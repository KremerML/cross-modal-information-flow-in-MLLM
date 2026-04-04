"""Utilities for attention knockout flow definitions and token ranges — PaliGemma 2 edition."""

from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Image token count
# ---------------------------------------------------------------------------

def get_image_token_count(model) -> int:
    """Return the number of image tokens for this model.

    For PaliGemma 2 this is stored in ``model.config.num_image_tokens``
    (256 for the 224-px variant, 1024 for 448-px).
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "num_image_tokens"):
        return int(cfg.num_image_tokens)
    # Fallback: inspect model name from config
    name = ""
    if cfg is not None:
        name = str(getattr(cfg, "_name_or_path", "") or getattr(cfg, "name_or_path", ""))
    return get_paligemma_image_token_count(name)


def get_paligemma_image_token_count(model_name: str) -> int:
    """Return image token count based on PaliGemma 2 model name suffix."""
    return 1024 if "448" in model_name else 256


# ---------------------------------------------------------------------------
# Token range helpers
# ---------------------------------------------------------------------------

def get_image_token_range(image_token_count: int) -> List[int]:
    """Image tokens occupy positions 0 .. image_token_count-1 in PaliGemma's input_ids."""
    return list(range(0, image_token_count))


def get_question_token_range(
    input_ids: torch.Tensor,
    question_text: str,
    tokenizer,
) -> List[int]:
    """Return the full-sequence positions of the question text tokens.

    For PaliGemma 2, ``input_ids`` already contains the expanded image tokens
    at the start, so no placeholder-expansion offset is needed.  We search for
    the question token IDs as a sub-sequence and return their literal positions.
    """
    from sae_experiments.utils.token_utils import get_question_token_range as _impl

    # Passing image_token_count=1 and image_token_index=None disables the
    # LLaVA-style expansion offset; positions are already in full-sequence space.
    return _impl(
        input_ids[0] if input_ids.dim() > 1 else input_ids,
        image_token_count=1,
        question_text=question_text,
        tokenizer=tokenizer,
        image_token_index=None,
    )


def get_last_token_range(input_ids: torch.Tensor) -> List[int]:
    """Return the index of the last token in the sequence."""
    return [input_ids.shape[-1] - 1]


def resolve_flow_ranges(
    flow: str,
    input_ids: torch.Tensor,
    image_token_count: int,
    question_text: str,
    tokenizer,
    model_name: str = "",
) -> Tuple[List[int], List[int]]:
    """Compute (source_range, target_range) for a given attention flow string.

    Supported flows: ``"Image->Question"``, ``"Image->Last"``.
    """
    source_name, target_name = [p.strip() for p in flow.split("->")]

    if source_name == "Image":
        source_range = get_image_token_range(image_token_count)
    else:
        raise ValueError(f"Unsupported source flow: {source_name}")

    if target_name == "Question":
        target_range = get_question_token_range(input_ids, question_text, tokenizer)
    elif target_name == "Last":
        target_range = get_last_token_range(input_ids)
    else:
        raise ValueError(f"Unsupported target flow: {target_name}")

    if not source_range or not target_range:
        return [], []

    return source_range, target_range


# ---------------------------------------------------------------------------
# Block config builder
# ---------------------------------------------------------------------------

def build_block_config(
    layer: int,
    num_layers: int,
    window: int,
    src_tgt_pairs: List[Tuple[int, int]],
) -> Dict[int, List[Tuple[int, int]]]:
    """Build a block config dict for :func:`set_block_attn_hooks_paligemma`.

    Args:
        layer: Target layer index.
        num_layers: Total number of layers (for window clamping).
        window: Number of consecutive layers to block (centred on ``layer``).
        src_tgt_pairs: ``(target_token, source_token)`` pairs to block.

    Returns:
        Dict mapping layer index → list of ``(tgt, src)`` pairs.
    """
    if window <= 1:
        return {layer: src_tgt_pairs}
    half = window // 2
    layer_list = list(range(max(0, layer - half), min(num_layers, layer + half + 1)))
    return {lay: list(src_tgt_pairs) for lay in layer_list}


# ---------------------------------------------------------------------------
# Logprob scoring
# ---------------------------------------------------------------------------

def sequence_logprob(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    answer_text: str,
    normalize: bool = True,
    block_config: Optional[Dict[int, List[Tuple[int, int]]]] = None,
) -> Optional[float]:
    """Compute the (mean) log-probability of ``answer_text`` given ``input_ids``.

    For PaliGemma 2, ``pixel_values`` replaces the LLaVA ``images`` +
    ``image_sizes`` arguments.  Image tokens are already present in
    ``input_ids``, so no expansion offset is applied.

    Args:
        model: ``PaliGemmaForConditionalGeneration``.
        tokenizer: The processor's tokenizer (``processor.tokenizer``).
        input_ids: Tokenised context ``[1, seq_len]`` (includes image tokens).
        pixel_values: Processed image tensor ``[1, C, H, W]``.
        answer_text: String whose log-probability to score.
        normalize: If True, return mean per-token log-prob; else return sum.
        block_config: Optional attention knockout configuration.

    Returns:
        Float log-probability, or ``None`` if scoring fails.
    """
    if not answer_text:
        return None

    answer_ids = tokenizer.encode(f" {answer_text.strip()}", add_special_tokens=False)
    if not answer_ids:
        return None

    device = input_ids.device
    answer_tensor = torch.tensor([answer_ids], device=device, dtype=input_ids.dtype)
    input_ids_full = torch.cat([input_ids, answer_tensor], dim=1)

    hooks = None
    if block_config:
        from sae_experiments.utils.paligemma_hooks import (
            set_block_attn_hooks_paligemma,
            remove_wrapper_paligemma,
        )
        hooks = set_block_attn_hooks_paligemma(model, block_config)

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids_full,
                pixel_values=pixel_values,
                use_cache=False,
            )
    finally:
        if hooks:
            from sae_experiments.utils.paligemma_hooks import remove_wrapper_paligemma
            remove_wrapper_paligemma(model, hooks)

    logits = outputs.logits
    log_probs = torch.log_softmax(logits[0], dim=-1)

    # For PaliGemma, input_ids already contains image tokens — no expansion.
    # The logit at position (start - 1) predicts the first answer token.
    start = input_ids.shape[1]
    token_logps = []
    for i, tok_id in enumerate(answer_ids):
        idx = start + i - 1
        if idx < 0 or idx >= log_probs.shape[0]:
            continue
        token_logps.append(log_probs[idx, tok_id].item())

    if not token_logps:
        return None
    if normalize:
        return float(sum(token_logps) / len(token_logps))
    return float(sum(token_logps))

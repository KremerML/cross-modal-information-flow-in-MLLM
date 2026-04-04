"""Attention-blocking hooks for PaliGemma 2 (Gemma-2 language backbone).

These replace ``methods.set_block_attn_hooks_llava`` for PaliGemma.

PaliGemma 2 layer access path:
    model.language_model.model.layers[i].self_attn

HuggingFace Gemma-2 passes a 4-D additive attention mask of shape
``[batch, 1, q_len, kv_len]`` where 0 means "attend" and a large
negative value means "block".  The hooks below clone the mask and write
``-inf`` (dtype min) into every (tgt_token, src_token) slot listed in
``block_config``.
"""

from typing import Dict, List, Tuple

import torch


def set_block_attn_hooks_paligemma(
    model,
    block_config: Dict[int, List[Tuple[int, int]]],
) -> List[Tuple[int, object]]:
    """Install attention-blocking forward hooks on the specified layers.

    Args:
        model: A ``PaliGemmaForConditionalGeneration`` instance.
        block_config: Mapping from layer index to a list of
            ``(target_token, source_token)`` pairs to block.
            Blocking means ``target`` cannot attend to ``source``.

    Returns:
        List of ``(layer_idx, original_forward_fn)`` tuples; pass this to
        :func:`remove_wrapper_paligemma` to restore the originals.
    """
    hooks: List[Tuple[int, object]] = []

    for layer_idx, pairs in block_config.items():
        layer = model.language_model.model.layers[layer_idx]
        original_forward = layer.self_attn.forward

        # Capture pairs in closure
        def make_hooked_forward(orig_fwd, block_pairs):
            def hooked_forward(hidden_states, attention_mask=None, **kwargs):
                if attention_mask is not None:
                    # Clone to avoid in-place modification issues with autograd
                    attention_mask = attention_mask.clone()
                    q_len = hidden_states.shape[1]
                    kv_len = attention_mask.shape[-1]
                    dtype_min = float(
                        torch.finfo(hidden_states.dtype).min
                    )
                    for tgt, src in block_pairs:
                        if tgt < q_len and src < kv_len:
                            attention_mask[0, 0, tgt, src] = dtype_min
                else:
                    # No mask provided — construct a zero (all-attend) mask
                    # and apply blocking
                    q_len = hidden_states.shape[1]
                    kv_len = q_len  # self-attention
                    device = hidden_states.device
                    dtype = hidden_states.dtype
                    attention_mask = torch.zeros(
                        1, 1, q_len, kv_len, device=device, dtype=dtype
                    )
                    dtype_min = float(torch.finfo(dtype).min)
                    for tgt, src in block_pairs:
                        if tgt < q_len and src < kv_len:
                            attention_mask[0, 0, tgt, src] = dtype_min

                return orig_fwd(
                    hidden_states, attention_mask=attention_mask, **kwargs
                )

            return hooked_forward

        layer.self_attn.forward = make_hooked_forward(original_forward, list(pairs))
        hooks.append((layer_idx, original_forward))

    return hooks


def remove_wrapper_paligemma(model, hooks: List[Tuple[int, object]]) -> None:
    """Restore original ``self_attn.forward`` methods after a knockout pass.

    Args:
        model: The same model passed to :func:`set_block_attn_hooks_paligemma`.
        hooks: The list returned by :func:`set_block_attn_hooks_paligemma`.
    """
    for layer_idx, original_forward in hooks:
        model.language_model.model.layers[layer_idx].self_attn.forward = original_forward

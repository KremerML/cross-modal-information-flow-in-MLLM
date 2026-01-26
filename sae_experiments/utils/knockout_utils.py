"""Utilities for attention knockout flow definitions and token ranges."""

from typing import Dict, List, Optional, Tuple

import torch

try:
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    IMAGE_TOKEN_INDEX = -200


def estimate_inputs_embeds_shape(model, input_ids, image_tensor, image_sizes) -> Optional[Tuple[int, int, int]]:
    if not hasattr(model, "prepare_inputs_labels_for_multimodal"):
        return None
    try:
        _, _, _, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor,
            ["image"],
            image_sizes=image_sizes,
        )
        return tuple(inputs_embeds.shape)
    except Exception:
        return None


def find_token_range(tokenizer, token_array, substring: str, model_name: str) -> Tuple[int, int]:
    """Find token start/end indices for a substring within a token array."""
    toks = tokenizer.convert_ids_to_tokens(token_array)
    model_name = model_name or ""
    if model_name in ("llava-v1.6-vicuna-7b", "llava-v1.5-7b", "llava-v1.5-13b"):
        whole_string = "".join(toks).replace("▁", " ")
    else:
        whole_string = "".join(toks).replace("Ġ", " ").replace("Ċ", "\n")

    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def _split_input_ids_by_image_tokens(input_ids: torch.Tensor) -> List[torch.Tensor]:
    image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [
        input_ids[0].shape[0]
    ]
    input_ids_noim = []
    for i in range(len(image_token_indices) - 1):
        input_ids_noim.append(input_ids[0][image_token_indices[i] + 1 : image_token_indices[i + 1]])
    return input_ids_noim


def get_image_token_range(input_ids: torch.Tensor, inputs_embeds_shape: Tuple[int, int, int]) -> List[int]:
    image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
    image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
    return [x for x in range(image_index, image_index + image_dim)]


def get_question_token_range(
    input_ids: torch.Tensor,
    inputs_embeds_shape: Tuple[int, int, int],
    question_text: str,
    tokenizer,
    model_name: str,
) -> List[int]:
    image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
    input_ids_noim = _split_input_ids_by_image_tokens(input_ids)
    try:
        question_range = find_token_range(tokenizer, input_ids_noim[1], question_text, model_name)
    except ValueError:
        return []
    return [
        x
        for x in range(
            question_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
            question_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1,
        )
    ]


def get_last_token_range(input_ids: torch.Tensor, inputs_embeds_shape: Tuple[int, int, int]) -> List[int]:
    image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
    ntoks = input_ids.shape[1] + image_dim - 1
    return [ntoks - 1]


def resolve_flow_ranges(
    flow: str,
    input_ids: torch.Tensor,
    inputs_embeds_shape: Tuple[int, int, int],
    question_text: str,
    tokenizer,
    model_name: str,
) -> Tuple[List[int], List[int]]:
    source, target = [part.strip() for part in flow.split("->")]
    if source == "Image":
        source_range = get_image_token_range(input_ids, inputs_embeds_shape)
    else:
        raise ValueError(f"Unsupported source flow: {source}")

    if target == "Question":
        target_range = get_question_token_range(
            input_ids, inputs_embeds_shape, question_text, tokenizer, model_name
        )
    elif target == "Last":
        target_range = get_last_token_range(input_ids, inputs_embeds_shape)
    else:
        raise ValueError(f"Unsupported target flow: {target}")

    if not source_range or not target_range:
        return [], []

    return source_range, target_range


def build_block_config(
    layer: int,
    num_layers: int,
    window: int,
    src_tgt_pairs: List[Tuple[int, int]],
) -> Dict[int, List[Tuple[int, int]]]:
    if window <= 1:
        return {layer: src_tgt_pairs}
    half = window // 2
    layerlist = list(range(max(0, layer - half), min(num_layers, layer + half + 1)))
    return {l: list(src_tgt_pairs) for l in layerlist}

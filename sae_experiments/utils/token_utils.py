"""Token-related helpers."""

from typing import Iterable, List, Tuple
import re


ATTRIBUTE_VOCAB = {
    "color": [
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
        "grey",
        "gold",
        "silver",
        "tan",
        "beige",
    ],
    "size": [
        "small",
        "large",
        "big",
        "tiny",
        "tall",
        "short",
        "wide",
        "long",
    ],
    "shape": [
        "round",
        "square",
        "rectangular",
        "triangle",
        "circular",
        "oval",
    ],
    "material": [
        "wood",
        "wooden",
        "metal",
        "plastic",
        "glass",
        "ceramic",
        "leather",
    ],
    "state": [
        "open",
        "closed",
        "full",
        "empty",
        "broken",
        "clean",
        "dirty",
    ],
}


def extract_attribute_words(question: str) -> List[Tuple[str, str, int]]:
    """Return list of (word, category, index) for attribute words."""
    tokens = re.findall(r"[a-zA-Z']+", question.lower())
    found = []
    for idx, word in enumerate(tokens):
        for category, vocab in ATTRIBUTE_VOCAB.items():
            if word in vocab:
                found.append((word, category, idx))
    return found


def get_token_positions(text: str, target_words: Iterable[str], tokenizer) -> List[int]:
    """Return token positions for target words within tokenized text."""
    if tokenizer is None:
        return []
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    positions: List[int] = []
    for word in target_words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            continue
        for i in range(len(input_ids) - len(word_ids) + 1):
            if input_ids[i : i + len(word_ids)] == word_ids:
                positions.append(i)
    return positions


def align_tokens_to_positions(tokens: Iterable[str], input_ids: List[int], tokenizer) -> List[int]:
    """Align token strings to token indices in input_ids."""
    positions = []
    if tokenizer is None:
        return positions
    for token in tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if not token_ids:
            continue
        for i in range(len(input_ids) - len(token_ids) + 1):
            if input_ids[i : i + len(token_ids)] == token_ids:
                positions.append(i)
    return positions


def get_question_token_range(
    input_ids,
    image_token_count: int,
    question_text: str = "",
    tokenizer=None,
    image_token_index: int = None,
) -> List[int]:
    """Return full-sequence positions for question tokens."""
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    image_idx = ids.index(image_token_index) if image_token_index in ids else None

    if question_text and tokenizer is not None:
        question_ids = tokenizer.encode(question_text, add_special_tokens=False)
        match_start = _find_sublist(ids, question_ids)
        if match_start is not None:
            offset = 0
            if image_idx is not None and match_start > image_idx:
                offset = max(0, image_token_count - 1)
            start = match_start + offset
            end = start + len(question_ids)
            return list(range(start, end))

    if image_idx is None:
        return list(range(len(ids)))

    start = image_idx + image_token_count
    end = len(ids) - 1 + image_token_count
    if start >= end:
        return []
    return list(range(start, end))


def _find_sublist(haystack: List[int], needle: List[int]):
    if not needle:
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None

"""PaliGemma 2 data loader for the ChooseAttr VQA task."""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PaliGemmaChooseAttrDataset(Dataset):
    """Dataset for ChooseAttr rows, processed for PaliGemma 2.

    Each item returns ``(input_ids, pixel_values, question_text, row_dict)``
    where ``input_ids`` and ``pixel_values`` are ready to pass to
    ``PaliGemmaForConditionalGeneration``.

    Args:
        questions: List of row dicts (each must have ``"q_id"`` and match a key in
            ``dataset_dict``).
        dataset_dict: Mapping ``question_id → row dict`` (the full CSV, indexed).
        image_folder: Path to the folder containing image files.
        processor: HuggingFace ``AutoProcessor`` for PaliGemma 2.
    """

    def __init__(
        self,
        questions: List[Dict[str, Any]],
        dataset_dict: Dict[str, Dict[str, Any]],
        image_folder: str,
        processor,
    ):
        self.questions = questions
        self.dataset_dict = dataset_dict
        self.image_folder = image_folder
        self.processor = processor
        # Expose tokenizer so FeatureAblator / KnockoutRunner can reach it
        self.tokenizer = processor.tokenizer

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        line = self.questions[idx]
        q_id = line["q_id"]
        row = self.dataset_dict[q_id]

        question_text = row.get("question", "")
        image_filename = row.get("image", row.get("img_id", ""))
        image_path = os.path.join(self.image_folder, str(image_filename))

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Return a blank image so the batch doesn't break during sweeps
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # PaliGemma processor accepts text + images and returns tensors
        inputs = self.processor(
            text=question_text,
            images=image,
            return_tensors="pt",
        )
        # Squeeze the batch dimension added by the processor
        input_ids = inputs["input_ids"].squeeze(0)      # (seq_len,)
        pixel_values = inputs["pixel_values"].squeeze(0)  # (C, H, W)

        return input_ids, pixel_values, question_text, line

    def create_dataloader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=_paligemma_collate_fn,
        )


def _paligemma_collate_fn(batch):
    """Collate a list of ``(input_ids, pixel_values, question_text, row_dict)`` items.

    ``input_ids`` are left-padded so all sequences in the batch have the same
    length.  ``pixel_values`` are stacked normally.
    """
    input_ids_list, pixel_values_list, question_texts, row_dicts = zip(*batch)

    # Pad input_ids to the longest sequence in the batch
    max_len = max(t.shape[0] for t in input_ids_list)
    pad_id = 0  # PaliGemma tokenizer pads with 0 (or use processor.tokenizer.pad_token_id)
    padded_ids = torch.full((len(input_ids_list), max_len), fill_value=pad_id, dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        padded_ids[i, : ids.shape[0]] = ids

    pixel_values = torch.stack(pixel_values_list, dim=0)
    return padded_ids, pixel_values, list(question_texts), list(row_dicts)


def create_paligemma_data_loader(
    questions: List[Dict[str, Any]],
    dataset_dict: Dict[str, Dict[str, Any]],
    image_folder: str,
    processor,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """Convenience wrapper that builds a :class:`PaliGemmaChooseAttrDataset` and
    returns its :class:`DataLoader`.

    Args:
        questions: List of question dicts (each has ``"q_id"``).
        dataset_dict: Full dataset dict keyed by ``question_id``.
        image_folder: Path to image files.
        processor: HuggingFace ``AutoProcessor`` for PaliGemma 2.
        batch_size: Batch size (default 1 — recommended for knockout sweeps).
        num_workers: DataLoader worker count.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A :class:`DataLoader` yielding
        ``(input_ids, pixel_values, question_texts, row_dicts)`` batches.
    """
    dataset = PaliGemmaChooseAttrDataset(
        questions=questions,
        dataset_dict=dataset_dict,
        image_folder=image_folder,
        processor=processor,
    )
    return dataset.create_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

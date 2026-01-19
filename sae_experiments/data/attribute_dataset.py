"""Dataset helpers for attribute-centric VQA tasks."""

from typing import Any, Dict, List, Optional

import os
import pandas as pd

from InformationFlow import create_data_loader
from sae_experiments.utils import token_utils


class AttributeVQADataset:
    """Wrapper around GQA CSVs with attribute annotations."""

    def __init__(
        self,
        refined_dataset: str,
        image_folder: str,
        tokenizer,
        image_processor,
        model_config,
        task_type: str = "ChooseAttr",
        conv_mode: str = "vicuna_v1",
    ):
        self.refined_dataset = refined_dataset
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.task_type = task_type
        self.conv_mode = conv_mode

        self.df = pd.read_csv(refined_dataset, dtype={"question_id": str}).fillna("")
        self.filter_by_task()

        self.dataset_dict = self.df.set_index("question_id").T.to_dict("dict")
        self.questions = [
            {**detail, "q_id": q_id} for q_id, detail in self.dataset_dict.items()
        ]
        self.id_to_index = {entry["q_id"]: idx for idx, entry in enumerate(self.questions)}
        self.extract_attribute_info()

    def filter_by_task(self) -> None:
        if not self.task_type:
            return
        task_lower = self.task_type.lower()
        for col in ("type_detailed", "type_structural", "type_semantic"):
            if col in self.df.columns:
                mask = self.df[col].astype(str).str.lower().str.contains(task_lower)
                if mask.any():
                    self.df = self.df[mask]
                    return

    def extract_attribute_info(self) -> None:
        """Extract attribute words and token positions for each question."""
        for entry in self.questions:
            question = entry.get("question", "")
            attrs = token_utils.extract_attribute_words(question)
            attr_entries = []
            for word, category, _ in attrs:
                positions = token_utils.get_token_positions(
                    question,
                    [word],
                    self.tokenizer,
                )
                attr_entries.append(
                    {
                        "word": word,
                        "category": category,
                        "positions": positions,
                    }
                )
            entry["attribute_tokens"] = attr_entries

    def get_item_with_metadata(self, idx: int) -> Dict[str, Any]:
        entry = self.questions[idx]
        detail = self.dataset_dict[entry["q_id"]]
        return {
            "question_id": entry["q_id"],
            "question": detail.get("question", ""),
            "answer": detail.get("answer", ""),
            "true_option": detail.get("true option", ""),
            "false_option": detail.get("false option", ""),
            "bboxes": self._get_bboxes(detail),
            "attribute_tokens": entry.get("attribute_tokens", []),
        }

    def get_item_with_metadata_by_id(self, question_id: str) -> Dict[str, Any]:
        idx = self.id_to_index.get(question_id)
        if idx is None:
            return {}
        return self.get_item_with_metadata(idx)

    def create_control_dataset(self, task_type: str = "ChooseRel") -> "AttributeVQADataset":
        dataset_path = self.refined_dataset
        if task_type and "ChooseAttr" in dataset_path:
            candidate = dataset_path.replace("ChooseAttr", task_type)
            if os.path.exists(candidate):
                dataset_path = candidate
        return AttributeVQADataset(
            refined_dataset=dataset_path,
            image_folder=self.image_folder,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            model_config=self.model_config,
            task_type=task_type,
            conv_mode=self.conv_mode,
        )

    def create_dataloader(self, batch_size: int = 1, num_workers: int = 0):
        return create_data_loader(
            self.questions,
            self.image_folder,
            batch_size,
            num_workers,
            self.tokenizer,
            self.image_processor,
            self.model_config,
            self.task_type,
            self.conv_mode,
        )

    def _get_bboxes(self, detail: Dict[str, Any]) -> Optional[List[tuple]]:
        if self.task_type in ("CompareAttr", "ChooseRel", "LogicalObj"):
            boxes = []
            if detail.get("object1 x", "") != "":
                boxes.append(
                    (
                        int(detail["object1 x"]),
                        int(detail["object1 y"]),
                        int(detail["object1 x"]) + int(detail["object1 w"]),
                        int(detail["object1 y"]) + int(detail["object1 h"]),
                    )
                )
            if detail.get("object2 x", "") not in ("", "-"):
                boxes.append(
                    (
                        int(detail["object2 x"]),
                        int(detail["object2 y"]),
                        int(detail["object2 x"]) + int(detail["object2 w"]),
                        int(detail["object2 y"]) + int(detail["object2 h"]),
                    )
                )
            return boxes if boxes else None
        if self.task_type in ("ChooseAttr", "ChooseCat", "QueryAttr"):
            if detail.get("central object x", "") == "":
                return None
            return [
                (
                    int(detail["central object x"]),
                    int(detail["central object y"]),
                    int(detail["central object x"]) + int(detail["central object w"]),
                    int(detail["central object y"]) + int(detail["central object h"]),
                )
            ]
        return None

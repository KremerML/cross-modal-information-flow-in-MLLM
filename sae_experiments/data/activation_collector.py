"""Collects hidden activations from target model layers."""

from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    IMAGE_TOKEN_INDEX = -200

from sae_experiments.utils.hook_utils import HookManager, create_activation_capture_hook, get_target_module
from sae_experiments.utils.knockout_utils import estimate_image_token_count
from sae_experiments.utils import token_utils


class ActivationCollector:
    """Collects activations from a specific layer in a model."""

    def __init__(self, model: torch.nn.Module, layer_idx: int, activation_site: str = "residual"):
        self.model = model
        self.layer_idx = layer_idx
        self.activation_site = activation_site
        self.hook_manager = HookManager(model)
        self.storage: Dict[str, torch.Tensor] = {}

    def register_hooks(self) -> None:
        layer = get_target_module(self.model, self.layer_idx, self.activation_site)
        self.hook_manager.register_forward_hook(
            layer, create_activation_capture_hook(self.storage, "acts")
        )

    def collect_from_dataset(
        self,
        dataset,
        position_type: str = "question",
        tokenizer=None,
        max_samples: Optional[int] = None,
        device: Optional[str] = None,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        if hasattr(dataset, "create_dataloader"):
            data_loader = dataset.create_dataloader()
            questions = dataset.questions
            dataset_dict = dataset.dataset_dict
        else:
            data_loader = dataset
            questions = getattr(dataset, "questions", None)
            dataset_dict = getattr(dataset, "dataset_dict", {})
            if questions is None:
                raise ValueError("dataset must provide questions list")

        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        self.register_hooks()
        activations: List[torch.Tensor] = []
        metadata: List[Dict[str, Any]] = []
        offset = 0
        try:
            iterator = zip(data_loader, questions)
            total = len(questions) if questions is not None else None
            if max_samples is not None and total is not None:
                total = min(total, max_samples)
            if show_progress:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=total, desc="Collecting activations")

            for idx, (batch, line) in enumerate(iterator):
                if max_samples is not None and idx >= max_samples:
                    break

                input_ids, image_tensor, image_sizes, prompts, _ = batch
                input_ids = input_ids.to(device)
                image_tensor = [img.to(device) for img in image_tensor]

                image_token_count = estimate_image_token_count(
                    self.model, input_ids, image_tensor, image_sizes
                )

                self.storage.pop("acts", None)
                with torch.no_grad():
                    _ = self.model(
                        input_ids=input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        use_cache=False,
                    )

                acts = self.storage.get("acts")
                if acts is None:
                    continue
                acts = self._normalize_activation_tensor(acts)
                if acts is None:
                    continue

                question_text = dataset_dict[line["q_id"]]["question"]
                positions = self._select_positions(
                    position_type,
                    input_ids,
                    image_token_count,
                    question_text,
                    tokenizer,
                    line,
                )
                if not positions:
                    continue

                activations.append(acts[positions])
                metadata.append(
                    {
                        "question_id": line["q_id"],
                        "question": question_text,
                        "answer": dataset_dict[line["q_id"]].get("answer", ""),
                        "positions": positions,
                        "attribute_tokens": line.get("attribute_tokens", []),
                        "start_idx": offset,
                        "count": len(positions),
                    }
                )
                offset += len(positions)
        finally:
            self.hook_manager.remove_hooks()

        if not activations:
            return torch.empty(0), metadata

        return torch.cat(activations, dim=0), metadata

    @staticmethod
    def _normalize_activation_tensor(acts):
        if acts is None:
            return None
        if isinstance(acts, (tuple, list)):
            acts = acts[0]
        if acts is None:
            return None
        # Most model hooks return [batch, seq, hidden]. Some wrappers can add batch dims.
        while acts.ndim > 3 and acts.shape[0] == 1:
            acts = acts[0]
        if acts.ndim != 3:
            return None
        return acts[0]

    def _select_positions(
        self,
        position_type: str,
        input_ids: torch.Tensor,
        image_token_count: int,
        question_text: str,
        tokenizer,
        line: Dict[str, Any],
    ) -> List[int]:
        if position_type == "all":
            return list(range(input_ids.shape[-1]))

        question_range = token_utils.get_question_token_range(
            input_ids[0],
            image_token_count,
            question_text=question_text,
            tokenizer=tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
        )

        if position_type == "last":
            ntoks = input_ids.shape[-1] + image_token_count - 1
            return [max(0, ntoks - 1)]

        if position_type == "question":
            return question_range

        if position_type == "attribute":
            attr_positions = []
            for attr in line.get("attribute_tokens", []):
                attr_positions.extend(attr.get("positions", []))
            if not question_range:
                return []
            start = question_range[0]
            end = question_range[-1]
            positions = [start + pos for pos in attr_positions]
            positions = [pos for pos in positions if start <= pos <= end]
            return sorted(set(positions))

        return question_range

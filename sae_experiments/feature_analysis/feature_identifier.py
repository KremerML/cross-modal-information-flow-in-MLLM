"""Identify discriminative SAE features for attribute binding."""

from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch

from sae_experiments.data.activation_collector import ActivationCollector


class FeatureIdentifier:
    """Compute feature statistics and select discriminative features."""

    def __init__(self, sae, model, dataset, layer_idx: int):
        self.sae = sae
        self.model = model
        self.dataset = dataset
        self.layer_idx = layer_idx
        self.feature_stats: Dict[int, Dict[str, float]] = {}
        self.feature_acts: Optional[np.ndarray] = None
        self.metadata: Optional[List[dict]] = None

    def compute_feature_activations(
        self,
        position_type: str = "attribute",
        max_samples: Optional[int] = None,
        include_predictions: bool = False,
    ) -> Tuple[np.ndarray, List[dict]]:
        collector = ActivationCollector(self.model, self.layer_idx)
        activations, metadata = collector.collect_from_dataset(
            self.dataset,
            position_type=position_type,
            tokenizer=self.dataset.tokenizer,
            max_samples=max_samples,
        )
        if activations.numel() == 0:
            return np.empty((0, self.sae.n_features)), []

        device = next(self.sae.parameters()).device
        activations = activations.to(device)
        with torch.no_grad():
            feats = self.sae.encode(activations).cpu().numpy()

        per_sample = []
        for meta in metadata:
            start = meta["start_idx"]
            count = meta["count"]
            if count == 0:
                per_sample.append(np.zeros((self.sae.n_features,)))
                continue
            per_sample.append(feats[start : start + count].mean(axis=0))
        per_sample = np.stack(per_sample, axis=0)

        if include_predictions:
            predictions = self._compute_predictions(max_samples=max_samples)
            for meta in metadata:
                pred = predictions.get(meta["question_id"], "")
                meta["predicted_answer"] = pred
                meta["is_correct"] = pred == meta.get("answer", "").strip().lower()

        self.feature_acts = per_sample
        self.metadata = metadata
        return per_sample, metadata

    def find_discriminative_features(self, threshold: float = 2.0) -> List[int]:
        if self.feature_acts is None or self.metadata is None:
            raise ValueError("Run compute_feature_activations first")

        correct_mask = np.array(
            [meta.get("is_correct", False) for meta in self.metadata], dtype=bool
        )
        if correct_mask.sum() == 0 or (~correct_mask).sum() == 0:
            return []

        correct_mean = self.feature_acts[correct_mask].mean(axis=0)
        incorrect_mean = self.feature_acts[~correct_mask].mean(axis=0)
        ratio = (correct_mean + 1e-8) / (incorrect_mean + 1e-8)

        features = np.where(ratio > threshold)[0].tolist()
        for idx in features:
            self.feature_stats[idx] = {
                "correct_mean": float(correct_mean[idx]),
                "incorrect_mean": float(incorrect_mean[idx]),
                "ratio": float(ratio[idx]),
            }
        return features

    def find_attribute_specific_features(self, attribute_type: str = "color") -> List[int]:
        if self.feature_acts is None or self.metadata is None:
            raise ValueError("Run compute_feature_activations first")

        mask = []
        for meta in self.metadata:
            attr_types = {a["category"] for a in meta.get("attribute_tokens", [])}
            mask.append(attribute_type in attr_types)
        mask = np.array(mask, dtype=bool)

        if mask.sum() == 0:
            return []

        attr_mean = self.feature_acts[mask].mean(axis=0)
        other_mean = self.feature_acts[~mask].mean(axis=0)
        ratio = (attr_mean + 1e-8) / (other_mean + 1e-8)

        features = np.argsort(ratio)[::-1].tolist()
        for idx in features[: min(200, len(features))]:
            self.feature_stats[idx] = {
                "attr_mean": float(attr_mean[idx]),
                "other_mean": float(other_mean[idx]),
                "ratio": float(ratio[idx]),
            }
        return features

    def get_top_k_features(self, k: int = 50) -> List[int]:
        if not self.feature_stats:
            return []
        ranked = sorted(self.feature_stats.items(), key=lambda x: x[1].get("ratio", 0), reverse=True)
        return [idx for idx, _ in ranked[:k]]

    def save_feature_statistics(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.feature_stats, handle, indent=2)

    def _compute_predictions(self, max_samples: Optional[int] = None) -> Dict[str, str]:
        data_loader = self.dataset.create_dataloader()
        predictions: Dict[str, str] = {}
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        for idx, (batch, line) in enumerate(zip(data_loader, self.dataset.questions)):
            if max_samples is not None and idx >= max_samples:
                break
            input_ids, image_tensor, image_sizes, _, _ = batch
            input_ids = input_ids.to(device=device)
            image_tensor = [img.to(device=device) for img in image_tensor]
            inps = {
                "inputs": input_ids,
                "images": image_tensor,
                "image_sizes": image_sizes,
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 1,
                "use_cache": True,
                "return_dict_in_generate": True,
                "output_scores": False,
                "pad_token_id": self.dataset.tokenizer.eos_token_id,
            }
            with torch.inference_mode():
                output = self.model.generate(**inps)
            answer = self.dataset.tokenizer.batch_decode(
                output["sequences"], skip_special_tokens=True
            )[0].strip().lower()
            predictions[line["q_id"]] = answer
        return predictions

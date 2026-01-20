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
        correctness_metric: str = "string_match",
        logprob_normalize: bool = True,
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

        sae_param = next(self.sae.parameters())
        device = sae_param.device
        dtype = sae_param.dtype
        activations = activations.to(device=device, dtype=dtype)
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

        predictions = None
        if include_predictions or correctness_metric == "string_match":
            predictions = self._compute_predictions(max_samples=max_samples)
        if correctness_metric == "option_logprob":
            option_scores = self._compute_option_logprobs(
                max_samples=max_samples,
                normalize=logprob_normalize,
            )
        else:
            option_scores = {}
        for meta in metadata:
            qid = meta["question_id"]
            pred = predictions.get(qid, "") if predictions is not None else ""
            meta["predicted_answer"] = pred
            if correctness_metric == "option_logprob" and qid in option_scores:
                scores = option_scores[qid]
                meta["true_option_logprob"] = scores.get("true")
                meta["false_option_logprob"] = scores.get("false")
                meta["correctness_score"] = scores.get("margin")
                meta["is_correct"] = scores.get("is_correct")
            else:
                meta["is_correct"] = pred == meta.get("answer", "").strip().lower()

        self.feature_acts = per_sample
        self.metadata = metadata
        return per_sample, metadata

    def find_discriminative_features(
        self,
        threshold: float = 2.0,
        min_activation: float = 0.0,
        min_diff: float = 0.0,
    ) -> List[int]:
        if self.feature_acts is None or self.metadata is None:
            raise ValueError("Run compute_feature_activations first")

        correct_mask = np.array(
            [meta.get("is_correct", False) for meta in self.metadata], dtype=bool
        )
        if correct_mask.sum() == 0 or (~correct_mask).sum() == 0:
            return []

        correct_mean = self.feature_acts[correct_mask].mean(axis=0)
        incorrect_mean = self.feature_acts[~correct_mask].mean(axis=0)
        denom = np.maximum(incorrect_mean, min_activation if min_activation > 0 else 1e-8)
        ratio = (correct_mean + 1e-8) / (denom + 1e-8)
        diff = correct_mean - incorrect_mean

        mask = (ratio > threshold) & (correct_mean >= min_activation) & (diff >= min_diff)
        features = np.where(mask)[0].tolist()
        for idx in features:
            self.feature_stats[idx] = {
                "correct_mean": float(correct_mean[idx]),
                "incorrect_mean": float(incorrect_mean[idx]),
                "ratio": float(ratio[idx]),
                "diff": float(diff[idx]),
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

    def _compute_option_logprobs(
        self,
        max_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        data_loader = self.dataset.create_dataloader()
        results: Dict[str, Dict[str, float]] = {}
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
            detail = self.dataset.dataset_dict[line["q_id"]]
            true_option = detail.get("true option", "").strip()
            false_option = detail.get("false option", "").strip()
            if not true_option or not false_option:
                continue
            true_lp = self._sequence_logprob(
                input_ids,
                image_tensor,
                image_sizes,
                true_option,
                normalize=normalize,
            )
            false_lp = self._sequence_logprob(
                input_ids,
                image_tensor,
                image_sizes,
                false_option,
                normalize=normalize,
            )
            if true_lp is None or false_lp is None:
                continue
            results[line["q_id"]] = {
                "true": true_lp,
                "false": false_lp,
                "margin": true_lp - false_lp,
                "is_correct": true_lp > false_lp,
            }
        return results

    def _sequence_logprob(
        self,
        input_ids,
        image_tensor,
        image_sizes,
        answer_text: str,
        normalize: bool = True,
    ) -> Optional[float]:
        if not answer_text:
            return None
        answer_ids = self.dataset.tokenizer.encode(
            f" {answer_text.strip()}",
            add_special_tokens=False,
        )
        if not answer_ids:
            return None
        device = input_ids.device
        answer_tensor = torch.tensor([answer_ids], device=device, dtype=input_ids.dtype)
        input_ids_full = torch.cat([input_ids, answer_tensor], dim=1)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids_full,
                images=image_tensor,
                image_sizes=image_sizes,
                use_cache=False,
            )
        logits = outputs.logits
        log_probs = torch.log_softmax(logits[0], dim=-1)
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

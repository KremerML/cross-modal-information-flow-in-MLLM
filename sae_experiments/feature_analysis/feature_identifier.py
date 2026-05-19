"""Identify discriminative SAE features for attribute binding."""

from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch

from sae_experiments.data.activation_collector import ActivationCollector


class FeatureIdentifier:
    """Compute feature statistics and select discriminative features."""

    def __init__(self, sae, model, dataset, layer_idx: int, activation_site: str = "residual"):
        self.sae = sae
        self.model = model
        self.dataset = dataset
        self.layer_idx = layer_idx
        self.activation_site = activation_site
        self.feature_stats: Dict[int, Dict[str, float]] = {}
        self.feature_acts: Optional[np.ndarray] = None
        self.metadata: Optional[List[dict]] = None
        self.feature_distribution_summary: Optional[Dict[str, object]] = None

    def compute_feature_activations(
        self,
        position_type: str = "attribute",
        max_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        include_predictions: bool = False,
        correctness_metric: str = "string_match",
        logprob_normalize: bool = True,
        aggregation: str = "mean",
        show_progress: bool = False,
    ) -> Tuple[np.ndarray, List[dict]]:
        collector = ActivationCollector(
            self.model,
            self.layer_idx,
            activation_site=self.activation_site,
        )
        activations, metadata = collector.collect_from_dataset(
            self.dataset,
            position_type=position_type,
            tokenizer=self.dataset.tokenizer,
            max_samples=max_samples,
            show_progress=show_progress,
        )
        if activations.numel() == 0:
            return np.empty((0, self.sae.n_features)), []

        sae_param = next(self.sae.parameters())
        device = sae_param.device
        dtype = sae_param.dtype
        if batch_size is None:
            batch_size = 2048
        n_features = self.sae.n_features
        aggregation = str(aggregation).lower()

        per_sample = []
        per_sample_metadata: List[dict] = []

        encode_batch_size = batch_size
        total_rows = activations.shape[0]
        if show_progress:
            from tqdm import tqdm

        sample_iter = iter(enumerate(metadata))
        if show_progress:
            sample_iter = tqdm(sample_iter, total=len(metadata), desc="Encoding SAE")

        for _si, meta in sample_iter:
            count = meta["count"]
            if count == 0:
                continue
            start = meta["start_idx"]
            sample_acts = activations[start : start + count]

            feats_chunks = []
            with torch.no_grad():
                for s in range(0, count, encode_batch_size):
                    chunk = sample_acts[s : s + encode_batch_size].to(
                        device=device, dtype=dtype
                    )
                    feats_chunks.append(self.sae.encode(chunk).cpu().numpy())
            token_feats = np.concatenate(feats_chunks, axis=0) if len(feats_chunks) > 1 else feats_chunks[0]

            if aggregation == "none":
                for ti in range(token_feats.shape[0]):
                    per_sample.append(token_feats[ti])
                    tm = dict(meta)
                    tm["token_position_index"] = ti
                    per_sample_metadata.append(tm)
            elif aggregation == "max":
                per_sample.append(token_feats.max(axis=0))
                per_sample_metadata.append(meta)
            else:
                per_sample.append(token_feats.mean(axis=0))
                per_sample_metadata.append(meta)

        if not per_sample:
            return np.empty((0, n_features)), []
        per_sample = np.stack(per_sample, axis=0)
        metadata = per_sample_metadata

        predictions = None
        if include_predictions or correctness_metric == "string_match":
            predictions = self._compute_predictions(max_samples=max_samples, show_progress=show_progress)
        if correctness_metric == "option_logprob":
            option_scores = self._compute_option_logprobs(
                max_samples=max_samples,
                normalize=logprob_normalize,
                show_progress=show_progress,
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
        selection_method: str = "ratio",
    ) -> List[int]:
        if self.feature_acts is None or self.metadata is None:
            raise ValueError("Run compute_feature_activations first")

        selection_method = str(selection_method).strip().lower()
        if selection_method in ("diff", "absolute_diff"):
            selection_method = "abs_diff"
        if selection_method not in {"ratio", "abs_diff"}:
            raise ValueError(
                f"Unsupported selection_method={selection_method}. "
                "Expected one of: ratio, abs_diff"
            )

        correct_mask = np.array(
            [meta.get("is_correct", False) for meta in self.metadata], dtype=bool
        )
        if correct_mask.sum() == 0 or (~correct_mask).sum() == 0:
            self.feature_distribution_summary = {
                "selection_method": selection_method,
                "n_features": int(self.feature_acts.shape[1]),
                "n_samples": int(self.feature_acts.shape[0]),
                "n_correct": int(correct_mask.sum()),
                "n_incorrect": int((~correct_mask).sum()),
                "empty_split": True,
            }
            return []

        correct_mean = self.feature_acts[correct_mask].mean(axis=0)
        incorrect_mean = self.feature_acts[~correct_mask].mean(axis=0)
        # Always compute ratio as a diagnostic signal, even when not used for gating.
        denom_diag = np.maximum(incorrect_mean, 1e-8)
        ratio_diag = (correct_mean + 1e-8) / (denom_diag + 1e-8)
        diff = correct_mean - incorrect_mean

        if selection_method == "abs_diff":
            mask = (correct_mean >= min_activation) & (diff >= min_diff)
        else:
            denom_gate = np.maximum(incorrect_mean, min_activation if min_activation > 0 else 1e-8)
            ratio_gate = (correct_mean + 1e-8) / (denom_gate + 1e-8)
            mask = (ratio_gate > threshold) & (correct_mean >= min_activation) & (diff >= min_diff)

        self.feature_distribution_summary = self._build_feature_distribution_summary(
            selection_method=selection_method,
            correct_mask=correct_mask,
            correct_mean=correct_mean,
            incorrect_mean=incorrect_mean,
            diff=diff,
            ratio=ratio_diag,
        )

        features = np.where(mask)[0].tolist()
        for idx in features:
            self.feature_stats[idx] = {
                "correct_mean": float(correct_mean[idx]),
                "incorrect_mean": float(incorrect_mean[idx]),
                "ratio": float(ratio_diag[idx]),
                "diff": float(diff[idx]),
            }
        return features

    def get_top_k_features(self, k: int = 50, score_key: str = "ratio") -> List[int]:
        if not self.feature_stats:
            return []
        ranked = sorted(
            self.feature_stats.items(),
            key=lambda x: x[1].get(score_key, x[1].get("ratio", 0)),
            reverse=True,
        )
        return [idx for idx, _ in ranked[:k]]

    def save_feature_statistics(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.feature_stats, handle, indent=2)

    def get_feature_distribution_summary(self) -> Dict[str, object]:
        if self.feature_distribution_summary is None:
            return {}
        return dict(self.feature_distribution_summary)

    def save_feature_distribution_summary(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.get_feature_distribution_summary(), handle, indent=2)

    @staticmethod
    def _percentile_stats(values: np.ndarray) -> Dict[str, float]:
        if values.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "p99_5": 0.0,
            }
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "p99_5": float(np.percentile(values, 99.5)),
        }

    def _build_feature_distribution_summary(
        self,
        selection_method: str,
        correct_mask: np.ndarray,
        correct_mean: np.ndarray,
        incorrect_mean: np.ndarray,
        diff: np.ndarray,
        ratio: np.ndarray,
    ) -> Dict[str, object]:
        return {
            "selection_method": selection_method,
            "n_features": int(correct_mean.shape[0]),
            "n_samples": int(self.feature_acts.shape[0]) if self.feature_acts is not None else 0,
            "n_correct": int(correct_mask.sum()),
            "n_incorrect": int((~correct_mask).sum()),
            "correct_mean": self._percentile_stats(correct_mean),
            "incorrect_mean": self._percentile_stats(incorrect_mean),
            "diff": self._percentile_stats(diff),
            "ratio": self._percentile_stats(ratio),
        }

    def _compute_predictions(
        self,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
    ) -> Dict[str, str]:
        tokenizer = getattr(self.dataset, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "eos_token_id"):
            predictions: Dict[str, str] = {}
            total = len(self.dataset.questions)
            if max_samples is not None:
                total = min(total, max_samples)
            for idx in range(total):
                qid = self.dataset.questions[idx]["q_id"]
                predictions[qid] = ""
            return predictions

        data_loader = self.dataset.create_dataloader()
        predictions: Dict[str, str] = {}
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        iterator = zip(data_loader, self.dataset.questions)
        total = len(self.dataset.questions)
        if max_samples is not None:
            total = min(total, max_samples)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=total, desc="Generating predictions")
        for idx, (batch, line) in enumerate(iterator):
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
                "pad_token_id": tokenizer.eos_token_id,
            }
            with torch.inference_mode():
                output = self.model.generate(**inps)
            answer = tokenizer.batch_decode(
                output["sequences"], skip_special_tokens=True
            )[0].strip().lower()
            predictions[line["q_id"]] = answer
        return predictions

    def _compute_option_logprobs(
        self,
        max_samples: Optional[int] = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        if getattr(self.dataset, "tokenizer", None) is None:
            return {}
        data_loader = self.dataset.create_dataloader()
        results: Dict[str, Dict[str, float]] = {}
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        iterator = zip(data_loader, self.dataset.questions)
        total = len(self.dataset.questions)
        if max_samples is not None:
            total = min(total, max_samples)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=total, desc="Scoring options")
        for idx, (batch, line) in enumerate(iterator):
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
        # Account for multimodal expansion (image tokens) when locating answer logits.
        start = input_ids.shape[1]
        if hasattr(self.model, "prepare_inputs_labels_for_multimodal"):
            try:
                _, _, _, _, inputs_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,
                    None,
                    None,
                    None,
                    image_tensor,
                    ["image"],
                    image_sizes=image_sizes,
                )
                image_dim = inputs_embeds.shape[1] - (input_ids.shape[-1] - 1)
                start = input_ids.shape[1] + image_dim - 1
            except Exception:
                pass
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

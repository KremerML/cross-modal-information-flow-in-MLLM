"""Feature ablation utilities — PaliGemma 2 edition."""

from typing import Any, Dict, List, Optional, Tuple

import torch

from tqdm import tqdm

from sae_experiments.utils import token_utils
from sae_experiments.utils.hook_utils import HookManager, get_target_module
from sae_experiments.utils.knockout_utils import get_image_token_count, sequence_logprob
from sae_experiments.utils.paligemma_hooks import (
    set_block_attn_hooks_paligemma,
    remove_wrapper_paligemma,
)


class FeatureAblator:
    """Ablates SAE features by zeroing them in the hidden state."""

    def __init__(self, model, sae, layer_idx: int, activation_site: str = "residual"):
        self.model = model
        self.sae = sae
        self.layer_idx = layer_idx
        self.activation_site = activation_site
        self.hook_manager = HookManager(model)

    def create_ablation_hook(
        self,
        feature_indices: List[int],
        positions: Optional[List[int]] = None,
        mode: str = "residual",
        delta_scale: float = 1.0,
        operation: str = "zero",
        operation_scale: float = 1.0,
        diagnostics_buffer: Optional[List[Dict[str, float]]] = None,
    ):
        feature_indices = torch.tensor(feature_indices, dtype=torch.long)

        def hook(module, inputs, output):
            acts = output[0] if isinstance(output, (tuple, list)) else output
            sae_param = next(self.sae.parameters())
            acts_dtype = acts.dtype
            acts_device = acts.device
            sae_device = sae_param.device
            sae_dtype = sae_param.dtype
            acts_for_sae = acts.to(device=sae_device, dtype=sae_dtype)
            feats_full = self.sae.encode(acts_for_sae)
            idx = feature_indices.to(feats_full.device)
            feats_mod = feats_full.clone()
            op = str(operation).lower()
            if op == "scaled_zero":
                feats_mod[:, idx] = feats_mod[:, idx] * (1.0 - float(operation_scale))
            elif op == "signed_scale":
                feats_mod[:, idx] = feats_mod[:, idx] * float(-abs(operation_scale))
            else:
                feats_mod[:, idx] = 0.0

            if mode == "replace":
                recon_mod = self.sae.decode(feats_mod, target_shape=acts.shape)
                out = recon_mod.to(device=acts_device, dtype=acts_dtype)
                out = self._apply_positions(acts, out, positions)
            else:
                recon_full = self.sae.decode(feats_full, target_shape=acts.shape)
                recon_mod = self.sae.decode(feats_mod, target_shape=acts.shape)
                delta = (recon_mod - recon_full).to(device=acts_device, dtype=acts_dtype)
                if delta_scale != 1.0:
                    delta = delta * delta_scale
                if positions:
                    mask = torch.zeros_like(acts, dtype=delta.dtype, device=acts_device)
                    mask[:, positions, :] = 1.0
                    delta = delta * mask
                out = acts + delta

            if diagnostics_buffer is not None:
                delta_tensor = (out - acts).detach().float()
                acts_tensor = acts.detach().float()
                delta_norm = torch.linalg.norm(delta_tensor, dim=-1).mean().item()
                acts_norm = torch.linalg.norm(acts_tensor, dim=-1).mean().item()
                diagnostics_buffer.append(
                    {
                        "delta_norm": float(delta_norm),
                        "acts_norm": float(acts_norm),
                        "relative_norm": float(delta_norm / (acts_norm + 1e-8)),
                        "affected_tokens": float(len(positions) if positions else acts.shape[1]),
                    }
                )

            if isinstance(output, tuple):
                return (out,) + output[1:]
            if isinstance(output, list):
                new_output = list(output)
                new_output[0] = out
                return new_output
            return out

        return hook

    def run_with_ablation(self, sample: Tuple, feature_indices: List[int], tokenizer) -> Tuple[str, float]:
        input_ids, pixel_values, _, _ = sample
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = input_ids.to(device=device)
        pixel_values = pixel_values.to(device=device)

        layer = get_target_module(self.model, self.layer_idx, self.activation_site)
        hook = layer.register_forward_hook(self.create_ablation_hook(feature_indices))

        inps = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 1,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        try:
            with torch.inference_mode():
                output = self.model.generate(**inps)
        finally:
            hook.remove()

        answer_token_id = output["sequences"][:, 0]
        logits_first = output["scores"][0]
        prob = torch.softmax(logits_first, dim=-1)[0][answer_token_id].item()
        prediction = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)[0].strip().lower()
        return prediction, prob

    def compute_baseline_cache(
        self,
        dataset,
        logprob_normalize: bool = True,
        max_samples: Optional[int] = None,
        score_options: bool = True,
        show_progress: bool = False,
        progress_desc: str = "Baseline",
    ) -> List[Dict[str, Any]]:
        baseline_cache: List[Dict[str, Any]] = []
        data_loader = dataset.create_dataloader()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        iterator = zip(data_loader, dataset.questions)
        if show_progress:
            total = len(dataset.questions)
            if max_samples is not None:
                total = min(total, max_samples)
            iterator = tqdm(iterator, total=total, desc=progress_desc)

        for idx, (batch, line) in enumerate(iterator):
            if max_samples is not None and idx >= max_samples:
                break
            input_ids, pixel_values, _, _ = batch
            input_ids = input_ids.to(device=device)
            pixel_values = pixel_values.to(device=device)
            baseline_record = self._compute_baseline_record(
                input_ids=input_ids,
                pixel_values=pixel_values,
                dataset=dataset,
                line=line,
                logprob_normalize=logprob_normalize,
                score_options=score_options,
            )
            baseline_cache.append(baseline_record)
        return baseline_cache

    def batch_ablation_experiment(
        self,
        dataset,
        feature_indices: List[int],
        position_type: str = "all",
        mode: str = "residual",
        delta_scale: float = 1.0,
        operation: str = "zero",
        operation_scale: float = 1.0,
        logprob_normalize: bool = True,
        attn_block_config: Optional[dict] = None,
        apply_sae: bool = True,
        attn_block_resolver: Optional[callable] = None,
        show_progress: bool = False,
        max_samples: Optional[int] = None,
        baseline_cache: Optional[Any] = None,
        score_options: bool = True,
    ) -> List[dict]:
        results = []
        data_loader = dataset.create_dataloader()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        image_token_count = get_image_token_count(self.model)

        iterator = zip(data_loader, dataset.questions)
        if show_progress:
            total = len(dataset.questions)
            if max_samples is not None:
                total = min(total, max_samples)
            iterator = tqdm(iterator, total=total, desc="Ablation")

        for idx, (batch, line) in enumerate(iterator):
            if max_samples is not None and idx >= max_samples:
                break
            input_ids, pixel_values, _, _ = batch
            input_ids = input_ids.to(device=device)
            pixel_values = pixel_values.to(device=device)

            inps = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 1,
                "use_cache": True,
                "return_dict_in_generate": True,
                "output_scores": True,
                "pad_token_id": dataset.tokenizer.eos_token_id,
            }

            baseline_record = self._resolve_cached_baseline(
                baseline_cache=baseline_cache,
                sample_idx=idx,
                question_id=line.get("q_id"),
            )
            if baseline_record is None:
                baseline_record = self._compute_baseline_record(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    dataset=dataset,
                    line=line,
                    logprob_normalize=logprob_normalize,
                    score_options=score_options,
                )
            baseline_pred = baseline_record.get("baseline_pred", "")
            baseline_prob = baseline_record.get("baseline_prob", 0.0)
            baseline_gt_prob = baseline_record.get("baseline_gt_prob")
            baseline_true_lp = baseline_record.get("baseline_true_logprob")
            baseline_false_lp = baseline_record.get("baseline_false_logprob")
            baseline_margin = baseline_record.get("baseline_margin")

            gt_token_id = self._get_answer_token_id(
                dataset.dataset_dict[line["q_id"]].get("answer", ""),
                dataset.tokenizer,
            )
            true_option = dataset.dataset_dict[line["q_id"]].get("true option", "").strip()
            false_option = dataset.dataset_dict[line["q_id"]].get("false option", "").strip()

            positions = self._resolve_positions(
                position_type,
                input_ids,
                image_token_count,
                dataset,
                line,
            )
            layer = get_target_module(self.model, self.layer_idx, self.activation_site)
            hook = None
            sample_diagnostics: List[Dict[str, float]] = []
            if apply_sae:
                hook = layer.register_forward_hook(
                    self.create_ablation_hook(
                        feature_indices,
                        positions=positions,
                        mode=mode,
                        delta_scale=delta_scale,
                        operation=operation,
                        operation_scale=operation_scale,
                        diagnostics_buffer=sample_diagnostics,
                    )
                )
            attn_hooks = None
            resolved_block_config = attn_block_config
            if attn_block_resolver is not None:
                resolved_block_config = attn_block_resolver(
                    input_ids,
                    pixel_values,
                    dataset,
                    line,
                )
            if resolved_block_config:
                attn_hooks = set_block_attn_hooks_paligemma(self.model, resolved_block_config)
            try:
                with torch.inference_mode():
                    ablated = self.model.generate(**inps)
                    if score_options:
                        ablated_true_lp = sequence_logprob(
                            self.model,
                            dataset.tokenizer,
                            input_ids,
                            pixel_values,
                            true_option,
                            normalize=logprob_normalize,
                        )
                        ablated_false_lp = sequence_logprob(
                            self.model,
                            dataset.tokenizer,
                            input_ids,
                            pixel_values,
                            false_option,
                            normalize=logprob_normalize,
                        )
                    else:
                        ablated_true_lp = None
                        ablated_false_lp = None
            finally:
                if hook:
                    hook.remove()
                if attn_hooks:
                    remove_wrapper_paligemma(self.model, attn_hooks)

            ablated_pred = dataset.tokenizer.batch_decode(
                ablated["sequences"], skip_special_tokens=True
            )[0].strip().lower()
            ablated_logits = ablated["scores"][0]
            ablated_probs = torch.softmax(ablated_logits, dim=-1)
            ablated_prob = ablated_probs[0][ablated["sequences"][:, 0]].item()
            ablated_gt_prob = (
                ablated_probs[0][gt_token_id].item() if gt_token_id is not None else None
            )
            ablated_margin = (
                ablated_true_lp - ablated_false_lp
                if ablated_true_lp is not None and ablated_false_lp is not None
                else None
            )

            answer = baseline_record.get(
                "answer",
                dataset.dataset_dict[line["q_id"]].get("answer", "").strip().lower(),
            )
            results.append(
                {
                    "question_id": line["q_id"],
                    "answer": answer,
                    "baseline_pred": baseline_pred,
                    "ablated_pred": ablated_pred,
                    "baseline_prob": baseline_prob,
                    "ablated_prob": ablated_prob,
                    "baseline_gt_prob": baseline_gt_prob,
                    "ablated_gt_prob": ablated_gt_prob,
                    "baseline_true_logprob": baseline_true_lp,
                    "baseline_false_logprob": baseline_false_lp,
                    "baseline_margin": baseline_margin,
                    "ablated_true_logprob": ablated_true_lp,
                    "ablated_false_logprob": ablated_false_lp,
                    "ablated_margin": ablated_margin,
                    "perturb_mean_delta_norm": (
                        sum(d["delta_norm"] for d in sample_diagnostics) / len(sample_diagnostics)
                        if sample_diagnostics else None
                    ),
                    "perturb_mean_acts_norm": (
                        sum(d["acts_norm"] for d in sample_diagnostics) / len(sample_diagnostics)
                        if sample_diagnostics else None
                    ),
                    "perturb_relative_norm": (
                        sum(d["relative_norm"] for d in sample_diagnostics) / len(sample_diagnostics)
                        if sample_diagnostics else None
                    ),
                    "perturb_calls": len(sample_diagnostics),
                    "perturb_mode": mode,
                    "perturb_operation": operation,
                    "perturb_position_count": len(positions) if positions else None,
                }
            )

        return results

    def _compute_baseline_record(
        self,
        input_ids,
        pixel_values,
        dataset,
        line: Dict[str, Any],
        logprob_normalize: bool = True,
        score_options: bool = True,
    ) -> Dict[str, Any]:
        inps = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 1,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": dataset.tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            baseline = self.model.generate(**inps)

        baseline_pred = dataset.tokenizer.batch_decode(
            baseline["sequences"], skip_special_tokens=True
        )[0].strip().lower()
        baseline_logits = baseline["scores"][0]
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        baseline_prob = baseline_probs[0][baseline["sequences"][:, 0]].item()
        gt_token_id = self._get_answer_token_id(
            dataset.dataset_dict[line["q_id"]].get("answer", ""),
            dataset.tokenizer,
        )
        baseline_gt_prob = (
            baseline_probs[0][gt_token_id].item() if gt_token_id is not None else None
        )

        if score_options:
            true_option = dataset.dataset_dict[line["q_id"]].get("true option", "").strip()
            false_option = dataset.dataset_dict[line["q_id"]].get("false option", "").strip()
            baseline_true_lp = sequence_logprob(
                self.model,
                dataset.tokenizer,
                input_ids,
                pixel_values,
                true_option,
                normalize=logprob_normalize,
            )
            baseline_false_lp = sequence_logprob(
                self.model,
                dataset.tokenizer,
                input_ids,
                pixel_values,
                false_option,
                normalize=logprob_normalize,
            )
        else:
            baseline_true_lp = None
            baseline_false_lp = None
        baseline_margin = (
            baseline_true_lp - baseline_false_lp
            if baseline_true_lp is not None and baseline_false_lp is not None
            else None
        )
        answer = dataset.dataset_dict[line["q_id"]].get("answer", "").strip().lower()
        return {
            "question_id": line["q_id"],
            "answer": answer,
            "baseline_pred": baseline_pred,
            "baseline_prob": baseline_prob,
            "baseline_gt_prob": baseline_gt_prob,
            "baseline_true_logprob": baseline_true_lp,
            "baseline_false_logprob": baseline_false_lp,
            "baseline_margin": baseline_margin,
        }

    @staticmethod
    def _resolve_cached_baseline(
        baseline_cache: Optional[Any],
        sample_idx: int,
        question_id: Any,
    ) -> Optional[Dict[str, Any]]:
        if baseline_cache is None:
            return None
        if isinstance(baseline_cache, dict):
            entry = baseline_cache.get(question_id)
            if entry is None:
                entry = baseline_cache.get(str(question_id))
            return entry if isinstance(entry, dict) else None
        if isinstance(baseline_cache, list):
            if sample_idx < 0 or sample_idx >= len(baseline_cache):
                return None
            entry = baseline_cache[sample_idx]
            if not isinstance(entry, dict):
                return None
            cached_qid = entry.get("question_id")
            if cached_qid is None or str(cached_qid) == str(question_id):
                return entry
        return None

    def compute_ablation_effect(self, results: List[dict]) -> dict:
        baseline_correct = [r["baseline_pred"] == r["answer"] for r in results]
        ablated_correct = [r["ablated_pred"] == r["answer"] for r in results]
        baseline_acc = sum(baseline_correct) / max(1, len(baseline_correct))
        ablated_acc = sum(ablated_correct) / max(1, len(ablated_correct))

        prob_drop = [r["baseline_prob"] - r["ablated_prob"] for r in results]
        gt_pairs = [
            (r["baseline_gt_prob"], r["ablated_gt_prob"])
            for r in results
            if r["baseline_gt_prob"] is not None and r["ablated_gt_prob"] is not None
        ]
        gt_baseline = [b for b, _ in gt_pairs]
        gt_ablated = [a for _, a in gt_pairs]
        gt_drop = [b - a for b, a in gt_pairs]
        margin_pairs = [
            (r.get("baseline_margin"), r.get("ablated_margin"))
            for r in results
            if r.get("baseline_margin") is not None and r.get("ablated_margin") is not None
        ]
        margin_base = [b for b, _ in margin_pairs]
        margin_abl = [a for _, a in margin_pairs]
        margin_drop = [b - a for b, a in margin_pairs]
        rel_perturb = [
            r.get("perturb_relative_norm")
            for r in results
            if r.get("perturb_relative_norm") is not None
        ]
        return {
            "baseline_accuracy": baseline_acc,
            "ablated_accuracy": ablated_acc,
            "accuracy_drop": baseline_acc - ablated_acc,
            "mean_probability_drop": sum(prob_drop) / max(1, len(prob_drop)),
            "baseline_gt_probability": sum(gt_baseline) / max(1, len(gt_baseline)) if gt_baseline else None,
            "ablated_gt_probability": sum(gt_ablated) / max(1, len(gt_ablated)) if gt_ablated else None,
            "mean_gt_probability_drop": sum(gt_drop) / max(1, len(gt_drop)) if gt_drop else None,
            "baseline_margin": sum(margin_base) / max(1, len(margin_base)) if margin_base else None,
            "ablated_margin": sum(margin_abl) / max(1, len(margin_abl)) if margin_abl else None,
            "mean_margin_drop": sum(margin_drop) / max(1, len(margin_drop)) if margin_drop else None,
            "mean_relative_perturbation": sum(rel_perturb) / len(rel_perturb) if rel_perturb else None,
        }

    def _resolve_positions(
        self,
        position_type: str,
        input_ids,
        image_token_count: int,
        dataset,
        line,
    ) -> Optional[List[int]]:
        if position_type in (None, "all"):
            return None

        seq_len = input_ids.shape[-1]

        if position_type == "image":
            return list(range(0, min(image_token_count, seq_len)))

        if position_type == "last":
            return [seq_len - 1]

        question_text = dataset.dataset_dict[line["q_id"]].get("question", "")
        question_range = token_utils.get_question_token_range(
            input_ids[0] if input_ids.dim() > 1 else input_ids,
            image_token_count=1,
            question_text=question_text,
            tokenizer=dataset.tokenizer,
            image_token_index=None,
        )
        if not question_range:
            return []

        if position_type == "question":
            return question_range

        if position_type == "attribute":
            attr_positions = []
            for attr in line.get("attribute_tokens", []):
                attr_positions.extend(attr.get("positions", []))
            if not attr_positions:
                return question_range
            start = question_range[0]
            end = question_range[-1]
            positions = [start + pos for pos in attr_positions]
            positions = [pos for pos in positions if start <= pos <= end]
            return sorted(set(positions))

        return question_range

    @staticmethod
    def _apply_positions(original: torch.Tensor, replaced: torch.Tensor, positions: Optional[List[int]]):
        if not positions:
            return replaced
        out = original.clone()
        out[:, positions, :] = replaced[:, positions, :]
        return out

    @staticmethod
    def _get_answer_token_id(answer: str, tokenizer) -> Optional[int]:
        if not answer or tokenizer is None:
            return None
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        if not token_ids:
            return None
        return token_ids[0]

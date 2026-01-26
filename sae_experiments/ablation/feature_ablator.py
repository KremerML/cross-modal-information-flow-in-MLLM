"""Feature ablation utilities."""

from typing import List, Optional, Tuple

import torch

from tqdm import tqdm

try:
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    IMAGE_TOKEN_INDEX = -200

from sae_experiments.utils import token_utils
from sae_experiments.utils.hook_utils import HookManager
from methods import remove_wrapper_llava, set_block_attn_hooks_llava


class FeatureAblator:
    """Ablates SAE features by zeroing them in the hidden state."""

    def __init__(self, model, sae, layer_idx: int):
        self.model = model
        self.sae = sae
        self.layer_idx = layer_idx
        self.hook_manager = HookManager(model)

    def create_ablation_hook(
        self,
        feature_indices: List[int],
        positions: Optional[List[int]] = None,
        mode: str = "residual",
        delta_scale: float = 1.0,
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

            if isinstance(output, tuple):
                return (out,) + output[1:]
            if isinstance(output, list):
                new_output = list(output)
                new_output[0] = out
                return new_output
            return out

        return hook

    def run_with_ablation(self, sample: Tuple, feature_indices: List[int], tokenizer) -> Tuple[str, float]:
        input_ids, image_tensor, image_sizes, _, _ = sample
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = input_ids.to(device=device)
        image_tensor = [img.to(device=device) for img in image_tensor]

        layer = self._get_layer_module(self.layer_idx)
        hook = layer.register_forward_hook(self.create_ablation_hook(feature_indices))

        inps = {
            "inputs": input_ids,
            "images": image_tensor,
            "image_sizes": image_sizes,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 1,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            output = self.model.generate(**inps)
        hook.remove()

        answer_token_id = output["sequences"][:, 0]
        logits_first = output["scores"][0]
        prob = torch.softmax(logits_first, dim=-1)[0][answer_token_id].item()
        prediction = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)[0].strip().lower()
        return prediction, prob

    def batch_ablation_experiment(
        self,
        dataset,
        feature_indices: List[int],
        position_type: str = "all",
        mode: str = "residual",
        delta_scale: float = 1.0,
        logprob_normalize: bool = True,
        attn_block_config: Optional[dict] = None,
        apply_sae: bool = True,
        attn_block_resolver: Optional[callable] = None,
        show_progress: bool = False,
        max_samples: Optional[int] = None,
    ) -> List[dict]:
        results = []
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
            iterator = tqdm(iterator, total=total, desc="Ablation")
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
            true_option = dataset.dataset_dict[line["q_id"]].get("true option", "").strip()
            false_option = dataset.dataset_dict[line["q_id"]].get("false option", "").strip()
            baseline_true_lp = self._sequence_logprob(
                input_ids,
                image_tensor,
                image_sizes,
                true_option,
                dataset.tokenizer,
                normalize=logprob_normalize,
            )
            baseline_false_lp = self._sequence_logprob(
                input_ids,
                image_tensor,
                image_sizes,
                false_option,
                dataset.tokenizer,
                normalize=logprob_normalize,
            )
            baseline_margin = (
                baseline_true_lp - baseline_false_lp
                if baseline_true_lp is not None and baseline_false_lp is not None
                else None
            )

            positions = self._resolve_positions(
                position_type,
                input_ids,
                image_tensor,
                image_sizes,
                dataset,
                line,
            )
            layer = self._get_layer_module(self.layer_idx)
            hook = None
            if apply_sae:
                hook = layer.register_forward_hook(
                    self.create_ablation_hook(
                        feature_indices,
                        positions=positions,
                        mode=mode,
                        delta_scale=delta_scale,
                    )
                )
            attn_hooks = None
            resolved_block_config = attn_block_config
            if attn_block_resolver is not None:
                resolved_block_config = attn_block_resolver(
                    input_ids,
                    image_tensor,
                    image_sizes,
                    dataset,
                    line,
                )
            if resolved_block_config:
                attn_hooks = set_block_attn_hooks_llava(self.model, resolved_block_config)
            with torch.inference_mode():
                ablated = self.model.generate(**inps)
                ablated_true_lp = self._sequence_logprob(
                    input_ids,
                    image_tensor,
                    image_sizes,
                    true_option,
                    dataset.tokenizer,
                    normalize=logprob_normalize,
                )
                ablated_false_lp = self._sequence_logprob(
                    input_ids,
                    image_tensor,
                    image_sizes,
                    false_option,
                    dataset.tokenizer,
                    normalize=logprob_normalize,
                )
            if hook:
                hook.remove()
            if attn_hooks:
                remove_wrapper_llava(self.model, attn_hooks)

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

            answer = dataset.dataset_dict[line["q_id"]].get("answer", "").strip().lower()
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
                }
            )

        return results

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
        }

    def _get_layer_module(self, layer_idx: int):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        if hasattr(self.model, "layers"):
            return self.model.layers[layer_idx]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        raise ValueError("Unsupported model type for layer access")

    def _estimate_image_token_count(self, input_ids, image_tensor, image_sizes) -> int:
        if not hasattr(self.model, "prepare_inputs_labels_for_multimodal"):
            return 0
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
            image_token_count = inputs_embeds.shape[1] - (input_ids.shape[-1] - 1)
            return int(image_token_count)
        except Exception:
            return 0

    def _resolve_positions(
        self,
        position_type: str,
        input_ids,
        image_tensor,
        image_sizes,
        dataset,
        line,
    ) -> Optional[List[int]]:
        if position_type in (None, "all"):
            return None

        question_text = dataset.dataset_dict[line["q_id"]].get("question", "")
        image_token_count = self._estimate_image_token_count(
            input_ids, image_tensor, image_sizes
        )
        question_range = token_utils.get_question_token_range(
            input_ids[0],
            image_token_count,
            question_text=question_text,
            tokenizer=dataset.tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
        )
        if not question_range:
            return []

        if position_type == "question":
            return question_range

        if position_type == "last":
            ntoks = input_ids.shape[-1] + image_token_count - 1
            return [max(0, ntoks - 1)]

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

    def _sequence_logprob(
        self,
        input_ids,
        image_tensor,
        image_sizes,
        answer_text: str,
        tokenizer,
        normalize: bool = True,
    ) -> Optional[float]:
        if not answer_text or tokenizer is None:
            return None
        answer_ids = tokenizer.encode(f" {answer_text.strip()}", add_special_tokens=False)
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

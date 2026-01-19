"""Feature ablation utilities."""

from typing import List, Tuple

import torch

from sae_experiments.utils.hook_utils import HookManager


class FeatureAblator:
    """Ablates SAE features by zeroing them in the hidden state."""

    def __init__(self, model, sae, layer_idx: int):
        self.model = model
        self.sae = sae
        self.layer_idx = layer_idx
        self.hook_manager = HookManager(model)

    def create_ablation_hook(self, feature_indices: List[int]):
        feature_indices = torch.tensor(feature_indices, dtype=torch.long)

        def hook(module, inputs, output):
            acts = output
            if isinstance(acts, (tuple, list)):
                acts = acts[0]
            feats = self.sae.encode(acts)
            idx = feature_indices.to(feats.device)
            feats[:, idx] = 0.0
            recon = self.sae.decode(feats, target_shape=acts.shape)
            return recon

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

    def batch_ablation_experiment(self, dataset, feature_indices: List[int]) -> List[dict]:
        results = []
        data_loader = dataset.create_dataloader()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        for batch, line in zip(data_loader, dataset.questions):
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
            baseline_prob = torch.softmax(baseline["scores"][0], dim=-1)[0][
                baseline["sequences"][:, 0]
            ].item()

            layer = self._get_layer_module(self.layer_idx)
            hook = layer.register_forward_hook(self.create_ablation_hook(feature_indices))
            with torch.inference_mode():
                ablated = self.model.generate(**inps)
            hook.remove()

            ablated_pred = dataset.tokenizer.batch_decode(
                ablated["sequences"], skip_special_tokens=True
            )[0].strip().lower()
            ablated_prob = torch.softmax(ablated["scores"][0], dim=-1)[0][
                ablated["sequences"][:, 0]
            ].item()

            answer = dataset.dataset_dict[line["q_id"]].get("answer", "").strip().lower()
            results.append(
                {
                    "question_id": line["q_id"],
                    "answer": answer,
                    "baseline_pred": baseline_pred,
                    "ablated_pred": ablated_pred,
                    "baseline_prob": baseline_prob,
                    "ablated_prob": ablated_prob,
                }
            )

        return results

    def compute_ablation_effect(self, results: List[dict]) -> dict:
        baseline_correct = [r["baseline_pred"] == r["answer"] for r in results]
        ablated_correct = [r["ablated_pred"] == r["answer"] for r in results]
        baseline_acc = sum(baseline_correct) / max(1, len(baseline_correct))
        ablated_acc = sum(ablated_correct) / max(1, len(ablated_correct))

        prob_drop = [r["baseline_prob"] - r["ablated_prob"] for r in results]
        return {
            "baseline_accuracy": baseline_acc,
            "ablated_accuracy": ablated_acc,
            "accuracy_drop": baseline_acc - ablated_acc,
            "mean_probability_drop": sum(prob_drop) / max(1, len(prob_drop)),
        }

    def _get_layer_module(self, layer_idx: int):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        if hasattr(self.model, "layers"):
            return self.model.layers[layer_idx]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        raise ValueError("Unsupported model type for layer access")

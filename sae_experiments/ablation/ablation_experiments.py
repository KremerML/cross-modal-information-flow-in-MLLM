"""High-level ablation experiment runner."""

from typing import Dict, List
import json
import os
import random

import torch

from sae_experiments.ablation.feature_ablator import FeatureAblator
from methods import trace_with_attn_block_llava


class AblationExperiment:
    """Runs ablation studies for identified features."""

    def __init__(self, model, sae, config):
        self.model = model
        self.sae = sae
        self.config = config

    def run_three_condition_test(self, dataset, binding_features: List[int]) -> Dict[str, dict]:
        ablator = FeatureAblator(self.model, self.sae, self.config.get("model", {}).get("target_layer", 0))
        ablation_cfg = self.config.get("ablation", {})
        position_type = ablation_cfg.get("position_type", "all")
        mode = ablation_cfg.get("mode", "residual")
        delta_scale = ablation_cfg.get("delta_scale", 1.0)

        binding_results = ablator.batch_ablation_experiment(
            dataset,
            binding_features,
            position_type=position_type,
            mode=mode,
            delta_scale=delta_scale,
        )
        binding_summary = ablator.compute_ablation_effect(binding_results)

        n_random = self.config.get("ablation", {}).get("n_random_features", len(binding_features))
        random_features = random.sample(range(self.sae.n_features), k=min(n_random, self.sae.n_features))
        random_results = ablator.batch_ablation_experiment(
            dataset,
            random_features,
            position_type=position_type,
            mode=mode,
            delta_scale=delta_scale,
        )
        random_summary = ablator.compute_ablation_effect(random_results)

        baseline_summary = {
            "baseline_accuracy": binding_summary["baseline_accuracy"],
        }

        return {
            "baseline": baseline_summary,
            "binding": binding_summary,
            "random": random_summary,
            "binding_results": binding_results,
            "random_results": random_results,
            "ablation_settings": {
                "position_type": position_type,
                "mode": mode,
                "delta_scale": delta_scale,
            },
            "evaluation_settings": {
                "primary_metric": self.config.get("evaluation", {}).get("primary_metric", "pred_token_prob"),
            },
        }

    def test_task_specificity(self, binding_features: List[int], choose_attr_data, choose_rel_data) -> Dict[str, dict]:
        attr_results = self.run_three_condition_test(choose_attr_data, binding_features)
        rel_results = self.run_three_condition_test(choose_rel_data, binding_features)
        return {
            "choose_attr": attr_results,
            "choose_rel": rel_results,
        }

    def feature_importance_ranking(self, feature_list: List[int], dataset) -> Dict[int, float]:
        ablator = FeatureAblator(self.model, self.sae, self.config.get("model", {}).get("target_layer", 0))
        ranking = {}
        for feature_idx in feature_list:
            results = ablator.batch_ablation_experiment(dataset, [feature_idx])
            summary = ablator.compute_ablation_effect(results)
            ranking[feature_idx] = summary.get("accuracy_drop", 0.0)
        return ranking

    def save_results(self, results: Dict, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    def run_attention_knockout_baseline(self, dataset, block_config: Dict[int, list]) -> Dict[str, float]:
        """Optional baseline using the existing attention knockout implementation."""
        data_loader = dataset.create_dataloader()
        scores = []
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
            output = self.model.generate(**inps)
            first_answer_token_id = output["sequences"][:, 0]
            base_score = trace_with_attn_block_llava(
                self.model,
                inps,
                block_config,
                first_answer_token_id,
                "AttentionKnockout",
                self.model.config._name_or_path,
            )
            scores.append(base_score.item() if hasattr(base_score, "item") else float(base_score))
        return {"mean_score": sum(scores) / max(1, len(scores))}

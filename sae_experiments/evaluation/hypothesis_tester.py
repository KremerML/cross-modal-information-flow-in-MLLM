"""Hypothesis testing for attribute-binding features."""

from typing import Dict

from sae_experiments.ablation import statistical_analysis as stats


class HypothesisTester:
    """Run hypothesis tests on ablation results."""

    def __init__(self, config):
        self.config = config

    def test_causal_necessity(self, ablation_results: Dict) -> Dict[str, float]:
        binding = ablation_results.get("binding_results", [])
        random_results = ablation_results.get("random_results", [])
        metric = self.config.get("evaluation", {}).get("primary_metric", "pred_token_prob")

        binding_drop = self._extract_drops(binding, metric)
        random_drop = self._extract_drops(random_results, metric)

        t_stat, p_val = stats.paired_t_test(binding_drop, random_drop)
        effect = stats.effect_size_cohens_d(binding_drop, random_drop)

        supported = bool(p_val < self.config.get("evaluation", {}).get("significance_level", 0.05))
        return {
            "hypothesis_supported": supported,
            "p_value": p_val,
            "effect_size": effect,
            "primary_metric": metric,
        }

    def test_task_specificity(self, choose_attr_results: Dict, choose_rel_results: Dict) -> Dict[str, float]:
        attr_drop = choose_attr_results.get("binding", {}).get("accuracy_drop", 0.0)
        rel_drop = choose_rel_results.get("binding", {}).get("accuracy_drop", 0.0)
        specificity_score = attr_drop - rel_drop
        return {
            "specificity_score": specificity_score,
        }

    def test_feature_interpretability(self, feature_catalog) -> Dict[str, float]:
        categories = feature_catalog.categorize_features()
        return {"num_categories": float(len(categories))}

    def generate_hypothesis_report(self, ablation_results: Dict) -> Dict[str, float]:
        return self.test_causal_necessity(ablation_results)

    @staticmethod
    def _extract_drops(results, metric: str):
        if metric == "gt_token_prob":
            pairs = [
                (r.get("baseline_gt_prob"), r.get("ablated_gt_prob"))
                for r in results
                if r.get("baseline_gt_prob") is not None and r.get("ablated_gt_prob") is not None
            ]
            return [b - a for b, a in pairs]
        return [r.get("baseline_prob", 0.0) - r.get("ablated_prob", 0.0) for r in results]

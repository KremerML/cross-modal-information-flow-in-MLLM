"""Statistical tests and plots for ablation results."""

from typing import Dict, Iterable, Tuple

import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

import matplotlib.pyplot as plt


def compute_accuracy(predictions: Iterable[str], ground_truth: Iterable[str]) -> float:
    preds = list(predictions)
    gt = list(ground_truth)
    if not preds:
        return 0.0
    correct = [p == g for p, g in zip(preds, gt)]
    return sum(correct) / len(correct)


def compute_probability_drop(baseline: Iterable[float], ablated: Iterable[float]) -> float:
    base = np.array(list(baseline))
    abl = np.array(list(ablated))
    if base.size == 0:
        return 0.0
    return float(np.mean(base - abl))


def paired_t_test(baseline_probs: Iterable[float], ablated_probs: Iterable[float]) -> Tuple[float, float]:
    base = np.array(list(baseline_probs))
    abl = np.array(list(ablated_probs))
    if stats is None or base.size == 0:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(base, abl)
    return float(t_stat), float(p_val)


def effect_size_cohens_d(baseline: Iterable[float], ablated: Iterable[float]) -> float:
    base = np.array(list(baseline))
    abl = np.array(list(ablated))
    if base.size == 0:
        return 0.0
    diff = base - abl
    return float(np.mean(diff) / (np.std(diff) + 1e-8))


def bootstrap_confidence_interval(data: Iterable[float], n_bootstrap: int = 1000) -> Tuple[float, float]:
    values = np.array(list(data))
    if values.size == 0:
        return 0.0, 0.0
    samples = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(values, size=len(values), replace=True)
        samples.append(np.mean(resample))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def plot_ablation_comparison(results_dict: Dict[str, Dict], save_path: str) -> None:
    labels = ["baseline", "binding", "random"]
    values = [
        results_dict.get("baseline", {}).get("baseline_accuracy", 0.0),
        results_dict.get("binding", {}).get("ablated_accuracy", 0.0),
        results_dict.get("random", {}).get("ablated_accuracy", 0.0),
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.title("Ablation comparison")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_statistical_report(results: Dict) -> Dict[str, float]:
    binding = results.get("binding_results", [])
    baseline_probs = [r["baseline_prob"] for r in binding]
    ablated_probs = [r["ablated_prob"] for r in binding]
    t_stat, p_val = paired_t_test(baseline_probs, ablated_probs)
    effect = effect_size_cohens_d(baseline_probs, ablated_probs)
    ci_low, ci_high = bootstrap_confidence_interval(baseline_probs)
    report = {
        "t_stat": t_stat,
        "p_value": p_val,
        "effect_size": effect,
        "baseline_ci_low": ci_low,
        "baseline_ci_high": ci_high,
    }

    gt_pairs = [
        (r.get("baseline_gt_prob"), r.get("ablated_gt_prob"))
        for r in binding
        if r.get("baseline_gt_prob") is not None and r.get("ablated_gt_prob") is not None
    ]
    if gt_pairs:
        gt_baseline = [b for b, _ in gt_pairs]
        gt_ablated = [a for _, a in gt_pairs]
        gt_t, gt_p = paired_t_test(gt_baseline, gt_ablated)
        gt_effect = effect_size_cohens_d(gt_baseline, gt_ablated)
        report.update(
            {
                "gt_t_stat": gt_t,
                "gt_p_value": gt_p,
                "gt_effect_size": gt_effect,
                "gt_mean_drop": compute_probability_drop(gt_baseline, gt_ablated),
            }
        )
    return report

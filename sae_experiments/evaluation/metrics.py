"""Evaluation metrics for SAE experiments."""

from typing import Iterable

import numpy as np


def accuracy_at_k(predictions: Iterable[str], targets: Iterable[str], k: int = 1) -> float:
    preds = list(predictions)
    targs = list(targets)
    if not preds:
        return 0.0
    correct = [p == t for p, t in zip(preds, targs)]
    return sum(correct) / len(correct)


def mean_probability_drop(baseline_probs: Iterable[float], ablated_probs: Iterable[float]) -> float:
    baseline = np.array(list(baseline_probs))
    ablated = np.array(list(ablated_probs))
    if baseline.size == 0:
        return 0.0
    return float(np.mean(baseline - ablated))


def forced_choice_margin(true_logprobs: Iterable[float], false_logprobs: Iterable[float]) -> float:
    true_vals = np.array(list(true_logprobs))
    false_vals = np.array(list(false_logprobs))
    if true_vals.size == 0 or false_vals.size == 0:
        return 0.0
    return float(np.mean(true_vals - false_vals))


def mean_margin_drop(baseline_margins: Iterable[float], ablated_margins: Iterable[float]) -> float:
    base = np.array(list(baseline_margins))
    abl = np.array(list(ablated_margins))
    if base.size == 0:
        return 0.0
    return float(np.mean(base - abl))


def feature_selectivity(feature_acts: np.ndarray, correct_mask: np.ndarray) -> float:
    if feature_acts.size == 0:
        return 0.0
    correct_mean = feature_acts[correct_mask].mean()
    incorrect_mean = feature_acts[~correct_mask].mean()
    return float((correct_mean + 1e-8) / (incorrect_mean + 1e-8))


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.size == 0:
        return 0.0
    return float(np.mean((original - reconstructed) ** 2))


def sparsity_metric(feature_activations: np.ndarray) -> dict:
    if feature_activations.size == 0:
        return {"l0": 0.0, "l1": 0.0}
    l0 = np.mean((feature_activations != 0).sum(axis=1))
    l1 = np.mean(np.abs(feature_activations))
    return {"l0": float(l0), "l1": float(l1)}


def cross_task_transfer(features: np.ndarray, task1_data: np.ndarray, task2_data: np.ndarray) -> float:
    if features.size == 0:
        return 0.0
    proj1 = task1_data @ features.T
    proj2 = task2_data @ features.T
    return float(np.corrcoef(proj1.flatten(), proj2.flatten())[0, 1])

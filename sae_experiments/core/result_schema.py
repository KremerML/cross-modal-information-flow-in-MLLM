"""Lightweight validation for ablation result payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def validate_ablation_results(results: Dict[str, Any]) -> None:
    """Validate ablation results structure.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    if not isinstance(results, dict):
        raise ValueError("Ablation results must be a dictionary.")

    required_top_level = ["baseline", "binding", "random", "binding_results", "random_results"]
    missing = [key for key in required_top_level if key not in results]
    if missing:
        raise ValueError(f"Missing top-level ablation keys: {missing}")

    _validate_summary("baseline", results.get("baseline", {}), required_keys=["baseline_accuracy"])
    _validate_summary("binding", results.get("binding", {}))
    _validate_summary("random", results.get("random", {}))

    _validate_results_rows("binding_results", results.get("binding_results", []))
    _validate_results_rows("random_results", results.get("random_results", []))

    set_summaries = results.get("random_set_summaries")
    if set_summaries is not None:
        if not isinstance(set_summaries, list):
            raise ValueError("random_set_summaries must be a list when present.")
        for idx, item in enumerate(set_summaries):
            if not isinstance(item, dict):
                raise ValueError(f"random_set_summaries[{idx}] must be a dictionary.")


def _validate_summary(
    name: str,
    summary: Dict[str, Any],
    required_keys: Iterable[str] | None = None,
) -> None:
    if not isinstance(summary, dict):
        raise ValueError(f"{name} summary must be a dictionary.")

    required_keys = list(required_keys or [])
    for key in required_keys:
        if key not in summary:
            raise ValueError(f"{name} summary missing required key: {key}")

    numeric_or_none_keys = [
        "baseline_accuracy",
        "ablated_accuracy",
        "accuracy_drop",
        "mean_probability_drop",
        "baseline_gt_probability",
        "ablated_gt_probability",
        "mean_gt_probability_drop",
        "baseline_margin",
        "ablated_margin",
        "mean_margin_drop",
        "mean_relative_perturbation",
    ]
    for key in numeric_or_none_keys:
        if key in summary and not _is_number_or_none(summary[key]):
            raise ValueError(f"{name}.{key} must be numeric or null.")


def _validate_results_rows(name: str, rows: Any) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be a list.")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{name}[{idx}] must be a dictionary.")

        required = ["baseline_pred", "ablated_pred", "baseline_prob", "ablated_prob"]
        for key in required:
            if key not in row:
                raise ValueError(f"{name}[{idx}] missing required key: {key}")
        if not _is_number_or_none(row.get("baseline_prob")):
            raise ValueError(f"{name}[{idx}].baseline_prob must be numeric or null.")
        if not _is_number_or_none(row.get("ablated_prob")):
            raise ValueError(f"{name}[{idx}].ablated_prob must be numeric or null.")


def _is_number_or_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


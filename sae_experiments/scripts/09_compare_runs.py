"""Aggregate and compare ablation outcomes across multiple run directories."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_get_metric(blob: Dict[str, Any], path: List[str], default=None):
    cur = blob
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _collect_rows(root: str, pattern: str) -> List[Dict[str, Any]]:
    results_paths = sorted(glob.glob(os.path.join(root, pattern)))
    rows: List[Dict[str, Any]] = []
    for results_path in results_paths:
        run_dir = str(Path(results_path).parents[1])
        run_name = Path(run_dir).name
        config_path = os.path.join(run_dir, "config.yaml")
        results = _load_json(results_path)

        binding_acc_drop = _safe_get_metric(results, ["binding", "accuracy_drop"])
        binding_margin_drop = _safe_get_metric(results, ["binding", "mean_margin_drop"])
        binding_prob_drop = _safe_get_metric(results, ["binding", "mean_probability_drop"])
        binding_gt_drop = _safe_get_metric(results, ["binding", "mean_gt_probability_drop"])
        random_acc_drop = _safe_get_metric(results, ["random", "accuracy_drop"])
        random_margin_drop = _safe_get_metric(results, ["random", "mean_margin_drop"])
        random_n_sets = _safe_get_metric(results, ["random", "n_sets"], 1)
        sig_acc_p = _safe_get_metric(results, ["significance", "accuracy_drop", "empirical_p_value"])
        sig_margin_p = _safe_get_metric(results, ["significance", "mean_margin_drop", "empirical_p_value"])

        row = {
            "run_name": run_name,
            "run_dir": run_dir,
            "config_path": config_path if os.path.exists(config_path) else None,
            "binding_accuracy_drop": binding_acc_drop,
            "random_accuracy_drop": random_acc_drop,
            "binding_minus_random_accuracy_drop": (
                None
                if binding_acc_drop is None or random_acc_drop is None
                else float(binding_acc_drop) - float(random_acc_drop)
            ),
            "binding_mean_margin_drop": binding_margin_drop,
            "random_mean_margin_drop": random_margin_drop,
            "binding_mean_probability_drop": binding_prob_drop,
            "binding_mean_gt_probability_drop": binding_gt_drop,
            "random_n_sets": random_n_sets,
            "empirical_p_accuracy_drop": sig_acc_p,
            "empirical_p_margin_drop": sig_margin_p,
        }
        rows.append(row)
    return rows


def _sort_rows(rows: List[Dict[str, Any]], key: str, descending: bool = True) -> List[Dict[str, Any]]:
    def sort_key(row: Dict[str, Any]):
        value = row.get(key)
        if value is None:
            return float("-inf") if descending else float("inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf") if descending else float("inf")

    return sorted(rows, key=sort_key, reverse=descending)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="output/sae_experiments/sweeps",
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument(
        "--results_glob",
        type=str,
        default="*/results/ablation_results.json",
        help="Glob pattern under --root used to find ablation JSON files.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="binding_minus_random_accuracy_drop",
        help="Metric key used to sort the output table.",
    )
    parser.add_argument("--ascending", action="store_true", help="Sort ascending instead of descending.")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    rows = _collect_rows(args.root, args.results_glob)
    rows = _sort_rows(rows, key=args.sort_by, descending=not args.ascending)

    output_json = args.output_json or os.path.join(args.root, "comparison_summary.json")
    output_csv = args.output_csv or os.path.join(args.root, "comparison_summary.csv")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    _write_csv(output_csv, rows)

    print(f"Compared {len(rows)} runs")
    print(f"Saved JSON summary to {output_json}")
    print(f"Saved CSV summary to {output_csv}")


if __name__ == "__main__":
    main()


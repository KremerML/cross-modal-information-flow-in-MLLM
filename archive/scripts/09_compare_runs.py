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
    """Load a JSON file into a dictionary.

    Args:
        path (str): JSON file path.

    Returns:
        Dict[str, Any]: Parsed JSON object.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file content is invalid JSON.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_get_metric(blob: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    """Traverse nested dictionaries safely and return a metric value.

    Args:
        blob (Dict[str, Any]): Input mapping.
        path (List[str]): Ordered key path to traverse.
        default (Any): Fallback value when the path does not exist.

    Returns:
        Any: Resolved value or ``default``.

    Raises:
        None: Missing keys are handled by returning ``default``.
    """
    cur = blob
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _collect_rows(root: str, pattern: str) -> List[Dict[str, Any]]:
    """Collect per-run comparison rows from ablation result files.

    Args:
        root (str): Root directory containing run subdirectories.
        pattern (str): Glob pattern under ``root`` that matches result JSON files.

    Returns:
        List[Dict[str, Any]]: One summary row per discovered run.

    Raises:
        OSError: If result files cannot be read.
        json.JSONDecodeError: If a result file is malformed.
    """
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
    """Sort run summary rows by a numeric metric key.

    Args:
        rows (List[Dict[str, Any]]): Comparison rows.
        key (str): Metric field used for ordering.
        descending (bool): Sort direction flag.

    Returns:
        List[Dict[str, Any]]: Sorted rows.

    Raises:
        None: Non-numeric values are handled with sentinel ordering.
    """
    def sort_key(row: Dict[str, Any]) -> float:
        """Build a numeric sort key with stable handling for missing/non-numeric values.

        Args:
            row (Dict[str, Any]): Comparison row.

        Returns:
            float: Numeric sort key used by ``sorted``.

        Raises:
            None: Conversion failures map to sentinel values.
        """
        value = row.get(key)
        if value is None:
            return float("-inf") if descending else float("inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf") if descending else float("inf")

    return sorted(rows, key=sort_key, reverse=descending)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write comparison rows to CSV.

    Args:
        path (str): Output CSV file path.
        rows (List[Dict[str, Any]]): Rows to write.

    Returns:
        None: Persists CSV output to disk.

    Raises:
        OSError: If output file creation or writing fails.
    """
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
    """Aggregate and export run comparison summaries across experiment directories.

    Args:
        None: CLI arguments are parsed in this function.

    Returns:
        None: Writes JSON and CSV summary files.

    Raises:
        OSError: If results cannot be discovered or written.
        json.JSONDecodeError: If a discovered results file is malformed.
    """
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

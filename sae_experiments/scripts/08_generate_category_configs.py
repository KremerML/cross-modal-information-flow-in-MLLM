"""Generate per-category SAE configs from a base config."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


DEFAULT_CATEGORIES = ["color", "size", "shape", "material", "state"]


def _count_rows(csv_path: str) -> int:
    """Count rows in a CSV file, excluding the header.

    Args:
        csv_path (str): CSV file path.

    Returns:
        int: Number of non-header rows.

    Raises:
        OSError: If ``csv_path`` cannot be read.
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _as_repo_relative(path: str) -> str:
    """Convert a path to repository-relative form when possible.

    Args:
        path (str): Input path (absolute or relative).

    Returns:
        str: Path relative to repository root when resolvable, otherwise absolute path.

    Raises:
        None: Falls back to absolute path if relative conversion is not possible.
    """
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, start=str(ROOT))
    except ValueError:
        return abs_path


def main() -> None:
    """Generate per-category config files from a base SAE config template.

    Args:
        None: CLI arguments are parsed in this function.

    Returns:
        None: Writes one config per category plus a generation manifest.

    Raises:
        FileNotFoundError: If the base config or datasets are missing.
        ValueError: If categories are empty or base config structure is invalid.
        yaml.YAMLError: If YAML parsing/serialization fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="datasets/by_attribute_category")
    parser.add_argument("--output_dir", type=str, default="configs/sae_categories")
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated categories to generate.",
    )
    parser.add_argument("--min_rows", type=int, default=100)
    args = parser.parse_args()

    categories = [part.strip().lower() for part in args.categories.split(",") if part.strip()]
    if not categories:
        raise ValueError("No categories provided.")

    base_config_path = os.path.expanduser(args.base_config)
    dataset_dir = os.path.expanduser(args.dataset_dir)
    output_root = os.path.expanduser(args.output_dir)
    profile_name = Path(base_config_path).stem
    output_dir = os.path.join(output_root, profile_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(base_config_path, "r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle) or {}
    if not isinstance(base_cfg, dict):
        raise ValueError(f"Base config is not a mapping: {base_config_path}")

    dataset_cfg = base_cfg.setdefault("dataset", {})
    dataset_cfg.setdefault("task_types", ["ChooseAttr"])
    experiment_cfg = base_cfg.setdefault("experiment", {})

    base_name = str(experiment_cfg.get("name") or Path(base_config_path).stem).strip()
    if not base_name:
        base_name = Path(base_config_path).stem

    generated: List[Dict[str, object]] = []
    for category in categories:
        category_csv = os.path.join(dataset_dir, f"ChooseAttr_{category}.csv")
        if not os.path.exists(category_csv):
            generated.append(
                {
                    "category": category,
                    "status": "missing_dataset",
                    "dataset_path": category_csv,
                }
            )
            continue

        row_count = _count_rows(category_csv)
        category_cfg = copy.deepcopy(base_cfg)
        category_cfg.setdefault("dataset", {})
        category_cfg["dataset"]["refined_dataset"] = _as_repo_relative(category_csv)
        category_cfg["dataset"]["task_types"] = ["ChooseAttr"]

        category_cfg.setdefault("experiment", {})
        category_experiment_name = f"{base_name}_{category}"
        category_cfg["experiment"]["name"] = category_experiment_name
        if category_cfg["experiment"].get("output_base"):
            category_cfg["experiment"]["output_dir"] = os.path.join(
                category_cfg["experiment"]["output_base"],
                category_experiment_name,
            )
        elif category_cfg["experiment"].get("output_dir"):
            current_output_dir = str(category_cfg["experiment"]["output_dir"])
            parent = os.path.dirname(current_output_dir.rstrip("/")) or current_output_dir
            category_cfg["experiment"]["output_dir"] = os.path.join(parent, category_experiment_name)

        output_cfg_path = os.path.join(output_dir, f"{category}.yaml")
        with open(output_cfg_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(category_cfg, handle, sort_keys=False)

        generated.append(
            {
                "category": category,
                "status": "ready" if row_count >= int(args.min_rows) else "too_small",
                "dataset_path": _as_repo_relative(category_csv),
                "config_path": _as_repo_relative(output_cfg_path),
                "rows": row_count,
            }
        )

    manifest = {
        "base_config": _as_repo_relative(base_config_path),
        "dataset_dir": _as_repo_relative(dataset_dir),
        "output_root": _as_repo_relative(output_root),
        "profile_name": profile_name,
        "output_dir": _as_repo_relative(output_dir),
        "categories": categories,
        "min_rows_threshold": int(args.min_rows),
        "generated": generated,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote category configs to {output_dir}")
    print(f"Manifest: {manifest_path}")
    for entry in generated:
        print(
            f"  {entry['category']:>8}: {entry['status']}"
            + (f", rows={entry['rows']}" if "rows" in entry else "")
        )


if __name__ == "__main__":
    main()

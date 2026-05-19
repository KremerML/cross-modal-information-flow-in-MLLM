"""Build per-category ChooseAttr datasets from a refined CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path
import re
import sys
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.utils import token_utils


DEFAULT_CATEGORIES = ["color", "size", "shape", "material", "state"]


def _split_words(raw: str) -> List[str]:
    """Split a delimited attribute field into normalized tokens.

    Args:
        raw (str): Raw attribute text, potentially delimited by commas/semicolons/pipes.

    Returns:
        List[str]: Lower-cased, trimmed tokens preserving original order.

    Raises:
        None: Empty input returns an empty list.
    """
    if not raw:
        return []
    return [part.strip().lower() for part in re.split(r"[,;|]", str(raw)) if part.strip()]


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    """Deduplicate a sequence while preserving first-seen order.

    Args:
        values (Sequence[str]): Input sequence of category tokens.

    Returns:
        List[str]: Deduplicated values in stable order.

    Raises:
        None.
    """
    return list(OrderedDict((v, None) for v in values).keys())


def _extract_field_categories(
    row: Dict[str, str],
    allowed: set[str],
) -> List[str]:
    """Extract allowed categories from the explicit attribute metadata field.

    Args:
        row (Dict[str, str]): Dataset row.
        allowed (set[str]): Allowed category labels.

    Returns:
        List[str]: Unique matched categories from ``central object question attribute``.

    Raises:
        KeyError: If expected row fields are missing.
    """
    words = _split_words(row.get("central object question attribute", ""))
    out: List[str] = []
    for word in words:
        cat = token_utils.categorize_attribute(word)
        if cat in allowed:
            out.append(cat)
    return _dedupe_preserve_order(out)


def _extract_question_categories(
    row: Dict[str, str],
    allowed: set[str],
) -> List[str]:
    """Extract allowed categories from question text via token-based matching.

    Args:
        row (Dict[str, str]): Dataset row.
        allowed (set[str]): Allowed category labels.

    Returns:
        List[str]: Unique matched categories inferred from question tokens.

    Raises:
        KeyError: If expected row fields are missing.
    """
    question = row.get("question", "")
    out: List[str] = []
    for _, category, _ in token_utils.extract_attribute_words(question):
        if category in allowed:
            out.append(category)
    return _dedupe_preserve_order(out)


def _assign_categories(
    row: Dict[str, str],
    allowed_categories: Sequence[str],
) -> Tuple[List[str], str]:
    """Assign categories for a row with source provenance.

    Args:
        row (Dict[str, str]): Dataset row.
        allowed_categories (Sequence[str]): Candidate categories in priority order.

    Returns:
        Tuple[List[str], str]: Matched categories and source label
        (``central_object_question_attribute``, ``question_text``, or ``none``).

    Raises:
        KeyError: If required row fields are missing.
    """
    allowed = set(allowed_categories)
    field_cats = _extract_field_categories(row, allowed)
    if field_cats:
        return field_cats, "central_object_question_attribute"
    question_cats = _extract_question_categories(row, allowed)
    if question_cats:
        return question_cats, "question_text"
    return [], "none"


def _count_rows(csv_path: str) -> int:
    """Count data rows in a CSV file, excluding the header.

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


def main() -> None:
    """Split a refined ChooseAttr CSV into per-category dataset shards.

    Args:
        None: CLI arguments are parsed inside this function.

    Returns:
        None: Writes per-category CSV files and a manifest JSON.

    Raises:
        FileNotFoundError: If the input CSV cannot be read.
        ValueError: If categories or CSV schema are invalid.
        OSError: If output files cannot be written.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/by_attribute_category",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated categories to split (default: color,size,shape,material,state).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["first", "drop_multi", "duplicate"],
        default="first",
        help=(
            "How to handle rows matching multiple categories: "
            "'first' picks the first category in --categories, "
            "'drop_multi' skips row, "
            "'duplicate' writes to all matched categories."
        ),
    )
    parser.add_argument(
        "--min_rows",
        type=int,
        default=100,
        help="Minimum row count threshold for reporting category as runnable.",
    )
    args = parser.parse_args()

    categories = [part.strip().lower() for part in args.categories.split(",") if part.strip()]
    if not categories:
        raise ValueError("No categories provided.")
    category_set = set(categories)

    input_csv = os.path.expanduser(args.input_csv)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"Input CSV has no header: {input_csv}")
        rows = list(reader)

    total_rows = len(rows)

    by_category: Dict[str, List[Dict[str, str]]] = {category: [] for category in categories}
    unlabeled_rows: List[Dict[str, str]] = []
    ambiguous_examples: List[Dict[str, object]] = []
    source_counts: Dict[str, int] = {"central_object_question_attribute": 0, "question_text": 0, "none": 0}
    ambiguous_count = 0

    for row in rows:
        matched, source = _assign_categories(row, categories)
        source_counts[source] = source_counts.get(source, 0) + 1

        if not matched:
            unlabeled_rows.append(row)
            continue

        if len(matched) > 1:
            ambiguous_count += 1
            if len(ambiguous_examples) < 30:
                ambiguous_examples.append(
                    {
                        "question_id": row.get("question_id"),
                        "categories": matched,
                        "source": source,
                        "question": row.get("question", ""),
                        "attribute_field": row.get("central object question attribute", ""),
                    }
                )

        if len(matched) == 1:
            by_category[matched[0]].append(row)
            continue

        if args.policy == "drop_multi":
            continue
        if args.policy == "duplicate":
            for category in matched:
                by_category[category].append(row)
        else:  # first
            # Deterministic priority based on --categories ordering.
            for category in categories:
                if category in matched:
                    by_category[category].append(row)
                    break

    category_paths: Dict[str, str] = {}
    category_counts: Dict[str, int] = {}
    for category in categories:
        out_path = os.path.join(output_dir, f"ChooseAttr_{category}.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in by_category[category]:
                writer.writerow(row)
        category_paths[category] = out_path
        category_counts[category] = _count_rows(out_path)

    manifest = {
        "input_csv": input_csv,
        "output_dir": output_dir,
        "total_rows": total_rows,
        "assigned_rows_total": int(sum(category_counts.values())),
        "unlabeled_rows": len(unlabeled_rows),
        "ambiguous_rows": ambiguous_count,
        "policy": args.policy,
        "categories": categories,
        "source_counts": source_counts,
        "category_counts": category_counts,
        "category_paths": category_paths,
        "runnable_categories": [
            category for category in categories if category_counts.get(category, 0) >= int(args.min_rows)
        ],
        "min_rows_threshold": int(args.min_rows),
        "ambiguous_examples": ambiguous_examples,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote category datasets to {output_dir}")
    print(f"Manifest: {manifest_path}")
    for category in categories:
        print(f"  {category:>8}: {category_counts[category]}")
    print(f"  unlabeled: {len(unlabeled_rows)}")
    print(f"  ambiguous: {ambiguous_count} (policy={args.policy})")


if __name__ == "__main__":
    main()

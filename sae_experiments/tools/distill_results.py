"""Distill large result JSONs into small committed summaries.

The pipeline writes several files that are far too large to keep in git or to load
into an LLM context: per-feature dictionaries over all 32768 SAE features, and
ablation results carrying one record per evaluated sample. This tool reduces each
to a `*.summary.json` sibling that preserves everything the writeup needs — the
top-scoring features, the distribution the scores came from, and the aggregate
statistics — while dropping the long tail.

The originals stay on disk (gitignored); the summaries are what gets committed.

    $PY sae_experiments/tools/distill_results.py --root output --dry_run
    $PY sae_experiments/tools/distill_results.py --root output

Stdlib only, so it runs without the LLaVA venv.
"""

import argparse
import json
import math
import os
import sys

# Per-sample lists longer than this are replaced by summary statistics.
SAMPLE_LIST_THRESHOLD = 20

# How many top features each distilled catalog retains.
DEFAULT_TOP_K = 500

# Files below this stay tracked as-is — small enough that the repo can carry them
# at full fidelity. Sources above it are distilled, and any summary that would not
# actually be smaller than its source is skipped.
DEFAULT_MIN_BYTES = 128 * 1024


def _percentiles(values, points=(50, 90, 99, 99.9)):
    """Nearest-rank percentiles of an already-sorted ascending list."""
    if not values:
        return {}
    out = {}
    n = len(values)
    for p in points:
        idx = min(n - 1, max(0, int(math.ceil(p / 100.0 * n)) - 1))
        out[f"p{p:g}"] = values[idx]
    return out


def _describe(values):
    """Mean/std/min/max/percentiles for a list of numbers."""
    vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    ordered = sorted(vals)
    desc = {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": ordered[0],
        "max": ordered[-1],
        "frac_positive": sum(1 for v in vals if v > 0) / n,
    }
    desc.update(_percentiles(ordered))
    return desc


def distill_feature_dict(data, score_key, top_k, extra_keys=()):
    """Reduce a {feature_index: {metric: value}} dict to top-k plus distribution.

    Also records the activation magnitude of the retained features against the
    global distribution — the statistic that distinguishes v2's causal features
    from the v1 "ghost features" that drove the null results.
    """
    items = []
    for fid, rec in data.items():
        if not isinstance(rec, dict):
            continue
        score = rec.get(score_key)
        if isinstance(score, (int, float)):
            items.append((abs(score), fid, rec))
    items.sort(key=lambda t: t[0], reverse=True)

    scores = sorted(abs(s) for s, _, _ in items)
    total_mass = sum(scores)
    top = items[:top_k]

    summary = {
        "n_features_total": len(data),
        "n_features_scored": len(items),
        "score_key": score_key,
        "top_k_retained": len(top),
        "score_distribution": _describe(scores),
        "score_mass_total": total_mass,
        "n_nonzero": sum(1 for s in scores if s > 0),
    }

    for k in (50, 200, 500):
        if k <= len(items):
            captured = sum(s for s, _, _ in items[:k])
            summary[f"score_mass_frac_top{k}"] = (
                captured / total_mass if total_mass else 0.0
            )

    for key in extra_keys:
        all_vals = [
            r.get(key) for _, _, r in items if isinstance(r.get(key), (int, float))
        ]
        top_vals = [
            r.get(key) for _, _, r in top if isinstance(r.get(key), (int, float))
        ]
        if all_vals:
            summary[f"{key}_all"] = _describe([abs(v) for v in all_vals])
        if top_vals:
            summary[f"{key}_top_k"] = _describe([abs(v) for v in top_vals])

    summary["top_features"] = [
        dict(feature=int(fid) if str(fid).isdigit() else fid, **rec) for _, fid, rec in top
    ]
    return summary


def distill_sample_lists(node):
    """Recursively replace long per-sample lists with per-field summary statistics."""
    if isinstance(node, dict):
        return {k: distill_sample_lists(v) for k, v in node.items()}

    if isinstance(node, list):
        if len(node) > SAMPLE_LIST_THRESHOLD and all(
            isinstance(x, dict) for x in node
        ):
            fields = {}
            for key in {k for rec in node for k in rec}:
                desc = _describe([rec.get(key) for rec in node])
                if desc is not None:
                    fields[key] = desc
            return {
                "_distilled": True,
                "_original_length": len(node),
                "field_summaries": fields,
            }
        if len(node) > SAMPLE_LIST_THRESHOLD and all(
            isinstance(x, (int, float)) for x in node
        ):
            return {
                "_distilled": True,
                "_original_length": len(node),
                "summary": _describe(node),
            }
        return [distill_sample_lists(x) for x in node]

    return node


def derive_ablation_fields(data):
    """Add the per-sample quantities the writeup needs but the runner never stored.

    `margin_drop` is the headline metric yet only its two operands are recorded per
    sample, and prediction flips are not counted anywhere. Both are recoverable, but
    only from the raw lists — so they must be computed before those lists are
    collapsed into summary statistics.
    """
    for key in ("binding_results", "random_results"):
        records = data.get(key)
        if not isinstance(records, list):
            continue
        flips = heals = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            base, abl = rec.get("baseline_margin"), rec.get("ablated_margin")
            if isinstance(base, (int, float)) and isinstance(abl, (int, float)):
                rec["margin_drop"] = base - abl
            answer = rec.get("answer")
            bp, ap = rec.get("baseline_pred"), rec.get("ablated_pred")
            if answer is not None and bp is not None and ap is not None:
                if bp == answer and ap != answer:
                    flips += 1
                elif bp != answer and ap == answer:
                    heals += 1
        data[f"{key}_prediction_changes"] = {
            "n": len(records),
            "correct_to_wrong": flips,
            "wrong_to_correct": heals,
        }
    return data


def distill_file(path, top_k):
    """Dispatch on filename. Returns (summary_dict, output_path) or None to skip."""
    base = os.path.basename(path)
    with open(path) as fh:
        data = json.load(fh)

    stem = path[: -len(".json")] if path.endswith(".json") else path
    out_path = stem + ".summary.json"

    if base == "causal_feature_stats.json":
        summary = distill_feature_dict(
            data,
            score_key="causal_score",
            top_k=top_k,
            extra_keys=("activation_mean", "gradient_mean"),
        )
    elif base == "feature_stats.json":
        summary = distill_feature_dict(
            data,
            score_key="ratio",
            top_k=top_k,
            extra_keys=("correct_mean", "incorrect_mean", "diff"),
        )
        by_diff = distill_feature_dict(data, score_key="diff", top_k=top_k)
        summary["top_features_by_diff"] = by_diff["top_features"]
        summary["diff_distribution"] = by_diff["score_distribution"]
    elif "ablation" in base and base.endswith(".json"):
        summary = distill_sample_lists(derive_ablation_fields(data))
    else:
        return None

    summary["_source"] = os.path.relpath(path)
    summary["_source_bytes"] = os.path.getsize(path)
    return summary, out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output", help="Directory tree to scan")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min_bytes", type=int, default=DEFAULT_MIN_BYTES,
                        help="Skip sources smaller than this (they stay tracked as-is)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be written without writing")
    args = parser.parse_args()

    targets = []
    for dirpath, _, filenames in os.walk(args.root):
        for name in filenames:
            if not name.endswith(".json") or name.endswith(".summary.json"):
                continue
            if name in ("causal_feature_stats.json", "feature_stats.json") or (
                "ablation" in name and "results" in name
            ):
                path = os.path.join(dirpath, name)
                if os.path.getsize(path) >= args.min_bytes:
                    targets.append(path)

    targets.sort()
    total_in = total_out = 0
    written = skipped = 0

    for path in targets:
        try:
            result = distill_file(path, args.top_k)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  SKIP {path}: {exc}", file=sys.stderr)
            continue
        if result is None:
            continue
        summary, out_path = result
        blob = json.dumps(summary, indent=1)
        in_sz = os.path.getsize(path)
        out_sz = len(blob.encode())

        if out_sz >= in_sz:
            print(f"  SKIP (no gain) {path}")
            skipped += 1
            continue

        total_in += in_sz
        total_out += out_sz
        pct = 100.0 * out_sz / in_sz
        print(f"  {in_sz/1048576:7.2f} MB -> {out_sz/1048576:6.3f} MB ({pct:5.1f}%)  {out_path}")
        if not args.dry_run:
            with open(out_path, "w") as fh:
                fh.write(blob)
            written += 1

    print(
        f"\n{len(targets) - skipped} distilled: {total_in/1048576:.1f} MB -> "
        f"{total_out/1048576:.1f} MB ({100.0*total_out/total_in if total_in else 0:.1f}%), "
        f"{written} written, {skipped} skipped"
    )


if __name__ == "__main__":
    main()

"""Turn multi-layer condition outputs into the numbers the writeup needs.

Three quantities, in descending order of how much weight they can carry:

**R(S) = A(S) / K(S)** -- the redundancy index. Joint ablation drop over a span divided by
the multi-layer knockout drop over the *same span, same samples, same filter policy*. This
generalises "% of knockout ceiling" and, unlike the published figures, has a numerator and a
denominator from the same sample set. The hypothesis predicts R(S) well above the 0.50-0.68
single layers reach. It assumes no additivity model at all, which is why it leads.

**Comparative saturation** -- fit A(S_k) and K(S_k) over span size k. If ablation and
knockout saturate at the same rate the shortfall is a constant fraction rather than
redundancy; if ablation saturates faster, features become redundant faster than the flow
does. The null here is fitted to the knockout data itself, so no arbitrary curve is assumed.

**Interaction index** -- A(S) against the sum of single-layer drops. Sub-additivity alone
proves nothing, because the metric saturates even under true independence; so the ratio is
reported *calibrated by the knockout's own sub-additivity*, which the same run measures.

    PY=LLaVA-NeXT/.venv/bin/python
    $PY sae_experiments/tools/analyze_multilayer_ablation.py \
        --experiment_dir output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sae_experiments.ablation.statistical_analysis import (  # noqa: E402
    paired_bootstrap_ci,
    ratio_statistic,
    wilcoxon_vs_controls,
    z_score_standard_error,
)

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


def load_conditions(experiment_dir):
    """Load every completed condition, keyed by id, with per-sample margin drops."""
    root = os.path.join(experiment_dir, "conditions")
    if not os.path.isdir(root):
        raise SystemExit(f"no conditions directory under {experiment_dir}")

    conditions = {}
    for condition_id in sorted(os.listdir(root)):
        path = os.path.join(root, condition_id, "results.json")
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            payload = json.load(handle)
        conditions[condition_id] = {
            "summary": payload.get("summary", {}),
            "significance": payload.get("significance", {}),
            "meta": payload.get("meta", {}),
            "control_summaries": payload.get("control_summaries", []),
            "margin_drops": margin_drops(payload.get("per_sample", [])),
        }
    return conditions


def margin_drops(rows):
    """Per-sample baseline_margin - ablated_margin, the quantity everything is built on."""
    drops = []
    for row in rows:
        baseline = row.get("baseline_margin")
        ablated = row.get("ablated_margin")
        if baseline is None or ablated is None:
            continue
        drops.append(float(baseline) - float(ablated))
    return drops


def redundancy_index(conditions, ablation_id, knockout_id, seed=42):
    """R = A/K with a paired bootstrap CI, or a reason it cannot be reported."""
    if ablation_id not in conditions or knockout_id not in conditions:
        return {"status": "missing_condition"}

    ablation = conditions[ablation_id]["margin_drops"]
    knockout = conditions[knockout_id]["margin_drops"]
    if not ablation or not knockout:
        return {"status": "no_per_sample_data"}
    if len(ablation) != len(knockout):
        return {"status": "sample_count_mismatch"}

    mean_knockout = float(np.mean(knockout))
    result = {
        "ablation_condition": ablation_id,
        "knockout_condition": knockout_id,
        "ablation_mean": float(np.mean(ablation)),
        "knockout_mean": mean_knockout,
        "n_samples": len(ablation),
    }

    # A negative ceiling makes the ratio meaningless, not merely noisy. Layer 13's knockout
    # is inhibitory, so this is a real case and must be reported as undefined.
    if mean_knockout <= 0:
        result["status"] = "undefined_negative_ceiling"
        result["ratio"] = None
        return result

    point, low, high = paired_bootstrap_ci(
        {"ablation": ablation, "knockout": knockout},
        ratio_statistic("ablation", "knockout"),
        seed=seed,
    )
    result.update(
        {"status": "ok", "ratio": point, "ci_low": low, "ci_high": high}
    )
    return result


def saturation_fit(span_sizes, values):
    """Fit y = a(1 - exp(-b k)); returns (a, b) or None when scipy is unavailable."""
    if curve_fit is None or len(span_sizes) < 3:
        return None

    def model(k, a, b):
        return a * (1.0 - np.exp(-b * k))

    try:
        params, _ = curve_fit(
            model,
            np.asarray(span_sizes, dtype=float),
            np.asarray(values, dtype=float),
            p0=[max(values) if values else 1.0, 0.5],
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None
    return {"a": float(params[0]), "b": float(params[1])}


def nested_curves(conditions):
    """Ablation and knockout saturation as a function of span size."""
    ablation_points, knockout_points = [], []
    for condition_id, payload in conditions.items():
        summary = payload["summary"]
        size = len(summary.get("layers") or summary.get("knockout_layers") or [])
        if not size:
            continue
        drop = summary.get("mean_margin_drop")
        if drop is None:
            continue
        if condition_id.startswith(("nested_L", "joint_L")):
            ablation_points.append((size, float(drop)))
        elif condition_id.startswith(("nested_knockout_L", "span_knockout_L")):
            knockout_points.append((size, float(drop)))

    ablation_points.sort()
    knockout_points.sort()
    out = {
        "ablation": [{"span_size": s, "margin_drop": d} for s, d in ablation_points],
        "knockout": [{"span_size": s, "margin_drop": d} for s, d in knockout_points],
        "ablation_fit": saturation_fit(*zip(*ablation_points)) if len(ablation_points) >= 3 else None,
        "knockout_fit": saturation_fit(*zip(*knockout_points)) if len(knockout_points) >= 3 else None,
    }
    if out["ablation_fit"] and out["knockout_fit"]:
        out["b_minus_d"] = out["ablation_fit"]["b"] - out["knockout_fit"]["b"]
        out["interpretation"] = (
            "ablation saturates faster than the flow does (redundancy)"
            if out["b_minus_d"] > 0
            else "ablation and knockout saturate alike; the shortfall looks like a constant fraction"
        )
    return out


def interaction_index(conditions, span, joint_id, span_knockout_id):
    """A(S) vs the sum of single-layer effects, calibrated by the knockout's own sub-additivity."""
    single_ablation, single_knockout = [], []
    for layer in span:
        nested = conditions.get(f"nested_L{layer}")
        if nested:
            single_ablation.append(nested["summary"].get("mean_margin_drop"))
        knockout = conditions.get(f"knockout_L{layer}")
        if knockout:
            single_knockout.append(knockout["summary"].get("mean_margin_drop"))

    result = {"span": list(span)}
    joint = conditions.get(joint_id, {}).get("summary", {}).get("mean_margin_drop")
    span_knockout = conditions.get(span_knockout_id, {}).get("summary", {}).get("mean_margin_drop")

    if joint is not None and all(v is not None for v in single_ablation) and single_ablation:
        total = float(sum(single_ablation))
        result["joint_ablation"] = float(joint)
        result["sum_single_ablation"] = total
        result["rho_ablation"] = float(joint) / total if total else None

    if span_knockout is not None and all(v is not None for v in single_knockout) and single_knockout:
        total = float(sum(single_knockout))
        result["span_knockout"] = float(span_knockout)
        result["sum_single_knockout"] = total
        result["rho_knockout"] = float(span_knockout) / total if total else None

    rho_a, rho_k = result.get("rho_ablation"), result.get("rho_knockout")
    if rho_a is not None and rho_k:
        # The calibrated quantity: how sub-additive the ablation is *relative to* how
        # sub-additive the intervention that removes everything is. Above 1 means the
        # features combine less redundantly than the flow itself does.
        result["rho_ratio"] = rho_a / rho_k
    return result


def leave_one_out(conditions, span, joint_id):
    """In-context marginal contribution of each layer, against its standalone effect."""
    joint = conditions.get(joint_id, {}).get("summary", {}).get("mean_margin_drop")
    if joint is None:
        return []

    rows = []
    for layer in span:
        without = conditions.get(f"loo_drop{layer}", {}).get("summary", {}).get("mean_margin_drop")
        standalone = conditions.get(f"nested_L{layer}", {}).get("summary", {}).get("mean_margin_drop")
        if without is None:
            continue
        marginal = float(joint) - float(without)
        rows.append(
            {
                "layer": layer,
                "joint": float(joint),
                "without_layer": float(without),
                "marginal": marginal,
                "standalone": float(standalone) if standalone is not None else None,
                # Much smaller in context than alone is the redundancy signature: the other
                # layers already carry what this one contributes.
                "marginal_over_standalone": (
                    marginal / float(standalone)
                    if standalone not in (None, 0)
                    else None
                ),
            }
        )
    return rows


def significance_report(conditions):
    """Restate significance with the p-floor and the z precision made explicit."""
    rows = []
    for condition_id, payload in conditions.items():
        significance = payload.get("significance", {}).get("mean_margin_drop")
        if not significance:
            continue
        n_sets = len(payload.get("control_summaries", []))
        z = significance.get("z_score")
        controls = [
            c.get("mean_margin_drop") for c in payload.get("control_summaries", [])
        ]
        binding = payload["margin_drops"]

        entry = {
            "condition_id": condition_id,
            "binding": significance.get("binding"),
            "random_mean": significance.get("random_mean"),
            "random_std": significance.get("random_std"),
            "z_score": z,
            "z_standard_error": z_score_standard_error(z, n_sets) if z else None,
            "n_control_sets": n_sets,
            "empirical_p": significance.get("empirical_p_value"),
            "empirical_p_floor": 1.0 / (n_sets + 1) if n_sets else None,
            "control_regime": payload.get("meta", {}).get("control_effective"),
        }
        if binding and controls and all(c is not None for c in controls):
            control_mean = float(np.mean(controls))
            _, p_value = wilcoxon_vs_controls(binding, [control_mean] * len(binding))
            entry["wilcoxon_p"] = p_value
        rows.append(entry)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--span", default="10,11,12,13,14")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    span = [int(x) for x in args.span.split(",")]
    span_tag = f"{span[0]}-{span[-1]}" if span == list(range(span[0], span[-1] + 1)) else ",".join(map(str, span))
    joint_id = f"joint_L{span_tag}"
    span_knockout_id = f"span_knockout_L{span_tag}"

    conditions = load_conditions(args.experiment_dir)
    print(f"Loaded {len(conditions)} conditions from {args.experiment_dir}\n")

    report = {
        "experiment_dir": args.experiment_dir,
        "span": span,
        "n_conditions": len(conditions),
        "redundancy_index": redundancy_index(conditions, joint_id, span_knockout_id, args.seed),
        "single_layer_ratios": {
            str(layer): redundancy_index(
                conditions, f"nested_L{layer}", f"knockout_L{layer}", args.seed
            )
            for layer in span
        },
        "saturation": nested_curves(conditions),
        "interaction": interaction_index(conditions, span, joint_id, span_knockout_id),
        "leave_one_out": leave_one_out(conditions, span, joint_id),
        "significance": significance_report(conditions),
    }

    out_dir = os.path.join(args.experiment_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "multilayer_summary.json")
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2)

    markdown = render_markdown(report)
    md_path = os.path.join(out_dir, "multilayer_summary.md")
    with open(md_path, "w") as handle:
        handle.write(markdown)

    print(markdown)
    print(f"\nWrote {json_path}\n      {md_path}")


def render_markdown(report):
    lines = ["# Multi-layer ablation summary", ""]
    span = report["span"]
    lines.append(f"Span: {span}. {report['n_conditions']} conditions.")
    lines.append("")

    lines.append("## Redundancy index R(S) = A(S) / K(S)")
    lines.append("")
    index = report["redundancy_index"]
    if index.get("status") == "ok":
        lines.append(
            f"**R(S) = {index['ratio']:.3f}** "
            f"(95% CI {index['ci_low']:.3f} - {index['ci_high']:.3f}), "
            f"from A = {index['ablation_mean']:.4f} and K = {index['knockout_mean']:.4f} "
            f"over n = {index['n_samples']}."
        )
    else:
        lines.append(f"Not available: `{index.get('status')}`.")
    lines.append("")

    lines.append("Single layers, for comparison (same samples for numerator and denominator):")
    lines.append("")
    lines.append("| layer | ablation | knockout | ratio | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for layer, entry in report["single_layer_ratios"].items():
        if entry.get("status") == "ok":
            lines.append(
                f"| {layer} | {entry['ablation_mean']:.4f} | {entry['knockout_mean']:.4f} | "
                f"{entry['ratio']:.3f} | {entry['ci_low']:.3f} - {entry['ci_high']:.3f} |"
            )
        else:
            lines.append(f"| {layer} | — | — | undefined | {entry.get('status')} |")
    lines.append("")

    saturation = report["saturation"]
    if saturation.get("b_minus_d") is not None:
        lines.append("## Comparative saturation")
        lines.append("")
        lines.append(
            f"Ablation b = {saturation['ablation_fit']['b']:.4f}, "
            f"knockout d = {saturation['knockout_fit']['b']:.4f}, "
            f"b - d = {saturation['b_minus_d']:+.4f}."
        )
        lines.append("")
        lines.append(saturation["interpretation"])
        lines.append("")

    interaction = report["interaction"]
    if interaction.get("rho_ablation") is not None:
        lines.append("## Interaction index")
        lines.append("")
        lines.append(
            f"Joint ablation {interaction['joint_ablation']:.4f} against a sum of single "
            f"layers of {interaction['sum_single_ablation']:.4f}: rho_A = "
            f"{interaction['rho_ablation']:.3f}."
        )
        if interaction.get("rho_knockout"):
            lines.append(
                f"The knockout's own sub-additivity over the same span is rho_K = "
                f"{interaction['rho_knockout']:.3f}, so the calibrated ratio is "
                f"**{interaction['rho_ratio']:.3f}**. Raw sub-additivity is expected from "
                f"metric saturation even under independence; the calibrated ratio is the "
                f"part that carries information."
            )
        lines.append("")

    if report["leave_one_out"]:
        lines.append("## Leave-one-out")
        lines.append("")
        lines.append("| layer | joint | without | marginal | standalone | marginal/standalone |")
        lines.append("|---|---|---|---|---|---|")
        for row in report["leave_one_out"]:
            ratio = row["marginal_over_standalone"]
            lines.append(
                f"| {row['layer']} | {row['joint']:.4f} | {row['without_layer']:.4f} | "
                f"{row['marginal']:+.4f} | "
                f"{row['standalone']:.4f} | {ratio:.3f} |"
                if row["standalone"] is not None and ratio is not None
                else f"| {row['layer']} | {row['joint']:.4f} | {row['without_layer']:.4f} | "
                     f"{row['marginal']:+.4f} | — | — |"
            )
        lines.append("")
        lines.append(
            "A marginal contribution far below the standalone effect at every layer is the "
            "strong redundancy signature: the rest of the span already carries what that "
            "layer contributes."
        )
        lines.append("")

    if report["significance"]:
        lines.append("## Significance")
        lines.append("")
        lines.append("| condition | binding | z | z SE | control sets | p floor | Wilcoxon p | control regime |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in report["significance"]:
            z = row["z_score"]
            lines.append(
                f"| {row['condition_id']} | {row['binding']:.4f} | "
                f"{z:.0f} | {row['z_standard_error']:.0f} | {row['n_control_sets']} | "
                f"{row['empirical_p_floor']:.4f} | "
                f"{row.get('wilcoxon_p', float('nan')):.2e} | {row['control_regime']} |"
            )
        lines.append("")
        lines.append(
            "z is reported to one significant figure: it is estimated from a handful of "
            "control draws and its own standard error is large. The empirical p cannot go "
            "below its floor, so the Wilcoxon test over questions is the one with power."
        )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()

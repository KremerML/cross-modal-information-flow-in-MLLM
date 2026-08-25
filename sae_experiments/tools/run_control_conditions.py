"""Run the redesigned random-control conditions.

Why this exists
---------------
The published significance statistic is a z over N random control sets. That
estimand -- "how unusual is the binding set among comparable feature sets" --
is not identified for this experiment. The selection rule is a top-k cut on
`causal_score`, so the number of non-selected features whose score falls inside
the binding range is exactly zero. Matching on activation instead fails because
causal_score is close to a monotone function of activation (Spearman 0.949 at
layer 11): only 228 non-selected features at layer 11 and 185 at layer 14 lie
inside the binding activation range, against 200 needed. Fifteen "independent"
matched sets therefore overlap 97% and carry no between-set variance.

The fix is to stop sampling controls from a pool that does not exist, and to
use controls that are disjoint *by construction*:

  rank-band control  ablate causal ranks k..2k at the same layers. Disjoint
                     from the binding set by definition, same feature count,
                     comparable activation. Perturbation matching is verified
                     per condition rather than assumed (measured at layer 11:
                     0.01844 against the binding set's 0.01855).

  activation control the top-k features by activation alone, ignoring the
                     causal score. Carries MORE activation mass and MORE
                     perturbation than the binding set, so it is biased in
                     favour of the null: if the binding set still wins, the
                     conclusion does not depend on matching.

Inference moves from "over feature sets" to "over questions" -- a real
superpopulation -- via paired per-question contrasts on the shared 256.

Pairing
-------
Every condition is evaluated on the sample cache of the published multi-layer
run, so new conditions are paired question-for-question with the published
ones and per-sample contrasts are meaningful.

Outputs go to a NEW experiment directory. No published result file is touched.

    PY=LLaVA-NeXT/.venv/bin/python
    CFG=configs/clevr_lite/multilayer_l10-14_attn_out_question.yaml

    $PY sae_experiments/tools/run_control_conditions.py --config $CFG --dry_run
    $PY sae_experiments/tools/run_control_conditions.py --config $CFG --phases single
    $PY sae_experiments/tools/run_control_conditions.py --config $CFG --phases multi

Resumable: one fsync'd JSONL line per completed condition; completed ids are
skipped on restart.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sae_experiments.ablation.multilayer_experiments import (  # noqa: E402
    KIND_PASSTHROUGH,
    KIND_SAE,
    Condition,
    MultiLayerAblationExperiment,
)
from sae_experiments.ablation.sample_cache import load_sample_cache  # noqa: E402
from sae_experiments.core.config import load_config  # noqa: E402
from sae_experiments.tools.run_multilayer_ablation import (  # noqa: E402
    build_dataset,
    load_catalogs_and_stats,
    load_multilayer_saes,
    write_condition,
)
from sae_experiments.utils.script_utils import (  # noqa: E402
    load_llava_components,
    setup_experiment,
)

PHASES = ("single", "gradient", "doseresponse", "subsets", "multi")

# The published run whose sample cache and conditions we pair against.
PUBLISHED = "output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question"

LAYERS = (10, 11, 12, 13, 14)


def log(message="", indent=0):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {'  ' * indent}{message}", flush=True)


# --------------------------------------------------------------------------- rankings


def build_rankings(stats):
    """Per-layer feature orderings, computed once.

    `by_causal` is the selection order (the catalogs are its top 200).
    `by_activation` is the same dictionary ordered by mean activation alone --
    the control that ignores the gradient term.
    """
    rankings = {}
    for layer, layer_stats in stats.items():
        if not layer_stats:
            continue
        by_causal = [
            f for f, _ in sorted(
                layer_stats.items(),
                key=lambda kv: (-float(kv[1].get("causal_score", 0.0)), kv[0]),
            )
        ]
        by_activation = [
            f for f, _ in sorted(
                layer_stats.items(),
                key=lambda kv: (-float(kv[1].get("activation_mean", 0.0)), kv[0]),
            )
        ]
        rankings[layer] = {"causal": by_causal, "activation": by_activation}
    return rankings


def band(rankings, layer, lo, hi, order="causal"):
    """Features at causal ranks [lo, hi) for one layer."""
    return list(rankings[layer][order][lo:hi])


def activation_matched(stats, rankings, layer, binding, seed=42):
    """Nearest-neighbour activation matches for `binding`, excluding it.

    Kept for comparison with the published matched arm. The pool is known to be
    exhausted at k=200, which is the point: this condition documents what the
    matched sampler actually produces rather than assuming it works.
    """
    import bisect

    layer_stats = stats[layer]
    binding_set = set(binding)
    pool = sorted(
        (f for f in layer_stats if f not in binding_set),
        key=lambda f: float(layer_stats[f].get("activation_mean", 0.0)),
    )
    pool_vals = [float(layer_stats[f].get("activation_mean", 0.0)) for f in pool]
    chosen, used = [], set()
    for b in binding:
        target = float(layer_stats[b].get("activation_mean", 0.0))
        i = bisect.bisect_left(pool_vals, target)
        picked = None
        for off in range(len(pool)):
            for j in (i + off, i - off):
                if 0 <= j < len(pool) and pool[j] not in used:
                    picked = pool[j]
                    break
            if picked is not None:
                break
        if picked is None:
            break
        chosen.append(picked)
        used.add(picked)
    return chosen


# --------------------------------------------------------------------------- conditions


def single_layer_conditions(rankings, stats, k=200):
    """Per layer: the binding set and the controls it should be judged against."""
    conditions = []
    for layer in LAYERS:
        if layer not in rankings:
            continue
        binding = band(rankings, layer, 0, k)
        conditions += [
            Condition(f"ctl_passthrough_L{layer}", KIND_PASSTHROUGH,
                      features={layer: []},
                      label=f"L{layer} pass-through: SAE reconstruction floor"),
            Condition(f"bind_causal_1_{k}_L{layer}", KIND_SAE,
                      features={layer: binding},
                      label=f"L{layer} binding: causal ranks 1-{k}"),
            Condition(f"ctl_band_{k}_{2*k}_L{layer}", KIND_SAE,
                      features={layer: band(rankings, layer, k, 2 * k)},
                      label=f"L{layer} control: causal ranks {k+1}-{2*k}"),
            Condition(f"ctl_band_{2*k}_{3*k}_L{layer}", KIND_SAE,
                      features={layer: band(rankings, layer, 2 * k, 3 * k)},
                      label=f"L{layer} control: causal ranks {2*k+1}-{3*k}"),
            Condition(f"ctl_acttop_{k}_L{layer}", KIND_SAE,
                      features={layer: band(rankings, layer, 0, k, order="activation")},
                      label=f"L{layer} control: top-{k} by activation alone"),
            Condition(f"ctl_actmatched_{k}_L{layer}", KIND_SAE,
                      features={layer: activation_matched(stats, rankings, layer, binding)},
                      label=f"L{layer} control: activation-matched to binding"),
        ]
    return conditions


def gradient_isolation_conditions(rankings, layers=(11, 14), k=200):
    """The features on which the causal and activation rankings disagree.

    causal_score = |grad| x |activation|, so the two rankings overlap heavily
    (141/200 at layer 11). The symmetric difference isolates what the gradient
    term contributes: if the causal-only features move the margin and the
    activation-only ones do not, the gradient factor is doing real work.
    """
    conditions = []
    for layer in layers:
        if layer not in rankings:
            continue
        causal = set(band(rankings, layer, 0, k))
        act = set(band(rankings, layer, 0, k, order="activation"))
        causal_only = sorted(causal - act)
        act_only = sorted(act - causal)
        conditions += [
            Condition(f"grad_causal_only_L{layer}", KIND_SAE,
                      features={layer: causal_only},
                      label=f"L{layer}: {len(causal_only)} features causal-only"),
            Condition(f"grad_act_only_L{layer}", KIND_SAE,
                      features={layer: act_only},
                      label=f"L{layer}: {len(act_only)} features activation-only"),
            Condition(f"grad_shared_L{layer}", KIND_SAE,
                      features={layer: sorted(causal & act)},
                      label=f"L{layer}: {len(causal & act)} features in both rankings"),
        ]
    return conditions


def dose_response_conditions(rankings, layers=(11, 14), width=40, n_bands=10):
    """Disjoint equal-size bands down the causal ranking.

    Fixed size removes the budget confound: every band ablates `width`
    features, so a decline across bands cannot be a count effect. Bands are
    disjoint, so no candidate pool is needed.
    """
    conditions = []
    for layer in layers:
        if layer not in rankings:
            continue
        for i in range(n_bands):
            lo, hi = i * width, (i + 1) * width
            conditions.append(
                Condition(f"dose_L{layer}_r{lo}_{hi}", KIND_SAE,
                          features={layer: band(rankings, layer, lo, hi)},
                          label=f"L{layer} causal ranks {lo+1}-{hi}")
            )
    return conditions


def subset_conditions(rankings, layer=11, width=40, n_sets=12, seed=42):
    """Random equal-size subsets of two deep pools.

    This is the one place a set-level null IS estimable: subsets of the top-200
    and of ranks 201-1000 overlap by 20% and 5% respectively, not 97%, so
    between-set variance is real. It also measures whether the effect is spread
    across the selected set or concentrated in a few of its members.
    """
    conditions = []
    rng = random.Random(seed)
    top = band(rankings, layer, 0, 200)
    tail = band(rankings, layer, 200, 1000)
    for i in range(n_sets):
        conditions.append(
            Condition(f"subset_top200_L{layer}_{i:02d}", KIND_SAE,
                      features={layer: sorted(rng.sample(top, width))},
                      label=f"L{layer}: random {width} of causal ranks 1-200")
        )
    for i in range(n_sets):
        conditions.append(
            Condition(f"subset_tail_L{layer}_{i:02d}", KIND_SAE,
                      features={layer: sorted(rng.sample(tail, width))},
                      label=f"L{layer}: random {width} of causal ranks 201-1000")
        )
    return conditions


def multi_layer_conditions(rankings):
    """Rank-band controls for the multi-layer conditions that lack them.

    38 of the 47 published conditions have no control arm, including the whole
    concentrated budget curve. The headline comparison is therefore asymmetric:
    the spread arm is control-adjusted and the concentrated arm is not. Each
    control here mirrors its binding condition exactly -- same layers, same
    per-layer count -- shifted one band down the causal ranking.
    """
    conditions = []

    # Budget curve: concentrated at layer 11, k features -> control ranks k..2k
    for k in (40, 100, 200, 400, 800):
        conditions.append(
            Condition(f"ctl_budget_concentrated_L11_k{k}", KIND_SAE,
                      features={11: band(rankings, 11, k, 2 * k)},
                      label=f"control for budget_concentrated_L11_k{k}")
        )
    # Spread arm: 40 per layer across 10-14 -> control ranks 40..80 per layer
    conditions.append(
        Condition("ctl_budget_spread40x5", KIND_SAE,
                  features={l: band(rankings, l, 40, 80) for l in LAYERS},
                  label="control for budget_spread40x5")
    )
    # Redundancy spans: 200 per layer -> control ranks 200..400 per layer
    spans = {
        "nested_L14": (14,),
        "nested_L13-14": (13, 14),
        "nested_L12-14": (12, 13, 14),
        "nested_L11-14": (11, 12, 13, 14),
        "joint_L10-14": (10, 11, 12, 13, 14),
        "nonnested_L10-12": (10, 11, 12),
        "nonnested_L10,12,14": (10, 12, 14),
    }
    for name, layers in spans.items():
        conditions.append(
            Condition(f"ctl_{name}", KIND_SAE,
                      features={l: band(rankings, l, 200, 400) for l in layers},
                      label=f"control for {name}")
        )
    # Leave-one-out: the span minus one layer
    for dropped in LAYERS:
        layers = [l for l in LAYERS if l != dropped]
        conditions.append(
            Condition(f"ctl_loo_drop{dropped}", KIND_SAE,
                      features={l: band(rankings, l, 200, 400) for l in layers},
                      label=f"control for loo_drop{dropped}")
        )
    # Downstream anchors: single layer ablation, with the knockout arm untouched
    for anchor in (11, 14):
        conditions.append(
            Condition(f"ctl_downstream_ablate_L{anchor}", KIND_SAE,
                      features={anchor: band(rankings, anchor, 200, 400)},
                      label=f"control for downstream_ablate_L{anchor}")
        )
    return conditions


def build_conditions(rankings, stats, phases):
    conditions = []
    if "single" in phases:
        conditions += single_layer_conditions(rankings, stats)
    if "gradient" in phases:
        conditions += gradient_isolation_conditions(rankings)
    if "doseresponse" in phases:
        conditions += dose_response_conditions(rankings)
    if "subsets" in phases:
        conditions += subset_conditions(rankings)
    if "multi" in phases:
        conditions += multi_layer_conditions(rankings)
    return conditions


# --------------------------------------------------------------------------- checkpoint


def completed_conditions(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["condition_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_checkpoint(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phases", default="single",
                        help=f"comma-separated from {list(PHASES)}, or 'all'")
    parser.add_argument("--conditions", default=None, help="comma-separated condition ids")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--experiment_name", default="controls_v3_clevr_lite_l10-14")
    parser.add_argument("--sample_cache", default=None,
                        help="sample cache to pair against (default: the published run's)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    phases = list(PHASES) if args.phases == "all" else [
        p.strip() for p in args.phases.split(",") if p.strip()
    ]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        parser.error(f"unknown phase(s) {unknown}; choose from {list(PHASES)}")

    catalogs, stats = load_catalogs_and_stats(config)
    rankings = build_rankings(stats)
    missing = [l for l in LAYERS if l not in rankings]
    if missing:
        log(f"WARNING: no stats for layers {missing}; their conditions are skipped")

    conditions = build_conditions(rankings, stats, phases)
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in conditions if c.condition_id in wanted]

    if args.dry_run:
        log(f"{len(conditions)} conditions in phases {phases}")
        for condition in conditions:
            counts = ", ".join(
                f"L{l}:{len(condition.features[l])}" for l in sorted(condition.features)
            )
            log(f"{condition.condition_id:<34} [{counts}]  {condition.label}", indent=1)
        est = len(conditions) * 85
        log(f"estimated runtime {format_duration(est)} at 85s per condition")
        return

    experiment_dir, seed = setup_experiment(args, config)
    log(f"experiment dir: {experiment_dir}")
    os.makedirs(experiment_dir, exist_ok=True)

    cache_path = args.sample_cache or os.path.join(PUBLISHED, "sample_cache.json")
    if not os.path.exists(cache_path):
        sys.exit(f"sample cache not found: {cache_path}")
    sample_records = load_sample_cache(cache_path)
    log(f"paired against {len(sample_records)} samples from {cache_path}")

    checkpoint_path = os.path.join(experiment_dir, "checkpoint.jsonl")
    done = set() if args.force else completed_conditions(checkpoint_path)
    pending = [c for c in conditions if c.condition_id not in done]
    log(f"{len(conditions)} conditions, {len(done & {c.condition_id for c in conditions})} "
        f"already done, {len(pending)} to run")
    if not pending:
        log("nothing to do")
        return

    tokenizer, model, image_processor = load_llava_components(config.get("model", {}))
    model.eval()
    dataset = build_dataset(config, tokenizer, image_processor, model)
    saes = load_multilayer_saes(config, model)
    experiment = MultiLayerAblationExperiment(model, saes, catalogs, stats, config)

    with open(os.path.join(experiment_dir, "conditions_index.json"), "w") as handle:
        json.dump(
            [{"condition_id": c.condition_id, "label": c.label, "kind": c.kind,
              "features_per_layer": {str(l): len(c.features[l]) for l in sorted(c.features)},
              "total_features": c.total_features} for c in conditions],
            handle, indent=1,
        )

    started = time.time()
    for idx, condition in enumerate(pending, 1):
        elapsed = time.time() - started
        eta = (elapsed / (idx - 1) * (len(pending) - idx + 1)) if idx > 1 else None
        log(f"[{idx}/{len(pending)}] {condition.condition_id}"
            + (f"  (elapsed {format_duration(elapsed)}, ETA {format_duration(eta)})" if eta else ""))
        log(condition.label, indent=1)

        rows, summary = experiment.run_condition(
            dataset, condition, sample_records,
            max_samples=args.max_samples,
            show_progress=not args.no_progress,
        )
        payload = {
            "condition_id": condition.condition_id,
            "label": condition.label,
            "summary": summary,
            "control_summaries": [],
            "control_feature_sets": [],
            "significance": {},
            "meta": {
                "feature_selection": "causal_v2",
                "intervention": "control_design_v3",
                "control_design": "disjoint_rank_band",
                "layers": condition.layers,
                "features_per_layer": {
                    str(l): len(condition.features[l]) for l in sorted(condition.features)
                },
                "knockout_layers": list(condition.knockout_layers),
                "mode": condition.mode,
                "encode_mode": config.get("multilayer", {}).get("encode_mode", "live"),
                "n_samples": len(rows),
                "seed": seed,
                "paired_sample_cache": cache_path,
                "ablation_version": "v3_rank_band_controls",
            },
            "feature_ids": {str(l): condition.features[l] for l in sorted(condition.features)},
        }
        write_condition(experiment_dir, condition.condition_id, rows, payload)
        append_checkpoint(checkpoint_path, {
            "condition_id": condition.condition_id,
            "mean_margin_drop": summary.get("mean_margin_drop"),
            "mean_relative_perturbation": summary.get("mean_relative_perturbation"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        })
        log(f"drop {summary.get('mean_margin_drop'):+.4f}  "
            f"perturbation {summary.get('mean_relative_perturbation'):.5f}  "
            f"({summary.get('runtime_seconds'):.0f}s)", indent=1)

    log(f"done: {len(pending)} conditions in {format_duration(time.time() - started)}")


if __name__ == "__main__":
    main()

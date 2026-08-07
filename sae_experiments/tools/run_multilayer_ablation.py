"""Run the multi-layer ablation condition matrix.

Tests whether the 30-50% shortfall between single-layer SAE ablation and the attention-
knockout ceiling is explained by cross-layer redundancy -- whether ablating one layer leaves
the model free to re-read the image at layers L+1..31.

Lives in tools/ rather than pipeline/NN_ because it reads many experiment directories at
once (like tools/knockout_sae_pipeline.py) and because being importable makes the condition
builders testable.

    PY=LLaVA-NeXT/.venv/bin/python
    CFG=configs/clevr_lite/multilayer_l10-14_attn_out_question.yaml

    # Review the matrix before spending any GPU time
    $PY sae_experiments/tools/run_multilayer_ablation.py --config $CFG --dry_run

    # The gate first. A0 must reproduce the published single-layer layer-11 number
    # (0.2131) and gate_none must be exactly 0.0, or the harness is not trustworthy.
    $PY sae_experiments/tools/run_multilayer_ablation.py --config $CFG --phases gate

    # Then the rest
    $PY sae_experiments/tools/run_multilayer_ablation.py \
        --config $CFG --phases primary,nested,leave_one_out

Resumable: one fsync'd JSONL line per completed condition, and completed ids are skipped on
restart. Conditions take ~90s, so per-condition granularity is enough.
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sae_experiments.ablation.multilayer_experiments import (  # noqa: E402
    KIND_COMBINED,
    KIND_KNOCKOUT,
    KIND_NONE,
    KIND_PASSTHROUGH,
    KIND_SAE,
    Condition,
    MultiLayerAblationExperiment,
    _compact_layer_range,
)
from sae_experiments.ablation.sample_cache import (  # noqa: E402
    build_sample_cache,
    load_sample_cache,
    save_sample_cache,
)
from sae_experiments.core.config import load_config  # noqa: E402
from sae_experiments.core.sparse_autoencoder import SparseAutoencoder  # noqa: E402
from sae_experiments.utils.config_utils import resolve_dtype  # noqa: E402
from sae_experiments.utils.script_utils import (  # noqa: E402
    load_llava_components,
    setup_experiment,
)

PHASES = (
    "gate",
    "primary",
    "nested",
    "non_nested",
    "leave_one_out",
    "budget",
    "downstream",
    "sensitivity",
)


# --------------------------------------------------------------------------- conditions


def build_conditions(experiment, config, phases):
    """Enumerate every condition the requested phases call for.

    Returns them in dependency order: the gate first, because nothing downstream means
    anything until A0 reproduces the published single-layer number.
    """
    ml_cfg = config.get("multilayer", {})
    cond_cfg = config.get("conditions", {})
    span = [int(l) for l in ml_cfg.get("layers", [])]
    k = ml_cfg.get("features_per_layer", 200)
    mode = config.get("ablation", {}).get("mode", "replace")
    flow = (config.get("knockout", {}).get("flows") or ["Image->Question"])[0]

    conditions = []

    if "gate" in phases:
        # A0 is the regression test for the whole feature_ablator refactor: one layer,
        # top-200, through the multi-layer harness, must reproduce 0.2131.
        conditions.append(
            Condition(
                condition_id="A0_regression_L11",
                kind=KIND_SAE,
                features=experiment.features_for([11], 200),
                mode=mode,
                label="harness regression vs published layer-11 result",
            )
        )
        conditions.append(
            Condition(
                condition_id="gate_none",
                kind=KIND_NONE,
                features={},
                mode=mode,
                label="no intervention; must be exactly 0.0",
            )
        )
        # Pass-through over each nested span: replace mode swaps the hidden state for the
        # SAE reconstruction, so stacking N of them stacks N layers of reconstruction error.
        # A layer mapped to [] is hooked but ablates nothing, which is what measures it.
        for nested in cond_cfg.get("nested", []):
            layers = [int(l) for l in nested]
            conditions.append(
                Condition(
                    condition_id=f"gate_passthrough_L{_compact_layer_range(layers)}",
                    kind=KIND_PASSTHROUGH,
                    features={layer: [] for layer in layers},
                    mode=mode,
                    label="reconstruction error only, no features ablated",
                )
            )
        conditions.append(
            Condition(
                condition_id=f"gate_passthrough_delta_L{_compact_layer_range(span)}",
                kind=KIND_PASSTHROUGH,
                features={layer: [] for layer in span},
                mode="residual",
                label="delta mode passthrough; ~0 by construction, a harness check",
            )
        )

    if "primary" in phases:
        conditions.append(
            Condition(
                condition_id=f"joint_L{_compact_layer_range(span)}",
                kind=KIND_SAE,
                features=experiment.features_for(span, k),
                mode=mode,
                label="the headline joint ablation",
            )
        )
        conditions.append(
            Condition(
                condition_id=f"span_knockout_L{_compact_layer_range(span)}",
                kind=KIND_KNOCKOUT,
                features={},
                knockout_layers=tuple(span),
                flow=flow,
                label="the ceiling, measured on the ablation's own samples",
            )
        )
        # Single-layer knockouts on the same samples, so every "% of ceiling" has a
        # denominator from the same sample set as its numerator. The published ratios
        # divided an n=256 unfiltered ablation by an n=7084 filtered knockout.
        for layer in span:
            conditions.append(
                Condition(
                    condition_id=f"knockout_L{layer}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=(layer,),
                    flow=flow,
                    label="single-layer ceiling on the shared sample set",
                )
            )

    if "nested" in phases:
        for nested in cond_cfg.get("nested", []):
            layers = [int(l) for l in nested]
            if layers == span:
                continue  # already covered by the primary joint condition
            tag = _compact_layer_range(layers)
            conditions.append(
                Condition(
                    condition_id=f"nested_L{tag}",
                    kind=KIND_SAE,
                    features=experiment.features_for(layers, k),
                    mode=mode,
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"nested_knockout_L{tag}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=tuple(layers),
                    flow=flow,
                    label="knockout saturation curve, the null the ablation curve is read against",
                )
            )

    if "non_nested" in phases:
        for group in cond_cfg.get("non_nested", []):
            layers = [int(l) for l in group]
            tag = _compact_layer_range(layers)
            conditions.append(
                Condition(
                    condition_id=f"nonnested_L{tag}",
                    kind=KIND_SAE,
                    features=experiment.features_for(layers, k),
                    mode=mode,
                    label="separates span size from span depth",
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"nonnested_knockout_L{tag}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=tuple(layers),
                    flow=flow,
                )
            )

    if "leave_one_out" in phases and cond_cfg.get("leave_one_out", True):
        for dropped in span:
            remaining = [l for l in span if l != dropped]
            conditions.append(
                Condition(
                    condition_id=f"loo_drop{dropped}",
                    kind=KIND_SAE,
                    features=experiment.features_for(remaining, k),
                    mode=mode,
                    label=f"in-context marginal contribution of layer {dropped}",
                )
            )

    if "budget" in phases:
        budget = cond_cfg.get("budget_matched", {})
        spread = int(budget.get("spread_per_layer", 40))
        conditions.append(
            Condition(
                condition_id=f"budget_spread{spread}x{len(span)}",
                kind=KIND_SAE,
                features=experiment.features_for(span, spread),
                mode=mode,
                label=f"{spread * len(span)} features spread across {len(span)} layers",
            )
        )
        # The concentrated arm is a curve, not a point: a single k=200 comparison would
        # confound feature rank and count saturation with the layer distribution.
        concentrated_layer = int(budget.get("concentrated_layer", 11))
        for count in budget.get("concentrated_k", []):
            conditions.append(
                Condition(
                    condition_id=f"budget_concentrated_L{concentrated_layer}_k{count}",
                    kind=KIND_SAE,
                    features=experiment.features_for([concentrated_layer], int(count)),
                    mode=mode,
                    label="concentrated count curve at equal and unequal budgets",
                )
            )

    if "downstream" in phases:
        downstream = cond_cfg.get("downstream_knockout", {})
        last_layer = int(downstream.get("downstream_to", 31))
        for anchor in downstream.get("anchor_layers", []):
            anchor = int(anchor)
            tail = tuple(range(anchor + 1, last_layer + 1))
            features = experiment.features_for([anchor], k)
            conditions.append(
                Condition(
                    condition_id=f"downstream_ablate_L{anchor}",
                    kind=KIND_SAE,
                    features=features,
                    mode=mode,
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"downstream_knockout_L{anchor + 1}-{last_layer}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=tail,
                    flow=flow,
                    label="blocks re-reading without ablating anything",
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"downstream_combined_L{anchor}",
                    kind=KIND_COMBINED,
                    features=features,
                    knockout_layers=tail,
                    flow=flow,
                    mode=mode,
                    label="re-reading blocked AND features removed",
                )
            )

    if "sensitivity" in phases:
        sens = [int(l) for l in cond_cfg.get("sensitivity_span", [])]
        if sens:
            tag = _compact_layer_range(sens)
            conditions.append(
                Condition(
                    condition_id=f"sensitivity_joint_L{tag}",
                    kind=KIND_SAE,
                    features=experiment.features_for(sens, k),
                    mode=mode,
                    label="span without the sign-inconsistent layer 13",
                )
            )
            conditions.append(
                Condition(
                    condition_id=f"sensitivity_knockout_L{tag}",
                    kind=KIND_KNOCKOUT,
                    features={},
                    knockout_layers=tuple(sens),
                    flow=flow,
                )
            )
        # Isolates the last layer's ablation contribution from its reconstruction error:
        # ablate 10..13 while layer 14 carries a pass-through hook.
        if len(span) > 1:
            head, tail_layer = span[:-1], span[-1]
            features = experiment.features_for(head, k)
            features[tail_layer] = []
            conditions.append(
                Condition(
                    condition_id=f"sensitivity_passthrough_tail_L{tail_layer}",
                    kind=KIND_SAE,
                    features=features,
                    mode=mode,
                    label=f"layers {_compact_layer_range(head)} ablated, layer {tail_layer} pass-through",
                )
            )

    return conditions


def controls_for(condition, phases, config):
    """How many joint control sets a condition gets.

    Only conditions that carry a significance claim need controls, and only the primary
    one needs the full 15.
    """
    if condition.kind not in (KIND_SAE, KIND_COMBINED):
        return 0
    n_full = int(config.get("random_control", {}).get("n_random_sets", 15))
    if condition.condition_id.startswith(("joint_", "A0_")):
        return n_full
    if condition.condition_id.startswith(("nested_L", "nonnested_L", "budget_spread")):
        return 1
    return 0


# --------------------------------------------------------------------------- loading


def load_multilayer_saes(config, model):
    ml_cfg = config.get("multilayer", {})
    dtype = resolve_dtype(ml_cfg.get("sae_dtype", "float32"))
    device = next(model.parameters()).device if model is not None else "cpu"

    saes = {}
    for layer in ml_cfg.get("layers", []):
        layer = int(layer)
        path = ml_cfg["sae_paths"][layer]
        checkpoint = torch_load(path)
        state = checkpoint.get("state", {})
        sae_state = state.get("sae_state", checkpoint)
        # Take dimensions from the checkpoint rather than the config: with one config
        # covering several layers there is no single n_features to read.
        n_features, d_model = sae_state["encoder.weight"].shape
        sae = SparseAutoencoder(
            d_model=int(d_model),
            n_features=int(n_features),
            l1_coeff=float(config.get("sae", {}).get("l1_coeff", 1e-3)),
        )
        sae.load_state_dict(sae_state)
        sae.to(device=device, dtype=dtype)
        sae.eval()
        saes[layer] = sae
        print(f"[multilayer] layer {layer}: SAE {n_features}x{d_model} from {path}")
    return saes


def torch_load(path):
    import torch

    return torch.load(path, map_location="cpu")


def load_catalogs_and_stats(config):
    ml_cfg = config.get("multilayer", {})
    catalogs, stats = {}, {}
    for layer in ml_cfg.get("layers", []):
        layer = int(layer)
        with open(ml_cfg["catalog_paths"][layer]) as handle:
            catalog = json.load(handle)
        # Catalogs are written in descending causal-score order; preserve it.
        catalogs[layer] = [int(k) for k in catalog.keys()]

        stats_path = ml_cfg.get("stats_paths", {}).get(layer)
        if stats_path and os.path.exists(stats_path):
            with open(stats_path) as handle:
                stats[layer] = {int(k): v for k, v in json.load(handle).items()}
        else:
            stats[layer] = {}
        print(
            f"[multilayer] layer {layer}: {len(catalogs[layer])} catalog features, "
            f"{len(stats[layer])} scored features"
        )
    return catalogs, stats


# --------------------------------------------------------------------------- checkpoint


def completed_conditions(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["condition_id"])
    return done


def append_checkpoint(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phases",
        default="gate",
        help=f"comma-separated subset of {','.join(PHASES)}, or 'all'",
    )
    parser.add_argument("--conditions", default=None, help="comma-separated condition ids")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument(
        "--mode",
        default=None,
        help="override ablation.mode, e.g. 'residual' if the pass-through gate trips",
    )
    parser.add_argument("--force", action="store_true", help="re-run completed conditions")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print the condition matrix and exit without loading the model",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode:
        config.data["ablation"] = dict(config.get("ablation", {}), mode=args.mode)

    phases = list(PHASES) if args.phases == "all" else [
        p.strip() for p in args.phases.split(",") if p.strip()
    ]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        parser.error(f"unknown phase(s) {unknown}; choose from {list(PHASES)}")

    if args.dry_run:
        return dry_run(config, phases, args)

    experiment_dir, seed = setup_experiment(args, config)
    show_progress = not args.no_progress

    tokenizer, model, image_processor = load_llava_components(config.get("model", {}))
    model.eval()
    dataset = build_dataset(config, tokenizer, image_processor, model)

    saes = load_multilayer_saes(config, model)
    catalogs, stats = load_catalogs_and_stats(config)
    experiment = MultiLayerAblationExperiment(model, saes, catalogs, stats, config)

    conditions = build_conditions(experiment, config, phases)
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in conditions if c.condition_id in wanted]

    checkpoint_path = os.path.join(experiment_dir, "checkpoint.jsonl")
    done = set() if args.force else completed_conditions(checkpoint_path)

    # One cache for the whole run: the baseline and the resolved positions do not depend
    # on the condition, so paying for them once instead of once per condition is roughly
    # half the wall clock across the matrix.
    cache_path = os.path.join(experiment_dir, "sample_cache.json")
    if os.path.exists(cache_path) and not args.force:
        sample_records = load_sample_cache(cache_path)
        print(f"[multilayer] reusing sample cache: {len(sample_records)} samples")
    else:
        print("[multilayer] building sample cache...")
        sample_records = build_sample_cache(
            experiment.ablator,
            dataset,
            position_type=experiment.position_type,
            logprob_normalize=experiment.logprob_normalize,
            max_samples=args.max_samples,
            show_progress=show_progress,
        )
        save_sample_cache(sample_records, cache_path)
    print(f"[multilayer] {len(sample_records)} samples cached\n")

    rc_cfg = config.get("random_control", {})
    sampling = str(rc_cfg.get("sampling", "matched"))
    matched_metric = str(rc_cfg.get("matched_metric", "activation_mean"))
    strict = bool(rc_cfg.get("strict_matching", True))
    print(f"[multilayer] control regime per layer: "
          f"{experiment.effective_sampling(sampling, matched_metric)}\n")

    for index, condition in enumerate(conditions, start=1):
        if condition.condition_id in done:
            print(f"[SKIP {index}/{len(conditions)}] {condition.condition_id}")
            continue

        print(f"[RUN {index}/{len(conditions)}] {condition.describe()}")
        started = time.time()
        rows, summary = experiment.run_condition(
            dataset,
            condition,
            sample_records,
            max_samples=args.max_samples,
            show_progress=show_progress,
        )

        n_controls = controls_for(condition, phases, config)
        control_summaries, control_sets = [], []
        if n_controls:
            rng = random.Random(int(rc_cfg.get("seed", seed)))
            for set_index in range(n_controls):
                control_features = experiment.sample_joint_random(
                    condition.features,
                    rng=rng,
                    sampling=sampling,
                    matched_metric=matched_metric,
                    strict=strict,
                )
                control_condition = Condition(
                    condition_id=f"{condition.condition_id}__control{set_index}",
                    kind=condition.kind,
                    features=control_features,
                    knockout_layers=condition.knockout_layers,
                    flow=condition.flow,
                    mode=condition.mode,
                )
                _, control_summary = experiment.run_condition(
                    dataset,
                    control_condition,
                    sample_records,
                    max_samples=args.max_samples,
                    show_progress=False,
                )
                control_summary["set_index"] = set_index
                control_summaries.append(control_summary)
                control_sets.append(
                    {str(l): v for l, v in sorted(control_features.items())}
                )
                print(
                    f"    control {set_index + 1}/{n_controls}: "
                    f"margin_drop {control_summary.get('mean_margin_drop')}"
                )

        payload = {
            "condition_id": condition.condition_id,
            "label": condition.label,
            "summary": summary,
            "control_summaries": control_summaries,
            "control_feature_sets": control_sets,
            "significance": (
                experiment.compare_to_controls(summary, control_summaries)
                if control_summaries
                else {}
            ),
            "meta": {
                "feature_selection": "causal_v2",
                "intervention": "multilayer_sae_ablation",
                "layers": condition.layers,
                "knockout_layers": list(condition.knockout_layers),
                "mode": condition.mode,
                "encode_mode": config.get("multilayer", {}).get("encode_mode", "live"),
                "control": f"{sampling}_{matched_metric}" + ("_strict" if strict else ""),
                "control_effective": experiment.effective_sampling(sampling, matched_metric),
                "n_random_sets": n_controls,
                "n_samples": summary.get("n_samples"),
                "seed": seed,
                # Kept for readers of the older single-layer result files.
                "ablation_version": "v2_causal_features",
            },
        }
        write_condition(experiment_dir, condition.condition_id, rows, payload)
        append_checkpoint(checkpoint_path, {
            "condition_id": condition.condition_id,
            "mean_margin_drop": summary.get("mean_margin_drop"),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(
            f"    margin_drop {summary.get('mean_margin_drop'):+.4f} "
            f"in {time.time() - started:.0f}s\n"
        )

    print(f"[multilayer] done. Results under {experiment_dir}/conditions/")


def write_condition(experiment_dir, condition_id, rows, payload):
    target = os.path.join(experiment_dir, "conditions", condition_id)
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "results.json"), "w") as handle:
        json.dump({"per_sample": rows, **payload}, handle, indent=2)
    with open(os.path.join(target, "summary.json"), "w") as handle:
        json.dump(payload, handle, indent=2)


def build_dataset(config, tokenizer, image_processor, model):
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    if dataset_cfg.get("format") == "clevr_lite":
        from sae_experiments.data.clevr_lite_dataset import CLEVRLiteVQADataset

        return CLEVRLiteVQADataset(
            data_dir=dataset_cfg.get("data_dir", "datasets/clevr_lite"),
            split=dataset_cfg.get("split", "val"),
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_config=model.config,
            conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
            filter_held_out=dataset_cfg.get("filter_held_out"),
        )

    from sae_experiments.data.attribute_dataset import AttributeVQADataset

    return AttributeVQADataset(
        csv_path=dataset_cfg.get("refined_dataset"),
        image_folder=dataset_cfg.get("image_folder"),
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
    )


def dry_run(config, phases, args):
    """Print the matrix without loading the model, so it can be reviewed first."""
    catalogs, stats = load_catalogs_and_stats(config)
    ml_cfg = config.get("multilayer", {})

    experiment = MultiLayerAblationExperiment.__new__(MultiLayerAblationExperiment)
    experiment.catalogs = catalogs
    experiment.feature_stats = stats
    experiment.saes = {}
    experiment.config = config

    conditions = build_conditions(experiment, config, phases)
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in conditions if c.condition_id in wanted]

    total_runs = 0
    print(f"\nPhases: {', '.join(phases)}")
    print(f"Span: {ml_cfg.get('layers')}  excluded: {ml_cfg.get('excluded_layers')}")
    print(f"Samples per condition: {args.max_samples}\n")
    print(f"{'#':>3}  {'condition':<44} {'runs':>5}  detail")
    print("-" * 118)
    for index, condition in enumerate(conditions, start=1):
        n_controls = controls_for(condition, phases, config)
        runs = 1 + n_controls
        total_runs += runs
        print(f"{index:>3}  {condition.condition_id:<44} {runs:>5}  {condition.describe()}")
        if condition.label:
            print(f"{'':>3}  {'':<44} {'':>5}  -> {condition.label}")

    rc_cfg = config.get("random_control", {})
    print("-" * 118)
    print(f"{len(conditions)} conditions, {total_runs} evaluation runs of "
          f"{args.max_samples} samples each")
    print(
        f"\nControl regime: sampling={rc_cfg.get('sampling')} "
        f"metric={rc_cfg.get('matched_metric')} strict={rc_cfg.get('strict_matching')}"
    )
    print(f"Effective per layer: {experiment.effective_sampling(str(rc_cfg.get('sampling', 'matched')), str(rc_cfg.get('matched_metric', 'activation_mean')))}")
    return 0


if __name__ == "__main__":
    main()

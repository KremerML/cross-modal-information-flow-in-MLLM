"""Run v2 ablation experiments with error-term preservation and pass-through baseline.

Changes from 03_run_ablation.py:
- Always uses error-preserving delta mode (acts + SAE_mod - SAE_orig)
- Adds SAE pass-through baseline (zero features zeroed) to quantify reconstruction error
- Loads causal feature catalogs from 02b output
- Reports pass-through effect alongside binding/random comparison
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.ablation.ablation_experiments import AblationExperiment
from sae_experiments.ablation.feature_ablator import FeatureAblator
from sae_experiments.core.config import load_config
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.utils.config_utils import resolve_primary_task_type, resolve_task_types
from sae_experiments.utils.script_utils import setup_experiment, load_llava_components, load_sae


def run_passthrough_baseline(
    ablator: FeatureAblator,
    dataset,
    position_type: str,
    max_samples=None,
    show_progress: bool = False,
) -> dict:
    """Run SAE pass-through (zero features zeroed) to measure reconstruction error."""
    results = ablator.batch_ablation_experiment(
        dataset,
        feature_indices=[],
        position_type=position_type,
        mode="residual",
        show_progress=show_progress,
        max_samples=max_samples,
        score_options=True,
    )
    return ablator.compute_ablation_effect(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--skip_passthrough", action="store_true",
                        help="Skip the SAE pass-through baseline")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    ablation_cfg = config.get("ablation", {})
    eval_cfg = config.get("evaluation", {})
    experiment_dir, seed = setup_experiment(args, config)
    tokenizer, model, image_processor = load_llava_components(model_cfg)

    dataset_format = data_cfg.get("format", "csv")
    if dataset_format == "clevr_lite":
        from sae_experiments.data.clevr_lite_dataset import CLEVRLiteVQADataset
        dataset = CLEVRLiteVQADataset(
            data_dir=data_cfg.get("data_dir", "datasets/clevr_lite"),
            split="val",
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_config=model.config,
            conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
        )
    else:
        from sae_experiments.data.attribute_dataset import AttributeVQADataset
        dataset = AttributeVQADataset(
            refined_dataset=data_cfg.get("refined_dataset", ""),
            image_folder=data_cfg.get("image_folder", ""),
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_config=model.config,
            task_type=resolve_primary_task_type(data_cfg.get("task_types")),
            conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
        )

    checkpoint_path = args.sae_checkpoint or os.path.join(experiment_dir, "sae_checkpoint.pt")
    sae = load_sae(config, model, checkpoint_path)

    target_layer = model_cfg.get("target_layer", 0)
    activation_site = model_cfg.get("activation_site", "residual")
    position_type = ablation_cfg.get("position_type", "question")
    show_progress = not args.no_progress

    print(f"[03b] layer={target_layer}, site={activation_site}, "
          f"position_type={position_type}, n_items={len(dataset.questions)}")

    ablator = FeatureAblator(model, sae, target_layer, activation_site=activation_site)

    # Phase 1: SAE pass-through baseline
    passthrough_summary = None
    if not args.skip_passthrough:
        print("[03b] Phase 1: SAE pass-through baseline (zero features zeroed)...")
        passthrough_summary = run_passthrough_baseline(
            ablator, dataset, position_type,
            max_samples=args.max_samples, show_progress=show_progress,
        )
        pt_md = passthrough_summary.get("mean_margin_drop", 0) or 0
        pt_rp = passthrough_summary.get("mean_relative_perturbation", 0) or 0
        print(f"[03b] Pass-through: margin_drop={pt_md:+.4f}, rel_perturb={pt_rp:.4f}")
        if abs(pt_md) > 0.05:
            print(f"[03b] WARNING: Pass-through margin_drop is {pt_md:+.4f} — "
                  f"reconstruction error may confound results")

    # Phase 2: Load causal features
    causal_dir = experiment_dir + "_causal"
    features_path = args.features or os.path.join(causal_dir, "causal_feature_catalog.json")
    if not os.path.exists(features_path):
        features_path = os.path.join(experiment_dir, "feature_catalog.json")
    catalog = FeatureCatalog()
    catalog.load_from_json(features_path)
    binding_features = list(catalog.features.keys())
    print(f"[03b] Loaded {len(binding_features)} features from {features_path}")

    feature_stats_path = os.path.join(causal_dir, "causal_feature_stats.json")
    feature_stats = None
    if os.path.exists(feature_stats_path):
        with open(feature_stats_path) as f:
            feature_stats = {int(k): v for k, v in json.load(f).items()}

    # Phase 3: Three-condition ablation (binding vs random)
    print("[03b] Phase 2: Three-condition ablation test...")
    experiment = AblationExperiment(model, sae, config)
    results = experiment.run_three_condition_test(
        dataset,
        binding_features,
        feature_stats=feature_stats,
        show_progress=show_progress,
        max_samples=args.max_samples,
    )

    results["passthrough_baseline"] = passthrough_summary
    results["meta"] = {
        "seed": seed,
        "activation_site": activation_site,
        "ablation_version": "v2_error_preserving",
        "features_source": features_path,
    }

    # Save results
    output_path = args.output or os.path.join(
        causal_dir, "results", "ablation_v2_results.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    b = results.get("binding", {})
    r = results.get("random", {})
    sig = results.get("significance", {})
    pt = passthrough_summary or {}

    print(f"[03b] ── Results ────────────────────────────────────────")
    print(f"[03b]   Baseline accuracy  : {results.get('baseline', {}).get('baseline_accuracy', 0):.3f}")
    if pt:
        print(f"[03b]   Pass-through      │ margin_drop={pt.get('mean_margin_drop', 0):+.4f}  "
              f"rel_perturb={pt.get('mean_relative_perturbation', 0):.4f}")
    print(f"[03b]   Binding  (n={len(binding_features):>3})  │ "
          f"margin_drop={b.get('mean_margin_drop', 0):+.4f}  "
          f"acc_drop={b.get('accuracy_drop', 0):+.4f}  "
          f"rel_perturb={b.get('mean_relative_perturbation', 0):.4f}")
    print(f"[03b]   Random             │ "
          f"margin_drop={r.get('mean_margin_drop', 0):+.4f}  "
          f"acc_drop={r.get('accuracy_drop', 0):+.4f}  "
          f"rel_perturb={r.get('mean_relative_perturbation', 0):.4f}")

    md_sig = sig.get("mean_margin_drop", {})
    if md_sig:
        print(f"[03b]   Significance       │ z={md_sig.get('z_score', 'N/A'):.1f}, "
              f"p={md_sig.get('empirical_p_value', 'N/A')}")

    # Corrected margin drop (subtract pass-through baseline)
    if pt and pt.get("mean_margin_drop") is not None and b.get("mean_margin_drop") is not None:
        corrected = b["mean_margin_drop"] - pt["mean_margin_drop"]
        print(f"[03b]   Corrected binding  │ margin_drop={corrected:+.4f} "
              f"(after subtracting pass-through)")

    print(f"[03b] ──────────────────────────────────────────────────")
    print(f"[03b] Saved to {output_path}")


if __name__ == "__main__":
    main()

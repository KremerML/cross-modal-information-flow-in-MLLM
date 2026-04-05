"""Run ablation experiments for SAE features."""

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
from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.analysis.result_schema import validate_ablation_results
from sae_experiments.utils.config_utils import resolve_primary_task_type, resolve_task_types
from sae_experiments.utils.script_utils import setup_experiment, load_llava_components, load_sae


def main() -> None:
    """Run three-condition feature ablation and persist structured results.

    Args:
        None: Arguments are consumed from CLI flags parsed in this function.

    Returns:
        None: Writes validated ablation results JSON to disk.

    Raises:
        FileNotFoundError: If config, checkpoint, or feature catalog files are missing.
        ValueError: If the resolved task configuration is invalid.
        RuntimeError: If ablation execution fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    experiment_dir, seed = setup_experiment(args, config)
    tokenizer, model, image_processor = load_llava_components(model_cfg)

    task_types = resolve_task_types(data_cfg.get("task_types"))
    task_type = resolve_primary_task_type(task_types)

    dataset = AttributeVQADataset(
        refined_dataset=data_cfg.get("refined_dataset", ""),
        image_folder=data_cfg.get("image_folder", ""),
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        task_type=task_type,
        conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
    )

    checkpoint_path = args.sae_checkpoint or os.path.join(experiment_dir, "sae_checkpoint.pt")
    sae = load_sae(config, model, checkpoint_path)

    ablation_cfg = config.get("ablation", {})
    model_cfg_log = config.get("model", {})
    random_cfg = config.get("random_control", {})
    print(f"[03] layer={model_cfg_log.get('target_layer')}, "
          f"site={model_cfg_log.get('activation_site')}, "
          f"position_type={ablation_cfg.get('position_type')}, "
          f"mode={ablation_cfg.get('mode')}, "
          f"n_random_sets={ablation_cfg.get('n_random_sets', random_cfg.get('n_random_sets', 1))}, "
          f"dataset={data_cfg.get('refined_dataset','')}, n_items={len(dataset.questions)}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[03] GPU memory: {free/1024**3:.2f} GB free / {total/1024**3:.2f} GB total")

    catalog = FeatureCatalog()
    features_path = args.features or os.path.join(experiment_dir, "feature_catalog.json")
    catalog.load_from_json(features_path)
    binding_features = list(catalog.features.keys())
    print(f"[03] binding features loaded: {len(binding_features)} from {features_path}")
    feature_stats_path = os.path.join(experiment_dir, "feature_stats.json")
    feature_stats = None
    if os.path.exists(feature_stats_path):
        with open(feature_stats_path, "r", encoding="utf-8") as handle:
            feature_stats = {int(k): v for k, v in json.load(handle).items()}

    experiment = AblationExperiment(model, sae, config)
    show_progress = not args.no_progress
    results = experiment.run_three_condition_test(
        dataset,
        binding_features,
        feature_stats=feature_stats,
        progress_label=task_type,
        show_progress=show_progress,
        max_samples=args.max_samples,
    )
    results["meta"] = {
        "seed": seed,
        "deterministic": bool(config.get("reproducibility", {}).get("deterministic", True)),
        "activation_site": model_cfg.get("activation_site", "residual"),
        "task_type": task_type,
        "configured_task_types": task_types,
    }
    results["task_specificity"] = {
        "skipped": True,
        "reason": "03_run_ablation.py now runs only dataset.task_types[0] from config.",
    }

    output_path = args.output or os.path.join(experiment_dir, "results", "ablation_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    validate_ablation_results(results)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    # Print key outcome summary
    b = results.get("binding", {})
    r = results.get("random", {})
    sig = results.get("significance", {})
    print(f"[03] ── Results summary ──────────────────────────────")
    print(f"[03]   Baseline accuracy  : {results.get('baseline', {}).get('baseline_accuracy', 0):.3f}")
    print(f"[03]   Binding  │ acc_drop={b.get('accuracy_drop', 0):+.4f}  margin_drop={b.get('mean_margin_drop', 0):+.4f}  rel_perturb={b.get('mean_relative_perturbation', 0):.4f}")
    print(f"[03]   Random   │ acc_drop={r.get('accuracy_drop', 0):+.4f}  margin_drop={r.get('mean_margin_drop', 0):+.4f}  rel_perturb={r.get('mean_relative_perturbation', 0):.4f}")
    print(f"[03]   Significance: acc p={sig.get('accuracy_drop_p', 'N/A')}, margin p={sig.get('margin_drop_p', 'N/A')}")
    print(f"[03]   Reference ceilings: L0 all-position ceiling=0.314, knockout=0.540")
    print(f"[03] ────────────────────────────────────────────────")
    print(f"Saved ablation results to {output_path}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()

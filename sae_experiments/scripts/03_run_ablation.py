"""Run ablation experiments for SAE features."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.ablation.ablation_experiments import AblationExperiment
from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder
from sae_experiments.analysis.result_schema import validate_ablation_results
from sae_experiments.utils.checkpoint_utils import resolve_experiment_dir
from sae_experiments.utils.config_utils import resolve_primary_task_type, resolve_task_types
from sae_experiments.utils.random_utils import resolve_seed, set_global_seed


def _resolve_dtype(value: str) -> torch.dtype:
    """Map a config dtype string to a torch dtype.

    Args:
        value (str): Text dtype token (for example ``float32`` or ``fp16``).

    Returns:
        torch.dtype: Resolved dtype for SAE loading/evaluation.

    Raises:
        None: Unknown values default to ``torch.float32``.
    """
    value = str(value).lower()
    if value in ("float16", "fp16", "half"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


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
    reproducibility_cfg = config.get("reproducibility", {})
    training_cfg = config.get("training", {})
    seed = resolve_seed(
        reproducibility_cfg.get("seed", training_cfg.get("seed")),
        fallback_seed=42,
    )
    set_global_seed(
        seed,
        deterministic=bool(reproducibility_cfg.get("deterministic", True)),
        benchmark=bool(reproducibility_cfg.get("benchmark", False)),
    )
    experiment_cfg = dict(config.get("experiment", {}))
    if args.experiment_name:
        experiment_cfg["name"] = args.experiment_name
        experiment_cfg.pop("output_dir", None)
    experiment_dir = resolve_experiment_dir(experiment_cfg, args.experiment_dir)

    model_path = os.path.expanduser(model_cfg.get("name", ""))
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_cfg.get("model_base"),
        model_name,
        device_map="auto",
        attn_implementation=None,
    )
    model.eval()

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

    sae = SparseAutoencoder(
        d_model=model_cfg.get("d_model", 4096),
        n_features=config.get("sae", {}).get("n_features", 32768),
        l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
    )
    checkpoint_path = args.sae_checkpoint or os.path.join(experiment_dir, "sae_checkpoint.pt")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sae.load_state_dict(ckpt.get("state", {}).get("sae_state", ckpt))
    train_cfg = config.get("training", {})
    sae.to(device=next(model.parameters()).device, dtype=_resolve_dtype(train_cfg.get("dtype", "float32")))
    sae.eval()

    catalog = FeatureCatalog()
    features_path = args.features or os.path.join(experiment_dir, "feature_catalog.json")
    catalog.load_from_json(features_path)
    binding_features = list(catalog.features.keys())
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
        "deterministic": bool(reproducibility_cfg.get("deterministic", True)),
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

    print(f"Saved ablation results to {output_path}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()

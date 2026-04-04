"""Run ablation experiments for SAE features."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.ablation.ablation_experiments import AblationExperiment
from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.analysis.result_schema import validate_ablation_results
from sae_experiments.utils.config_utils import resolve_primary_task_type, resolve_task_types
from sae_experiments.data.paligemma_dataset import PaliGemmaChooseAttrDataset
from sae_experiments.utils.script_utils import setup_experiment, load_paligemma_components, load_gemma_scope_sae


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
    processor, model = load_paligemma_components(model_cfg)
    tokenizer = processor.tokenizer

    import pandas as pd
    refined_dataset = data_cfg.get("refined_dataset", "")
    df = pd.read_csv(refined_dataset, dtype={"question_id": str}).fillna("")
    dataset_dict = df.set_index("question_id").T.to_dict("dict")
    questions = [{**detail, "q_id": qu_id} for qu_id, detail in dataset_dict.items()]

    dataset = PaliGemmaChooseAttrDataset(
        questions=questions,
        dataset_dict=dataset_dict,
        image_folder=data_cfg.get("image_folder", ""),
        processor=processor,
    )

    sae = load_gemma_scope_sae(config, model)

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

    print(f"Saved ablation results to {output_path}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()

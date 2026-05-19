"""Visualize SAE features and generate dashboards."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.feature_analysis.feature_identifier import FeatureIdentifier
from sae_experiments.feature_analysis.feature_visualizer import FeatureVisualizer
from sae_experiments.utils.config_utils import resolve_primary_task_type
from sae_experiments.utils.script_utils import setup_experiment, load_llava_components, load_sae


def main() -> None:
    """Build a feature dashboard for the selected SAE feature catalog.

    Args:
        None: CLI arguments are parsed within this function.

    Returns:
        None: Writes feature visualizations and dashboard assets to disk.

    Raises:
        FileNotFoundError: If checkpoint or catalog inputs are missing.
        ValueError: If configuration values are invalid.
        RuntimeError: If activation collection or rendering fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--catalog", type=str, default=None)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    experiment_dir, _ = setup_experiment(args, config)
    tokenizer, model, image_processor = load_llava_components(model_cfg)

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

    catalog = FeatureCatalog()
    catalog_path = args.catalog or os.path.join(experiment_dir, "feature_catalog.json")
    catalog.load_from_json(catalog_path)
    features = list(catalog.features.keys())

    identifier = FeatureIdentifier(
        sae,
        model,
        dataset,
        model_cfg.get("target_layer", 12),
        activation_site=model_cfg.get("activation_site", "residual"),
    )
    identifier.compute_feature_activations(
        position_type="attribute",
        max_samples=args.max_samples,
        include_predictions=False,
    )

    output_dir = args.output or os.path.join(experiment_dir, "feature_dashboard")
    visualizer = FeatureVisualizer(sae, model, dataset, identifier.feature_acts, identifier.metadata)
    visualizer.create_feature_dashboard(features, output_dir)

    print(f"Saved feature dashboard to {output_dir}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()

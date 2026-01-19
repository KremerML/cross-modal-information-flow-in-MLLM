"""Visualize SAE features and generate dashboards."""

import argparse
import os

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.feature_analysis.feature_identifier import FeatureIdentifier
from sae_experiments.feature_analysis.feature_visualizer import FeatureVisualizer
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--catalog", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})

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

    dataset = AttributeVQADataset(
        refined_dataset=data_cfg.get("refined_dataset", ""),
        image_folder=data_cfg.get("image_folder", ""),
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        task_type=data_cfg.get("task_types", ["ChooseAttr"])[0],
        conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
    )

    sae = SparseAutoencoder(
        d_model=model_cfg.get("d_model", 4096),
        n_features=config.get("sae", {}).get("n_features", 32768),
        l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
    )
    ckpt = torch.load(args.sae_checkpoint, map_location="cpu")
    sae.load_state_dict(ckpt.get("state", {}).get("sae_state", ckpt))
    sae.eval()

    catalog = FeatureCatalog()
    catalog.load_from_json(args.catalog)
    features = list(catalog.features.keys())

    identifier = FeatureIdentifier(sae, model, dataset, model_cfg.get("target_layer", 12))
    identifier.compute_feature_activations(
        position_type="attribute",
        max_samples=args.max_samples,
        include_predictions=False,
    )

    output_dir = args.output or os.path.join(os.path.dirname(args.catalog), "feature_dashboard")
    visualizer = FeatureVisualizer(sae, model, dataset, identifier.feature_acts, identifier.metadata)
    visualizer.create_feature_dashboard(features, output_dir)

    print(f"Saved feature dashboard to {output_dir}")


if __name__ == "__main__":
    main()

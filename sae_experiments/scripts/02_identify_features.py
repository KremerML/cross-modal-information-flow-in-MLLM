"""Identify discriminative SAE features."""

import argparse
import os
from pathlib import Path
import sys

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.feature_analysis.feature_identifier import FeatureIdentifier
from sae_experiments.feature_analysis.feature_visualizer import FeatureVisualizer
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder


def _resolve_dtype(value: str) -> torch.dtype:
    value = str(value).lower()
    if value in ("float16", "fp16", "half"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    feat_cfg = config.get("feature_identification", {})
    paths_cfg = config.get("paths", {})

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
    train_cfg = config.get("training", {})
    sae.to(device=next(model.parameters()).device, dtype=_resolve_dtype(train_cfg.get("dtype", "float32")))
    sae.eval()

    identifier = FeatureIdentifier(sae, model, dataset, model_cfg.get("target_layer", 12))
    identifier.compute_feature_activations(
        position_type="attribute",
        max_samples=args.max_samples,
        include_predictions=True,
    )

    features = identifier.find_discriminative_features(
        threshold=feat_cfg.get("discrimination_threshold", 2.0)
    )
    top_features = features[: feat_cfg.get("top_k", 50)]

    catalog = FeatureCatalog()
    for feature_idx in top_features:
        stats = identifier.feature_stats.get(feature_idx, {})
        catalog.add_feature(
            feature_idx,
            {
                "name": f"feature_{feature_idx}",
                "type": "binding",
                "discrimination_score": stats.get("ratio", 0.0),
            },
        )

    catalog_path = paths_cfg.get("feature_catalog")
    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
    catalog.export_to_json(catalog_path)
    identifier.save_feature_statistics(os.path.join(os.path.dirname(catalog_path), "feature_stats.json"))

    visualizer = FeatureVisualizer(sae, model, dataset, identifier.feature_acts, identifier.metadata)
    for feature_idx in top_features[: min(10, len(top_features))]:
        visualizer.visualize_feature(
            feature_idx,
            os.path.join(os.path.dirname(catalog_path), f"feature_{feature_idx}.png"),
        )

    print(f"Saved feature catalog to {catalog_path}")


if __name__ == "__main__":
    main()

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
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
    control_dataset = dataset.create_control_dataset(task_type="ChooseRel")

    sae = SparseAutoencoder(
        d_model=model_cfg.get("d_model", 4096),
        n_features=config.get("sae", {}).get("n_features", 32768),
        l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
    )
    ckpt = torch.load(args.sae_checkpoint, map_location="cpu")
    sae.load_state_dict(ckpt.get("state", {}).get("sae_state", ckpt))
    sae.eval()

    catalog = FeatureCatalog()
    catalog.load_from_json(args.features)
    binding_features = list(catalog.features.keys())

    experiment = AblationExperiment(model, sae, config)
    results = experiment.run_three_condition_test(dataset, binding_features)
    specificity = experiment.test_task_specificity(binding_features, dataset, control_dataset)
    results["task_specificity"] = specificity

    output_path = args.output or os.path.join(paths_cfg.get("results_dir", "output/sae_experiments/results"), "ablation_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved ablation results to {output_path}")


if __name__ == "__main__":
    main()

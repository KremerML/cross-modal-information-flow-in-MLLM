"""Train SAE on LLaVA activations."""

import argparse
import os

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder
from sae_experiments.models.sae_trainer import SAETrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
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

    sae = SparseAutoencoder(
        d_model=model_cfg.get("d_model", 4096),
        n_features=config.get("sae", {}).get("n_features", 32768),
        l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
    )

    trainer = SAETrainer(
        sae=sae,
        config=config,
        target_layer=model_cfg.get("target_layer", 12),
        llava_model=model,
    )

    activations, _ = trainer.collect_activations(
        dataset,
        position_type="question",
        tokenizer=tokenizer,
        max_samples=args.max_samples,
    )

    history = trainer.train(activations)
    checkpoint_path = paths_cfg.get("sae_checkpoint")
    trainer.save_checkpoint(checkpoint_path, metadata={"history": history})

    print(f"Saved SAE checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()

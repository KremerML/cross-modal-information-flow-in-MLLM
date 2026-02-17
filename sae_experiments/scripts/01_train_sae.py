"""Train SAE on LLaVA activations."""

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

from sae_experiments.config.sae_config import load_config, save_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder
from sae_experiments.models.sae_trainer import SAETrainer
from sae_experiments.utils.checkpoint_utils import resolve_experiment_dir
from sae_experiments.utils.config_utils import (
    resolve_primary_task_type,
    resolve_training_position_type,
)
from sae_experiments.utils.sae_validation import compute_activation_stats, reconstruction_loss
from sae_experiments.utils.random_utils import resolve_seed, set_global_seed


def _parse_bool(value: object) -> bool:
    """Parse a permissive CLI boolean value.

    Args:
        value (object): Raw CLI value (bool or string-like token).

    Returns:
        bool: Parsed boolean value.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a supported boolean token.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> None:
    """Train an SAE checkpoint from collected LLaVA activations.

    Args:
        None: Arguments are supplied through CLI flags parsed in this function.

    Returns:
        None: Saves a trained SAE checkpoint and resolved config to the experiment directory.

    Raises:
        FileNotFoundError: If required dataset or checkpoint paths are missing.
        ValueError: If configuration values are invalid.
        RuntimeError: If model loading, activation collection, or training fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--position_type", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--show_progress", type=_parse_bool, default=True)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    reproducibility_cfg = config.get("reproducibility", {})
    training_cfg = config.get("training", {})
    feat_cfg = config.get("feature_identification", {})
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

    checkpoint_path = args.checkpoint_path or os.path.join(experiment_dir, "sae_checkpoint.pt")

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
        task_type=resolve_primary_task_type(data_cfg.get("task_types")),
        conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
    )

    sae = SparseAutoencoder(
        d_model=model_cfg.get("d_model", 4096),
        n_features=config.get("sae", {}).get("n_features", 32768),
        l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
    )

    target_layer = args.target_layer if args.target_layer is not None else model_cfg.get("target_layer", 12)
    trainer = SAETrainer(
        sae=sae,
        config=config,
        target_layer=target_layer,
        activation_site=model_cfg.get("activation_site", "residual"),
        llava_model=model,
    )

    activations, _ = trainer.collect_activations(
        dataset,
        position_type=resolve_training_position_type(
            args.position_type,
            training_cfg,
            feat_cfg,
            default="question",
        ),
        tokenizer=tokenizer,
        max_samples=args.max_samples,
    )

    history = trainer.train(activations, show_progress=args.show_progress)
    activation_stats = compute_activation_stats(activations)
    recon_loss = reconstruction_loss(trainer.sae, activations)
    trainer.save_checkpoint(
        checkpoint_path,
        metadata={
            "history": history,
            "activation_stats": activation_stats,
            "reconstruction_loss": recon_loss,
            "activation_samples": int(activations.shape[0]),
            "seed": seed,
            "deterministic": bool(reproducibility_cfg.get("deterministic", True)),
        },
    )
    save_config(config, os.path.join(experiment_dir, "config.yaml"))

    print(f"Saved SAE checkpoint to {checkpoint_path}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()

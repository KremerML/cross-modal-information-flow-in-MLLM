"""Train SAE on LLaVA activations."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.config.sae_config import load_config, save_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder
from sae_experiments.models.sae_trainer import SAETrainer
from sae_experiments.utils.config_utils import (
    resolve_primary_task_type,
    resolve_training_position_type,
)
from sae_experiments.utils.sae_validation import compute_activation_stats
from sae_experiments.utils.script_utils import setup_experiment, load_llava_components


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


def _split_activation_rows_by_question(
    metadata: List[Dict[str, Any]],
    test_ratio: float,
    split_seed: int,
    min_test_samples: int,
) -> Dict[str, Any]:
    """Split activation row indices by question ID for holdout evaluation.

    Args:
        metadata (List[Dict[str, Any]]): Activation metadata entries from collector output.
        test_ratio (float): Fraction of unique questions assigned to test split.
        split_seed (int): Seed for deterministic question-level shuffling.
        min_test_samples (int): Minimum number of unique test questions when possible.

    Returns:
        Dict[str, Any]: Split payload with question IDs and activation row indices.

    Raises:
        ValueError: If metadata is empty or cannot produce non-empty train/test splits.
    """
    spans: List[Dict[str, Any]] = []
    for item in metadata:
        question_id = str(item.get("question_id", "")).strip()
        if not question_id:
            continue
        try:
            start_idx = int(item.get("start_idx", 0))
            count = int(item.get("count", 0))
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        spans.append({"question_id": question_id, "start_idx": start_idx, "count": count})

    if not spans:
        raise ValueError("No activation metadata spans available for holdout split.")

    question_ids = sorted({span["question_id"] for span in spans})
    n_questions = len(question_ids)
    if n_questions < 2:
        raise ValueError("Need at least 2 questions to create a holdout split.")

    ratio = float(max(0.0, min(1.0, test_ratio)))
    rng = random.Random(int(split_seed))
    rng.shuffle(question_ids)

    test_count = int(round(n_questions * ratio))
    if ratio > 0.0 and test_count == 0:
        test_count = 1
    test_count = max(test_count, int(min_test_samples))
    test_count = min(test_count, n_questions - 1)
    if test_count < int(min_test_samples):
        warnings.warn(
            (
                "Could not satisfy holdout.min_test_samples with available questions. "
                f"Using {test_count} test questions out of {n_questions}."
            ),
            RuntimeWarning,
        )
    if test_count <= 0:
        raise ValueError("Holdout split produced zero test questions.")

    test_question_ids = question_ids[:test_count]
    test_question_set = set(test_question_ids)
    train_question_ids = question_ids[test_count:]

    train_indices: List[int] = []
    test_indices: List[int] = []
    for span in spans:
        row_range = list(range(span["start_idx"], span["start_idx"] + span["count"]))
        if span["question_id"] in test_question_set:
            test_indices.extend(row_range)
        else:
            train_indices.extend(row_range)

    if not train_indices or not test_indices:
        raise ValueError(
            "Holdout split produced an empty train or test activation set. "
            f"train_rows={len(train_indices)}, test_rows={len(test_indices)}"
        )

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_question_ids": train_question_ids,
        "test_question_ids": test_question_ids,
        "total_questions": n_questions,
    }


def _take_rows(activations: torch.Tensor, row_indices: Sequence[int]) -> torch.Tensor:
    """Select activation rows by index on the same device as input activations.

    Args:
        activations (torch.Tensor): Activation matrix of shape [n_rows, d_model].
        row_indices (Sequence[int]): Row indices to select.

    Returns:
        torch.Tensor: Selected activation rows.

    Raises:
        ValueError: If any row index is out of bounds.
    """
    if not row_indices:
        return activations[:0]
    max_index = max(int(idx) for idx in row_indices)
    if max_index >= int(activations.shape[0]):
        raise ValueError(
            f"Row index out of range for activations: max_index={max_index}, rows={activations.shape[0]}"
        )
    index_tensor = torch.tensor(
        list(row_indices),
        dtype=torch.long,
        device=activations.device,
    )
    return activations.index_select(0, index_tensor)


def _compute_reconstruction_metrics(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Compute reconstruction and sparsity metrics for a set of activations.

    Args:
        sae (SparseAutoencoder): Trained SAE model.
        activations (torch.Tensor): Activation rows to evaluate.
        batch_size (int): Evaluation batch size.

    Returns:
        Dict[str, Any]: Metrics including MSE, normalized MSE, explained variance, mean L0, and dead-feature fraction.

    Raises:
        RuntimeError: If SAE forward pass fails.
    """
    if activations.numel() == 0:
        return {
            "rows": 0,
            "mse": 0.0,
            "normalized_mse": None,
            "explained_variance": None,
            "mean_l0": 0.0,
            "dead_feature_fraction": 1.0,
        }

    sae.eval()
    param = next(sae.parameters())
    eval_batch_size = max(1, int(batch_size))

    total_se = 0.0
    total_sum = 0.0
    total_sum2 = 0.0
    total_values = 0
    total_l0 = 0.0
    total_rows = 0
    active_feature_mask = torch.zeros(sae.n_features, dtype=torch.bool, device=param.device)

    with torch.no_grad():
        for start in range(0, activations.shape[0], eval_batch_size):
            batch = activations[start : start + eval_batch_size]
            if batch.device != param.device or batch.dtype != param.dtype:
                batch = batch.to(device=param.device, dtype=param.dtype)

            recon, feats = sae.forward(batch)
            error = recon - batch
            total_se += torch.sum(error * error).item()
            total_sum += batch.sum().item()
            total_sum2 += torch.sum(batch * batch).item()
            total_values += int(batch.numel())

            feature_on = feats > 0
            total_l0 += feature_on.sum(dim=1).float().sum().item()
            total_rows += int(feature_on.shape[0])
            active_feature_mask |= feature_on.any(dim=0)

    mse = float(total_se / max(1, total_values))
    mean_value = float(total_sum / max(1, total_values))
    variance = float((total_sum2 / max(1, total_values)) - (mean_value * mean_value))
    variance = max(variance, 0.0)

    normalized_mse: Optional[float]
    explained_variance: Optional[float]
    if variance <= 1e-12:
        normalized_mse = None
        explained_variance = None
    else:
        normalized_mse = float(mse / variance)
        explained_variance = float(1.0 - normalized_mse)

    mean_l0 = float(total_l0 / max(1, total_rows))
    dead_feature_fraction = float(1.0 - active_feature_mask.float().mean().item())
    return {
        "rows": int(activations.shape[0]),
        "mse": mse,
        "normalized_mse": normalized_mse,
        "explained_variance": explained_variance,
        "mean_l0": mean_l0,
        "dead_feature_fraction": dead_feature_fraction,
    }


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
    parser.add_argument("--activations_path", type=str, default=None,
                        help="Directory of pre-collected activations (output of "
                             "collect_activations_multilayer.py). Skips forward pass.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    holdout_cfg = config.get("holdout", {})
    feat_cfg = config.get("feature_identification", {})
    training_cfg = config.get("training", {})
    experiment_dir, seed = setup_experiment(args, config)

    checkpoint_path = args.checkpoint_path or os.path.join(experiment_dir, "sae_checkpoint.pt")
    target_layer = args.target_layer if args.target_layer is not None else model_cfg.get("target_layer", 12)
    train_position_type = resolve_training_position_type(
        args.position_type,
        training_cfg,
        feat_cfg,
        default="question",
    )

    sae = SparseAutoencoder(
        d_model=model_cfg.get("d_model", 4096),
        n_features=config.get("sae", {}).get("n_features", 32768),
        l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
    )

    if args.activations_path:
        # Pre-collected activations: skip model and dataset loading entirely.
        layer_dir = os.path.join(args.activations_path, f"layer_{target_layer}")
        acts_file = os.path.join(layer_dir, "activations.pt")
        meta_file = os.path.join(layer_dir, "metadata.json")
        if not os.path.exists(acts_file):
            raise FileNotFoundError(f"Pre-collected activations not found: {acts_file}")
        print(f"[01_train_sae] Loading activations from {acts_file}")
        activations = torch.load(acts_file, weights_only=True, map_location="cpu")
        with open(meta_file) as _f:
            metadata = json.load(_f)
        if args.max_samples is not None and activations.shape[0] > args.max_samples:
            activations = activations[: args.max_samples]
            metadata = metadata[: args.max_samples]
            print(f"[01_train_sae] Subsampled to {activations.shape[0]:,} rows (--max_samples)")
        print(f"[01_train_sae] Loaded {activations.shape[0]:,} rows × {activations.shape[1]} dims")
        chunk_files = None
        trainer = SAETrainer(
            sae=sae,
            config=config,
            target_layer=target_layer,
            activation_site=model_cfg.get("activation_site", "residual"),
            llava_model=None,
        )
    else:
        tokenizer, model, image_processor = load_llava_components(model_cfg)
        dataset_format = data_cfg.get("format", "csv")
        if dataset_format == "clevr_lite":
            from sae_experiments.data.clevr_lite_dataset import CLEVRLiteVQADataset
            dataset = CLEVRLiteVQADataset(
                data_dir=data_cfg.get("data_dir", "datasets/clevr_lite"),
                split=data_cfg.get("split", "train"),
                tokenizer=tokenizer,
                image_processor=image_processor,
                model_config=model.config,
                conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
            )
        else:
            dataset = AttributeVQADataset(
                refined_dataset=data_cfg.get("refined_dataset", ""),
                image_folder=data_cfg.get("image_folder", ""),
                tokenizer=tokenizer,
                image_processor=image_processor,
                model_config=model.config,
                task_type=resolve_primary_task_type(data_cfg.get("task_types")),
                conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
            )
        trainer = SAETrainer(
            sae=sae,
            config=config,
            target_layer=target_layer,
            activation_site=model_cfg.get("activation_site", "residual"),
            llava_model=model,
        )
        act_cache_dir = os.path.join(experiment_dir, "_activation_cache")
        result = trainer.collect_activations(
            dataset,
            position_type=train_position_type,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            show_progress=args.show_progress,
            checkpoint_dir=act_cache_dir,
        )
        if isinstance(result[0], list):
            metadata, chunk_files = result
            activations = None
        else:
            activations, metadata = result
            chunk_files = None

    if not args.activations_path and torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("[01_train_sae] Released LLaVA model to free memory for training")

    if chunk_files is not None:
        from sae_experiments.data.activation_collector import reassemble_chunks
        print(f"[01_train_sae] Reassembling {len(chunk_files)} chunks after model release...")
        activations = reassemble_chunks(chunk_files, cleanup=True)
        print(f"[01_train_sae] Activations: {activations.shape[0]:,} rows × {activations.shape[1]} dims")

    holdout_enabled = bool(holdout_cfg.get("enabled", False))
    train_activations = activations
    test_activations: Optional[torch.Tensor] = None
    holdout_split: Optional[Dict[str, Any]] = None
    if holdout_enabled:
        split = _split_activation_rows_by_question(
            metadata=metadata,
            test_ratio=float(holdout_cfg.get("test_ratio", 0.2)),
            split_seed=int(holdout_cfg.get("split_seed", seed)),
            min_test_samples=int(holdout_cfg.get("min_test_samples", 50)),
        )
        train_activations = _take_rows(activations, split["train_indices"])
        test_activations = _take_rows(activations, split["test_indices"])
        holdout_split = {
            "enabled": True,
            "test_ratio": float(holdout_cfg.get("test_ratio", 0.2)),
            "split_seed": int(holdout_cfg.get("split_seed", seed)),
            "min_test_samples": int(holdout_cfg.get("min_test_samples", 50)),
            "total_questions": int(split["total_questions"]),
            "train_question_count": int(len(split["train_question_ids"])),
            "test_question_count": int(len(split["test_question_ids"])),
            "train_activation_rows": int(train_activations.shape[0]),
            "test_activation_rows": int(test_activations.shape[0]),
            "train_question_ids": split["train_question_ids"],
            "test_question_ids": split["test_question_ids"],
        }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    history = trainer.train(train_activations, show_progress=args.show_progress)
    activation_stats = compute_activation_stats(train_activations)
    eval_batch_size = int(holdout_cfg.get("eval_batch_size", 512))
    reconstruction_eval = {
        "train": _compute_reconstruction_metrics(trainer.sae, train_activations, batch_size=eval_batch_size),
    }
    if test_activations is not None:
        reconstruction_eval["test"] = _compute_reconstruction_metrics(
            trainer.sae,
            test_activations,
            batch_size=eval_batch_size,
        )
    recon_loss = float(reconstruction_eval["train"]["mse"])
    trainer.save_checkpoint(
        checkpoint_path,
        metadata={
            "history": history,
            "activation_stats": activation_stats,
            "reconstruction_loss": recon_loss,
            "activation_samples": int(train_activations.shape[0]),
            "activation_samples_total": int(activations.shape[0]),
            "seed": seed,
            "deterministic": bool(config.get("reproducibility", {}).get("deterministic", True)),
            "position_type": train_position_type,
            "holdout": holdout_split or {"enabled": False},
            "reconstruction_eval": reconstruction_eval,
        },
    )
    save_config(config, os.path.join(experiment_dir, "config.yaml"))

    if holdout_split is not None:
        holdout_split_path = os.path.join(experiment_dir, "holdout_split.json")
        with open(holdout_split_path, "w", encoding="utf-8") as handle:
            json.dump(holdout_split, handle, indent=2)

    reconstruction_eval_path = os.path.join(experiment_dir, "reconstruction_eval.json")
    with open(reconstruction_eval_path, "w", encoding="utf-8") as handle:
        json.dump(reconstruction_eval, handle, indent=2)

    print(f"Saved SAE checkpoint to {checkpoint_path}")
    print(f"Saved reconstruction evaluation to {reconstruction_eval_path}")
    if holdout_split is not None:
        print(
            "Holdout split:",
            f"train_questions={holdout_split['train_question_count']},",
            f"test_questions={holdout_split['test_question_count']},",
            f"train_rows={holdout_split['train_activation_rows']},",
            f"test_rows={holdout_split['test_activation_rows']}",
        )
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()

"""Identify discriminative SAE features."""

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

from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.feature_analysis.feature_identifier import FeatureIdentifier
from sae_experiments.feature_analysis.feature_visualizer import FeatureVisualizer
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder
from sae_experiments.utils.checkpoint_utils import resolve_experiment_dir
from sae_experiments.utils.config_utils import resolve_primary_task_type
from sae_experiments.utils.random_utils import resolve_seed, set_global_seed


def _resolve_dtype(value: str) -> torch.dtype:
    """Map a config dtype string to a torch dtype.

    Args:
        value (str): Text dtype token (for example ``float32`` or ``bf16``).

    Returns:
        torch.dtype: Resolved dtype used for SAE inference.

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
    """Identify discriminative SAE features and export catalogs/statistics.

    Args:
        None: Arguments are provided through CLI flags parsed in this function.

    Returns:
        None: Writes ``feature_catalog.json``, ``feature_stats.json``, and feature visualizations.

    Raises:
        FileNotFoundError: If required config or checkpoint files are missing.
        ValueError: If resolved feature settings are invalid.
        RuntimeError: If model execution or feature analysis fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    feat_cfg = config.get("feature_identification", {})
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
    checkpoint_path = args.sae_checkpoint or os.path.join(experiment_dir, "sae_checkpoint.pt")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sae.load_state_dict(ckpt.get("state", {}).get("sae_state", ckpt))
    train_cfg = config.get("training", {})
    sae.to(device=next(model.parameters()).device, dtype=_resolve_dtype(train_cfg.get("dtype", "float32")))
    sae.eval()

    identifier = FeatureIdentifier(
        sae,
        model,
        dataset,
        model_cfg.get("target_layer", 12),
        activation_site=model_cfg.get("activation_site", "residual"),
    )
    show_progress = not args.no_progress
    identifier.compute_feature_activations(
        position_type=feat_cfg.get("position_type", "attribute"),
        max_samples=args.max_samples,
        include_predictions=True,
        correctness_metric=feat_cfg.get("correctness_metric", "string_match"),
        logprob_normalize=feat_cfg.get("logprob_normalize", True),
        aggregation=feat_cfg.get("aggregation", "mean"),
        show_progress=show_progress,
    )

    top_k = feat_cfg.get("top_k", 50)
    selection_method = str(feat_cfg.get("selection_method", "ratio")).lower()
    discrimination_method = str(
        feat_cfg.get("discrimination_method", selection_method)
    ).strip().lower()
    if discrimination_method in ("causal_hybrid", ""):
        discrimination_method = "ratio"
    if discrimination_method in ("diff", "absolute_diff"):
        discrimination_method = "abs_diff"
    if discrimination_method == "abs_diff" and feat_cfg.get("discrimination_threshold") is not None:
        print(
            "[info] feature_identification.discrimination_threshold is unused in abs_diff mode."
        )
    top_features = []
    features = identifier.find_discriminative_features(
        threshold=feat_cfg.get("discrimination_threshold", 2.0),
        min_activation=feat_cfg.get("min_activation", 0.0),
        min_diff=feat_cfg.get("min_diff", 0.0),
        selection_method=discrimination_method,
    )
    if not features:
        fallback = feat_cfg.get("fallback", {})
        fallback_threshold = fallback.get("discrimination_threshold", 1.1)
        fallback_min_activation = fallback.get("min_activation", 0.0)
        fallback_min_diff = fallback.get("min_diff", 0.0)
        features = identifier.find_discriminative_features(
            threshold=fallback_threshold,
            min_activation=fallback_min_activation,
            min_diff=fallback_min_diff,
            selection_method=discrimination_method,
        )
        if features:
            print(
                "No features found with primary thresholds; using fallback thresholds:",
                f"threshold={fallback_threshold}, min_activation={fallback_min_activation}, min_diff={fallback_min_diff}",
            )

    if selection_method == "causal_hybrid":
        causal_scores_path = feat_cfg.get("causal_scores_path")
        if causal_scores_path and os.path.exists(causal_scores_path):
            with open(causal_scores_path, "r", encoding="utf-8") as handle:
                score_blob = json.load(handle)
            scores = score_blob.get("scores", score_blob)
            ranked = sorted(
                ((int(k), float(v)) for k, v in scores.items()),
                key=lambda x: x[1],
                reverse=True,
            )
            top_features = [idx for idx, _ in ranked[:top_k]]
        elif causal_scores_path:
            print(f"[warning] causal_scores_path not found: {causal_scores_path}. Falling back to activation ranking.")
    if not top_features:
        top_features = identifier.get_top_k_features(
            top_k,
            score_key=feat_cfg.get("score_key", "ratio"),
        )
    if not top_features:
        top_features = features[:top_k]

    catalog = FeatureCatalog()
    for feature_idx in top_features:
        stats = identifier.feature_stats.get(feature_idx, {})
        catalog.add_feature(
            feature_idx,
            {
                "name": f"feature_{feature_idx}",
                "type": "binding",
                "discrimination_score": stats.get("ratio", 0.0),
                "ratio": stats.get("ratio"),
                "diff": stats.get("diff"),
                "correct_mean": stats.get("correct_mean"),
                "incorrect_mean": stats.get("incorrect_mean"),
            },
        )

    catalog_path = os.path.join(experiment_dir, "feature_catalog.json")
    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
    catalog.export_to_json(catalog_path)
    identifier.save_feature_statistics(os.path.join(experiment_dir, "feature_stats.json"))
    identifier.save_feature_distribution_summary(
        os.path.join(experiment_dir, "feature_stats_summary.json")
    )

    visualizer = FeatureVisualizer(sae, model, dataset, identifier.feature_acts, identifier.metadata)
    viz_features = top_features[: min(10, len(top_features))]
    viz_iter = viz_features
    if show_progress and viz_features:
        from tqdm import tqdm

        viz_iter = tqdm(viz_features, desc="Visualizing features")
    for feature_idx in viz_iter:
        visualizer.visualize_feature(
            feature_idx,
            os.path.join(os.path.dirname(catalog_path), f"feature_{feature_idx}.png"),
        )

    print(f"Saved feature catalog to {catalog_path}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()

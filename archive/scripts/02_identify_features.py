"""Identify discriminative SAE features."""

import argparse
import json
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
    experiment_dir, seed = setup_experiment(args, config)
    tokenizer, model, image_processor = load_llava_components(model_cfg)

    dataset_format = data_cfg.get("format", "csv")
    if dataset_format == "clevr_lite":
        from sae_experiments.data.clevr_lite_dataset import CLEVRLiteVQADataset
        dataset = CLEVRLiteVQADataset(
            data_dir=data_cfg.get("data_dir", "datasets/clevr_lite"),
            split="val",
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

    checkpoint_path = args.sae_checkpoint or os.path.join(experiment_dir, "sae_checkpoint.pt")
    sae = load_sae(config, model, checkpoint_path)

    position_type = feat_cfg.get("position_type", "attribute")
    target_layer = model_cfg.get("target_layer", 12)
    activation_site = model_cfg.get("activation_site", "residual")
    top_k = feat_cfg.get("top_k", 50)
    selection_method = str(feat_cfg.get("selection_method", "ratio")).lower()
    print(f"[02] layer={target_layer}, site={activation_site}, position_type={position_type}, "
          f"top_k={top_k}, selection={selection_method}, "
          f"dataset={data_cfg.get('refined_dataset','')}, n_items={len(dataset.questions)}")

    identifier = FeatureIdentifier(
        sae,
        model,
        dataset,
        target_layer,
        activation_site=activation_site,
    )
    show_progress = not args.no_progress
    identifier.compute_feature_activations(
        position_type=position_type,
        max_samples=args.max_samples,
        include_predictions=True,
        correctness_metric=feat_cfg.get("correctness_metric", "string_match"),
        logprob_normalize=feat_cfg.get("logprob_normalize", True),
        aggregation=feat_cfg.get("aggregation", "mean"),
        show_progress=show_progress,
    )

    n_correct = sum(1 for m in identifier.metadata if m.get("is_correct", False))
    n_total = len(identifier.metadata)
    print(f"[02] activation collection done: {n_total} items, "
          f"{n_correct} correct ({100*n_correct/max(1,n_total):.1f}%), "
          f"{len(identifier.feature_acts)} feature activation rows")

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
    print(f"[02] primary threshold: {len(features)} features passed "
          f"(threshold={feat_cfg.get('discrimination_threshold', 2.0)}, method={discrimination_method})")
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
            print(f"[02] fallback threshold: {len(features)} features passed "
                  f"(threshold={fallback_threshold})")
        else:
            print("[02] WARNING: zero features found even with fallback thresholds — "
                  "catalog will be empty. Check position_type and SAE quality.")

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
        score_key = feat_cfg.get("score_key", "ratio")
        top_features = identifier.get_top_k_features(top_k, score_key=score_key)
    if not top_features:
        top_features = features[:top_k]

    if top_features:
        score_key = feat_cfg.get("score_key", "ratio")
        scores = [identifier.feature_stats.get(f, {}).get(score_key, 0.0) for f in top_features]
        print(f"[02] top-{len(top_features)} features by '{score_key}': "
              f"max={max(scores):.4f}, min={min(scores):.4f}, mean={sum(scores)/len(scores):.4f}")
    else:
        print("[02] WARNING: top_features is empty — no catalog will be written.")

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

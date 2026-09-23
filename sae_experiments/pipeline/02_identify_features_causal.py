"""Identify causally important SAE features via gradient attribution.

Uses the same config and checkpoint as 02_identify_features.py, but replaces
correlational feature selection (activation ratio) with causal selection
(gradient * activation through the model's computation graph).

Usage:
    python sae_experiments/pipeline/02_identify_features_causal.py \
        --config configs/clevr_lite/sae_layer11_attn_out_question.yaml \
        --max_samples 50  # smoke test
"""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.core.config import load_config
from sae_experiments.feature_analysis.causal_feature_identifier import (
    CausalFeatureIdentifier,
)
from sae_experiments.utils.config_utils import resolve_primary_task_type
from sae_experiments.utils.script_utils import setup_experiment, load_llava_components, load_sae


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal SAE feature identification")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--target", type=str, default="margin",
                        choices=["margin", "correct_logit"])
    parser.add_argument("--position_type", type=str, default=None,
                        help="Override position_type from config")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Override top_k from config")
    parser.add_argument("--output_suffix", type=str, default="causal",
                        help="Suffix for output directory")
    parser.add_argument("--no_progress", action="store_true")
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
        from sae_experiments.data.attribute_dataset import AttributeVQADataset
        dataset = AttributeVQADataset(
            refined_dataset=data_cfg.get("refined_dataset", ""),
            image_folder=data_cfg.get("image_folder", ""),
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_config=model.config,
            task_type=resolve_primary_task_type(data_cfg.get("task_types")),
            conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
        )

    checkpoint_path = args.sae_checkpoint or os.path.join(
        experiment_dir, "sae_checkpoint.pt"
    )
    sae = load_sae(config, model, checkpoint_path)

    target_layer = model_cfg.get("target_layer", 12)
    activation_site = model_cfg.get("activation_site", "residual")
    position_type = args.position_type or feat_cfg.get("position_type", "question")
    top_k = args.top_k or feat_cfg.get("top_k", 200)

    n_items = len(dataset.questions)
    effective_n = min(n_items, args.max_samples) if args.max_samples else n_items
    print(f"[02b] Causal feature identification")
    print(f"  layer={target_layer}, site={activation_site}, "
          f"position_type={position_type}, target={args.target}")
    print(f"  dataset={dataset_format}, n_items={n_items}, "
          f"processing={effective_n}")

    identifier = CausalFeatureIdentifier(
        sae, model, dataset, target_layer, activation_site=activation_site,
    )

    show_progress = not args.no_progress
    feature_stats = identifier.compute_causal_scores(
        position_type=position_type,
        max_samples=args.max_samples,
        target=args.target,
        show_progress=show_progress,
    )

    if not feature_stats:
        print("[02b] ERROR: No samples processed successfully.")
        return

    summary = identifier.summary
    print(f"\n[02b] Done: {summary['n_processed']} samples processed, "
          f"{summary['n_skipped']} skipped")

    top_features = identifier.get_top_k_features(top_k)
    if top_features:
        scores = [feature_stats[f]["causal_score"] for f in top_features]
        acts = [feature_stats[f]["activation_mean"] for f in top_features]
        grads = [feature_stats[f]["gradient_mean"] for f in top_features]
        print(f"\n[02b] Top-{len(top_features)} features by causal_score:")
        print(f"  causal: max={max(scores):.6f}, min={min(scores):.6f}, "
              f"mean={sum(scores)/len(scores):.6f}")
        print(f"  activation: max={max(acts):.6f}, min={min(acts):.6f}")
        print(f"  gradient: max={max(grads):.6f}, min={min(grads):.6f}")

        print(f"\n  {'Rank':>4} {'Feature':>8} {'Causal':>12} "
              f"{'Activation':>12} {'Gradient':>12}")
        print(f"  {'----':>4} {'-------':>8} {'------':>12} "
              f"{'----------':>12} {'--------':>12}")
        for rank, feat_idx in enumerate(top_features[:20], 1):
            s = feature_stats[feat_idx]
            print(f"  {rank:4d} {feat_idx:8d} {s['causal_score']:12.6f} "
                  f"{s['activation_mean']:12.6f} {s['gradient_mean']:12.6f}")

    output_dir = experiment_dir
    if args.output_suffix:
        output_dir = experiment_dir + f"_{args.output_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    identifier.save_results(output_dir)
    catalog_path = identifier.export_catalog(output_dir, top_k=top_k)

    print(f"\nSaved causal feature stats to {output_dir}/")
    print(f"Saved feature catalog to {catalog_path}")

    if summary:
        cs = summary.get("causal_score", {})
        am = summary.get("activation_mean", {})
        gm = summary.get("gradient_mean", {})
        print(f"\n[02b] Distribution summary (all {summary['n_features']} features):")
        print(f"  causal_score:    p50={cs.get('p50',0):.2e}, "
              f"p90={cs.get('p90',0):.2e}, p99={cs.get('p99',0):.2e}, "
              f"max={cs.get('max',0):.2e}")
        print(f"  activation_mean: p50={am.get('p50',0):.2e}, "
              f"p90={am.get('p90',0):.2e}, p99={am.get('p99',0):.2e}, "
              f"max={am.get('max',0):.2e}")
        print(f"  gradient_mean:   p50={gm.get('p50',0):.2e}, "
              f"p90={gm.get('p90',0):.2e}, p99={gm.get('p99',0):.2e}, "
              f"max={gm.get('max',0):.2e}")


if __name__ == "__main__":
    main()

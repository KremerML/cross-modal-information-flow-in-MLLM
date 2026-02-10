"""Rank SAE features by direct causal ablation effect."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.ablation.feature_ablator import FeatureAblator
from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder
from sae_experiments.utils.checkpoint_utils import resolve_experiment_dir
from sae_experiments.utils.config_utils import resolve_primary_task_type
from sae_experiments.utils.random_utils import resolve_seed, set_global_seed


def _resolve_dtype(value: str) -> torch.dtype:
    value = str(value).lower()
    if value in ("float16", "fp16", "half"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def _load_candidates(
    candidates_path: str | None,
    feature_stats_path: str,
    pool_size: int,
    score_key: str,
    n_features: int,
) -> List[int]:
    if candidates_path and os.path.exists(candidates_path):
        with open(candidates_path, "r", encoding="utf-8") as handle:
            blob = json.load(handle)
        if isinstance(blob, dict):
            if "features" in blob and isinstance(blob["features"], list):
                return [int(x) for x in blob["features"]]
            if "scores" in blob and isinstance(blob["scores"], dict):
                ranked = sorted(
                    ((int(k), float(v)) for k, v in blob["scores"].items()),
                    key=lambda item: item[1],
                    reverse=True,
                )
                return [idx for idx, _ in ranked[:pool_size]]
            ranked = sorted(
                ((int(k), float(v)) for k, v in blob.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            return [idx for idx, _ in ranked[:pool_size]]
        if isinstance(blob, list):
            return [int(x) for x in blob]

    if os.path.exists(feature_stats_path):
        with open(feature_stats_path, "r", encoding="utf-8") as handle:
            stats_blob = json.load(handle)
        parsed: Dict[int, dict] = {}
        for key, value in stats_blob.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                parsed[idx] = value
        ranked = sorted(
            parsed.items(),
            key=lambda item: float(item[1].get(score_key, item[1].get("ratio", 0.0))),
            reverse=True,
        )
        return [idx for idx, _ in ranked[:pool_size]]

    return list(range(min(pool_size, n_features)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--candidates", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--candidate_pool_k", type=int, default=None)
    parser.add_argument("--metric", type=str, default="mean_margin_drop")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    feat_cfg = config.get("feature_identification", {})
    ablation_cfg = config.get("ablation", {})
    eval_cfg = config.get("evaluation", {})
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
    sae.to(device=next(model.parameters()).device, dtype=_resolve_dtype(training_cfg.get("dtype", "float32")))
    sae.eval()

    candidate_pool_k = int(args.candidate_pool_k or feat_cfg.get("candidate_pool_k", 200))
    feature_stats_path = os.path.join(experiment_dir, "feature_stats.json")
    candidates = _load_candidates(
        args.candidates,
        feature_stats_path=feature_stats_path,
        pool_size=candidate_pool_k,
        score_key=feat_cfg.get("score_key", "ratio"),
        n_features=sae.n_features,
    )
    candidates = sorted({int(x) for x in candidates if 0 <= int(x) < sae.n_features})
    if not candidates:
        raise ValueError("No candidate features available for causal ranking.")

    ablator = FeatureAblator(
        model,
        sae,
        layer_idx=model_cfg.get("target_layer", 12),
        activation_site=model_cfg.get("activation_site", "residual"),
    )
    show_progress = not args.no_progress
    iterator = candidates
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(candidates, desc="Ranking causal features")

    position_type = ablation_cfg.get("position_type", "attribute")
    mode = ablation_cfg.get("mode", "residual")
    delta_scale = float(ablation_cfg.get("delta_scale", 1.0))
    operation = str(ablation_cfg.get("operation", "zero")).lower()
    operation_scale = float(ablation_cfg.get("operation_scale", 1.0))
    logprob_normalize = bool(eval_cfg.get("logprob_normalize", True))

    metric = str(args.metric)
    scores: Dict[int, float] = {}
    summaries: Dict[int, Dict] = {}
    for feature_idx in iterator:
        results = ablator.batch_ablation_experiment(
            dataset,
            [feature_idx],
            position_type=position_type,
            mode=mode,
            delta_scale=delta_scale,
            operation=operation,
            operation_scale=operation_scale,
            logprob_normalize=logprob_normalize,
            show_progress=False,
            max_samples=args.max_samples,
        )
        summary = ablator.compute_ablation_effect(results)
        score = summary.get(metric)
        if score is None:
            score = summary.get("accuracy_drop", 0.0)
        scores[feature_idx] = float(score)
        summaries[feature_idx] = summary

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    output = {
        "metric": metric,
        "candidate_count": len(candidates),
        "scores": {str(k): v for k, v in scores.items()},
        "ranked_features": [idx for idx, _ in ranked],
        "top_features": [{"feature_idx": idx, "score": score} for idx, score in ranked[:50]],
        "feature_summaries": {str(k): v for k, v in summaries.items()},
        "settings": {
            "position_type": position_type,
            "mode": mode,
            "delta_scale": delta_scale,
            "operation": operation,
            "operation_scale": operation_scale,
            "max_samples": args.max_samples,
        },
    }

    output_path = args.output or os.path.join(experiment_dir, "causal_feature_scores.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Saved causal feature ranking to {output_path}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()

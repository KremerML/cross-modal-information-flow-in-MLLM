"""Rank SAE features by direct causal ablation effect."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.ablation.feature_ablator import FeatureAblator
from sae_experiments.config.sae_config import load_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.utils.config_utils import resolve_primary_task_type
from sae_experiments.utils.script_utils import setup_experiment, load_llava_components, load_sae


def _load_candidates(
    candidates_path: str | None,
    feature_stats_path: str,
    pool_size: int,
    score_key: str,
    n_features: int,
) -> List[int]:
    """Load and rank candidate feature indices from multiple input formats.

    Args:
        candidates_path (str | None): Optional JSON path with explicit feature candidates or scores.
        feature_stats_path (str): Fallback path to ``feature_stats.json``.
        pool_size (int): Maximum number of candidates to keep.
        score_key (str): Statistic key used when ranking from feature stats.
        n_features (int): SAE latent dimensionality used for fallback range generation.

    Returns:
        List[int]: Candidate feature indices ordered by descending priority.

    Raises:
        OSError: If an input file exists but cannot be read.
        json.JSONDecodeError: If a candidate JSON file is malformed.
        ValueError: If numeric conversions fail for candidate indices/scores.
    """
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


def _rank_candidates(
    *,
    ablator: FeatureAblator,
    dataset: AttributeVQADataset,
    candidates: List[int],
    metric: str,
    position_type: str,
    mode: str,
    delta_scale: float,
    operation: str,
    operation_scale: float,
    logprob_normalize: bool,
    max_samples: Optional[int],
    score_options: bool,
    baseline_cache: Optional[Any],
    show_progress: bool,
    progress_desc: str,
) -> Dict[str, Any]:
    """Rank candidate features by direct intervention effect.

    Args:
        ablator (FeatureAblator): Ablation engine used to run feature interventions.
        dataset (AttributeVQADataset): Evaluation dataset used for per-feature scoring.
        candidates (List[int]): Feature indices to evaluate.
        metric (str): Summary metric key used for ranking (for example ``mean_margin_drop``).
        position_type (str): Token position subset for intervention.
        mode (str): SAE intervention mode (for example ``residual`` or ``replace``).
        delta_scale (float): Scale for residual-delta interventions.
        operation (str): Feature operation (for example ``zero``).
        operation_scale (float): Operation scaling factor.
        logprob_normalize (bool): Whether to length-normalize option log-probabilities.
        max_samples (Optional[int]): Optional cap on evaluated dataset samples.
        score_options (bool): Whether to compute option scores required for margin metrics.
        baseline_cache (Optional[Any]): Optional cached baseline outputs aligned to ``dataset``.
        show_progress (bool): Whether to show a progress bar over candidates.
        progress_desc (str): Progress-bar description label.

    Returns:
        Dict[str, Any]: Ranking payload containing raw scores, summaries, and sorted candidates.

    Raises:
        RuntimeError: If ablation execution fails for any candidate.
        ValueError: If a candidate index or requested metric is invalid.
    """
    scores: Dict[int, float] = {}
    summaries: Dict[int, Dict[str, Any]] = {}
    iterator = candidates
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(candidates, desc=progress_desc)

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
            max_samples=max_samples,
            baseline_cache=baseline_cache,
            score_options=score_options,
        )
        summary = ablator.compute_ablation_effect(results)
        score = summary.get(metric)
        if score is None:
            score = summary.get("accuracy_drop", 0.0)
        scores[feature_idx] = float(score)
        summaries[feature_idx] = summary

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "metric": metric,
        "candidate_count": len(candidates),
        "max_samples": max_samples,
        "score_options": score_options,
        "scores": scores,
        "summaries": summaries,
        "ranked": ranked,
    }


def _serialize_stage(stage_name: str, stage: Dict[str, Any], top_n: int = 50) -> Dict[str, Any]:
    """Convert an in-memory ranking stage payload to JSON-friendly output.

    Args:
        stage_name (str): Stage label (for example ``coarse`` or ``fine``).
        stage (Dict[str, Any]): Raw stage payload returned by ``_rank_candidates``.
        top_n (int): Number of top features to include in ``top_features``.

    Returns:
        Dict[str, Any]: Serialized stage payload with stringified JSON object keys.

    Raises:
        ValueError: If stage payload fields are inconsistent.
    """
    ranked = stage.get("ranked", [])
    scores = stage.get("scores", {})
    summaries = stage.get("summaries", {})
    return {
        "stage": stage_name,
        "metric": stage.get("metric"),
        "candidate_count": stage.get("candidate_count"),
        "max_samples": stage.get("max_samples"),
        "score_options": stage.get("score_options"),
        "scores": {str(k): v for k, v in scores.items()},
        "ranked_features": [idx for idx, _ in ranked],
        "top_features": [{"feature_idx": idx, "score": score} for idx, score in ranked[:top_n]],
        "feature_summaries": {str(k): v for k, v in summaries.items()},
    }


def main() -> None:
    """Run single-stage or two-stage causal ranking for SAE features.

    Args:
        None: CLI arguments are parsed in this function.

    Returns:
        None: Writes ranked feature scores and optional stage breakdowns to a JSON file.

    Raises:
        FileNotFoundError: If required config/checkpoint inputs are missing.
        ValueError: If no valid candidate features are available.
        RuntimeError: If model loading or ranking execution fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--candidates", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--candidate_pool_k", type=int, default=None)
    parser.add_argument("--metric", type=str, default="mean_margin_drop")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--skip_option_scores",
        action="store_true",
        help="Skip true/false option logprob scoring (faster, but margin metrics become unavailable).",
    )
    parser.add_argument(
        "--coarse_top_k",
        type=int,
        default=None,
        help="Enable two-stage ranking: coarse over all candidates then fine over top-k.",
    )
    parser.add_argument("--coarse_metric", type=str, default="accuracy_drop")
    parser.add_argument("--coarse_max_samples", type=int, default=None)
    parser.add_argument(
        "--coarse_skip_option_scores",
        action="store_true",
        help="Skip option scoring in coarse stage.",
    )
    parser.add_argument("--fine_metric", type=str, default=None)
    parser.add_argument("--fine_max_samples", type=int, default=None)
    parser.add_argument(
        "--fine_skip_option_scores",
        action="store_true",
        help="Skip option scoring in fine stage.",
    )
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
    experiment_dir, seed = setup_experiment(args, config)
    tokenizer, model, image_processor = load_llava_components(model_cfg)

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

    position_type = ablation_cfg.get("position_type", "attribute")
    mode = ablation_cfg.get("mode", "residual")
    delta_scale = float(ablation_cfg.get("delta_scale", 1.0))
    operation = str(ablation_cfg.get("operation", "zero")).lower()
    operation_scale = float(ablation_cfg.get("operation_scale", 1.0))
    logprob_normalize = bool(eval_cfg.get("logprob_normalize", True))

    baseline_cache_store: Dict[Tuple[Optional[int], bool], List[Dict[str, Any]]] = {}

    def _get_baseline_cache(
        max_samples: Optional[int],
        score_options: bool,
        stage_name: str,
    ) -> List[Dict[str, Any]]:
        """Return cached baseline outputs keyed by stage sampling/scoring settings.

        Args:
            max_samples (Optional[int]): Optional cap on baseline sample count.
            score_options (bool): Whether baseline rows include option-score fields.
            stage_name (str): Stage label used only for progress messaging.

        Returns:
            List[Dict[str, Any]]: Baseline rows aligned with dataset iteration order.

        Raises:
            RuntimeError: If baseline cache computation fails.
        """
        cache_key = (max_samples, score_options)
        if cache_key not in baseline_cache_store:
            baseline_cache_store[cache_key] = ablator.compute_baseline_cache(
                dataset,
                logprob_normalize=logprob_normalize,
                max_samples=max_samples,
                score_options=score_options,
                show_progress=show_progress,
                progress_desc=f"Baseline ({stage_name})",
            )
        return baseline_cache_store[cache_key]

    stage_payloads: Dict[str, Dict[str, Any]] = {}

    coarse_top_k = args.coarse_top_k
    use_two_stage = coarse_top_k is not None and int(coarse_top_k) > 0
    if use_two_stage:
        coarse_top_k = max(1, min(int(coarse_top_k), len(candidates)))
        coarse_metric = str(args.coarse_metric)
        coarse_max_samples = args.coarse_max_samples
        if coarse_max_samples is None:
            coarse_max_samples = args.max_samples
        coarse_score_options = not bool(args.coarse_skip_option_scores)
        coarse_baseline_cache = _get_baseline_cache(
            coarse_max_samples,
            coarse_score_options,
            stage_name="coarse",
        )
        coarse_stage = _rank_candidates(
            ablator=ablator,
            dataset=dataset,
            candidates=candidates,
            metric=coarse_metric,
            position_type=position_type,
            mode=mode,
            delta_scale=delta_scale,
            operation=operation,
            operation_scale=operation_scale,
            logprob_normalize=logprob_normalize,
            max_samples=coarse_max_samples,
            score_options=coarse_score_options,
            baseline_cache=coarse_baseline_cache,
            show_progress=show_progress,
            progress_desc="Ranking causal features (coarse)",
        )
        coarse_selected = [idx for idx, _ in coarse_stage["ranked"][:coarse_top_k]]
        coarse_serialized = _serialize_stage("coarse", coarse_stage)
        coarse_serialized["selected_for_next_stage"] = coarse_selected
        stage_payloads["coarse"] = coarse_serialized

        fine_metric = str(args.fine_metric or args.metric)
        fine_max_samples = args.fine_max_samples
        if fine_max_samples is None:
            fine_max_samples = args.max_samples
        fine_score_options = not bool(args.fine_skip_option_scores)
        fine_baseline_cache = _get_baseline_cache(
            fine_max_samples,
            fine_score_options,
            stage_name="fine",
        )
        final_stage = _rank_candidates(
            ablator=ablator,
            dataset=dataset,
            candidates=coarse_selected,
            metric=fine_metric,
            position_type=position_type,
            mode=mode,
            delta_scale=delta_scale,
            operation=operation,
            operation_scale=operation_scale,
            logprob_normalize=logprob_normalize,
            max_samples=fine_max_samples,
            score_options=fine_score_options,
            baseline_cache=fine_baseline_cache,
            show_progress=show_progress,
            progress_desc="Ranking causal features (fine)",
        )
        stage_payloads["fine"] = _serialize_stage("fine", final_stage)
    else:
        single_metric = str(args.metric)
        single_max_samples = args.max_samples
        single_score_options = not bool(args.skip_option_scores)
        single_baseline_cache = _get_baseline_cache(
            single_max_samples,
            single_score_options,
            stage_name="single",
        )
        final_stage = _rank_candidates(
            ablator=ablator,
            dataset=dataset,
            candidates=candidates,
            metric=single_metric,
            position_type=position_type,
            mode=mode,
            delta_scale=delta_scale,
            operation=operation,
            operation_scale=operation_scale,
            logprob_normalize=logprob_normalize,
            max_samples=single_max_samples,
            score_options=single_score_options,
            baseline_cache=single_baseline_cache,
            show_progress=show_progress,
            progress_desc="Ranking causal features",
        )

    ranked = final_stage["ranked"]
    scores = final_stage["scores"]
    summaries = final_stage["summaries"]
    output = {
        "metric": final_stage["metric"],
        "candidate_count": final_stage["candidate_count"],
        "initial_candidate_count": len(candidates),
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
            "max_samples": final_stage["max_samples"],
            "score_options": final_stage["score_options"],
        },
    }
    if stage_payloads:
        output["stages"] = stage_payloads

    output_path = args.output or os.path.join(experiment_dir, "causal_feature_scores.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Saved causal feature ranking to {output_path}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()

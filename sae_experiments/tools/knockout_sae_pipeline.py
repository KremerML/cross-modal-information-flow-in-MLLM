"""End-to-end knockout-aligned SAE pipeline."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.core.config import load_config, save_config
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.feature_analysis.feature_identifier import FeatureIdentifier
from sae_experiments.tools.knockout_runner import run_knockout_sweep
from sae_experiments.training.sae_trainer import SAETrainer
from sae_experiments.core.sparse_autoencoder import SparseAutoencoder
from sae_experiments.utils.checkpoint_utils import load_checkpoint, save_checkpoint
from sae_experiments.utils.config_utils import resolve_primary_task_type, resolve_dtype
from sae_experiments.hooks.knockout_utils import build_block_config, estimate_inputs_embeds_shape, resolve_flow_ranges
from sae_experiments.utils.script_utils import setup_experiment, load_llava_components
from sae_experiments.training.sae_validation import (
    compute_activation_stats,
    reconstruction_loss,
    should_reuse_sae,
)
from sae_experiments.ablation.feature_ablator import FeatureAblator


FLOW_POSITION_MAP = {
    "Image->Question": "question",
    "Image->Last": "last",
}


def _select_top_layers(summary: List[Dict], flow: str, top_k: int) -> List[int]:
    """Select the strongest knockout layers for a flow by effect size.

    Args:
        summary (List[Dict]): Knockout summary rows containing ``flow``, ``layer``, and ``effect_size``.
        flow (str): Flow label to filter (for example ``Image->Question``).
        top_k (int): Number of top layers to return.

    Returns:
        List[int]: Layer indices sorted by descending estimated effect.

    Raises:
        KeyError: If required summary fields are missing.
        TypeError: If summary rows are malformed.
    """
    rows = [row for row in summary if row.get("flow") == flow]
    rows = sorted(rows, key=lambda r: r.get("effect_size", 0.0), reverse=True)
    return [row["layer"] for row in rows[:top_k]]


def _ensure_sae_for_layer(
    model: Any,
    dataset: AttributeVQADataset,
    config: Dict[str, Any],
    layer: int,
    position_type: str,
    sae_dir: str,
    reuse_cfg: Dict[str, Any],
    train_max_samples: Optional[int] = None,
    skip_reuse: bool = False,
) -> Tuple[SparseAutoencoder, str, Dict[str, Any]]:
    """Load/reuse/train an SAE for a target layer and save its checkpoint.

    Args:
        model: Loaded LLaVA model used for activation collection.
        dataset: Dataset object with tokenizer/image processing context.
        config: Full experiment config mapping.
        layer (int): Transformer layer used for activation collection/intervention.
        position_type (str): Token position subset used during activation collection.
        sae_dir (str): Output directory for checkpoint persistence.
        reuse_cfg (Dict[str, Any]): Reuse policy and quality-threshold settings.
        train_max_samples (Optional[int]): Optional cap on SAE training activation samples.
        skip_reuse (bool): If ``True``, force retraining and ignore reuse candidates.

    Returns:
        Tuple[SparseAutoencoder, str, Dict[str, Any]]: Loaded/trained SAE, checkpoint path, and reuse metadata.

    Raises:
        ValueError: If checkpoint paths are invalid.
        FileNotFoundError: If configured reuse paths reference missing checkpoints.
        RuntimeError: If activation collection or SAE training fails.
    """
    os.makedirs(sae_dir, exist_ok=True)
    checkpoint_path = os.path.join(sae_dir, "sae_checkpoint.pt")
    model_cfg = config.get("model", {})
    sae_cfg = config.get("sae", {})
    train_cfg = config.get("training", {})

    sae = SparseAutoencoder(
        d_model=model_cfg.get("d_model", 4096),
        n_features=sae_cfg.get("n_features", 32768),
        l1_coeff=sae_cfg.get("l1_coeff", 1e-3),
    )

    trainer = SAETrainer(
        sae=sae,
        config=config,
        target_layer=layer,
        activation_site=model_cfg.get("activation_site", "residual"),
        llava_model=model,
    )

    reuse_info = {"reused": False}
    activations_cache = None
    if os.path.exists(checkpoint_path) and not os.path.isfile(checkpoint_path):
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
    if os.path.exists(checkpoint_path) and not skip_reuse:
        state, metadata = load_checkpoint(checkpoint_path)
        if state.get("target_layer") == layer and state.get(
            "activation_site", "residual"
        ) == model_cfg.get("activation_site", "residual"):
            sae.load_state_dict(state["sae_state"])
            activations_cache, _ = trainer.collect_activations(
                dataset,
                position_type=position_type,
                tokenizer=dataset.tokenizer,
                max_samples=reuse_cfg.get("sample_size", 256),
            )
            if activations_cache.numel() > 0:
                sae.to(device=activations_cache.device, dtype=activations_cache.dtype)
            ref_stats = metadata.get("activation_stats", {})
            if not ref_stats and not reuse_cfg.get("allow_missing_stats", True):
                reuse_info.update({"reason": "missing_activation_stats"})
            else:
                check = should_reuse_sae(
                    sae,
                    activations_cache,
                    ref_stats or compute_activation_stats(activations_cache),
                    recon_threshold=reuse_cfg.get("recon_threshold", 0.1),
                    kl_threshold=reuse_cfg.get("kl_threshold", 0.5),
                )
                reuse_info.update(check)
                if check.get("reuse"):
                    reuse_info["reused"] = True
                    return sae, checkpoint_path, reuse_info

    # Search for reusable checkpoints in other directories
    if not skip_reuse:
        search_paths = reuse_cfg.get("search_paths", [])
        for base in search_paths:
            if not os.path.exists(base):
                continue
            for root, _, files in os.walk(base):
                if "sae_checkpoint.pt" not in files:
                    continue
                candidate_path = os.path.join(root, "sae_checkpoint.pt")
                state, metadata = load_checkpoint(candidate_path)
                if state.get("target_layer") != layer:
                    continue
                if state.get("activation_site", "residual") != model_cfg.get(
                    "activation_site", "residual"
                ):
                    continue
                sae.load_state_dict(state["sae_state"])
                if activations_cache is None:
                    activations_cache, _ = trainer.collect_activations(
                        dataset,
                        position_type=position_type,
                        tokenizer=dataset.tokenizer,
                        max_samples=reuse_cfg.get("sample_size", 256),
                    )
                if activations_cache.numel() > 0:
                    sae.to(device=activations_cache.device, dtype=activations_cache.dtype)
                ref_stats = metadata.get("activation_stats", {})
                if not ref_stats and not reuse_cfg.get("allow_missing_stats", True):
                    continue
                check = should_reuse_sae(
                    sae,
                    activations_cache,
                    ref_stats or compute_activation_stats(activations_cache),
                    recon_threshold=reuse_cfg.get("recon_threshold", 0.1),
                    kl_threshold=reuse_cfg.get("kl_threshold", 0.5),
                )
                if check.get("reuse"):
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    save_checkpoint(state, checkpoint_path, metadata)
                    check["reused"] = True
                    return sae, checkpoint_path, check

    # Train new SAE
    activations, _ = trainer.collect_activations(
        dataset,
        position_type=position_type,
        tokenizer=dataset.tokenizer,
        max_samples=train_max_samples,
    )
    history = trainer.train(
        activations,
        show_progress=True,
        progress_desc=f"SAE training (layer {layer}, {position_type})",
    )
    activation_stats = compute_activation_stats(activations)
    recon_loss = reconstruction_loss(sae, activations)
    metadata = {
        "history": history,
        "activation_stats": activation_stats,
        "activation_samples": int(activations.shape[0]),
        "reconstruction_loss": recon_loss,
    }
    state = {
        "sae_state": sae.state_dict(),
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "target_layer": layer,
        "activation_site": model_cfg.get("activation_site", "residual"),
    }
    save_checkpoint(state, checkpoint_path, metadata)
    reuse_info.update({"reused": False, "trained": True})
    return sae, checkpoint_path, reuse_info


def _make_attn_block_resolver(
    flow: str,
    layer: int,
    window: int,
    model: Any,
) -> Callable[..., Optional[Dict[str, Any]]]:
    """Create a resolver that builds per-sample attention block configs.

    Args:
        flow (str): Knockout flow name used for source/target token ranges.
        layer (int): Center layer index for attention blocking.
        window (int): Layer window width passed to block config generation.
        model: Loaded LLaVA model instance.

    Returns:
        Callable[..., Optional[Dict[str, Any]]]: Resolver callable accepted by ``FeatureAblator``.

    Raises:
        RuntimeError: Raised indirectly if range/config resolution fails.
    """
    def resolver(
        input_ids: Any,
        image_tensor: Any,
        image_sizes: Any,
        dataset: AttributeVQADataset,
        line: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Resolve a knockout block specification for one dataset item.

        Args:
            input_ids: Token IDs for the sample prompt.
            image_tensor: Processed image tensor for the sample.
            image_sizes: Original image size metadata.
            dataset: Dataset object exposing ``dataset_dict`` and tokenizer.
            line: Sample row containing at least ``q_id``.

        Returns:
            Optional[Dict[str, Any]]: Blocking config or ``None`` if ranges cannot be resolved.

        Raises:
            KeyError: If required row fields are missing.
        """
        inputs_embeds_shape = estimate_inputs_embeds_shape(
            model, input_ids, image_tensor, image_sizes
        )
        if inputs_embeds_shape is None:
            return None
        question_text = dataset.dataset_dict[line["q_id"]].get("question", "")
        source_range, target_range = resolve_flow_ranges(
            flow,
            input_ids,
            inputs_embeds_shape,
            question_text,
            dataset.tokenizer,
        )
        if not source_range or not target_range:
            return None
        pairs = [(tgt, src) for src in source_range for tgt in target_range]
        return build_block_config(layer, model.config.num_hidden_layers, window, pairs)

    return resolver


def main() -> None:
    """Run an end-to-end knockout-aligned SAE discovery pipeline.

    Args:
        None: CLI arguments are parsed in this function.

    Returns:
        None: Writes knockout, SAE, and ablation artifacts under the experiment directory.

    Raises:
        FileNotFoundError: If required config or dataset files are missing.
        ValueError: If layer/flow selections are invalid.
        RuntimeError: If any pipeline stage fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--ablation_max_samples", type=int, default=None)
    parser.add_argument("--top_k_layers", type=int, default=None)
    parser.add_argument("--feature_max_samples", type=int, default=None)
    parser.add_argument("--force_knockout", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_ablation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    knockout_cfg = config.get("knockout", {})
    reuse_cfg = config.get("sae_reuse", {})
    feat_cfg = config.get("feature_identification", {})
    ablation_cfg = config.get("ablation", {})

    experiment_dir, _ = setup_experiment(args, config)

    knockout_dir = os.path.join(experiment_dir, "knockout")
    sae_root = os.path.join(experiment_dir, "sae")
    ablation_root = os.path.join(experiment_dir, "ablation")
    analysis_dir = os.path.join(experiment_dir, "analysis")
    os.makedirs(knockout_dir, exist_ok=True)
    os.makedirs(sae_root, exist_ok=True)
    os.makedirs(ablation_root, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    save_config(config, os.path.join(experiment_dir, "config.yaml"))

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

    # Knockout sweep
    knockout_results_path = os.path.join(knockout_dir, "knockout_results.json")
    knockout_summary_path = os.path.join(knockout_dir, "knockout_summary.json")
    if args.force_knockout or not os.path.exists(knockout_summary_path):
        data_loader = dataset.create_dataloader(
            batch_size=knockout_cfg.get("batch_size", 1),
            num_workers=knockout_cfg.get("num_workers", 2),
        )
        results, summaries = run_knockout_sweep(
            model=model,
            tokenizer=tokenizer,
            dataset_dict=dataset.dataset_dict,
            questions=dataset.questions,
            data_loader=data_loader,
            flows=knockout_cfg.get("flows", []),
            window=knockout_cfg.get("window", 1),
            max_samples=args.max_samples or knockout_cfg.get("max_samples"),
            filter_correct=knockout_cfg.get("filter_correct", True),
            normalize_logprob=knockout_cfg.get("normalize_logprob", True),
            progress_desc="Knockout sweep",
        )
        with open(knockout_results_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        with open(knockout_summary_path, "w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)
    else:
        with open(knockout_summary_path, "r", encoding="utf-8") as handle:
            summaries = json.load(handle)

    flows = knockout_cfg.get("flows", [])
    top_k = args.top_k_layers if args.top_k_layers is not None else knockout_cfg.get("top_k_layers", 5)

    layer_map: Dict[str, List[int]] = {}
    for flow in flows:
        layer_map[flow] = _select_top_layers(summaries, flow, top_k)

    selection_path = os.path.join(analysis_dir, "selected_layers.json")
    with open(selection_path, "w", encoding="utf-8") as handle:
        json.dump(layer_map, handle, indent=2)

    # Train/reuse SAE per selected layer
    sae_catalog: Dict[str, Dict] = {}
    for flow, layers in layer_map.items():
        position_type = FLOW_POSITION_MAP.get(flow, "question")
        for layer in layers:
            layer_dir = os.path.join(sae_root, flow.replace("->", "_"), f"layer_{layer}")
            if args.force_train:
                if os.path.exists(os.path.join(layer_dir, "sae_checkpoint.pt")):
                    os.remove(os.path.join(layer_dir, "sae_checkpoint.pt"))
            sae, checkpoint_path, reuse_info = _ensure_sae_for_layer(
                model,
                dataset,
                config,
                layer,
                position_type,
                layer_dir,
                reuse_cfg,
                train_max_samples=args.train_max_samples,
                skip_reuse=args.force_train,
            )

            identifier = FeatureIdentifier(
                sae,
                model,
                dataset,
                layer,
                activation_site=model_cfg.get("activation_site", "residual"),
            )
            identifier.compute_feature_activations(
                position_type=position_type,
                max_samples=args.feature_max_samples,
                batch_size=feat_cfg.get("batch_size"),
                include_predictions=True,
                correctness_metric=feat_cfg.get("correctness_metric", "option_logprob"),
                logprob_normalize=feat_cfg.get("logprob_normalize", True),
                aggregation=feat_cfg.get("aggregation", "mean"),
            )
            features = identifier.find_discriminative_features(
                threshold=feat_cfg.get("discrimination_threshold", 2.0),
                min_activation=feat_cfg.get("min_activation", 0.0),
                min_diff=feat_cfg.get("min_diff", 0.0),
            )
            if not features:
                fallback = feat_cfg.get("fallback", {})
                features = identifier.find_discriminative_features(
                    threshold=fallback.get("discrimination_threshold", 1.1),
                    min_activation=fallback.get("min_activation", 0.0),
                    min_diff=fallback.get("min_diff", 0.0),
                )
            top_features = identifier.get_top_k_features(
                feat_cfg.get("top_k", 50),
                score_key=feat_cfg.get("score_key", "ratio"),
            )
            if not top_features:
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
            catalog_path = os.path.join(layer_dir, "feature_catalog.json")
            catalog.export_to_json(catalog_path)
            identifier.save_feature_statistics(os.path.join(layer_dir, "feature_stats.json"))

            sae_catalog[f"{flow}/layer_{layer}"] = {
                "checkpoint": checkpoint_path,
                "feature_catalog": catalog_path,
                "reuse_info": reuse_info,
                "position_type": position_type,
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(os.path.join(analysis_dir, "sae_catalog.json"), "w", encoding="utf-8") as handle:
        json.dump(sae_catalog, handle, indent=2)

    # Ablation runs (SAE only, KO only, combined)
    ablation_results = {}
    for flow, layers in layer_map.items():
        position_type = FLOW_POSITION_MAP.get(flow, "question")
        for layer in layers:
            key = f"{flow}/layer_{layer}"
            sae_info = sae_catalog[key]
            sae_state, _ = load_checkpoint(sae_info["checkpoint"])
            sae = SparseAutoencoder(
                d_model=model_cfg.get("d_model", 4096),
                n_features=config.get("sae", {}).get("n_features", 32768),
                l1_coeff=config.get("sae", {}).get("l1_coeff", 1e-3),
            )
            sae.load_state_dict(sae_state["sae_state"])
            sae.to(
                device=next(model.parameters()).device,
                dtype=resolve_dtype(config.get("training", {}).get("dtype", "float32")),
            )
            sae.eval()

            with open(sae_info["feature_catalog"], "r", encoding="utf-8") as handle:
                catalog_data = json.load(handle)
            feature_indices = [int(idx) for idx in catalog_data.keys()]

            ablator = FeatureAblator(
                model,
                sae,
                layer,
                activation_site=model_cfg.get("activation_site", "residual"),
            )
            resolver = _make_attn_block_resolver(
                flow, layer, knockout_cfg.get("window", 1), model
            )

            result_dir = os.path.join(ablation_root, flow.replace("->", "_"), f"layer_{layer}")
            summary_path = os.path.join(result_dir, "summary.json")
            if os.path.exists(summary_path) and not args.force_ablation:
                ablation_results[key] = {
                    "sae_only": os.path.join(result_dir, "sae_only.json"),
                    "ko_only": os.path.join(result_dir, "ko_only.json"),
                    "combined": os.path.join(result_dir, "combined.json"),
                    "summary": summary_path,
                }
                continue

            sae_only = ablator.batch_ablation_experiment(
                dataset,
                feature_indices,
                position_type=position_type,
                mode=ablation_cfg.get("mode", "residual"),
                delta_scale=ablation_cfg.get("delta_scale", 1.0),
                operation=ablation_cfg.get("operation", "zero"),
                operation_scale=ablation_cfg.get("operation_scale", 1.0),
                logprob_normalize=config.get("evaluation", {}).get("logprob_normalize", True),
                show_progress=True,
                max_samples=args.ablation_max_samples,
            )
            sae_only_summary = ablator.compute_ablation_effect(sae_only)
            ko_only = ablator.batch_ablation_experiment(
                dataset,
                feature_indices,
                position_type=position_type,
                mode=ablation_cfg.get("mode", "residual"),
                delta_scale=ablation_cfg.get("delta_scale", 1.0),
                operation=ablation_cfg.get("operation", "zero"),
                operation_scale=ablation_cfg.get("operation_scale", 1.0),
                logprob_normalize=config.get("evaluation", {}).get("logprob_normalize", True),
                apply_sae=False,
                attn_block_resolver=resolver,
                show_progress=True,
                max_samples=args.ablation_max_samples,
            )
            ko_only_summary = ablator.compute_ablation_effect(ko_only)
            combined = ablator.batch_ablation_experiment(
                dataset,
                feature_indices,
                position_type=position_type,
                mode=ablation_cfg.get("mode", "residual"),
                delta_scale=ablation_cfg.get("delta_scale", 1.0),
                operation=ablation_cfg.get("operation", "zero"),
                operation_scale=ablation_cfg.get("operation_scale", 1.0),
                logprob_normalize=config.get("evaluation", {}).get("logprob_normalize", True),
                apply_sae=True,
                attn_block_resolver=resolver,
                show_progress=True,
                max_samples=args.ablation_max_samples,
            )
            combined_summary = ablator.compute_ablation_effect(combined)

            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, "sae_only.json"), "w", encoding="utf-8") as handle:
                json.dump(sae_only, handle, indent=2)
            with open(os.path.join(result_dir, "ko_only.json"), "w", encoding="utf-8") as handle:
                json.dump(ko_only, handle, indent=2)
            with open(os.path.join(result_dir, "combined.json"), "w", encoding="utf-8") as handle:
                json.dump(combined, handle, indent=2)
            with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "sae_only": sae_only_summary,
                        "ko_only": ko_only_summary,
                        "combined": combined_summary,
                    },
                    handle,
                    indent=2,
                )

            ablation_results[key] = {
                "sae_only": os.path.join(result_dir, "sae_only.json"),
                "ko_only": os.path.join(result_dir, "ko_only.json"),
                "combined": os.path.join(result_dir, "combined.json"),
                "summary": os.path.join(result_dir, "summary.json"),
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(os.path.join(analysis_dir, "ablation_index.json"), "w", encoding="utf-8") as handle:
        json.dump(ablation_results, handle, indent=2)

    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()

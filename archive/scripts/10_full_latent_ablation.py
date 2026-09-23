"""Run a quick full-latent SAE ablation experiment.

This script ablates *all* SAE latent features (for example, all 32768 features)
and evaluates model behavior under maximal latent corruption at the configured
layer and activation site.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
LLAVA_ROOT = ROOT / "LLaVA-NeXT"
for _path in (str(ROOT), str(LLAVA_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _ensure_llava_transformers_compat() -> None:
    """Patch Transformers helper symbol locations expected by LLaVA.

    Some LLaVA modules import helper utilities from ``transformers.modeling_utils``.
    In newer Transformers versions, these helpers moved to
    ``transformers.pytorch_utils``. This shim copies the attributes onto
    ``transformers.modeling_utils`` at runtime so this script can run without
    modifying the vendored LLaVA codebase.
    """
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "apply_chunking_to_forward"):
        return

    from transformers.pytorch_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )

    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    modeling_utils.prune_linear_layer = prune_linear_layer


def _parse_position_types(value: str) -> List[str]:
    """Parse a comma-separated position-type list.

    Args:
        value (str): Comma-separated position types (for example ``attribute,all``).

    Returns:
        List[str]: Deduplicated position types preserving order.

    Raises:
        ValueError: If no valid position types are provided.
    """
    out: List[str] = []
    seen = set()
    for raw in str(value).split(","):
        pos = raw.strip()
        if not pos or pos in seen:
            continue
        out.append(pos)
        seen.add(pos)
    if not out:
        raise ValueError("No position types provided.")
    return out


def main() -> None:
    """Run full-latent ablation and save summary JSON results.

    Args:
        None: Arguments are parsed from CLI.

    Returns:
        None: Writes ablation summaries to the output JSON path.

    Raises:
        FileNotFoundError: If config or checkpoint files are missing.
        ValueError: If no position types are configured or parsing fails.
        RuntimeError: If model loading or ablation execution fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--position_types", type=str, default="attribute,all")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--include_sample_results", action="store_true")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()

    _ensure_llava_transformers_compat()
    from sae_experiments.ablation.feature_ablator import FeatureAblator
    from sae_experiments.config.sae_config import load_config
    from sae_experiments.data.attribute_dataset import AttributeVQADataset
    from sae_experiments.utils.config_utils import resolve_primary_task_type
    from sae_experiments.utils.script_utils import setup_experiment, load_llava_components, load_sae

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
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

    ablator = FeatureAblator(
        model,
        sae,
        layer_idx=model_cfg.get("target_layer", 0),
        activation_site=model_cfg.get("activation_site", "residual"),
    )

    mode = ablation_cfg.get("mode", "replace")
    delta_scale = float(ablation_cfg.get("delta_scale", 1.0))
    operation = str(ablation_cfg.get("operation", "zero")).lower()
    operation_scale = float(ablation_cfg.get("operation_scale", 1.0))
    logprob_normalize = bool(eval_cfg.get("logprob_normalize", True))
    show_progress = not args.no_progress

    full_feature_set = list(range(int(sae.n_features)))
    position_types = _parse_position_types(args.position_types)

    payload_runs: Dict[str, Dict] = {}
    for position_type in position_types:
        results = ablator.batch_ablation_experiment(
            dataset,
            full_feature_set,
            position_type=position_type,
            mode=mode,
            delta_scale=delta_scale,
            operation=operation,
            operation_scale=operation_scale,
            logprob_normalize=logprob_normalize,
            show_progress=show_progress,
            max_samples=args.max_samples,
        )
        run_payload: Dict[str, object] = {
            "summary": ablator.compute_ablation_effect(results),
            "sample_count": len(results),
        }
        if args.include_sample_results:
            run_payload["sample_results"] = results
        payload_runs[position_type] = run_payload

    output = {
        "meta": {
            "experiment_dir": experiment_dir,
            "config_path": args.config,
            "checkpoint_path": checkpoint_path,
            "task_type": resolve_primary_task_type(data_cfg.get("task_types")),
            "dataset_path": data_cfg.get("refined_dataset", ""),
            "layer_idx": model_cfg.get("target_layer", 0),
            "activation_site": model_cfg.get("activation_site", "residual"),
            "n_features_ablated": int(sae.n_features),
            "mode": mode,
            "operation": operation,
            "operation_scale": operation_scale,
            "delta_scale": delta_scale,
            "logprob_normalize": logprob_normalize,
            "max_samples": args.max_samples,
            "position_types": position_types,
            "seed": seed,
        },
        "runs": payload_runs,
    }

    output_path = args.output or os.path.join(
        experiment_dir,
        "results",
        "full_latent_ablation_quick.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Saved full-latent ablation results to {output_path}")
    for position_type, run_payload in payload_runs.items():
        summary = run_payload.get("summary", {})
        print(
            f"[{position_type}] "
            f"accuracy_drop={summary.get('accuracy_drop')}, "
            f"mean_margin_drop={summary.get('mean_margin_drop')}, "
            f"mean_relative_perturbation={summary.get('mean_relative_perturbation')}"
        )


if __name__ == "__main__":
    main()

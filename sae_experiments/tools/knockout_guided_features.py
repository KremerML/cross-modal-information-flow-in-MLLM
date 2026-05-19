"""Knockout-guided SAE feature identification (Plan E).

For each sample, runs two forward passes:
  1. Normal pass — captures layer-N attn_out activations
  2. Layer-K knockout pass — same, but with attention from image tokens to question
     tokens blocked at layer K

Features whose SAE activations change most between the two passes at image token
positions are those that receive the Image→Question cross-modal signal at layer N.
These are selected as the "knockout-guided" binding features.

Usage:
    python sae_experiments/tools/knockout_guided_features.py \
        --config configs/sae_categories/sae_layer11_attn_out_v2/color.yaml \
        --knockout_layer 0
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.core.config import load_config
from sae_experiments.data.activation_collector import ActivationCollector
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.utils.config_utils import resolve_primary_task_type
from sae_experiments.hooks.hook_utils import HookManager, create_activation_capture_hook, get_target_module
from sae_experiments.hooks.knockout_utils import (
    estimate_inputs_embeds_shape,
    resolve_flow_ranges,
    build_block_config,
)
from sae_experiments.utils.script_utils import setup_experiment, load_llava_components, load_sae

from sae_experiments.hooks.attention_hooks import set_block_attn_hooks_llava, remove_wrapper_llava


def _encode_batched(sae, activations: torch.Tensor, batch_size: int = 2048, device=None) -> torch.Tensor:
    """Encode activations through SAE in CPU-resident mini-batches."""
    if device is None:
        device = next(sae.parameters()).device
    dtype = next(sae.parameters()).dtype
    activations = activations.to(dtype=dtype)
    feats_list = []
    starts = range(0, activations.shape[0], batch_size)
    with torch.no_grad():
        for start in starts:
            batch = activations[start : start + batch_size].to(device=device)
            feats_list.append(sae.encode(batch).cpu())
    return torch.cat(feats_list, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--output_catalog", type=str, default=None)
    parser.add_argument("--knockout_layer", type=int, default=0,
                        help="Layer at which to apply the Image->Question attention knockout.")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    feat_cfg = config.get("feature_identification", {})
    experiment_dir, seed = setup_experiment(args, config)

    tokenizer, model, image_processor = load_llava_components(model_cfg)
    device = next(model.parameters()).device

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

    collection_layer = model_cfg.get("target_layer", 11)
    knockout_layer = args.knockout_layer
    top_k = args.top_k or feat_cfg.get("top_k", 50)
    show_progress = not args.no_progress

    print(f"[07] collection_layer={collection_layer}, knockout_layer={knockout_layer}, "
          f"top_k={top_k}, n_items={len(dataset.questions)}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[07] GPU memory: {free/1024**3:.2f} GB free / {total/1024**3:.2f} GB total")

    target_module = get_target_module(model, collection_layer, model_cfg.get("activation_site", "attn_out"))
    _collector = ActivationCollector(model, collection_layer, activation_site=model_cfg.get("activation_site", "attn_out"))

    data_loader = dataset.create_dataloader()
    questions = dataset.questions

    position_type = feat_cfg.get("position_type", "image")
    print(f"[07] position_type={position_type}")

    # Per-sample: store feature-level difference at selected positions on CPU.
    # Shape after loop: [n_samples * n_positions, d_model] — aggregated to [n_sae_features].
    all_normal_acts = []
    all_ko_acts = []
    skipped = 0

    iterator = zip(data_loader, questions)
    total = len(questions)
    if args.max_samples is not None:
        total = min(total, args.max_samples)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=total, desc="Paired forward passes")

    storage = {}

    for idx, (batch, line) in enumerate(iterator):
        if args.max_samples is not None and idx >= args.max_samples:
            break

        input_ids, image_tensor, image_sizes, _, _ = batch
        input_ids = input_ids.to(device)
        image_tensor = [img.to(device) for img in image_tensor]

        inputs_embeds_shape = estimate_inputs_embeds_shape(model, input_ids, image_tensor, image_sizes)
        if inputs_embeds_shape is None:
            skipped += 1
            continue

        image_token_count = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        question_text = dataset.dataset_dict[line["q_id"]]["question"]
        positions = _collector._select_positions(
            position_type,
            input_ids,
            image_token_count,
            question_text,
            tokenizer,
            line,
        )
        if not positions:
            skipped += 1
            continue

        # ── Normal forward pass ──────────────────────────────────────────
        hook_mgr = HookManager(model)
        hook_mgr.register_forward_hook(target_module, create_activation_capture_hook(storage, "acts"))
        storage.pop("acts", None)
        with torch.no_grad():
            model(input_ids=input_ids, images=image_tensor, image_sizes=image_sizes, use_cache=False)
        hook_mgr.remove_hooks()

        acts = storage.pop("acts", None)
        acts = ActivationCollector._normalize_activation_tensor(acts)
        if acts is None:
            skipped += 1
            continue
        normal_img = acts[positions].cpu()

        # ── Knockout forward pass ────────────────────────────────────────
        source_range, target_range = resolve_flow_ranges(
            "Image->Question",
            input_ids,
            inputs_embeds_shape,
            question_text,
            tokenizer,
        )
        if not source_range or not target_range:
            skipped += 1
            continue

        src_tgt_pairs = [(tgt, src) for src in source_range for tgt in target_range]
        block_config = build_block_config(knockout_layer, model.config.num_hidden_layers, 1, src_tgt_pairs)

        hook_mgr2 = HookManager(model)
        hook_mgr2.register_forward_hook(target_module, create_activation_capture_hook(storage, "acts"))
        storage.pop("acts", None)
        ko_hooks = set_block_attn_hooks_llava(model, block_config)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, images=image_tensor, image_sizes=image_sizes, use_cache=False)
        finally:
            remove_wrapper_llava(model, ko_hooks)
            hook_mgr2.remove_hooks()

        acts_ko = storage.pop("acts", None)
        acts_ko = ActivationCollector._normalize_activation_tensor(acts_ko)
        if acts_ko is None:
            skipped += 1
            continue
        ko_img = acts_ko[positions].cpu()

        all_normal_acts.append(normal_img)
        all_ko_acts.append(ko_img)

    print(f"[07] Collected {len(all_normal_acts)} paired samples ({skipped} skipped)")
    if not all_normal_acts:
        print("[07] ERROR: no samples collected — check dataset and model.")
        return

    # Encode through SAE ─────────────────────────────────────────────────
    # Flatten to [total_img_tokens, d_model] for batched encoding.
    normal_flat = torch.cat(all_normal_acts, dim=0)   # [N_total, d_model]
    ko_flat = torch.cat(all_ko_acts, dim=0)            # [N_total, d_model]

    print(f"[07] Encoding {normal_flat.shape[0]:,} activation vectors through SAE ...")
    feats_normal = _encode_batched(sae, normal_flat)   # [N_total, n_sae_features]
    feats_ko = _encode_batched(sae, ko_flat)            # [N_total, n_sae_features]

    # Per-feature knockout effect score: mean absolute change across all positions/samples.
    diff = (feats_normal - feats_ko).abs()              # [N_total, n_sae_features]
    knockout_effect = diff.mean(dim=0).numpy()          # [n_sae_features]

    # Also compute the mean activation under normal conditions (diagnostic).
    normal_mean = feats_normal.mean(dim=0).numpy()

    print(f"[07] Knockout effect scores: "
          f"max={float(knockout_effect.max()):.5f}, "
          f"mean={float(knockout_effect.mean()):.5f}, "
          f"p95={float(sorted(knockout_effect)[-max(1, int(0.05 * len(knockout_effect)))]):.5f}")

    import numpy as np
    top_indices = np.argsort(knockout_effect)[::-1][:top_k].tolist()
    top_scores = [float(knockout_effect[i]) for i in top_indices]
    print(f"[07] Top-{len(top_indices)} features by knockout effect: "
          f"max={max(top_scores):.5f}, min={min(top_scores):.5f}, "
          f"mean={sum(top_scores)/len(top_scores):.5f}")

    # Save knockout effect scores for all features.
    ko_scores_path = os.path.join(experiment_dir, "knockout_feature_scores.json")
    with open(ko_scores_path, "w", encoding="utf-8") as handle:
        json.dump(
            {str(i): float(knockout_effect[i]) for i in range(len(knockout_effect))},
            handle,
            indent=2,
        )
    print(f"[07] Saved knockout effect scores to {ko_scores_path}")

    # Build feature catalog.
    catalog = FeatureCatalog()
    for rank, feature_idx in enumerate(top_indices):
        catalog.add_feature(
            feature_idx,
            {
                "name": f"feature_{feature_idx}",
                "type": "binding",
                "knockout_effect": float(knockout_effect[feature_idx]),
                "normal_mean_activation": float(normal_mean[feature_idx]),
                "rank": rank,
                "knockout_layer": knockout_layer,
                "collection_layer": collection_layer,
            },
        )

    catalog_path = args.output_catalog or os.path.join(experiment_dir, "knockout_feature_catalog.json")
    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
    catalog.export_to_json(catalog_path)
    print(f"[07] Saved knockout feature catalog ({len(top_indices)} features) to {catalog_path}")
    print(f"[07] Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()

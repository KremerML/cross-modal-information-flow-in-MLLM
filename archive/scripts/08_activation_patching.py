"""Activation patching experiment to address cross-layer redundancy.

The core idea: instead of zeroing SAE features (which the model compensates for
by re-reading image tokens at later layers), we *replace* the layer-N attn_out
activations at image positions with the values they would have had under a
layer-K attention knockout.  Layers N+1 onwards then receive the counterfactual
signal and process it normally — there is nothing to compensate for.

For each sample:
  1. Normal pass       → baseline margin + acts_normal at layer-N image positions
  2. Knockout pass     → acts_ko at layer-N image positions (layer-K ko active)
  3. Encode both through SAE: feats_normal, feats_ko
  4. Build patched features: feats_patched = feats_normal; patch binding features
     with their knockout values: feats_patched[:, binding_feats] = feats_ko[:, binding_feats]
  5. Reconstruct patched activations: patched_acts = sae.decode(feats_patched)
  6. Patched pass      → hook replaces layer-N output at image positions with
                         patched_acts → patched margin
  7. margin_drop = baseline_margin - patched_margin

Repeat steps 4-6 for random feature sets as null distribution.

Usage:
    python sae_experiments/scripts/08_activation_patching.py \
        --config configs/sae_categories/sae_layer11_attn_out_v2/color.yaml \
        --features output/.../knockout_feature_catalog.json \
        --n_random_sets 0   # quick diagnostic; remove for full run
"""

import argparse
import json
import os
import random
import statistics
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.ablation.statistical_analysis import paired_t_test, effect_size_cohens_d
from sae_experiments.config.sae_config import load_config
from sae_experiments.data.activation_collector import ActivationCollector
from sae_experiments.data.attribute_dataset import AttributeVQADataset
from sae_experiments.feature_analysis.feature_catalog import FeatureCatalog
from sae_experiments.utils.config_utils import resolve_primary_task_type
from sae_experiments.utils.hook_utils import HookManager, create_activation_capture_hook, get_target_module
from sae_experiments.utils.knockout_utils import (
    estimate_inputs_embeds_shape,
    get_image_token_range,
    resolve_flow_ranges,
    build_block_config,
    sequence_logprob,
)
from sae_experiments.utils.script_utils import setup_experiment, load_llava_components, load_sae

try:
    from methods import set_block_attn_hooks_llava, remove_wrapper_llava
except ImportError as exc:
    raise ImportError(
        "Cannot import set_block_attn_hooks_llava from methods.py. "
        "Run from the repo root so that methods.py is on sys.path."
    ) from exc


# ── Hook helpers ─────────────────────────────────────────────────────────────

def create_replacement_hook(patched_acts: torch.Tensor, image_positions: List[int]):
    """Return a forward hook that replaces layer output at image_positions.

    patched_acts: [n_img_tokens, d_model] CPU tensor — the pre-computed
                  reconstruction to inject at the given positions.
    """
    def hook(module, inputs, output):
        out = ActivationCollector._normalize_activation_tensor(output)
        if out is None:
            return output
        result = out.clone()
        result[image_positions] = patched_acts.to(result.device, dtype=result.dtype)
        if isinstance(output, tuple):
            return (result,) + output[1:]
        if isinstance(output, list):
            new_out = list(output)
            new_out[0] = result
            return new_out
        return result
    return hook


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _encode(sae, acts: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    """CPU-resident SAE encoding: acts on CPU, move mini-batches to SAE device."""
    device = next(sae.parameters()).device
    dtype = next(sae.parameters()).dtype
    acts = acts.to(dtype=dtype)
    out = []
    with torch.no_grad():
        for start in range(0, acts.shape[0], batch_size):
            out.append(sae.encode(acts[start : start + batch_size].to(device)).cpu())
    return torch.cat(out, dim=0)


def _decode(sae, feats: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    """CPU-resident SAE decoding."""
    device = next(sae.parameters()).device
    dtype = next(sae.parameters()).dtype
    feats = feats.to(dtype=dtype)
    out = []
    with torch.no_grad():
        for start in range(0, feats.shape[0], batch_size):
            out.append(sae.decode(feats[start : start + batch_size].to(device)).cpu())
    return torch.cat(out, dim=0)


# ── Per-condition patching logic ──────────────────────────────────────────────

def _run_patched_pass(
    model,
    sae,
    target_module,
    feats_normal: torch.Tensor,
    feats_ko: torch.Tensor,
    feature_indices: List[int],
    image_positions: List[int],
    input_ids,
    image_tensor,
    image_sizes,
    tokenizer,
    logprob_normalize: bool,
    true_option: str,
    false_option: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Patch selected features with their knockout values and score the options."""
    feats_patched = feats_normal.clone()
    feats_patched[:, feature_indices] = feats_ko[:, feature_indices]
    patched_acts = _decode(sae, feats_patched)  # [n_img, d_model] on CPU

    hook_mgr = HookManager(model)
    hook_mgr.register_forward_hook(
        target_module,
        create_replacement_hook(patched_acts, image_positions),
    )
    try:
        true_lp = sequence_logprob(
            model, tokenizer, input_ids, image_tensor, image_sizes,
            true_option, normalize=logprob_normalize,
        )
        false_lp = sequence_logprob(
            model, tokenizer, input_ids, image_tensor, image_sizes,
            false_option, normalize=logprob_normalize,
        )
    finally:
        hook_mgr.remove_hooks()

    return true_lp, false_lp


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--features", type=str, default=None,
                        help="Path to feature catalog JSON (knockout_feature_catalog.json).")
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--knockout_layer", type=int, default=0,
                        help="Layer at which to apply Image->Question knockout for the ko pass.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Use only the top-k features from the catalog (default: all).")
    parser.add_argument("--n_random_sets", type=int, default=None,
                        help="Number of random control sets (default: from config; 0 = diagnostic only).")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    eval_cfg = config.get("evaluation", {})
    ablation_cfg = config.get("ablation", {})
    random_cfg = config.get("random_control", {})
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
    sae.eval()

    collection_layer = model_cfg.get("target_layer", 11)
    activation_site = model_cfg.get("activation_site", "attn_out")
    knockout_layer = args.knockout_layer
    logprob_normalize = bool(eval_cfg.get("logprob_normalize", True))
    model_name = model_cfg.get("name", "")

    n_random_sets = args.n_random_sets
    if n_random_sets is None:
        n_random_sets = int(ablation_cfg.get("n_random_sets", random_cfg.get("n_random_sets", 15)))
    rng_seed = int(random_cfg.get("seed", config.get("reproducibility", {}).get("seed", 42)))
    rng = random.Random(rng_seed)

    # Load binding features from catalog.
    catalog = FeatureCatalog()
    features_path = args.features or os.path.join(experiment_dir, "knockout_feature_catalog.json")
    catalog.load_from_json(features_path)
    binding_features = sorted(catalog.features.keys())
    if args.top_k is not None:
        binding_features = binding_features[: args.top_k]
    print(f"[08] collection_layer={collection_layer}, site={activation_site}, "
          f"knockout_layer={knockout_layer}, n_binding={len(binding_features)}, "
          f"n_random_sets={n_random_sets}, n_items={len(dataset.questions)}")
    if torch.cuda.is_available():
        free, total_mem = torch.cuda.mem_get_info()
        print(f"[08] GPU memory: {free/1024**3:.2f} GB free / {total_mem/1024**3:.2f} GB total")

    target_module = get_target_module(model, collection_layer, activation_site)
    storage: Dict[str, torch.Tensor] = {}

    data_loader = dataset.create_dataloader()
    questions = dataset.questions
    show_progress = not args.no_progress

    # Per-sample records: {baseline_margin, binding_margin, random_margins[]}
    baseline_margins: List[float] = []
    binding_margins: List[float] = []  # margin under binding patch
    random_margins_per_set: List[List[float]] = [[] for _ in range(n_random_sets)]
    skipped = 0

    total = min(len(questions), args.max_samples) if args.max_samples else len(questions)
    iterator = zip(data_loader, questions)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=total, desc="Activation patching")

    ko_diff_norms: List[float] = []  # sanity check: mean |normal - ko| at image positions

    for idx, (batch, line) in enumerate(iterator):
        if args.max_samples is not None and idx >= args.max_samples:
            break

        input_ids, image_tensor, image_sizes, _, _ = batch
        input_ids = input_ids.to(device)
        image_tensor = [img.to(device) for img in image_tensor]

        detail = dataset.dataset_dict[line["q_id"]]
        true_option = detail.get("true option", "").strip()
        false_option = detail.get("false option", "").strip()
        if not true_option or not false_option:
            skipped += 1
            continue

        inputs_embeds_shape = estimate_inputs_embeds_shape(model, input_ids, image_tensor, image_sizes)
        if inputs_embeds_shape is None:
            skipped += 1
            continue

        image_positions = get_image_token_range(input_ids, inputs_embeds_shape)
        if not image_positions:
            skipped += 1
            continue

        # ── Pass 1: Normal ────────────────────────────────────────────────
        hook_mgr = HookManager(model)
        hook_mgr.register_forward_hook(target_module, create_activation_capture_hook(storage, "acts"))
        storage.pop("acts", None)
        base_true_lp = sequence_logprob(
            model, tokenizer, input_ids, image_tensor, image_sizes,
            true_option, normalize=logprob_normalize,
        )
        hook_mgr.remove_hooks()

        # Capture acts_normal from the last forward pass inside sequence_logprob.
        # sequence_logprob calls model() internally — the hook fires during that call.
        # Re-register and run a dedicated capture pass instead.
        hook_mgr = HookManager(model)
        hook_mgr.register_forward_hook(target_module, create_activation_capture_hook(storage, "acts"))
        storage.pop("acts", None)
        with torch.no_grad():
            model(input_ids=input_ids, images=image_tensor, image_sizes=image_sizes, use_cache=False)
        hook_mgr.remove_hooks()

        acts_raw = storage.pop("acts", None)
        acts_normal_full = ActivationCollector._normalize_activation_tensor(acts_raw)
        if acts_normal_full is None:
            skipped += 1
            continue

        base_false_lp = sequence_logprob(
            model, tokenizer, input_ids, image_tensor, image_sizes,
            false_option, normalize=logprob_normalize,
        )
        if base_true_lp is None or base_false_lp is None:
            skipped += 1
            continue

        base_margin = base_true_lp - base_false_lp
        acts_normal = acts_normal_full[image_positions].cpu()  # [n_img, d_model]

        # ── Pass 2: Knockout ──────────────────────────────────────────────
        question_text = detail.get("question", "")
        source_range, target_range = resolve_flow_ranges(
            "Image->Question", input_ids, inputs_embeds_shape,
            question_text, tokenizer, model_name,
        )
        if not source_range or not target_range:
            skipped += 1
            continue

        src_tgt_pairs = [(tgt, src) for src in source_range for tgt in target_range]
        block_config = build_block_config(
            knockout_layer, model.config.num_hidden_layers, 1, src_tgt_pairs
        )

        hook_mgr = HookManager(model)
        hook_mgr.register_forward_hook(target_module, create_activation_capture_hook(storage, "acts"))
        storage.pop("acts", None)
        ko_hooks = set_block_attn_hooks_llava(model, block_config)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, images=image_tensor, image_sizes=image_sizes, use_cache=False)
        finally:
            remove_wrapper_llava(model, ko_hooks)
            hook_mgr.remove_hooks()

        acts_ko_raw = storage.pop("acts", None)
        acts_ko_full = ActivationCollector._normalize_activation_tensor(acts_ko_raw)
        if acts_ko_full is None:
            skipped += 1
            continue

        acts_ko = acts_ko_full[image_positions].cpu()  # [n_img, d_model]

        # Sanity: record knockout effect norm.
        ko_diff_norms.append(float((acts_normal - acts_ko).abs().mean()))

        # ── Encode both ───────────────────────────────────────────────────
        feats_normal = _encode(sae, acts_normal)  # [n_img, n_sae]
        feats_ko = _encode(sae, acts_ko)           # [n_img, n_sae]

        # ── Pass 3a: Binding patch ────────────────────────────────────────
        b_true_lp, b_false_lp = _run_patched_pass(
            model, sae, target_module,
            feats_normal, feats_ko, binding_features, image_positions,
            input_ids, image_tensor, image_sizes,
            tokenizer, logprob_normalize, true_option, false_option,
        )
        if b_true_lp is None or b_false_lp is None:
            skipped += 1
            continue

        binding_margin = b_true_lp - b_false_lp
        baseline_margins.append(base_margin)
        binding_margins.append(binding_margin)

        # ── Pass 3b: Random patches ───────────────────────────────────────
        n_total_features = int(sae.n_features)
        binding_set = set(binding_features)
        pool = [i for i in range(n_total_features) if i not in binding_set]
        n_rand = len(binding_features)

        for set_idx in range(n_random_sets):
            rand_features = rng.sample(pool, k=min(n_rand, len(pool)))
            r_true_lp, r_false_lp = _run_patched_pass(
                model, sae, target_module,
                feats_normal, feats_ko, rand_features, image_positions,
                input_ids, image_tensor, image_sizes,
                tokenizer, logprob_normalize, true_option, false_option,
            )
            if r_true_lp is not None and r_false_lp is not None:
                random_margins_per_set[set_idx].append(r_true_lp - r_false_lp)
            else:
                random_margins_per_set[set_idx].append(base_margin)

    print(f"[08] Processed {len(baseline_margins)} samples ({skipped} skipped)")
    if ko_diff_norms:
        print(f"[08] Knockout effect (mean|acts_normal - acts_ko| at image pos): "
              f"{statistics.fmean(ko_diff_norms):.5f}")

    if not baseline_margins:
        print("[08] ERROR: no samples collected.")
        return

    # ── Aggregate results ─────────────────────────────────────────────────────
    def _margin_drop_stats(base_list: List[float], patch_list: List[float]) -> Dict:
        drops = [b - p for b, p in zip(base_list, patch_list)]
        n = len(drops)
        mean_drop = statistics.fmean(drops)
        acc_base = sum(1 for m in base_list if m > 0) / max(1, n)
        acc_patch = sum(1 for m in patch_list if m > 0) / max(1, n)
        t_stat, p_val = paired_t_test(base_list, patch_list)
        d = effect_size_cohens_d(base_list, patch_list)
        return {
            "mean_margin_drop": mean_drop,
            "accuracy_drop": acc_base - acc_patch,
            "baseline_accuracy": acc_base,
            "patched_accuracy": acc_patch,
            "t_stat": t_stat,
            "p_value": p_val,
            "cohens_d": d,
            "n_samples": n,
        }

    binding_stats = _margin_drop_stats(baseline_margins, binding_margins)

    random_set_stats = []
    for set_margins in random_margins_per_set:
        if len(set_margins) == len(baseline_margins):
            random_set_stats.append(_margin_drop_stats(baseline_margins, set_margins))

    if random_set_stats:
        rand_drops = [s["mean_margin_drop"] for s in random_set_stats]
        random_summary = {
            "mean_margin_drop": statistics.fmean(rand_drops),
            "mean_margin_drop_std": statistics.pstdev(rand_drops) if len(rand_drops) > 1 else 0.0,
            "n_sets": len(random_set_stats),
        }
        # Significance: binding margin drop vs. random distribution
        rand_drops_arr = [s["mean_margin_drop"] for s in random_set_stats]
        binding_drop = binding_stats["mean_margin_drop"]
        empirical_p = sum(1 for r in rand_drops_arr if r >= binding_drop) / max(1, len(rand_drops_arr))
        significance = {
            "binding_vs_random_empirical_p": empirical_p,
            "binding_margin_drop": binding_drop,
            "random_mean_margin_drop": random_summary["mean_margin_drop"],
        }
    else:
        random_summary = {}
        significance = {}

    results = {
        "baseline": {
            "baseline_accuracy": binding_stats["baseline_accuracy"],
            "mean_baseline_margin": statistics.fmean(baseline_margins),
        },
        "binding": binding_stats,
        "random": random_summary,
        "random_set_stats": random_set_stats,
        "significance": significance,
        "patching_settings": {
            "knockout_layer": knockout_layer,
            "collection_layer": collection_layer,
            "activation_site": activation_site,
            "n_features_patched": len(binding_features),
            "n_samples": len(baseline_margins),
            "features_path": features_path,
        },
        "meta": {
            "seed": seed,
            "logprob_normalize": logprob_normalize,
        },
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    b = results["binding"]
    r = results["random"]
    sig = results["significance"]
    print(f"[08] ── Results summary ──────────────────────────────────────────")
    print(f"[08]   Baseline accuracy  : {results['baseline']['baseline_accuracy']:.3f}")
    print(f"[08]   Binding  │ margin_drop={b['mean_margin_drop']:+.4f}  acc_drop={b['accuracy_drop']:+.4f}  d={b['cohens_d']:.3f}")
    if r:
        print(f"[08]   Random   │ margin_drop={r['mean_margin_drop']:+.4f}  (±{r['mean_margin_drop_std']:.4f}, n={r['n_sets']})")
        print(f"[08]   Significance: empirical p={sig.get('binding_vs_random_empirical_p', 'N/A')}")
    print(f"[08]   Reference: L11 knockout=0.170, full-latent ceiling=0.169")
    print(f"[08] ────────────────────────────────────────────────────────────")

    output_path = args.output or os.path.join(experiment_dir, "results", "patching_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[08] Saved patching results to {output_path}")


if __name__ == "__main__":
    main()

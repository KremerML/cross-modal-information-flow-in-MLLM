"""Shared setup helpers for experiment scripts."""

import os

import torch

from sae_experiments.utils.checkpoint_utils import resolve_experiment_dir
from sae_experiments.utils.config_utils import resolve_dtype
from sae_experiments.utils.random_utils import resolve_seed, set_global_seed


def setup_experiment(args, config):
    """Resolve seed, set global seed, create experiment dir. Return (experiment_dir, seed)."""
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
    if getattr(args, "experiment_name", None):
        experiment_cfg["name"] = args.experiment_name
        experiment_cfg.pop("output_dir", None)
    experiment_dir = resolve_experiment_dir(
        experiment_cfg, getattr(args, "experiment_dir", None)
    )
    return experiment_dir, seed


def load_paligemma_components(model_cfg):
    """Load PaliGemma 2 processor and model. Return (processor, model).

    The returned ``processor`` acts as both tokenizer and image processor.
    Use ``processor.tokenizer`` to access the tokenizer directly.
    """
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    model_path = os.path.expanduser(model_cfg.get("name", ""))
    processor = AutoProcessor.from_pretrained(model_path)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=resolve_dtype(model_cfg.get("dtype", "bfloat16")),
    )
    model.eval()
    return processor, model


def load_gemma_scope_sae(config, model):
    """Load a Gemma Scope JumpReLU SAE from HuggingFace. Return sae.

    Config keys used (under ``sae``):
        gemma_scope_repo : HuggingFace repo id (default: ``google/gemma-scope-2b-pt-att``)
        target_layer     : transformer layer index (required)
        width            : feature width string (default: ``"16k"``)
        dtype            : float dtype string (default: ``"float32"``)
    """
    from sae_experiments.models.gemma_scope_sae import GemmaScopeJumpReLUSAE

    sae_cfg = config.get("sae", {})
    repo_id = sae_cfg.get("gemma_scope_repo", "google/gemma-scope-2b-pt-att")
    layer_idx = sae_cfg.get("target_layer")
    if layer_idx is None:
        raise ValueError("sae.target_layer must be set in config to load a Gemma Scope SAE")
    width = sae_cfg.get("width", "16k")
    dtype_str = sae_cfg.get("dtype", config.get("training", {}).get("dtype", "float32"))
    dtype = resolve_dtype(dtype_str)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    sae = GemmaScopeJumpReLUSAE.from_hf(
        layer_idx=int(layer_idx),
        width=width,
        repo_id=repo_id,
        device=str(device),
        dtype=dtype,
    )
    return sae

"""Utilities for registering model hooks."""

from typing import Callable, Dict

import torch


class HookManager:
    """Simple forward-hook manager with context support."""

    def __init__(self, model):
        self.model = model
        self.hooks = []

    def register_forward_hook(self, layer, hook_fn: Callable):
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle

    def register_forward_pre_hook(self, layer, hook_fn: Callable):
        handle = layer.register_forward_pre_hook(hook_fn)
        self.hooks.append(handle)
        return handle

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove_hooks()


def create_activation_capture_hook(storage_dict: Dict, key: str):
    def hook(module, inputs, output):
        storage_dict[key] = output
    return hook


def create_intervention_hook(
    sae,
    feature_indices,
    ablate: bool = True,
    positions=None,
    mode: str = "residual",
    delta_scale: float = 1.0,
):
    def hook(module, inputs, output):
        acts = output[0] if isinstance(output, (tuple, list)) else output
        sae_param = next(sae.parameters())
        acts_dtype = acts.dtype
        acts_device = acts.device
        acts_for_sae = acts.to(device=sae_param.device, dtype=sae_param.dtype)
        feats_full = sae.encode(acts_for_sae)
        feats_mod = feats_full.clone()
        if ablate:
            feats_mod[:, feature_indices] = 0.0
        if mode == "replace":
            recon_mod = sae.decode(feats_mod, target_shape=acts.shape)
            out = recon_mod.to(device=acts_device, dtype=acts_dtype)
            if positions:
                original = acts
                out_full = original.clone()
                out_full[:, positions, :] = out[:, positions, :]
                out = out_full
        else:
            recon_full = sae.decode(feats_full, target_shape=acts.shape)
            recon_mod = sae.decode(feats_mod, target_shape=acts.shape)
            delta = (recon_mod - recon_full).to(device=acts_device, dtype=acts_dtype)
            if delta_scale != 1.0:
                delta = delta * delta_scale
            if positions:
                mask = torch.zeros_like(acts, dtype=delta.dtype, device=acts_device)
                mask[:, positions, :] = 1.0
                delta = delta * mask
            out = acts + delta
        if isinstance(output, tuple):
            return (out,) + output[1:]
        if isinstance(output, list):
            new_output = list(output)
            new_output[0] = out
            return new_output
        return out
    return hook

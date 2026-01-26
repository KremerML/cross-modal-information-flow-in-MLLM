"""Helpers for validating SAE reuse across activation distributions."""

from typing import Dict, Tuple

import numpy as np
import torch


def compute_activation_stats(activations: torch.Tensor, bins: int = 100) -> Dict[str, object]:
    if activations.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "hist_counts": [], "hist_bins": []}
    values = activations.detach().float().view(-1).cpu().numpy()
    mean = float(values.mean())
    std = float(values.std())
    hist_counts, hist_bins = np.histogram(values, bins=bins, density=True)
    return {
        "mean": mean,
        "std": std,
        "hist_counts": hist_counts.tolist(),
        "hist_bins": hist_bins.tolist(),
    }


def kl_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    p = p_counts.astype(np.float64) + 1e-12
    q = q_counts.astype(np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compare_activation_stats(reference: Dict[str, object], current: Dict[str, object]) -> Tuple[float, float]:
    ref_counts = np.array(reference.get("hist_counts", []), dtype=np.float64)
    cur_counts = np.array(current.get("hist_counts", []), dtype=np.float64)
    if ref_counts.size == 0 or cur_counts.size == 0:
        return 0.0, 0.0
    kl = kl_divergence(ref_counts, cur_counts)
    mean_delta = abs(float(reference.get("mean", 0.0)) - float(current.get("mean", 0.0)))
    return kl, mean_delta


def reconstruction_loss(sae, activations: torch.Tensor) -> float:
    if activations.numel() == 0:
        return 0.0
    sae_param = next(sae.parameters())
    activations = activations.to(device=sae_param.device, dtype=sae_param.dtype)
    sae.eval()
    with torch.no_grad():
        recon, _ = sae.forward(activations)
        loss = torch.mean((recon - activations) ** 2)
    return float(loss.item())


def should_reuse_sae(
    sae,
    activations: torch.Tensor,
    reference_stats: Dict[str, object],
    recon_threshold: float,
    kl_threshold: float,
) -> Dict[str, object]:
    current_stats = compute_activation_stats(activations)
    kl, mean_delta = compare_activation_stats(reference_stats, current_stats)
    recon = reconstruction_loss(sae, activations)
    return {
        "reuse": (recon <= recon_threshold) and (kl <= kl_threshold),
        "recon_loss": recon,
        "kl_divergence": kl,
        "mean_delta": mean_delta,
        "current_stats": current_stats,
    }

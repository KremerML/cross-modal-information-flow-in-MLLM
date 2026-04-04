"""Gemma Scope JumpReLU Sparse Autoencoder — loads pre-trained weights from HuggingFace."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GemmaScopeJumpReLUSAE(nn.Module):
    """SAE with JumpReLU activation, weight-compatible with Gemma Scope checkpoints.

    Forward pass:
        x_cent = x - b_dec
        pre    = x_cent @ W_enc + b_enc      # (*, n_features)
        z      = relu(pre) * (pre > threshold)  # JumpReLU gate
        x_hat  = z @ W_dec + b_dec

    Weight convention (Gemma Scope):
        W_enc : (d_model, n_features)
        b_enc : (n_features,)
        W_dec : (n_features, d_model)
        b_dec : (d_model,)   — centering bias; also added back in decode
        threshold : (n_features,)
    """

    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # Raw parameters — shapes match Gemma Scope convention
        self.W_enc = nn.Parameter(torch.zeros(d_model, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.zeros(n_features, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.threshold = nn.Parameter(torch.zeros(n_features))

    # ------------------------------------------------------------------
    # Core interface (matches SparseAutoencoder.encode / .decode)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return JumpReLU feature activations for input x."""
        x_flat, _ = self._flatten(x)
        x_cent = x_flat - self.b_dec
        pre = x_cent @ self.W_enc + self.b_enc
        z = F.relu(pre) * (pre > self.threshold).to(pre.dtype)
        return z

    def decode(self, z: torch.Tensor, target_shape=None) -> torch.Tensor:
        """Reconstruct from feature activations z."""
        recon = z @ self.W_dec + self.b_dec
        if target_shape is not None:
            recon = recon.view(target_shape)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat, original_shape = self._flatten(x)
        z = self.encode(x_flat)
        recon = self.decode(z).view(original_shape)
        return recon, z

    # ------------------------------------------------------------------
    # HuggingFace loader
    # ------------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        layer_idx: int,
        width: str = "16k",
        repo_id: str = "google/gemma-scope-2b-pt-att",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "GemmaScopeJumpReLUSAE":
        """Download and load a Gemma Scope SAE from HuggingFace.

        Args:
            layer_idx: Transformer layer index (0-based).
            width: Feature width string, e.g. ``"16k"`` or ``"131k"``.
            repo_id: HuggingFace repo containing the Gemma Scope SAEs.
            device: Target device for the loaded tensors.
            dtype: Target dtype for the loaded tensors.

        Returns:
            Loaded and eval-mode ``GemmaScopeJumpReLUSAE``.
        """
        from huggingface_hub import list_repo_files, hf_hub_download

        # Find the params.npz file under layer_X/width_Y/average_l0_*/
        prefix = f"layer_{layer_idx}/width_{width}/"
        candidates = [
            f for f in list_repo_files(repo_id)
            if f.startswith(prefix) and f.endswith("params.npz")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No params.npz found under '{prefix}' in '{repo_id}'. "
                f"Available files starting with this prefix: "
                f"{[f for f in list_repo_files(repo_id) if f.startswith(prefix)]}"
            )
        # Pick the first match (typically one per layer/width combination)
        params_path = hf_hub_download(repo_id=repo_id, filename=candidates[0])

        data = np.load(params_path)
        # Verify expected keys are present
        required_keys = {"W_enc", "b_enc", "W_dec", "b_dec", "threshold"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(
                f"params.npz is missing expected keys: {missing}. "
                f"Found keys: {list(data.keys())}"
            )

        d_model, n_features = data["W_enc"].shape
        sae = cls(d_model=d_model, n_features=n_features)

        with torch.no_grad():
            sae.W_enc.copy_(torch.from_numpy(data["W_enc"].astype(np.float32)))
            sae.b_enc.copy_(torch.from_numpy(data["b_enc"].astype(np.float32)))
            sae.W_dec.copy_(torch.from_numpy(data["W_dec"].astype(np.float32)))
            sae.b_dec.copy_(torch.from_numpy(data["b_dec"].astype(np.float32)))
            sae.threshold.copy_(torch.from_numpy(data["threshold"].astype(np.float32)))

        sae.to(device=device, dtype=dtype)
        sae.eval()
        return sae

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        if x.dim() <= 2:
            return x, x.shape
        original_shape = x.shape
        return x.view(-1, original_shape[-1]), original_shape

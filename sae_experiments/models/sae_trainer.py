"""Training utilities for sparse autoencoders."""

from typing import Optional, Tuple
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sae_experiments.data.activation_collector import ActivationCollector
from sae_experiments.utils.checkpoint_utils import save_checkpoint, load_checkpoint


class SAETrainer:
    """Trainer for SAE models with activation collection support."""

    def __init__(
        self,
        sae: nn.Module,
        config,
        target_layer: int,
        llava_model: Optional[nn.Module] = None,
    ):
        self.sae = sae
        self.config = config
        self.target_layer = target_layer
        self.llava_model = llava_model

    def collect_activations(
        self,
        dataset,
        position_type: str = "question",
        tokenizer=None,
        max_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, list]:
        if self.llava_model is None:
            raise ValueError("llava_model is required for activation collection")
        collector = ActivationCollector(self.llava_model, self.target_layer)
        activations, metadata = collector.collect_from_dataset(
            dataset,
            position_type=position_type,
            tokenizer=tokenizer,
            max_samples=max_samples,
        )
        return activations, metadata

    def train(
        self,
        activations: torch.Tensor,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        device: Optional[str] = None,
        show_progress: bool = False,
        progress_desc: str = "SAE training",
    ) -> dict:
        train_cfg = self.config.get("training", {})
        batch_size = self._coerce_int(batch_size or train_cfg.get("batch_size", 32))
        learning_rate = self._coerce_float(learning_rate or train_cfg.get("learning_rate", 1e-4))
        epochs = self._coerce_int(epochs or train_cfg.get("epochs", 10))

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self._resolve_dtype(train_cfg.get("dtype", "float32"))
        self.sae.to(device=device, dtype=dtype)
        activations = activations.to(device=device, dtype=dtype)

        dataset = TensorDataset(activations)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=learning_rate)

        history = {
            "loss": [],
            "recon_loss": [],
            "l1_loss": [],
        }

        self.sae.train()
        epoch_iter = range(epochs)
        if show_progress:
            try:
                from tqdm import tqdm

                epoch_iter = tqdm(epoch_iter, desc=progress_desc)
            except ImportError:
                epoch_iter = range(epochs)

        for _ in epoch_iter:
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_l1 = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                total, recon_loss, l1_loss = self.sae.get_loss(batch)
                total.backward()
                optimizer.step()

                epoch_loss += total.item()
                epoch_recon += recon_loss.item()
                epoch_l1 += l1_loss.item()

            denom = max(1, len(loader))
            history["loss"].append(epoch_loss / denom)
            history["recon_loss"].append(epoch_recon / denom)
            history["l1_loss"].append(epoch_l1 / denom)

        self.sae.eval()
        return history

    @staticmethod
    def _coerce_float(value) -> float:
        if isinstance(value, str):
            return float(value)
        return float(value)

    @staticmethod
    def _coerce_int(value) -> int:
        if isinstance(value, str):
            return int(float(value))
        return int(value)

    @staticmethod
    def _resolve_dtype(value) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            value = value.lower()
            if value in ("float16", "fp16", "half"):
                return torch.float16
            if value in ("bfloat16", "bf16"):
                return torch.bfloat16
            return torch.float32
        return torch.float32

    def save_checkpoint(self, path: str, metadata: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "sae_state": self.sae.state_dict(),
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config,
            "target_layer": self.target_layer,
        }
        save_checkpoint(state, path, metadata)

    def load_checkpoint(self, path: str) -> dict:
        state, metadata = load_checkpoint(path)
        self.sae.load_state_dict(state["sae_state"])
        return {"state": state, "metadata": metadata}

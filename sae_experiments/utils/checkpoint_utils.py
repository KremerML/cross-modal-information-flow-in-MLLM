"""Checkpoint helpers for SAE experiments."""

from typing import Any, Dict, Tuple
import json
import os
import time

import torch


def save_checkpoint(state_dict: Dict[str, Any], path: str, metadata: Dict[str, Any] = None) -> None:
    payload = {
        "state": state_dict,
        "metadata": metadata or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    return payload.get("state", {}), payload.get("metadata", {})


def save_experiment_state(sae, feature_catalog, results: Dict[str, Any], path: str) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(sae.state_dict(), os.path.join(path, "sae_state.pt"))
    feature_catalog.export_to_json(os.path.join(path, "feature_catalog.json"))
    with open(os.path.join(path, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def create_experiment_directory(base_path: str, experiment_name: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_path, f"{experiment_name}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path

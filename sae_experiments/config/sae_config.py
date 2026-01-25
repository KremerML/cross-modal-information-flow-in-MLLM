"""Configuration helpers for SAE experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict
import copy
import os

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "llava-hf/llava-1.5-7b-hf",
        "target_layer": 12,
        "d_model": 4096,
        "conv_mode": "vicuna_v1",
        "model_base": None,
    },
    "sae": {
        "n_features": 32768,
        "l1_coeff": 0.001,
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
        "seed": 42,
        "dtype": "float32",
    },
    "experiment": {
        "output_dir": "output/sae_experiments/exp_default",
        "output_base": "output/sae_experiments",
        "name": "experiment",
        "use_timestamp": False,
    },
    "dataset": {
        "task_types": ["ChooseAttr"],
        "split": "validation",
        "refined_dataset": "datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv",
        "image_folder": "datasets/images",
    },
    "feature_identification": {
        "discrimination_threshold": 2.0,
        "min_activation": 0.1,
        "top_k": 50,
        "min_diff": 0.0,
        "position_type": "attribute",
        "correctness_metric": "option_logprob",
        "logprob_normalize": True,
        "fallback": {
            "discrimination_threshold": 1.1,
            "min_activation": 0.0,
            "min_diff": 0.0,
        },
    },
    "ablation": {
        "n_random_features": 50,
        "n_bootstrap": 1000,
        "position_type": "attribute",
        "mode": "residual",
        "delta_scale": 1.0,
    },
    "evaluation": {
        "significance_level": 0.05,
        "primary_metric": "pred_token_prob",
    },
    "paths": {
        "output_dir": "output/sae_experiments",
        "sae_checkpoint": "output/sae_experiments/sae_checkpoint.pt",
        "feature_catalog": "output/sae_experiments/feature_catalog.json",
        "results_dir": "output/sae_experiments/results",
    },
}


@dataclass
class Config:
    """Dataclass wrapper around the YAML configuration."""

    data: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str) -> Config:
    """Load YAML config and merge with defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            updates = yaml.safe_load(handle) or {}
        cfg = _deep_update(cfg, updates)
    return Config(cfg)


def save_config(config: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

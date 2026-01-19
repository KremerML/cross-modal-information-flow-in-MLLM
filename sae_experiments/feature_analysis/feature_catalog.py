"""Structured catalog for SAE features."""

from typing import Dict, List
import json


class FeatureCatalog:
    """Stores metadata for identified features."""

    def __init__(self):
        self.features: Dict[int, Dict] = {}

    def add_feature(self, feature_idx: int, metadata: Dict) -> None:
        self.features[feature_idx] = metadata

    def categorize_features(self) -> Dict[str, List[int]]:
        categories: Dict[str, List[int]] = {}
        for idx, meta in self.features.items():
            category = meta.get("type", "unknown")
            categories.setdefault(category, []).append(idx)
        return categories

    def export_to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.features, handle, indent=2)

    def load_from_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.features = {int(k): v for k, v in data.items()}

    def get_features_by_type(self, attr_type: str) -> Dict[int, Dict]:
        return {idx: meta for idx, meta in self.features.items() if meta.get("type") == attr_type}

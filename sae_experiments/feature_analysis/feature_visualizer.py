"""Visualization helpers for SAE feature interpretations."""

from typing import List, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sae_experiments.utils.visualization_utils import display_image_with_bbox, generate_html_report


class FeatureVisualizer:
    """Visualizes activations and top examples for features."""

    def __init__(self, sae, model, dataset, feature_acts=None, metadata=None):
        self.sae = sae
        self.model = model
        self.dataset = dataset
        self.feature_acts = feature_acts
        self.metadata = metadata

    def get_max_activating_examples(self, feature_idx: int, n: int = 10) -> List[Tuple[float, dict]]:
        if self.feature_acts is None or self.metadata is None:
            return []
        acts = self.feature_acts[:, feature_idx]
        top_idx = np.argsort(acts)[::-1][:n]
        return [(float(acts[i]), self.metadata[i]) for i in top_idx]

    def visualize_feature(self, feature_idx: int, save_path: str, n: int = 6) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        examples = self.get_max_activating_examples(feature_idx, n=n)
        if not examples:
            return

        cols = min(3, len(examples))
        rows = int(np.ceil(len(examples) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_1d(axes).reshape(rows, cols)

        for ax, (activation, meta) in zip(axes.flatten(), examples):
            detail = self.dataset.dataset_dict.get(meta["question_id"], {})
            img_id = str(detail.get("img_id", ""))
            image_path = os.path.join(self.dataset.image_folder, f"{img_id}.jpg")
            try:
                image = Image.open(image_path).convert("RGB")
                bboxes = self.dataset.get_item_with_metadata_by_id(
                    meta["question_id"]
                ).get("bboxes", None)
                if bboxes:
                    image = display_image_with_bbox(image, bboxes)
                ax.imshow(image)
            except Exception:
                ax.axis("off")
            question = meta.get("question", "")
            ax.set_title(f"{activation:.3f}\n{question}", fontsize=8)
            ax.axis("off")

        for ax in axes.flatten()[len(examples) :]:
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def create_feature_dashboard(self, feature_list: List[int], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        items = []
        for feature_idx in feature_list:
            image_path = os.path.join(output_dir, f"feature_{feature_idx}.png")
            self.visualize_feature(feature_idx, image_path)
            items.append({"feature": feature_idx, "image": os.path.basename(image_path)})
        template = os.path.join(output_dir, "template.html")
        generate_html_report(items, template, os.path.join(output_dir, "index.html"))

    def plot_activation_distribution(self, feature_idx: int, save_path: str) -> None:
        if self.feature_acts is None:
            return
        acts = self.feature_acts[:, feature_idx]
        plt.figure(figsize=(6, 4))
        plt.hist(acts, bins=50)
        plt.title(f"Feature {feature_idx} activation distribution")
        plt.xlabel("Activation")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

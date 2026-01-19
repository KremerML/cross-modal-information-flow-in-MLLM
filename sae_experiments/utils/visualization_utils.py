"""Visualization utilities for SAE experiments."""

from typing import Any, Dict, List
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def plot_feature_activation_heatmap(activations: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.imshow(activations, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Feature activation heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def display_image_with_bbox(image: Image.Image, bboxes: List[tuple], color: str = "red") -> Image.Image:
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, width=2)
    return image


def create_comparison_plot(baseline: Dict[str, float], ablated: Dict[str, float], random: Dict[str, float], save_path: str) -> None:
    labels = ["baseline", "binding", "random"]
    values = [
        baseline.get("baseline_accuracy", 0.0),
        ablated.get("ablated_accuracy", 0.0),
        random.get("ablated_accuracy", 0.0),
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.title("Ablation comparison")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_html_report(feature_data: List[Dict[str, Any]], template_path: str, output_path: str) -> None:
    if not os.path.exists(template_path):
        template = """
        <html>
        <head><title>SAE Feature Dashboard</title></head>
        <body>
        <h1>Feature Dashboard</h1>
        {items}
        </body>
        </html>
        """
    else:
        with open(template_path, "r", encoding="utf-8") as handle:
            template = handle.read()

    items_html = "\n".join(
        f"<div><h3>Feature {item['feature']}</h3><img src='{item['image']}' /></div>"
        for item in feature_data
    )
    html = template.replace("{items}", items_html)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def plot_layer_wise_effects(layer_results: Dict[int, float], save_path: str) -> None:
    layers = sorted(layer_results.keys())
    values = [layer_results[layer] for layer in layers]
    plt.figure(figsize=(6, 4))
    plt.plot(layers, values, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Effect")
    plt.title("Layer-wise effects")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

"""Analyze ablation results and generate reports."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sae_experiments.ablation import statistical_analysis
from sae_experiments.config.sae_config import load_config
from sae_experiments.evaluation.hypothesis_tester import HypothesisTester


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.results, "r", encoding="utf-8") as handle:
        results = json.load(handle)

    tester = HypothesisTester(config)
    hypothesis = tester.test_causal_necessity(results)
    report = statistical_analysis.generate_statistical_report(results)
    report.update(hypothesis)

    output_dir = args.output or os.path.join(config.get("paths", {}).get("results_dir", "output/sae_experiments/results"), "analysis")
    os.makedirs(output_dir, exist_ok=True)

    statistical_analysis.plot_ablation_comparison(
        results,
        os.path.join(output_dir, "ablation_comparison.png"),
    )

    with open(os.path.join(output_dir, "hypothesis_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved analysis report to {output_dir}")


if __name__ == "__main__":
    main()

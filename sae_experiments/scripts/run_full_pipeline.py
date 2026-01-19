"""Run the full SAE experiment pipeline."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--feature_catalog", type=str, default=None)
    parser.add_argument("--results", type=str, default=None)
    args = parser.parse_args()

    python = sys.executable

    subprocess.check_call([python, "sae_experiments/scripts/01_train_sae.py", "--config", args.config])

    sae_checkpoint = args.sae_checkpoint or "output/sae_experiments/sae_checkpoint.pt"
    subprocess.check_call(
        [python, "sae_experiments/scripts/02_identify_features.py", "--config", args.config, "--sae_checkpoint", sae_checkpoint]
    )

    feature_catalog = args.feature_catalog or "output/sae_experiments/feature_catalog.json"
    subprocess.check_call(
        [python, "sae_experiments/scripts/03_run_ablation.py", "--config", args.config, "--features", feature_catalog, "--sae_checkpoint", sae_checkpoint]
    )

    results_path = args.results or "output/sae_experiments/results/ablation_results.json"
    subprocess.check_call(
        [python, "sae_experiments/scripts/04_analyze_results.py", "--config", args.config, "--results", results_path]
    )

    subprocess.check_call(
        [python, "sae_experiments/scripts/05_visualize_features.py", "--config", args.config, "--catalog", feature_catalog, "--sae_checkpoint", sae_checkpoint]
    )


if __name__ == "__main__":
    main()

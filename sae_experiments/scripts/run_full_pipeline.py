"""Run the full SAE experiment pipeline."""

import argparse
import os
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
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    experiment_dir = args.experiment_dir
    if args.experiment_name and not experiment_dir:
        experiment_dir = f"output/sae_experiments/{args.experiment_name}"

    common = []
    if experiment_dir:
        common.extend(["--experiment_dir", experiment_dir])
    if args.experiment_name:
        common.extend(["--experiment_name", args.experiment_name])

    if not args.skip_train:
        subprocess.check_call(
            [python, "sae_experiments/scripts/01_train_sae.py", "--config", args.config] + common
        )

    sae_checkpoint = args.sae_checkpoint or (
        f"{experiment_dir}/sae_checkpoint.pt"
        if experiment_dir
        else "output/sae_experiments/sae_checkpoint.pt"
    )
    if args.skip_train and not os.path.exists(sae_checkpoint):
        raise FileNotFoundError(
            f"SAE checkpoint not found at {sae_checkpoint}. Provide --sae_checkpoint or run training."
        )
    subprocess.check_call(
        [python, "sae_experiments/scripts/02_identify_features.py", "--config", args.config, "--sae_checkpoint", sae_checkpoint] + common
    )

    feature_catalog = args.feature_catalog or (f"{experiment_dir}/feature_catalog.json" if experiment_dir else "output/sae_experiments/feature_catalog.json")
    subprocess.check_call(
        [python, "sae_experiments/scripts/03_run_ablation.py", "--config", args.config, "--features", feature_catalog, "--sae_checkpoint", sae_checkpoint] + common
    )

    results_path = args.results or (f"{experiment_dir}/results/ablation_results.json" if experiment_dir else "output/sae_experiments/results/ablation_results.json")
    subprocess.check_call(
        [python, "sae_experiments/scripts/04_analyze_results.py", "--config", args.config, "--results", results_path] + common
    )

    subprocess.check_call(
        [python, "sae_experiments/scripts/05_visualize_features.py", "--config", args.config, "--catalog", feature_catalog, "--sae_checkpoint", sae_checkpoint] + common
    )


if __name__ == "__main__":
    main()

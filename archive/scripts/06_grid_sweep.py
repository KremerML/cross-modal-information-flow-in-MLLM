"""Run a grid sweep over SAE identification + ablation settings."""

from __future__ import annotations

import argparse
import gc
import json
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path (str): YAML file path.

    Returns:
        Dict[str, Any]: Parsed YAML mapping, or an empty mapping when the file is empty.

    Raises:
        OSError: If ``path`` cannot be read.
        yaml.YAMLError: If file contents are invalid YAML.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(data: Dict[str, Any], path: str) -> None:
    """Write a dictionary to YAML.

    Args:
        data (Dict[str, Any]): YAML-serializable mapping.
        path (str): Output YAML path.

    Returns:
        None: Creates parent directories and writes YAML content.

    Raises:
        OSError: If the file cannot be written.
        yaml.YAMLError: If ``data`` cannot be serialized.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _grid_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Compute Cartesian product combinations from a sweep grid.

    Args:
        grid (Dict[str, List[Any]]): Mapping from hyperparameter key to candidate values.

    Returns:
        List[Dict[str, Any]]: One dictionary per grid combination.

    Raises:
        ValueError: If a grid value list is malformed.
    """
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def _run(cmd: List[str], dry_run: bool, log_path: Optional[str] = None) -> None:
    """Execute a subprocess command optionally with command logging.

    Args:
        cmd (List[str]): Command tokens passed to ``subprocess.run``.
        dry_run (bool): If ``True``, print command instead of executing it.
        log_path (Optional[str]): Optional file path for stdout/stderr redirection.

    Returns:
        None: Executes command or prints dry-run preview.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero code.
        OSError: If log file creation fails.
    """
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(cmd)}\n")
            handle.flush()
            try:
                subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as exc:
                handle.write(f"\n[command failed] returncode={exc.returncode}\n")
                handle.flush()
                raise
        return
    subprocess.run(cmd, check=True)


def _log_status(path: str, payload: Dict[str, Any]) -> None:
    """Append a timestamped JSON line entry to the sweep status log.

    Args:
        path (str): JSONL status file path.
        payload (Dict[str, Any]): Event payload to append.

    Returns:
        None: Appends one JSON line to ``path``.

    Raises:
        OSError: If the status file cannot be written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(payload)
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _cleanup_cuda() -> None:
    """Release Python and CUDA memory between sweep stages.

    Args:
        None.

    Returns:
        None: Attempts best-effort cache cleanup.

    Raises:
        None: Exceptions from CUDA cleanup are intentionally swallowed.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _training_cache_key(run: Dict[str, Any], run_config: Dict[str, Any], train_position: str) -> str:
    """Build a deterministic key for reusable training-equivalent runs.

    Args:
        run (Dict[str, Any]): Current sweep run specification.
        run_config (Dict[str, Any]): Resolved run configuration mapping.
        train_position (str): Position type used during SAE training.

    Returns:
        str: Stable JSON key used to cache/reuse compatible checkpoints.

    Raises:
        TypeError: If payload fields are not JSON serializable.
    """
    model_cfg = run_config.get("model", {})
    dataset_cfg = run_config.get("dataset", {})
    payload = {
        "layer": int(run["layer"]),
        "train_position": str(train_position),
        "model": {
            "name": model_cfg.get("name"),
            "model_base": model_cfg.get("model_base"),
            "target_layer": model_cfg.get("target_layer"),
            "activation_site": model_cfg.get("activation_site", "residual"),
            "d_model": model_cfg.get("d_model"),
        },
        "sae": run_config.get("sae", {}),
        "training": run_config.get("training", {}),
        "dataset": {
            "refined_dataset": dataset_cfg.get("refined_dataset"),
            "image_folder": dataset_cfg.get("image_folder"),
            "task_types": dataset_cfg.get("task_types"),
            "split": dataset_cfg.get("split"),
        },
    }
    return json.dumps(payload, sort_keys=True)


def main() -> None:
    """Execute the configured grid sweep for train/identify/ablation stages.

    Args:
        None: CLI arguments are parsed inside this function.

    Returns:
        None: Produces per-run configs, logs, and result artifacts.

    Raises:
        ValueError: If required sweep configuration keys are missing.
        subprocess.CalledProcessError: If a stage command fails and stop-on-error is enabled.
        OSError: If run artifacts cannot be created.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Sweep config YAML.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have a results file.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop the sweep immediately if a run fails.",
    )
    args = parser.parse_args()

    sweep_cfg = _load_yaml(args.config).get("sweep", {})
    base_config_path = sweep_cfg.get("base_config")
    if not base_config_path:
        raise ValueError("sweep.base_config is required")
    base_config = _load_yaml(base_config_path)

    sweep_name = sweep_cfg.get("name", "sae_grid")
    output_base = sweep_cfg.get("output_base", "output/sae_experiments/sweeps")
    layers = sweep_cfg.get("layers", [base_config.get("model", {}).get("target_layer", 12)])
    train_positions = sweep_cfg.get("train_position_types", ["question"])
    feature_positions = sweep_cfg.get("feature_position_types", ["attribute"])
    grid = sweep_cfg.get("grid", {})
    steps = sweep_cfg.get("steps", {"train": True, "identify": True, "ablate": True})
    reuse_checkpoints = bool(sweep_cfg.get("reuse_checkpoints", True))
    identify_max_samples = sweep_cfg.get("identify_max_samples")
    ablation_max_samples = sweep_cfg.get("ablation_max_samples")
    dry_run = args.dry_run or bool(sweep_cfg.get("dry_run", False))

    combos = _grid_product(grid)
    if not combos:
        combos = [{}]

    runs = []
    for layer in layers:
        for train_position in train_positions:
            for feature_position in feature_positions:
                for combo in combos:
                    run = {
                        "layer": layer,
                        "train_position": train_position,
                        "feature_position": feature_position,
                        **combo,
                    }
                    runs.append(run)

    trained_checkpoint_cache: Dict[str, str] = {}

    for idx, run in enumerate(runs, start=1):
        run_name = (
            f"{sweep_name}_layer{run['layer']}"
            f"_train-{run['train_position']}"
            f"_feat-{run['feature_position']}"
        )
        for key in ("top_k", "min_activation", "mode", "delta_scale"):
            if key in run:
                run_name += f"_{key}{run[key]}"
        experiment_dir = os.path.join(output_base, run_name)

        run_config = json.loads(json.dumps(base_config))
        run_config.setdefault("experiment", {})
        run_config["experiment"]["output_dir"] = experiment_dir
        run_config["experiment"]["name"] = run_name
        run_config.setdefault("model", {})
        run_config["model"]["target_layer"] = int(run["layer"])

        run_config.setdefault("feature_identification", {})
        run_config["feature_identification"]["position_type"] = run["feature_position"]
        if "top_k" in run:
            run_config["feature_identification"]["top_k"] = int(run["top_k"])
        if "min_activation" in run:
            run_config["feature_identification"]["min_activation"] = float(run["min_activation"])

        run_config.setdefault("ablation", {})
        if "mode" in run:
            run_config["ablation"]["mode"] = run["mode"]
        if "delta_scale" in run:
            run_config["ablation"]["delta_scale"] = float(run["delta_scale"])
        run_config["ablation"]["position_type"] = run["feature_position"]

        run_config_path = os.path.join(experiment_dir, "config.yaml")
        _write_yaml(run_config, run_config_path)
        train_key = _training_cache_key(run, run_config, run["train_position"])

        sae_checkpoint = os.path.join(experiment_dir, "sae_checkpoint.pt")
        feature_catalog = os.path.join(experiment_dir, "feature_catalog.json")
        ablation_output = os.path.join(experiment_dir, "results", "ablation_results.json")
        run_log = os.path.join(experiment_dir, "run.log")
        status_log = os.path.join(output_base, f"{sweep_name}_status.jsonl")

        print(f"[{idx}/{len(runs)}] {run_name}")
        os.makedirs(experiment_dir, exist_ok=True)
        if not os.path.exists(run_log):
            with open(run_log, "a", encoding="utf-8") as handle:
                handle.write(f"[run start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        _log_status(
            status_log,
            {
                "run_name": run_name,
                "event": "start",
                "index": idx,
                "total": len(runs),
                "experiment_dir": experiment_dir,
            },
        )

        if args.resume and os.path.exists(ablation_output):
            print(f"Skipping run; results exist at {ablation_output}")
            if reuse_checkpoints and os.path.exists(sae_checkpoint):
                trained_checkpoint_cache[train_key] = sae_checkpoint
            _log_status(
                status_log,
                {
                    "run_name": run_name,
                    "event": "skip_existing_results",
                    "index": idx,
                    "total": len(runs),
                    "experiment_dir": experiment_dir,
                },
            )
            continue

        try:
            if steps.get("train", True):
                if not reuse_checkpoints or not os.path.exists(sae_checkpoint):
                    cached_ckpt = trained_checkpoint_cache.get(train_key) if reuse_checkpoints else None
                    if cached_ckpt and os.path.exists(cached_ckpt):
                        if dry_run:
                            print(f"DRY RUN: reuse checkpoint {cached_ckpt} -> {sae_checkpoint}")
                        else:
                            shutil.copy2(cached_ckpt, sae_checkpoint)
                            print(f"Reused checkpoint from {cached_ckpt}")
                    else:
                        cmd = [
                            sys.executable,
                            str(ROOT / "sae_experiments" / "scripts" / "01_train_sae.py"),
                            "--config",
                            run_config_path,
                            "--target_layer",
                            str(run["layer"]),
                            "--position_type",
                            run["train_position"],
                            "--experiment_dir",
                            experiment_dir,
                        ]
                        _run(cmd, dry_run, log_path=run_log)
                else:
                    print(f"Skipping training; checkpoint exists at {sae_checkpoint}")
                if reuse_checkpoints and os.path.exists(sae_checkpoint):
                    trained_checkpoint_cache[train_key] = sae_checkpoint

            _cleanup_cuda()

            if steps.get("identify", True):
                cmd = [
                    sys.executable,
                    str(ROOT / "sae_experiments" / "scripts" / "02_identify_features.py"),
                    "--config",
                    run_config_path,
                    "--sae_checkpoint",
                    sae_checkpoint,
                    "--experiment_dir",
                    experiment_dir,
                ]
                if identify_max_samples is not None:
                    cmd += ["--max_samples", str(identify_max_samples)]
                _run(cmd, dry_run, log_path=run_log)

            _cleanup_cuda()

            if steps.get("ablate", True):
                cmd = [
                    sys.executable,
                    str(ROOT / "sae_experiments" / "scripts" / "03_run_ablation.py"),
                    "--config",
                    run_config_path,
                    "--features",
                    feature_catalog,
                    "--sae_checkpoint",
                    sae_checkpoint,
                    "--output",
                    ablation_output,
                    "--experiment_dir",
                    experiment_dir,
                ]
                if ablation_max_samples is not None:
                    cmd += ["--max_samples", str(ablation_max_samples)]
                _run(cmd, dry_run, log_path=run_log)

            _log_status(
                status_log,
                {
                    "run_name": run_name,
                    "event": "completed",
                    "index": idx,
                    "total": len(runs),
                    "experiment_dir": experiment_dir,
                },
            )
        except subprocess.CalledProcessError as exc:
            _log_status(
                status_log,
                {
                    "run_name": run_name,
                    "event": "failed",
                    "index": idx,
                    "total": len(runs),
                    "experiment_dir": experiment_dir,
                    "returncode": exc.returncode,
                },
            )
            print(f"Run failed (return code {exc.returncode}). See {run_log} for details.")
            if args.stop_on_error:
                raise
        finally:
            _cleanup_cuda()


if __name__ == "__main__":
    main()

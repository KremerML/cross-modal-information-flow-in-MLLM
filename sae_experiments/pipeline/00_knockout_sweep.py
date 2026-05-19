"""Run per-layer attention knockout sweeps for specified flows."""

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from InformationFlow import create_data_loader
from sae_experiments.core.config import load_config, save_config
from sae_experiments.tools.knockout_runner import run_knockout_sweep
from sae_experiments.utils.checkpoint_utils import resolve_experiment_dir
from sae_experiments.utils.config_utils import resolve_primary_task_type


def main() -> None:
    """Run a configurable layerwise attention knockout sweep.

    Args:
        None: Arguments are provided via CLI flags parsed in this function.

    Returns:
        None: Writes knockout result artifacts to disk.

    Raises:
        FileNotFoundError: If the configured dataset path cannot be read.
        ValueError: If required configuration values are invalid.
        RuntimeError: If model loading or sweep execution fails.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--flows", type=str, default=None, help="Comma-separated flow names.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    knockout_cfg = config.get("knockout", {})
    experiment_cfg = dict(config.get("experiment", {}))
    if args.experiment_name:
        experiment_cfg["name"] = args.experiment_name
        experiment_cfg.pop("output_dir", None)

    experiment_dir = resolve_experiment_dir(experiment_cfg, args.experiment_dir)
    knockout_dir = os.path.join(experiment_dir, knockout_cfg.get("output_subdir", "knockout"))
    os.makedirs(knockout_dir, exist_ok=True)

    save_config(config, os.path.join(experiment_dir, "config.yaml"))

    model_path = os.path.expanduser(model_cfg.get("name", ""))
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_cfg.get("model_base"),
        model_name,
        device_map="auto",
        attn_implementation=None,
    )
    model.eval()

    dataset_format = data_cfg.get("format", "csv")
    if dataset_format == "clevr_lite":
        from sae_experiments.data.clevr_lite_dataset import CLEVRLiteVQADataset
        dataset_obj = CLEVRLiteVQADataset(
            data_dir=data_cfg.get("data_dir", "datasets/clevr_lite"),
            split=data_cfg.get("split", "val"),
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_config=model.config,
            conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
        )
        dataset_dict = dataset_obj.dataset_dict
        questions = dataset_obj.questions
        data_loader = dataset_obj.create_dataloader(
            batch_size=knockout_cfg.get("batch_size", 1),
            num_workers=knockout_cfg.get("num_workers", 0),
        )
    else:
        refined_dataset = data_cfg.get("refined_dataset", "")
        df = pd.read_csv(refined_dataset, dtype={"question_id": str}).fillna("")
        dataset_dict = df.set_index("question_id").T.to_dict("dict")
        questions = [{**detail, "q_id": qu_id} for qu_id, detail in dataset_dict.items()]

        task_name = resolve_primary_task_type(data_cfg.get("task_types"), default="ChooseAttr")
        data_loader = create_data_loader(
            questions,
            data_cfg.get("image_folder", ""),
            knockout_cfg.get("batch_size", 1),
            knockout_cfg.get("num_workers", 2),
            tokenizer,
            image_processor,
            model.config,
            task_name,
            model_cfg.get("conv_mode", "vicuna_v1"),
        )

    flows = knockout_cfg.get("flows", ["Image->Question", "Image->Last"])
    if args.flows:
        flows = [flow.strip() for flow in args.flows.split(",") if flow.strip()]

    window = args.window if args.window is not None else knockout_cfg.get("window", 1)
    max_samples = args.max_samples if args.max_samples is not None else knockout_cfg.get("max_samples")
    filter_correct = knockout_cfg.get("filter_correct", True)
    normalize_logprob = knockout_cfg.get("normalize_logprob", True)

    checkpoint_path = os.path.join(knockout_dir, "checkpoint.jsonl")

    results, summaries = run_knockout_sweep(
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset_dict,
        questions=questions,
        data_loader=data_loader,
        flows=flows,
        model_name=model_name,
        window=window,
        max_samples=max_samples,
        filter_correct=filter_correct,
        normalize_logprob=normalize_logprob,
        progress_desc="Knockout sweep",
        checkpoint_path=checkpoint_path,
    )

    results_path = os.path.join(knockout_dir, "knockout_results.json")
    summary_path = os.path.join(knockout_dir, "knockout_summary.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    print(f"Saved knockout results ({len(results)} rows) to {results_path}")
    print(f"Saved knockout summary ({len(summaries)} entries) to {summary_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()

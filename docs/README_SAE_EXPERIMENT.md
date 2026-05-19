# SAE-Based Attribute Binding Experiments

> **Note (May 2026):** This doc describes the original GQA-based pipeline (Gen 1-2). The active pipeline uses CLEVR-Lite with v2 causal feature identification. Many scripts and configs referenced below have been archived or renamed — see `docs/CLAUDE.md` for the current structure.

This directory adds a Sparse Autoencoder (SAE) pipeline for identifying and testing attribute-binding features in LLaVA models.

## Overview
The pipeline is organized around five steps:
1. Train an SAE on question-token activations.
2. Identify discriminative features for attribute binding.
3. Run ablations to test causal necessity.
4. Analyze results with statistical tests.
5. Visualize top features and examples.

## Installation
Use the existing LLaVA-NeXT environment, then install additional dependencies:
```
pip install -r requirements_sae.txt
```

## Quick Start
Run the full pipeline into a dedicated experiment folder:
```
~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/run_full_pipeline.py \
  --config configs/sae_config_llava15_7b.yaml \
  --experiment_name exp_run1
```

Reuse an existing SAE checkpoint (skip training):
```
~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/run_full_pipeline.py \
  --config configs/sae_config_llava15_7b.yaml \
  --experiment_name exp_run1 \
  --skip_train \
  --sae_checkpoint output/sae_experiments/exp_run1/sae_checkpoint.pt
```

Or run the steps individually (same output folder):
```
~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/01_train_sae.py \
  --config configs/sae_config_llava15_7b.yaml \
  --experiment_name exp_run1

~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/02_identify_features.py \
  --config configs/sae_config_llava15_7b.yaml \
  --experiment_name exp_run1

~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/03_run_ablation.py \
  --config configs/sae_config_llava15_7b.yaml \
  --experiment_name exp_run1

~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/04_analyze_results.py \
  --config configs/sae_config_llava15_7b.yaml \
  --experiment_name exp_run1

~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/05_visualize_features.py \
  --config configs/sae_config_llava15_7b.yaml \
  --experiment_name exp_run1
```

To explore variants, swap the config, for example:
```
~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/run_full_pipeline.py \
  --config configs/sae_config_llava15_7b_attr_residual_strong.yaml \
  --experiment_name exp_attr_strong
```

## Configuration
- `configs/sae_config_llava15_7b.yaml`: Full configuration for training, feature discovery, and ablation.
- `configs/experiment_config.yaml`: Optional experiment-only settings.
- `configs/sae_config_llava15_7b_*.yaml`: Example sweep configs (attribute/question/all + residual/replace).

Update these fields to point to your dataset CSV and image folder:
- `dataset.refined_dataset`
- `dataset.image_folder`

To keep multiple runs separate, set either:
- `experiment.output_dir` in the config, or
- `--experiment_dir` on the CLI for each script.
You can also pass `--experiment_name`, which creates `output/sae_experiments/<name>`.

Useful knobs for ablation + evaluation:
- `ablation.position_type`: `attribute`, `question`, or `all`
- `ablation.mode`: `residual` or `replace`
- `ablation.delta_scale`: scales the residual ablation strength
- `evaluation.primary_metric`: `pred_token_prob` or `gt_token_prob`

Feature identification knobs:
- `feature_identification.correctness_metric`: `option_logprob` or `string_match`
- `feature_identification.position_type`: `attribute`, `question`, or `all`
- `feature_identification.min_activation` / `min_diff`: filters for stable feature ratios
- `feature_identification.logprob_normalize`: normalize option log-probs by length
Note: `option_logprob` requires `true option`/`false option` columns in the dataset.

## Outputs
- SAE checkpoints: `output/sae_experiments/exp_run1/sae_checkpoint.pt`
- Feature catalog: `output/sae_experiments/exp_run1/feature_catalog.json`
- Ablation results: `output/sae_experiments/exp_run1/results/ablation_results.json`
- Analysis report: `output/sae_experiments/exp_run1/analysis/hypothesis_report.json`
- Visualization dashboard: `output/sae_experiments/exp_run1/feature_dashboard/`

## Category-Split Datasets (ChooseAttr)
You can split `ChooseAttr` into per-attribute-category datasets and generate matching configs:

```bash
python sae_experiments/scripts/07_build_category_datasets.py \
  --input_csv datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv \
  --output_dir datasets/by_attribute_category \
  --policy first

python sae_experiments/scripts/08_generate_category_configs.py \
  --base_config configs/sae_first_layer11_attn_out.yaml \
  --dataset_dir datasets/by_attribute_category \
  --output_dir configs/sae_categories
```

Generated files:
- Datasets: `datasets/by_attribute_category/ChooseAttr_<category>.csv`
- Split manifest: `datasets/by_attribute_category/manifest.json`
- Configs: `configs/sae_categories/<base_config_stem>/<category>.yaml`
- Config manifest: `configs/sae_categories/<base_config_stem>/manifest.json`

## Notes
- The pipeline reuses the existing LLaVA data loader from `InformationFlow.py`.
- For attention-knockout baselines, see `AblationExperiment.run_attention_knockout_baseline`.
- `run_full_pipeline.py` accepts `--experiment_dir` / `--experiment_name` to keep outputs grouped.

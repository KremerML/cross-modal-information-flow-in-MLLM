# SAE-Based Attribute Binding Experiments

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
```
# Train SAE
~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/01_train_sae.py \
  --config configs/sae_config_llava15_7b.yaml

# Identify features
~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/02_identify_features.py \
  --config configs/sae_config_llava15_7b.yaml \
  --sae_checkpoint output/sae_experiments/sae_checkpoint.pt

# Run ablations
~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/03_run_ablation.py \
  --config configs/sae_config_llava15_7b.yaml \
  --features output/sae_experiments/feature_catalog.json \
  --sae_checkpoint output/sae_experiments/sae_checkpoint.pt

# Analyze results
~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/04_analyze_results.py \
  --config configs/sae_config_llava15_7b.yaml \
  --results output/sae_experiments/results/ablation_results.json

# Visualize features
~/Documents/Github/cross-modal-information-flow-in-MLLM/LLaVA-NeXT/.venv/bin/python \
  sae_experiments/scripts/05_visualize_features.py \
  --config configs/sae_config_llava15_7b.yaml \
  --catalog output/sae_experiments/feature_catalog.json \
  --sae_checkpoint output/sae_experiments/sae_checkpoint.pt
```

## Configuration
- `configs/sae_config_llava15_7b.yaml`: Full configuration for training, feature discovery, and ablation.
- `configs/experiment_config.yaml`: Optional experiment-only settings.

Update these fields to point to your dataset CSV and image folder:
- `dataset.refined_dataset`
- `dataset.image_folder`

## Outputs
- SAE checkpoints: `output/sae_experiments/sae_checkpoint.pt`
- Feature catalog: `output/sae_experiments/feature_catalog.json`
- Ablation results: `output/sae_experiments/results/ablation_results.json`
- Analysis report: `output/sae_experiments/results/analysis/hypothesis_report.json`
- Visualization dashboard: `output/sae_experiments/feature_dashboard/`

## Notes
- The pipeline reuses the existing LLaVA data loader from `InformationFlow.py`.
- For attention-knockout baselines, see `AblationExperiment.run_attention_knockout_baseline`.

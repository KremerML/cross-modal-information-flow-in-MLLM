#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "LLaVA-NeXT/.venv/bin/activate" ]]; then
  echo "Missing virtual environment activation script: .venv/bin/activate" >&2
  exit 1
fi

# shellcheck disable=SC1091
source LLaVA-NeXT/.venv/bin/activate

CFG="${1:-configs/sae_first_layer11_attn_out.yaml}"
RUN="${2:-output/sae_experiments/rerun_layer11_attn_out_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG" >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "Config: $CFG"
echo "Run dir: $RUN"
echo "CUBLAS_WORKSPACE_CONFIG: $CUBLAS_WORKSPACE_CONFIG"

# 0) Recompute attention-knockout baseline (fresh run)
python sae_experiments/scripts/00_knockout_sweep.py \
  --config "$CFG" \
  --experiment_dir "$RUN"

# 1) Train SAE on attention outputs (layer 11, attribute positions)
python sae_experiments/scripts/01_train_sae.py \
  --config "$CFG" \
  --target_layer 11 \
  --position_type attribute \
  --experiment_dir "$RUN"

# 2) Identify features
python sae_experiments/scripts/02_identify_features.py \
  --config "$CFG" \
  --sae_checkpoint "$RUN/sae_checkpoint.pt" \
  --experiment_dir "$RUN"

# 3) Run binding vs matched-random ablation
python sae_experiments/scripts/03_run_ablation.py \
  --config "$CFG" \
  --features "$RUN/feature_catalog.json" \
  --sae_checkpoint "$RUN/sae_checkpoint.pt" \
  --output "$RUN/results/ablation_results.json" \
  --experiment_dir "$RUN"

# 4) Statistical analysis/report
python sae_experiments/scripts/04_analyze_results.py \
  --config "$CFG" \
  --results "$RUN/results/ablation_results.json" \
  --output "$RUN/analysis" \
  --experiment_dir "$RUN"

# 5) Causal ranking pass (single-feature ablations)
python sae_experiments/scripts/04_rank_features_causally.py \
  --config "$CFG" \
  --sae_checkpoint "$RUN/sae_checkpoint.pt" \
  --output "$RUN/causal_feature_scores.json" \
  --experiment_dir "$RUN"

echo "Completed full attn_out rerun at: $RUN"

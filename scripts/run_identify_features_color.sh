#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${1:-configs/sae_categories/sae_first_layer11_attn_out_color.yaml}"
RUN_DIR="${2:-output/sae_experiments/first_pass_layer11_attn_out_color}"
SAE_CKPT="${3:-$RUN_DIR/sae_checkpoint.pt}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"

if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG" >&2
  exit 1
fi

if [[ ! -f "$SAE_CKPT" ]]; then
  echo "SAE checkpoint not found: $SAE_CKPT" >&2
  exit 1
fi

echo "Python: $PYTHON_BIN"
echo "Config: $CFG"
echo "Run dir: $RUN_DIR"
echo "SAE checkpoint: $SAE_CKPT"
echo "Show progress: $SHOW_PROGRESS"
if [[ -n "$MAX_SAMPLES" ]]; then
  echo "Max samples: $MAX_SAMPLES"
fi

CMD=(
  "$PYTHON_BIN" sae_experiments/scripts/02_identify_features.py
  --config "$CFG"
  --sae_checkpoint "$SAE_CKPT"
  --experiment_dir "$RUN_DIR"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max_samples "$MAX_SAMPLES")
fi

if [[ "${SHOW_PROGRESS,,}" == "false" ]]; then
  CMD+=(--no_progress)
fi

"${CMD[@]}"


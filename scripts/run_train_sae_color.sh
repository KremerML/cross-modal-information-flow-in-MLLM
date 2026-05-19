#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${1:-configs/sae_categories/sae_first_layer11_attn_out_color.yaml}"
RUN_DIR="${2:-output/sae_experiments/first_pass_layer11_attn_out_color}"
POSITION_TYPE="${3:-attribute}"
TARGET_LAYER="${4:-11}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"

if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG" >&2
  exit 1
fi

echo "Python: $PYTHON_BIN"
echo "Config: $CFG"
echo "Run dir: $RUN_DIR"
echo "Position type: $POSITION_TYPE"
echo "Target layer: $TARGET_LAYER"
echo "Show progress: $SHOW_PROGRESS"
if [[ -n "$MAX_SAMPLES" ]]; then
  echo "Max samples: $MAX_SAMPLES"
fi

CMD=(
  "$PYTHON_BIN" sae_experiments/pipeline/01_train_sae.py
  --config "$CFG"
  --target_layer "$TARGET_LAYER"
  --position_type "$POSITION_TYPE"
  --show_progress "$SHOW_PROGRESS"
  --experiment_dir "$RUN_DIR"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max_samples "$MAX_SAMPLES")
fi

"${CMD[@]}"


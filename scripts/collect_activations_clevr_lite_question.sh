#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source LLaVA-NeXT/.venv/bin/activate

python sae_experiments/scripts/collect_activations.py \
    --config configs/clevr_lite/sae_layer0_attn_out_question.yaml \
    --layers 0,10,11,12,13,14 \
    --output_dir "/run/media/ron/External SSD/clevr_lite_activations/question" \
    --batch_size 8 \
    --num_workers 4 \
    --show_progress true

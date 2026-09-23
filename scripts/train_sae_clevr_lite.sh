#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source LLaVA-NeXT/.venv/bin/activate

ACTS_DIR="/run/media/ron/External SSD/clevr_lite_activations/question"

for layer in 0 10 11 12 13 14; do
    echo "========================================"
    echo " Training SAE — layer ${layer}"
    echo "========================================"
    python sae_experiments/pipeline/01_train_sae.py \
        --config "configs/clevr_lite/sae_layer${layer}_attn_out_question.yaml" \
        --target_layer "${layer}" \
        --activations_path "${ACTS_DIR}" \
        --show_progress true
done

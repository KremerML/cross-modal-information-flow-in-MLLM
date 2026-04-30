#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source LLaVA-NeXT/.venv/bin/activate

ACTS="/run/media/ron/External SSD/clevr_lite_activations/question"

for layer in 0 10 11 12 13 14; do
    echo "=== Training SAE layer ${layer} (question positions, from cache) ==="
    python sae_experiments/scripts/01_train_sae.py \
        --config "configs/clevr_lite/sae_layer${layer}_attn_out_question.yaml" \
        --activations_path "${ACTS}" \
        --show_progress true
done

echo "=== All layers complete ==="

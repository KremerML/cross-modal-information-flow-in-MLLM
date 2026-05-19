#!/usr/bin/env bash
# Full pipeline: train remaining SAEs + run causal feature ID on all layers.
# Designed to be resumable — checks for existing outputs before running each step.

set -euo pipefail

cd /home/ron/Documents/Github/cross-modal-information-flow-in-MLLM
PYTHON="LLaVA-NeXT/.venv/bin/python"
LAYERS=(0 10 11 12 13 14)
LOGFILE="output/sae_experiments/pipeline_log.txt"
ACTIVATIONS_PATH="/run/media/ron/External SSD/clevr_lite_activations/question"

mkdir -p output/sae_experiments
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "Pipeline started: $(date)"
echo "=========================================="

# Phase 1: Train SAEs for layers that don't have checkpoints
echo ""
echo "=== PHASE 1: SAE Training ==="
for layer in "${LAYERS[@]}"; do
    CHECKPOINT="output/sae_experiments/sae_clevr_lite_layer${layer}_attn_out_question/sae_checkpoint.pt"
    CONFIG="configs/clevr_lite/sae_layer${layer}_attn_out_question.yaml"

    if [ -f "$CHECKPOINT" ]; then
        echo "[SKIP] Layer $layer: checkpoint already exists"
        continue
    fi

    echo ""
    echo "[TRAIN] Layer $layer: starting at $(date)"
    $PYTHON sae_experiments/pipeline/01_train_sae.py \
        --config "$CONFIG" \
        --activations_path "$ACTIVATIONS_PATH" \
        --show_progress true
    echo "[TRAIN] Layer $layer: finished at $(date)"
done

echo ""
echo "=== PHASE 1 COMPLETE ==="

# Phase 2: Run causal feature identification on all layers
echo ""
echo "=== PHASE 2: Causal Feature Identification ==="
for layer in "${LAYERS[@]}"; do
    CAUSAL_DIR="output/sae_experiments/sae_clevr_lite_layer${layer}_attn_out_question_causal"
    CATALOG="$CAUSAL_DIR/causal_feature_catalog.json"
    CONFIG="configs/clevr_lite/sae_layer${layer}_attn_out_question.yaml"

    if [ -f "$CATALOG" ]; then
        echo "[SKIP] Layer $layer: causal catalog already exists"
        continue
    fi

    echo ""
    echo "[CAUSAL] Layer $layer: starting at $(date)"
    $PYTHON sae_experiments/pipeline/02_identify_features_causal.py \
        --config "$CONFIG" \
        --target margin \
        --position_type question \
        --top_k 200
    echo "[CAUSAL] Layer $layer: finished at $(date)"
done

echo ""
echo "=========================================="
echo "Pipeline complete: $(date)"
echo "=========================================="

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
for layer in "${LAYERS[@]}"; do
    CHECKPOINT="output/sae_experiments/sae_clevr_lite_layer${layer}_attn_out_question/sae_checkpoint.pt"
    CATALOG="output/sae_experiments/sae_clevr_lite_layer${layer}_attn_out_question_causal/causal_feature_catalog.json"
    SUMMARY="output/sae_experiments/sae_clevr_lite_layer${layer}_attn_out_question_causal/causal_summary.json"

    train_status="MISSING"
    causal_status="MISSING"
    [ -f "$CHECKPOINT" ] && train_status="OK"
    [ -f "$CATALOG" ] && causal_status="OK"

    echo "Layer $layer: train=$train_status, causal=$causal_status"
    if [ -f "$SUMMARY" ]; then
        $PYTHON -c "
import json
with open('$SUMMARY') as f:
    s = json.load(f)
cs = s.get('causal_score', {})
print(f'  n_processed={s.get(\"n_processed\",\"?\")}, causal_score: p99={cs.get(\"p99\",0):.2e}, max={cs.get(\"max\",0):.2e}')
"
    fi
done

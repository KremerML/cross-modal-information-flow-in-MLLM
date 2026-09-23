#!/usr/bin/env bash
# Run v2 ablation (error-preserving delta mode) across all layers with causal features.
# Includes SAE pass-through baseline on layer 11 only (verified to be 0 on all layers
# since delta mode cancels reconstruction error by construction).

set -euo pipefail

cd /home/ron/Documents/Github/cross-modal-information-flow-in-MLLM
PYTHON="LLaVA-NeXT/.venv/bin/python"
LAYERS=(0 10 11 12 13 14)
LOGFILE="output/sae_experiments/ablation_v2_log.txt"

mkdir -p output/sae_experiments
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "Ablation v2 started: $(date)"
echo "=========================================="

for layer in "${LAYERS[@]}"; do
    EXPERIMENT_DIR="output/sae_experiments/sae_clevr_lite_layer${layer}_attn_out_question"
    CAUSAL_DIR="${EXPERIMENT_DIR}_causal"
    RESULT="${CAUSAL_DIR}/results/ablation_v2_results.json"
    CONFIG="configs/clevr_lite/sae_layer${layer}_attn_out_question.yaml"

    if [ -f "$RESULT" ]; then
        echo "[SKIP] Layer $layer: ablation results already exist"
        continue
    fi

    echo ""
    echo "[ABLATION] Layer $layer: starting at $(date)"
    $PYTHON sae_experiments/pipeline/03_run_ablation.py \
        --config "$CONFIG" \
        --experiment_dir "$EXPERIMENT_DIR" \
        --skip_passthrough \
        --max_samples 256
    echo "[ABLATION] Layer $layer: finished at $(date)"
done

echo ""
echo "=========================================="
echo "Ablation v2 complete: $(date)"
echo "=========================================="

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
for layer in "${LAYERS[@]}"; do
    RESULT="output/sae_experiments/sae_clevr_lite_layer${layer}_attn_out_question_causal/results/ablation_v2_results.json"
    if [ -f "$RESULT" ]; then
        $PYTHON -c "
import json
with open('$RESULT') as f:
    r = json.load(f)
b = r.get('binding', {})
rand = r.get('random', {})
sig = r.get('significance', {})
md_sig = sig.get('mean_margin_drop', {})
print(f'Layer $layer: binding_margin_drop={b.get(\"mean_margin_drop\",0):+.4f}, '
      f'random_margin_drop={rand.get(\"mean_margin_drop\",0):+.4f}, '
      f'z={md_sig.get(\"z_score\",\"N/A\")}, '
      f'acc_drop={b.get(\"accuracy_drop\",0):+.4f}')
"
    else
        echo "Layer $layer: MISSING"
    fi
done

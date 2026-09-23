#!/usr/bin/env bash
# Multi-layer SAE ablation across layers 10-14, CLEVR-Lite, question positions.
#
# Tests whether the 30-50% shortfall between single-layer ablation and the attention-knockout
# ceiling is carried redundantly across layers.
#
# Resumable: the runner appends one fsync'd line per completed condition to checkpoint.jsonl
# and skips completed ids on restart, so re-running this script picks up where it stopped.
#
#   bash scripts/run_multilayer_ablation_clevr_lite_question.sh            # every phase
#   bash scripts/run_multilayer_ablation_clevr_lite_question.sh gate       # just the gate
#   PHASES="primary,nested" bash scripts/run_multilayer_ablation_clevr_lite_question.sh

set -euo pipefail

cd /home/ron/Documents/Github/cross-modal-information-flow-in-MLLM
PYTHON="LLaVA-NeXT/.venv/bin/python"
CONFIG="configs/clevr_lite/multilayer_l10-14_attn_out_question.yaml"
EXPERIMENT_DIR="output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question"
LOGFILE="${EXPERIMENT_DIR}/run_log.txt"
MAX_SAMPLES="${MAX_SAMPLES:-256}"

# Five fp32 SAEs sit alongside the 7B model; this keeps fragmentation from eating the margin.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Everything below is piped through tee, so stdout is not a tty and Python block-buffers it.
# Without this, log lines sit in a 4-8KB buffer for minutes while tqdm (which writes to
# stderr) keeps updating -- which looks exactly like the run having gone silent.
export PYTHONUNBUFFERED=1

# Phases run in dependency order. The gate comes first because nothing downstream means
# anything until A0 reproduces the published single-layer layer-11 number.
DEFAULT_PHASES="gate primary nested non_nested leave_one_out budget downstream sensitivity"
if [ "$#" -gt 0 ]; then
    PHASE_LIST="$*"
else
    PHASE_LIST="${PHASES:-$DEFAULT_PHASES}"
    PHASE_LIST="${PHASE_LIST//,/ }"
fi

mkdir -p "$EXPERIMENT_DIR"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "Multi-layer ablation started: $(date)"
echo "Phases: $PHASE_LIST"
echo "Samples per condition: $MAX_SAMPLES"
echo "=========================================="

echo ""
echo "[PLAN] condition matrix:"
$PYTHON sae_experiments/tools/run_multilayer_ablation.py \
    --config "$CONFIG" \
    --phases "$(echo "$PHASE_LIST" | tr ' ' ',')" \
    --max_samples "$MAX_SAMPLES" \
    --dry_run

PHASE_TOTAL=$(echo "$PHASE_LIST" | wc -w)
PHASE_INDEX=0
RUN_START=$(date +%s)

for phase in $PHASE_LIST; do
    PHASE_INDEX=$((PHASE_INDEX + 1))
    PHASE_START=$(date +%s)
    echo ""
    echo "=========================================="
    echo "[PHASE $PHASE_INDEX/$PHASE_TOTAL] $phase: starting at $(date '+%H:%M:%S')"
    echo "=========================================="
    $PYTHON sae_experiments/tools/run_multilayer_ablation.py \
        --config "$CONFIG" \
        --phases "$phase" \
        --experiment_dir "$EXPERIMENT_DIR" \
        --max_samples "$MAX_SAMPLES"
    PHASE_ELAPSED=$(( $(date +%s) - PHASE_START ))
    TOTAL_ELAPSED=$(( $(date +%s) - RUN_START ))
    echo "[PHASE $PHASE_INDEX/$PHASE_TOTAL] $phase: finished at $(date '+%H:%M:%S')" \
         "(took $((PHASE_ELAPSED / 60))m$((PHASE_ELAPSED % 60))s," \
         "total $((TOTAL_ELAPSED / 3600))h$(((TOTAL_ELAPSED % 3600) / 60))m)"
    echo "[PHASE $PHASE_INDEX/$PHASE_TOTAL] completed conditions so far:" \
         "$(wc -l < "$EXPERIMENT_DIR/checkpoint.jsonl" 2>/dev/null || echo 0)/47"

    # The gate is a hard stop: if the harness cannot reproduce the published single-layer
    # result, every number after it is meaningless.
    if [ "$phase" = "gate" ]; then
        echo ""
        echo "[GATE] checking A0 against the published layer-11 result (0.2131)..."
        $PYTHON - <<'PY'
import json
import os
import sys

base = "output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question/conditions"
expected = 0.21307707345113158

def drop(condition_id):
    path = os.path.join(base, condition_id, "summary.json")
    if not os.path.exists(path):
        print(f"[GATE] MISSING {condition_id}")
        sys.exit(1)
    with open(path) as handle:
        return json.load(handle)["summary"]["mean_margin_drop"]

a0 = drop("A0_regression_L11")
none = drop("gate_none")
print(f"[GATE] A0_regression_L11 margin_drop = {a0:.6f} (published {expected:.6f})")
print(f"[GATE] gate_none         margin_drop = {none:.6f} (must be exactly 0)")

failed = False
if abs(a0 - expected) > 1e-4:
    print("[GATE] FAIL: the harness does not reproduce the published result.")
    failed = True
if none != 0.0:
    print("[GATE] FAIL: a no-op condition moved the margin; the harness is not inert.")
    failed = True

passthrough = drop("gate_passthrough_L10-14")
print(f"[GATE] 5-layer replace passthrough = {passthrough:+.6f}")
if abs(passthrough) >= 0.05:
    print("[GATE] RED: stacked reconstruction error is large. Re-run with --mode residual.")
    failed = True
elif abs(passthrough) >= 0.02:
    print("[GATE] AMBER: report binding minus passthrough, and add a delta robustness arm.")
else:
    print("[GATE] GREEN: replace mode is safe to keep as primary.")

sys.exit(1 if failed else 0)
PY
        echo "[GATE] passed."
    fi
done

echo ""
echo "=========================================="
echo "Multi-layer ablation complete: $(date)"
echo "=========================================="

echo ""
echo "[NEXT] analysis:"
echo "  $PYTHON sae_experiments/tools/analyze_multilayer_ablation.py --experiment_dir $EXPERIMENT_DIR"
echo "[NEXT] distil summaries per the repo artifact policy:"
echo "  $PYTHON sae_experiments/tools/distill_results.py --root output"

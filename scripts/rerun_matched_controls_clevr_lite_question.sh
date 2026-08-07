#!/usr/bin/env bash
# Re-run all six single-layer ablations with MATCHED random controls.
#
# Every published v2 run requested matched controls and silently got uniform ones: the
# configs match on `correct_mean`, a key the v2 stats files do not carry, so every metric
# lookup returned None and the sampler fell through to a uniform draw. At layer 11 that put
# the controls at a median activation of 6.1e-08 against the binding set's 0.117 -- the same
# near-dead "ghost features" this project diagnosed in v1 selection, surviving in the control
# arm. Every published z-score is inflated by it.
#
# This puts the whole table on one control standard. Expect every z to fall; the margin drops
# and per-sample positive fractions are unaffected, since only the control arm changes.
#
# Results are written alongside the originals as ablation_matched_controls.json, never over
# them, so both remain citable and the comparison stays visible.
#
#   bash scripts/rerun_matched_controls_clevr_lite_question.sh
#   LAYERS="11 14" bash scripts/rerun_matched_controls_clevr_lite_question.sh

set -euo pipefail

cd /home/ron/Documents/Github/cross-modal-information-flow-in-MLLM
PYTHON="LLaVA-NeXT/.venv/bin/python"
LAYERS="${LAYERS:-0 10 11 12 13 14}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
LOGFILE="output/sae_experiments/rerun_matched_controls_log.txt"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p output/sae_experiments
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "Matched-control re-run started: $(date)"
echo "Layers: $LAYERS   samples: $MAX_SAMPLES"
echo "=========================================="

for layer in $LAYERS; do
    EXPERIMENT_DIR="output/sae_experiments/sae_clevr_lite_layer${layer}_attn_out_question"
    CAUSAL_DIR="${EXPERIMENT_DIR}_causal"
    OUTPUT="${CAUSAL_DIR}/results/ablation_matched_controls.json"
    CONFIG="configs/clevr_lite/sae_layer${layer}_attn_out_question.yaml"

    if [ -f "$OUTPUT" ]; then
        echo "[SKIP] Layer $layer: matched-control results already exist"
        continue
    fi

    echo ""
    echo "[RERUN] Layer $layer: starting at $(date)"
    # --strict_matching turns a missing metric into an error rather than a silent uniform
    # draw, which is the failure this whole re-run exists to correct.
    $PYTHON sae_experiments/pipeline/03_run_ablation.py \
        --config "$CONFIG" \
        --experiment_dir "$EXPERIMENT_DIR" \
        --output "$OUTPUT" \
        --matched_metric activation_mean \
        --strict_matching \
        --skip_passthrough \
        --max_samples "$MAX_SAMPLES"
    echo "[RERUN] Layer $layer: finished at $(date)"
done

echo ""
echo "=== UNIFORM (published) vs MATCHED (this run) ==="
for layer in $LAYERS; do
    CAUSAL_DIR="output/sae_experiments/sae_clevr_lite_layer${layer}_attn_out_question_causal"
    $PYTHON - "$layer" "$CAUSAL_DIR" <<'PY'
import json
import os
import sys

layer, causal_dir = sys.argv[1], sys.argv[2]
results = os.path.join(causal_dir, "results")

# Layers 10 and 11 report from ablation_results.json; the rest from ablation_v2_results.json.
published = None
for name in ("ablation_results.json", "ablation_v2_results.json"):
    path = os.path.join(results, name)
    if os.path.exists(path):
        published = path
        break

matched = os.path.join(results, "ablation_matched_controls.json")
if not os.path.exists(matched):
    print(f"Layer {layer}: matched run MISSING")
    sys.exit(0)


def read(path):
    with open(path) as handle:
        sig = json.load(handle)["significance"]["mean_margin_drop"]
    return sig["binding"], sig["z_score"]


drop_new, z_new = read(matched)
if published:
    drop_old, z_old = read(published)
    print(
        f"Layer {layer}: margin_drop {drop_old:+.4f} -> {drop_new:+.4f}  "
        f"z {z_old:.1f} -> {z_new:.1f}  ({100 * (z_new / z_old - 1):+.0f}%)"
    )
else:
    print(f"Layer {layer}: margin_drop {drop_new:+.4f}  z {z_new:.1f} (no published file found)")
PY
done

echo ""
echo "[NEXT] distil summaries per the repo artifact policy:"
echo "  $PYTHON sae_experiments/tools/distill_results.py --root output"
echo "[NEXT] then update the z column in LLM_TECHNICAL_SUMMARY.md and docs/MEMORY.md."

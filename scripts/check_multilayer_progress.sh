#!/usr/bin/env bash
# One-screen status for the detached multi-layer ablation run.
#
# The run lives in a tmux session so it survives SSH disconnects; this reads its state from
# disk rather than attaching, so it is safe to call from anywhere (phone, second shell).
#
#   bash scripts/check_multilayer_progress.sh          # snapshot
#   watch -n 60 bash scripts/check_multilayer_progress.sh
#
# To watch it live instead:  tmux attach -t multilayer   (detach again with Ctrl-b then d)

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

EXPERIMENT_DIR="output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question"
SESSION="${SESSION:-multilayer}"
TOTAL=47

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION': ALIVE"
else
    echo "tmux session '$SESSION': GONE (finished, or killed)"
fi

DONE=$(wc -l < "$EXPERIMENT_DIR/checkpoint.jsonl" 2>/dev/null || echo 0)
echo "conditions:  $DONE/$TOTAL complete"

# The runner writes one fsync'd line per completed condition, so the file mtime is the
# best available heartbeat -- a long gap means the current condition is still in its
# forward passes, or the run has wedged.
if [ -f "$EXPERIMENT_DIR/checkpoint.jsonl" ]; then
    LAST=$(( $(date +%s) - $(stat -c %Y "$EXPERIMENT_DIR/checkpoint.jsonl") ))
    echo "last result: $((LAST / 60))m$((LAST % 60))s ago  (conditions take ~1.5-3m)"
fi

echo "GPU:         $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

echo
echo "--- recent log (tqdm bars stripped) ---"
# tqdm redraws with \r into the same tee'd file, so split on \r and drop the bar frames.
tr '\r' '\n' < "$EXPERIMENT_DIR/run_log.txt" 2>/dev/null \
    | grep -vE 'it/s\]|s/it\]|\?it/s\]' \
    | grep -vE '^\s*$' \
    | tail -15

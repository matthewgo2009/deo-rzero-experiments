#!/bin/bash
# Minimal persistence probe: verify a LARGE (checkpoint-sized) dir + eval json
# persist to the OUT mount on CLEAN completion. ~10 min, no training/GPU use.
set -u
OUT=${1:-/tmp/out}
echo "=== persist_probe start | OUT=$OUT ==="
mkdir -p "$OUT/probe/sub"; echo hi > "$OUT/probe/sub/small.txt"
[ "$(cat "$OUT/probe/sub/small.txt" 2>/dev/null)" = hi ] && echo "[probe] small file OK" || { echo "[probe] small FAIL"; exit 1; }

# fake large checkpoint (~3 GB) at the real depth + a fake eval results file with a response
W=/tmp/work
mkdir -p "$W/R-Zero_run/models/qwen3-4b-base-rzero_solver_v1/global_step_1/actor/huggingface"
mkdir -p "$W/eval7/evaluation/model_x"
echo '[{"question":"q","answer":"a","response":"long response text","score":0}]' > "$W/eval7/evaluation/model_x/results_math.json"
echo '{"model":"x"}' > "$W/R-Zero_run/models/qwen3-4b-base-rzero_solver_v1/global_step_1/actor/huggingface/config.json"
dd if=/dev/zero of="$W/R-Zero_run/models/qwen3-4b-base-rzero_solver_v1/global_step_1/actor/huggingface/model.safetensors" bs=1M count=3072 2>/dev/null
echo "[probe] created fake ckpt: $(du -sh "$W/R-Zero_run/models" | cut -f1)"

# copy to OUT exactly like the pipeline's persist (cp -a; container has no rsync)
cpdir(){ [ -d "$1" ] || return 0; mkdir -p "$2"; cp -a "$1/." "$2/"; }
t0=$(date +%s)
cpdir "$W/R-Zero_run/models" "$OUT/R-Zero_run/models"
cpdir "$W/eval7/evaluation" "$OUT/eval7/evaluation"
sync
echo "[probe] cp took $(( $(date +%s) - t0 ))s"

echo "[probe] OUT huggingface dirs   = $(find "$OUT" -type d -name huggingface 2>/dev/null | wc -l)"
echo "[probe] OUT results_*.json     = $(find "$OUT" -name 'results_*.json' 2>/dev/null | wc -l)"
echo "[probe] OUT ckpt bytes         = $(du -sh "$OUT/R-Zero_run/models" 2>/dev/null | cut -f1)"
find "$OUT" -type f 2>/dev/null | sed "s#$OUT/##" | head
echo "=== persist_probe done (clean exit) ==="

#!/bin/bash
# Full pipeline inside one AzureML Command Job (8xH100 single node):
#   Phase 1  DEO baseline_drift native, 5 iters   (GPU base0, solver1/4/5, verl2/3, eval6)
#   Phase 2  R-Zero penalty-questioner, 5 iters    (GPU 0-3, native)
#   Phase 3  eval base + DEO v1-5 + R-Zero v1-5 on 7 math sets, fan-out 8 GPUs
#
# Trains on fast local /tmp/work; a background daemon mirrors DURABLE artifacts
# (merged HF checkpoints, datasets, summaries, eval json, logs) to the output
# mount every few minutes. On (re)start, prior artifacts are restored from the
# mount first, so an interrupted/resubmitted job resumes (idempotent per-iter).
#
# Invoked by job_full.yaml as:  bash azureml/run_pipeline_job.sh ${{outputs.work}}
set -u
OUT=${1:-/tmp/out}
ROOT=$PWD
WORK=/tmp/work
MODE=${MODE:-full}                 # full | deo | rzero | eval
echo "=== [$(date '+%F %T')] PIPELINE MODE=$MODE OUT=$OUT ROOT=$ROOT ==="
nvidia-smi -L

export WANDB_MODE=disabled
python3 -c "import json,os;open('$ROOT/R-Zero/tokens.json','w').write(json.dumps({'huggingface':os.environ['HF_TOKEN'],'openai':os.environ['OPENAI_API_KEY'],'anthropic':os.environ.get('ANTHROPIC_API_KEY',''),'wandb':os.environ.get('WANDB_API_KEY','')}))"

export HUGGINGFACENAME=${HUGGINGFACENAME:-yuyang322}
export MODEL_NAME=Qwen/Qwen3-4B-Base
export HF_HOME=/tmp/hf_cache HUGGINGFACE_HUB_CACHE=/tmp/hf_cache/hub
export RZERO_DIR=$ROOT/R-Zero RZERO_CODE_DIR=$ROOT/R-Zero
export VLLM_PIDDIR=/tmp/vllm_pids VLLM_LOGDIR=/tmp/vllm_logs
DEO_STORAGE=$WORK/DEO; RZERO_STORAGE=$WORK/R-Zero_run; EVAL_ROOT=$WORK/eval7
mkdir -p "$DEO_STORAGE" "$RZERO_STORAGE" "$EVAL_ROOT/evaluation" "$WORK/logs" "$OUT"

# ---- restore prior durable artifacts from the mount (resume) ----
if [ -d "$OUT/DEO" ] || [ -d "$OUT/R-Zero_run" ]; then
  echo "=== restoring prior artifacts from $OUT -> $WORK ==="
  rsync -a "$OUT/DEO/"        "$DEO_STORAGE/"     2>/dev/null || true
  rsync -a "$OUT/R-Zero_run/" "$RZERO_STORAGE/"   2>/dev/null || true
  rsync -a "$OUT/eval7/"      "$EVAL_ROOT/"       2>/dev/null || true
fi

# ---- background sync daemon: WORK durable artifacts -> OUT every 300s ----
sync_once(){
  mkdir -p "$OUT/DEO" "$OUT/R-Zero_run" "$OUT/eval7" "$OUT/logs"
  rsync -a --prune-empty-dirs --include='*/' \
    --include='results_summary*.json' --include='final_results.jsonl' \
    --include='datasets/***' --include='evaluation/***' --include='generated_question/***' \
    --include='*/global_step_*/actor/huggingface/***' \
    --exclude='*' "$DEO_STORAGE/" "$OUT/DEO/" 2>/dev/null
  rsync -a --prune-empty-dirs --include='*/' \
    --include='final_results.jsonl' --include='iter_wallclock.tsv' --include='evaluation/***' --include='generated_question/***' \
    --include='*/global_step_*/actor/huggingface/***' \
    --exclude='*' "$RZERO_STORAGE/" "$OUT/R-Zero_run/" 2>/dev/null
  rsync -a "$EVAL_ROOT/" "$OUT/eval7/" 2>/dev/null
  cp -f "$WORK"/logs/* "$OUT/logs/" 2>/dev/null
  cp -f "$ROOT/R-Zero/final_results.jsonl" "$OUT/final_results.jsonl" 2>/dev/null
  cp -f "$ROOT/RESULTS_7sets.md" "$OUT/RESULTS_7sets.md" 2>/dev/null
}
( while true; do sleep 300; sync_once; done ) &
SYNC_PID=$!
trap 'echo "[trap] final sync"; sync_once; kill $SYNC_PID 2>/dev/null' EXIT

# Thoroughly free all GPUs between phases: kill every process nvidia-smi reports
# as using a GPU (vllm spawns orphan EngineCore workers that a name-based pkill
# misses, leaving ~68GB resident and OOM-ing the next phase), then wait until
# total used memory drops near zero.
free_gpus(){
  echo "[free_gpus] killing leftover GPU processes..."
  pkill -9 -f "vllm" 2>/dev/null || true
  pkill -9 -f "baseline_drift_native_main" 2>/dev/null || true
  for r in 1 2 3; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | sort -u)
    [ -z "$pids" ] && break
    echo "[free_gpus] round $r killing: $pids"
    for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    sleep 5
  done
  for i in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
    echo "[free_gpus] total GPU mem used = ${used} MiB"
    [ "${used:-99999}" -lt 4000 ] && { echo "[free_gpus] GPUs clear"; return 0; }
    sleep 5
  done
  echo "[free_gpus] WARNING: GPUs not fully clear after wait"
}

run_deo(){
  echo "=== [$(date '+%T')] PHASE 1 DEO native (5 iters) ==="
  export STORAGE_PATH=$DEO_STORAGE PYTHONPATH=$ROOT/R-Zero DEO_NUM_ITERS=5
  bash "$ROOT/DEO/start_vllm_native.sh"
  python3 "$ROOT/DEO/baseline_drift_native_main.py"
  free_gpus
  sync_once
}
# curriculum-DEO: canonical walk + solver labeler + annealed BETA schedule
run_curriculum(){
  echo "=== [$(date '+%T')] curriculum-DEO (annealed beta, ${DEO_NUM_ITERS:-5} iters) ==="
  export STORAGE_PATH=$DEO_STORAGE PYTHONPATH=$ROOT/R-Zero DEO_NUM_ITERS=${DEO_NUM_ITERS:-5}
  bash "$ROOT/DEO/start_vllm_native.sh"
  python3 "$ROOT/DEO/curriculum_deo_native_main.py"
  free_gpus; sync_once
}
# canonical DEO (MCMC walk) with Claude (Sonnet 4.5) as labeler
run_canon(){
  echo "=== [$(date '+%T')] canonical DEO + Claude labeler (${DEO_NUM_ITERS:-5} iters) ==="
  export STORAGE_PATH=$DEO_STORAGE PYTHONPATH=$ROOT/R-Zero DEO_NUM_ITERS=${DEO_NUM_ITERS:-5}
  bash "$ROOT/DEO/start_vllm_native.sh"
  python3 "$ROOT/DEO/canonical_claude_label_native_main.py"
  free_gpus; sync_once
}
run_eval_canon(){
  local abbr=${DEO_ABBR:-deo_canon_claudelabel}
  echo "=== [$(date '+%T')] eval ${abbr} ladder on 7 sets ==="
  free_gpus
  cd "$ROOT/R-Zero"
  export STORAGE_PATH=$EVAL_ROOT PYTHONPATH=$ROOT/R-Zero VLLM_DISABLE_COMPILE_CACHE=1
  MODELS=("Qwen/Qwen3-4B-Base")
  for n in 1 2 3 4 5; do MODELS+=("$DEO_STORAGE/models/${abbr}_solver_v$n/global_step_15/actor/huggingface"); done
  _eval_fanout "${MODELS[@]}"
  ORDER=("Qwen_Qwen3-4B-Base")
  for n in 1 2 3 4 5; do ORDER+=("$(echo "$DEO_STORAGE/models/${abbr}_solver_v$n/global_step_15/actor/huggingface" | tr '/' '_')"); done
  python3 evaluation/aggregate_7sets.py --eval_root "$EVAL_ROOT/evaluation" --models "${ORDER[@]}" --out "$ROOT/RESULTS_7sets_${abbr}.md"
  cd "$ROOT"; sync_once
  echo "===== ${abbr} RESULTS ====="; cat "$ROOT/RESULTS_7sets_${abbr}.md" 2>/dev/null
}
run_rzero(){
  local iters=${RZERO_ITERS:-5}
  echo "=== [$(date '+%T')] PHASE 2 R-Zero ($iters iters) ==="
  free_gpus
  # R-Zero scripts assume these storage subdirs pre-exist (per upstream cold-start).
  # caller_penalty.py writes $STORAGE_PATH/temp_results/*.json — a missing dir there
  # crashes the questioner reward -> no checkpoint -> whole ladder cascades to empty.
  for d in evaluation models generated_question temp_results datasets logs; do
    mkdir -p "$RZERO_STORAGE/$d"
  done
  cd "$ROOT/R-Zero"
  export STORAGE_PATH=$RZERO_STORAGE
  # NCCL hang mitigation: the questioner verl (TP=2 on GPU 0,1) deadlocked in the
  # "Compute log probs" allreduce in this container (NCCL warned the rank->GPU
  # mapping was unknown). Disable P2P/IB/SHM transports so NCCL uses the reliable
  # path; also let allocator expand segments. Scoped to the R-Zero phase only.
  export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1
  export NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  source ./run_env.sh
  bash scripts/run_rzero_gpu03.sh Qwen/Qwen3-4B-Base qwen3-4b-base-rzero "$iters"
  cd "$ROOT"; free_gpus; sync_once
}
run_eval(){
  echo "=== [$(date '+%T')] PHASE 3 eval 7 sets ==="
  cd "$ROOT/R-Zero"
  export STORAGE_PATH=$EVAL_ROOT PYTHONPATH=$ROOT/R-Zero VLLM_DISABLE_COMPILE_CACHE=1
  MODELS=("Qwen/Qwen3-4B-Base")
  for n in 1 2 3 4 5; do MODELS+=("$DEO_STORAGE/models/deo_baseline_drift_solver_v$n/global_step_15/actor/huggingface"); done
  for n in 1 2 3 4 5; do MODELS+=("$RZERO_STORAGE/models/qwen3-4b-base-rzero_solver_v$n/global_step_15/actor/huggingface"); done
  _eval_fanout "${MODELS[@]}"
  ORDER=("Qwen_Qwen3-4B-Base")
  for n in 1 2 3 4 5; do ORDER+=("$(echo "$DEO_STORAGE/models/deo_baseline_drift_solver_v$n/global_step_15/actor/huggingface" | tr '/' '_')"); done
  for n in 1 2 3 4 5; do ORDER+=("$(echo "$RZERO_STORAGE/models/qwen3-4b-base-rzero_solver_v$n/global_step_15/actor/huggingface" | tr '/' '_')"); done
  python3 evaluation/aggregate_7sets.py --eval_root "$EVAL_ROOT/evaluation" --models "${ORDER[@]}" --out "$ROOT/RESULTS_7sets.md"
  cd "$ROOT"; sync_once
  echo "===== RESULTS ====="; cat "$ROOT/RESULTS_7sets.md"
}
# fan eval_7sets across 8 GPUs for the given model list
_eval_fanout(){
  local gpu=0; local pids=()
  for m in "$@"; do
    if [ "$m" != "Qwen/Qwen3-4B-Base" ] && [ ! -d "$m" ]; then echo "skip missing $m"; continue; fi
    bash evaluation/eval_7sets.bash "$m" "$gpu" > "$WORK/logs/eval_g${gpu}_$(echo "$m" | tr '/' '_').log" 2>&1 &
    pids+=($!); gpu=$(((gpu+1)%8))
    if [ ${#pids[@]} -ge 8 ]; then wait "${pids[@]}"; pids=(); fi
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
}
# eval ONLY the R-Zero ladder (used when DEO 7-set numbers are already in hand)
run_eval_rzero(){
  echo "=== [$(date '+%T')] eval R-Zero ladder on 7 sets ==="
  free_gpus
  cd "$ROOT/R-Zero"
  export STORAGE_PATH=$EVAL_ROOT PYTHONPATH=$ROOT/R-Zero VLLM_DISABLE_COMPILE_CACHE=1
  MODELS=("Qwen/Qwen3-4B-Base")
  for n in 1 2 3 4 5; do MODELS+=("$RZERO_STORAGE/models/qwen3-4b-base-rzero_solver_v$n/global_step_15/actor/huggingface"); done
  _eval_fanout "${MODELS[@]}"
  ORDER=("Qwen_Qwen3-4B-Base")
  for n in 1 2 3 4 5; do ORDER+=("$(echo "$RZERO_STORAGE/models/qwen3-4b-base-rzero_solver_v$n/global_step_15/actor/huggingface" | tr '/' '_')"); done
  python3 evaluation/aggregate_7sets.py --eval_root "$EVAL_ROOT/evaluation" --models "${ORDER[@]}" --out "$ROOT/RESULTS_7sets_rzero.md"
  cd "$ROOT"; sync_once
  echo "===== R-ZERO RESULTS ====="; cat "$ROOT/RESULTS_7sets_rzero.md"
}

# fast smoke: ONE R-Zero iter to confirm the questioner no longer NCCL-hangs and
# solver_v1 checkpoint is produced (~1.5-2h), before committing to the full run.
run_rzero_smoke(){
  RZERO_ITERS=1 run_rzero
  ck="$RZERO_STORAGE/models/qwen3-4b-base-rzero_solver_v1/global_step_15/actor/huggingface"
  if [ -d "$ck" ]; then echo "SMOKE_OK: solver_v1 checkpoint present at $ck"; ls "$ck" | head;
  else echo "SMOKE_FAIL: solver_v1 checkpoint MISSING ($ck)"; fi
}

case $MODE in
  deo)         run_deo ;;
  rzero)       run_rzero ;;
  eval)        run_eval ;;
  rzero_eval)  run_rzero; run_eval_rzero ;;
  rzero_smoke) run_rzero_smoke ;;
  canon_claudelabel) run_canon; run_eval_canon ;;
  canon_claude_smoke) DEO_NUM_ITERS=1 run_canon ;;
  curriculum) run_curriculum; run_eval_canon ;;
  full)        run_deo; run_rzero; run_eval ;;
esac
sync_once
echo "=== [$(date '+%F %T')] PIPELINE MODE=$MODE COMPLETE ==="

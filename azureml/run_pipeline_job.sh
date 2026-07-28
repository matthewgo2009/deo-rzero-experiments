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
# base model is env-overridable (e.g. BASE_MODEL=Qwen/Qwen3-8B-Base); DEO reads BASE_MODEL too
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B-Base}
export MODEL_NAME=$BASE_MODEL
RZERO_ABBR=${RZERO_ABBR:-qwen3-4b-base-rzero}   # R-Zero checkpoint prefix
BASE_UND=$(echo "$BASE_MODEL" | tr "/" "_")   # underscored base name for eval dir
export HF_HOME=/tmp/hf_cache HUGGINGFACE_HUB_CACHE=/tmp/hf_cache/hub
# Pre-fetch base weights to local cache BEFORE any vllm launch, so vllm does a pure
# (fast, deterministic) load instead of racing a 16GB download against the readiness
# timeout — the 8B first-load intermittently exceeded even 1800s otherwise.
echo "[prefetch] downloading $BASE_MODEL to cache..."
python3 -c "import os;from huggingface_hub import snapshot_download;snapshot_download(os.environ['BASE_MODEL'],max_workers=8)" 2>&1 | tail -2 || echo "[prefetch] warn: snapshot_download returned nonzero (vllm will fall back to its own download)"
echo "[prefetch] done"
export RZERO_DIR=$ROOT/R-Zero RZERO_CODE_DIR=$ROOT/R-Zero
export VLLM_PIDDIR=/tmp/vllm_pids VLLM_LOGDIR=/tmp/vllm_logs
DEO_STORAGE=$WORK/DEO; RZERO_STORAGE=$WORK/R-Zero_run; EVAL_ROOT=$WORK/eval7
mkdir -p "$DEO_STORAGE" "$RZERO_STORAGE" "$EVAL_ROOT/evaluation" "$WORK/logs" "$OUT"

# NOTE: the container has NO rsync (only cp). Every prior job's dir-sync used
# `rsync -a` and failed silently ("rsync: command not found"), so checkpoints and
# eval dirs never persisted — only cp'd files (final_results.jsonl) survived. All
# copying below therefore uses cp.
cpdir(){ [ -d "$1" ] || return 0; mkdir -p "$2"; cp -a "$1/." "$2/" 2>/dev/null; }

# ---- self-test persistence to OUT at startup; abort in <1 min if it fails ----
mkdir -p "$OUT/.persist_selftest/sub"; echo probe > "$OUT/.persist_selftest/sub/f.txt"
if [ "$(cat "$OUT/.persist_selftest/sub/f.txt" 2>/dev/null)" != "probe" ]; then
  echo "FATAL: cannot persist to OUT ($OUT) — aborting before wasting compute"; exit 1
fi
echo "[persist] self-test OK: dir+file persisted to OUT"

# ---- restore prior durable artifacts from the mount (resume) ----
# (skipped for MODE=regrade — it reads checkpoints straight from the mount, so no
#  need to copy ~80-100GB of checkpoints + FSDP shards to local disk.)
if [ "$MODE" != regrade ] && { [ -d "$OUT/DEO" ] || [ -d "$OUT/R-Zero_run" ]; }; then
  echo "=== restoring prior artifacts from $OUT -> $WORK ==="
  cpdir "$OUT/DEO" "$DEO_STORAGE"
  cpdir "$OUT/R-Zero_run" "$RZERO_STORAGE"
  cpdir "$OUT/eval7" "$EVAL_ROOT"
fi

# ---- durable persistence: WORK -> OUT (explicit cp; robust to dir depth) ----
# Copies EVERY merged HF checkpoint (find .../global_step_*/actor/huggingface at
# any depth), the full evaluation dirs (results_*.json keep per-problem responses,
# so any grader can be re-applied offline), datasets, generated questions, summaries,
# wallclock, logs.
persist_ckpts(){  # $1 = storage base, $2 = out subdir
  local base=$1 outb=$2
  [ -d "$base/models" ] || return 0
  find "$base/models" -type d -path "*/global_step_*/actor/huggingface" 2>/dev/null | while read -r hf; do
    cpdir "$hf" "$outb/${hf#$base/}"
  done
}
sync_once(){
  mkdir -p "$OUT/DEO" "$OUT/R-Zero_run" "$OUT/eval7" "$OUT/logs"
  cp -f "$DEO_STORAGE"/results_summary*.json "$OUT/DEO/" 2>/dev/null
  cp -f "$RZERO_STORAGE"/iter_wallclock.tsv "$OUT/R-Zero_run/" 2>/dev/null
  cp -f "$ROOT/R-Zero/final_results.jsonl" "$OUT/final_results.jsonl" 2>/dev/null
  for sub in datasets evaluation generated_question; do
    cpdir "$DEO_STORAGE/$sub" "$OUT/DEO/$sub"
    cpdir "$RZERO_STORAGE/$sub" "$OUT/R-Zero_run/$sub"
  done
  cpdir "$EVAL_ROOT" "$OUT/eval7"
  persist_ckpts "$DEO_STORAGE" "$OUT/DEO"
  persist_ckpts "$RZERO_STORAGE" "$OUT/R-Zero_run"
  cp -f "$WORK"/logs/* "$OUT/logs/" 2>/dev/null
}
persist_verify(){   # foreground final persist + report counts (proof it landed)
  echo "[persist] FINAL sync starting..."; sync_once; sync
  echo "[persist] OUT huggingface ckpt dirs: $(find "$OUT" -type d -name huggingface 2>/dev/null | wc -l)"
  echo "[persist] OUT results_*.json (with responses): $(find "$OUT" -name 'results_*.json' 2>/dev/null | wc -l)"
  find "$OUT" -type d -name huggingface 2>/dev/null | sed "s#$OUT/##" | head -20
  echo "[persist] FINAL sync done."
}
( while true; do sleep 300; sync_once; done ) &
SYNC_PID=$!
trap 'echo "[trap] final persist+verify"; kill $SYNC_PID 2>/dev/null; persist_verify' EXIT

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
# adaptive-temperature DEO: canonical walk + solver labeler + beta updated each
# iter to keep in-band r_unc fraction near delta (DEO paper §1.4)
run_adaptive(){
  echo "=== [$(date '+%T')] adaptive-temp DEO (auto-beta, ${DEO_NUM_ITERS:-5} iters) ==="
  export STORAGE_PATH=$DEO_STORAGE PYTHONPATH=$ROOT/R-Zero DEO_NUM_ITERS=${DEO_NUM_ITERS:-5}
  bash "$ROOT/DEO/start_vllm_native.sh"
  python3 "$ROOT/DEO/adaptive_temp_deo_native_main.py"
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
  MODELS=("$BASE_MODEL")
  for n in 1 2 3 4 5; do MODELS+=("$DEO_STORAGE/models/${abbr}_solver_v$n/global_step_15/actor/huggingface"); done
  _eval_fanout "${MODELS[@]}"
  ORDER=("$BASE_UND")
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
  bash scripts/run_rzero_gpu03.sh "$BASE_MODEL" "$RZERO_ABBR" "$iters"
  cd "$ROOT"; free_gpus; sync_once
}
run_eval(){
  echo "=== [$(date '+%T')] PHASE 3 eval 7 sets ==="
  cd "$ROOT/R-Zero"
  export STORAGE_PATH=$EVAL_ROOT PYTHONPATH=$ROOT/R-Zero VLLM_DISABLE_COMPILE_CACHE=1
  MODELS=("$BASE_MODEL")
  for n in 1 2 3 4 5; do MODELS+=("$DEO_STORAGE/models/deo_baseline_drift_solver_v$n/global_step_15/actor/huggingface"); done
  for n in 1 2 3 4 5; do MODELS+=("$RZERO_STORAGE/models/${RZERO_ABBR}_solver_v$n/global_step_15/actor/huggingface"); done
  _eval_fanout "${MODELS[@]}"
  ORDER=("$BASE_UND")
  for n in 1 2 3 4 5; do ORDER+=("$(echo "$DEO_STORAGE/models/deo_baseline_drift_solver_v$n/global_step_15/actor/huggingface" | tr '/' '_')"); done
  for n in 1 2 3 4 5; do ORDER+=("$(echo "$RZERO_STORAGE/models/${RZERO_ABBR}_solver_v$n/global_step_15/actor/huggingface" | tr '/' '_')"); done
  python3 evaluation/aggregate_7sets.py --eval_root "$EVAL_ROOT/evaluation" --models "${ORDER[@]}" --out "$ROOT/RESULTS_7sets.md"
  cd "$ROOT"; sync_once
  echo "===== RESULTS ====="; cat "$ROOT/RESULTS_7sets.md"
}
# fan eval_7sets across 8 GPUs for the given model list
_eval_fanout(){
  local gpu=0; local pids=()
  for m in "$@"; do
    if [ "$m" != "$BASE_MODEL" ] && [ ! -d "$m" ]; then echo "skip missing $m"; continue; fi
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
  MODELS=("$BASE_MODEL")
  for n in 1 2 3 4 5; do MODELS+=("$RZERO_STORAGE/models/${RZERO_ABBR}_solver_v$n/global_step_${RZ_S_GSTEP:-15}/actor/huggingface"); done
  _eval_fanout "${MODELS[@]}"
  ORDER=("$BASE_UND")
  for n in 1 2 3 4 5; do ORDER+=("$(echo "$RZERO_STORAGE/models/${RZERO_ABBR}_solver_v$n/global_step_${RZ_S_GSTEP:-15}/actor/huggingface" | tr '/' '_')"); done
  python3 evaluation/aggregate_7sets.py --eval_root "$EVAL_ROOT/evaluation" --models "${ORDER[@]}" --out "$ROOT/RESULTS_7sets_rzero.md"
  cd "$ROOT"; sync_once
  echo "===== R-ZERO RESULTS ====="; cat "$ROOT/RESULTS_7sets_rzero.md"
}

# REGRADE: fresh 7-set generation (raw math_verify) on the persisted R-Zero
# checkpoints, then DUAL grading (ours=gpt-4o-mini boxed | paper=gpt-4o full-text)
# without mutating raw scores. Answers "does R-Zero reach ~49 MATH AVG under the
# paper's lenient grader?". Mount an existing R-Zero run's OUT (checkpoints in
# $RZERO_STORAGE). RZ_S_GSTEP defaults to 15 (matches the full run).
run_regrade(){
  echo "=== [$(date '+%T')] REGRADE: fresh 7-set gen + dual grader ==="
  free_gpus
  cd "$ROOT/R-Zero"
  local RGROOT=$OUT/regrade   # write responses + compare straight to the mount
  mkdir -p "$RGROOT/evaluation"
  export STORAGE_PATH=$RGROOT PYTHONPATH=$ROOT/R-Zero VLLM_DISABLE_COMPILE_CACHE=1
  # read checkpoints straight from the mounted prior run (no local restore).
  # REGRADE_CKROOT: dir containing models/ (R-Zero_run default; DEO variants use $OUT/DEO).
  # REGRADE_ABBR:   solver checkpoint prefix (qwen3-4b-base-rzero | deo_curriculum | deo_adaptive | ...).
  local CKROOT=${REGRADE_CKROOT:-$OUT/${REGRADE_CKSUB:-R-Zero_run}}
  local ABBR=${REGRADE_ABBR:-qwen3-4b-base-rzero}
  local MODELS=("$BASE_MODEL")
  for n in 1 2 3 4 5; do MODELS+=("$CKROOT/models/${ABBR}_solver_v$n/global_step_${RZ_S_GSTEP:-15}/actor/huggingface"); done
  local DATASETS=(math gsm8k amc minerva olympiad aime2024 aime2025)
  # 1) fresh generation (raw), one model per GPU (generate.py only — NO recheck mutation)
  #    REGRADE_SKIP_GEN=1 -> responses already generated; just (re)grade them.
  if [ -z "${REGRADE_SKIP_GEN:-}" ]; then
    local gpu=0; local pids=()
    for m in "${MODELS[@]}"; do
      if [ "$m" != "$BASE_MODEL" ] && [ ! -d "$m" ]; then echo "REGRADE skip missing $m"; continue; fi
      ( for ds in "${DATASETS[@]}"; do
          CUDA_VISIBLE_DEVICES=$gpu python evaluation/generate.py --model "$m" --dataset "$ds"
        done ) > "$WORK/logs/regen_g${gpu}_$(echo "$m" | tr '/' '_').log" 2>&1 &
      pids+=($!); gpu=$(((gpu+1)%8))
      if [ ${#pids[@]} -ge 8 ]; then wait "${pids[@]}"; pids=(); fi
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
    sync_once
  else
    echo "[regrade] REGRADE_SKIP_GEN set — grading existing responses in $RGROOT/evaluation"
  fi
  # 2) dual grade (CPU + OpenAI API only) — iterate the ACTUAL response dirs present
  #    (generate.py names dirs by the full checkpoint path, which carries a per-job id,
  #    so glob what exists instead of reconstructing paths).
  rm -f "$ROOT/R-Zero/regrade_compare.jsonl"
  for d in "$RGROOT"/evaluation/*/; do
    [ -d "$d" ] || continue
    dn=$(basename "$d"); lab="$dn"
    case "$dn" in
      *solver_v1*) lab=solver_v1;; *solver_v2*) lab=solver_v2;; *solver_v3*) lab=solver_v3;;
      *solver_v4*) lab=solver_v4;; *solver_v5*) lab=solver_v5;; *Qwen3-*Base*) lab=base;;
    esac
    python3 evaluation/dual_grade.py --eval_dir "$d" --label "$lab" --workers 16 --grader "${REGRADE_GRADER:-openai}"
  done
  cp -f "$ROOT/R-Zero/regrade_compare.jsonl" "$RGROOT/regrade_compare.jsonl" 2>/dev/null || true
  echo "===== REGRADE COMPARE (raw | ours=4o-mini boxed | paper=4o full-text) ====="
  cat "$RGROOT/regrade_compare.jsonl"
  cd "$ROOT"; sync_once
}

# fast smoke: ONE R-Zero iter to confirm the questioner no longer NCCL-hangs and
# solver_v1 checkpoint is produced (~1.5-2h), before committing to the full run.
run_rzero_smoke(){
  RZERO_ITERS=1 run_rzero
  ck="$RZERO_STORAGE/models/${RZERO_ABBR}_solver_v1/global_step_${RZ_S_GSTEP:-15}/actor/huggingface"
  if [ -d "$ck" ]; then echo "SMOKE_OK: solver_v1 checkpoint present at $ck"; ls "$ck" | head;
  else echo "SMOKE_FAIL: solver_v1 checkpoint MISSING ($ck)"; fi
}

case $MODE in
  deo)         run_deo ;;
  rzero)       run_rzero ;;
  eval)        run_eval ;;
  rzero_eval)  run_rzero; run_eval_rzero ;;
  regrade)     run_regrade ;;
  rzero_smoke) run_rzero_smoke ;;
  canon_claudelabel) run_canon; run_eval_canon ;;
  canon_claude_smoke) DEO_NUM_ITERS=1 run_canon ;;
  curriculum) run_curriculum; run_eval_canon ;;
  adaptive) run_adaptive; run_eval_canon ;;
  full)        run_deo; run_rzero; run_eval ;;
esac
sync_once
echo "=== [$(date '+%F %T')] PIPELINE MODE=$MODE COMPLETE ==="

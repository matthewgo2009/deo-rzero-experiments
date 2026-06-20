#!/bin/bash
# Controlled R-Zero orchestrator for the GPU-0-3 (2+2) fair comparison vs DEO.
# Mirrors scripts/main.sh but: parameterized iteration count, per-stage logging,
# evals the base model first (iter 0), and uses our MATH-500 + GPT-mini grader
# (via the edited evaluation/evaluate.bash).
#
# Usage:
#   bash scripts/run_rzero_gpu03.sh <Base_Model> <Model_abbr> <MAX_ITERS>
# Example (smoke = 1 iter):
#   bash scripts/run_rzero_gpu03.sh Qwen/Qwen3-4B-Base qwen3-4b-base-rzero 1
set -u
cd "$(dirname "$0")/.."
source ./run_env.sh

Base_model=${1:-Qwen/Qwen3-4B-Base}
Model_abbr=${2:-qwen3-4b-base-rzero}
MAX_ITERS=${3:-5}
START_ITER=${START_ITER:-1}   # set >1 to resume, reusing already-trained earlier iters
LOG=${STORAGE_PATH}/logs/run_${Model_abbr}_$(date +%Y%m%d_%H%M%S).log
mkdir -p "${STORAGE_PATH}/logs"

echo "=== R-Zero GPU-0-3 run | base=$Base_model abbr=$Model_abbr iters=${START_ITER}..${MAX_ITERS} ===" | tee -a "$LOG"
echo "log: $LOG"

run() { echo "[$(date '+%F %T')] >>> $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

if [ "${START_ITER}" -le 1 ]; then
    # ---- iter 0: base model MATH-500 eval (our grader) ----
    run bash evaluation/evaluate.bash "$Base_model"
    # ---- iter 1: challenger from base, solver from base ----
    run bash scripts/questioner_train_penalty.sh "$Base_model" "$Base_model" ${Model_abbr}_questioner_v1
    run bash scripts/solver_train.sh "$Base_model" ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface ${Model_abbr}_solver_v1
    START_ITER=2
fi

# ---- iters START_ITER..MAX_ITERS ----
for i in $(seq ${START_ITER} ${MAX_ITERS}); do
    prev=$((i-1))
    run bash scripts/questioner_train_penalty.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_5/actor/huggingface \
        ${Model_abbr}_questioner_v${i}
    run bash scripts/solver_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/huggingface \
        ${Model_abbr}_solver_v${i}
done

echo "[$(date '+%F %T')] === run complete ===" | tee -a "$LOG"

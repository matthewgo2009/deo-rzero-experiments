#!/bin/bash
# Controlled R-Zero orchestrator (fair comparison vs DEO).
# Mirrors scripts/main.sh but: parameterized iteration count, per-stage logging,
# evals the base model first (iter 0), uses our MATH-500 + GPT-mini grader
# (via evaluation/evaluate.bash), and RECORDS per-iteration wall-clock time.
#
# Usage:
#   bash scripts/run_rzero_gpu03.sh <Base_Model> <Model_abbr> <MAX_ITERS>
set -u
cd "$(dirname "$0")/.."
source ./run_env.sh

Base_model=${1:-Qwen/Qwen3-4B-Base}
Model_abbr=${2:-qwen3-4b-base-rzero}
MAX_ITERS=${3:-5}
START_ITER=${START_ITER:-1}   # set >1 to resume, reusing already-trained earlier iters
LOG=${STORAGE_PATH}/logs/run_${Model_abbr}_$(date +%Y%m%d_%H%M%S).log
WCLOG=${STORAGE_PATH}/iter_wallclock.tsv
mkdir -p "${STORAGE_PATH}/logs"
[ -f "$WCLOG" ] || printf "iter\tphase\tseconds\tminutes\tstart\tend\n" > "$WCLOG"

echo "=== R-Zero run | base=$Base_model abbr=$Model_abbr iters=${START_ITER}..${MAX_ITERS} ===" | tee -a "$LOG"
echo "log: $LOG  wallclock: $WCLOG"

run() { echo "[$(date '+%F %T')] >>> $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

# time a stage: rec <iter> <phase> <cmd...>  -> runs cmd, appends wall-clock to WCLOG + LOG
rec() {
    local it=$1 ph=$2; shift 2
    local t0 t1 s
    t0=$(date +%s); local start; start=$(date '+%F %T')
    run "$@"
    t1=$(date +%s); s=$((t1 - t0))
    printf "%s\t%s\t%d\t%.1f\t%s\t%s\n" "$it" "$ph" "$s" "$(awk "BEGIN{print $s/60}")" "$start" "$(date '+%F %T')" | tee -a "$WCLOG" | tee -a "$LOG"
    echo "[WALLCLOCK] iter=$it $ph = ${s}s ($(awk "BEGIN{printf \"%.1f\",$s/60}") min)" | tee -a "$LOG"
}

iter_total_start=""
iter_total() { # print iter total = sum of its phases from WCLOG
    local it=$1
    awk -F'\t' -v it="$it" 'NR>1 && $1==it {tot+=$3} END{printf "[WALLCLOCK] iter=%s TOTAL = %ds (%.1f min)\n", it, tot, tot/60}' "$WCLOG" | tee -a "$LOG"
}

if [ "${START_ITER}" -le 1 ]; then
    run bash evaluation/evaluate.bash "$Base_model"     # iter 0 baseline eval
    rec 1 questioner bash scripts/questioner_train_penalty.sh "$Base_model" "$Base_model" ${Model_abbr}_questioner_v1
    rec 1 solver     bash scripts/solver_train.sh "$Base_model" ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface ${Model_abbr}_solver_v1
    iter_total 1
    START_ITER=2
fi

for i in $(seq ${START_ITER} ${MAX_ITERS}); do
    prev=$((i-1))
    rec $i questioner bash scripts/questioner_train_penalty.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_5/actor/huggingface \
        ${Model_abbr}_questioner_v${i}
    rec $i solver bash scripts/solver_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/huggingface \
        ${Model_abbr}_solver_v${i}
    iter_total $i
done

echo "[$(date '+%F %T')] === run complete ===" | tee -a "$LOG"
echo "=== per-iteration wall-clock (${WCLOG}) ==="; cat "$WCLOG" | tee -a "$LOG"

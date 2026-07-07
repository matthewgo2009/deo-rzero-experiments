#!/bin/bash
# Evaluate ONE model on all 7 R-Zero math benchmarks, graded with our fair-compare
# grader: math_verify primary (generate.py) + gpt-4o-mini boxed-only recheck.
#
# Usage: bash evaluation/eval_7sets.bash <model_path_or_name> [gpu_id]
# Output: $STORAGE_PATH/evaluation/<model_/>/results_<dataset>.json (per dataset)
#         one line per (model,dataset) appended to ./final_results.jsonl
set -u
export VLLM_DISABLE_COMPILE_CACHE=1
model_name=$1
gpu=${2:-0}
DATASETS=(math gsm8k amc minerva olympiad aime2024 aime2025)

echo "=== [$(date '+%F %T')] eval_7sets model=${model_name} gpu=${gpu} ==="
for ds in "${DATASETS[@]}"; do
    echo "--> [$(date '+%T')] generate ${ds}"
    CUDA_VISIBLE_DEVICES=${gpu} python evaluation/generate.py --model "${model_name}" --dataset "${ds}"
    echo "--> [$(date '+%T')] gpt-mini recheck ${ds}"
    python evaluation/recheck_mini.py --model_name "${model_name}" --dataset "${ds}" --workers 16
done
echo "=== [$(date '+%F %T')] eval_7sets DONE model=${model_name} ==="

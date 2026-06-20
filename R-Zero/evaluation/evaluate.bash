#!/bin/bash
# GPU-0-3 fair-comparison eval: MATH-500 ONLY (matches the DEO paper_data eval set),
# graded with OUR grader = math_verify primary (generate.py) + GPT-4o-mini boxed-only
# recheck (results_recheck_math500_mini.py). All other R-Zero benchmarks
# (gsm8k/amc/minerva/olympiad/aime/supergpqa/bbeh/mmlupro) are intentionally skipped —
# DEO has no counterpart data and MATH-500 is the agreed comparison basis.
export VLLM_DISABLE_COMPILE_CACHE=1
model_name=$1

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] MATH-500 generate on GPU 0 for model [${model_name}]"
CUDA_VISIBLE_DEVICES=0 python evaluation/generate.py --model "${model_name}" --dataset "math"

echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPT-4o-mini boxed-only recheck (our grader)"
python evaluation/results_recheck_math500_mini.py --model_name "${model_name}" --workers 12

echo "==> MATH-500 eval + recheck finished for [${model_name}]"

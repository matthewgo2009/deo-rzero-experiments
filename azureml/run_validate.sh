#!/bin/bash
# Validation job: run ONE real DEO baseline_drift iteration end-to-end inside the
# AzureML container, to prove the native path works (env build, 8 GPUs, native
# vllm serve, MCMC gen, HF push, verl GRPO w/ KL-pin, global_step_15 merge,
# MATH-500 eval + gpt-mini recheck) before committing the multi-day full run.
# Trains on fast local /tmp; copies the durable artifacts to the output mount.
#
# Invoked by job_validate.yaml as:  bash azureml/run_validate.sh ${{outputs.work}}
set -u
OUT=${1:-/tmp/out}
ROOT=$PWD
echo "=== [$(date '+%F %T')] VALIDATE start | OUT=$OUT ROOT=$ROOT ==="
nvidia-smi -L
python3 -c "import torch,vllm,flash_attn;print('torch',torch.__version__,'vllm',vllm.__version__,'cuda',torch.cuda.is_available(),'ngpu',torch.cuda.device_count())"

# tokens.json (read by DEO HF push + generate.py + gpt recheck + verl), from job env vars.
# include a wandb placeholder (verl/trainer/main.py reads it); WANDB_MODE=disabled keeps it offline.
export WANDB_MODE=disabled
python3 -c "import json,os;open('$ROOT/R-Zero/tokens.json','w').write(json.dumps({'huggingface':os.environ['HF_TOKEN'],'openai':os.environ['OPENAI_API_KEY'],'wandb':os.environ.get('WANDB_API_KEY','')}))"

export HUGGINGFACENAME=${HUGGINGFACENAME:-yuyang322}
export MODEL_NAME=Qwen/Qwen3-4B-Base
export HF_HOME=/tmp/hf_cache HUGGINGFACE_HUB_CACHE=/tmp/hf_cache/hub
export RZERO_DIR=$ROOT/R-Zero
export PYTHONPATH=$ROOT/R-Zero
export VLLM_PIDDIR=/tmp/vllm_pids VLLM_LOGDIR=/tmp/vllm_logs
export STORAGE_PATH=/tmp/work/DEO            # fast local scratch
mkdir -p "$STORAGE_PATH" "$OUT"

# one real iteration (defaults: TOTAL_QUESTIONS=1500, verl max_steps=20 -> merge global_step_15)
export DEO_NUM_ITERS=1

echo "=== launching 4 native vLLM servers (base GPU0, solver DP GPU1/4/5) ==="
bash "$ROOT/DEO/start_vllm_native.sh"

echo "=== running DEO baseline_drift native (1 iter) ==="
python3 "$ROOT/DEO/baseline_drift_native_main.py"
rc=$?
echo "=== DEO native rc=$rc ==="

echo "=== copying durable artifacts to output mount ==="
mkdir -p "$OUT/DEO"
cp -f "$STORAGE_PATH/results_summary_baseline_drift.json" "$OUT/DEO/" 2>/dev/null
rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='*/global_step_15/actor/huggingface/***' \
  --include='datasets/***' --include='evaluation/***' --include='logs/***' \
  --exclude='*' "$STORAGE_PATH/" "$OUT/DEO/" 2>/dev/null
echo "=== artifacts in OUT/DEO ==="; ls -R "$OUT/DEO" | head -40
echo "=== summary ==="; cat "$STORAGE_PATH/results_summary_baseline_drift.json" 2>/dev/null
pkill -9 -f "vllm serve" 2>/dev/null || true
echo "=== [$(date '+%F %T')] VALIDATE done rc=$rc ==="
exit $rc

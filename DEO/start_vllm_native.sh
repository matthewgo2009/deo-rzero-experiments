#!/bin/bash
# Launch the 4 DEO vLLM servers as NATIVE background processes (no docker), for
# running inside an AzureML Command Job container (8xH100, single node).
#   vllm_base   GPU 0  port 8000  Qwen3-4B-Base (questioner / mutator)
#   vllm_solver GPU 1  port 8001  current solver (r_unc scorer, DP shard 1)
#   vllm_solver GPU 4  port 8004  current solver (r_unc scorer, DP shard 2)
#   vllm_solver GPU 5  port 8005  current solver (r_unc scorer, DP shard 3)
# verl trains on GPU 2,3; eval runs on GPU 6 (see baseline_drift_native_main.py).
# PIDs are written to $PIDDIR/<port>.pid so reload_vllm_solver can restart shards.
set -u
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Base}
PIDDIR=${VLLM_PIDDIR:-/tmp/vllm_pids}
LOGDIR=${VLLM_LOGDIR:-/tmp/vllm_logs}
mkdir -p "$PIDDIR" "$LOGDIR"

launch() {  # name gpu port model
  local name=$1 gpu=$2 port=$3 model=$4
  echo "[start_vllm_native] $name gpu=$gpu port=$port model=$model"
  CUDA_VISIBLE_DEVICES=$gpu VLLM_DISABLE_COMPILE_CACHE=1 \
    vllm serve "$model" \
      --served-model-name "$MODEL_NAME" \
      --dtype bfloat16 --max-model-len 6144 --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.85 --port "$port" \
      > "$LOGDIR/${name}_${port}.log" 2>&1 &
  echo $! > "$PIDDIR/${port}.pid"
}

launch vllm_base   0 8000 "$MODEL_NAME"
launch vllm_solver 1 8001 "$MODEL_NAME"
launch vllm_solver 4 8004 "$MODEL_NAME"
launch vllm_solver 5 8005 "$MODEL_NAME"
echo "[start_vllm_native] launched 4 servers; pids in $PIDDIR"

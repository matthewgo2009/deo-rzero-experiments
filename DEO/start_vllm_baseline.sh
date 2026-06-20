#!/bin/bash
# BASELINE ABLATION (no MCMC walk) — runs in parallel with main MCMC fair-compare.
#   - vllm_base_baseline   on GPU 4, port 8002: Qwen3-4B-Base (questioner only, no mutator)
#   - vllm_solver_baseline on GPU 5, port 8003: current solver (reloaded between iters)

set -e

HF_CACHE="/eph/nvme0/yyd/hf_cache"
DEO_STORAGE="/eph/nvme0/yyd/DEO_baseline"
RZERO_STORAGE="/eph/nvme0/yyd/R-Zero"
MODEL_NAME="Qwen/Qwen3-4B-Base"

mkdir -p "$HF_CACHE" "$DEO_STORAGE/models" "$DEO_STORAGE/logs"

# Tear down any previous baseline containers
sudo docker stop vllm_base_baseline vllm_solver_baseline 2>/dev/null || true
sudo docker rm   vllm_base_baseline vllm_solver_baseline 2>/dev/null || true

# --- vllm_base_baseline: GPU 4, port 8002 ---
sudo docker run -d \
    --name vllm_base_baseline \
    --gpus '"device=4"' \
    --network host \
    --shm-size 16g \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    vllm/vllm-openai:latest \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --dtype bfloat16 \
    --max-model-len 6144 \
    --tensor-parallel-size 1 \
    --port 8002

# --- vllm_solver_baseline: GPU 5, port 8003 ---
sudo docker run -d \
    --name vllm_solver_baseline \
    --gpus '"device=5"' \
    --network host \
    --shm-size 16g \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${DEO_STORAGE}:/storage" \
    -v "${RZERO_STORAGE}:/storage_rzero" \
    vllm/vllm-openai:latest \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --dtype bfloat16 \
    --max-model-len 6144 \
    --tensor-parallel-size 1 \
    --port 8003

echo ""
echo "vllm_base_baseline   on GPU 4 -> http://localhost:8002"
echo "vllm_solver_baseline on GPU 5 -> http://localhost:8003"
echo ""
echo "Wait ~60-120s for both to load."
echo "Stop:    sudo docker stop vllm_base_baseline vllm_solver_baseline && sudo docker rm vllm_base_baseline vllm_solver_baseline"

#!/bin/bash
# DEO baseline (no MCMC walk) — Docker, GPU 0-3, fair vs the R-Zero GPU-0-3 rerun.
# Layout:
#   host GPU 0  vllm_base_baseline    (Qwen3-4B-Base, port 8002)  question proposals
#   host GPU 1  vllm_solver_baseline  (current solver, port 8003) r_unc M-vote scorer (reloaded each iter)
#   host GPU 2,3 deo_runner_baseline  (deo-rzero:latest)          verl KL-pin training + eval
#
# Usage: sudo bash run_baseline_docker_gpu03.sh [NUM_ITERATIONS]
set -e
NUM_ITERATIONS=${1:-5}

HF_CACHE=/eph/nvme0/yyd/hf_cache
DEO_STORAGE=/eph/nvme0/yyd/DEO_baseline
RZERO=/home/azureuser/yyd/R-Zero
DEO=/home/azureuser/yyd/DEO
MODEL=Qwen/Qwen3-4B-Base

mkdir -p "$HF_CACHE" "$DEO_STORAGE"/{models,logs,datasets,evaluation,generated_question,temp_results}

echo "=== tearing down any prior baseline containers ==="
docker rm -f vllm_base_baseline vllm_solver_baseline deo_runner_baseline 2>/dev/null || true

echo "=== start vllm_base_baseline (GPU0:8002) + vllm_solver_baseline (GPU1:8003) ==="
docker run -d --name vllm_base_baseline --gpus '"device=0"' --network host --shm-size 16g \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    vllm/vllm-openai:latest --model "$MODEL" --served-model-name "$MODEL" \
    --dtype bfloat16 --max-model-len 6144 --tensor-parallel-size 1 --port 8002

docker run -d --name vllm_solver_baseline --gpus '"device=1"' --network host --shm-size 16g \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${DEO_STORAGE}:/storage" \
    vllm/vllm-openai:latest --model "$MODEL" --served-model-name "$MODEL" \
    --dtype bfloat16 --max-model-len 6144 --tensor-parallel-size 1 --port 8003

echo "=== waiting for both vllm endpoints ==="
until curl -fsS http://localhost:8002/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8003/v1/models >/dev/null 2>&1; do sleep 10; done
echo "ALL VLLM READY"

echo "=== launch deo_runner_baseline (GPU2,3) running baseline_no_mcmc.py, NUM_ITERATIONS=$NUM_ITERATIONS ==="
docker run -d --name deo_runner_baseline --gpus '"device=2,3"' --network host --shm-size 32g \
    -e STORAGE_PATH=/storage -e HUGGINGFACENAME=yuyang322 -e HF_HOME=/root/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub -e PYTHONPATH=/workspace \
    -e NUM_ITERATIONS=$NUM_ITERATIONS -e WANDB_MODE=disabled -e VLLM_DISABLE_COMPILE_CACHE=1 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${DEO_STORAGE}:/storage" \
    -v "${RZERO}:/workspace" \
    -v "${DEO}:/deo" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -w /workspace \
    deo-rzero:latest \
    python /deo/baseline_no_mcmc.py

echo "deo_runner_baseline started. Follow: docker logs -f deo_runner_baseline"

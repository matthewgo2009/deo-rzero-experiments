#!/bin/bash
# Launch vllm endpoints for DEO:
#   - vllm_base        on GPU 0, port 8000: Qwen3-4B-Base (questioner/mutator)
#   - vllm_solver      on GPU 1, port 8001: current solver instance #1 (r_unc scorer)
#   - vllm_solver_dp1  on GPU 4, port 8004: current solver instance #2 (r_unc scorer, DP)
#   - vllm_solver_dp2  on GPU 5, port 8005: current solver instance #3 (r_unc scorer, DP)
# The three solver containers run data-parallel; mcmc_deo_vllm.py shards r_unc prompts
# across them via SOLVER_INSTANCES in Config. All three reload together between iters.
# Initial solver = base, so all start with the same model.

set -e

HF_CACHE="/eph/nvme0/yyd/hf_cache"
DEO_STORAGE="/eph/nvme0/yyd/DEO"
RZERO_STORAGE="/eph/nvme0/yyd/R-Zero"
MODEL_NAME="Qwen/Qwen3-4B-Base"

mkdir -p "$HF_CACHE" "$DEO_STORAGE/models" "$DEO_STORAGE/logs"

# Tear down any previous containers from earlier runs
sudo docker stop vllm_base vllm_solver vllm_solver_dp1 vllm_solver_dp2 2>/dev/null || true
sudo docker rm   vllm_base vllm_solver vllm_solver_dp1 vllm_solver_dp2 2>/dev/null || true

# --- vllm_base: GPU 0, port 8000 ---
sudo docker run -d \
    --name vllm_base \
    --gpus '"device=0"' \
    --network host \
    --shm-size 16g \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    vllm/vllm-openai:v0.9.1 \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --dtype bfloat16 \
    --max-model-len 6144 \
    --tensor-parallel-size 1 \
    --port 8000

# --- vllm_solver instances (DP=3): GPUs 1, 4, 5  /  ports 8001, 8004, 8005 ---
# Each is identical; mcmc_deo_vllm.py round-robins r_unc prompts across them.
launch_solver() {
    local NAME=$1 GPU=$2 PORT=$3
    sudo docker run -d \
        --name "${NAME}" \
        --gpus '"device='"${GPU}"'"' \
        --network host \
        --shm-size 16g \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${DEO_STORAGE}:/storage" \
        -v "${RZERO_STORAGE}:/storage_rzero" \
        vllm/vllm-openai:v0.9.1 \
        --model "${MODEL_NAME}" \
        --served-model-name "${MODEL_NAME}" \
        --dtype bfloat16 \
        --max-model-len 6144 \
        --tensor-parallel-size 1 \
        --port "${PORT}"
}

launch_solver vllm_solver     1 8001
launch_solver vllm_solver_dp1 4 8004
launch_solver vllm_solver_dp2 5 8005

echo ""
echo "vllm_base        on GPU 0 -> http://localhost:8000  (Qwen3-4B-Base, always)"
echo "vllm_solver      on GPU 1 -> http://localhost:8001  (solver DP shard 1)"
echo "vllm_solver_dp1  on GPU 4 -> http://localhost:8004  (solver DP shard 2)"
echo "vllm_solver_dp2  on GPU 5 -> http://localhost:8005  (solver DP shard 3)"
echo ""
echo "Wait ~60-120s for all to load the model."
echo "Logs:    sudo docker logs -f vllm_base | vllm_solver | vllm_solver_dp1 | vllm_solver_dp2"
echo "Stop:    sudo docker stop vllm_base vllm_solver vllm_solver_dp1 vllm_solver_dp2 \\"
echo "                  && sudo docker rm vllm_base vllm_solver vllm_solver_dp1 vllm_solver_dp2"

#!/bin/bash
# Launch base_only_pseudo ablation: pseudo-label always from base model.
# Minimal infra — only vllm_base needed (no solver DP).

set -e

SESSION="deo_bp"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO"
ARCHIVE="/eph/nvme0/yyd/DEO_archived_20260515_204605"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

mkdir -p "$DEO_STORAGE"/{models,evaluation,datasets,logs,generated_question,temp_results}

# Copy canonical solver_v1 as starting ckpt (same as other ablations).
if [ ! -d "$DEO_STORAGE/models/deo_base_pseudo_solver_v1" ]; then
    echo "Copying solver_v1 ckpt from canonical archive..."
    sudo cp -r "$ARCHIVE/models/deo_qwen3_4b_base_solver_v1" \
        "$DEO_STORAGE/models/deo_base_pseudo_solver_v1"
    sudo chown -R azureuser:azureuser "$DEO_STORAGE/models/deo_base_pseudo_solver_v1"
fi

# Bust HF cache.
sudo rm -rf "$HF_CACHE"/datasets/yuyang322___deo_base_pseudo* \
            "$HF_CACHE"/datasets/_*yuyang322___deo_base_pseudo*.lock 2>/dev/null || true

# Start ONLY vllm_base on GPU 0. No solver containers needed.
echo "Starting vllm_base on GPU 0..."
sudo docker stop vllm_base 2>/dev/null || true
sudo docker rm   vllm_base 2>/dev/null || true
sudo docker run -d \
    --name vllm_base \
    --gpus '"device=0"' \
    --network host \
    --shm-size 16g \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    vllm/vllm-openai:latest \
    --model "Qwen/Qwen3-4B-Base" \
    --served-model-name "Qwen/Qwen3-4B-Base" \
    --dtype bfloat16 \
    --max-model-len 6144 \
    --tensor-parallel-size 1 \
    --port 8000

echo "Waiting for vllm_base on port 8000..."
until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 10; done
echo "vllm_base ready."

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_bp_runner \
    --gpus '\"device=2,3\"' \
    --network host \
    --shm-size 32g \
    -v ${DEO_CODE}:/workspace_deo \
    -v ${RZERO_CODE}:/workspace \
    -v ${DEO_STORAGE}:/storage \
    -v /eph/nvme0/yyd/R-Zero:/storage_rzero \
    -v ${HF_CACHE}:/root/.cache/huggingface \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker:ro \
    -e PYTHONPATH=/workspace \
    -e STORAGE_PATH=/storage \
    -e HF_HOME=/root/.cache/huggingface \
    -e HUGGINGFACENAME=yuyang322 \
    -e WANDB_MODE=disabled \
    -w /workspace_deo \
    --entrypoint python3 \
    rzero:latest \
    base_only_pseudo_main.py 2>&1 | tee ${DEO_STORAGE}/logs/run_base_pseudo_\$(date +%Y%m%d_%H%M%S).log" Enter

echo ""
echo "base_only_pseudo running in tmux session '${SESSION}'"
echo "  Results: ${DEO_STORAGE}/results_summary_base_pseudo.json"

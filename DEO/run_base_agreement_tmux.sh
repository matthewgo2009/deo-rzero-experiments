#!/bin/bash
# Launch base-agreement ablation in tmux.
# Each iter (>=2) adds an extra filter: base-model M-vote majority must
# agree with the current solver's M-vote majority before a question enters
# training set. Tests whether base-concurrence prevents per-iter solver drift.

set -e

SESSION="deo_ba"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO"
ARCHIVE="/eph/nvme0/yyd/DEO_archived_20260515_204605"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

mkdir -p "$DEO_STORAGE"/{models,evaluation,datasets,logs,generated_question,temp_results}

# Copy canonical solver_v1 ckpt to live storage under our ablation's prefix.
if [ ! -d "$DEO_STORAGE/models/deo_base_agreement_solver_v1" ]; then
    echo "Copying solver_v1 ckpt from canonical archive..."
    sudo cp -r "$ARCHIVE/models/deo_qwen3_4b_base_solver_v1" \
        "$DEO_STORAGE/models/deo_base_agreement_solver_v1"
    sudo chown -R azureuser:azureuser "$DEO_STORAGE/models/deo_base_agreement_solver_v1"
fi

# Bust HF cache for our new datasets to prevent stale shadow.
sudo rm -rf "$HF_CACHE"/datasets/yuyang322___deo_base_agreement* \
            "$HF_CACHE"/datasets/_*yuyang322___deo_base_agreement*.lock 2>/dev/null || true

# Start canonical 4-vllm layout: base on GPU 0, solver DP×3 on GPU 1/4/5.
# Initially all loaded with Qwen3-4B-Base; main script reloads solvers to v1.
echo "Starting vllm endpoints (base + 3 solver DP)..."
bash "$DEO_CODE/start_vllm.sh"

echo "Waiting for vllm endpoints..."
until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8001/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8004/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8005/v1/models >/dev/null 2>&1; do
    sleep 15
done
echo "All vllm endpoints ready."

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Run base_agreement_main.py inside rzero docker on GPU 2,3 (for verl).
tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_ba_runner \
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
    base_agreement_main.py 2>&1 | tee ${DEO_STORAGE}/logs/run_base_agreement_\$(date +%Y%m%d_%H%M%S).log" Enter

echo ""
echo "Base-agreement ablation running in tmux session '${SESSION}'"
echo ""
echo "  tmux attach -t ${SESSION}        # view live"
echo "  Ctrl+B, D                        # detach"
echo "  tmux kill-session -t ${SESSION}  # stop"
echo ""
echo "Results: ${DEO_STORAGE}/results_summary_base_agreement.json"

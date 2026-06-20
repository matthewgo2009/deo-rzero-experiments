#!/bin/bash
# Launch canonical + conf filter ablation in tmux.
# Reuses already-running vllm containers (base + 3 solver DP).
# IMPORTANT: vllm_solver currently has baseline_klfix's solver_v1 loaded;
# canonical pipeline expects base initially, so reload first.

set -e

SESSION="deo_cf"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

mkdir -p "$DEO_STORAGE"/{models,evaluation,datasets,logs,generated_question,temp_results}

# Restart vllm endpoints so solver containers start with base (canonical iter 1 needs base).
echo "Restarting vllm endpoints (solvers need to start with base for iter 1)..."
bash "$DEO_CODE/start_vllm.sh"
echo "Waiting for endpoints..."
until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8001/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8004/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8005/v1/models >/dev/null 2>&1; do
    sleep 15
done
echo "All endpoints ready."

# Bust HF cache for new dataset names
sudo rm -rf "$HF_CACHE"/datasets/yuyang322___deo_conf_filter* \
            "$HF_CACHE"/datasets/_*yuyang322___deo_conf_filter*.lock 2>/dev/null || true

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_cf_runner \
    --gpus '\"device=2,3\"' \
    --network host --shm-size 32g \
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
    -w /workspace_deo --entrypoint python3 \
    rzero:latest \
    conf_filter_canonical_main.py 2>&1 | tee ${DEO_STORAGE}/logs/run_conf_filter_\$(date +%Y%m%d_%H%M%S).log" Enter

echo ""
echo "Conf-filter canonical ablation running in tmux session '${SESSION}'"
echo "  tmux attach -t ${SESSION}"
echo "Results: ${DEO_STORAGE}/results_summary_conf_filter.json"

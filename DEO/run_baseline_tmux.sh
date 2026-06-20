#!/bin/bash
# BASELINE ABLATION (no MCMC walk) — runs in parallel with main fair-compare on GPUs 4-7.
# tmux session: deo_baseline.  docker container: deo_runner_baseline.
# Prerequisites:
#   bash /home/azureuser/yyd/DEO/start_vllm_baseline.sh  (vllm on GPU 4,5 / ports 8002,8003)

set -e

SESSION="deo_baseline"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO_baseline"
RZERO_STORAGE="/eph/nvme0/yyd/R-Zero"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

for port in 8002 8003; do
    if ! curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
        echo "WARN: baseline vllm on port ${port} not responding (will block until ready)."
    fi
done

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Baseline runs on GPU 6,7 (vllm_base_baseline owns 4, vllm_solver_baseline owns 5).
tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_runner_baseline \
    --gpus '\"device=6,7\"' \
    --network host \
    --shm-size 32g \
    -v ${DEO_CODE}:/workspace_deo \
    -v ${RZERO_CODE}:/workspace \
    -v ${DEO_STORAGE}:/storage \
    -v ${RZERO_STORAGE}:/storage_rzero \
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
    baseline_no_mcmc.py 2>&1 | tee ${DEO_STORAGE}/logs/run_\$(date +%Y%m%d_%H%M%S).log" Enter

echo ""
echo "BASELINE running in tmux session '${SESSION}'"
echo "  tmux attach -t ${SESSION}        # view live"
echo "  Ctrl+B, D                        # detach"
echo "  tmux kill-session -t ${SESSION}  # stop"
echo ""
echo "Logs: ${DEO_STORAGE}/logs/run_<timestamp>.log"

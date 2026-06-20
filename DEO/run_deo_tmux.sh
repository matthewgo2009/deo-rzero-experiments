#!/bin/bash
# Launch DEO + R-Zero verl pipeline inside an rzero:latest container, in tmux.
#
# Prerequisites (run by hand once):
#   1. tokens.json populated at /home/azureuser/yyd/R-Zero/tokens.json (HF token)
#   2. bash /home/azureuser/yyd/DEO/start_vllm.sh   (starts vllm_base + vllm_solver)
#      -> wait ~60-120s for both to load (check `sudo docker logs vllm_base`)
#
# Usage: bash run_deo_tmux.sh

set -e

SESSION="deo"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO"
RZERO_STORAGE="/eph/nvme0/yyd/R-Zero"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

# Sanity: vllm endpoints reachable?
for port in 8000 8001; do
    if ! curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
        echo "WARN: vllm endpoint on port ${port} not responding."
        echo "      Run 'bash ${DEO_CODE}/start_vllm.sh' first and wait for both to load."
        echo "      DEO will block waiting for them, but you may want to verify before launching."
    fi
done

# Kill stale session
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -x 220 -y 50

# Run DEO inside rzero:latest:
#   GPU 2,3 only (vllm_base on GPU 0, vllm_solver on GPU 1 — already running)
#   /workspace      = R-Zero  (so 'import verl' works; verl reads tokens.json from cwd)
#   /workspace_deo  = DEO code
#   /storage        = DEO storage (verl writes models here; STORAGE_PATH=/storage)
#   /storage_rzero  = R-Zero existing storage (for cross-reference if needed)
#   docker.sock     = so DEO can stop/restart vllm_solver between iterations
tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_runner \
    --gpus '\"device=2,3\"' \
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
    mcmc_deo_vllm.py 2>&1 | tee ${DEO_STORAGE}/logs/run_\$(date +%Y%m%d_%H%M%S).log" Enter

echo ""
echo "DEO running in tmux session '${SESSION}'"
echo ""
echo "  tmux attach -t ${SESSION}        # view live"
echo "  Ctrl+B, D                        # detach (keeps running)"
echo "  tmux kill-session -t ${SESSION}  # stop"
echo ""
echo "Logs also tee'd to: ${DEO_STORAGE}/logs/run_<timestamp>.log"

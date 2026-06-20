#!/bin/bash
# Launch base_agreement_v2 ablation. Reuses canonical's archived mcmc pools
# (so no per-iter MCMC walk), re-scores with both current solver and base,
# keeps mutual-agreement entries. ~4x faster than v1.
#
# Prereqs: vllm_base + 3 solver DP containers already running on GPU 0/1/4/5,
# solver_v1 ckpt already copied to /eph/nvme0/yyd/DEO/models/deo_base_agreement_solver_v1/,
# canonical mcmc pools copied to /eph/nvme0/yyd/DEO/canonical_mcmc_pools/.

set -e

SESSION="deo_ba_v2"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

# Sanity: vllm endpoints up?
for port in 8000 8001 8004 8005; do
    if ! curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
        echo "WARN: vllm endpoint on port ${port} not responding; run start_vllm.sh first"
    fi
done

# Bust HF cache for our v2 datasets to prevent stale shadow.
sudo rm -rf "$HF_CACHE"/datasets/yuyang322___deo_base_agreement_v2* \
            "$HF_CACHE"/datasets/_*yuyang322___deo_base_agreement_v2*.lock 2>/dev/null || true

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_ba_v2_runner \
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
    base_agreement_v2_main.py 2>&1 | tee ${DEO_STORAGE}/logs/run_base_agreement_v2_\$(date +%Y%m%d_%H%M%S).log" Enter

echo "base_agreement_v2 launched in tmux session '${SESSION}'"
echo "  Results: ${DEO_STORAGE}/results_summary_base_agreement_v2.json"

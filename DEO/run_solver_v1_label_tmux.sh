#!/bin/bash
# Launch solver_v1_label ablation: solver_v1 frozen as labeler across all iters.
# Uses existing vllm_solver_check (already loaded with solver_v1) on port 8001.

set -e

SESSION="deo_sv1l"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

# Symlink solver_v1 ckpt under this ablation's prefix (saves 148GB vs cp).
# Use RELATIVE link target so it resolves both on host and inside docker
# (docker mounts /eph/nvme0/yyd/DEO at /storage, so absolute /eph/... target
# would be unresolvable inside the container).
if [ ! -e "$DEO_STORAGE/models/deo_solver_v1_label_solver_v1" ]; then
    echo "Symlinking solver_v1 ckpt under new prefix (relative)..."
    (cd "$DEO_STORAGE/models" && \
     ln -s deo_base_pseudo_solver_v1 deo_solver_v1_label_solver_v1)
fi

# Bust HF cache for new dataset names.
sudo rm -rf "$HF_CACHE"/datasets/yuyang322___deo_solver_v1_label* \
            "$HF_CACHE"/datasets/_*yuyang322___deo_solver_v1_label*.lock 2>/dev/null || true

# Sanity-check labeling endpoint is up
if ! curl -fsS http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "ERROR: vllm_solver_check (port 8001) not responding. Start it first."
    exit 1
fi
echo "Label endpoint (vllm_solver_check on port 8001, solver_v1 loaded) is ready."

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_sv1l_runner \
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
    solver_v1_label_main.py 2>&1 | tee ${DEO_STORAGE}/logs/run_solver_v1_label_\$(date +%Y%m%d_%H%M%S).log" Enter

echo ""
echo "solver_v1_label ablation running in tmux session '${SESSION}'"
echo "  Results: ${DEO_STORAGE}/results_summary_solver_v1_label.json"

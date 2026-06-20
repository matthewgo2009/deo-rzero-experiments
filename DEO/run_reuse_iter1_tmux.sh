#!/bin/bash
# Launch reuse_iter1 ablation in tmux. Trains solver_v2..v5 on the FIXED
# iter 1 dataset, no per-iter MCMC. Starts from canonical DEO archive's
# solver_v1 ckpt.
#
# No vllm containers needed — eval_math500 spawns its own vllm.LLM inside
# the deo_runner process.

set -e

SESSION="deo_reuse"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO_reuse_iter1"
ARCHIVE="/eph/nvme0/yyd/DEO_archived_20260515_204605"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

# Fresh storage dir for this ablation, separate from canonical DEO so the
# canonical archive stays untouched.
mkdir -p "$DEO_STORAGE"/{models,evaluation,datasets,logs,generated_question,temp_results}

# Copy (not symlink — verl rewrites inside) the canonical solver_v1 ckpt as
# our starting point. This is what eval_math500 will load and what verl
# uses as the actor init for iter 2's training.
if [ ! -d "$DEO_STORAGE/models/deo_reuse_iter1_solver_v1" ]; then
    echo "Copying solver_v1 ckpt from canonical archive..."
    sudo cp -r "$ARCHIVE/models/deo_qwen3_4b_base_solver_v1" \
        "$DEO_STORAGE/models/deo_reuse_iter1_solver_v1"
    sudo chown -R azureuser:azureuser "$DEO_STORAGE/models/deo_reuse_iter1_solver_v1"
    echo "  done: $(du -sh "$DEO_STORAGE/models/deo_reuse_iter1_solver_v1" | cut -f1)"
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Run the ablation script inside rzero docker. Mount /eph/nvme0/yyd/DEO_reuse_iter1
# at /storage so STORAGE_PATH=/storage works the same as canonical DEO.
# Use GPU 2,3 for verl, GPU 0 (single) for eval.
tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_reuse_runner \
    --gpus '\"device=0,1,2,3\"' \
    --network host \
    --shm-size 32g \
    -v ${DEO_CODE}:/workspace_deo \
    -v ${RZERO_CODE}:/workspace \
    -v ${DEO_STORAGE}:/storage \
    -v ${HF_CACHE}:/root/.cache/huggingface \
    -e PYTHONPATH=/workspace \
    -e STORAGE_PATH=/storage \
    -e HF_HOME=/root/.cache/huggingface \
    -e HUGGINGFACENAME=yuyang322 \
    -e WANDB_MODE=disabled \
    -w /workspace_deo \
    --entrypoint python3 \
    rzero:latest \
    reuse_iter1_main.py 2>&1 | tee ${DEO_STORAGE}/logs/run_reuse_iter1_\$(date +%Y%m%d_%H%M%S).log" Enter

echo ""
echo "Reuse-iter1 ablation running in tmux session '${SESSION}'"
echo ""
echo "  tmux attach -t ${SESSION}        # view live"
echo "  Ctrl+B, D                        # detach"
echo "  tmux kill-session -t ${SESSION}  # stop"
echo ""
echo "Results: ${DEO_STORAGE}/results_summary_reuse_iter1.json"

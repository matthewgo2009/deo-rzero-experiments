#!/bin/bash
# Canonical DEO pipeline + pipelined MCMC. Same labeler behavior as canonical
# (reload vllm_solver every iter), but MCMC walk uses mcmc_pipelined.py for
# ~5x speedup on the MCMC portion (~1.7x per-iter wall clock).
#
# Used to benchmark pipelined MCMC vs canonical at iso-acc on iter 1.

set -e

SESSION="deo_pipe"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

mkdir -p "$DEO_STORAGE"/{models,evaluation,datasets,logs,generated_question,temp_results}

# Stop old tmux sessions (the previous frozen_sv1_full run left an orphan session)
tmux kill-session -t deo_fsv1 2>/dev/null || true

# Bring up vllm endpoints fresh (base on GPU 0, 3-DP solver on GPU 1/4/5 all loaded
# with Qwen3-4B-Base initially — pipelined_canonical_main.py reloads to solver_vN
# after each iter, same as canonical).
echo "Starting vllm endpoints fresh (base model on all)..."
bash "$DEO_CODE/start_vllm.sh"

echo "Waiting for endpoints..."
until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8001/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8004/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8005/v1/models >/dev/null 2>&1; do
    sleep 15
done
echo "All vllm endpoints ready."

# Bust HF cache for this experiment's datasets (prevents stale-cache shadow)
sudo rm -rf "$HF_CACHE"/datasets/yuyang322___deo_canonical_pipelined* \
            "$HF_CACHE"/datasets/_*yuyang322___deo_canonical_pipelined*.lock 2>/dev/null || true

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_pipe_runner \
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
    pipelined_canonical_main.py 2>&1 | tee ${DEO_STORAGE}/logs/run_pipelined_canonical_\$(date +%Y%m%d_%H%M%S).log" Enter

echo ""
echo "Pipelined-canonical running in tmux session '${SESSION}'"
echo "  tmux attach -t ${SESSION}        # view live"
echo "  Ctrl+B, D                        # detach"
echo "  tmux kill-session -t ${SESSION}  # stop"
echo ""
echo "Results: ${DEO_STORAGE}/results_summary_pipelined_canonical.json"

#!/bin/bash
# Launch KL-fix baseline_no_mcmc in tmux. Same canonical DEO pipeline
# (regenerate MCMC each iter, reload labeler each iter, KL ref pinned to
# base), but MCMC_STEPS=0 so only the init pool (1500 questions sampled
# from base) is used — no mutation walk.
#
# Reuses any already-running vllm containers (4 of them: vllm_base +
# vllm_solver_{,_dp1,_dp2}) if available; otherwise calls start_vllm.sh.

set -e

SESSION="deo_bkf"
DEO_CODE="/home/azureuser/yyd/DEO"
RZERO_CODE="/home/azureuser/yyd/R-Zero"
DEO_STORAGE="/eph/nvme0/yyd/DEO"
HF_CACHE="/eph/nvme0/yyd/hf_cache"

mkdir -p "$DEO_STORAGE"/{models,evaluation,datasets,logs,generated_question,temp_results}

# Verify vllm endpoints are alive; start if not.
ALIVE=1
for p in 8000 8001 8004 8005; do
    curl -fsS "http://localhost:$p/v1/models" >/dev/null 2>&1 || ALIVE=0
done
if [ "$ALIVE" = "0" ]; then
    echo "Some vllm endpoints down. Restarting all 4 fresh..."
    bash "$DEO_CODE/start_vllm.sh"
    until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1 \
       && curl -fsS http://localhost:8001/v1/models >/dev/null 2>&1 \
       && curl -fsS http://localhost:8004/v1/models >/dev/null 2>&1 \
       && curl -fsS http://localhost:8005/v1/models >/dev/null 2>&1; do
        sleep 15
    done
fi
echo "All vllm endpoints ready."

# Bust HF cache for this experiment's dataset names (prevent stale-cache shadow).
sudo rm -rf "$HF_CACHE"/datasets/yuyang322___deo_baseline_klfix* \
            "$HF_CACHE"/datasets/_*yuyang322___deo_baseline_klfix*.lock 2>/dev/null || true

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "sudo docker run --rm \
    --name deo_bkf_runner \
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
    baseline_no_mcmc_klfix_main.py 2>&1 | tee ${DEO_STORAGE}/logs/run_baseline_klfix_\$(date +%Y%m%d_%H%M%S).log" Enter

echo ""
echo "baseline_no_mcmc_klfix running in tmux session '${SESSION}'"
echo "  tmux attach -t ${SESSION}        # view live"
echo "  Ctrl+B, D                        # detach"
echo "  tmux kill-session -t ${SESSION}  # stop"
echo ""
echo "Results: ${DEO_STORAGE}/results_summary_baseline_klfix.json"

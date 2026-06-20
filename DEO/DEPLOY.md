# DEO Deployment Guide (post-migration)

This is a self-contained deployment guide for the DEO (Direct Self-evolving Optimization) project.
If you (Claude or the user) are reading this on a fresh cluster — **start here**.

---

## What this project is

DEO replaces R-Zero's trained challenger with **MCMC sampling from base model** for self-evolving math
LLM training. The pipeline is:

```
For each iter t = 1..5:
  1. MCMC sample 1500 questions from Qwen3-4B-Base, score r_unc with current solver
  2. Filter [0.3, 0.8] -> push HF dataset (~1000 questions)
  3. R-Zero verl GRPO training on that dataset (max_steps=20, rollout_batch=512)
  4. MATH-500 eval on the trained solver
  5. Reload vllm_solver with new checkpoint -> next iter
```

We run **two parallel pipelines**:
- **Main**: full DEO with MCMC mutation walk
- **Baseline ablation**: same pipeline but `SKIP_MCMC_WALK=True` (init pool only, no mutation)

The two together let us isolate MCMC's contribution.

---

## ⚠️ Critical patches we applied to R-Zero verl

We modified verl to **pin the KL reference policy to the base model** instead of resetting it
to the previous-iter actor each iter. Without this, cumulative KL drift collapses solver
back to base after 5 iters. The diffs:

1. `R-Zero/verl/workers/actor/config.py` — `RefConfig` gained `model: ModelConfig` field
2. `R-Zero/verl/workers/config.py` — `WorkerConfig.post_init()` falls back ref.model to actor.model when no override
3. `R-Zero/verl/workers/fsdp_workers.py` — ref worker uses `self.config.ref.model` instead of `self.config.actor.model`
4. `DEO/mcmc_deo_vllm.py` and `DEO/baseline_no_mcmc.py` — verl call passes `worker.ref.model.model_path=Qwen/Qwen3-4B-Base`

If you ever rebuild from upstream R-Zero, **reapply these 4 patches** or all KL-drift fixes are undone.

---

## Hardware required

- **8x H100 80GB** (or any 80GB GPUs)
- Docker + NVIDIA Container Toolkit
- ~50 GB disk for HF model cache, ~500 GB for training checkpoints
- Network: outbound to huggingface.co, openai (optional), wandb (optional, disabled by default)

GPU layout used by all scripts (hardcoded — change deliberately):
```
GPU 0   vllm_base              (Qwen3-4B-Base, port 8000)        main: question generator
GPU 1   vllm_solver            (current solver, port 8001)        main: r_unc scorer (reloaded each iter)
GPU 2,3 deo_runner             (verl training, FSDP TP=2)         main: solver training
GPU 4   vllm_base_baseline     (Qwen3-4B-Base, port 8002)        baseline: question generator
GPU 5   vllm_solver_baseline   (current solver, port 8003)        baseline: r_unc scorer
GPU 6,7 deo_runner_baseline    (verl training, FSDP TP=2)         baseline: solver training
```

---

## Path layout (hardcoded — must match)

```
~/yyd/DEO/                              <-- DEO code (this directory)
~/yyd/R-Zero/                           <-- R-Zero codebase + our verl patches
~/.claude/projects/-home-azureuser-yyd/ <-- Claude conversation history (slug == working dir)

/eph/nvme0/<user>/DEO/                  <-- main run storage (verl writes here)
/eph/nvme0/<user>/DEO_baseline/         <-- baseline run storage
/eph/nvme0/<user>/hf_cache/             <-- HuggingFace model + dataset cache
```

If your username is **not** `azureuser`, hardcoded paths in `run_deo_tmux.sh` /
`run_baseline_tmux.sh` need editing, OR `sudo ln -s /home/<user> /home/azureuser` as a shortcut.

---

## Cold start: deploy from scratch

```bash
# 0. Verify Docker + GPU
docker --version
nvidia-smi -L | head    # should show 8 GPUs

# 1. Extract migration bundle (already done if reading this file)
# tar xzf migrate_essentials.tar.gz -C ~/

# 2. Create storage dirs
mkdir -p /eph/nvme0/$(whoami)/{DEO,DEO_baseline,hf_cache}
for d in datasets logs models evaluation generated_question temp_results; do
    mkdir -p /eph/nvme0/$(whoami)/DEO/$d
    mkdir -p /eph/nvme0/$(whoami)/DEO_baseline/$d
done

# 3. Build rzero docker (~10 min, includes the patched verl)
cd ~/yyd/R-Zero && bash build_docker.sh
docker images | grep rzero    # rzero:latest should exist

# 4. Verify HF token in tokens.json
cat ~/yyd/R-Zero/tokens.json
# Make sure "huggingface" key has a valid token (NOT the placeholder).
# WandB key can be placeholder (we run with WANDB_MODE=disabled).

# 5. Start vllm endpoints (first time downloads ~8 GB Qwen3-4B-Base, ~5-10 min)
cd ~/yyd/DEO
bash start_vllm.sh
bash start_vllm_baseline.sh

# Wait until all 4 endpoints respond
until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8001/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8002/v1/models >/dev/null 2>&1 \
   && curl -fsS http://localhost:8003/v1/models >/dev/null 2>&1; do
    sleep 10
done && echo "ALL VLLM READY"

# 6. Launch the two runs
bash run_deo_tmux.sh        # tmux session: deo
bash run_baseline_tmux.sh   # tmux session: deo_baseline

# 7. Watch
tmux attach -t deo                 # ctrl+B D to detach
sudo docker logs -f deo_runner     # alternative
tail -f /eph/nvme0/$(whoami)/DEO/logs/run_*.log
```

Expected total wall-clock: **~30-40 hours** for 5 iterations of each pipeline.

---

## Resume Claude conversation

```bash
claude /login                    # use yuyang.deng@accenture.com
cd ~/yyd                         # working dir slug must match: -home-<user>-yyd
claude --resume 4fc528a2-9e4e-495c-a5a1-6d9228e4968d
```

If `~/.claude/projects/-home-<user>-yyd/` doesn't exist, the slug doesn't match.
Either move conversations to the right slug or work from `/home/azureuser/yyd` (matches existing slug).

---

## Common operations

### Check progress

```bash
# Aggregate accuracy so far
cat /eph/nvme0/$(whoami)/DEO/results_summary.json
cat /eph/nvme0/$(whoami)/DEO_baseline/results_summary.json

# Containers + GPU
sudo docker ps --format "table {{.Names}}\t{{.Status}}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# Latest milestones
LOG=$(ls -1t /eph/nvme0/$(whoami)/DEO/logs/run_*.log | head -1)
grep -E "MATH-500 acc|Iteration|verl\] training|reloading|filtered count|FIRST attempt failed|ERROR|Traceback" $LOG | tail -20
```

### MCMC trajectory inspection (per iter, includes OLD/NEW question pairs and r_c deltas)

```bash
ls /eph/nvme0/$(whoami)/DEO/logs/mcmc_iter_*.log
# Sample 3 ACCEPTED mutations from iter 1:
awk '/^\[Step / {if (record && status=="ACCEPTED") {print record"\n---"; n++; if(n>=3) exit}; record=$0; status=($0~/ACCEPTED/?"ACCEPTED":"REJECTED"); next} {record=record"\n"$0}' \
    /eph/nvme0/$(whoami)/DEO/logs/mcmc_iter_1.log
```

### Stop everything

```bash
sudo docker stop deo_runner deo_runner_baseline vllm_base vllm_solver vllm_base_baseline vllm_solver_baseline
sudo docker rm   deo_runner deo_runner_baseline vllm_base vllm_solver vllm_base_baseline vllm_solver_baseline
tmux kill-session -t deo
tmux kill-session -t deo_baseline
```

### Restart fresh (clean previous state)

```bash
TS=$(date +%Y%m%d_%H%M%S)
sudo mv /eph/nvme0/$(whoami)/DEO          /eph/nvme0/$(whoami)/DEO_archived_${TS}
sudo mv /eph/nvme0/$(whoami)/DEO_baseline /eph/nvme0/$(whoami)/DEO_baseline_archived_${TS}
# Clean HF dataset cache for our pushed datasets (prevents stale-cache bug)
rm -rf /eph/nvme0/$(whoami)/hf_cache/datasets/yuyang322___deo_qwen3_4b_base*

# Recreate storage
mkdir -p /eph/nvme0/$(whoami)/{DEO,DEO_baseline}
for d in datasets logs models evaluation generated_question temp_results; do
    mkdir -p /eph/nvme0/$(whoami)/DEO/$d
    mkdir -p /eph/nvme0/$(whoami)/DEO_baseline/$d
done

# then re-run start_vllm + run_*_tmux
```

---

## Configuration knobs

### `~/yyd/DEO/mcmc_deo_vllm.py` Config class

| Knob | Current | What it controls |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-4B-Base` | base model for both ref and initial solver |
| `MODEL_ABBR` | `deo_qwen3_4b_base` | prefix for HF datasets and ckpt dirs |
| `TOTAL_QUESTIONS` | 1500 | MCMC pool size; ensures filter ≥ 512 |
| `MCMC_STEPS` | 5 | MCMC walk length |
| `MUTATE_BATCH_SIZE` | 20 | proposals per vllm batch call |
| `M_SAMPLES` | 9 | majority-vote samples per question (matches R-Zero) |
| `BETA` | 0.1 | MH temperature |
| `LAMBDA_REP` | 10.0 | BLEU-cluster repetition penalty weight |
| `MIN/MAX_SCORE` | 0.3 / 0.8 | filter range on majority-vote rate |
| `NUM_ITERATIONS` | 5 | outer self-evolving loops |

### `baseline_no_mcmc.py` extra knob

| Knob | Value | Effect |
|---|---|---|
| `SKIP_MCMC_WALK` | `True` | bypass mutation walk; init pool only |

### Verl args we override (in `run_verl_solver`)

```
worker.actor.model.model_path     <- current solver checkpoint  (changes each iter)
worker.ref.model.model_path       <- Qwen/Qwen3-4B-Base          (PINNED, KL-fix patch)
trainer.max_steps                 = 20
worker.actor.global_batch_size    = 16
data.max_response_length          = 4096
data.format_prompt                = ./examples/format_prompt/solver.jinja
trainer.val_freq                  = 4
data.train_files                  = HF dataset name@train         (pushed by filter_and_push)
worker.rollout.tensor_parallel_size = 2
```

---

## Known gotchas

### 1. Filter pass rate < 512 -> verl assertion fails

If `[upload] X/1500 passed filter` shows X < 512, verl's `assert len(train_dataloader) >= 1`
will trip. Mitigation:
- Either bump `TOTAL_QUESTIONS` higher (e.g. 2000)
- Or override `data.rollout_batch_size` to a smaller value in run_verl_solver

This bit us in baseline iter 4 of the broken-KL run (476/1500 = 32% < 512).

### 2. Ray "0 GPUs available" on later iters

We have a defensive `cleanup_ray()` in `run_verl_solver` that runs `ray stop --force` and
`rm -rf /tmp/ray*` before each verl call, plus a single retry. Past iter 4 main runs hit this
once with retry succeeding. If you see `FIRST attempt failed (rc=1). Cleaning Ray + retrying once...`
in the log, the retry handled it.

### 3. HF dataset cache poisoning

We bust the local `/root/.cache/huggingface/datasets/<user>___<exp_name>` directory before each
verl call. If somehow verl loads 0 examples while HF has 600+ (possible after manual upload
mishaps), check that cache and delete it.

### 4. vllm_solver hot-reload

Full FT can't hot-swap weights via vllm API; we stop+remove+rerun the docker container with the
new `--model` path. Takes ~60-90s. Code lives in `reload_vllm_solver()`. Requires:
- `/var/run/docker.sock` mounted into the runner container
- `/usr/bin/docker` mounted from host (binary only, since vllm/vllm-openai base doesn't have it)

### 5. Working-dir slug for Claude history

The slug `-home-azureuser-yyd` only matches if you `cd /home/azureuser/yyd` (or have a softlink).
If your homedir is `/home/foo`, slug becomes `-home-foo-yyd` and Claude won't find old conversations.

---

## Paper data (broken-KL archive)

Optional separate bundle `migrate_paper_data.tar.gz` contains the **broken-KL** run's:
- 5 iter MCMC datasets (`mcmc_iter_*.json`)
- 5 iter trajectory logs (OLD/NEW question pairs, r_c, alpha)
- per-iter MATH-500 eval results
- `results_summary.json`

Numbers (R-Zero default, ref=prev_iter, no fix):
```
                        baseline  iter1  iter2  iter3  iter4  iter5
Main (MCMC)              58.6     64.2   62.4   61.2   60.8   58.6   <-- collapses back to base
Baseline (no MCMC)       58.8     62.6   61.0   60.8   CRASH  -      <-- iter 4 crashed (filter < 512)
```

Use this for the "before" curve when comparing against KL-fix runs.

Extract:
```bash
tar xzf migrate_paper_data.tar.gz -C ~/paper_data
```

---

## What to ask Claude after migration

If you're handing this off to Claude on the new cluster, key context to seed:

1. *"This is a self-evolving math LLM training pipeline replacing R-Zero's challenger with MCMC sampling. We've patched verl to pin KL ref to base."*
2. *"Two pipelines run in parallel: main (mcmc_deo_vllm.py) and baseline ablation (baseline_no_mcmc.py, SKIP_MCMC_WALK=True)."*
3. *"Watch for filter pass rate < 512 (verl crashes), Ray '0 GPUs' bug (we have retry), and pseudo-label correctness on iter 1/2 (~30-40% wrong)."*

The full conversation history in `~/.claude/projects/-home-<user>-yyd/` covers everything we've debugged.

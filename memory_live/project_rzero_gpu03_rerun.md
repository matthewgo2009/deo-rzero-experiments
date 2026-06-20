---
name: project-rzero-gpu03-rerun
description: "2026-06-19 fresh R-Zero fair-comparison rerun on GPU 0-3 (2+2 layout), env/scripts/locations + iter-0/1 results"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7d1219bf-2527-4009-9884-30d66ffca789
---

Fresh R-Zero-vs-DEO fair comparison started 2026-06-19 on the migrated cluster
(the old run's checkpoints/repo/docker/`/eph/nvme0` were all gone). Goal: run
official R-Zero (github.com/Chengsong-Huang/R-Zero, 2025-08 penalty-questioner
version) on Qwen3-4B-Base, eval MATH-500 with OUR grader, compare vs DEO
paper_data (DEO side already rechecked, not re-run).

**Locations (all big artifacts on the 27TB nvme0):**
- Repo: `/home/azureuser/yyd/R-Zero` (cloned fresh, scripts adapted in-place)
- venv: `/eph/nvme0/yyd/rzero_venv` (torch2.7.0+cu126, vllm0.9.1, flash_attn2.7.4.post1; `av` pinned down to 14.2.0, `stopit` + `setuptools<81` added)
- STORAGE_PATH: `/eph/nvme0/yyd/R-Zero_run` (models/evaluation/logs/generated_question)
- HF cache: `/eph/nvme0/yyd/hf_cache`; env file `R-Zero/run_env.sh` (source before any run; sets PYTHONPATH=repo root — generate.py does package-style `import evaluation.*`)
- tokens.json has hf (yuyang322, write-OK) + openai (gpt-4o-mini grader)

**GPU 0-3 (2+2) adaptation (no algorithm/hparam change):**
- `vllm_service_init/start.sh`: 4→2 solver-scoring servers on GPU 2,3 (ports 5000,5001)
- `caller_penalty.py`: reward sharding `N_SERVERS=2` (was hardcoded 4)
- `questioner_train_penalty.sh`: CUDA 0,1 + n_gpus_per_node=2 (verl on 0,1 while vllm scores on 2,3)
- `solver_train.sh`: CUDA 0,1,2,3 + n_gpus_per_node=4 + `rollout_batch_size=64` + `global_batch_size=16` (must divide rollout_batch — see [[verl-rollout-batch-size-gotcha]]) + val disabled (val_freq=-1, val_before_train=false — result-neutral, saves ~3h/iter on unused math12k val)
- `question_generate.bash`/`question_evaluate/evaluate.sh`: 8→4 shards (suffix 0-3, exactly what upload.py consumes)
- `evaluation/evaluate.bash`: MATH-500 ONLY on GPU 0 + our `results_recheck_math500_mini.py` (GPT-4o-mini boxed recheck). R-Zero kept STOCK KL (ref reset to prev actor) — the KL-pin-to-base patch is DEO's, not applied here.
- Orchestrator: `scripts/run_rzero_gpu03.sh <base> <abbr> <MAX_ITERS>`; `START_ITER=N` env resumes reusing earlier iters.

**Results so far (MATH-500, GPT-mini rechecked):** iter0 base = 71.8 (math_verify 58.2 pre-recheck), iter1 solver_v1 = 76.6 (filtered set 1642). Full iter2-5 running in tmux `rzero`. Compare against canonical DEO 76.8→69.0 in [[project-rzero-vs-deo-findings]]. Eval methodology [[project-math500-eval]].

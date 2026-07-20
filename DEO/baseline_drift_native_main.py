"""
DEO baseline_drift (no MCMC walk + DRIFTING labeler) — NATIVE runner for an
AzureML Command Job container (8xH100, single node, no docker-in-docker).

Same algorithm as baseline_drift_main.py (MCMC_STEPS=0; each iter's labeler is
the previous iter's solver; KL ref pinned to base via the verl patch). The only
differences are infrastructure plumbing, applied here by monkeypatching the deo
module so mcmc_deo_vllm.py stays untouched:
  * STORAGE_ROOT / RZERO_DIR come from env (AzureML output mount + code dir)
  * vLLM servers run as native processes (start_vllm_native.sh), not docker
  * reload_vllm_solver restarts the native solver shards in place
  * verl trains on GPU 2,3; per-iter MATH-500 eval runs on GPU 6
GPU layout: base=0, solver DP=1/4/5, verl=2/3, eval=6 (7 spare).
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcmc_deo_vllm as deo

# ----- config from env -----
deo.config.STORAGE_ROOT = os.environ["STORAGE_PATH"]          # AzureML output mount
deo.config.RZERO_DIR = os.environ.get("RZERO_DIR", "/workspace")
deo.config.HF_USER = os.environ.get("HUGGINGFACENAME", "yuyang322")
deo.config.MODEL_ABBR = os.environ.get("DEO_ABBR", "deo_baseline_drift")
deo.config.MCMC_STEPS = 0                                     # no walk (init pool only)

VERL_GPUS = os.environ.get("DEO_VERL_GPUS", "2,3")
EVAL_GPU = os.environ.get("DEO_EVAL_GPU", "6")
# rollout_batch_size=64 default (small filtered sets); smoke runs pass a smaller
# value + trainer.max_steps=1 via DEO_VERL_EXTRA (space-separated Hydra overrides).
VERL_OVERRIDES = ["data.rollout_batch_size=64"] + os.environ.get("DEO_VERL_EXTRA", "").split()
# scale knobs (full run = defaults; smoke run overrides via env)
deo.config.NUM_ITERATIONS = int(os.environ.get("DEO_NUM_ITERS", deo.config.NUM_ITERATIONS))
deo.config.TOTAL_QUESTIONS = int(os.environ.get("DEO_TOTAL_Q", deo.config.TOTAL_QUESTIONS))
PIDDIR = os.environ.get("VLLM_PIDDIR", "/tmp/vllm_pids")
LOGDIR = os.environ.get("VLLM_LOGDIR", "/tmp/vllm_logs")


# ----- native vLLM solver reload (replaces docker version) -----
def _launch_vllm(gpu, port, model):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    os.makedirs(PIDDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    log = open(f"{LOGDIR}/solver_{port}.log", "ab")
    p = subprocess.Popen(
        ["vllm", "serve", model,
         "--served-model-name", deo.config.MODEL_NAME,
         "--dtype", "bfloat16", "--max-model-len", "6144",
         "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.85",
         "--port", str(port)],
        env=env, stdout=log, stderr=log,
    )
    with open(f"{PIDDIR}/{port}.pid", "w") as f:
        f.write(str(p.pid))
    return p


def native_reload_vllm_solver(new_model_path):
    n = len(deo.config.SOLVER_INSTANCES)
    print(f"\n[vllm-solver native] reloading {n} shards with {new_model_path}", flush=True)
    # tear down all solver shards first so GPU memory frees before relaunch
    for gpu_id, port, _name in deo.config.SOLVER_INSTANCES:
        subprocess.run(["pkill", "-9", "-f", f"--port {port}"], check=False)
    time.sleep(8)
    for gpu_id, port, _name in deo.config.SOLVER_INSTANCES:
        _launch_vllm(gpu_id, port, new_model_path)
    for url in deo.config.VLLM_SOLVER_URLS:
        deo.wait_for_vllm_ready(url, label=f"vllm-solver@{url}")
    deo._clients_solver = None


deo.reload_vllm_solver = native_reload_vllm_solver

# ----- verl on GPU 2,3 (wrap so the subprocess inherits the right CUDA mask) -----
_orig_run_verl = deo.run_verl_solver


def native_run_verl(*args, **kwargs):
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = VERL_GPUS
    try:
        return _orig_run_verl(*args, **kwargs)
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


deo.run_verl_solver = native_run_verl


# ----- per-iter MATH-500 eval on GPU 6 (native copy of deo.eval_math500) -----
def native_eval_math500(model_path, label):
    print(f"\n=== MATH-500 eval: {label}  (model={model_path}) ===", flush=True)
    env = os.environ.copy()
    env["STORAGE_PATH"] = deo.config.STORAGE_ROOT
    env["CUDA_VISIBLE_DEVICES"] = EVAL_GPU
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    subprocess.run(
        ["python3", "evaluation/generate.py", "--model", model_path, "--dataset", "math"],
        cwd=deo.config.RZERO_DIR, check=True, env=env,
    )
    results_path = (
        f"{deo.config.STORAGE_ROOT}/evaluation/"
        f"{model_path.replace('/', '_')}/results_math.json"
    )
    old_avg, new_avg, n_bumped = deo.gpt_recheck_math500(results_path)
    print(f"=== {label}: MATH-500 acc = {new_avg:.4f} ({new_avg*100:.2f}%) "
          f"[raw {old_avg:.4f}, +{n_bumped} GPT-bumped] ===\n", flush=True)
    return new_avg


deo.eval_math500 = native_eval_math500


def main():
    deo.config.HF_TOKEN = deo.load_hf_token()
    deo.hf_login(token=deo.config.HF_TOKEN)
    for d in ["datasets", "logs", "models", "evaluation", "temp_results", "generated_question"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{d}", exist_ok=True)

    tokenizer = deo.AutoTokenizer.from_pretrained(deo.config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    deo.wait_for_vllm_ready(deo.config.VLLM_BASE_URL, label="vllm-base")
    for url in deo.config.VLLM_SOLVER_URLS:
        deo.wait_for_vllm_ready(url, label=f"vllm-solver@{url}")

    eval_history = {}
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_baseline_drift.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    # idempotent resume: reload prior summary if present
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            eval_history = json.load(f)

    if "iter_0_baseline" not in eval_history:
        eval_history["iter_0_baseline"] = native_eval_math500(deo.config.MODEL_NAME, "iter 0 baseline")
        save()

    current_solver = deo.config.MODEL_NAME
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        done_key = f"iter_{it}"
        ckpt = f"{deo.config.STORAGE_ROOT}/models/deo_baseline_drift_solver_v{it}/global_step_15/actor/huggingface"
        if done_key in eval_history and os.path.isdir(ckpt):
            print(f"=== iter {it} already complete (acc={eval_history[done_key]}), skipping ===", flush=True)
            current_solver = ckpt
            if it < deo.config.NUM_ITERATIONS:
                native_reload_vllm_solver(ckpt)
            continue

        print(f"\n{'='*60}\n=== Iteration {it}/{deo.config.NUM_ITERATIONS} (native, no walk + drift) ===\n{'='*60}", flush=True)
        exp_name = f"deo_baseline_drift_solver_v{it}"
        repo_name = exp_name
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_baseline_drift.log"

        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_baseline_drift.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        deo.filter_and_push(train_data, repo_name, exp_name)
        merged_ckpt = native_run_verl(
            current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name,
            extra_verl_overrides=VERL_OVERRIDES,
        )

        eval_history[done_key] = native_eval_math500(merged_ckpt, f"iter {it}")
        save()

        if it < deo.config.NUM_ITERATIONS:
            native_reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\n" + "=" * 60, flush=True)
    print("FINAL TRAJECTORY (native, no walk + drifting labeler):", flush=True)
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f} ({acc*100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()

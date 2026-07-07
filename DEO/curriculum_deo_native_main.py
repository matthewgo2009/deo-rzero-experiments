"""
Curriculum-DEO — NATIVE runner (AzureML job, 8xH100, no docker).

Canonical DEO (MCMC walk ON) + solver M-vote self-labeling (the ORIGINAL
deo.filter_and_push, NOT Claude), but with a per-iteration BETA schedule that
ANNEALS the MH temperature down each iter. Acceptance is
    alpha = min(1, exp((E' - E) / BETA))   # maximizes total r_unc
so a SMALLER beta = greedier climb toward the max-uncertainty (p_hat=0.5) edge.
Schedule high->low = explore broadly early, then sharpen focus on hard edge
questions as the solver strengthens (curriculum).

Default schedule (iters 1..5): 1.0, 0.5, 0.25, 0.1, 0.05  (env DEO_BETA_SCHEDULE).
Everything else mirrors the validated native path. GPU: base=0, solver DP=1/4/5,
verl=2/3, eval=6.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcmc_deo_vllm as deo

deo.config.STORAGE_ROOT = os.environ["STORAGE_PATH"]
deo.config.RZERO_DIR = os.environ.get("RZERO_DIR", "/workspace")
deo.config.HF_USER = os.environ.get("HUGGINGFACENAME", "yuyang322")
deo.config.MODEL_ABBR = os.environ.get("DEO_ABBR", "deo_curriculum")
deo.config.MCMC_STEPS = int(os.environ.get("DEO_MCMC_STEPS", "5"))     # canonical walk
deo.config.NUM_ITERATIONS = int(os.environ.get("DEO_NUM_ITERS", deo.config.NUM_ITERATIONS))
deo.config.TOTAL_QUESTIONS = int(os.environ.get("DEO_TOTAL_Q", deo.config.TOTAL_QUESTIONS))

# per-iteration MH temperature schedule (annealed down => greedier toward hard edge)
BETA_SCHEDULE = [float(x) for x in
                 os.environ.get("DEO_BETA_SCHEDULE", "1.0,0.5,0.25,0.1,0.05").split(",")]

VERL_GPUS = os.environ.get("DEO_VERL_GPUS", "2,3")
EVAL_GPU = os.environ.get("DEO_EVAL_GPU", "6")
VERL_OVERRIDES = ["data.rollout_batch_size=64"] + os.environ.get("DEO_VERL_EXTRA", "").split()
PIDDIR = os.environ.get("VLLM_PIDDIR", "/tmp/vllm_pids")
LOGDIR = os.environ.get("VLLM_LOGDIR", "/tmp/vllm_logs")


def _launch_vllm(gpu, port, model):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    os.makedirs(PIDDIR, exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)
    log = open(f"{LOGDIR}/solver_{port}.log", "ab")
    p = subprocess.Popen(
        ["vllm", "serve", model, "--served-model-name", deo.config.MODEL_NAME,
         "--dtype", "bfloat16", "--max-model-len", "6144", "--tensor-parallel-size", "1",
         "--gpu-memory-utilization", "0.85", "--port", str(port)],
        env=env, stdout=log, stderr=log,
    )
    with open(f"{PIDDIR}/{port}.pid", "w") as f:
        f.write(str(p.pid))
    return p


def native_reload_vllm_solver(new_model_path):
    print(f"\n[vllm-solver native] reloading with {new_model_path}", flush=True)
    for _gpu, port, _name in deo.config.SOLVER_INSTANCES:
        subprocess.run(["pkill", "-9", "-f", f"--port {port}"], check=False)
    time.sleep(8)
    for gpu_id, port, _name in deo.config.SOLVER_INSTANCES:
        _launch_vllm(gpu_id, port, new_model_path)
    for url in deo.config.VLLM_SOLVER_URLS:
        deo.wait_for_vllm_ready(url, label=f"vllm-solver@{url}")
    deo._clients_solver = None


deo.reload_vllm_solver = native_reload_vllm_solver

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


def native_eval_math500(model_path, label):
    print(f"\n=== MATH-500 eval: {label} (model={model_path}) ===", flush=True)
    env = os.environ.copy()
    env["STORAGE_PATH"] = deo.config.STORAGE_ROOT
    env["CUDA_VISIBLE_DEVICES"] = EVAL_GPU
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    subprocess.run(
        ["python3", "evaluation/generate.py", "--model", model_path, "--dataset", "math"],
        cwd=deo.config.RZERO_DIR, check=True, env=env,
    )
    results_path = (f"{deo.config.STORAGE_ROOT}/evaluation/"
                    f"{model_path.replace('/', '_')}/results_math.json")
    old_avg, new_avg, n_bumped = deo.gpt_recheck_math500(results_path)
    print(f"=== {label}: MATH-500 acc = {new_avg:.4f} ({new_avg*100:.2f}%) "
          f"[raw {old_avg:.4f}, +{n_bumped} GPT-bumped] ===\n", flush=True)
    return new_avg


deo.eval_math500 = native_eval_math500


def beta_for_iter(it):
    return BETA_SCHEDULE[it - 1] if it - 1 < len(BETA_SCHEDULE) else BETA_SCHEDULE[-1]


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

    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_{deo.config.MODEL_ABBR}.json"
    eval_history = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            eval_history = json.load(f)

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    print(f"[curriculum] BETA schedule (iter1..): {BETA_SCHEDULE}", flush=True)

    if "iter_0_baseline" not in eval_history:
        eval_history["iter_0_baseline"] = native_eval_math500(deo.config.MODEL_NAME, "iter 0 baseline")
        save()

    current_solver = deo.config.MODEL_NAME
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        done_key = f"iter_{it}"
        ckpt = f"{deo.config.STORAGE_ROOT}/models/{deo.config.MODEL_ABBR}_solver_v{it}/global_step_15/actor/huggingface"
        if done_key in eval_history and os.path.isdir(ckpt):
            print(f"=== iter {it} already complete (acc={eval_history[done_key]}), skipping ===", flush=True)
            current_solver = ckpt
            if it < deo.config.NUM_ITERATIONS:
                native_reload_vllm_solver(ckpt)
            continue

        deo.config.BETA = beta_for_iter(it)
        print(f"\n{'='*60}\n=== Iteration {it}/{deo.config.NUM_ITERATIONS} (curriculum: BETA={deo.config.BETA}) ===\n{'='*60}", flush=True)
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.log"

        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        deo.filter_and_push(train_data, exp_name, exp_name)   # ORIGINAL solver M-vote labeler
        merged_ckpt = native_run_verl(
            current_solver, f"{deo.config.HF_USER}/{exp_name}", exp_name,
            extra_verl_overrides=VERL_OVERRIDES,
        )

        eval_history[done_key] = native_eval_math500(merged_ckpt, f"iter {it}")
        save()

        if it < deo.config.NUM_ITERATIONS:
            native_reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\nFINAL TRAJECTORY (curriculum-DEO, annealed beta):", flush=True)
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f} ({acc*100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()

"""
Canonical DEO (MCMC walk ON) but with GPT-4o as the LABELER — NATIVE runner for
an AzureML Command Job (8xH100, single node, no docker).

Canonical DEO == MCMC_STEPS=5 (mutation walk) + drifting solver (reloaded to the
just-trained ckpt each iter). The ONLY change here vs canonical is the training
label: instead of the 4B solver's M-vote majority (pseudo_label), each selected
question is SOLVED by GPT-4o and that boxed answer becomes the training label.
The solver is still used for question SELECTION (p_hat in [0.3,0.8]) and the MCMC
energy (r_unc), so the curriculum is unchanged — only label quality changes.

Infra plumbing (native vLLM, reload, GPU placement, resume) mirrors
baseline_drift_native_main.py. GPU layout: base=0, solver DP=1/4/5, verl=2/3,
eval=6. GPT labeling is an API call (no GPU).
"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcmc_deo_vllm as deo
import openai

# ----- config from env -----
deo.config.STORAGE_ROOT = os.environ["STORAGE_PATH"]
deo.config.RZERO_DIR = os.environ.get("RZERO_DIR", "/workspace")
deo.config.HF_USER = os.environ.get("HUGGINGFACENAME", "yuyang322")
deo.config.MODEL_ABBR = "deo_canon_gptlabel"
deo.config.MCMC_STEPS = int(os.environ.get("DEO_MCMC_STEPS", "5"))   # canonical walk
deo.config.NUM_ITERATIONS = int(os.environ.get("DEO_NUM_ITERS", deo.config.NUM_ITERATIONS))
deo.config.TOTAL_QUESTIONS = int(os.environ.get("DEO_TOTAL_Q", deo.config.TOTAL_QUESTIONS))

VERL_GPUS = os.environ.get("DEO_VERL_GPUS", "2,3")
EVAL_GPU = os.environ.get("DEO_EVAL_GPU", "6")
VERL_OVERRIDES = ["data.rollout_batch_size=64"] + os.environ.get("DEO_VERL_EXTRA", "").split()
PIDDIR = os.environ.get("VLLM_PIDDIR", "/tmp/vllm_pids")
LOGDIR = os.environ.get("VLLM_LOGDIR", "/tmp/vllm_logs")
LABEL_MODEL = os.environ.get("DEO_LABEL_MODEL", "gpt-4o")
LABEL_WORKERS = int(os.environ.get("DEO_LABEL_WORKERS", "24"))


# ======== GPT-4o labeler ========
def _gpt_solve(client, question, retries=4):
    """Solve a math question with the labeler model; return its last \\boxed{} answer."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LABEL_MODEL,
                messages=[
                    {"role": "system", "content": deo.RZERO_SOLVER_SYSTEM},
                    {"role": "user", "content": question},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            return deo._extract_last_boxed(resp.choices[0].message.content)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def filter_and_push_gpt(train_data, repo_name, config_name):
    """Canonical selection (regex already applied + p_hat band + solver answered),
    but RELABEL each surviving question with GPT-4o's solution before push."""
    n_total = len(train_data)
    # Stage 1 — canonical question selection: p_hat in band, solver produced an answer
    stage1 = [
        d for d in train_data
        if d["pseudo_label"] not in (None, "", "None")
        and deo.config.MIN_SCORE <= d["p_hat"] <= deo.config.MAX_SCORE
    ]
    print(f"[gpt-filter] phat∈[{deo.config.MIN_SCORE},{deo.config.MAX_SCORE}]+pseudo: "
          f"{len(stage1)}/{n_total} selected", flush=True)

    # Stage 2 — relabel with GPT-4o (concurrent)
    api_key = deo._load_openai_key()
    if not api_key:
        raise SystemExit("ERROR: openai key required for GPT labeler (tokens.json)")
    client = openai.OpenAI(api_key=api_key)
    t0 = time.time()
    gpt_answers = [None] * len(stage1)
    with ThreadPoolExecutor(max_workers=LABEL_WORKERS) as ex:
        futs = {ex.submit(_gpt_solve, client, d["question"]): i for i, d in enumerate(stage1)}
        for fut in as_completed(futs):
            gpt_answers[futs[fut]] = fut.result()
    n_labeled = sum(1 for a in gpt_answers if a)
    print(f"[gpt-filter] GPT-4o labeled {n_labeled}/{len(stage1)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    filtered = [
        {"problem": d["question"], "answer": ans, "score": d["p_hat"]}
        for d, ans in zip(stage1, gpt_answers) if ans
    ]
    print(f"[gpt-upload] {len(filtered)}/{n_total} final (GPT-labeled)", flush=True)
    if not filtered:
        raise SystemExit("ERROR: 0 GPT-labeled questions — cannot train empty dataset.")

    os.makedirs(f"{deo.config.STORAGE_ROOT}/datasets", exist_ok=True)
    with open(f"{deo.config.STORAGE_ROOT}/datasets/filtered_{repo_name}.json", "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    repo_full = f"{deo.config.HF_USER}/{repo_name}"
    ds = deo.DatasetDict({"train": deo.Dataset.from_list(filtered)})
    ds.push_to_hub(repo_full, private=True, config_name=config_name)
    print(f"[gpt-upload] pushed to https://huggingface.co/datasets/{repo_full}", flush=True)
    return len(filtered)


deo.filter_and_push = filter_and_push_gpt


# ======== native vLLM solver reload (same as baseline_drift_native) ========
def _launch_vllm(gpu, port, model):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    os.makedirs(PIDDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
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

    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_canon_gptlabel.json"
    eval_history = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            eval_history = json.load(f)

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    if "iter_0_baseline" not in eval_history:
        eval_history["iter_0_baseline"] = native_eval_math500(deo.config.MODEL_NAME, "iter 0 baseline")
        save()

    current_solver = deo.config.MODEL_NAME
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        done_key = f"iter_{it}"
        ckpt = f"{deo.config.STORAGE_ROOT}/models/deo_canon_gptlabel_solver_v{it}/global_step_15/actor/huggingface"
        if done_key in eval_history and os.path.isdir(ckpt):
            print(f"=== iter {it} already complete (acc={eval_history[done_key]}), skipping ===", flush=True)
            current_solver = ckpt
            if it < deo.config.NUM_ITERATIONS:
                native_reload_vllm_solver(ckpt)
            continue

        print(f"\n{'='*60}\n=== Iteration {it}/{deo.config.NUM_ITERATIONS} (canonical MCMC walk + GPT-4o labeler) ===\n{'='*60}", flush=True)
        exp_name = f"deo_canon_gptlabel_solver_v{it}"
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_canon_gptlabel.log"

        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_canon_gptlabel.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        deo.filter_and_push(train_data, exp_name, exp_name)
        merged_ckpt = native_run_verl(
            current_solver, f"{deo.config.HF_USER}/{exp_name}", exp_name,
            extra_verl_overrides=VERL_OVERRIDES,
        )

        eval_history[done_key] = native_eval_math500(merged_ckpt, f"iter {it}")
        save()

        if it < deo.config.NUM_ITERATIONS:
            native_reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\nFINAL TRAJECTORY (canonical MCMC walk + GPT-4o labeler):", flush=True)
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f} ({acc*100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()

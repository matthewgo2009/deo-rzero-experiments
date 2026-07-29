"""
Adaptive-Temperature DEO — NATIVE runner (AzureML job, 8xH100, no docker).

Canonical DEO (MCMC walk ON) + solver M-vote self-labeling (the ORIGINAL
deo.filter_and_push, NOT Claude), but instead of a FIXED beta schedule
(curriculum_deo_native_main.py) the MH temperature beta is UPDATED after each
iteration by the adaptive-temperature rule of DEO paper §1.4.

Goal: keep questions' uncertainty r_unc(x;theta) inside a target band [r_min, r_max].
delta is the VIOLATION rate (allowed fraction OUTSIDE the band). Constrained
problem for beta (paper §1.4, Eq 5-6):

    max_{beta in [bmin,bmax]} beta   s.t.   1 - E_{X~q(beta)}[V(X)] <= delta,
    V(X) = (1/N) sum_x 1{ r_min <= r_unc(x;theta) <= r_max }   (in-band fraction),
    r_c(X) = sum_x max(0, r_unc(x) - lambda_rep*r_rep(x)),   q(beta) ~ pi_base*exp(r_c/beta)

Lagrangian L = -log beta + lambda (1 - E[V] - delta), lambda >= 0, min_beta max_lambda.
Using dE[V]/dbeta = -(1/beta^2) Cov_{q(beta)}(V, r_c), estimated from this iter's
B batch-chunk samples X_1..X_B (V_i, R_i=r_c(X_i)):

    grad_beta L   = -1/beta + (lambda/beta^2) * Cov_hat(V, r_c)      # sample cov, 1/(B-1)  (Eq 11)
    grad_lambda L = 1 - mean(V) - delta
    beta   <- clamp_[bmin,bmax]( beta - eta_beta * grad_beta L )
    lambda <- [ lambda + eta_lambda * grad_lambda L ]_+
smaller delta => tighter (want more in-band); e.g. delta=0.2 targets >=80% in-band.

Note V_i and r_c_i depend only on the current solver theta (not on beta), so
within one iteration mean(V)/Cov are constants; beta moves via the -1/beta and
-lambda/beta^2 terms over NGDA micro-steps, and the real "did beta change the
in-band fraction" feedback shows up NEXT iter when the walk re-samples with the
new beta. beta/lambda persist across iters (restored on resume).

Defaults: r_min=0.3, r_max=0.8, delta=0.5, beta0=1.0, lambda0=1.0,
eta_beta=eta_lambda=0.1, NGDA=20, beta clamp [0.02, 2.0].
GPU: base=0, solver DP=1/4/5, verl=2/3, eval=6.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcmc_deo_vllm as deo

deo.config.STORAGE_ROOT = os.environ["STORAGE_PATH"]
deo.config.RZERO_DIR = os.environ.get("RZERO_DIR", "/workspace")
deo.config.HF_USER = os.environ.get("HUGGINGFACENAME", "yuyang322")
deo.config.MODEL_ABBR = os.environ.get("DEO_ABBR", "deo_adaptive")
deo.config.MCMC_STEPS = int(os.environ.get("DEO_MCMC_STEPS", "5"))     # canonical walk
deo.config.NUM_ITERATIONS = int(os.environ.get("DEO_NUM_ITERS", deo.config.NUM_ITERATIONS))
deo.config.TOTAL_QUESTIONS = int(os.environ.get("DEO_TOTAL_Q", deo.config.TOTAL_QUESTIONS))
deo.config.LAMBDA_REP = float(os.environ.get("DEO_LAMBDA_REP", deo.config.LAMBDA_REP))

# --- adaptive-temperature knobs (DEO paper §1.4) ---
R_MIN = float(os.environ.get("DEO_RMIN", "0.3"))
R_MAX = float(os.environ.get("DEO_RMAX", "0.8"))
DELTA = float(os.environ.get("DEO_DELTA", "0.5"))         # VIOLATION rate cap (out-of-band frac); smaller=more in-band
ETA_BETA = float(os.environ.get("DEO_ETA_BETA", "0.1"))
ETA_LAMBDA = float(os.environ.get("DEO_ETA_LAMBDA", "0.1"))
BETA0 = float(os.environ.get("DEO_BETA0", "1.0"))
LAMBDA0 = float(os.environ.get("DEO_LAMBDA0", "1.0"))
NGDA = int(os.environ.get("DEO_NGDA", "20"))
# V(X) and r_c(X) are BATCH-level quantities (r_rep = within-batch BLEU clusters),
# so the B samples X_1..X_K used to estimate Cov(V, r_c) must each be a *chunk*
# (batch) of questions, not a single question. Split the iter's MCMC pool into
# chunks of CHUNK_SIZE and compute one (V_i, R_i) pair per chunk.
CHUNK_SIZE = int(os.environ.get("DEO_CHUNK_SIZE", "64"))
_clamp = os.environ.get("DEO_BETA_CLAMP", "0.02,2.0").split(",")
BMIN, BMAX = float(_clamp[0]), float(_clamp[1])

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


def chunk_stats(train_data):
    """Split the iter's MCMC pool into K chunks (each a 'batch of questions'
    X_i as in paper §1.4) and return per-chunk (V_i, R_i):
        V_i = fraction of chunk with r_min <= r_unc <= r_max   (in-band fraction)
        R_i = mean r_c over chunk, r_c = max(0, r_unc - LAMBDA_REP*r_rep),
              r_rep = within-chunk BLEU cluster size / |chunk|.
    Also returns the overall mean r_unc (diagnostic).
    """
    n = len(train_data)
    cs = max(2, min(CHUNK_SIZE, n))
    K = max(2, n // cs)
    Vs, Rs, all_runc = [], [], []
    for i in range(K):
        lo = i * cs
        hi = n if i == K - 1 else (i + 1) * cs
        chunk = train_data[lo:hi]
        if len(chunk) < 2:
            continue
        qs = [d["question"] for d in chunk]
        r_unc = np.asarray([float(d["r_unc"]) for d in chunk], dtype=float)
        _, _, cluster_size = deo.calculate_batch_energy(qs, r_unc)   # within-chunk BLEU
        r_rep = cluster_size / len(chunk)
        inband = ((r_unc >= R_MIN) & (r_unc <= R_MAX)).astype(float)
        Vs.append(float(inband.mean()))   # V(X): in-band FRACTION of the batch
        # batch reward = MCMC energy of the batch = sum of PER-QUESTION clamped r_c
        # r_c(X) = sum_x max(0, r_unc(x) - lambda_rep * r_rep(x))  (matches energy_from_state;
        # NOTE: paper §1.4 writes max(0, sum(...)) with the max OUTSIDE — that is inconsistent
        # with p.2's per-question r_c and with the sampler, so we clamp per-question then sum.)
        Rs.append(float(np.maximum(0.0, r_unc - deo.config.LAMBDA_REP * r_rep).sum()))
        all_runc.append(r_unc)
    runc_mean = float(np.concatenate(all_runc).mean()) if all_runc else 0.0
    return np.asarray(Vs), np.asarray(Rs), runc_mean


def update_beta_lambda(beta, lam, train_data):
    """GDA update of (beta, lambda) from this iter's B=K batch samples (paper §1.4).

    Returns (new_beta, new_lambda, stats).
    """
    V, R, runc_mean = chunk_stats(train_data)
    B = len(V)
    Vbar = float(V.mean()) if B else 0.0
    # sample covariance Cov_hat(V, r_c) over the K batch samples, 1/(B-1)
    if B > 1:
        cov = float(np.cov(V, R, ddof=1)[0, 1])
    else:
        cov = 0.0
    b, l = beta, lam
    for _ in range(NGDA):
        # constraint on VIOLATION rate: 1 - E[V] <= delta  (paper §1.4)
        # L = -log b + lambda*(1 - E[V] - delta);  dE[V]/db = -(1/b^2)Cov(V,r_c)
        g_beta = -1.0 / b + (l / (b * b)) * cov          # grad_beta L (Eq 11)
        b = b - ETA_BETA * g_beta
        b = min(BMAX, max(BMIN, b))
        g_lam = 1.0 - Vbar - DELTA                       # grad_lambda L = 1 - mean(V) - delta
        l = max(0.0, l + ETA_LAMBDA * g_lam)
    stats = {"beta_used": beta, "Vbar": Vbar, "violation": 1.0 - Vbar, "cov_V_rc": cov,
             "r_unc_mean": runc_mean, "R_mean": float(R.mean()) if B else 0.0,
             "beta_new": b, "lambda_new": l, "K_chunks": B}
    return b, l, stats


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
    beta_traj_path = f"{deo.config.STORAGE_ROOT}/beta_traj_{deo.config.MODEL_ABBR}.json"
    eval_history = {}
    beta_traj = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            eval_history = json.load(f)
    if os.path.exists(beta_traj_path):
        with open(beta_traj_path) as f:
            beta_traj = json.load(f)

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)
        with open(beta_traj_path, "w") as f:
            json.dump(beta_traj, f, indent=2)

    print(f"[adaptive] band=[{R_MIN},{R_MAX}] delta={DELTA} beta0={BETA0} lambda0={LAMBDA0} "
          f"eta_b={ETA_BETA} eta_l={ETA_LAMBDA} ngda={NGDA} clamp=[{BMIN},{BMAX}]", flush=True)

    # restore (beta, lambda) from last recorded iter, else start at (beta0, lambda0)
    beta, lam = BETA0, LAMBDA0
    if beta_traj:
        last = beta_traj[sorted(beta_traj, key=lambda k: int(k.split("_")[1]))[-1]]
        beta, lam = last["beta_new"], last["lambda_new"]
        print(f"[adaptive] resumed beta={beta:.4f} lambda={lam:.4f}", flush=True)

    if "iter_0_baseline" not in eval_history:
        eval_history["iter_0_baseline"] = native_eval_math500(deo.config.MODEL_NAME, "iter 0 baseline")
        save()

    WARM_START = os.getenv("DEO_WARM_START", "0") == "1"
    if WARM_START:
        print("[adaptive] WARM START enabled: next-iter MCMC starts from prev-iter mutated pool", flush=True)
    prev_pool = None

    current_solver = deo.config.MODEL_NAME
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        done_key = f"iter_{it}"
        ckpt = f"{deo.config.STORAGE_ROOT}/models/{deo.config.MODEL_ABBR}_solver_v{it}/global_step_15/actor/huggingface"
        if done_key in eval_history and os.path.isdir(ckpt):
            print(f"=== iter {it} already complete (acc={eval_history[done_key]}), skipping ===", flush=True)
            current_solver = ckpt
            if WARM_START:  # resume-safe: reload this iter's pool so the next iter can warm-start
                mcmc_json = f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.json"
                if os.path.exists(mcmc_json):
                    with open(mcmc_json) as f:
                        prev_pool = json.load(f)
            if it < deo.config.NUM_ITERATIONS:
                native_reload_vllm_solver(ckpt)
            continue

        deo.config.BETA = beta
        print(f"\n{'='*60}\n=== Iteration {it}/{deo.config.NUM_ITERATIONS} "
              f"(adaptive: BETA={beta:.4f}, lambda={lam:.4f}) ===\n{'='*60}", flush=True)
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.log"

        train_data = deo.generate_batch_mcmc(
            tokenizer, deo.config.TOTAL_QUESTIONS, log_path,
            init_pool=prev_pool if WARM_START else None,
        )
        prev_pool = train_data   # warm-start seed for the next iter
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        # --- adaptive temperature: update (beta, lambda) from this iter's samples ---
        beta, lam, stats = update_beta_lambda(beta, lam, train_data)
        beta_traj[f"iter_{it}"] = stats
        print(f"[adaptive] iter {it}: in-band={stats['Vbar']:.3f} violation={stats['violation']:.3f} "
              f"(cap delta={DELTA}) cov(V,r_c)={stats['cov_V_rc']:.4f} r_unc_mean={stats['r_unc_mean']:.3f} "
              f"-> beta {stats['beta_used']:.4f}->{beta:.4f}, lambda->{lam:.4f}", flush=True)
        save()

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

    print("\nFINAL TRAJECTORY (adaptive-temperature DEO):", flush=True)
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f} ({acc*100:.2f}%)", flush=True)
    print("\nBETA/LAMBDA TRAJECTORY:", flush=True)
    for k in sorted(beta_traj, key=lambda x: int(x.split("_")[1])):
        s = beta_traj[k]
        print(f"  {k:10s} beta {s['beta_used']:.4f}->{s['beta_new']:.4f} "
              f"lambda->{s['lambda_new']:.4f} Vbar={s['Vbar']:.3f} cov={s['cov_V_rc']:.4f}", flush=True)


if __name__ == "__main__":
    main()

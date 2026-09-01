"""SGLD-DEO — NATIVE runner (DEO_SGLD.pdf, Algorithm 1).

Challenger-free self-evolution where the question distribution itself MOVES:
per-question soft-prefix latents z_i steer the frozen base model's generation,
sampled by SGLD toward the latent Gibbs best-response (uncertainty − repetition).
Replaces the discrete MCMC walk of the canonical runner; filtering, pseudo-
labeling, verl GRPO and eval are identical to the walk runs.

GPU layout: HF base (generation + score backprop) = GPU 7 (free in the native
layout), solver vLLM DP = 1/4/5, verl = 2/3, eval = 6. The base vLLM on GPU 0
is still launched by start_vllm_native.sh but unused here (kept to avoid
touching the launcher; its 0.85 mem util on GPU0 is idle).

Env knobs:
  SGLD_STEPS (SWEEPS, default 10; one sweep = every latent updated once)
  SGLD_ETA (1e-3)  SGLD_SIGMA (1.0)  SGLD_TAU (0.1 = beta; exact for the separable term)
  SGLD_ALPHA ("" = auto: 5% of prompt-embedding RMS)  SGLD_LAMBDA_REP (0 = off, default)
  SGLD_MINIBATCH (500)  SGLD_GEN_BS (16)  SGLD_SCORE_BS (4)  SGLD_MAXTOK (1024)
  DEO_TOTAL_Q (n, default 2000)  DEO_NUM_ITERS (5)
"""
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcmc_deo_vllm as deo

deo.config.STORAGE_ROOT = os.environ["STORAGE_PATH"]
deo.config.RZERO_DIR = os.environ.get("RZERO_DIR", "/workspace")
deo.config.HF_USER = os.environ.get("HUGGINGFACENAME", "yuyang322")
deo.config.MODEL_ABBR = os.environ.get("DEO_ABBR", "deo_sgld")
deo.config.NUM_ITERATIONS = int(os.environ.get("DEO_NUM_ITERS", deo.config.NUM_ITERATIONS))
deo.config.TOTAL_QUESTIONS = int(os.environ.get("DEO_TOTAL_Q", deo.config.TOTAL_QUESTIONS))
# Repetition penalty DISABLED by default for the first correctness experiment
# (review §P0/P1: per-question credit is exact only for the separable term).
SGLD_LAMBDA_REP = float(os.environ.get("SGLD_LAMBDA_REP", "0"))

SGLD_STEPS = int(os.environ.get("SGLD_STEPS", "10"))
SGLD_ETA = float(os.environ.get("SGLD_ETA", "1e-3"))
SGLD_SIGMA = float(os.environ.get("SGLD_SIGMA", "1.0"))
SGLD_TAU = float(os.environ.get("SGLD_TAU", "0.1"))
SGLD_MINIBATCH = int(os.environ.get("SGLD_MINIBATCH", "500"))
SGLD_GEN_BS = int(os.environ.get("SGLD_GEN_BS", "16"))
SGLD_SCORE_BS = int(os.environ.get("SGLD_SCORE_BS", "4"))
SGLD_MAXTOK = int(os.environ.get("SGLD_MAXTOK", "1024"))
SGLD_ALPHA = os.environ.get("SGLD_ALPHA", "")   # "" = auto (5% of prompt-embedding RMS)
SGLD_GPU = os.environ.get("SGLD_GPU", "7")

VERL_GPUS = os.environ.get("DEO_VERL_GPUS", "2,3")
EVAL_GPU = os.environ.get("DEO_EVAL_GPU", "6")
VERL_OVERRIDES = ["data.rollout_batch_size=64"] + os.environ.get("DEO_VERL_EXTRA", "").split()
PIDDIR = os.environ.get("VLLM_PIDDIR", "/tmp/vllm_pids")
LOGDIR = os.environ.get("VLLM_LOGDIR", "/tmp/vllm_logs")


# ---------- native vllm/verl/eval scaffolding (mirrors curriculum main) ----------
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


# ---------- SGLD question generation phase (replaces generate_batch_mcmc) ----------
def sgld_generate_pool(sampler, tokenizer, num_questions, log_path, it):
    """Run K_SGLD SGLD sweeps, then sample the final pool from z^{K_SGLD}."""
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("=" * 50 + f"\nSGLD Phase (iter {it})\n" + "=" * 50 + "\n")
    forbidden = ["prove that", "show that", "justify", "explain", "true or false", "yes or no"]

    def parse_valid(raw_texts):
        """completion text -> (question, gt) or None (parse+garbage+leak gate)."""
        out = []
        for t in raw_texts:
            q, gt = deo.extract_challenger_output(t)
            ok = (q and len(q) > 30 and not any(w in q.lower() for w in forbidden)
                  and not (deo.config.STRIP_LEAKS and deo.question_is_leaky(q)))
            out.append((q, gt) if ok else None)
        return out

    import torch as _t
    g = _t.Generator().manual_seed(1234 + it)
    for sweep in range(SGLD_STEPS):
        # one sweep = every latent updated exactly once (review §P1)
        for mb_i, idxs in enumerate(sampler.sweep_minibatches(SGLD_MINIBATCH, generator=g)):
            t0 = time.time()
            raw, ids_list = sampler.generate(idxs)          # exact sampled ids kept
            parsed = parse_valid(raw)
            valid_ix = [i for i, p in enumerate(parsed) if p is not None]
            qs = [parsed[i][0] for i in valid_ix]
            r_uncs, p_hats = [], []
            if qs:
                r_uncs, p_hats, _pl = deo.evaluate_r_unc_vllm(tokenizer, qs)[:3]
            rrep = [0.0] * len(qs)
            if SGLD_LAMBDA_REP > 0 and len(qs) > 1:         # off by default (surrogate)
                toks = [q.split() for q in qs]
                for a in range(len(qs)):
                    c = 1
                    for b in range(len(qs)):
                        if a != b and deo._is_close(toks[a], toks[b]):
                            c += 1
                    rrep[a] = c / len(qs)
            rewards = {}
            for j, i in enumerate(valid_ix):
                rewards[idxs[i]] = r_uncs[j] - SGLD_LAMBDA_REP * rrep[j]
            for i, p in enumerate(parsed):
                if p is None:
                    rewards[idxs[i]] = 0.0                  # malformed → zero utility
            grads, logps = sampler.score_grads(idxs, ids_list, score_bs=SGLD_SCORE_BS)
            stats = sampler.step(idxs, rewards, grads, eta=SGLD_ETA)
            inband = sum(1 for j in range(len(qs))
                         if deo.config.MIN_SCORE <= p_hats[j] <= deo.config.MAX_SCORE)
            mean_lp = sum(logps.values()) / max(1, len(logps))
            msg = (f"[sgld] iter{it} sweep {sweep+1}/{SGLD_STEPS} mb {mb_i+1}: "
                   f"valid {len(valid_ix)}/{len(idxs)} in-band {inband}/{max(1,len(qs))} "
                   f"adv={stats['mean_adv']:+.3f} |g/τ|={stats['mean_gnorm']:.2e} "
                   f"z_rms={stats['latent_rms']:.3f} Δ/prompt={stats['delta_to_prompt']:.3f} "
                   f"drift={stats['mean_driftnorm']:.2e} logp={mean_lp:.0f} "
                   f"baseline={stats['baseline']:.3f} ({time.time()-t0:.0f}s)")
            print(msg, flush=True)
            log_file.write(msg + "\n"); log_file.flush()

    # final pool: one question per z_i, scored for filtering/labels
    print(f"[sgld] sampling final pool of {num_questions} from z^{SGLD_STEPS}...", flush=True)
    pool_q, pool_gt = [None] * num_questions, [None] * num_questions
    all_ix = list(range(num_questions))
    raw, _ids = sampler.generate(all_ix)
    parsed = parse_valid(raw)
    need_resample = [i for i, p in enumerate(parsed) if p is None]
    for r in range(3):                                       # up to 3 resample rounds
        if not need_resample:
            break
        raw2, _ids2 = sampler.generate(need_resample)
        parsed2 = parse_valid(raw2)
        still = []
        for i, p in zip(need_resample, parsed2):
            if p is not None:
                parsed[i] = p
            else:
                still.append(i)
        need_resample = still
    kept = [i for i, p in enumerate(parsed) if p is not None]
    log_file.write(f"[sgld] final pool: {len(kept)}/{num_questions} valid after resampling\n")
    qs = [parsed[i][0] for i in kept]
    r_uncs, p_hats, pseudos = deo.evaluate_r_unc_vllm(tokenizer, qs)[:3]
    train_data = []
    for j, i in enumerate(kept):
        train_data.append({"question": qs[j], "gt": parsed[i][1], "p_hat": p_hats[j],
                           "pseudo_label": pseudos[j], "r_unc": r_uncs[j],
                           "topic": "sgld"})
    log_file.close()
    return train_data


def main():
    import torch
    from transformers import AutoModelForCausalLM
    from sgld_soft_prefix import SoftPrefixSGLD

    deo.config.HF_TOKEN = deo.load_hf_token()
    deo.hf_login(token=deo.config.HF_TOKEN)
    for d in ["datasets", "logs", "models", "evaluation", "temp_results", "generated_question"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{d}", exist_ok=True)

    tokenizer = deo.AutoTokenizer.from_pretrained(deo.config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # HF frozen base for soft-prefix generation + score backprop (GPU SGLD_GPU)
    device = f"cuda:{SGLD_GPU}"
    print(f"[sgld] loading HF base {deo.config.MODEL_NAME} on {device}...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        deo.config.MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    # p0 = the SAME challenger prompt as the walk runs (topic instruction omitted:
    # diversity now comes from n independent latents + SGLD noise)
    user_p = "Generate one new, challenging reasoning question now."
    p0 = deo.apply_chat_template(tokenizer, deo.CHALLENGER_SYSTEM_PROMPT, user_p)
    sampler = SoftPrefixSGLD(base, tokenizer, p0, n=deo.config.TOTAL_QUESTIONS,
                             alpha=(float(SGLD_ALPHA) if SGLD_ALPHA else None),
                             sigma=SGLD_SIGMA, tau=SGLD_TAU, eta=SGLD_ETA,
                             device=device, gen_bs=SGLD_GEN_BS,
                             max_new_tokens=SGLD_MAXTOK)
    zpath = f"{deo.config.STORAGE_ROOT}/datasets/sgld_Z_{deo.config.MODEL_ABBR}.pt"
    sampler.load(zpath)   # warm-start across restarts/outer iters (Algorithm 1 step 3b)

    print(f"[sgld] n={deo.config.TOTAL_QUESTIONS} K={sampler.K} d={sampler.d} "
          f"sweeps={SGLD_STEPS} mb={SGLD_MINIBATCH} eta={SGLD_ETA} sigma={SGLD_SIGMA} "
          f"tau={SGLD_TAU} lambda_rep={SGLD_LAMBDA_REP} alpha={sampler.alpha:.5f}", flush=True)

    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_{deo.config.MODEL_ABBR}.json"
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
        ckpt = (f"{deo.config.STORAGE_ROOT}/models/{deo.config.MODEL_ABBR}_solver_v{it}"
                f"/global_step_15/actor/huggingface")
        if done_key in eval_history and os.path.isdir(ckpt):
            print(f"=== iter {it} already complete, skipping ===", flush=True)
            current_solver = ckpt
            if it < deo.config.NUM_ITERATIONS:
                native_reload_vllm_solver(ckpt)
            continue

        print(f"\n{'='*60}\n=== SGLD-DEO Iteration {it}/{deo.config.NUM_ITERATIONS} ===\n{'='*60}",
              flush=True)
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        log_path = f"{deo.config.STORAGE_ROOT}/logs/sgld_iter_{it}_{deo.config.MODEL_ABBR}.log"

        train_data = sgld_generate_pool(sampler, tokenizer, deo.config.TOTAL_QUESTIONS,
                                        log_path, it)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.json",
                  "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        sampler.save(zpath)   # persist Z (warm-start next iter / restart)

        deo.filter_and_push(train_data, exp_name, exp_name)
        merged_ckpt = native_run_verl(current_solver, f"{deo.config.HF_USER}/{exp_name}",
                                      exp_name, extra_verl_overrides=VERL_OVERRIDES)
        eval_history[done_key] = native_eval_math500(merged_ckpt, f"iter {it}")
        save()
        if it < deo.config.NUM_ITERATIONS:
            native_reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\nFINAL TRAJECTORY (SGLD-DEO):", flush=True)
    for label, acc in eval_history.items():
        print(f"  {label}: {acc}", flush=True)


if __name__ == "__main__":
    main()

"""Planning-SGLD DEO — NATIVE runner (DEO_SGLD.pdf v2, Algorithm 1). MODE=plansgld.

Interpretable challenger-free self-evolution: M particles of planning logits
A ∈ R^{K×L} are sampled by annealed SGLD toward the plan-level best response;
every generation renders a HARD plan into a text instruction for the frozen base
model via the ordinary vLLM server (no embeddings, no backprop anywhere).
Filtering, pseudo-labeling, verl GRPO and eval identical to all walk runs.

GPU layout identical to curriculum runs: base vLLM=0 (used for generation),
solver DP=1/4/5, verl=2/3, eval=6.

Env knobs:
  PLAN_M (8 particles)  PLAN_B (25 probes/particle/step)  PLAN_S (30 SGLD steps/iter)
  PLAN_ETA (0.3)  PLAN_T0/T1 (0.2/0.02, reward-scale aware)  PLAN_GAMMA0/GAMMA1 (1.0/0.3)
  PLAN_TAUPLAN (0.05)  PLAN_LAMBDA_A (1e-3)  PLAN_LAMBDA_REP (1.0, pooled MB batch)
  PLAN_LEN_PENALTY (1: out-of-bin length => reward 0 for that probe)
  DEO_TOTAL_Q (2000)  DEO_NUM_ITERS (5)
"""
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcmc_deo_vllm as deo
from plan_sgld import PlanSGLD, render_plan, LENGTH_BINS, LENGTH_AXIS

deo.config.STORAGE_ROOT = os.environ["STORAGE_PATH"]
deo.config.RZERO_DIR = os.environ.get("RZERO_DIR", "/workspace")
deo.config.HF_USER = os.environ.get("HUGGINGFACENAME", "yuyang322")
deo.config.MODEL_ABBR = os.environ.get("DEO_ABBR", "deo_plansgld")
deo.config.NUM_ITERATIONS = int(os.environ.get("DEO_NUM_ITERS", deo.config.NUM_ITERATIONS))
deo.config.TOTAL_QUESTIONS = int(os.environ.get("DEO_TOTAL_Q", deo.config.TOTAL_QUESTIONS))

PLAN_M = int(os.environ.get("PLAN_M", "8"))
PLAN_B = int(os.environ.get("PLAN_B", "25"))
PLAN_S = int(os.environ.get("PLAN_S", "30"))
PLAN_ETA = float(os.environ.get("PLAN_ETA", "0.3"))
PLAN_T0 = float(os.environ.get("PLAN_T0", "0.2"))
PLAN_T1 = float(os.environ.get("PLAN_T1", "0.02"))
PLAN_G0 = float(os.environ.get("PLAN_GAMMA0", "1.0"))
PLAN_G1 = float(os.environ.get("PLAN_GAMMA1", "0.3"))
PLAN_TAUPLAN = float(os.environ.get("PLAN_TAUPLAN", "0.05"))
PLAN_LAMBDA_A = float(os.environ.get("PLAN_LAMBDA_A", "1e-3"))
PLAN_LAMBDA_REP = float(os.environ.get("PLAN_LAMBDA_REP", "1.0"))
PLAN_LEN_PENALTY = os.environ.get("PLAN_LEN_PENALTY", "1") == "1"

VERL_GPUS = os.environ.get("DEO_VERL_GPUS", "2,3")
EVAL_GPU = os.environ.get("DEO_EVAL_GPU", "6")
VERL_OVERRIDES = ["data.rollout_batch_size=64"] + os.environ.get("DEO_VERL_EXTRA", "").split()
PIDDIR = os.environ.get("VLLM_PIDDIR", "/tmp/vllm_pids")
LOGDIR = os.environ.get("VLLM_LOGDIR", "/tmp/vllm_logs")


# ---------- native scaffolding (identical to curriculum main) ----------
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
    for _g, port, _n in deo.config.SOLVER_INSTANCES:
        subprocess.run(["pkill", "-9", "-f", f"--port {port}"], check=False)
    time.sleep(8)
    for gpu_id, port, _n in deo.config.SOLVER_INSTANCES:
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
    subprocess.run(["python3", "evaluation/generate.py", "--model", model_path,
                    "--dataset", "math"], cwd=deo.config.RZERO_DIR, check=True, env=env)
    results_path = (f"{deo.config.STORAGE_ROOT}/evaluation/"
                    f"{model_path.replace('/', '_')}/results_math.json")
    old_avg, new_avg, n_bumped = deo.gpt_recheck_math500(results_path)
    print(f"=== {label}: MATH-500 acc = {new_avg:.4f} ({new_avg*100:.2f}%) "
          f"[raw {old_avg:.4f}, +{n_bumped} GPT-bumped] ===\n", flush=True)
    return new_avg


deo.eval_math500 = native_eval_math500

FORBIDDEN = ["prove that", "show that", "justify", "explain", "true or false", "yes or no"]


def gen_questions(tokenizer, instructions):
    """Batched base-model generation from rendered plan instructions (vLLM, text prompts)."""
    prompts = [deo.apply_chat_template(tokenizer, deo.CHALLENGER_SYSTEM_PROMPT, ins)
               for ins in instructions]
    resp = deo.base_client().completions.create(
        model=deo.config.MODEL_NAME, prompt=prompts,
        max_tokens=1536, temperature=1.0, top_p=0.95)
    out = []
    for c in resp.choices:
        q, gt = deo.extract_challenger_output(c.text)
        ok = (q and len(q) > 30 and not any(w in q.lower() for w in FORBIDDEN)
              and not (deo.config.STRIP_LEAKS and deo.question_is_leaky(q)))
        out.append((q, gt) if ok else None)
    return out


def probe_and_score(tokenizer, plans_rows):
    """One planning step: generate B probes per particle, pool for the repetition
    penalty (Eq 22 pooling), return per-particle Û (tent-r_unc − λ_rep·r_rep;
    malformed / out-of-length-bin probes contribute 0)."""
    M = len(plans_rows)
    instructions, owner = [], []
    for p, rows in enumerate(plans_rows):
        instructions += [render_plan(rows)] * PLAN_B
        owner += [p] * PLAN_B
    parsed = gen_questions(tokenizer, instructions)
    valid_ix = [i for i, x in enumerate(parsed) if x is not None]
    qs = [parsed[i][0] for i in valid_ix]
    r_uncs = []
    if qs:
        r_uncs, _ph, _pl = deo.evaluate_r_unc_vllm(tokenizer, qs)[:3]
    # pooled repetition penalty across ALL particles' probes
    toks = [q.split() for q in qs]
    rep = [0.0] * len(qs)
    if PLAN_LAMBDA_REP > 0 and len(qs) > 1:
        for a in range(len(qs)):
            c = 1
            for b in range(len(qs)):
                if a != b and deo._is_close(toks[a], toks[b]):
                    c += 1
            rep[a] = c / len(qs)
    # per-question utility; length-bin conformity gate
    util = {}
    for j, i in enumerate(valid_ix):
        u = max(0.0, r_uncs[j] - PLAN_LAMBDA_REP * rep[j])
        if PLAN_LEN_PENALTY:
            lo, hi = LENGTH_BINS[plans_rows[owner[i]][LENGTH_AXIS]]
            ntok = len(tokenizer(qs[j]).input_ids)
            if ntok < 0.5 * lo or ntok > 2.0 * hi:
                u = 0.0                       # far outside the plan's bin: no reward
        util[i] = u
    U = []
    for p in range(M):
        mine = [util.get(i, 0.0) for i in range(p * PLAN_B, (p + 1) * PLAN_B)]
        U.append(sum(mine) / PLAN_B)          # malformed probes count as 0 (Eq 15-16)
    n_valid = len(valid_ix)
    return U, n_valid, len(instructions)


def main():
    deo.config.HF_TOKEN = deo.load_hf_token()
    deo.hf_login(token=deo.config.HF_TOKEN)
    for d in ["datasets", "logs", "models", "evaluation", "temp_results", "generated_question"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{d}", exist_ok=True)
    tokenizer = deo.AutoTokenizer.from_pretrained(deo.config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    planner = PlanSGLD(M=PLAN_M, eta=PLAN_ETA, T0=PLAN_T0, T1=PLAN_T1,
                       gamma0=PLAN_G0, gamma1=PLAN_G1, tau_plan=PLAN_TAUPLAN,
                       lambda_A=PLAN_LAMBDA_A)
    apath = f"{deo.config.STORAGE_ROOT}/datasets/plan_A_{deo.config.MODEL_ABBR}.pt"
    planner.load(apath)
    print(f"[plan] M={PLAN_M} B={PLAN_B} S={PLAN_S} eta={PLAN_ETA} "
          f"T:{PLAN_T0}->{PLAN_T1} gamma:{PLAN_G0}->{PLAN_G1} "
          f"tau_plan={PLAN_TAUPLAN} lambda_A={PLAN_LAMBDA_A} "
          f"lambda_rep={PLAN_LAMBDA_REP}", flush=True)

    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_{deo.config.MODEL_ABBR}.json"
    eval_history = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            eval_history = json.load(f)

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    if "iter_0_baseline" not in eval_history:
        eval_history["iter_0_baseline"] = native_eval_math500(deo.config.MODEL_NAME,
                                                              "iter 0 baseline")
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

        print(f"\n{'='*60}\n=== Planning-SGLD Iteration {it}/{deo.config.NUM_ITERATIONS} "
              f"===\n{'='*60}", flush=True)
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        log_path = f"{deo.config.STORAGE_ROOT}/logs/plan_iter_{it}_{deo.config.MODEL_ABBR}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        g = torch.Generator().manual_seed(4321 + it)

        if it > 1:
            planner.add_refresh_noise(0.05)   # Alg 1 step 3c

        # ---- S annealed planning steps ----
        for s in range(PLAN_S):
            t0 = time.time()
            plans, Ps = [], []
            _T, gamma = planner.temps(s, PLAN_S)
            for p in range(PLAN_M):
                rows, P = planner.sample_plan(p, gamma, generator=g)
                plans.append(rows); Ps.append(P)
            U, n_valid, n_total = probe_and_score(tokenizer, plans)
            stats = None
            for p in range(PLAN_M):
                stats = planner.step(p, plans[p], Ps[p], U[p], s, PLAN_S, generator=g)
            msg = (f"[plan] iter{it} step {s+1}/{PLAN_S}: valid {n_valid}/{n_total} "
                   f"U(mean)={sum(U)/len(U):.3f} U(max)={max(U):.3f} "
                   f"T={stats['T']:.2f} gamma={stats['gamma']:.2f} ({time.time()-t0:.0f}s)")
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.write(json.dumps({"step": s + 1, "plans": [planner.describe(r) for r in plans],
                                       "U": [round(u, 4) for u in U]}) + "\n")
            log_file.flush()

        # ---- quantize + report the learned plans (interpretability payoff) ----
        hard = [planner.quantize(p) for p in range(PLAN_M)]
        for p, rows in enumerate(hard):
            line = f"[plan] iter{it} FINAL particle {p}: {planner.describe(rows)}"
            print(line, flush=True); log_file.write(line + "\n")
        log_file.write(json.dumps(planner.logits_snapshot()) + "\n")
        planner.save(apath)

        # ---- generate the training pool from the M hard plans ----
        n = deo.config.TOTAL_QUESTIONS
        per = [n // PLAN_M] * PLAN_M
        for i in range(n - sum(per)):
            per[i] += 1
        train_data = []
        for p, rows in enumerate(hard):
            need, tries = per[p], 0
            while need > 0 and tries < 4:
                batch = gen_questions(tokenizer, [render_plan(rows)] * min(need + 20, 300))
                got = [x for x in batch if x is not None][:need]
                if got:
                    qs = [q for q, _ in got]
                    r_uncs, p_hats, pseudos = deo.evaluate_r_unc_vllm(tokenizer, qs)[:3]
                    for j, (q, gt) in enumerate(got):
                        train_data.append({"question": q, "gt": gt, "p_hat": p_hats[j],
                                           "pseudo_label": pseudos[j], "r_unc": r_uncs[j],
                                           "topic": f"plan{p}"})
                    need -= len(got)
                tries += 1
        log_file.write(f"[plan] final pool: {len(train_data)}/{n}\n")
        log_file.close()
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.json",
                  "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        deo.filter_and_push(train_data, exp_name, exp_name)
        merged_ckpt = native_run_verl(current_solver, f"{deo.config.HF_USER}/{exp_name}",
                                      exp_name, extra_verl_overrides=VERL_OVERRIDES)
        eval_history[done_key] = native_eval_math500(merged_ckpt, f"iter {it}")
        save()
        if it < deo.config.NUM_ITERATIONS:
            native_reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\nFINAL TRAJECTORY (Planning-SGLD DEO):", flush=True)
    for label, acc in eval_history.items():
        print(f"  {label}: {acc}", flush=True)


if __name__ == "__main__":
    main()

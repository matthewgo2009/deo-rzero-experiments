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
  PLAN_TAUPLAN (0.05)  PLAN_LAMBDA_A (1e-3)
  PLAN_LAMBDA_REP (0 = UNCERTAINTY-ONLY default; >0 = pooled-diversity variant)
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
from plan_sgld import (PlanSGLD, render_plan, length_ok, gated_utility,
                       allocate_quota, coordinate_refine)

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
PLAN_TAUMIX = float(os.environ.get("PLAN_TAUMIX", "0.05"))   # final-quota softmax temp
PLAN_LAMBDA_A = float(os.environ.get("PLAN_LAMBDA_A", "1e-3"))
# 0 (default) => per-particle local utility is the EXACT score coefficient;
# >0 => repetition couples particles, so the runner switches to the shared
# pooled-batch reward for every particle (review P0-1).
PLAN_LAMBDA_REP = float(os.environ.get("PLAN_LAMBDA_REP", "0"))

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


def probe_and_score(tokenizer, plans_rows, B=None):
    """One planning step / validation pass. Gate order (audit P0-6):
        parse -> LENGTH gate -> solver rollouts -> pseudo/p_hat gate -> judge
    so cheap gates run first and the judge only sees in-band survivors.
    Judge calls are per-item fault-isolated with a circuit breaker: a widespread
    judge outage ABORTS the iteration rather than silently passing everything.
    Returns (U_local list, U_global, diag)."""
    from concurrent.futures import ThreadPoolExecutor
    B = B or PLAN_B
    M = len(plans_rows)
    instructions, owner = [], []
    for p, rows in enumerate(plans_rows):
        instructions += [render_plan(rows)] * B
        owner += [p] * B
    parsed = gen_questions(tokenizer, instructions)

    # length gate BEFORE solver rollouts
    lens, len_pass = {}, []
    for i, x in enumerate(parsed):
        if x is None:
            continue
        ntok = len(tokenizer(x[0], add_special_tokens=False).input_ids)
        lens[i] = ntok
        if length_ok(ntok, plans_rows[owner[i]]):
            len_pass.append(i)
    qs = [parsed[i][0] for i in len_pass]
    r_uncs, p_hats, pseudos = [], [], []
    if qs:
        r_uncs, p_hats, pseudos = deo.evaluate_r_unc_vllm(tokenizer, qs)[:3]

    # pseudo/p_hat band gate BEFORE the judge
    band_ix = [j for j in range(len(qs))
               if pseudos[j] not in (None, "", "None")
               and deo.config.MIN_SCORE <= p_hats[j] <= deo.config.MAX_SCORE]

    def _judge(j):
        try:
            return deo.judge_one_validity(qs[j], pseudos[j])
        except Exception:
            return "ERR"
    judge_ok, judge_err = {}, 0
    if band_ix:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for j, v in zip(band_ix, ex.map(_judge, band_ix)):
                if v == "ERR":
                    judge_err += 1
                    judge_ok[j] = False
                else:
                    judge_ok[j] = bool(v)
        if judge_err > 0.5 * len(band_ix):
            raise RuntimeError(f"[plan] judge endpoint outage: {judge_err}/{len(band_ix)} "
                               "errors — aborting instead of silently changing the objective")

    rep = [0.0] * len(qs)
    if PLAN_LAMBDA_REP > 0 and len(qs) > 1:
        toks = [q.split() for q in qs]
        for a in range(len(qs)):
            c = 1
            for b2 in range(len(qs)):
                if a != b2 and deo._is_close(toks[a], toks[b2]):
                    c += 1
            rep[a] = c / (M * B)

    util = {}
    for j in band_ix:
        i = len_pass[j]
        util[i] = gated_utility(r_uncs[j], rep[j], pseudos[j], p_hats[j],
                                judge_ok.get(j, False), True, PLAN_LAMBDA_REP,
                                deo.config.MIN_SCORE, deo.config.MAX_SCORE)
    U_local = []
    for p in range(M):
        mine = [util.get(i, 0.0) for i in range(p * B, (p + 1) * B)]
        U_local.append(sum(mine) / B)
    U_global = sum(util.values()) / (M * B)
    plen = {p: [lens[i] for i in range(p * B, (p + 1) * B) if i in lens] for p in range(M)}
    diag = {"valid": sum(1 for x in parsed if x is not None), "total": len(instructions),
            "len_ok": len(len_pass), "inband": len(band_ix),
            "judge_drop": sum(1 for j in band_ix if not judge_ok.get(j, False)) - judge_err,
            "judge_err": judge_err,
            "mean_len": {p: (sum(v) / len(v) if v else 0) for p, v in plen.items()}}
    return U_local, U_global, diag


def gen_final_pool(tokenizer, alloc, n):
    """Final training pool with POST-FILTER quota accounting (audit P0-4): a
    candidate counts against its plan's quota only after passing ALL the gates
    the downstream filter applies (parse + strict length + pseudo + p_hat band +
    LLM judge) and global exact-dedup. Shortfalls are refilled within a bounded
    budget; the runner asserts a verl-viable dataset size before upload."""
    from concurrent.futures import ThreadPoolExecutor
    train_data, seen = [], set()
    for p, (rows, quota) in enumerate(alloc):
        need, tries = quota, 0
        while need > 0 and tries < 8:
            tries += 1
            batch = gen_questions(tokenizer, [render_plan(rows)] * min(need * 3 + 30, 300))
            cands = []
            for x in batch:
                if x is None:
                    continue
                q, gt = x
                if q in seen:
                    continue
                ntok = len(tokenizer(q, add_special_tokens=False).input_ids)
                if length_ok(ntok, rows):
                    cands.append((q, gt, ntok))
            if not cands:
                continue
            qs = [q for q, _g, _n in cands]
            r_uncs, p_hats, pseudos = deo.evaluate_r_unc_vllm(tokenizer, qs)[:3]
            keep = [j for j in range(len(qs))
                    if pseudos[j] not in (None, "", "None")
                    and deo.config.MIN_SCORE <= p_hats[j] <= deo.config.MAX_SCORE]
            if keep:
                with ThreadPoolExecutor(max_workers=16) as ex:
                    verd = dict(zip(keep, ex.map(
                        lambda j: bool(deo.judge_one_validity(qs[j], pseudos[j])), keep)))
                keep = [j for j in keep if verd.get(j, False)]
            for j in keep:
                if need <= 0:
                    break
                q, gt, _nt = cands[j]
                seen.add(q)
                train_data.append({"question": q, "gt": gt, "p_hat": p_hats[j],
                                   "pseudo_label": pseudos[j], "r_unc": r_uncs[j],
                                   "topic": f"plan{p}"})
                need -= 1
        if need > 0:
            print(f"[plan] WARNING: plan {rows} short by {need} after refill budget",
                  flush=True)
    # realized marginals (audit P0-2 option (a): observe, don't fix)
    from collections import Counter
    plan_counts = Counter(d["topic"] for d in train_data)
    print(f"[plan] realized post-filter plan counts: {dict(plan_counts)} "
          f"(target n={n}, achieved {len(train_data)})", flush=True)
    if len(train_data) < 64:
        raise RuntimeError(f"[plan] post-filter dataset has {len(train_data)} rows "
                           "< verl rollout_batch_size=64 — aborting before training")
    return train_data


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
            planner.add_refresh_noise(0.05)   # adaptive: contracts saturated particles

        # ---- S annealed planning steps ----
        n_gen = n_roll = 0
        for st in range(PLAN_S):
            t0 = time.time()
            plans, Ps = [], []
            _T, gamma = planner.temps(st, PLAN_S)
            for p in range(PLAN_M):
                rows, P = planner.sample_plan(p, gamma, generator=g)
                plans.append(rows); Ps.append(P)
            U_local, U_global, diag = probe_and_score(tokenizer, plans)
            n_gen += diag["total"]
            m_per = deo.config.USTAT_N if deo.config.CD_ENABLE else deo.config.M_SAMPLES
            n_roll += diag["valid"] * m_per
            stats = None
            for p in range(PLAN_M):
                # lambda_rep couples particles through pooled repetition -> shared
                # batch reward is the valid score coefficient (review P0-1);
                # separable case (lambda_rep=0) uses the exact local utility.
                U_use = U_global if PLAN_LAMBDA_REP > 0 else U_local[p]
                stats = planner.step(p, plans[p], Ps[p], U_use, st, PLAN_S, generator=g)
            ents = [round(planner.row_entropy(p), 2) for p in range(PLAN_M)]
            msg = (f"[plan] iter{it} step {st+1}/{PLAN_S}: valid {diag['valid']}/{diag['total']} "
                   f"judge_drop {diag['judge_drop']} inband {diag['inband']} "
                   f"len_ok {diag['len_ok']} U(mean)={sum(U_local)/len(U_local):.3f} "
                   f"U(max)={max(U_local):.3f} T={stats['T']:.2f} gamma={stats['gamma']:.2f} "
                   f"ent={ents} ({time.time()-t0:.0f}s)")
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.write(json.dumps({"step": st + 1,
                                       "plans": [planner.describe(r) for r in plans],
                                       "U_local": [round(u, 4) for u in U_local],
                                       "U_global": round(U_global, 4),
                                       "mean_len": diag["mean_len"]}) + "\n")
            log_file.flush()
        log_file.write(f"[plan] iter{it} planning budget: {n_gen} generations, "
                       f"~{n_roll} solver rollouts (excludes final-pool scoring)\n")

        # ---- quantize, VALIDATE with a fresh probe pass, dedup + weight quotas ----
        hard = [planner.quantize(p) for p in range(PLAN_M)]
        Uval, _g2, vdiag = probe_and_score(tokenizer, hard)   # fresh validation batch
        for p, rows in enumerate(hard):
            line = (f"[plan] iter{it} FINAL particle {p}: {planner.describe(rows)} "
                    f"validated_U={Uval[p]:.3f}")
            print(line, flush=True); log_file.write(line + "\n")
        # coordinate-refine the top validated plan (one sweep, K*L probe batches):
        # repairs the single-stuck-axis mode of short annealed chains.
        top = max(range(PLAN_M), key=lambda p: Uval[p])
        refined = coordinate_refine(
            hard[top], lambda variants: probe_and_score(tokenizer, variants)[0])
        if refined != hard[top]:
            Uref = probe_and_score(tokenizer, [refined])[0][0]
            line = (f"[plan] iter{it} REFINED top plan {planner.describe(hard[top])} -> "
                    f"{planner.describe(refined)} (U {Uval[top]:.3f} -> {Uref:.3f})")
            print(line, flush=True); log_file.write(line + "\n")
            if Uref >= Uval[top]:
                hard[top], Uval[top] = refined, Uref
        alloc = allocate_quota(hard, Uval, deo.config.TOTAL_QUESTIONS, tau_mix=PLAN_TAUMIX)
        if alloc is None:
            # all plans validated to zero (audit P0-3): one bigger revalidation, then abort
            print("[plan] all validated utilities are ZERO — retrying with 3x batch", flush=True)
            Uval, _g3, _d3 = probe_and_score(tokenizer, hard, B=3 * PLAN_B)
            alloc = allocate_quota(hard, Uval, deo.config.TOTAL_QUESTIONS, tau_mix=PLAN_TAUMIX)
            if alloc is None:
                raise RuntimeError("[plan] every quantized plan validated to zero utility "
                                   "twice — planner failed this iteration; not training on "
                                   "unusable plans")
        for rows, q in alloc:
            line = f"[plan] iter{it} QUOTA {q}: {planner.describe(rows)}"
            print(line, flush=True); log_file.write(line + "\n")
        log_file.write(json.dumps(planner.logits_snapshot()) + "\n")
        planner.save(apath)

        # ---- generate the training pool from the weighted plans (strict length) ----
        train_data = gen_final_pool(tokenizer, alloc, deo.config.TOTAL_QUESTIONS)
        lens = [len(tokenizer(d["question"], add_special_tokens=False).input_ids)
                for d in train_data]
        if lens:
            from plan_sgld import LENGTH_BINS
            marg = {f"{lo}-{hi}": sum(1 for x in lens if lo <= x < hi)
                    for lo, hi in LENGTH_BINS.values()}
            log_file.write(f"[plan] final pool: {len(train_data)} qs, "
                           f"len mean={sum(lens)/len(lens):.0f} min={min(lens)} "
                           f"max={max(lens)} realized_length_marginal={marg}\n")
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

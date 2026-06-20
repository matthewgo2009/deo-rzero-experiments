"""
Pipelined MCMC walk for DEO. Drop-in replacement for
mcmc_deo_vllm.generate_batch_mcmc() — same signature, same return type,
same semantics (sequential MH accept/reject is preserved). The only
change is that base-model proposals and solver M-vote scoring are done
in ONE big vllm call per phase, instead of being interleaved in a
75-batch inner loop per walk step.

This eliminates per-batch HTTP overhead and lets vllm run at peak
batched throughput continuously. Estimated speedup: ~5-6× on the MCMC
walk portion, which is ~50% of an iter, so ~1.7× per-iter wall clock.

Semantic invariant: MH accept/reject within a step is still sequential
(updates cluster_size + neighbor matrix per accept). What's batched is
the *proposal generation* and *solver scoring* — both of which only
depend on the pool state at step-start (since shuffled-indices are
non-overlapping within a step, so accepts don't change other proposals'
seeds).

Usage:
    import mcmc_pipelined
    train_data = mcmc_pipelined.generate_batch_mcmc_pipelined(tokenizer, N, log_path)

This module does NOT modify mcmc_deo_vllm.py. Existing callers continue
to use the canonical implementation unchanged.
"""
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo

# Re-export commonly used names for clarity
config = deo.config
apply_chat_template = deo.apply_chat_template
extract_challenger_output = deo.extract_challenger_output
extract_mutation_strategy = deo.extract_mutation_strategy
extract_solver_answer = deo.extract_solver_answer
neighbor_row_for = deo.neighbor_row_for
energy_from_state = deo.energy_from_state
calculate_batch_energy = deo.calculate_batch_energy
base_client = deo.base_client
solver_clients = deo.solver_clients
CHALLENGER_SYSTEM_PROMPT = deo.CHALLENGER_SYSTEM_PROMPT
MUTATOR_SYSTEM_PROMPT = deo.MUTATOR_SYSTEM_PROMPT
MUTATOR_USER_TEMPLATE = deo.MUTATOR_USER_TEMPLATE
RZERO_SOLVER_SYSTEM = deo.RZERO_SOLVER_SYSTEM
MATH_TOPICS = deo.MATH_TOPICS

# Chunking caps for big batch calls — protect against HTTP timeout / vllm
# memory pressure when sending several thousand prompts at once.
BASE_CHUNK_PROMPTS = 256       # ~256 prompts per base call
SOLVER_CHUNK_QUESTIONS = 100   # 100 questions × 9 = 900 prompts per solver shard call


# ---------------------------------------------------------------------------
# Big-batch base generation (chunked, single endpoint = vllm_base)
# ---------------------------------------------------------------------------
def _base_generate_big_batch(prompts, max_tokens=1536, temperature=1.0, top_p=0.95):
    """Generate completions for many prompts in one logical call.
    Chunked to BASE_CHUNK_PROMPTS to stay below HTTP timeout."""
    if not prompts:
        return []
    out_texts = []
    for s in range(0, len(prompts), BASE_CHUNK_PROMPTS):
        chunk = prompts[s:s + BASE_CHUNK_PROMPTS]
        resp = base_client().completions.create(
            model=config.MODEL_NAME,
            prompt=chunk,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        out_texts.extend(c.text for c in resp.choices)
    return out_texts


# ---------------------------------------------------------------------------
# Big-batch solver M-vote (uses DP across solver instances; chunked)
# ---------------------------------------------------------------------------
def _solver_score_big_batch(tokenizer, questions):
    """Same return as deo.evaluate_r_unc_vllm but chunks per-shard so each
    HTTP call stays under timeout even on large pools (~1500 questions =
    13500 prompts).

    Returns: (r_unc_list, p_hat_list, pseudo_list)
    """
    if not questions:
        return [], [], []
    m = config.M_SAMPLES
    clients = solver_clients()
    n_dp = len(clients)

    # Shard questions across DP instances
    base_sz, rem = divmod(len(questions), n_dp)
    shard_questions = []
    start = 0
    for i in range(n_dp):
        size = base_sz + (1 if i < rem else 0)
        shard_questions.append(questions[start:start + size])
        start += size

    def _run_shard(idx):
        qs = shard_questions[idx]
        if not qs:
            return idx, []
        out = []
        for s in range(0, len(qs), SOLVER_CHUNK_QUESTIONS):
            q_sub = qs[s:s + SOLVER_CHUNK_QUESTIONS]
            prompts = [
                apply_chat_template(tokenizer, RZERO_SOLVER_SYSTEM, q)
                for q in q_sub for _ in range(m)
            ]
            resp = clients[idx].completions.create(
                model=config.MODEL_NAME,
                prompt=prompts,
                max_tokens=config.SOLVER_MAX_TOKENS,
                temperature=config.SOLVER_TEMP,
                top_p=config.SOLVER_TOP_P,
                extra_body={"top_k": config.SOLVER_TOP_K},
            )
            out.extend(c.text for c in resp.choices)
        return idx, out

    texts_by_shard = [None] * n_dp
    with ThreadPoolExecutor(max_workers=n_dp) as ex:
        for fut in as_completed([ex.submit(_run_shard, i) for i in range(n_dp)]):
            idx, txt = fut.result()
            texts_by_shard[idx] = txt
    texts = [t for shard in texts_by_shard for t in shard]
    answers = [extract_solver_answer(t) for t in texts]

    r_unc_list, p_hat_list, pseudo_list = [], [], []
    for i in range(len(questions)):
        chunk = answers[i * m: (i + 1) * m]
        valid = [a for a in chunk if a is not None and a != "GUESSED_FAIL_FORMAT"]
        if not valid:
            r_unc_list.append(0.0)
            p_hat_list.append(0.0)
            pseudo_list.append(None)
            continue
        major_ans, count = Counter(valid).most_common(1)[0]
        p_hat = count / m
        r_unc = max(0.0, 1.0 - 2.0 * abs(p_hat - 0.5))
        is_garbage = (
            "text" in major_ans.lower()
            or "\\text" in major_ans
            or len(major_ans) > 100
        )
        r_unc_list.append(r_unc)
        p_hat_list.append(p_hat)
        pseudo_list.append(None if is_garbage else major_ans)

    return r_unc_list, p_hat_list, pseudo_list


# ---------------------------------------------------------------------------
# Init pool — one big base call, then one big solver call, repeat until N
# ---------------------------------------------------------------------------
def _init_pool_pipelined(tokenizer, num_questions, forbidden):
    pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo = [], [], [], [], []
    pbar = tqdm(total=num_questions, desc="Init")
    # Heuristic: at ~60% extract success and ~70% non-forbidden, generate
    # ~2.5x the remaining count per round.
    OVERHEAD = 2.5
    MAX_PROMPTS_PER_ROUND = 2500

    while len(pool_q) < num_questions:
        needed = num_questions - len(pool_q)
        bs = min(int(needed * OVERHEAD), MAX_PROMPTS_PER_ROUND)
        c_prompts = []
        for _ in range(bs):
            topic = random.choice(MATH_TOPICS)
            user_p = (
                "Generate one new, challenging reasoning question now. "
                f"YOU MUST STRICTLY FOCUS ON: **{topic}**."
            )
            c_prompts.append(apply_chat_template(tokenizer, CHALLENGER_SYSTEM_PROMPT, user_p))

        texts = _base_generate_big_batch(
            c_prompts, max_tokens=1536, temperature=1.0, top_p=0.95
        )

        valid_qs, valid_gts = [], []
        for t in texts:
            q, gt = extract_challenger_output(t)
            if q and len(q) > 30 and not any(w in q.lower() for w in forbidden):
                valid_qs.append(q)
                valid_gts.append(gt)
        if not valid_qs:
            continue

        r_uncs, p_hats, pseudos = _solver_score_big_batch(tokenizer, valid_qs)
        for q, gt, ru, ph, ps in zip(valid_qs, valid_gts, r_uncs, p_hats, pseudos):
            if len(pool_q) >= num_questions:
                break
            pool_q.append(q)
            pool_gt.append(gt)
            pool_runc.append(ru)
            pool_phat.append(ph)
            pool_pseudo.append(ps)
            pbar.update(1)
    pbar.close()
    return pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo


# ---------------------------------------------------------------------------
# One MCMC walk step — all 1500 proposals generated and scored upfront,
# then sequential MH (same as canonical from this point)
# ---------------------------------------------------------------------------
def _walk_step_pipelined(
    step_idx, num_questions, pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo,
    pool_tokens, neighbor, cluster_size, r_unc_arr, energy, tokenizer, log_file,
):
    n = num_questions
    idx_perm = list(range(num_questions))
    random.shuffle(idx_perm)

    # === STAGE 1: build all proposal prompts upfront ===
    m_prompts = [
        apply_chat_template(
            tokenizer, MUTATOR_SYSTEM_PROMPT,
            MUTATOR_USER_TEMPLATE.format(seed=pool_q[k]),
        )
        for k in idx_perm
    ]

    # === STAGE 2: ONE big base call (chunked internally) ===
    texts = _base_generate_big_batch(
        m_prompts, max_tokens=1536, temperature=1.1, top_p=0.95
    )

    # === STAGE 3: extract proposals; track which idx in idx_perm produced each ===
    proposals = []
    for j, k in enumerate(idx_perm):
        t = texts[j]
        qp, gtp = extract_challenger_output(t)
        strat = extract_mutation_strategy(t) or "?"
        if qp and len(qp) > 30 and qp != pool_q[k]:
            proposals.append({"k": k, "q": qp, "gt": gtp, "strat": strat,
                              "old": pool_q[k]})

    if not proposals:
        return pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo, \
               pool_tokens, neighbor, cluster_size, r_unc_arr, energy

    # === STAGE 4: ONE big solver call across ALL proposals ===
    qs_new = [p["q"] for p in proposals]
    rus, phs, pls = _solver_score_big_batch(tokenizer, qs_new)

    # === STAGE 5: sequential MH (same as canonical; updates state per accept) ===
    pbar_step = tqdm(total=len(proposals), desc=f"MCMC step {step_idx + 1}/{config.MCMC_STEPS} MH")
    for j, p in enumerate(proposals):
        k = p["k"]
        q_prime_tokens = p["q"].split()
        new_row = neighbor_row_for(q_prime_tokens, pool_tokens, self_idx=k)
        old_row = neighbor[k]
        delta = new_row.astype(int) - old_row.astype(int)
        delta[k] = 0
        new_cluster = cluster_size + delta
        new_cluster[k] = float(new_row.sum())

        new_runc = r_unc_arr.copy()
        new_runc[k] = rus[j]
        new_e = energy_from_state(new_runc, new_cluster, n)

        alpha = min(1.0, np.exp((new_e - energy) / config.BETA))
        accept = random.random() < alpha

        r_rep_k_old = float(cluster_size[k]) / n
        r_rep_k_new = float(new_cluster[k]) / n
        r_c_k_old = max(0.0, float(r_unc_arr[k]) - config.LAMBDA_REP * r_rep_k_old)
        r_c_k_new = max(0.0, rus[j] - config.LAMBDA_REP * r_rep_k_new)

        log_file.write(
            f"\n[Step {step_idx + 1} | Proposal {j}/{len(proposals)}] "
            f"Result: {'ACCEPTED' if accept else 'REJECTED'} | Strategy: {p['strat']}\n"
            f"--- [OLD QUESTION] ---\n{p['old']}\n"
            f"--- [NEW QUESTION] ---\n{p['q']}\n"
            f"r_unc:        {float(r_unc_arr[k]):.4f} -> {rus[j]:.4f}\n"
            f"r_rep[k]:     {r_rep_k_old:.4f} -> {r_rep_k_new:.4f}  "
            f"(cluster_size {int(cluster_size[k])} -> {int(new_cluster[k])})\n"
            f"r_c[k]:       {r_c_k_old:.4f} -> {r_c_k_new:.4f}\n"
            f"V(X) total:   {energy:.4f} -> {new_e:.4f}  (dE={new_e - energy:+.4f})\n"
            f"Alpha:        {alpha:.4f}\n"
            f"{'-' * 60}\n"
        )
        log_file.flush()
        if accept:
            pool_q[k] = p["q"]
            pool_tokens[k] = q_prime_tokens
            pool_gt[k] = pls[j] or p["gt"]
            pool_runc[k] = rus[j]
            pool_phat[k] = phs[j]
            pool_pseudo[k] = pls[j]
            neighbor[k, :] = new_row
            neighbor[:, k] = new_row
            cluster_size = new_cluster
            r_unc_arr = new_runc
            energy = new_e
        pbar_step.update(1)
        pbar_step.set_postfix({"V": f"{energy:.3f}"})
    pbar_step.close()

    return pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo, \
           pool_tokens, neighbor, cluster_size, r_unc_arr, energy


# ---------------------------------------------------------------------------
# Main entry point — same signature/return as deo.generate_batch_mcmc
# ---------------------------------------------------------------------------
def generate_batch_mcmc_pipelined(tokenizer, num_questions, log_path):
    """Pipelined version of deo.generate_batch_mcmc. Same return type:
       list of dicts {question, gt, p_hat, pseudo_label, r_unc}.
    """
    print(f"\n[MCMC-pipelined] Initializing pool of {num_questions} questions...")
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("=" * 50 + "\nMCMC Phase (PIPELINED)\n" + "=" * 50 + "\n")

    forbidden = ["prove that", "show that", "justify", "explain", "true or false", "yes or no"]
    pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo = _init_pool_pipelined(
        tokenizer, num_questions, forbidden
    )

    print("[MCMC-pipelined] Computing initial neighbor matrix (one-shot O(N^2))...")
    energy, neighbor, cluster_size = calculate_batch_energy(pool_q, pool_runc)
    pool_tokens = [q.split() for q in pool_q]
    r_unc_arr = np.asarray(pool_runc, dtype=float)
    log_file.write(f"[Init] V(X_0) = {energy:.4f}\n")

    for step_idx in range(config.MCMC_STEPS):
        (pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo,
         pool_tokens, neighbor, cluster_size, r_unc_arr, energy) = _walk_step_pipelined(
            step_idx, num_questions,
            pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo,
            pool_tokens, neighbor, cluster_size, r_unc_arr, energy,
            tokenizer, log_file,
        )

    log_file.close()
    return [
        {"question": pool_q[i], "gt": pool_gt[i], "p_hat": pool_phat[i],
         "pseudo_label": pool_pseudo[i], "r_unc": pool_runc[i]}
        for i in range(num_questions)
    ]

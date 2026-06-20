"""
Ablation v2 — cheaper version of base_agreement.

Skips per-iter MCMC walk (the expensive 3.5h step). Instead loads canonical
DEO's archived mcmc pools as the question source for each iter, then
re-scores them with BOTH current solver M-vote AND base model M-vote, and
keeps only questions where the two agree (mathruler.grade_answer).

Question set per iter is taken from canonical's archive (post-walk pool).
For iter 2, our solver_v1 = canonical solver_v1 so this is exact; for
iter 3+ the question set is from canonical's solver-evolution trajectory,
which is a small confound (the pool was curated for canonical's solvers,
not ours). Acceptable because the ablation's hypothesis is about LABEL
agreement, not question evolution.

Per-iter cost: ~80 min (vs ~5h with full MCMC walk). 4 iters: ~5-6h total.
"""
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo
import openai

deo.config.MODEL_ABBR = "deo_base_agreement_v2"

# Override deo's lazy-init clients with longer HTTP timeout (default 600s
# isn't enough for big batch calls like 1500*9 = 13500 prompts on a single
# vllm_base). 1800s = 30 min cap per request. Combined with explicit
# chunking below this is double-defensive.
deo._client_base = openai.OpenAI(
    api_key="EMPTY", base_url=deo.config.VLLM_BASE_URL, timeout=1800.0,
)
deo._clients_solver = [
    openai.OpenAI(api_key="EMPTY", base_url=url, timeout=1800.0)
    for url in deo.config.VLLM_SOLVER_URLS
]

# Chunk M-vote requests to stay within vllm batched-throughput sweet spot
# and well clear of the 1800s timeout cap.
SCORE_CHUNK_QUESTIONS = 100  # 100 * 9 = 900 prompts per HTTP call

START_CKPT = "/storage/models/deo_base_agreement_solver_v1/global_step_15/actor/huggingface"
CANONICAL_POOLS_DIR = "/storage/canonical_mcmc_pools"  # canonical mcmc_iter_{1..5}.json
VERL_OVERRIDES = ["data.rollout_batch_size=64"]


def score_with_solver_dp(questions, tokenizer):
    """M-vote each question with the current solver via DP. Returns
    (p_hat_list, pseudo_label_list). Chunks per-shard for HTTP timeout safety."""
    if not questions:
        return [], []
    m = deo.config.M_SAMPLES
    clients = deo.solver_clients()
    n_dp = len(clients)

    # Shard questions across DP instances (each shard chunked internally)
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
        for s in range(0, len(qs), SCORE_CHUNK_QUESTIONS):
            q_sub = qs[s:s + SCORE_CHUNK_QUESTIONS]
            prompts = [
                deo.apply_chat_template(tokenizer, deo.RZERO_SOLVER_SYSTEM, q)
                for q in q_sub for _ in range(m)
            ]
            resp = clients[idx].completions.create(
                model=deo.config.MODEL_NAME,
                prompt=prompts,
                max_tokens=deo.config.SOLVER_MAX_TOKENS,
                temperature=deo.config.SOLVER_TEMP,
                top_p=deo.config.SOLVER_TOP_P,
                extra_body={"top_k": deo.config.SOLVER_TOP_K},
            )
            out.extend(c.text for c in resp.choices)
        return idx, out

    texts_by_shard = [None] * n_dp
    with ThreadPoolExecutor(max_workers=n_dp) as ex:
        for fut in as_completed([ex.submit(_run_shard, i) for i in range(n_dp)]):
            idx, txt = fut.result()
            texts_by_shard[idx] = txt
    texts = [t for shard in texts_by_shard for t in shard]
    answers = [deo.extract_solver_answer(t) for t in texts]

    p_hat_list = []
    pseudo_list = []
    for i in range(len(questions)):
        chunk = answers[i * m: (i + 1) * m]
        valid = [a for a in chunk if a is not None and a != "GUESSED_FAIL_FORMAT"]
        if not valid:
            p_hat_list.append(0.0)
            pseudo_list.append(None)
            continue
        major, count = Counter(valid).most_common(1)[0]
        p_hat_list.append(count / m)
        is_garbage = "text" in major.lower() or "\\text" in major or len(major) > 100
        pseudo_list.append(None if is_garbage else major)
    return p_hat_list, pseudo_list


def score_with_base(questions, tokenizer):
    """M-vote each question with the base model. Chunked for HTTP timeout safety.
    Returns pseudo_label_list."""
    if not questions:
        return []
    m = deo.config.M_SAMPLES
    texts = []
    for s in range(0, len(questions), SCORE_CHUNK_QUESTIONS):
        q_sub = questions[s:s + SCORE_CHUNK_QUESTIONS]
        prompts = [
            deo.apply_chat_template(tokenizer, deo.RZERO_SOLVER_SYSTEM, q)
            for q in q_sub for _ in range(m)
        ]
        resp = deo.base_client().completions.create(
            model=deo.config.MODEL_NAME,
            prompt=prompts,
            max_tokens=deo.config.SOLVER_MAX_TOKENS,
            temperature=deo.config.SOLVER_TEMP,
            top_p=deo.config.SOLVER_TOP_P,
            extra_body={"top_k": deo.config.SOLVER_TOP_K},
        )
        texts.extend(c.text for c in resp.choices)
    answers = [deo.extract_solver_answer(t) for t in texts]
    pseudo_list = []
    for i in range(len(questions)):
        chunk = answers[i * m: (i + 1) * m]
        valid = [a for a in chunk if a is not None and a != "GUESSED_FAIL_FORMAT"]
        if not valid:
            pseudo_list.append(None)
            continue
        major, _ = Counter(valid).most_common(1)[0]
        pseudo_list.append(major)
    return pseudo_list


def filter_and_push_v2(questions, solver_pseudo, solver_phat, base_pseudo,
                      repo_name, config_name):
    """Apply filter cascade: BAD_RE → solver+base pseudo non-null → phat∈[0.3,0.8]
    → solver/base mutual-agreement (grade_answer). Push to HF + local dump."""
    kept = []
    stats = {"regex": 0, "no_solver_pseudo": 0, "no_base_pseudo": 0,
             "phat_oob": 0, "disagree": 0}
    for q, s_pseudo, s_phat, b_pseudo in zip(questions, solver_pseudo,
                                              solver_phat, base_pseudo):
        if deo.BAD_RE.search(q):
            stats["regex"] += 1
            continue
        if s_pseudo is None:
            stats["no_solver_pseudo"] += 1
            continue
        if b_pseudo is None:
            stats["no_base_pseudo"] += 1
            continue
        if not (deo.config.MIN_SCORE <= s_phat <= deo.config.MAX_SCORE):
            stats["phat_oob"] += 1
            continue
        if not deo.grade_answer(s_pseudo, b_pseudo):
            stats["disagree"] += 1
            continue
        kept.append({"problem": q, "answer": s_pseudo, "score": s_phat})

    n = len(questions)
    print(f"[filter] {n} candidates → {len(kept)} kept")
    print(f"  drops: regex={stats['regex']} | no_solver_pseudo={stats['no_solver_pseudo']} | "
          f"no_base_pseudo={stats['no_base_pseudo']} | phat_oob={stats['phat_oob']} | "
          f"disagree={stats['disagree']}")

    if not kept:
        raise SystemExit("ERROR: 0 questions passed filter — cannot train empty dataset.")

    local_path = f"{deo.config.STORAGE_ROOT}/datasets/filtered_{repo_name}.json"
    with open(local_path, "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"[upload] wrote local JSON: {local_path}")

    repo_full = f"{deo.config.HF_USER}/{repo_name}"
    ds = deo.DatasetDict({"train": deo.Dataset.from_list(kept)})
    ds.push_to_hub(repo_full, private=True, config_name=config_name)
    print(f"[upload] pushed to https://huggingface.co/datasets/{repo_full}")
    return len(kept)


def main():
    deo.config.HF_TOKEN = deo.load_hf_token()
    deo.hf_login(token=deo.config.HF_TOKEN)
    for d in ["models", "evaluation", "temp_results", "datasets", "logs",
              "generated_question"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{d}", exist_ok=True)

    tokenizer = deo.AutoTokenizer.from_pretrained(deo.config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    deo.wait_for_vllm_ready(deo.config.VLLM_BASE_URL, label="vllm-base")
    for url in deo.config.VLLM_SOLVER_URLS:
        deo.wait_for_vllm_ready(url, label=f"vllm-solver@{url}")

    if not os.path.exists(START_CKPT):
        sys.exit(f"ERROR: starting ckpt not found: {START_CKPT}")
    if not os.path.exists(CANONICAL_POOLS_DIR):
        sys.exit(f"ERROR: canonical mcmc pools dir not found: {CANONICAL_POOLS_DIR}")

    print(f"[ablation v2] reloading solver instances with iter-1 ckpt:")
    print(f"  {START_CKPT}")
    deo.reload_vllm_solver(START_CKPT)

    eval_history = {"iter_0_baseline": 0.722, "iter_1": 0.768}
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_base_agreement_v2.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)
    save()

    current_solver = START_CKPT

    for it in range(2, deo.config.NUM_ITERATIONS + 1):
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {it}/5 (base_agreement_v2: canonical pool + dual M-vote) ===")
        print(f"{'=' * 60}\n")

        # 1. Load canonical iter N's questions (skip MCMC regen).
        pool_path = f"{CANONICAL_POOLS_DIR}/mcmc_iter_{it}.json"
        pool = json.load(open(pool_path))
        questions = [d["question"] for d in pool]
        print(f"[pool] {len(questions)} questions loaded from canonical {pool_path}")

        # 2. Re-score with our CURRENT solver (M-vote, DP).
        t0 = time.time()
        s_phat, s_pseudo = score_with_solver_dp(questions, tokenizer)
        print(f"[score-solver] DP={len(deo.config.SOLVER_INSTANCES)} done in {time.time()-t0:.0f}s")

        # 3. Re-score with base model (M-vote, single).
        t0 = time.time()
        b_pseudo = score_with_base(questions, tokenizer)
        print(f"[score-base]   done in {time.time()-t0:.0f}s")

        # 4. Filter + push.
        filter_and_push_v2(questions, s_pseudo, s_phat, b_pseudo, repo_name, exp_name)

        # 5. verl GRPO (batch=64 override for small datasets).
        merged_ckpt = deo.run_verl_solver(
            current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name,
            extra_verl_overrides=VERL_OVERRIDES,
        )

        # 6. Eval + recheck (built into eval_math500).
        eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
        save()

        # 7. Reload vllm-solver for next iter.
        if it < deo.config.NUM_ITERATIONS:
            deo.reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (base_agreement_v2 — no MCMC walk):")
    print("=" * 60)
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f}  ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

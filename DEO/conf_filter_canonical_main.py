"""
Canonical DEO pipeline + extra stage-5 filter: keep only top-50% of
post-filter entries by majority-answer confidence (per-token logprob
average across trajectories that produced the majority).

Confidence definition:
  For each candidate (problem, pseudo) that survived the existing 4-stage
  filter (regex + p_hat + non-null pseudo + LLM-judge), run ONE additional
  M-vote with logprobs=1 on the solver DP cluster. For each trajectory
  matching the majority answer (pseudo), compute mean per-token logprob.
  Confidence(question) = mean across matching trajectories.
  Higher (less negative) = more confident.

Test: empirically (iter1 dataset, 924 entries graded by GPT-5 oracle) we
saw correctness ROC-AUC of 0.65 for per-token logprob alone, 0.75 for
M-vote consistency p_hat. Correct-rate jumps from 35% in lowest-conf
quartile to 68% in highest. So filtering for top-50% conf should bias
the training set toward more correct pseudo-labels.

Pipeline (identical to canonical except marked NEW):
  MCMC pool (init + walk)
  → filter_and_push:
      Stage 1: regex BAD_PATTERNS (already at extract time)
      Stage 2: p_hat ∈ [0.3, 0.8] + pseudo non-null
      Stage 3: LLM-judge validity
      Stage 4: [NEW] solver M-vote with logprobs → keep top 50% by conf
  → verl GRPO (rollout_batch_size=64 because dataset ~450 < default 512)
  → eval_math500 + GPT-mini recheck
  → reload vllm_solver to new ckpt (canonical drifting-labeler behavior)

Compare to canonical (76.8 → 75.0 → 71.8 → 71.2 → 69.0, drop −7.8).
"""
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo

deo.config.MODEL_ABBR = "deo_conf_filter"

VERL_OVERRIDES = ["data.rollout_batch_size=64"]
SCORE_CHUNK_QUESTIONS = 100  # per-shard chunk to stay under HTTP timeout


def score_with_solver_logprobs(questions, tokenizer):
    """Same as deo.evaluate_r_unc_vllm but with logprobs enabled. Returns
    per-question: (pseudo, per_token_logprob_of_majority_trajectories).
    """
    if not questions:
        return [], []
    m = deo.config.M_SAMPLES
    clients = deo.solver_clients()
    n_dp = len(clients)

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
            return idx, [], []
        out_texts = []
        out_lps = []
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
                logprobs=1,
            )
            for c in resp.choices:
                out_texts.append(c.text)
                lps = c.logprobs.token_logprobs if c.logprobs else []
                valid_lps = [l for l in lps if l is not None]
                out_lps.append(
                    sum(valid_lps) / len(valid_lps) if valid_lps else float("-inf")
                )
        return idx, out_texts, out_lps

    texts_by_shard = [None] * n_dp
    lps_by_shard = [None] * n_dp
    with ThreadPoolExecutor(max_workers=n_dp) as ex:
        for fut in as_completed([ex.submit(_run_shard, i) for i in range(n_dp)]):
            idx, txt, lps = fut.result()
            texts_by_shard[idx] = txt
            lps_by_shard[idx] = lps
    texts = [t for shard in texts_by_shard for t in shard]
    per_tok_lps = [l for shard in lps_by_shard for l in shard]
    answers = [deo.extract_solver_answer(t) for t in texts]

    pseudo_list = []
    conf_list = []
    for i in range(len(questions)):
        chunk_ans = answers[i * m: (i + 1) * m]
        chunk_lps = per_tok_lps[i * m: (i + 1) * m]
        valid = [(a, l) for a, l in zip(chunk_ans, chunk_lps)
                 if a is not None and a != "GUESSED_FAIL_FORMAT"]
        if not valid:
            pseudo_list.append(None)
            conf_list.append(float("-inf"))
            continue
        valid_ans = [a for a, _ in valid]
        major, _ = Counter(valid_ans).most_common(1)[0]
        matching_lps = [l for a, l in valid if a == major and l != float("-inf")]
        conf = sum(matching_lps) / len(matching_lps) if matching_lps else float("-inf")
        pseudo_list.append(major)
        conf_list.append(conf)
    return pseudo_list, conf_list


def filter_and_push_with_conf(train_data, repo_name, config_name, tokenizer):
    """Canonical 4-stage filter + NEW stage 5: solver M-vote with logprobs,
    keep top-50% by per-token logprob of majority-producing trajectories.
    """
    n_total = len(train_data)

    # Stages 1-2: phat + pseudo
    stage1 = [
        d for d in train_data
        if d["pseudo_label"] not in (None, "", "None")
        and deo.config.MIN_SCORE <= d["p_hat"] <= deo.config.MAX_SCORE
    ]
    print(f"[filter] phat∈[{deo.config.MIN_SCORE},{deo.config.MAX_SCORE}]+pseudo: "
          f"{len(stage1)}/{n_total} passed")

    # Stage 3: LLM judge
    if stage1:
        t0 = time.time()
        judge_ok = [True] * len(stage1)
        with ThreadPoolExecutor(max_workers=24) as ex:
            futs = {ex.submit(deo.judge_one_validity, d["question"], d["pseudo_label"]): i
                    for i, d in enumerate(stage1)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    judge_ok[i] = fut.result()
                except Exception:
                    judge_ok[i] = True
        n_judge_drop = sum(1 for v in judge_ok if not v)
        print(f"[filter] LLM-judge dropped {n_judge_drop}/{len(stage1)} "
              f"({time.time() - t0:.1f}s)")
        stage2 = [d for d, ok in zip(stage1, judge_ok) if ok]
    else:
        stage2 = []

    # Stage 4 NEW: confidence filter (keep top 50% by per-token logprob of majority)
    print(f"[filter] running solver M-vote w/ logprobs on {len(stage2)} entries...")
    t0 = time.time()
    questions = [d["question"] for d in stage2]
    pseudos, confs = score_with_solver_logprobs(questions, tokenizer)
    print(f"  done in {time.time() - t0:.0f}s")

    # Attach confidence to each entry. Use the freshly-computed pseudo (in
    # case it shifted from earlier M-vote) but keep the existing p_hat for
    # logging. Disambiguate: write `conf_pseudo` separately.
    for d, ps, cf in zip(stage2, pseudos, confs):
        d["conf_pseudo"] = ps
        d["confidence"] = cf

    # Drop any entry where confidence couldn't be computed (-inf)
    stage2 = [d for d in stage2 if d["confidence"] != float("-inf")]

    # Sort by confidence descending, keep top 50%.
    stage2.sort(key=lambda d: d["confidence"], reverse=True)
    keep_n = max(1, len(stage2) // 2)
    stage3 = stage2[:keep_n]
    print(f"[filter] conf filter: kept top {keep_n}/{len(stage2)} "
          f"(median conf={stage2[len(stage2)//2]['confidence']:.4f}, "
          f"top conf={stage3[0]['confidence']:.4f}, "
          f"cutoff conf={stage3[-1]['confidence']:.4f})")

    filtered = [
        {"problem": d["question"], "answer": d["pseudo_label"], "score": d["p_hat"]}
        for d in stage3
    ]
    print(f"[upload] {len(filtered)}/{n_total} final after regex+phat+pseudo+judge+conf")
    if not filtered:
        raise SystemExit("ERROR: 0 questions passed filter")

    local_path = f"{deo.config.STORAGE_ROOT}/datasets/filtered_{repo_name}.json"
    # Include confidence so user can analyze offline
    out_with_conf = [
        {"problem": d["question"], "answer": d["pseudo_label"],
         "score": d["p_hat"], "confidence": d["confidence"]}
        for d in stage3
    ]
    with open(local_path, "w") as f:
        json.dump(out_with_conf, f, indent=2, ensure_ascii=False)
    print(f"[upload] wrote local: {local_path}")

    repo_full = f"{deo.config.HF_USER}/{repo_name}"
    ds = deo.DatasetDict({"train": deo.Dataset.from_list(filtered)})
    ds.push_to_hub(repo_full, private=True, config_name=config_name)
    print(f"[upload] pushed to {repo_full}")
    return len(filtered)


def main():
    deo.config.HF_TOKEN = deo.load_hf_token()
    deo.hf_login(token=deo.config.HF_TOKEN)
    for d in ["datasets", "logs", "models", "evaluation", "temp_results",
              "generated_question"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{d}", exist_ok=True)

    tokenizer = deo.AutoTokenizer.from_pretrained(deo.config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    deo.wait_for_vllm_ready(deo.config.VLLM_BASE_URL, label="vllm-base")
    for url in deo.config.VLLM_SOLVER_URLS:
        deo.wait_for_vllm_ready(url, label=f"vllm-solver@{url}")

    eval_history = {}
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_conf_filter.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    eval_history["iter_0_baseline"] = deo.eval_math500(deo.config.MODEL_NAME, "iter 0 baseline")
    save()

    current_solver = deo.config.MODEL_NAME
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {it}/{deo.config.NUM_ITERATIONS} (canonical + conf filter) ===")
        print(f"{'=' * 60}")
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_conf_filter.log"

        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_conf_filter.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        filter_and_push_with_conf(train_data, repo_name, exp_name, tokenizer)

        merged_ckpt = deo.run_verl_solver(
            current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name,
            extra_verl_overrides=VERL_OVERRIDES,
        )

        eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
        save()

        # Canonical: drifting labeler — reload vllm_solver to new ckpt each iter
        if it < deo.config.NUM_ITERATIONS:
            deo.reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (canonical + conf filter):")
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

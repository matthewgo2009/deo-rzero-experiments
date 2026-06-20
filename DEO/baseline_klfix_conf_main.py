"""
Stack two best fixes:
  (1) baseline_klfix: MCMC_STEPS=0 (no walk, init pool only) + frozen
      solver_v1 labeler (reload vllm_solver only after iter 1, then never
      again).
  (2) conf_filter: add stage-5 filter to filter_and_push that re-M-votes
      with logprobs on post-stage-4 entries, keeps top-50% by majority-
      trajectory per-token logprob.

Same Qwen3-4B-Base. KL ref pinned to base. verl rollout_batch_size=64.

Hypothesis: both modifications independently rescue trajectory degradation
(baseline_klfix peak→iter5 = -0.2, conf_filter alone = -0.6). They attack
different failure modes (pseudo-label drift vs label noise) so stacking
may close to net positive trajectory (iter 5 ≥ 77).
"""
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo

deo.config.MODEL_ABBR = "deo_baseline_klfix_conf"
deo.config.MCMC_STEPS = 0  # baseline_klfix: no MCMC walk, init pool only

VERL_OVERRIDES = ["data.rollout_batch_size=64"]
SCORE_CHUNK_QUESTIONS = 100


def score_with_solver_logprobs(questions, tokenizer):
    """M-vote with solver_v1 (frozen labeler) + logprobs. Returns
    per-question (pseudo, mean per-token logprob over matching trajectories)."""
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
        out_texts, out_lps = [], []
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

    pseudo_list, conf_list = [], []
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
    n_total = len(train_data)
    stage1 = [
        d for d in train_data
        if d["pseudo_label"] not in (None, "", "None")
        and deo.config.MIN_SCORE <= d["p_hat"] <= deo.config.MAX_SCORE
    ]
    print(f"[filter] phat∈[{deo.config.MIN_SCORE},{deo.config.MAX_SCORE}]+pseudo: "
          f"{len(stage1)}/{n_total} passed")

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

    # Stage 5: solver M-vote w/ logprobs → top 50% by conf
    print(f"[filter] running solver M-vote w/ logprobs on {len(stage2)} entries...")
    t0 = time.time()
    questions = [d["question"] for d in stage2]
    _, confs = score_with_solver_logprobs(questions, tokenizer)
    for d, cf in zip(stage2, confs):
        d["confidence"] = cf
    stage2 = [d for d in stage2 if d["confidence"] != float("-inf")]
    stage2.sort(key=lambda d: d["confidence"], reverse=True)
    keep_n = max(1, len(stage2) // 2)
    stage3 = stage2[:keep_n]
    print(f"[filter] conf filter: kept top {keep_n}/{len(stage2)} "
          f"(top {stage3[0]['confidence']:.4f}, cutoff {stage3[-1]['confidence']:.4f}) "
          f"in {time.time() - t0:.0f}s")

    filtered = [
        {"problem": d["question"], "answer": d["pseudo_label"], "score": d["p_hat"]}
        for d in stage3
    ]
    print(f"[upload] {len(filtered)}/{n_total} final after regex+phat+pseudo+judge+conf")
    if not filtered:
        raise SystemExit("ERROR: 0 questions passed filter")

    local_path = f"{deo.config.STORAGE_ROOT}/datasets/filtered_{repo_name}.json"
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
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_baseline_klfix_conf.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    eval_history["iter_0_baseline"] = deo.eval_math500(deo.config.MODEL_NAME, "iter 0 baseline")
    save()

    current_solver = deo.config.MODEL_NAME
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {it}/{deo.config.NUM_ITERATIONS} (baseline_klfix + conf filter) ===")
        print(f"{'=' * 60}")
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_baseline_klfix_conf.log"

        # MCMC_STEPS=0 → init pool only (no walk)
        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_baseline_klfix_conf.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        filter_and_push_with_conf(train_data, repo_name, exp_name, tokenizer)

        merged_ckpt = deo.run_verl_solver(
            current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name,
            extra_verl_overrides=VERL_OVERRIDES,
        )

        # Eval wrapped in try/except (vllm device-empty bug after Ray cleanup
        # has bitten this exact setup before).
        try:
            eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
        except Exception as exc:
            print(f"[main] eval iter {it} failed in-process: {type(exc).__name__}; "
                  f"ckpt at {merged_ckpt}, can be redone in fresh container")
            eval_history[f"iter_{it}_eval_failed"] = True
        save()

        # FROZEN LABELER: reload vllm_solver only after iter 1.
        if it == 1:
            deo.reload_vllm_solver(merged_ckpt)
            print(f"[frozen-labeler] reloaded vllm_solver with solver_v1. "
                  f"Will NOT reload again for iter 2-5.")
        current_solver = merged_ckpt

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (baseline_klfix + conf filter):")
    for label, acc in eval_history.items():
        if isinstance(acc, float):
            print(f"  {label:25s} {acc:.4f} ({acc * 100:.2f}%)")
        else:
            print(f"  {label:25s} {acc}")


if __name__ == "__main__":
    main()

"""
Ablation: pseudo-label "double approval" filter.

For each candidate question in iter N (N>=2), we run M-vote with BOTH the
current solver (solver_v(N-1) ckpt) AND the original base model. The
question only enters the training set if BOTH models' majority answers
agree (mathruler grade_answer equivalence). This tests whether requiring
base-model concurrence prevents per-iter solver drift.

Compare against canonical DEO (76.8 → 75.0 → 71.8 → 71.2 → 69.0) and the
reuse_iter1 ablation (76.8 → 76.2 → 76.8 → 76.6 → 73.8).

Hypothesis: base_agreement falls somewhere between the two — better than
canonical (drops questions where solver's biased M-vote disagrees with
the unbiased base), possibly worse than reuse_iter1 (base is weak so it
may over-filter genuinely good questions whose base-model M-vote is just
noisy).

Starts from canonical solver_v1 ckpt (acc 76.8 rechecked). Runs iter 2-5.
"""
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo

# Separate output prefix so this ablation doesn't collide with canonical names
deo.config.MODEL_ABBR = "deo_base_agreement"

# Starting solver ckpt = canonical solver_v1, copied to live storage by launcher.
START_CKPT = "/storage/models/deo_base_agreement_solver_v1/global_step_15/actor/huggingface"

# Override verl rollout_batch_size to 64 (default 512 would crash if base-
# agreement filter drops dataset below it). This is the only param-level
# divergence from canonical; rest of the pipeline is identical.
VERL_OVERRIDES = ["data.rollout_batch_size=64"]


def base_agreement_filter(stage_entries, tokenizer):
    """For each entry, M-vote with the base model on entry["question"] and
    keep only entries where base's majority answer is mathematically
    equivalent (via mathruler.grade_answer) to entry["pseudo_label"] (which
    is the current solver's M-vote majority).

    Cost: ~M=9 base-model completions per entry; for typical post-LLM-judge
    set of ~900 entries this is ~8100 base completions = ~1-2 min on a
    single vllm_base instance.
    """
    if not stage_entries:
        return []
    m = deo.config.M_SAMPLES
    prompts = [
        deo.apply_chat_template(tokenizer, deo.RZERO_SOLVER_SYSTEM, d["question"])
        for d in stage_entries for _ in range(m)
    ]
    resp = deo.base_client().completions.create(
        model=deo.config.MODEL_NAME,
        prompt=prompts,
        max_tokens=deo.config.SOLVER_MAX_TOKENS,
        temperature=deo.config.SOLVER_TEMP,
        top_p=deo.config.SOLVER_TOP_P,
        extra_body={"top_k": deo.config.SOLVER_TOP_K},
    )
    texts = [c.text for c in resp.choices]
    answers = [deo.extract_solver_answer(t) for t in texts]

    kept = []
    for i, d in enumerate(stage_entries):
        chunk = answers[i * m: (i + 1) * m]
        valid = [a for a in chunk if a is not None and a != "GUESSED_FAIL_FORMAT"]
        if not valid:
            continue  # base couldn't M-vote on this question
        base_major, _ = Counter(valid).most_common(1)[0]
        # mathruler.grade_answer handles 1/2 vs 0.5, \\frac{1}{2} vs (1/2),
        # whitespace differences, etc. — so we measure semantic agreement.
        if deo.grade_answer(base_major, d["pseudo_label"]):
            kept.append(d)
    return kept


def filter_and_push_with_base_agreement(train_data, repo_name, config_name, tokenizer):
    """Same pipeline as canonical filter_and_push, but adds a base-agreement
    filter stage after the LLM-judge stage and before HF push.

    Stages: regex (already applied at extract) → p_hat∈[0.3,0.8] + pseudo
    non-null → LLM-judge validity → base-agreement → local dump + HF push.
    """
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
            futs = {
                ex.submit(deo.judge_one_validity, d["question"], d["pseudo_label"]): i
                for i, d in enumerate(stage1)
            }
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

    # NEW: base-agreement filter — only keep entries where base model's M-vote
    # majority matches the solver's M-vote majority (the pseudo_label).
    print(f"[filter] running base-agreement check on {len(stage2)} entries...")
    t0 = time.time()
    stage3 = base_agreement_filter(stage2, tokenizer)
    n_ba_drop = len(stage2) - len(stage3)
    print(f"[filter] base-agreement dropped {n_ba_drop}/{len(stage2)} "
          f"({time.time() - t0:.1f}s)")

    filtered = [
        {"problem": d["question"], "answer": d["pseudo_label"], "score": d["p_hat"]}
        for d in stage3
    ]
    print(f"[upload] {len(filtered)}/{n_total} final after "
          f"regex+phat+pseudo+judge+base-agreement")
    if not filtered:
        raise SystemExit("ERROR: 0 questions passed filter — cannot train empty dataset.")

    local_path = f"{deo.config.STORAGE_ROOT}/datasets/filtered_{repo_name}.json"
    with open(local_path, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"[upload] also wrote local JSON: {local_path}")

    repo_full = f"{deo.config.HF_USER}/{repo_name}"
    ds = deo.DatasetDict({"train": deo.Dataset.from_list(filtered)})
    ds.push_to_hub(repo_full, private=True, config_name=config_name)
    print(f"[upload] pushed to https://huggingface.co/datasets/{repo_full}")
    return len(filtered)


def main():
    deo.config.HF_TOKEN = deo.load_hf_token()
    deo.hf_login(token=deo.config.HF_TOKEN)
    for d in ["models", "evaluation", "temp_results", "datasets", "logs",
              "generated_question"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{d}", exist_ok=True)

    tokenizer = deo.AutoTokenizer.from_pretrained(deo.config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wait for all 4 vllm endpoints (base + 3 solver DP)
    deo.wait_for_vllm_ready(deo.config.VLLM_BASE_URL, label="vllm-base")
    for url in deo.config.VLLM_SOLVER_URLS:
        deo.wait_for_vllm_ready(url, label=f"vllm-solver@{url}")

    if not os.path.exists(START_CKPT):
        sys.exit(
            f"ERROR: starting ckpt not found: {START_CKPT}\n"
            "  Copy from canonical archive before launching this script."
        )

    # Reload all solver DP instances to iter-1 solver so MCMC walk uses the
    # right model for r_unc scoring.
    print(f"[ablation] reloading solver instances with iter-1 ckpt:")
    print(f"  {START_CKPT}")
    deo.reload_vllm_solver(START_CKPT)

    # iter 0/1 numbers reused from canonical DEO (same starting ckpt as ablation)
    eval_history = {"iter_0_baseline": 0.722, "iter_1": 0.768}
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_base_agreement.json"

    def save_summary():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    save_summary()

    current_solver = START_CKPT
    for it in range(2, deo.config.NUM_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {it}/{deo.config.NUM_ITERATIONS} (base-agreement filter) ===")
        print(f"{'=' * 60}\n")
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_ba.log"

        # 1. MCMC pool (uses base for proposals, current solver for r_unc).
        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_ba.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        # 2. Multi-stage filter (regex applied at extract earlier) + base-agreement.
        filter_and_push_with_base_agreement(train_data, repo_name, exp_name, tokenizer)

        # 3. verl GRPO — same as canonical, plus rollout_batch_size=64 override
        #    to survive datasets smaller than verl's 512 default.
        merged_ckpt = deo.run_verl_solver(
            current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name,
            extra_verl_overrides=VERL_OVERRIDES,
        )

        # 4. MATH-500 eval + GPT-mini boxed-only recheck (built into eval_math500).
        eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
        save_summary()

        # 5. Reload vllm-solver with the new ckpt for next iter's MCMC scoring.
        if it < deo.config.NUM_ITERATIONS:
            deo.reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (base-agreement ablation):")
    print("=" * 60)
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f}  ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

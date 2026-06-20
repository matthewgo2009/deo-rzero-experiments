"""
Ablation: pseudo-label always from solver_v1 (FROZEN labeler across all iters).

Use canonical mcmc pools as questions. For iter N ∈ [2..5], M-vote with
solver_v1 (NEVER updated for labeling — the same eval=76.8 ckpt throughout),
filter, train solver_vN from solver_v(N-1).

Compare against:
  canonical          76.8 → 75.0 → 71.8 → 71.2 → 69.0  (current solver labels, drift)
  reuse_iter1        76.8 → 76.2 → 76.8 → 76.6 → 73.8  (fixed iter1 data + labels)
  base_agreement_v2  76.8 → 74.8 → 75.4 → 76.0 → 75.0  (base+solver agree)
  base_only_pseudo   76.8 → 76.0 → 76.0 → 75.4 → 76.8  (base label only)
  solver_v1_label    76.8 → ?    → ?    → ?    → ?      (this run)

Hypothesis: solver_v1 labels (76.8 acc, p_hat 0.51 on filter-band) should be
higher quality than base labels (72.2 acc, p_hat 0.31 on filter-band) →
expect >= base_only_pseudo trajectory. But solver_v1 has its own biases (it
was trained), so labels may be more "self-confident" but not necessarily
more correct.
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo
import openai

deo.config.MODEL_ABBR = "deo_solver_v1_label"

# Override base_client to point at vllm_solver_check (port 8001) which is
# loaded with solver_v1. We use the "base_client" plumbing because that's
# what score_with_base() in this script would otherwise hit; redirecting it
# at the module level keeps the rest of the code identical to other ablations.
LABEL_URL = "http://localhost:8001/v1"
deo._client_base = openai.OpenAI(api_key="EMPTY", base_url=LABEL_URL, timeout=1800.0)

START_CKPT = "/storage/models/deo_solver_v1_label_solver_v1/global_step_15/actor/huggingface"
CANONICAL_POOLS_DIR = "/storage/canonical_mcmc_pools"
VERL_OVERRIDES = ["data.rollout_batch_size=64"]
SCORE_CHUNK_QUESTIONS = 100


def score_with_solver_v1(questions, tokenizer):
    """M-vote each question with the (frozen) solver_v1 model. Chunked for
    HTTP timeout safety. Returns (p_hat_list, pseudo_label_list)."""
    if not questions:
        return [], []
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


def filter_and_push(questions, p_hats, pseudos, repo_name, config_name):
    """Cascade: regex → non-null pseudo → p_hat ∈ [0.3, 0.8]."""
    kept = []
    stats = {"regex": 0, "no_pseudo": 0, "phat_oob": 0}
    for q, ph, ps in zip(questions, p_hats, pseudos):
        if deo.BAD_RE.search(q):
            stats["regex"] += 1
            continue
        if ps is None:
            stats["no_pseudo"] += 1
            continue
        if not (deo.config.MIN_SCORE <= ph <= deo.config.MAX_SCORE):
            stats["phat_oob"] += 1
            continue
        kept.append({"problem": q, "answer": ps, "score": ph})

    n = len(questions)
    print(f"[filter] {n} candidates → {len(kept)} kept")
    print(f"  drops: regex={stats['regex']} | no_pseudo={stats['no_pseudo']} | "
          f"phat_oob={stats['phat_oob']}")
    if not kept:
        raise SystemExit("ERROR: 0 questions passed filter")

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

    # vllm_solver_check (port 8001, loaded with solver_v1) — used for labeling all iters.
    # vllm_solver_check stays running; we never reload it (frozen labeler).
    deo.wait_for_vllm_ready(LABEL_URL, label="solver_v1-labeler@8001")

    if not os.path.exists(START_CKPT):
        sys.exit(f"ERROR: starting ckpt not found: {START_CKPT}")
    if not os.path.exists(CANONICAL_POOLS_DIR):
        sys.exit(f"ERROR: canonical mcmc pools dir not found: {CANONICAL_POOLS_DIR}")

    eval_history = {"iter_0_baseline": 0.722, "iter_1": 0.768}
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_solver_v1_label.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)
    save()

    current_solver = START_CKPT

    for it in range(2, deo.config.NUM_ITERATIONS + 1):
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {it}/5 (solver_v1-as-frozen-labeler) ===")
        print(f"{'=' * 60}\n")

        pool_path = f"{CANONICAL_POOLS_DIR}/mcmc_iter_{it}.json"
        pool = json.load(open(pool_path))
        questions = [d["question"] for d in pool]
        print(f"[pool] {len(questions)} questions from canonical {pool_path}")

        t0 = time.time()
        p_hats, pseudos = score_with_solver_v1(questions, tokenizer)
        print(f"[score-solver_v1] done in {time.time() - t0:.0f}s")

        filter_and_push(questions, p_hats, pseudos, exp_name, exp_name)

        merged_ckpt = deo.run_verl_solver(
            current_solver, f"{deo.config.HF_USER}/{exp_name}", exp_name,
            extra_verl_overrides=VERL_OVERRIDES,
        )
        eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
        save()
        current_solver = merged_ckpt
        # IMPORTANT: do NOT reload vllm_solver_check. solver_v1 stays frozen as the labeler.

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (solver_v1 frozen labeler):")
    print("=" * 60)
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f}  ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

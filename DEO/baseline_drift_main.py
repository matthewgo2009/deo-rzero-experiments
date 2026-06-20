"""
True KL-fix baseline (no MCMC walk, DRIFTING labeler — canonical-style).

Proper apples-to-apples "remove MCMC walk only" baseline:
  - MCMC_STEPS = 0  (init pool only, no mutation walk)
  - vllm_solver reloaded EVERY iter to the just-trained solver_vN
    (so each iter's labeler is the previous iter's solver = canonical
    behavior; only difference from canonical is MCMC walk is skipped)
  - KL ref pinned to base (canonical verl patch)
  - 4-stage filter (regex + p_hat + pseudo + LLM-judge), no conf filter
  - rollout_batch_size=64 (init-only pool ~450 entries after filter < default 512)

Compare to:
  canonical (walk + drift):                 76.8 → 69.0  drop -7.8
  baseline_klfix (no walk + FROZEN sv1):    77.6 → 77.4  drop -0.2  ← not pure baseline (frozen labeler is a separate variable)
  THIS (no walk + DRIFTING labeler):        ?            ?           ← isolates walk effect cleanly
"""
import json
import os
import sys

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo

deo.config.MODEL_ABBR = "deo_baseline_drift"
deo.config.MCMC_STEPS = 0

VERL_OVERRIDES = ["data.rollout_batch_size=64"]


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

    eval_history = {}
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_baseline_drift.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    eval_history["iter_0_baseline"] = deo.eval_math500(deo.config.MODEL_NAME, "iter 0 baseline")
    save()

    current_solver = deo.config.MODEL_NAME
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {it}/{deo.config.NUM_ITERATIONS} (no walk + DRIFTING labeler) ===")
        print(f"{'=' * 60}")
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_baseline_drift.log"

        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_baseline_drift.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        deo.filter_and_push(train_data, repo_name, exp_name)
        merged_ckpt = deo.run_verl_solver(
            current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name,
            extra_verl_overrides=VERL_OVERRIDES,
        )

        eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
        save()

        if it < deo.config.NUM_ITERATIONS:
            deo.reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (no walk + drifting labeler):")
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

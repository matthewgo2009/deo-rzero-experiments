"""
KL-fix baseline_no_mcmc with frozen solver_v1 labeler.

Combines TWO ablation knobs:
  1. MCMC_STEPS = 0  → no challenger walk; only init pool per iter
  2. Labeler frozen at solver_v1 after iter 1 (same as frozen_sv1_full)

Per iter:
  - init pool of 1500 questions sampled fresh from base via challenger prompt
  - r_unc scoring + pseudo-label by whatever solver vllm_solver currently has
  - filter + verl GRPO (KL ref pinned to base, verl patch)
  - eval + recheck
  - reload vllm_solver to solver_v1 ONLY after iter 1; never reload again

So iter 2-5 all use solver_v1 for both r_unc scoring (during init pool
sample acceptance) AND pseudo-label generation. The trainee solver
(verl actor) still evolves each iter.

Fills the missing iter 3-5 baseline data point that the original
broken-KL DEO_baseline never produced (it crashed at iter 4).
"""
import json
import os
import sys

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo

deo.config.MODEL_ABBR = "deo_baseline_klfix"
# Key knob: zero MCMC walk steps. Only the init pool is sampled.
deo.config.MCMC_STEPS = 0


def main():
    deo.config.HF_TOKEN = deo.load_hf_token()
    deo.hf_login(token=deo.config.HF_TOKEN)
    for d in ["datasets", "logs", "models", "evaluation", "temp_results",
              "generated_question"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{d}", exist_ok=True)

    tokenizer = deo.AutoTokenizer.from_pretrained(
        deo.config.MODEL_NAME, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    deo.wait_for_vllm_ready(deo.config.VLLM_BASE_URL, label="vllm-base")
    for url in deo.config.VLLM_SOLVER_URLS:
        deo.wait_for_vllm_ready(url, label=f"vllm-solver@{url}")

    eval_history = {}
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_baseline_klfix.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    eval_history["iter_0_baseline"] = deo.eval_math500(
        deo.config.MODEL_NAME, "iter 0 baseline"
    )
    save()

    current_solver = deo.config.MODEL_NAME
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        print(f"\n{'=' * 60}\n=== Iteration {it}/5 (baseline KL-fix, no MCMC walk) ===\n{'=' * 60}")
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_baseline_klfix.log"

        # MCMC_STEPS=0 → only init pool. Walk loop is a no-op.
        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_baseline_klfix.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        deo.filter_and_push(train_data, repo_name, exp_name)
        merged_ckpt = deo.run_verl_solver(
            current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name
        )
        eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
        save()

        # FROZEN LABELER: reload vllm_solver ONLY after iter 1 (so labeler
        # is solver_v1 from iter 2 onward, never updated to v2/v3/v4).
        if it == 1:
            deo.reload_vllm_solver(merged_ckpt)
            print(f"[frozen-labeler] reloaded vllm_solver with solver_v1. "
                  f"Will NOT reload again for iter 2-5.")
        current_solver = merged_ckpt

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (KL-fix baseline, no MCMC walk):")
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f}  ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

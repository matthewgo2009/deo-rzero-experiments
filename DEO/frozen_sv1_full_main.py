"""
Ablation: frozen solver_v1 as labeler, FULL pipeline (re-run MCMC each iter).

Identical to canonical DEO main(), with one single change: after iter 1
training, reload vllm_solver to solver_v1 (this is canonical behavior),
but DO NOT reload again after iter 2/3/4. The vllm_solver endpoints stay
loaded with solver_v1 for the entire iter 2-5, meaning:
  - MCMC walk r_unc scoring uses solver_v1
  - filter_and_push pseudo-labels use solver_v1

The trainee solver still evolves each iter (verl actor.model.model_path
= solver_v(N-1), produces solver_vN).

Compare to:
  canonical          (76.8 → 75.0 → 71.8 → 71.2 → 69.0, drop −7.8)
  solver_v1_label    (76.8 → 76.6 → 77.2 → 75.6 → 77.2, drop +0.4)
The latter used canonical's archived mcmc pools (no walk). This ablation
re-runs the full MCMC walk each iter with frozen solver_v1 driving r_unc.
"""
import json
import os
import sys

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo

deo.config.MODEL_ABBR = "deo_frozen_sv1_full"


def main():
    deo.config.HF_TOKEN = deo.load_hf_token()
    deo.hf_login(token=deo.config.HF_TOKEN)
    for d in ["datasets", "logs", "models", "evaluation", "temp_results",
              "generated_question"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{d}", exist_ok=True)

    tokenizer = deo.AutoTokenizer.from_pretrained(deo.config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wait for vllm endpoints (base + 3 solver DP — all initially loaded with base).
    deo.wait_for_vllm_ready(deo.config.VLLM_BASE_URL, label="vllm-base")
    for url in deo.config.VLLM_SOLVER_URLS:
        deo.wait_for_vllm_ready(url, label=f"vllm-solver@{url}")

    eval_history = {}
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_frozen_sv1_full.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    # Baseline (iter 0): MATH-500 acc on untouched Qwen3-4B-Base.
    eval_history["iter_0_baseline"] = deo.eval_math500(deo.config.MODEL_NAME, "iter 0 baseline")
    save()

    current_solver = deo.config.MODEL_NAME  # iter 1 trains from base
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {it}/{deo.config.NUM_ITERATIONS} (frozen solver_v1 from iter 2) ===")
        print(f"{'=' * 60}")
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_fsv1.log"

        # 1. MCMC pool + walk (uses base for proposals, current vllm_solver for r_unc).
        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_fsv1.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        # 2. Filter + push (regex + p_hat + LLM-judge + local dump).
        deo.filter_and_push(train_data, repo_name, exp_name)

        # 3. verl GRPO from current solver ckpt.
        merged_ckpt = deo.run_verl_solver(current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name)

        # 4. MATH-500 eval + GPT-mini recheck.
        eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
        save()

        # 5. KEY DIFFERENCE FROM CANONICAL: reload vllm_solver only ONCE,
        #    after iter 1, to load solver_v1. After that, never reload —
        #    solver_v1 stays as the labeler (and the MCMC r_unc model)
        #    for iter 2-5.
        if it == 1:
            deo.reload_vllm_solver(merged_ckpt)
            print(f"[frozen-labeler] reloaded vllm_solver with solver_v1. "
                  f"Will NOT reload again for iter 2-5.")

        # Trainee solver still advances each iter (verl input for next iter).
        current_solver = merged_ckpt

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (frozen solver_v1 + full MCMC walks):")
    print("=" * 60)
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f}  ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

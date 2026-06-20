"""
Canonical DEO pipeline (drifting solver-vN labeler + per-iter MCMC walk),
EXCEPT the MCMC walk uses the pipelined implementation from mcmc_pipelined.py.

Everything else (KL fix, regex filter, LLM-judge filter, GPT-mini recheck,
DP=3 solver, verl) is identical to canonical mcmc_deo_vllm.main(). Only
the MCMC walk is faster.

Used to benchmark wall-clock speedup of the pipelined MCMC implementation
against canonical at iso-acc. iter 1 result should match canonical iter 1
(76-77% rechecked) within noise.
"""
import json
import os
import sys

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo
import mcmc_pipelined as deop

deo.config.MODEL_ABBR = "deo_canonical_pipelined"


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
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_pipelined_canonical.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    eval_history["iter_0_baseline"] = deo.eval_math500(
        deo.config.MODEL_NAME, "iter 0 baseline"
    )
    save()

    current_solver = deo.config.MODEL_NAME
    for it in range(1, deo.config.NUM_ITERATIONS + 1):
        print(f"\n{'=' * 60}\n=== Iteration {it}/5 (canonical pipelined) ===\n{'=' * 60}")
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_pipelined.log"

        # KEY DIFFERENCE: pipelined MCMC instead of canonical generate_batch_mcmc
        train_data = deop.generate_batch_mcmc_pipelined(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_pipelined.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        deo.filter_and_push(train_data, repo_name, exp_name)
        merged_ckpt = deo.run_verl_solver(
            current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name
        )
        eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
        save()

        # Canonical labeler-drift behavior: reload vllm_solver every iter.
        if it < deo.config.NUM_ITERATIONS:
            deo.reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (canonical pipelined):")
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

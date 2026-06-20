"""
Resume baseline_no_mcmc_klfix from iter 4. Original run crashed at iter 3
eval (vllm device-empty error). solver_v3 merged ckpt is on disk and
already eval'd manually (74.0%).

Runs iter 4 + iter 5 with:
  - Frozen solver_v1 labeler (vllm_solver still loaded with solver_v1)
  - No MCMC walk (init pool only)
  - KL ref pinned to base (canonical verl patch)
  - Each iter splits verl + eval into separate steps; eval is wrapped in
    try/except so a single eval crash doesn't kill the chain.
"""
import json
import os
import sys
import subprocess

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo

deo.config.MODEL_ABBR = "deo_baseline_klfix"
deo.config.MCMC_STEPS = 0  # init pool only

SUMMARY_PATH = f"{deo.config.STORAGE_ROOT}/results_summary_baseline_klfix.json"
START_ITER = 4
START_CKPT = "/storage/models/deo_baseline_klfix_solver_v3/global_step_15/actor/huggingface"


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

    # Load existing summary (iter 0, 1, 2, 3 should be present)
    with open(SUMMARY_PATH) as f:
        eval_history = json.load(f)
    print(f"[resume] loaded existing summary: {eval_history}")

    def save():
        with open(SUMMARY_PATH, "w") as f:
            json.dump(eval_history, f, indent=2)

    if not os.path.exists(START_CKPT):
        sys.exit(f"ERROR: starting ckpt not found: {START_CKPT}")

    current_solver = START_CKPT

    for it in range(START_ITER, deo.config.NUM_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {it}/{deo.config.NUM_ITERATIONS} (resume baseline_klfix) ===")
        print(f"{'=' * 60}")
        exp_name = f"{deo.config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_baseline_klfix.log"

        # init pool only (MCMC_STEPS=0); labeling done by vllm_solver (= solver_v1, frozen)
        train_data = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_baseline_klfix.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        deo.filter_and_push(train_data, repo_name, exp_name)
        merged_ckpt = deo.run_verl_solver(current_solver, f"{deo.config.HF_USER}/{repo_name}", exp_name)

        # Eval wrapped in try/except so a vllm-device-empty bug doesn't kill the chain
        try:
            eval_history[f"iter_{it}"] = deo.eval_math500(merged_ckpt, f"iter {it}")
            save()
        except Exception as exc:
            print(f"[resume] eval iter {it} failed in-process: {type(exc).__name__}")
            print(f"[resume] merged_ckpt at {merged_ckpt}, eval can be redone in fresh container")
            eval_history[f"iter_{it}_eval_failed"] = True
            save()

        # NEVER reload vllm_solver — keep solver_v1 as frozen labeler
        current_solver = merged_ckpt

    print("\n" + "=" * 60)
    print("RESUMED RUN DONE. Final summary:")
    for k, v in eval_history.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

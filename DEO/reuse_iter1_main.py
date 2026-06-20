"""
Ablation: train solver_v2..v5 on iter 1's fixed HF dataset
(yuyang322/deo_qwen3_4b_base_solver_v1, 924 entries) instead of regenerating
a fresh MCMC pool each iter.

Question this answers: is DEO's monotonic degradation past iter 1 caused by
  (a) per-iter MCMC regeneration producing increasingly noisy pseudo-labels
      (each iter's pseudo-labels come from the just-trained solver, which is
      biased; so iter-2's dataset is built from solver_v1's noisy votes), OR
  (b) over-training on a static set of noisy-pseudo-label data?

If iter 2-5 keeps improving (or plateaus higher) on fixed iter1 data → (a).
If iter 2-5 still decays similarly → (b).

Starts from existing solver_v1 ckpt (canonical DEO archive, recheck'd acc 76.8).
No MCMC, no challenger, no filter — purely solver GRPO on fixed data.
"""
import json
import os
import sys

sys.path.insert(0, "/workspace_deo")
import mcmc_deo_vllm as deo


# Keep all outputs under separate prefix so we don't collide with canonical DEO
deo.config.MODEL_ABBR = "deo_reuse_iter1"

# The fixed dataset reused across all iters.
ITER1_HF = "yuyang322/deo_qwen3_4b_base_solver_v1"

# Starting point = solver_v1 from canonical run (restored to live storage by tmux launcher).
START_CKPT = "/storage/models/deo_reuse_iter1_solver_v1/global_step_15/actor/huggingface"


def main():
    # HF login (so verl can pull the private iter1 dataset if cache miss)
    deo.config.HF_TOKEN = deo.load_hf_token()
    from huggingface_hub import login as hf_login
    hf_login(token=deo.config.HF_TOKEN)

    for d in ["models", "evaluation", "temp_results", "datasets", "logs",
              "generated_question"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{d}", exist_ok=True)

    # Carry forward the canonical iter-0/iter-1 numbers (GPT-mini recheck'd).
    # solver_v1 ckpt is symlink/copy of canonical, so iter 1 acc IS 76.8.
    eval_history = {"iter_0_baseline": 0.722, "iter_1": 0.768}
    summary_path = f"{deo.config.STORAGE_ROOT}/results_summary_reuse_iter1.json"

    def save():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    save()

    if not os.path.exists(START_CKPT):
        sys.exit(
            f"ERROR: starting ckpt not found: {START_CKPT}\n"
            "  Copy from canonical archive before launching this script."
        )

    current = START_CKPT

    for it in range(2, 6):
        exp_name = f"deo_reuse_iter1_solver_v{it}"
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {it}/5 — training solver_v{it} on FIXED iter1 dataset ===")
        print(f"=== input solver:   {current}")
        print(f"=== train dataset:  {ITER1_HF}@train (canonical iter 1 filtered)")
        print(f"{'=' * 60}\n")

        merged_ckpt = deo.run_verl_solver(current, ITER1_HF, exp_name)
        acc = deo.eval_math500(merged_ckpt, f"iter {it}")
        eval_history[f"iter_{it}"] = acc
        save()
        current = merged_ckpt

    print("\n" + "=" * 60)
    print("FINAL TRAJECTORY (reuse_iter1 ablation):")
    print("=" * 60)
    for label, acc in eval_history.items():
        print(f"  {label:25s} {acc:.4f}  ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

---
name: project-rzero-vs-deo-findings
description: Key results from 2026-05-15/16 fair comparison of DEO (MCMC challenger) vs R-Zero (GRPO questioner) on Qwen3-4B-Base; both fail past iter 1 in different modes
metadata: 
  node_type: memory
  type: project
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

Fair comparison conducted 2026-05-15 → 05-16 on Qwen3-4B-Base. All artifacts
archived at `/eph/nvme0/yyd/paper_data/` with `SUMMARY.md`.

**Final per-iter MATH-500 acc (GPT-4o-mini boxed-only rechecked):**

| iter | DEO | R-Zero |
|---|---|---|
| 0 base | 72.2 | 72.2 |
| 1 | **76.8** | **76.2** |
| 2 | 75.0 | 74.6 |
| 3 | 71.8 | 76.8 (training no-op, only 74 entries) |
| 4 | 71.2 | CRASH (questioner produced 17 valid q's < verl batch) |
| 5 | 69.0 | — |

**Two failure modes:**

1. **DEO** ran all 5 iters, solver acc monotonically degrades past iter 1
   because MCMC picks `r_unc≈0.5` questions → ~50% wrong pseudo-labels → noisy
   self-training. KL anchor pinned to base **delays** the collapse (DEO 5-iter
   drop = −7.8 from peak vs broken-KL 5-iter drop ≈ −5.6 from peak) but
   doesn't prevent it.

2. **R-Zero** challenger format-fidelity collapses exponentially under GRPO:
   extract success 53% → 9% → 3% → **0.7%** by iter 4. Reward function only
   measures solver M-vote consistency on generated questions, no penalty for
   malformed `<question>...</question> \\boxed{}` output, so questioner
   drifts to "hard" while losing format. Iter 4 dataset (17 entries) too
   small even at our `rollout_batch_size=64` override → verl
   `train_dataloader >= 1` assertion fails. **R-Zero paper's README only
   reporting iter 1-3 for Qwen3-4B-Base is exactly this collapse — they
   didn't choose to stop at 3, the pipeline forces them to.**

**How to apply when planning next experiments:**
- Don't propose "just run R-Zero on Base for longer iters" — pipeline can't.
- For closing the pseudo-label-noise ceiling, both schemes need a per-question
  GPT grader on M-vote answer correctness; outside the "zero supervision" charter
  but might be the only way forward.
- For DEO trajectory analysis, the iter-1 peak is real; iter 2-5 monotonic
  decay is a robust pattern across broken-KL run AND KL-fix+filter run AND
  KL-fix+filter+DP run. Three different DEO configurations all show same
  collapse shape — it's the self-training dynamics, not a tunable.

Related: [[project-math500-eval]] (eval methodology), [[deploy-md-pointer]].

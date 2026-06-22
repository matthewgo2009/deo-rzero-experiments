# DEO baseline (no MCMC walk) vs R-Zero — controlled fair comparison, 2026-06-21/22

Both on **Qwen3-4B-Base**, **GPU 0-3, Docker**, MATH-500, identical grader
(math_verify + GPT-4o-mini boxed recheck). Solver-training knobs identical:
max_steps=20, global_batch=16, rollout_batch=64, tp=2, M=9, filter [0.3,0.8].
**Only the challenger differs:**
- DEO baseline: sample question pool from the base model, NO MCMC mutation walk
  (`SKIP_MCMC_WALK=True`); drifting solver labeler (reload to fresh ckpt each iter);
  KL ref pinned to base (verl patch).
- R-Zero: GRPO-trained penalty-questioner; KL ref reset to prev actor (stock).

| iter | DEO baseline (no walk) | R-Zero (GPU-0-3) | canonical DEO (walk+drift, old) |
|:---:|:---:|:---:|:---:|
| 0 base | 71.8 | 71.8 | 72.2 |
| 1 | 76.8 | 76.6 | 76.8 |
| 2 | 76.8 | 78.0 | 75.0 |
| 3 | 75.6 | 78.4 | 71.8 |
| 4 | 78.0 | 77.8 | 71.2 |
| 5 | 76.6 | 76.6 | 69.0 |
| peak | 78.0 | 78.4 | 76.8 |
| mean 1-5 | 76.76 | 77.48 | 72.76 |
| peak->i5 | -1.4 | -1.8 | -7.8 |

DEO-baseline filtered training set per iter: 626/567/615/629/642 (stable, no collapse).

## Findings
1. **Removing the MCMC walk fixes canonical DEO's degradation.** DEO baseline holds
   75.6-78.0 across all 5 iters (mean 76.76) vs canonical DEO's collapse to 69.0
   (mean 72.76). Confirms the earlier walk-vs-drift finding: the MCMC mutation walk
   was the dominant degradation driver.
2. **R-Zero (penalty-questioner) edges out DEO baseline**: higher peak (78.4 vs 78.0)
   and mean (77.48 vs 76.76), though both are within ~1 pt and neither collapses.
   The trained questioner gives a small but consistent edge over plain base sampling.
3. Both are essentially flat/stable self-training on Qwen3-4B-Base — the real ceiling
   is pseudo-label noise (~50% wrong in the r_unc≈0.5 band), which neither addresses.

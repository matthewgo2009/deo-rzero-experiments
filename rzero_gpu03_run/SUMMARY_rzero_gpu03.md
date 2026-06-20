# R-Zero (official, penalty-questioner) vs DEO — GPU-0-3 rerun, 2026-06-19/20

Fresh fair comparison on **Qwen3-4B-Base**, MATH-500, graded with the **same
grader as DEO**: `math_verify` primary (R-Zero `generate.py`) + GPT-4o-mini
boxed-only recheck (`results_recheck_math500_mini.py`).

R-Zero run on a **GPU-0-3 (2+2)** layout (verl training on GPU 0,1 while 2 vLLM
solver-scoring servers run on GPU 2,3; solver phase uses all 4). No algorithm /
hyperparameter change vs upstream — only GPU placement, plus the documented
`rollout_batch_size=64` + `global_batch_size=16` override (small filtered sets)
and disabled unused internal math12k validation. Stock R-Zero KL (ref reset to
prev actor each iter) — the KL-pin-to-base patch is DEO's, NOT applied here.

## Headline (MATH-500, GPT-mini rechecked)

| iter | R-Zero (this rerun) | DEO canonical | DEO solver_v1_label (best ablation) | old R-Zero (2026-05) |
|:---:|:---:|:---:|:---:|:---:|
| 0 base | 71.8 | 72.2 | 72.2 | 72.2 |
| 1 | 76.6 | 76.8 | 76.8 | 76.2 |
| 2 | 78.0 | 75.0 | 76.6 | 74.6 |
| 3 | **78.4** | 71.8 | 77.2 | 76.8 (≈no-op) |
| 4 | 77.8 | 71.2 | 75.6 | CRASH |
| 5 | 76.6 | 69.0 | 77.2 | — |
| **peak** | **78.4** | 76.8 | 77.2 | 76.8 |
| **peak→i5** | **−1.8** | −7.8 | +0.4 | — |

Per-iter filtered training set grew monotonically: 1642 → 2010 → 2028 → 2280 →
2768. No format collapse (solver_v5 boxed-missing rate = 1.4%).

## Findings

1. **The official penalty-questioner R-Zero does NOT collapse on Qwen3-4B-Base.**
   It climbs to a peak of 78.4 at iter 3 and holds 76.6–78.4 across all 5 iters,
   with format fidelity intact and a growing question pool. This **overturns the
   earlier conclusion** ("R-Zero can't sustain past iter 1 / crashes at iter 4 on
   Base") — that was an artifact of the *old* questioner with no format-penalty
   reward (extract success collapsed 53%→0.7%, iter-4 dataset 17 rows < batch).

2. **This rerun's R-Zero beats DEO canonical** (peak 78.4 vs 76.8; iter-5 76.6 vs
   69.0) and is on par with DEO's best ablation (solver_v1_label, peak 77.2).

3. **Grading caveat (resolved):** solver_v5's first recheck failed (OpenAI quota
   429, api-errs=186/186), leaving its score at the raw math_verify value 62.8 —
   a *grading artifact, not degradation*. Re-run with a funded key: 62.8 → 76.6.
   Raw math_verify was stable all iters (58–65), confirming no real iter-5 drop.

## Artifacts
- `results_summary_rzero_gpu03.json` — rechecked + raw + trainset sizes
- `evaluation/<model>/results_math.json` — per-iter 500 graded responses
- `final_results.jsonl` — one line per model
- `logs/*.log` — full run logs

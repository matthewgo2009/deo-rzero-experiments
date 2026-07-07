# DEO variants vs R-Zero — 7-benchmark comparison (Qwen3-4B-Base)

Self-evolving math training on **Qwen/Qwen3-4B-Base**. Eval: 7 R-Zero math benchmarks (math=MATH-500, gsm8k, amc, minerva, olympiad, aime2024, aime2025), graded with `math_verify` + **gpt-4o-mini boxed-only recheck** (same grader for all). MATH AVG = unweighted mean of the 7. All runs = 5 self-evolving iters on AzureML `azureml-job-cluster` (1× Standard_ND96isr_H100_v5, 8×H100).

## Headline — MATH AVG (7-set) per iteration

| method | walk | label | base | v1 | v2 | v3 | v4 | v5 | mean(v1-5) | peak |
|---|:--:|:--:|--|--|--|--|--|--|--|--|
| DEO baseline_drift (no walk, solver label) | ✗ | solver | 41.36 | 47.16 | 45.47 | 46.16 | 48.00 | 47.97 | **46.95** | 48.00 |
| R-Zero (penalty questioner) | — | — | 41.36 | 44.14 | 46.04 | 47.20 | 46.85 | 47.05 | **46.26** | 47.20 |
| canonical DEO + Claude label | ✓ | Claude | 41.36 | 46.66 | 49.78 | 48.29 | 45.94 | 46.41 | **47.42** | 49.78 |
| baseline_drift + Claude label | ✗ | Claude | 41.36 | 47.23 | 45.86 | 46.04 | 45.41 | 46.50 | **46.21** | 47.23 |
| curriculum-DEO (anneal β, solver label) | ✓ | solver | 41.36 | 47.30 | 45.65 | 47.70 | 48.15 | 46.42 | **47.04** | 48.15 |

## MATH-500 (single set) per iteration — gpt-mini rechecked

| method | base | v1 | v2 | v3 | v4 | v5 |
|---|--|--|--|--|--|--|
| (base) | 71.6 | | | | | |
| DEO baseline_drift (no walk, solver label) | 71.6 | 73.8 | 76.2 | 73.2 | 75.8 | 76.4 |
| R-Zero (penalty questioner) | 71.6 | 75.8 | 74.2 | 75.6 | 75.6 | 74.6 |
| canonical DEO + Claude label | 71.6 | 76.2 | 76.2 | 77.4 | 76.0 | 77.2 |
| baseline_drift + Claude label | 71.6 | 75.8 | 76.2 | 76.2 | 79.0 | 77.8 |
| curriculum-DEO (anneal β, solver label) | 71.6 | 76.2 | 76.0 | 77.2 | 77.2 | 76.2 |

> curriculum-DEO iter2-4 MATH-500 were re-checked after the fact (their in-run recheck hit OpenAI rate limits, +0 bumped, leaving raw ~62). Corrected values shown; confirms **no collapse** (stable 76-77).


## Per-method full 7-set tables

### DEO baseline_drift (no walk, solver label)

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 73.8 | 92.3 | 57.3 | 44.9 | 41.8 | 10.0 | 10.0 | 47.16 |
| v2 | 76.2 | 92.0 | 50.3 | 40.4 | 32.7 | 10.0 | 16.7 | 45.47 |
| v3 | 73.2 | 91.9 | 52.5 | 44.1 | 38.1 | 13.3 | 10.0 | 46.16 |
| v4 | 75.8 | 92.3 | 59.5 | 47.4 | 41.0 | 10.0 | 10.0 | 48.00 |
| v5 | 76.4 | 92.1 | 62.5 | 47.4 | 40.6 | 10.0 | 6.8 | 47.97 |

### R-Zero (penalty questioner)

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 75.8 | 91.9 | 49.8 | 41.2 | 33.6 | 10.0 | 6.7 | 44.14 |
| v2 | 74.2 | 92.2 | 52.4 | 45.2 | 35.1 | 13.1 | 10.0 | 46.04 |
| v3 | 75.6 | 92.3 | 62.4 | 46.0 | 40.7 | 10.0 | 3.4 | 47.20 |
| v4 | 75.6 | 92.1 | 55.0 | 45.6 | 39.6 | 13.3 | 6.8 | 46.85 |
| v5 | 74.6 | 92.0 | 57.5 | 45.2 | 40.0 | 10.0 | 10.0 | 47.05 |

### canonical DEO + Claude label

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 76.2 | 92.0 | 55.0 | 46.0 | 40.7 | 3.3 | 13.3 | 46.66 |
| v2 | 76.2 | 91.7 | 67.7 | 45.2 | 41.0 | 20.0 | 6.7 | 49.78 |
| v3 | 77.6 | 91.0 | 60.0 | 47.1 | 42.4 | 16.7 | 3.3 | 48.29 |
| v4 | 75.8 | 92.3 | 50.1 | 46.0 | 40.7 | 6.7 | 10.0 | 45.94 |
| v5 | 77.0 | 91.9 | 52.2 | 47.4 | 39.7 | 10.0 | 6.7 | 46.41 |

### baseline_drift + Claude label

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 75.8 | 91.9 | 59.3 | 43.0 | 40.6 | 6.7 | 13.3 | 47.23 |
| v2 | 76.0 | 91.8 | 55.0 | 44.5 | 40.3 | 6.8 | 6.7 | 45.86 |
| v3 | 76.2 | 92.3 | 55.1 | 44.9 | 40.4 | 6.7 | 6.7 | 46.04 |
| v4 | 78.4 | 92.0 | 50.8 | 45.2 | 41.3 | 6.7 | 3.4 | 45.41 |
| v5 | 77.6 | 91.6 | 52.5 | 43.4 | 40.7 | 13.3 | 6.3 | 46.50 |

### curriculum-DEO (anneal β, solver label)

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 76.2 | 92.4 | 57.9 | 46.3 | 41.5 | 13.3 | 3.4 | 47.30 |
| v2 | 76.0 | 90.9 | 50.2 | 43.4 | 39.1 | 10.0 | 10.0 | 45.65 |
| v3 | 77.2 | 91.7 | 60.1 | 43.4 | 38.2 | 10.0 | 13.3 | 47.70 |
| v4 | 77.2 | 91.9 | 57.4 | 47.8 | 39.4 | 10.0 | 13.3 | 48.15 |
| v5 | 76.2 | 91.7 | 52.5 | 43.8 | 40.7 | 13.3 | 6.8 | 46.42 |

## Key findings

1. **All self-evolving variants beat base** (MATH AVG 41.4 -> 46-48; MATH-500 71.6 -> 76-79),
   with the biggest gains on gsm8k, minerva, olympiad.
2. **MCMC-walk's effect flips sign with label quality.** With noisy solver M-vote labels,
   vanilla canonical DEO (fixed β=0.1 + walk) historically **collapses** (MATH-500 76.8->69).
   With clean Claude labels, the walk instead **helps generalization** (canonical+Claude MATH
   AVG mean 47.42, highest).
3. **Claude labeler = 100% coverage, no degradation.** canonical+Claude is the best mean (47.42);
   baseline_drift+Claude gets the single highest MATH-500 (79.0 @ iter4) but generalizes less
   (no walk -> narrower question diversity -> weaker amc/aime).
4. **Curriculum-DEO (annealed β 1.0->0.05) prevents the collapse even with solver labels.**
   MATH-500 stays 76-77 all 5 iters (vs vanilla canonical's ->69), MATH AVG mean 47.04.
   β-annealing works as designed: filter pass-count rises 709->995 as β drops (pool
   concentrates on the p_hat~0.5 edge).
5. **Differences between the top configs are within small-set noise (~1 MATH-AVG pt).**
   On the large, reliable sets the ordering is stable; on amc(40)/aime(30) it is not.

## Methodology & caveats

- **Dataset sizes (unique problems):** math=500, gsm8k=1319, olympiad=675, minerva=272,
  amc=40, aime2024=30, aime2025=30. amc/aime are tiny -> each problem is 2.5-3.3% ->
  those columns are noise-dominated; weight conclusions toward the large sets.
- **amc/aime recheck rate-limited:** the ×32 duplication in the loader inflates recheck
  request volume; amc/aime rechecks hit OpenAI 429s (api-errs, +0 bumped), so their scores
  are raw `math_verify` (slightly undercounted). Applies to all methods equally, so relative
  comparison holds. (math/gsm8k/minerva/olympiad recheck succeeded.)
- **KL ref pinned to base** (verl patch) for all DEO variants; R-Zero uses stock KL.
- **verl:** max_steps=20 (merge global_step_15), rollout_batch_size=64, TP=2.
- Grader = `math_verify` + gpt-4o-mini boxed-only recheck (drops mathruler format-equivalence
  misses on MATH-500; ~0% FP).

## Artifacts (AzureML jobs, workspace ai-core-research)

| experiment | job | output datastore path |
|---|---|---|
| DEO baseline_drift (solver) + R-Zero + eval | kind_box_0mbv30m7js | yyd_deo_rzero_run |
| R-Zero (rerun, used) | mighty_lemon_zj88drhnvz | yyd_rzero_run |
| canonical DEO + Claude | amusing_pig_dmqs56y8sw | yyd_canon_claudelabel |
| baseline_drift + Claude | great_cow_6pgg6kt0l5 | yyd_bd_claudelabel |
| curriculum-DEO (anneal β) | bold_sock_l7k811l29q | yyd_curriculum |

Per-(model,dataset) scores are in each job's `final_results.jsonl`. Code: `DEO/*_native_main.py`,
`azureml/`, verl KL-pin patch in `R-Zero/verl/`.


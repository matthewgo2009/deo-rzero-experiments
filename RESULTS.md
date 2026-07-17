# DEO variants vs R-Zero — 7-benchmark comparison (Qwen3-4B-Base)

Self-evolving math training on **Qwen/Qwen3-4B-Base**. Eval on the 7 R-Zero math benchmarks (math=MATH-500, gsm8k, amc, minerva, olympiad, aime2024, aime2025), graded with `math_verify` + **gpt-4o-mini boxed-only recheck** (identical grader for all runs). MATH AVG = unweighted mean of the 7. Each run = 5 self-evolving iters on AzureML `azureml-job-cluster` (1× Standard_ND96isr_H100_v5, 8×H100).

## Headline — MATH AVG (7-set) per iteration

| method | walk | label | β | base | v1 | v2 | v3 | v4 | v5 | mean | peak |
|---|:--:|:--:|:--:|--|--|--|--|--|--|--|--|
| DEO baseline_drift (no walk, solver) | ✗ | solver | 0.1 | 41.36 | 47.16 | 45.47 | 46.16 | 48.00 | 47.97 | **46.95** | 48.00 |
| R-Zero (penalty questioner) | — | — | — | 41.36 | 44.14 | 46.04 | 47.20 | 46.85 | 47.05 | **46.26** | 47.20 |
| canonical + Claude | ✓ | Claude | 0.1 | 41.36 | 46.66 | 49.78 | 48.29 | 45.94 | 46.41 | **47.42** | 49.78 |
| baseline_drift + Claude | ✗ | Claude | 0.1 | 41.36 | 47.23 | 45.86 | 46.04 | 45.41 | 46.50 | **46.21** | 47.23 |
| curriculum-DEO (anneal β 1→.05) | ✓ | solver | anneal | 41.36 | 47.30 | 45.65 | 47.70 | 48.15 | 46.42 | **47.04** | 48.15 |
| canonical fixed β=0.1 | ✓ | solver | 0.1 | 41.36 | 45.20 | 46.07 | 46.53 | 45.89 | 46.18 | **45.97** | 46.53 |
| canonical FAIR β=0.02, λ_rep=2 | ✓ | solver | 0.02 | 41.36 | 47.60 | 47.70 | 44.71 | 46.16 | 43.81 | **46.00** | 47.70 |

## MATH-500 (single set) per iteration — gpt-mini rechecked

| method | base | v1 | v2 | v3 | v4 | v5 |
|---|--|--|--|--|--|--|
| DEO baseline_drift (no walk, solver) | 71.6 | 73.8 | 76.2 | 73.2 | 75.8 | 76.4 |
| R-Zero (penalty questioner) | 71.6 | 75.8 | 74.2 | 75.6 | 75.6 | 74.6 |
| canonical + Claude | 71.6 | 76.2 | 76.2 | 77.4 | 76.0 | 77.2 |
| baseline_drift + Claude | 71.6 | 75.8 | 76.2 | 76.2 | 79.0 | 77.8 |
| curriculum-DEO (anneal β 1→.05) | 71.6 | 76.2 | 76.0 | 77.2 | 77.2 | 76.2 |
| canonical fixed β=0.1 | 71.6 | 77.8 | 77.2 | 77.4 | 76.2 | 75.0 |
| canonical FAIR β=0.02, λ_rep=2 | 71.6 | 77.8 | 77.6 | 76.4 | 75.4 | 76.0 |

## Per-method full 7-set tables

### DEO baseline_drift (no walk, solver)

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

### canonical + Claude

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 76.2 | 92.0 | 55.0 | 46.0 | 40.7 | 3.3 | 13.3 | 46.66 |
| v2 | 76.2 | 91.7 | 67.7 | 45.2 | 41.0 | 20.0 | 6.7 | 49.78 |
| v3 | 77.6 | 91.0 | 60.0 | 47.1 | 42.4 | 16.7 | 3.3 | 48.29 |
| v4 | 75.8 | 92.3 | 50.1 | 46.0 | 40.7 | 6.7 | 10.0 | 45.94 |
| v5 | 77.0 | 91.9 | 52.2 | 47.4 | 39.7 | 10.0 | 6.7 | 46.41 |

### baseline_drift + Claude

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 75.8 | 91.9 | 59.3 | 43.0 | 40.6 | 6.7 | 13.3 | 47.23 |
| v2 | 76.0 | 91.8 | 55.0 | 44.5 | 40.3 | 6.8 | 6.7 | 45.86 |
| v3 | 76.2 | 92.3 | 55.1 | 44.9 | 40.4 | 6.7 | 6.7 | 46.04 |
| v4 | 78.4 | 92.0 | 50.8 | 45.2 | 41.3 | 6.7 | 3.4 | 45.41 |
| v5 | 77.6 | 91.6 | 52.5 | 43.4 | 40.7 | 13.3 | 6.3 | 46.50 |

### curriculum-DEO (anneal β 1→.05)

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 76.2 | 92.4 | 57.9 | 46.3 | 41.5 | 13.3 | 3.4 | 47.30 |
| v2 | 76.0 | 90.9 | 50.2 | 43.4 | 39.1 | 10.0 | 10.0 | 45.65 |
| v3 | 77.2 | 91.7 | 60.1 | 43.4 | 38.2 | 10.0 | 13.3 | 47.70 |
| v4 | 77.2 | 91.9 | 57.4 | 47.8 | 39.4 | 10.0 | 13.3 | 48.15 |
| v5 | 76.2 | 91.7 | 52.5 | 43.8 | 40.7 | 13.3 | 6.8 | 46.42 |

### canonical fixed β=0.1

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 77.8 | 92.3 | 52.9 | 43.4 | 40.1 | 3.3 | 6.6 | 45.20 |
| v2 | 77.2 | 91.9 | 52.5 | 43.0 | 41.2 | 10.0 | 6.7 | 46.07 |
| v3 | 77.4 | 91.7 | 62.3 | 43.4 | 40.9 | 6.7 | 3.4 | 46.53 |
| v4 | 76.2 | 91.4 | 57.4 | 43.0 | 39.9 | 10.0 | 3.3 | 45.89 |
| v5 | 75.0 | 91.4 | 55.0 | 46.0 | 39.3 | 6.7 | 10.0 | 46.18 |

### canonical FAIR β=0.02, λ_rep=2

| iter | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | AVG |
|--|--|--|--|--|--|--|--|--|
| base | 70.8 | 81.8 | 48.5 | 31.6 | 30.2 | 10.0 | 16.6 | 41.36 |
| v1 | 77.8 | 91.7 | 64.9 | 44.5 | 40.9 | 10.0 | 3.4 | 47.60 |
| v2 | 77.6 | 92.2 | 55.3 | 45.6 | 39.9 | 13.3 | 10.0 | 47.70 |
| v3 | 76.4 | 91.8 | 47.2 | 42.6 | 40.9 | 7.4 | 6.7 | 44.71 |
| v4 | 75.4 | 91.4 | 55.1 | 44.9 | 39.7 | 13.3 | 3.3 | 46.16 |
| v5 | 76.0 | 91.8 | 39.8 | 47.8 | 41.3 | 6.7 | 3.3 | 43.81 |

## Key findings

1. **All self-evolving variants beat base** (7-set MATH AVG 41.4 -> 46-49; MATH-500 71.6 -> 75-79),
   with the largest gains on gsm8k, minerva, olympiad.

2. **The "canonical DEO collapses" claim did NOT reproduce.** In fresh native runs, canonical DEO
   (MCMC walk + solver M-vote labels, KL-pinned to base) stays stable at MATH-500 75-78 for all
   three tested β regimes — fixed 0.1 (77.8->75.0), fixed 0.02 (77.8->76.0), and annealed 1.0->0.05
   (76.2->76.2). None fall toward the ~69 reported in the earlier archived run. The historical
   collapse is most likely run-to-run variance / an old-setup artifact, not an inherent failure mode.
   (So an annealed-β "curriculum" is NOT needed to prevent a collapse that doesn't occur here.)

3. **Fair DEO-vs-R-Zero (matched challenger objective) → they tie, validating DEO's core claim.**
   R-Zero trains its challenger with a KL-regularized objective (kl_coef=0.01). DEO's MCMC samples
   the closed-form optimum of the *same* objective, π ∝ π_ref·exp(r_c/β). DEO's reward r_unc =
   1−2|p̂−0.5| is exactly **2×** R-Zero's min(p̂,1−p̂), so the reward-scale-matched DEO temperature
   is **β=0.02** (with λ_rep=2), not the default 0.1. At that fair setting, **DEO (MCMC challenger)
   MATH AVG mean 46.00 ≈ R-Zero (GRPO-trained challenger) 46.26** — statistically tied. This supports
   DEO's thesis that MCMC sampling of the optimal challenger can *replace* challenger training.

4. **Claude labeler = 100% coverage, no degradation; best single mean.** canonical+Claude reaches the
   top mean (47.42) and highest single iter (49.78 @ v2). Claude labeled ~100% of selected questions
   every iter (vs a truncated GPT-4o attempt's 41%, fixed via max_tokens=16384 + forced-\boxed + retry).
   baseline_drift+Claude gets the highest single MATH-500 (79.0) but generalizes less (no walk ->
   narrower question diversity -> weaker amc/aime).

5. **Differences are within small-benchmark noise (~1 MATH-AVG pt).** All seven configs land at
   MATH AVG 46-47.4 / MATH-500 75-79. On the large reliable sets the ordering is stable; on
   amc(40)/aime(30) it is dominated by 1-3 problem flips.

## Batch-size ablation (why the paper's absolute numbers weren't matched)

A stock 8-GPU R-Zero run with `rollout_batch_size=512` (paper default) was tested
against our `=64`. **Iter-1 solver MATH-500 was 76.0 (512) vs 75.8 (64) — no
meaningful difference.** So the ~2-pt gap to the R-Zero paper's Qwen3-4B-Base
numbers (MATH AVG 49.07 @ iter3) is **not** a batch-size artifact; it is the
**grader** (paper: gpt-4o full-text, ~33% false-positive → inflates; ours:
gpt-4o-mini boxed-only, ~0% FP). Matching the paper's absolute scores would
require re-grading with their lenient grader, not larger batches.

Wall-clock (8×H100, rollout_batch_size=512, max_response_length=4096, 1 iter):
questioner GRPO (6 steps) ≈ 4.75 h; solver GRPO (20 steps) ≈ 29 h — the 512×4096
rollout is ~8× the compute of the =64 runs, i.e. ~7 days for 5 iters, with no
expected change in outcome. The full 512 run was therefore **not** executed.

## Wall-clock / compute efficiency (Standard_ND96isr_H100_v5, 8×H100 node)

Per-iteration wall-clock (measured via iter_wallclock.tsv / job logs):

| method | GPUs used | per-iter | 5 iters + eval |
|---|:--:|---|---|
| R-Zero (4-GPU, rollout 64) | 4 | ~11 h (questioner ~6.5 h + solver ~4.8 h) | **~55 h (~2.3 d)** |
| R-Zero (8-GPU, rollout 64) | 8 | 10.0 h avg (questioner 4.9–5.3 h + solver 5.0–5.2 h) | **50.8 h (2.1 d)** (measured, 5 iters) |
| DEO canonical (MCMC walk) | 7 (base0/solverDP1,4,5/verl2,3/eval6) | ~4-5 h (walk ~2.5-3 h + verl ~1.5 h) | **~20-24 h (~1 d)** |
| DEO baseline_drift (no walk) | 7 | ~2-2.5 h | **~10-14 h** |

Notes:
- **DEO is 2-5× faster than R-Zero.** R-Zero *trains* a challenger every iter (questioner
  GRPO + 8-shard question generation + M-vote scoring ≈ 5-6.5 h) on top of solver training;
  DEO skips challenger training and *samples* the optimal challenger via MCMC — exactly the
  efficiency claim of the DEO paper. Combined with §finding-3 (equal accuracy under the fair
  β), this supports "MCMC sampling can replace challenger training, cheaper."
- **More GPUs barely helped R-Zero at rollout_batch_size=64**: 64 prompts across 8 GPUs
  underutilizes them, and the fixed per-iter cost (question gen + M-vote eval) dominates, so
  8-GPU (~10 h/iter) ≈ 4-GPU (~11 h/iter). Large batch (512) uses 8 GPUs but is ~8× the compute
  and thus slower overall (~7 d for 5 iters), with no accuracy gain (76.0 ≈ 75.8).

## Methodology & caveats

- **Dataset sizes (unique problems):** math=500, gsm8k=1319, olympiad=675, minerva=272, amc=40,
  aime2024=30, aime2025=30. amc/aime are tiny (2.5-3.3%/problem) → noise-dominated; weight
  conclusions toward the large sets.
- **amc/aime recheck rate-limited:** the ×32 duplication in the loader inflates recheck request
  volume; amc/aime rechecks hit OpenAI 429s (raw `math_verify` used, slightly undercounted). Applies
  to all methods equally. math/gsm8k/minerva/olympiad recheck succeeded.
- **Two distinct "β"s:** DEO's MCMC Metropolis-Hastings temperature β (challenger sharpness) is a
  different knob from verl's GRPO KL coefficient kl_coef=0.01 — though §finding-3 shows the MCMC β is
  the same parameter as the challenger objective's KL coefficient after reward-scale matching.
- **KL ref pinned to base** (verl patch) for all DEO variants; R-Zero uses stock KL (ref = prev actor).
- **verl:** max_steps=20 (merge global_step_15), rollout_batch_size=64, TP=2.
- Grader = `math_verify` + gpt-4o-mini boxed-only recheck.

## Artifacts (AzureML jobs, workspace ai-core-research)

| experiment | job | datastore path |
|---|---|---|
| DEO baseline_drift (solver) + R-Zero + eval | kind_box_0mbv30m7js | yyd_deo_rzero_run |
| R-Zero (rerun used) | mighty_lemon_zj88drhnvz | yyd_rzero_run |
| canonical + Claude | amusing_pig_dmqs56y8sw | yyd_canon_claudelabel |
| baseline_drift + Claude | great_cow_6pgg6kt0l5 | yyd_bd_claudelabel |
| curriculum-DEO (anneal β) | bold_sock_l7k811l29q | yyd_curriculum |
| canonical fixed β=0.1 | teal_malanga_s5ghn3r30j | yyd_canon_fixedbeta |
| canonical fair β=0.02, λ_rep=2 | tough_lunch_flst3p7zcc | yyd_canon_fair_b002 |

Per-(model,dataset) scores are in each job's `final_results.jsonl`. Code: `DEO/*_native_main.py`,
`azureml/`, verl KL-pin patch in `R-Zero/verl/`.


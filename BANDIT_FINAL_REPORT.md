# Bandit-Memory MCMC (deo_with_memory.pdf §1.4): final report

**Verdict: no gain.** A contextual Thompson-sampling memory over the five mutation operators
(A GENERALIZE / B COMPOSE / C INVERT / D CHANGE_OBJECTIVE / E DUALIZE) was implemented per the
paper, reviewed externally, run twice at 4B with fixes between runs, and closed: the best bandit
arm finishes **1.1 AVG7 below** the uniform-operator control, and the learned posteriors stay flat
through all 10 iterations of both runs — the five operators are statistically indistinguishable
in usefulness on this distribution.

## What was built

- `DEO/mutation_bandit.py`: Beta–Bernoulli posterior per (context, action); context =
  (seed topic × difficulty bucket Hard/Trainable/Easy of p̂); Thompson-sampling selection (Eq 14);
  success = Valid × in-band[0.3,0.8] × Novel(BLEU cluster ≤ 1) (Eq 16), plus an
  r_unc-non-degradation condition (ε=0.1) for already-Trainable states; frozen within an outer
  iteration, discounted update α←1+ρ(α−1)+S with ρ=0.9 afterwards (Eq 17); JSON persistence with
  per-run state files and per-iteration snapshots. MH acceptance untouched (bandit proposes,
  MCMC disposes). Unit-tested: TS converges on synthetic arms, unseen contexts stay explorative,
  discount exact, save/load roundtrip.
- External (GPT) review applied before running: operator-compliance attribution, pre-MH
  surface-validity gate (local-vLLM judge), per-run state isolation, Trainable-context reward
  tightening, seed-topic semantics, approximate-MH wording.

Common config of both runs: Qwen3-4B-Base, MODE=curriculum, fixed β=0.1×5, walk on, 2000-q pool,
no CD, KL→base. Control arm (identical, no bandit): `icy_sprout` (mutV1) — AVG7 48.79 @i5.

## Run 1 — strict compliance (`olden_boniato`, run_id bandit_fb_4b_r1)

Bandit picks operator a; the model must emit `<strategy>a</strategy>`; a mismatch = bandit
failure + proposal dropped pre-scoring.

- **Mismatch ≈ 62% flat across all 5 iters** (6115–6291/10k): Qwen3-4B-Base ignores the assigned
  operator most of the time. With the 15% surface-invalid rate, only ~2,200/10,000 proposals per
  iter reached MH — the chain ran at ~1/4 throughput.
- Successes 4–5%/iter; **action means flat (0.03–0.05)** — the reward was dominated by
  compliance noise (a model property, uniform across arms), so TS had nothing to learn.
- MATH-500: 71.6 → 74.8/76.6/76.0/76.6/**77.8** (peak i5). Slow monotone climb; ends above the
  uniform arm's i5 (74.2) but below its peak (78.2) — "fewer but pre-filtered proposals degrade
  the pool more slowly". 7-set not graded (superseded by r2).

## Redesign between runs (user/GPT feedback)

Single-operator prompt: since the bandit already chose the operator, the prompt describes ONLY
that operator (no A–E menu to deviate to), and a non-matching `<strategy>` tag is audit-only —
outcomes are credited to the chosen action (whose prompt generated the proposal), never dropped.

## Run 2 — single-operator prompt (`mango_lion`, run_id bandit_fb_4b_r2, fresh Beta(1,1))

- Throughput restored (mismatch now audit-only; note the tag still mis-echoes ~57% — the 4B base
  cannot even reliably copy a letter, validating the r1 diagnosis).
- Successes rose to ~6%/iter; **action means still flat** (A .06 B .06 C .07 D .07 E .05 —
  spread ≈ noise) through all 5 iters.
- MATH-500: 71.6 → 76.6/77.4/74.0/75.2/76.0 (peak 77.4 @i2).

## Final accuracy (Claude Haiku boxed, same grader)

| arm | AVG7 per iter | peak | HARD5 pk | COMP4 pk |
|--|--|--|--|--|
| **uniform (icy_sprout)** | 45.1 47.7 47.6 46.1 **48.79** | **48.79** @i5 | **34.94** | **31.62** |
| bandit r2 (mango_lion) | 45.2 47.0 45.9 47.2 **47.73** | 47.73 @i5 | 33.12 | 29.90 |

MATH-500 (GPT-bumped, in-run): uniform peak 78.2 / mean 76.2; r1 77.8 / 76.4; r2 77.4 / 75.8 —
statistically tied. The 7-set, however, has the bandit arm ~1.1 below uniform.

## Why it failed

1. **The arms are not different.** Across 10 iterations and ~100k proposals, no operator showed a
   success-rate edge beyond noise in any context. A bandit can only ever converge to the best arm;
   when arms are equal its ceiling is uniform-minus-overhead.
2. **Reward sparsity.** Even conditional on reaching scoring, success ≈ 18% (r1) / ~6% overall
   (r2 counts pre-gate failures); with 225 (context × action) cells and ρ=0.9 forgetting, per-cell
   evidence never accumulated past the prior.
3. **Executor unreliability.** The 4B base model neither follows operator assignments (62%) nor
   echoes tags (57%) — any operator-level control scheme at this scale is steering by suggestion.
4. **Overhead is real.** The pre-MH judge gate + audit machinery cost latency and dropped ~15-22%
   of proposals; with no learning gain this is pure tax (the −1.1 AVG7 plausibly reflects it plus
   run-to-run noise).

## What would make it worth revisiting

- A mutator that actually follows instructions (8B+/instruct model, or the current solver) — the
  compliance floor disappears and K_a becomes a real conditional kernel.
- Operators with genuine heterogeneity (finer-grained or domain-specific transformations; the
  current five all produce statistically similar in-band/novelty outcomes).
- Denser reward (e.g. continuous Δr_unc instead of the triple-gated binary success).

## Artifacts

- Runs: olden_boniato_7vw7zs989y (r1), mango_lion_cprrm4f7ld (r2); grade tidy_shoe_khqfxn7xcg.
- Per-iteration posterior snapshots: `DEO/datasets/bandit_state_bandit_fb_4b_r{1,2}_iter{1..5}.json`
  in the runs' outputs (`yyd_bandit_fb_4b`, `yyd_bandit_fb_4b_r2`).
- Chronology + full problem analysis: `BANDIT_RUN_ISSUES.md`; review: `bandit_mcmc_review_for_claude.md`.
- Code: `DEO/mutation_bandit.py`, integration in `DEO/mcmc_deo_vllm.py` (env `DEO_BANDIT=1`, off by default).

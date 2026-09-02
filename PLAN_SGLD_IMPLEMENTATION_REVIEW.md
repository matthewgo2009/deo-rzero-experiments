# Planning-SGLD Implementation Review

Reviewed commit: `7e58f29` on `curriculum-deo-claude-experiments`.

Files reviewed:

- `DEO/plan_sgld.py`
- `DEO/plan_sgld_native_main.py`
- `DEO/plan_sgld_tests.py`

## Important: do not change the uncertainty reward

The TENT uncertainty reward currently used by the project is intentional:

\[
r_{\mathrm{unc}}(x)=\max\left(0,1-2\left|\widehat p_{\mathrm{maj}}(x)-\tfrac12\right|\right).
\]

Do **not** replace it with `1 - p_hat`. The changes below concern credit assignment,
length control, validity feedback, particle selection, and testing.

## What is already correct

The core implementation in `plan_sgld.py` correctly includes:

- continuous logits `A` with shape `(M, K, L)`;
- hard row-wise categorical plans `Z`;
- the reward score term `((U - b) / gamma) * (Z - P)`;
- plan-prior KL and logit L2 regularization;
- Langevin noise `sqrt(2 * eta * T) * epsilon`;
- row centering and final row-wise argmax quantization;
- no gradient through the generator, generated text, solver, or reward.

The main remaining problems are in the integration between the planner and the
batch-level reward.

---

## P0. Use the full pooled reward for every particle gradient

### Current behavior

`probe_and_score()` computes repetition penalties using questions from all
particles, but returns a separate local utility `U[p]`. The runner then calls:

```python
planner.step(p, plans[p], Ps[p], U[p], ...)
```

This is not the score-function gradient of the pooled batch objective when the
repetition reward couples particles.

Let

\[
R(\mathbf Z)=R(Z^{(1)},\ldots,Z^{(M)})
\]

be the reward of the full pooled batch. The correct gradient is

\[
\nabla_{A^{(p)}}\mathbb E[R(\mathbf Z)]
=
\mathbb E\left[
  (R(\mathbf Z)-b_p)
  \nabla_{A^{(p)}}\log q_{A^{(p)}}(Z^{(p)})
\right].
\]

Using only `U[p]` omits the effect of particle `p` on the repetition penalties
of questions owned by other particles. The current estimator is valid only when
the reward is separable across particles, for example when `lambda_rep == 0`.

### Required change

Have `probe_and_score()` return one shared `U_global` for optimization, while
keeping per-particle utilities only as diagnostics:

```python
U_global = sum(all_question_utilities) / (PLAN_M * PLAN_B)

for p in range(PLAN_M):
    planner.step(p, plans[p], Ps[p], U_global, s, PLAN_S, generator=g)
```

All particles may keep separate history-dependent baselines, but the sampled
reward in the score term must be the same full-batch reward.

For lower variance, a counterfactual difference reward can be added later, but
the shared global reward is the correct first implementation.

### Repetition denominator

The repetition fraction should use the fixed batch size `PLAN_M * PLAN_B`, not
`len(valid_questions)`. Invalid generations should contribute zero utility but
should not silently change the normalization of the reward. Also handle the
single-valid-question case consistently: its cluster includes itself.

---

## P0. Make the length axis identifiable and enforce it in the final pool

### Current behavior

The probe gate accepts a generated length when

```python
0.5 * lo <= n_tokens <= 2.0 * hi
```

This produces the following effective intervals:

| Selected bin | Current accepted interval |
|---|---:|
| 80--150 | 40--300 |
| 150--250 | 75--500 |
| 250--400 | 125--800 |
| 400--600 | 200--1200 |

A 200-token question is accepted by every bin. Consequently, the length row
receives almost no discriminative learning signal.

The final training-pool generation does not check the selected length at all.
Even if a particle quantizes to `400--600`, its final training questions may
still all be approximately 200 tokens long.

### Required change

Create one shared helper and use it in both probing and final generation:

```python
def plan_length_ok(tokenizer, question, rows, tolerance=0.10):
    lo, hi = LENGTH_BINS[rows[LENGTH_AXIS]]
    n = len(tokenizer(question, add_special_tokens=False).input_ids)
    return (1.0 - tolerance) * lo <= n <= (1.0 + tolerance) * hi
```

- A probe outside the selected interval receives zero utility.
- A final-pool question outside the interval is rejected and resampled using a
  bounded resampling budget.
- Do not directly reward large length; only reward adherence to the selected
  categorical plan, so padding is not intrinsically beneficial.

### Required diagnostics

For every planning step and final hard plan, log:

- mean and standard deviation of observed question length;
- selected length bin;
- fraction inside the selected bin;
- final training-set length histogram.

The length curriculum should not be considered implemented until these metrics
show that different length categories induce measurably different question
distributions.

---

## P0. Feed validity and usability failures back into the planner

### Current behavior

During planning, a question is considered usable mainly when it can be parsed
and does not contain a forbidden phrase. `probe_and_score()` ignores the
returned pseudo-label, and the LLM validity judge is only run later inside
`filter_and_push()`.

This allows Planning-SGLD to reward ambiguous, inconsistent, or unsolvable
questions: such questions often have high solver uncertainty, so filtering
them only after optimization does not prevent planner reward hacking.

### Required change

Preserve the intentional TENT uncertainty reward, but gate it by usability:

\[
u_i
=
\mathbf 1\{\text{valid}_i\}
\mathbf 1\{\text{length-adherent}_i\}
\mathbf 1\{\text{usable pseudo-label}_i\}
\left[r_{\mathrm{unc},i}-\lambda_{\mathrm{rep}}r_{\mathrm{rep},i}\right]_+.
\]

At minimum, assign zero utility when:

- `pseudo_label is None`;
- the question is outside the configured trainable `p_hat` band;
- the selected length category is violated;
- the validity judge rejects the question.

If judging all probes is too expensive, use a cheap validity gate for every
probe and judge a cached or subsampled fraction. However, some validity signal
must be present during planning rather than only after the planner has finished.

---

## P1. Do not allocate final questions equally to unvalidated particles

### Current behavior

After quantization, every particle receives exactly `TOTAL_QUESTIONS / M`
generation slots. The directional unit test only requires one particle to find
the target plan. These two choices are inconsistent: if one of eight particles
is good, the current runner can still generate 87.5% of the dataset from worse
particles.

### Required change

After quantization:

1. Deduplicate identical hard plans.
2. Evaluate every unique hard plan with a fresh validation probe batch.
3. Drop or reinitialize clearly bad plans.
4. Allocate final questions using either top-plan selection or utility weights,
   for example

   \[
   w_p\propto\exp(\widehat U_p/\tau_{\mathrm{mix}}).
   \]

If diversity is desired, select a set of high-utility plans greedily with a
plan-level or question-level repulsion term instead of giving every particle an
equal quota unconditionally.

---

## P1. Strengthen the tests

The current tests do not validate the most important integration behavior.

### Problems

- `test_score_identity()` manually reconstructs the known estimator but does
  not call `PlanSGLD.grad()`. It can pass even if the implementation of
  `PlanSGLD.grad()` is wrong.
- `test_directional()` uses 1,500 planning steps, while production uses only
  30. It therefore does not validate the production schedule.
- There is no test for cross-particle reward coupling.
- There is no test that the final pool respects the quantized length plan.
- There is no test for invalid-question feedback or duplicate hard plans.

### Required tests

1. **Implemented-gradient test:** compare the expectation of
   `PlanSGLD.grad()` against exact enumeration/autograd, including KL and L2.
2. **Two-particle coupled-reward test:** define a reward that penalizes equal
   categorical choices; compare the shared-reward score estimator with exact
   enumeration over both particles. This test should fail under the current
   local-`U[p]` implementation.
3. **Production-horizon test:** run the actual 30-step defaults across many
   seeds and report regret, recovery rate, row entropy, and quantized-plan
   accuracy. Do not rely on a single 1,500-step success.
4. **Length integration test:** mock the generator with known output lengths
   and verify both probe rewards and final-pool rejection/resampling.
5. **Validity integration test:** confirm that malformed or judge-rejected
   questions receive zero planning reward.
6. **Particle-mixture test:** confirm deduplication and that low-utility
   particles do not receive equal final allocation.

---

## P1. Add budget and convergence diagnostics

The default planning budget per solver iteration is

\[
MBS=8\times25\times30=6000
\]

probe questions. With nine solver responses per probe, that is 54,000 solver
completions per outer iteration before final-pool scoring.

This may still be worthwhile because it removes challenger-model training and
its memory cost, but it must be measured explicitly. Log:

- total base-model generations and solver rollouts;
- wall-clock time per planning step and per outer iteration;
- reward mean/variance and baseline values;
- per-row entropy of `P(A; gamma)`;
- maximum valid-logit magnitude;
- number of unique sampled and quantized plans;
- improvement over the initial plan distribution.

Consider a smaller probe batch early in annealing and a larger batch near
quantization, rather than using `B=25` at every step.

---

## P2. Confirm remaining objective conventions

These are not requests to change the intentional TENT uncertainty reward, but
the code and paper should explicitly agree on them:

1. The code uses a positive-part utility
   `[r_unc - lambda_rep * r_rep]_+`; the current Planning-SGLD tex should show
   the same positive part if it is intentional.
2. The plan prior `rho0` is uniform. This is a design prior, not an estimate of
   the base model's natural distribution over semantic plans. Avoid describing
   it as an exact transformation of the base-model-anchored question
   distribution.
3. Some planning axes are not orthogonal. For example, `counting problem` is
   naturally compatible with combinatorics but may conflict with geometry.
   Either add compatibility masks or replace these values with attributes that
   are meaningful across every topic.
4. A warm-started particle with saturated logits may not be meaningfully
   refreshed by noise of scale `0.05`. Monitor entropy and either contract old
   logits, increase refresh noise adaptively, or reinitialize a subset of weak
   particles after each solver update.

---

## Minimum acceptance checklist before a full run

- [ ] TENT uncertainty reward remains unchanged.
- [ ] All particle gradients use the full pooled batch reward.
- [ ] Repetition normalization uses a fixed batch convention.
- [ ] Invalid, unusable, or length-violating probes receive zero reward.
- [ ] The same length constraint is enforced in final-pool generation.
- [ ] Quantized plans are freshly evaluated, deduplicated, and not allocated an
      equal quota regardless of quality.
- [ ] A coupled-particle exact-enumeration test passes.
- [ ] A 30-step multi-seed production-config test shows directional improvement.
- [ ] Length-bin adherence and planner entropy are logged.
- [ ] Planning-generation and solver-rollout costs are reported.


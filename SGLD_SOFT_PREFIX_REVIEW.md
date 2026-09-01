# SGLD Soft-Prefix Implementation Review and Required Fixes

## Scope

Review target:

- Repository: `matthewgo2009/deo-rzero-experiments`
- Branch: `curriculum-deo-claude-experiments`
- Commit: `bc33109eb1d332b0e5d281f4ccaaffe45a153b60`
- Main files:
  - `DEO/sgld_soft_prefix.py`
  - `DEO/sgld_deo_native_main.py`
  - `DEO/sgld_smoke.py`
  - `azureml/run_pipeline_job.sh`

The high-level implementation is correct: freeze the base model, add a continuous
soft-prefix perturbation, estimate a score-function gradient by teacher forcing,
and update the latent with

\[
z^{k+1}=z^k+\eta\left(-\frac{z^k}{\sigma^2}
+\frac{\widehat g}{\tau}\right)+\sqrt{2\eta}\,\epsilon.
\]

However, the current code does **not** sample from the latent Gibbs target claimed
in the paper. Do not launch the full five-iteration experiment until the P0 items
below are fixed.

## What Is Already Correct

- Base-model parameters are frozen.
- Gradients flow only to the latent soft prefix `z`.
- The teacher-forcing shift `logits[K - 1 + t] -> x[t]` is correct.
- The Gaussian prior score `-z / sigma**2` is correct.
- The Langevin noise scale `sqrt(2 * eta)` is correct for the chosen convention.
- The running baseline is used before being updated, so it is independent of the
  current sampled action.
- Saving and warm-starting `Z` across outer iterations is reasonable.
- All three new Python files pass static Python compilation.

## P0: Proposal and Score Distributions Do Not Match

### Current behavior

`generate()` samples using:

```python
do_sample=True
temperature=1.0
top_p=0.95
```

It does not explicitly disable `top_k`. Hugging Face generation defaults can apply
top-k truncation as well.

`score_grads()` then computes the score under the untruncated full softmax:

\[
\nabla_z\log \pi_0(x\mid p_0,z).
\]

Consequently, the implementation currently uses

\[
x\sim q_{\mathrm{truncated}}(\cdot\mid z),
\qquad
\widehat g\propto\nabla_z\log\pi_0(x\mid z),
\]

which is not a valid score-function estimator for either distribution.

### Required fix

For the first correct implementation, sample from the raw frozen model policy:

```python
gen = model.generate(
    ...,
    do_sample=True,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    # Also ensure no other logits warpers are active.
)
```

Alternatively, one could compute the normalized log probability of the exact
top-p/top-k proposal, but that is substantially more complicated and is not
recommended for the first experiment.

### Acceptance test

- Log the effective `model.generation_config`.
- Assert that `temperature == 1`, `top_p == 1`, `top_k == 0`, and that no other
  logits processor changes the distribution.
- Add a small test showing that the distribution used by `generate()` is the same
  distribution whose log probability is used by `score_grads()`.

## P0: Score the Exact Sampled Token Trajectory

### Current behavior

`generate()` decodes the generated token IDs into text with
`skip_special_tokens=True`. `score_grads()` then tokenizes that text again and
truncates it.

There are several resulting biases:

1. `encode(decode(ids))` is not guaranteed to recover the same token IDs.
2. EOS and other special tokens are removed.
3. The main runner generates up to 1536 tokens but scores only the first 1024.
4. The reward is computed from the full decoded completion while the score is
   computed from a truncated, retokenized completion.

The gradient is therefore not

\[
\nabla_z\log\pi_0(x_{\mathrm{sampled}}\mid p_0,z).
\]

### Required fix

Change `generate()` to return both:

```python
GeneratedSample(
    text=decoded_text,
    token_ids=exact_generated_ids,
    attention_mask=exact_generated_mask,
)
```

Then make `score_grads()` consume `token_ids` directly. Do not decode and
retokenize for scoring. Include the sampled EOS token when generation terminates
with EOS. If generation terminates because of `max_new_tokens`, score exactly the
sampled non-EOS trajectory.

Also use one consistent maximum length:

```python
max_new_tokens=SGLD_MAXTOK
```

instead of hard-coding 1536 in the runner and using 1024 in the scorer.

### Acceptance tests

- Assert exact equality between generated IDs and scored IDs.
- Test both EOS-terminated and max-length-terminated samples.
- Test a mixed-length padded batch and verify that padding contributes zero log
  probability.
- Compare batched scores against scores computed one sequence at a time.

## P0: Align the Implemented Reward with the Intended Objective

### Reward mismatch

The current TeX objective defines finite-rollout disagreement as

\[
\widehat r_{\mathrm{dis}}(x,\theta)
=\frac1m\sum_{j=1}^m
\mathbf 1\{y_j\neq\operatorname{Maj}(\mathbf y)\}
=1-\widehat p.
\]

The implementation instead reuses the R-Zero tent reward

\[
r_{\mathrm{unc}}=1-2\left|\widehat p-\frac12\right|
\]

and then clips the combined utility:

\[
u_i=\max(0,r_{\mathrm{unc},i}-\lambda r_{\mathrm{rep},i}).
\]

These are different objectives. For `p_hat >= 0.5`, the tent reward is twice the
disagreement reward, which can partly be absorbed into the temperature. For
`p_hat < 0.5`, they are qualitatively different. The outer `max(0, ...)` is an
additional objective change.

### Required decision

Choose one objective explicitly and make the paper, code, logging, and experiment
names agree.

Recommended default for this implementation task: implement the current TeX
objective exactly:

```python
r_dis = 1.0 - p_hat
u_i = r_dis - lambda_rep * r_rep_i
```

Do not apply `max(0, ...)` unless the TeX objective is changed to include it.

If retaining the R-Zero tent reward is intentional, document this as a different
surrogate objective and do not claim that the code implements the TeX objective.

## P0/P1: Per-Question Credit Is Exact Only for the Separable Reward

Claude's local-credit change is mathematically correct for the separable
uncertainty term. If

\[
R(X)=\frac1n\sum_{j=1}^n r(x_j),
\qquad \tau=\frac\beta n,
\]

then

\[
\frac1\tau\nabla_{z_i}\mathbb E[R(X)]
=\frac1\beta\mathbb E\left[
r(x_i)\nabla_{z_i}\log\pi_0(x_i\mid z_i)
\right].
\]

Therefore, using the local reward and setting the implementation temperature to
`beta` is exact for the separable term and has lower variance than multiplying
every score by the batch mean.

It is **not exact for diversity/repetition**, because changing `x_i` also changes
the repetition rewards assigned to other questions:

\[
\nabla_{z_i}\mathbb E[r_{\mathrm{rep}}(X)]
=\mathbb E\left[
r_{\mathrm{rep}}(X)
\nabla_{z_i}\log\pi_0(x_i\mid z_i)
\right].
\]

Using only `r_rep_i` omits the effect of `x_i` on `r_rep_j` for `j != i`.
Clipping each local utility makes this cross-credit even less reducible to a
simple local expression.

### Required fix

Use one of the following clearly documented options:

1. **Exact first baseline:** set `lambda_rep = 0`. Then local credit with
   `SGLD_TAU = beta` is correct.
2. **Exact batch reward:** multiply each score by a properly estimated full-batch
   reward and use the matching batch temperature.
3. **Derived pairwise estimator:** rewrite repetition as an explicit symmetric
   pairwise objective and derive an unbiased coordinate-gradient estimator,
   including both endpoints' contributions.
4. **Local surrogate:** keep the current local penalty but state explicitly that
   it optimizes a new surrogate, not the paper's batch objective.

For the first correctness experiment, option 1 is strongly recommended. Add the
repetition term only after the uncertainty-only sampler passes gradient and
distribution tests.

### `SGLD_GLOBAL_REWARD` scaling bug

`SGLD_GLOBAL_REWARD=1` does not currently restore the paper-literal update.
The code computes a minibatch mean over `b` selected questions but keeps
`tau = beta`. For a minibatch-mean coefficient, the matching scale is
`tau = beta / b`; for a true full-batch mean, it is `tau = beta / n`.

Either remove this option or make the reward scale and temperature adjustment
automatic and explicit.

## P1: Scale the Soft-Prefix Perturbation Separately from the Prior

The current code samples

```python
z ~ Normal(0, 1)
inputs_embeds = prompt_emb + z
```

This gives an RMS perturbation of approximately 1 per embedding component. It is
not guaranteed to be close to the original prompt-embedding scale and is likely
to dominate the base prompt.

Use a dimensionless standard-normal latent and a separate mapping scale:

\[
z\sim\mathcal N(0,I),
\qquad
E_{\mathrm{input}}=E(p_0)+\alpha z.
\]

The prior score remains `-z`, while `alpha` controls how strongly the latent can
change the model input.

### Required diagnostics

Log at initialization and after every SGLD step:

```text
prompt_embedding_rms
latent_rms
delta_embedding_rms
delta_to_prompt_rms_ratio
valid_parse_rate
mean_sequence_log_prob
score_gradient_norm
drift_norm
noise_norm
```

Start with a small `alpha` chosen relative to the observed prompt-embedding RMS,
not with a fixed perturbation RMS of 1.

## P1: Ten Minibatch Updates Are Not Ten SGLD Sweeps

With `n = 2000`, minibatch size `b = 500`, and ten steps, each latent is updated
only 2.5 times on average. The probability that a latent is never selected is

\[
\left(1-\frac{500}{2000}\right)^{10}\approx 5.6\%.
\]

Moreover, repetition is computed only among valid questions in the current
minibatch, rather than on the declared full batch.

### Required fix

Define one SGLD sweep as:

1. Randomly permute all `n` latent indices.
2. Partition them into minibatches.
3. Update every latent exactly once.

Then interpret `SGLD_STEPS` as the number of complete sweeps. If repetition is
enabled, use a global memory/state or a justified stochastic estimator of the
full-batch repetition objective.

## P1: Evaluation Runner Uses the Wrong Experiment Name

The shell dispatch currently contains:

```bash
sgld) run_sgld; run_eval_canon ;;
```

`run_eval_canon` defaults to `deo_canon_claudelabel`, while the SGLD runner saves
checkpoints under `deo_sgld`. The post-training evaluation will therefore look for
the wrong checkpoints unless `DEO_ABBR` happens to be externally set.

### Required fix

Either add a dedicated function:

```bash
run_eval_sgld() {
  DEO_ABBR=${DEO_ABBR:-deo_sgld} run_eval_canon
}
```

or refactor the evaluation function to take the experiment abbreviation as an
explicit argument:

```bash
sgld) run_sgld; run_eval_deo "${DEO_ABBR:-deo_sgld}" ;;
```

Add a dry-run test that prints every checkpoint path before launching evaluation
and asserts that it exists.

## P2: Memory and Runtime Risk

The latent tensor requires

\[
4nKd\ \text{bytes}
\]

in FP32. For `n = 2000`, `K = 200`, and Qwen3-4B hidden size `d = 2560`, `Z`
alone is approximately 3.8 GiB. Gradients for a 500-question minibatch add about
1 GiB, excluding temporary indexing copies. Backpropagation also materializes
large sequence activations and full-vocabulary logits.

Recommended follow-up after correctness is established:

\[
z_i\in\mathbb R^r,
\qquad
\Delta E_i=A z_i,
\qquad
r\ll Kd,
\]

where `A` is a fixed low-rank projection. This preserves the no-trained-challenger
property while substantially reducing memory and score variance.

Also avoid calling `torch.cuda.empty_cache()` after every scoring microbatch unless
profiling shows it is required; frequent cache clearing can be very expensive.

## Required Tests Before a Full Experiment

### 1. Exact-token score test

- Generate token IDs.
- Score those exact IDs.
- Assert no decode/encode round trip and no unexpected truncation.
- Compare batched and single-sequence log probabilities and gradients.

### 2. Finite-difference / exact-enumeration score-function test

Use a tiny toy categorical autoregressive model for which all short sequences can
be enumerated. Verify

\[
\nabla_z\mathbb E_{x\sim\pi(\cdot\mid z)}[R(x)]
=\mathbb E[R(x)\nabla_z\log\pi(x\mid z)]
\]

against the exact gradient or a finite-difference calculation.

The existing smoke test only checks that gradients are nonzero. A nonzero gradient
does not verify that it is the score of the sampling distribution.

### 3. Prior-only SGLD test

Set all rewards to a constant. After sufficient steps, each latent coordinate
should empirically have mean approximately zero and variance approximately
`sigma**2` under a sufficiently small step size.

### 4. Synthetic directional-reward test

Use a cheap reward such as the occurrence of one chosen token. Verify that SGLD
increases its empirical probability relative to the base while the KL/latent norm
remains controlled. Repeat across several random seeds and report confidence
intervals.

### 5. End-to-end small-pool test

Before using `n = 2000`:

```text
n = 16 or 32
lambda_rep = 0
top_p = 1
top_k = 0
exact generated token IDs
one or more complete latent sweeps
```

Confirm:

- high valid parse rate at initialization;
- no NaN or Inf gradients;
- stable latent RMS;
- reward improves over sweeps;
- solver reward uses the current solver checkpoint;
- final evaluation resolves the `deo_sgld` checkpoint path.

## Recommended Implementation Order

1. Make generation and teacher-forced scoring use the same untruncated policy.
2. Preserve and score exact generated token IDs, including EOS.
3. Remove length mismatch and hard-coded generation length.
4. Run the exact uncertainty-only objective with `lambda_rep = 0` and local
   credit at `tau = beta`.
5. Introduce a separate soft-prefix scale `alpha` and tune it using embedding-RMS
   diagnostics.
6. Replace random partial updates with complete shuffled sweeps.
7. Fix the `MODE=sgld` evaluation checkpoint name.
8. Add the exact-enumeration, prior-only, and small-pool tests.
9. Only then derive and add an unbiased batch-diversity gradient estimator.
10. After correctness, optimize memory with a low-dimensional latent projection.

## Definition of Done

The implementation is ready for a full experiment only when:

- sampled and scored token distributions are identical;
- exact sampled token IDs, including stopping behavior, are scored;
- the code and TeX use the same uncertainty and repetition objective;
- the temperature matches the reward normalization;
- diversity credit is either correct or explicitly disabled;
- one reported SGLD sweep updates every latent once;
- prior-only and exact-enumeration tests pass;
- the small-pool run improves the synthetic reward without latent explosion;
- the Azure runner evaluates the actual `deo_sgld` checkpoints.

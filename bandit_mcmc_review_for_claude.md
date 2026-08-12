# Bandit-MCMC Implementation Review: Issues and Required Fixes

Target files:

- `DEO/mutation_bandit.py`
- `DEO/mcmc_deo_vllm.py`

This note reviews the current contextual Thompson-sampling mutation-bandit integration. The high-level architecture is correct:

\[
x \rightarrow c(x) \rightarrow a \sim \pi_{\text{bandit}}(a|c)
\rightarrow x' \sim K_a(\cdot|x)
\rightarrow \text{MCMC accept/reject}.
\]

`MutationBandit` itself is largely correct: it stores a Beta posterior per `(context, action)`, samples actions using Thompson sampling, buffers observations during an outer iteration, applies a discounted update only at the end of the iteration, and persists state across iterations.

However, there are several integration issues that should be fixed before running a formal Bandit-vs-uniform MCMC experiment.

---

## 1. BLOCKING: The credited bandit action may differ from the action actually executed by the LLM

### Current behavior

The bandit selects an action:

```python
a, ctx = bandit.select(pool_topic[k], pool_phat[k])
chosen[k] = (a, ctx)
```

and the user prompt forces that action:

```python
MUTATOR_USER_TEMPLATE_FORCED.format(
    seed=pool_q[k],
    action=a,
    action_name=ACTION_NAMES[a],
)
```

However, the current mutator system prompt still contains instructions that allow or encourage changing strategies.

For example:

- V1 says not to use a strategy already exemplified by the seed.
- V2 says that if the selected strategy cannot produce a valid mutation, the model should choose another strategy.

These instructions conflict with the bandit-forced user prompt.

More importantly, after generation the code currently does:

```python
strat = extract_mutation_strategy(t) or "?"

if k in chosen:
    strat = chosen[k][0]
```

Therefore, even if the model outputs:

```text
<strategy>B</strategy>
```

while the bandit selected `A`, the proposal is still recorded/logged/credited as `A`.

This can directly corrupt the learned posterior:

\[
\text{success caused by B}
\quad\longrightarrow\quad
\alpha_{c,A} \text{ is increased}.
\]

That breaks the meaning of the bandit.

### Required fix

Create a dedicated bandit mutator system prompt, e.g.

```python
MUTATOR_SYSTEM_PROMPT_BANDIT
```

with semantics like:

> The mutation operator has already been selected externally.  
> You MUST execute exactly that operator.  
> Do not choose, substitute, or switch to another mutation strategy.

The system prompt should still contain the descriptions of A--E and the validity constraints, but it should remove instructions such as:

- "pick exactly one"
- "choose another strategy"
- "do not pick the strategy already exemplified by the seed"

because the operator has already been selected by the bandit.

Then verify the model output:

```python
actual_action = extract_mutation_strategy(t)
chosen_action, ctx = chosen[k]

if actual_action != chosen_action:
    bandit.record(ctx, chosen_action, False)
    # Treat as invalid proposal and do not let it enter MCMC.
    continue
```

Do not overwrite `actual_action` with `chosen_action` for bookkeeping.

### Acceptance criterion

For every bandit-generated proposal that reaches solver scoring:

```python
actual_action == chosen_action
```

must hold.

Any mismatch should:

1. count as a bandit failure;
2. be rejected before MCMC scoring/acceptance;
3. be explicitly logged.

---

## 2. BLOCKING: `Valid(x')` is documented in the bandit reward but is not actually used

### Intended reward

The implementation/documentation describes bandit success as:

\[
s(x')
=
\operatorname{Valid}(x')
\cdot
\mathbf 1\{p_{\min}\le \hat p(x')\le p_{\max}\}
\cdot
\operatorname{Novel}(x').
\]

### Current implementation

The actual code currently uses approximately:

```python
s_ok = (
    pls[j] not in (None, "", "None")
    and config.MIN_SCORE <= float(phs[j]) <= config.MAX_SCORE
    and int(new_cluster[k]) <= 1
)
```

This checks:

- pseudo-label exists;
- \(\hat p\) is in band;
- BLEU duplicate count is small.

It does not call `judge_one_validity()`.

The LLM validity judge is only called later in `filter_and_push()`, after the complete MCMC walk has already finished.

Therefore a malformed or mathematically broken question can currently receive:

```text
bandit success = 1
```

as long as it gives a non-null pseudo-label, lands in the desired p-hat range, and is novel.

Worse, that broken proposal may be accepted by MCMC and become the state from which later mutations are generated.

### Required fix

Move the validity gate to the proposal stage, before MCMC acceptance.

Recommended flow:

```text
LLM proposal
    ↓
parse / regex validity
    ↓
LLM validity judge
    ↓
if invalid:
    bandit failure
    reject proposal
    do not solver-score / do not enter MCMC
    ↓
if valid:
    solver score p_hat
    novelty check
    bandit success/failure
    MCMC acceptance
```

Concretely, after proposal parsing and before solver/MCMC use, evaluate:

```python
valid = judge_one_validity(q_prime, proposed_answer)
```

If `valid == False`:

```python
bandit.record(ctx, action, False)
continue
```

### Important terminology

The current judge is primarily a surface-validity / format-validity filter. It explicitly says "Do NOT solve the problem" and defaults to VALID when uncertain.

Therefore the paper/code should not imply that this guarantees mathematical correctness or well-posedness.

Prefer naming such as:

```text
surface-valid
format-valid
LLM validity filter
```

instead of claiming exact mathematical validity.

### Acceptance criterion

No proposal with:

```python
judge_one_validity(...) == False
```

should:

- receive positive bandit credit;
- be MCMC accepted;
- become a later chain state.

---

## 3. BLOCKING: `bandit_state.json` is shared globally across experiments

### Current behavior

The current code uses a fixed path similar to:

```python
bandit_path = f"{config.STORAGE_ROOT}/datasets/bandit_state.json"
```

This means two independent experimental runs can accidentally share the same learned bandit memory.

Example:

```text
run_A:
    iter1 -> iter5
    learns bandit_state.json

run_B:
    expected to start from Beta(1,1)
    actually loads run_A's posterior
```

This invalidates independent replicates.

If jobs run concurrently against the same storage directory, they may also overwrite each other's state.

### Required fix

Use a per-run state file.

For example:

```python
BANDIT_RUN_ID = os.getenv("DEO_BANDIT_RUN_ID", exp_name_or_job_name)

bandit_path = (
    f"{config.STORAGE_ROOT}/datasets/"
    f"bandit_state_{BANDIT_RUN_ID}.json"
)
```

The same five outer iterations of one experimental run must share one bandit state, while separate experimental runs must use separate state files.

Also save iteration snapshots for analysis:

```text
bandit_state_<run>_iter1.json
bandit_state_<run>_iter2.json
...
bandit_state_<run>_iter5.json
```

These snapshots will later allow plotting:

\[
P_t(a\mid c)
\]

or posterior means over iterations.

### Acceptance criterion

Two independent runs with different `DEO_BANDIT_RUN_ID` values must initialize from independent Beta(1,1) priors unless explicitly resuming the same run.

---

## 4. Recommended: The success definition for an already-trainable state is too weak

### Current issue

Suppose the old state is already trainable:

\[
\hat p(x)=0.50.
\]

A mutation produces:

\[
\hat p(x')=0.79.
\]

The current bandit reward still gives success because \(x'\) remains inside \([0.3,0.8]\).

But the proposal moved away from the maximum-uncertainty region and may even be rejected by the MCMC energy.

Thus the bandit can learn that an operator is "successful" even when it degrades the current state according to the MCMC objective.

### Suggested definition

For `Hard` or `Easy` contexts:

\[
s(x')
=
V(x')\,
I_{\rm band}(x')\,
N(x').
\]

For `Trainable` context, additionally require that uncertainty utility does not degrade substantially:

\[
s(x')
=
V(x')\,
I_{\rm band}(x')\,
N(x')\,
\mathbf 1
\left\{
r_{\rm unc}(x')
\ge
r_{\rm unc}(x)-\epsilon
\right\}.
\]

A reasonable first value is:

```python
epsilon = 0.1
```

This avoids training the bandit to prefer ultra-conservative operators merely because they keep questions barely inside the training band.

---

## 5. Recommended: Topic context is inherited from the original seed, not recomputed after mutation

### Current behavior

`pool_topic[k]` is assigned when the initial question is generated and is inherited through all accepted mutations.

Therefore the implementation is actually using:

\[
c(x_t)
=
(\operatorname{topic}(x_0),\, b(\hat p(x_t)))
\]

rather than:

\[
c(x_t)
=
(\operatorname{topic}(x_t),\, b(\hat p(x_t))).
\]

This is acceptable if mutations remain in the same mathematical domain, but the paper/code should be precise.

### Options

**Option A: keep current implementation**

Rename the variable/concept to:

```text
source_topic
seed_topic
```

and document that the topic is inherited along the chain.

**Option B: reclassify after accepted mutation**

After accepting \(x'\), infer its new topic and update:

```python
pool_topic[k] = classify_topic(x_prime)
```

This more closely matches the contextual-bandit formulation but adds classification cost/noise.

For the first experiment, Option A is probably sufficient.

---

## 6. Theoretical caveat: Contextual bandit makes proposal asymmetry more explicit

The effective proposal kernel is:

\[
K_{\mathcal B_t}(x'|x)
=
\sum_a
\pi_{\mathcal B_t}(a|c(x))K_a(x'|x).
\]

Even if the bandit is frozen within an outer iteration, generally:

\[
c(x)\neq c(x')
\]

and therefore:

\[
\pi(a|c(x))
\neq
\pi(a|c(x')).
\]

In addition, the LLM mutation kernels \(K_a\) themselves are not known to be symmetric.

Therefore the exact MH ratio should contain:

\[
\frac{
K_{\mathcal B_t}(x|x')
}{
K_{\mathcal B_t}(x'|x)
}.
\]

The current implementation still uses the approximate acceptance rule based only on the energy difference.

This is not a new bug introduced by the bandit—the original LLM mutation kernel already had the same issue—but the contextual policy makes the asymmetry even clearer.

### Recommendation

Do not claim exact MH invariance/stationarity for this implementation.

Use wording such as:

```text
approximate MH
pseudo-MH
energy-based accept/reject heuristic
```

unless a proposal-ratio estimator or reversible proposal mechanism is added later.

---

## 7. Experimental configuration recommendation

For the first clean Bandit-vs-uniform MCMC experiment, use:

```bash
DEO_BANDIT=1
DEO_KL_REF=prev
DEO_MUT_PROMPT=<bandit-specific prompt>
DEO_BANDIT_RUN_ID=<unique experiment ID>
```

Prefer a dedicated bandit/V3 prompt with:

```text
V1-style large structural steps
+
V2-style validity/self-check constraints
-
V2 locality / "stay close" requirements
```

Do not use the full conservative V2 prompt for the main bandit experiment.

---

# Recommended patch priority

## Must fix before experiment

1. Ensure executed strategy equals bandit-selected strategy.
2. Run validity filtering before bandit credit and before MCMC acceptance.
3. Use per-run bandit state files.

## Strongly recommended

4. Improve success definition for `Trainable` states.
5. Clarify whether context uses current topic or inherited source topic.
6. Keep the paper explicit that the current sampler is approximate MH.

---

# Desired end-to-end bandit proposal flow

```python
# Current chain state
x = pool_q[k]
p_old = pool_phat[k]
topic = pool_topic[k]

# 1. Thompson sampling chooses operator
action, ctx = bandit.select(topic, p_old)

# 2. LLM is forced to execute that operator
text = mutate_with_forced_operator(
    question=x,
    action=action,
)

# 3. Verify operator compliance
actual_action = extract_mutation_strategy(text)

if actual_action != action:
    bandit.record(ctx, action, False)
    reject_proposal()
    continue

# 4. Parse proposal
x_prime, gt_prime = extract_challenger_output(text)

if x_prime is None:
    bandit.record(ctx, action, False)
    reject_proposal()
    continue

# 5. Pre-MH validity gate
if not judge_one_validity(x_prime, gt_prime):
    bandit.record(ctx, action, False)
    reject_proposal()
    continue

# 6. Score proposal with solver
r_new, p_new, pseudo_new = evaluate(x_prime)

# 7. Novelty
novel = not_duplicate(x_prime)

# 8. Bandit feedback
if bandit.bucket(p_old) in ("Hard", "Easy"):
    success = (
        pmin <= p_new <= pmax
        and pseudo_new is not None
        and novel
    )
else:
    success = (
        pmin <= p_new <= pmax
        and pseudo_new is not None
        and novel
        and r_new >= r_old - epsilon
    )

bandit.record(ctx, action, success)

# 9. MCMC acceptance remains a separate decision
accept = mcmc_accept(...)

if accept:
    update_chain_state(...)
```

Crucially:

\[
\boxed{
\text{Bandit success}
\neq
\text{MCMC acceptance}.
}
\]

The bandit learns **which operator tends to generate useful proposals**.

MCMC still decides **whether a particular proposal should replace the current state**.

---

# Summary

The intended architecture is sound:

\[
\boxed{
\text{cross-iteration contextual bandit memory}
+
\text{operator-conditioned LLM mutation}
+
\text{MCMC selection}
}
\]

The current `MutationBandit` implementation is mostly correct. The main issues are in the integration:

- operator credit can be assigned to the wrong strategy;
- validity is not actually part of proposal-level bandit feedback;
- bandit state is not isolated across experimental runs.

Fixing these three items is necessary before interpreting any downstream gain or failure as evidence about whether bandit memory can reproduce the compounding benefit of a learned R-Zero challenger.

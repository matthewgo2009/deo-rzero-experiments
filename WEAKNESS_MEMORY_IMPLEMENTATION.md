# Minimal Weakness Memory for DEO

## Objective

Add one minimal cross-iteration memory loop to the existing DEO MCMC walk:

```text
m solver rollouts for a question
    -> one short weakness note
    -> one final note per MCMC chain
    -> merge all final notes into <=10 global weaknesses
    -> use one weakness to guide each chain in the next iteration
```

This is a clean **proposal-guidance ablation**. Do not change the existing:

- MCMC energy or acceptance probability;
- `p_hat`, `r_unc`, or pseudo-label calculation;
- filtering;
- beta/CD logic;
- solver training budget.

Enable it only when:

```bash
DEO_WEAKNESS_MEMORY=1
```

The disabled path must reproduce the current behavior.

## 1. What a memory item means

Do not build a large predefined skill taxonomy. A note has only:

```json
{
  "domain": "combinatorics",
  "weakness": "distinguishing ordered from unordered counting",
  "evidence": "the main answer clusters use permutations, combinations, and direct multiplication"
}
```

`domain` must be one of:

```text
algebra, geometry, number_theory, combinatorics,
probability, calculus, other
```

`weakness` describes a specific reasoning capability, not just the question
topic. Without a trusted answer, describe disagreement rather than claiming one
rollout is wrong.

```text
Good: The solver is uncertain whether order matters.
Bad:  The solver incorrectly used permutations.
```

## 2. Which of the m rollouts are used

For every evaluated question, the solver already produces
`m = config.M_SAMPLES` rollouts.

Use them as follows:

1. Extract all `m` final answers.
2. Build answer clusters using the same grouping currently used by
   `Counter(valid)`. Do not change existing `p_hat` semantics.
3. All `m` answers determine cluster counts, `p_hat`, `r_unc`, and the modal
   pseudo-label.
4. For the memory-writer prompt, include all cluster counts and one
   representative reasoning trace from each of the top three clusters.
5. Truncate each representative trace to 1,500 characters, keeping its beginning
   and ending.

A question is one observation. Never count its `m` rollouts as `m` separate
weakness observations.

Example for `m=9`:

```text
answer A: 4 rollouts
answer B: 3 rollouts
answer C: 2 rollouts
p_hat = 4/9
```

The writer sees the `4/3/2` counts plus one representative trace from A, B, and
C. The note is inferred from their reasoning divergence, not from one randomly
selected rollout.

## 3. Minimal data structures

### Per-question rollout details

Extend:

```python
evaluate_r_unc_vllm(tokenizer, questions, return_details=False)
```

Keep the current three return values when `return_details=False`. When true,
also return one details object per question:

```json
{
  "rollout_count": 9,
  "valid_answer_count": 9,
  "invalid_answer_count": 0,
  "clusters": [
    {"answer": "24", "count": 4, "representative_trace": "..."},
    {"answer": "12", "count": 3, "representative_trace": "..."},
    {"answer": "48", "count": 2, "representative_trace": "..."}
  ]
}
```

Do not persist all full rollout texts. Retain them only long enough to construct
the representatives and weakness note.

### Global memory

Save at most 10 items:

```json
[
  {
    "id": "memory_1",
    "domain": "geometry",
    "weakness": "composing multiple similarity ratios without reversing a mapping",
    "support": 61,
    "avg_p_hat": 0.43,
    "representative_evidence": "the second mapping splits into inverse answers"
  }
]
```

Paths:

```text
{STORAGE_ROOT}/weakness_memory/weakness_notes_iter_{t}.jsonl
{STORAGE_ROOT}/weakness_memory/global_weakness_memory_iter_{t}.json
```

## 4. Memory-writer call after each evaluated mutation

Use the fixed base/mutator model on `VLLM_BASE_URL` as the writer. Generate a
note only when:

```python
pseudo_label is not None
and config.MIN_SCORE <= p_hat <= config.MAX_SCORE
```

Writer system prompt:

```text
You summarize mathematical capabilities about which a solver is uncertain.

Given one problem and clusters of solver responses, return exactly one JSON
object with keys: domain, weakness, evidence.

domain must be one of algebra, geometry, number_theory, combinatorics,
probability, calculus, other.

weakness must describe the specific reasoning operation on which the response
clusters disagree, not merely the problem topic. Do not say which answer is
correct or incorrect because no verified answer is provided. Do not copy
problem-specific constants or wording. Keep each value under 30 words.
```

User prompt:

```text
PROBLEM:
{question}

SELF-CONSISTENCY:
p_hat={p_hat}; valid_answers={valid_count}/{m}

ANSWER CLUSTERS:
1. answer={answer_1}, count={count_1}
   representative reasoning={trace_1}
2. ...
```

Batch these requests through the base vLLM endpoint. If parsing fails, retry
once; after a second failure, set the note to `None` and continue the MCMC run.

## 5. One target and one current note per chain

Each pool index is one five-step MCMC chain. Maintain:

```python
pool_target_memory = [None] * num_questions
pool_weakness_note = [None] * num_questions
```

### Iteration 1

There is no previous memory. All chains use the original mutation prompt. Still
generate notes so iteration 1 creates memory for iteration 2.

### Iteration t > 1

Load `global_weakness_memory_iter_{t-1}.json` once and freeze it for the entire
iteration.

For each chain:

- with probability 0.8, sample one memory item;
- with probability 0.2, assign `None` and use the original unguided prompt.

Sample proportionally to:

```python
weight = support * (1.0 - abs(avg_p_hat - 0.5))
```

Keep the sampled target fixed across all five mutation steps.

### Guided mutation prompt

For `target=None`, keep the existing prompt unchanged. Otherwise append only the
selected item:

```text
KNOWN SOLVER WEAKNESS:
{weakness}

Mutate the seed so solving the new problem specifically requires this reasoning
capability. Preserve a unique, verifiable answer. Do not merely change numbers,
copy an old question, or include the solution/answer in the question. Keep the
mutation focused on this weakness while applying exactly one strategy A-E.
```

Do not include the full global memory in each prompt.

## 6. Note updates during the five-step walk

Generate eligible notes for the initial pool states. This matters when a chain
rejects all five proposals.

For every proposal:

1. Generate the proposal using the chain's fixed target.
2. Obtain `m` fresh solver rollouts.
3. Calculate the existing `p_hat`, `r_unc`, and pseudo-label unchanged.
4. Construct rollout details from all `m` answers.
5. Generate a weakness note if the proposal is eligible.
6. Run the existing MCMC accept/reject code unchanged.
7. Log the proposal note whether accepted or rejected.
8. Replace the chain's current note only on acceptance:

```python
if accept:
    pool_q[k] = proposal_q
    pool_phat[k] = proposal_phat
    pool_weakness_note[k] = proposal_note
else:
    # The old question remains the state, so its old note also remains.
    pass
```

Thus every mutation is diagnosed, but after five steps each chain contributes
only the note attached to its final MCMC state.

Add private metadata to returned pool records:

```python
{
    "_chain_id": i,
    "_target_memory_id": target_id_or_none,
    "_weakness_note": pool_weakness_note[i],
}
```

Do not upload underscore-prefixed fields to Hugging Face.

## 7. Build global memory after filtering

Use only final chain records that pass the existing complete filter:

1. regex checks;
2. non-null pseudo-label;
3. `p_hat` in `[MIN_SCORE, MAX_SCORE]`;
4. LLM validity judge.

Therefore invalid questions do not become “solver weaknesses.”

There is at most one input note per `_chain_id`.

### Map-reduce summary

The note list may not fit in one prompt, so use two simple LLM passes:

1. Split notes into chunks of 100. Merge semantic duplicates in each chunk into
   at most 10 provisional weaknesses.
2. Merge all provisional weaknesses once more into the final top 10.

For both passes, ask the model to return the source note indices belonging to
each cluster. Compute `support` and `avg_p_hat` in Python from those indices;
never trust LLM arithmetic.

Discard clusters supported by fewer than three distinct chains. Rank the rest by:

```python
score = support * (1.0 - abs(avg_p_hat - 0.5))
```

Save the top 10 as `global_weakness_memory_iter_{t}.json`. Do not explicitly
merge it with the old memory: the new file should describe the current solver.
Weaknesses that disappear from the current final pool should naturally expire.

## 8. Required code changes

Primary file:

```text
DEO/mcmc_deo_vllm.py
```

Implement:

1. Optional rollout details in `evaluate_r_unc_vllm`.
2. `generate_weakness_notes_batch(...)`.
3. `load_global_weakness_memory(iteration)`.
4. `sample_target_memory(memory)`.
5. `summarize_global_weakness_memory(notes, iteration)`.
6. Target assignment and current-note tracking in `generate_batch_mcmc`.
7. An optional return of fully filtered internal records from `filter_and_push`.
8. Load memory before generation and save new memory after filtering in the main
   iteration loop.

Add these environment-backed settings:

```python
WEAKNESS_MEMORY_ENABLED = bool(int(os.getenv("DEO_WEAKNESS_MEMORY", "0")))
MEMORY_GUIDED_PROB = float(os.getenv("DEO_MEMORY_GUIDED_PROB", "0.8"))
MEMORY_TOP_K = int(os.getenv("DEO_MEMORY_TOP_K", "10"))
MEMORY_MIN_SUPPORT = int(os.getenv("DEO_MEMORY_MIN_SUPPORT", "3"))
MEMORY_SUMMARY_CHUNK_SIZE = int(os.getenv("DEO_MEMORY_SUMMARY_CHUNK_SIZE", "100"))
MEMORY_TRACE_MAX_CHARS = int(os.getenv("DEO_MEMORY_TRACE_MAX_CHARS", "1500"))
```

First implement this only in the canonical fixed-beta, non-CD walk. Do not
generalize it to the other ablation entry points until the smoke test passes.

## 9. Required checks

1. `DEO_WEAKNESS_MEMORY=0` preserves current return values, prompts, and MCMC
   behavior under the same seed.
2. Synthetic `m=9` answers with counts `4/3/2` produce `p_hat=4/9` and the
   correct cluster counts.
3. A rejected proposal does not replace `pool_weakness_note[k]`.
4. An accepted proposal does replace it.
5. Five steps yield at most one final note per chain.
6. Out-of-band or judge-rejected final questions do not enter global memory.
7. Internal memory metadata is absent from the uploaded training dataset.
8. Iteration 1 is unguided and writes memory; iteration 2 loads that exact file.
9. A target selected for a chain remains fixed for all five steps.
10. Writer/summarizer failures never terminate the core DEO run.

Smoke-test with a small two-iteration run before launching 2,000 questions.

## 10. First experiment

Compare:

```text
Control: current 2000-question, fixed-beta, no-CD DEO walk
Memory:  identical setup + DEO_WEAKNESS_MEMORY=1
```

Keep the seed, beta, five mutation steps, `m`, filtering, and solver GRPO budget
fixed. Report per iteration:

- MATH-500 and AVG7;
- raw/final in-band counts;
- guided versus unguided proposal acceptance;
- global memory items and support;
- on a small audit sample, whether the final question actually tests its assigned
  weakness.

Success of the mechanism means it produces stable global memories and targeted
mutations. Higher in-band rate alone is not evidence of better downstream
training.

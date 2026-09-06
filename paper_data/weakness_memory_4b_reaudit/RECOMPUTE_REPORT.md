# Weakness-memory 4B (cool_fig) — offline re-audit of the ORIGINAL run data

Deliverable §7.2 of `WEAKNESS_MEMORY_FIXES.md`. Original data untouched at
`paper_data/weakness_memory_4b/` (commit 8cd1a81); everything here is recomputed
offline from those files. The fixed implementation lives in `DEO/mcmc_deo_vllm.py`
(v2, this commit) — nothing below is "rollout-verified": the original run did not
persist raw solver rollouts, which bounds what can still be attributed.

## 1. Confirmed cases (reproduced from the exported pools)

| case | question | note problem | status |
|--|--|--|--|
| iter1 chain530 | cubic polynomial, find f(2) | evidence: "incorrectly converted quarts to cups" | CONFIRMED — cross-question/off-topic evidence AND judgmental wording |
| iter1 memory_2 (global) | — | domain=calculus, evidence describes "a command to define a derivative" (a prompt, not solver disagreement) | CONFIRMED — degenerate note selected as representative |
| iter5 global | — | only 3 items while 22 exact-string groups have support≥3 | CONFIRMED — members silently dropped (see §2) |

**Unresolved (needs the raw rollouts the old run didn't save):** whether chain530-type
contamination comes from (a) the base-solver rollouts themselves drifting to another
problem and the writer faithfully describing an off-topic trace, (b) writer
hallucination, or (c) batch response misattribution. The v2 code persists raw
rollouts (gzipped, per writer-eligible question), maps batched responses by
`choice.index`, and requires a verbatim excerpt citation — so the next run can
attribute this class directly. Do not treat any single cause as established for the
old data.

## 2. Pure exact-string fallback recompute (all 5 iters)

`pure_fallback_recompute.json` holds the full listings. Summary (filtered final
notes → groups with support≥3 under pure same-string fallback vs what the run
actually produced):

| iter | notes | pure-fallback groups (≥3) | actual items | actual supports |
|--|--|--|--|--|
| 1 | 879 | 25 | 10 | 62, 44, 38, 34, 19, 18, 16, 15, 15, 14 |
| 2 | 885 | 29 | 10 | 67, 24, 15, 12, 11, 10, 8, 6, 5, 4 |
| 3 | 890 | 16 | 10 | 14, 12, 12, 8, 6, 6, 4, 4, 4, 4 |
| 4 | 889 | 27 | 10 | 18, 13, 10, 10, 10, 10, 10, 8, 7, 7 |
| 5 | 901 | 22 | 3 | 6, 5, 3 |

Two independent confirmations that the old run was NOT pure fallback (the earlier
README claim was wrong):

- iter1's actual top support (62) EXCEEDS the largest verbatim-duplicate count in
  its notes (29 × "solving quadratic equations") — only an LLM merge (including
  cross-domain mismerges) can produce it;
- iter5's 3 items vs 22 recomputable groups — an LLM reduce call partially
  succeeded, returned 3 valid groups, and the partial-return code path silently
  dropped every unassigned member (`_summarize_chunk_llm` pre-fix returned as soon
  as any legal group parsed).

So the mode was MIXED: some chunk calls fell back (log lines exist), at least one
merge per iter partially succeeded, and the partial-return + mid-level top-10
truncation bugs shaped the output. The v2 conservation invariants
(groups + unassigned == input at every level, no mid-level top-K, domain-bucketed
merging, audit file with trim reasons) make this class of loss structurally
impossible; `wm_events_iter_*.jsonl` now records llm_success / llm_repaired /
request_failure / parse_failure / fallback per call.

## 3. Corrections to the original README's claims

- "Every LLM merge call failed → all exact-string fallback": WRONG (see §2).
- "±0.85 run-to-run noise gauge from the iter1 replicate": WITHDRAWN — a single
  replicate cannot establish a noise band, and iter2's −1.85 was outside it anyway.
  Without multi-seed runs we report observed scores only, no significance claims.
- "Guidance never reached the questions": OVERSTATED — GPT's audit found compliant
  cases (iter2 chain1795 expected-value dice question on an expected-value target).
  Correct statement: guidance was followed inconsistently (sometimes followed,
  sometimes ignored, sometimes crudely spliced), and equal acceptance rates do not
  imply equal question distributions.

## 4. What changed in the v2 implementation (vs 8cd1a81)

- Writer: status taxonomy {weakness, problem_issue, insufficient_evidence} — only
  `weakness` notes enter memory; verbatim excerpt citation (cluster_id/rollout_id)
  structurally validated against the trace the writer actually saw; all cluster
  counts included (not just top-3); token-budget check shrinks traces instead of
  overflowing; per-question outcome statuses distinguish not_eligible /
  request_failed / parse_failed.
- Provenance: WmLogger writes eval/writer events with event_ids, gzipped raw
  rollouts for every writer-eligible question, and per-proposal note rows with
  target id + source iter; final global items carry source_event_ids/chains and a
  membership-verified representative evidence.
- Summary: domain-bucketed, token-budget-chunked map-reduce; conservation
  invariants; unassigned propagate as singletons; support = distinct chains,
  avg_p_hat recomputed in Python; audit JSON with every group and its trim reason.
- Guidance: seed-domain-compatible routing (trusted init-note domain, else one
  light batch classification; incompatible/unknown → unguided with recorded
  reason); guidance block asks to steer within the seed's structure and explicitly
  allows a normal mutation when incompatible; dedicated RNG isolates memory
  sampling from the MH stream.
- Batched LLM responses mapped by `choice.index`; missing responses recorded as
  failures instead of shifting later items.
- 10 offline acceptance tests (`DEO/weakness_memory_tests.py`) covering §6.1.

## 5. Full-experiment readiness

NOT yet cleared for a full 2000-question training run. Per §6.2 the gate is a
100-question, 2-iteration, no-training smoke run (MODE=wm_smoke) that must show:
local notes with verifiable citations, summary conservation (audit file), and
guided mutations producing plausible on-target questions at a nonzero rate.

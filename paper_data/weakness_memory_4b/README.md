# Weakness-memory DEO 4B (cool_fig) — full memory + per-iteration data

Data from the weakness-memory ablation run (`cool_fig_21ytnnypcq`, Qwen3-4B-Base,
2000q/iter, fixed β=0.1, no CD, V1 mutation prompt, 5 iters). The ONLY difference
from the control run `deo-mutv1-fixedbeta-4b` (Claude AVG7 peak 48.79 @i5) is
`DEO_WEAKNESS_MEMORY=1`. Method spec: `WEAKNESS_MEMORY_IMPLEMENTATION.md` (repo root);
implementation: `DEO/mcmc_deo_vllm.py` (search `[wm]` / `weakness`).

## Mechanism recap

1. Every evaluated proposal already has m=9 solver rollouts. For eligible questions
   (pseudo-label non-null, p̂∈[0.3,0.8]) the frozen base model writes one JSON note
   {domain, weakness, evidence} from the answer-cluster disagreement (counts + one
   truncated representative trace per top-3 cluster). Notes describe disagreement,
   never claim which answer is right.
2. Each of the 2000 MCMC chains keeps the note of its final accepted state.
3. After the complete filter (regex + pseudo + p̂ band + LLM judge), final-chain notes
   are merged into ≤10 global weaknesses; support/avg_p_hat computed in Python.
4. Next iteration, 80% of chains sample one weakness (weight = support·(1−|avg_p̂−0.5|))
   appended to the mutation prompt as "KNOWN SOLVER WEAKNESS: ..."; 20% run unguided.
   Target fixed across all 5 mutation steps.

## Files

| file | content |
|--|--|
| `global_weakness_memory_iter_{1..5}.json` | the ≤10 global weaknesses frozen for iter t+1: {id, domain, weakness, support, avg_p_hat, representative_evidence} |
| `weakness_notes_iter_{1..5}.jsonl` | EVERY proposal's note (accepted or rejected): {stage: init/step1..5, chain, accepted, target_id, domain, weakness, evidence}. `domain=null` = writer parse failure |
| `mcmc_iter_{1..5}_deo_wm_fb.json` | full 2000-question pre-filter pool. Private fields: `_chain_id`, `_target_memory_id` (the weakness this chain was guided by; null = unguided), `_weakness_note` (final-state note) |
| `filtered_deo_wm_fb_solver_v{1..5}.json` | post-filter training set actually pushed to the solver (problem/answer/score) |

## Outcome (Claude grader, AVG7)

| iter | control (mutV1) | weakness memory |
|--|--|--|
| i1 | 45.09 | 45.94 (unguided — a control replicate; Δ=+0.85 = run-to-run noise gauge) |
| i2 | 47.71 | 45.86 |
| i3 | 47.57 | 46.66 |
| i4 | 46.08 | 45.58 |
| i5 | 48.79 | 48.80 (AMC 63.83 single-set spike) |

Peak-to-peak a dead tie; mean −0.48. All within the ±0.85 noise gauge.

## Known issues found in our own audit (please verify / quantify independently)

1. **Writer quality**: notes are mostly faithful to the cluster disagreement, but many
   weakness strings are TOPIC-level ("solving quadratic equations" ×29 verbatim,
   "modular arithmetic" ×16) rather than reasoning-operation-level; 948 final notes
   contain only 754 unique strings.
2. **Summarizer degraded all 5 iters**: chunks of 100 notes exceeded the base vLLM's
   6144-token context → every LLM merge call failed → exact-string-grouping fallback.
   So `support` counts VERBATIM-duplicate generic phrases, not semantic clusters, and
   `representative_evidence` is just the first note's evidence (sometimes a degenerate
   note — e.g. iter1 memory_2's evidence describes a prompt, not solver disagreement).
3. **Guidance was ignored by the mutator (the headline finding)**: sampled guided
   chains show final questions unrelated to their target weakness (chains targeting
   "solving quadratic equations" produced a limit problem and a perimeter problem;
   "modular arithmetic" chains produced heptagon counting and LCM problems). Guided
   vs unguided acceptance rates are identical every iter (e.g. 57.1% vs 56.6% @i3).
   The seed problem dominates the mutation; a one-line appended guidance cannot pull
   a chain into a different domain. So the null result means "the guidance never
   reached the questions", NOT "targeting weaknesses doesn't help".

## Questions for the analyst

- Rate note faithfulness on a random sample: does {weakness, evidence} actually match
  the question and the described cluster disagreement?
- Quantify issue 3 properly: for each guided chain, does the FINAL question require the
  target weakness? (We only spot-checked ~6.) Per-target-memory breakdown welcome.
- Is there ANY detectable distribution shift between guided and unguided chains'
  final questions (topic mix, p̂, length)?
- Do the global memories drift/persist sensibly across iters given the solver is
  retrained each iter (e.g. "expected value" support 62 @i1 → does it shrink)?

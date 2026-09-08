# Weakness-memory v2 FULL run (willing_gyro) — the clean hypothesis test + all data

The v2.1.2 weakness-memory arm (post R1–R8 fixes, smoke-validated) at full scale:
Qwen3-4B-Base, 2000q/iter, fixed β=0.1, **no CD/U-stat** (plain MH acceptance,
m=9 rollouts), V1 mutation prompt, 5 iters. Identical to control
`deo-mutv1-fixedbeta-4b` except `DEO_WEAKNESS_MEMORY=1`. Implementation:
`DEO/mcmc_deo_vllm.py` @ v2.1.2; reviews & responses: `WEAKNESS_MEMORY_FIXES.md`,
`WEAKNESS_MEMORY_REVIEW_3BBE09D.md`, `WEAKNESS_MEMORY_R2_RESPONSE.md`.

Unlike the first attempt (`paper_data/weakness_memory_4b/`, cool_fig — where the
guidance demonstrably never reached the questions), THIS run has a verified
mechanism: real semantic memory, domain-routed targets, ~40% guided coverage.

## Outcome (Claude grader "ours", AVG7)

| iter | mutV1 control | WM v1 (cool_fig) | SGLD soft-prefix | **WM v2 (this run)** |
|--|--|--|--|--|
| i1 | 45.09 | 45.94 | 46.81 | 45.44 |
| i2 | 47.71 | 45.86 | 46.58 | 47.24 |
| i3 | 47.57 | 46.66 | 49.06 | **48.14** |
| i4 | 46.08 | 45.58 | 46.50 | 47.67 |
| i5 | 48.79 | 48.80 | 48.15 | 45.41 |
| peak | 48.79 | 48.80 | 49.06 | 48.14 |
| mean | 47.05 | 46.57 | 47.42 | 46.78 |

Per-dataset rows: `../claude_grade/4b_wm2_claude.jsonl` (raw + "ours" per iter × 7 sets).

**Verdict: with the mechanism verifiably working, weakness-guided mutation still
does not beat the plain walk** (peak −0.65, mean −0.27, no iter above the control's
same-iter score). Tenth falsified guidance-on-frozen-base intervention; graveyard
#12. The i5 dip is single-set shaped (AMC −7.8, AIME25 ≈ 0) but the sign is
consistent across all five iters. In-run MATH-500 raw: 58.2 → 60.8 → 63.0 → 62.4 →
63.4 → 63.4 (GPT-recheck +0 all iters — quota; trend only).

## Mechanism health at full scale (why this null is CLEAN)

| iter | final notes | global items (top supports) | guided chains | guided w/ ≥1 accept | merge llm_success/parse_fail/fallback |
|--|--|--|--|--|--|
| 1 | 646/2000 | 10 (25, 24, 17, 16…) | 0 (builds memory) | — | 55/45/19 |
| 2 | 648/2000 | 10 (17, 17, 14…) | 965 | 532 | 62/58/26 |
| 3 | 616/2000 | 10 (24, 16, 15…) | 771 | 425 | 74/39/15 |
| 4 | 605/2000 | 10 (22, 18, 16…) | 788 | 419 | 69/55/24 |
| 5 | 626/2000 | 10 (20, 19, 14…) | 743 | 427 | 75/63/25 |

- Guided acceptance ≈ unguided every iter (e.g. iter2: 54.1% vs 55.8%).
- Routing losses: no_seed_domain ~720/iter (domain classifier hits ~54%),
  no_compatible_target 81–277, unguided draw ~250 (by design, p=0.8).
- Smoke-run human sample (loving_bell): guided questions in-domain 5/6,
  on-exact-operation 2/6.

## Files (per iter t = 1..5)

| file | content |
|--|--|
| `mcmc_iter_t_deo_wm2_fb.json` | pre-filter 2000-pool. Private fields: `_chain_id`, `_target_memory_id` + `_target_source_iter` (guidance target), `_seed_domain`, `_guidance_reason` (guided / no_seed_domain / no_compatible_target / unguided_draw), `_weakness_note` (final-state note incl. citation), `_wm_event_id`, `_n_accepted` (0 = still the pristine seed) |
| `filtered_deo_wm2_fb_solver_vt.json` | post-filter training set (problem/answer/score) |
| `global_weakness_memory_iter_t.json` | the 10 weaknesses used to guide iter t+1: id, domain, weakness, support, avg_p_hat, representative evidence + event id, `source_event_ids`/`source_chains`, `rep_method` |
| `global_weakness_memory_iter_t_audit.json` | EVERY merge group incl. trimmed ones with `trim_reason`, plus per-run merge stats |
| `weakness_notes_iter_t.a*.jsonl` | every proposal's note row (stage, chain, accepted, target, note_status, domain/weakness/evidence/cluster_id/excerpt); `.a1` = walk attempt, `.a2` = summary-phase attempt |
| `wm_events_iter_t.a*.jsonl` | provenance events: eval (counts, p̂), writer (status, reason, FULL raw output, trace_chars), classify (match rate + samples), merge (call_id, member map, raw output, repairs) |

Raw solver rollouts (gzipped, per writer-eligible question) were persisted but are
too large for the repo — blob `yyd_wm2_fb_4b/DEO/weakness_memory/raw_rollouts_*.jsonl.gz`.

## Questions for the analyst

1. **Full target-hit audit** (the key one): for every guided chain with
   `_n_accepted > 0`, does the final question require `_target_memory_id`'s
   weakness? Rate: exact-operation / domain-only / miss, per target and per iter.
   Our 6-example smoke sample suggested ~1/3 exact — quantify it properly.
2. Guided vs unguided final-question distributions (p̂, topic mix via `_seed_domain`
   or your own classification, length, filter survival): any detectable shift?
3. Is the memory itself sensible at scale — spot-check `global_weakness_memory`
   items against their `source_event_ids` notes (writer citations included)?
4. Does the i5 collapse (AMC/AIME25) correlate with anything in the iter-5 pool
   (e.g. targets drifting to a degenerate weakness, memory_4 taking 236 chains)?
5. Given the four-way table: is there ANY reading under which guidance helped a
   subset (e.g. iters 2–4 mean vs control 2–4 mean), or is this closed?

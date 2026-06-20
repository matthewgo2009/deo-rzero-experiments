# DEO vs R-Zero on Qwen3-4B-Base — Fair Comparison Summary

Generated 2026-05-16. Base model: `Qwen/Qwen3-4B-Base`. Eval: MATH-500
(500 problems) with `mathruler.grade_answer()` + GPT-4o-mini boxed-only
secondary verification.

---

## Headline result

Both DEO (MCMC challenger + KL anchor + 2-stage filter) and R-Zero (GRPO
questioner) **fail to sustain monotonic improvement past iter 1**, but in
different ways:

- **DEO**: trains successfully 5 iters but solver acc monotonically decays
  past iter 1 peak; iter 5 falls below base.
- **R-Zero**: GRPO questioner loses output-format fidelity exponentially
  (extract success 53% → 9% → 3% → 0.7%); iter 4 dataset (17 entries)
  below verl `rollout_batch_size` even at our 64 override → assertion fail
  → pipeline crash. Cannot complete 5 iterations on Qwen3-4B-Base.

| iter | DEO (mathruler) | DEO (+GPT recheck) | R-Zero (+GPT recheck) | Δ R-Zero − DEO |
|:---:|:---:|:---:|:---:|:---:|
| 0 (base) | 58.6 | **72.2** | **72.2** | 0 |
| 1 | 62.0 | **76.8** | **76.2** | −0.6 |
| 2 | 61.4 | **75.0** | **74.6** | −0.4 |
| 3 | 59.8 | **71.8** | **76.8** † | +5.0 |
| 4 | 58.8 | **71.2** | CRASH ‡ | — |
| 5 | 57.2 | **69.0** | — | — |

† R-Zero iter 3 acc bounces back to iter 1 peak. This is **not** a sign of
better algorithm — training set is only 74 entries, so verl training at
`rollout_batch_size=64` is essentially a no-op (each prompt seen ~17× in
20 steps). Solver_v3 ≈ solver_v2 + ±2.2% MATH-500 noise.

‡ R-Zero iter 4 questioner produced 4000 raw prompts but only 17 passed
the [0.3, 0.8] score filter (extract success rate 0.7%), below the
`rollout_batch_size=64` we already lowered from default 512.

---

## R-Zero questioner format-collapse funnel

Per iter, R-Zero questioner is asked to produce a problem in the format
`<question>...</question> \boxed{answer}`. GRPO reward only measures the
solver M-vote consistency on the generated problem (no penalty for
malformed output), so the questioner drifts toward "hard" topics while
losing format adherence:

| iter | prompts sent | extract OK | extract % | filter [0.3,0.8] | → verl |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 (questioner_v1 = base raw) | 4000 | 2130 | **53%** | 709 | 709 |
| 2 | 4000 | 358 | **9%** | 127 | 127 |
| 3 | 4000 | 123 | **3%** | 74 | 74 |
| 4 | 4000 | **28** | **0.7%** | 17 | < `rollout_batch_size` → CRASH |

Extraction success halves+ every iter. This matches the R-Zero paper README
table on Qwen3-4B-Base only reporting up to iter 3 (peak MATH AVG 49.07);
their Instruct-2507 numbers come from the post-SFT-instruct model that
maintains format fidelity better.

---

## DEO trajectory analysis

DEO's MCMC challenger never trains — it always samples from base model, so
**no format-fidelity collapse**. Each iter consistently yields ~900-1000
filtered training entries:

| iter | MCMC pool | filtered | acc (recheck) |
|:---:|:---:|:---:|:---:|
| 1 | 1500 | 924 | 76.8 |
| 2 | 1500 | 968 | 75.0 |
| 3 | 1500 | 1006 | 71.8 |
| 4 | 1500 | 937 | 71.2 |
| 5 | 1500 | 920 | 69.0 |

But solver acc still degrades because:
1. MCMC picks questions with `r_unc ≈ 0.5` — by definition, ~50% of
   pseudo-labels are wrong.
2. Each iter solver is finetuned on noisy labels from prior solver.
3. KL anchor pinned to base (our verl patch) slows the drift but does not
   stop it. Self-training inherently amplifies pseudo-label noise.

Even though our additions (regex BAD_PATTERNS + GPT-mini LLM-judge filter)
clean format garbage, they cannot detect *wrong* pseudo-labels because
the judge doesn't try to solve the problem.

---

## Pipeline diff matrix

| Aspect | DEO | R-Zero |
|---|---|---|
| Base model | Qwen3-4B-Base | Qwen3-4B-Base (we matched) |
| Challenger | MCMC sampling from base | GRPO-trained `questioner_vN` |
| Challenger learnable? | No | Yes (5-step GRPO per iter) |
| KL ref in solver GRPO | Pinned to base (verl patch) | Reset to prev actor each iter |
| Question filter | regex BAD_PATTERNS + p_hat[0.3,0.8] + LLM-judge (gpt-mini boxed-only) | p_hat[0.3,0.8] + non-empty answer |
| Solver GRPO | max_steps=20, batch=16, TP=2, rollout_batch=512 | same, but rollout_batch=64 (override after crash) |
| Per-iter eval | MATH-500 via R-Zero generate.py + GPT-mini recheck | same |
| Per-iter wall-clock | ~7-8 h | iter 1 ~10 h, iter 2-3 ~1 h (small data) |
| Total wall-clock for 5 iter | ~38 h ✓ completed | crashed at iter 4 (~14 h elapsed) |

---

## Ablation: reuse-iter1 dataset (2026-05-17/18)

Trained solver_v2..v5 on the FIXED iter 1 dataset
(`yuyang322/deo_qwen3_4b_base_solver_v1@train`, 924 entries) instead of
regenerating an MCMC pool each iter. Same starting solver_v1 ckpt
(acc 76.8), same verl GRPO config, same MATH-500 eval + GPT-mini recheck.

| iter | canonical DEO | reuse_iter1 | Δ |
|:---:|:---:|:---:|:---:|
| 0 base | 72.2 | 72.2 | 0 |
| 1 | 76.8 | 76.8 | 0 |
| 2 | 75.0 | 76.2 | +1.2 |
| 3 | 71.8 | **76.8** | **+5.0** |
| 4 | 71.2 | **76.6** | **+5.4** |
| 5 | 69.0 | **73.8** | **+4.8** |
| **drop from peak** | **−7.8** | **−3.0** | — |

**Interpretation:**

The canonical −7.8 drop decomposes into:
- **~−4.8 from per-iter MCMC drift** (the gap reuse_iter1 doesn't experience): each iter
  uses the just-trained solver to M-vote pseudo-labels for the *new* MCMC pool,
  cumulatively injecting solver_v(N−1)'s biases into solver_vN's training data.
- **~−3.0 from over-training on noisy pseudo-labels** (what reuse_iter1 still suffers,
  visible at iter 5): even with fixed labels, repeated GRPO on questions whose
  pseudo-labels are ~50% wrong eventually makes the solver confidently wrong.

**Implications for self-evolving training design:**

- "Co-evolving challenger and solver each iter" is **not** strictly a curriculum benefit
  on this base model; it's a noise-amplification loop.
- A simpler and stricter setup — generate pseudo-labels **once with the base model's
  M-vote**, then run multiple GRPO iters on the same fixed dataset — yields better acc
  (76.6 at iter 4 vs canonical's 71.2) while taking less wall-clock per iter (no MCMC
  regeneration). reuse_iter1 reached the apparent self-training ceiling at iter 3 (76.8,
  same as iter 1) and held for one more iter before iter-5 drop appeared.
- The remaining ~50% pseudo-label error rate on `r_unc≈0.5` band questions is the real
  ceiling. Neither canonical DEO nor reuse_iter1 nor R-Zero addresses this — would
  require an external grader (e.g. GPT-4o on M-vote answer correctness) that
  contradicts the "zero external supervision" charter.

---

## Ablation: base-agreement filter (2026-05-18/19)

For iter ≥ 2, requires BOTH the current solver (M-vote majority) AND the
base model (M-vote majority) to agree on the pseudo-label (mathruler
`grade_answer` equivalence) before the question enters the training set.
Reuses canonical's archived `mcmc_iter_{N}.json` as question source (no
per-iter MCMC walk — cheaper). Re-scores each iter's questions with the
current iter's solver AND base, intersects. `data.rollout_batch_size=64`
override because filter is strict (drops to ~400 entries).

| iter | canonical | reuse_iter1 | **base_agreement_v2** |
|:---:|:---:|:---:|:---:|
| 0 base | 72.2 | 72.2 | 72.2 |
| 1 | 76.8 | 76.8 | 76.8 |
| 2 | 75.0 | 76.2 | 74.8 |
| 3 | 71.8 | 76.8 | 75.4 |
| 4 | 71.2 | 76.6 | **76.0** (back near peak) |
| 5 | 69.0 | 73.8 | 75.0 |
| **drop from iter1 peak** | **−7.8** | −3.0 | **−1.8 (smallest)** |

Filter funnel per iter (out of 1500 questions in canonical mcmc pool):

| iter | no_solver_pseudo | phat_oob | **disagree** | kept |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 94 | 475 | **449** | 482 |
| 3 | 97 | 483 | **466** | 454 |
| 4 | 79 | 567 | **464** | 390 |
| 5 | 92 | 583 | **461** | 362 |

The base/solver disagree filter consistently drops ~50% of the
`p_hat`-passing entries — the questions where solver has drifted from
base's natural answer. Removing them prevents the drifted pseudo-labels
from polluting training, at the cost of training set size (~400 vs
canonical's ~900).

### Refined interpretation of DEO degradation

Combining all three ablations decomposes the −7.8 peak-to-iter-5 drop:

| Configuration | iter 5 acc | drop | conclusion |
|---|:---:|:---:|---|
| canonical (per-iter MCMC + single solver M-vote) | 69.0 | −7.8 | baseline degradation |
| reuse_iter1 (no per-iter MCMC, fixed iter1 data) | 73.8 | −3.0 | MCMC drift accounts for ~4.8 |
| **base_agreement_v2 (per-iter MCMC + base/solver agreement filter)** | **75.0** | **−1.8** | **pseudo-label noise accounts for ~5–6; filter recovers most** |

**Implication**: per-iter MCMC question regeneration is not inherently
harmful — it provides fresh question variety without cost — *as long as*
you filter the resulting pseudo-labels to require an independent voter
(base model M-vote) to agree. The remaining ~1.8 drop is likely the
intrinsic ceiling from r_unc≈0.5 questions where even base+solver
agreement can both be wrong on the same answer.

### iter 1 training alignment diagnostic (2026-05-18/19)

To sanity-check that GRPO iter 1 actually did what it should:

| metric (on iter 1's 924-entry training set) | base now | solver_v1 |
|---|:---:|:---:|
| match rate (M-vote majority vs stored pseudo) | 74.2% | **86.3%** |
| mean p_hat (M-vote consistency) | 0.465 | **0.725** |
| median p_hat | 0.444 | **0.778** |

GRPO worked as intended: solver_v1 is +12pp more aligned with the
pseudo-labels than base, with median consistency jumping from 0.44 → 0.78.
But MATH-500 gain is only +4.6pp (72.2 → 76.8), so 67% of the
training-set alignment doesn't generalize — direct evidence of
pseudo-label noise as the ceiling. Saved: `diagnostics/iter1_alignment_report.json`.

### iter 2 dataset re-labeling diagnostic (2026-05-19/20)

Cross-check on canonical iter 2's 968 filtered entries: re-M-vote with
base and solver_v1, compare to stored pseudo (which canonical generated
with solver_v1 at iter 2 gen time):

| metric | base | solver_v1 (re-run) |
|---|:---:|:---:|
| match rate vs stored pseudo | 48.1% | 68.7% |
| mean p_hat | 0.314 | 0.509 |

Stored p_hat mean = 0.505 (matches solver_v1 re-run, confirming same model).
2×2 breakdown of match-vs-stored: both 40.2% | only base 8.0% | only
solver_v1 28.5% | neither 23.3%. The 23.3% "neither" cell is the worst
training noise — both base and solver_v1 reject the stored label, so
stored is most likely wrong on those 226 questions. Saved:
`diagnostics/iter2_relabel_comparison.json` (968 per-entry rows including
all three predictions and per-entry match flags).

### Ablation 4: frozen solver_v1 as labeler (2026-05-19/20) — BEST RESULT

Same setup as base_only_pseudo, but instead of base, use the frozen
solver_v1 (eval=76.8) to M-vote pseudo-labels every iter. Tests whether
a higher-quality fixed labeler outperforms a fixed low-quality labeler.

| iter | canonical | reuse_iter1 | base_agreement_v2 | base_only_pseudo | **solver_v1_label** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 72.2 | 72.2 | 72.2 | 72.2 | 72.2 |
| 1 peak | 76.8 | 76.8 | 76.8 | 76.8 | 76.8 |
| 2 | 75.0 | 76.2 | 74.8 | 76.0 | **76.6** |
| 3 | 71.8 | 76.8 | 75.4 | 76.0 | **77.2 ⭐** |
| 4 | 71.2 | 76.6 | 76.0 | 75.4 | 75.6 |
| 5 | 69.0 | 73.8 | 75.0 | 76.8 | **77.2 ⭐** |
| **drop from peak** | **−7.8** | −3.0 | −1.8 | 0.0 | **+0.4** ✅ |

**solver_v1_label is the only configuration that exceeds the iter-1 peak**.
iter 5 = 77.2 = iter 1 peak + 0.4. The average over iters 2-5 (76.65) is
also above iter 1 peak. This is the strongest self-training trajectory
across the entire ablation suite.

### Final recipe based on the full ablation suite

| design choice | how to set it |
|---|---|
| Question source | fresh MCMC pool per iter (diversity matters) |
| Labeler | frozen high-quality model (e.g. solver_v1, NOT current solver) |
| KL ref policy | pinned to base (verl patch) |
| `data.rollout_batch_size` | 64 override (default 512 brittle on small filtered sets) |
| Filter | regex BAD_PATTERNS + p_hat∈[0.3, 0.8] + non-null pseudo |
| Per-iter eval | MATH-500 generate.py + GPT-mini boxed-only recheck |

Versus canonical DEO (76.8 → 69.0, **−7.8**), this recipe achieves
76.8 → 77.2 (**+0.4**, an **8.2-point swing** at iter 5).

---

## Ablation: walk-vs-drift decomposition (2026-06-01/02)

Earlier ablations confounded two changes against canonical: `baseline_klfix`
removed MCMC walk *and* froze the labeler at solver_v1. To isolate, we
added two more variants:

- **baseline_drift**: MCMC_STEPS=0 (no walk) + canonical-style drifting
  labeler (vllm_solver reloaded every iter to the freshly-trained ckpt).
  Single-variable diff from canonical: walk only.
- **baseline_klfix_conf**: `baseline_klfix` (no walk, frozen sv1) +
  stage-5 conf filter (keep top-50% by per-token logprob of majority
  trajectories). Stacks two of the strongest individual fixes.
- **conf_filter_canonical**: canonical (walk + drift) + conf filter.
  Tests whether label-quality filter alone can fix canonical.

Per-iter MATH-500 acc (GPT-mini rechecked):

| iter | canonical | baseline_klfix (no walk + frozen sv1) | **baseline_drift** (no walk + drift) | conf_filter canonical | baseline_klfix + conf |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 72.2 | 72.2 | 72.2 | 72.2 | 72.2 |
| 1 | 76.8 | 77.6 | 76.0 | 75.8 | 76.6 |
| 2 | 75.0 | 76.0 | 76.8 | 76.4 | **78.4** |
| 3 | 71.8 | 74.0 | 77.8 | 76.6 | 75.8 |
| 4 | 71.2 | 75.8 | **78.4** | 75.8 | 78.0 |
| 5 | 69.0 | 77.4 | 75.2 | 76.0 | 75.8 |
| **peak → i5** | **−7.8** | −0.2 | −3.2 | −0.6 | −2.6 |
| **mean 2-5** | 71.75 | 75.80 | **77.05** | 76.20 | **77.00** |

### Decomposing canonical's −7.8 pp degradation (iter-5 view)

| condition | iter-5 acc | Δ vs canonical | knob isolated |
|---|---|---|---|
| canonical (walk + drift) | 69.0 | — | (baseline) |
| baseline_drift (no walk + drift) | 75.2 | **+6.2** | walk removed |
| baseline_klfix (no walk + frozen sv1) | 77.4 | +8.4 | walk + drift both removed |
| baseline_klfix + conf (stacked fixes) | 75.8 | +6.8 | walk + frozen + conf filter |

Roughly additive:
- **MCMC walk contributes ~6.2 pp of canonical's degradation** (single
  biggest factor; previously underestimated)
- **Labeler drift contributes ~2.2 pp on top** (76.8 peak → 77.4 frozen
  is +0.6, but iter-5 lift over baseline_drift is +2.2)
- **Conf filter alone (on canonical pipeline) recovers +7.0 pp**
  (69.0 → 76.0), validating that pseudo-label noise drives the loss

### Updated takeaway

The prior conclusion ("frozen labeler is the key fix") is **partially
correct but understated the role of MCMC walk**. The full picture:

1. **MCMC walk is the dominant source of degradation (~6.2 pp).** It
   pushes the question pool toward harder/noisier regions over iters via
   r_unc + Metropolis-Hastings, making iter N's pool out-of-distribution
   for iter N's solver. Removing it alone recovers most of the loss.
2. **Labeler drift adds ~2 pp on top.** Freezing the labeler at solver_v1
   (or any high-quality fixed model) buys an additional ~2 pp at iter 5.
3. **Conf filter is an orthogonal noise-reduction lever (~5-7 pp).**
   Keeping only the top-50% of pseudo-labels by per-token logprob is
   nearly as effective as the labeler swap (and stackable with it).
4. **Stacking returns diminish.** baseline_klfix+conf (no walk + frozen
   sv1 + conf filter) peaks at 78.4 (highest single iter across all
   ablations) but drops to 75.8 at iter 5, slightly worse than
   baseline_klfix alone (77.4). The conf filter cuts training data in
   half on top of an already-small no-walk pool (~225 entries), and the
   reduced training signal partially offsets the cleaner labels.

### Updated final recipe

| design choice | how to set it |
|---|---|
| MCMC walk | **disable (MCMC_STEPS=0)** — biggest single fix |
| Question source | fresh init pool per iter (no walk) |
| Labeler | frozen solver_v1 (canonical or this run's iter 1 ckpt) |
| Conf filter | optional; helps canonical (+7 pp), not stackable on small no-walk pools |
| KL ref policy | pinned to base (verl patch) |
| `data.rollout_batch_size` | 64 override (default 512 brittle on small filtered sets) |
| Filter | regex BAD_PATTERNS + p_hat∈[0.3, 0.8] + non-null pseudo + LLM-judge |

Best end-to-end iter-5: solver_v1_label (77.2, +0.4 over peak),
followed by baseline_klfix (77.4, −0.2 from peak). Highest single iter
across all ablations: baseline_klfix+conf iter 2 = baseline_drift iter
4 = 78.4 (+1.6 over canonical peak).

---

## Datasets shipped in this directory

```
DEO/                                       # canonical DEO 5-iter run (KL fix + filter + DP)
  results_summary_mathruler_only.json      mathruler-only acc (raw)
  results_summary_rechecked.json           GPT-mini boxed-only rechecked acc
  filtered_datasets/                       per-iter training set (problem, answer, score)
  mcmc_pools/                              per-iter pre-filter MCMC pool (1500 each)
  evaluation/                              raw MATH-500 results_math.json (bumped scores in place)
DEO_reuse_iter1_ablation/                  # ablation: same iter1 data for all 5 iters
  results_summary_reuse_iter1.json         GPT-mini rechecked per-iter acc
  evaluation/                              raw MATH-500 results_math.json for solver_v2..v5
  models/                                  config.json + tokenizer per solver_vN (full ckpts too big, skip)
DEO_base_agreement_v2_ablation/            # ablation: base/solver M-vote agreement filter
  results_summary_base_agreement_v2.json   GPT-mini rechecked per-iter acc
  filtered_datasets/                       per-iter training set (~400 entries each)
  evaluation/                              per-iter results_math.json with bumped scores
  models/                                  config.json + tokenizer per solver_vN
DEO_solver_v1_label_ablation/              # ablation: solver_v1 as frozen labeler (BEST iter-5)
  results_summary_solver_v1_label.json     GPT-mini rechecked per-iter acc (peak +0.4 @ iter5)
  filtered_datasets/                       per-iter training set (solver_v1-labeled)
  evaluation/                              per-iter results_math.json with bumped scores
  models/                                  config.json + tokenizer per solver_vN
DEO_baseline_drift_ablation/               # ablation: no walk + DRIFTING labeler (isolates walk only)
  results_summary_baseline_drift.json      iter 5 = 75.2; peak iter 4 = 78.4
  filtered_datasets/                       per-iter training set (~450 entries each)
  mcmc_pools/                              per-iter init pool (MCMC_STEPS=0, no walk)
  evaluation/                              per-iter results_math.json
  models/                                  config.json + tokenizer per solver_vN
DEO_baseline_klfix_conf_ablation/          # ablation: no walk + frozen sv1 + conf filter (stacked)
  results_summary_baseline_klfix_conf.json iter 2 = 78.4 (HIGHEST single iter across all ablations)
  filtered_datasets/                       per-iter training set (~225 entries, top-50% conf)
  mcmc_pools/                              per-iter init pool
  evaluation/                              per-iter results_math.json
  models/                                  config.json + tokenizer per solver_vN
diagnostics/                               # per-question diagnostics + alignment reports
  iter1_alignment_report.json              solver_v1 vs base on iter 1's 924 training entries
  iter2_relabel_comparison.json            base + solver_v1 re-vote on iter 2's 968 entries
  iter2_relabel_with_gt.json               same, plus challenger's gt joined in
R-Zero/                                    # R-Zero fair comparison, crashed at iter 4
  final_results.jsonl                      per-model MATH-500 score from results_recheck_math500_mini.py
  filtered_datasets/                       per-iter training set, iter 1-4
  raw_pre_eval/                            per-shard {save_name}_{i}_results.json with raw scores
  evaluation/                              raw MATH-500 results_math.json
```

---

## Methodological notes worth remembering

1. **Mathruler systematically undercounts ~13-14% on MATH-500** for
   Qwen3-4B outputs due to format equivalence misses (`\frac{1}{2}` vs
   `0.5`, etc.). GPT-4o-mini secondary verification with **boxed-only**
   prompt recovers most of these. Full-text GPT prompts have ~33% false
   positive rate; boxed-only ~0% in our eyeball.

2. **R-Zero paper's headline 69.8 on MATH-500 is on Instruct-2507, not
   Base.** Their `final_results.jsonl` model path confirms. README table
   for Qwen3-4B-Base only reports MATH AVG (49.07 at iter 3), an average
   over 7 math benchmarks (math, gsm8k, amc, minerva, olympiad, aime24/25).

3. **`data.rollout_batch_size` default 512 is the silent killer** in
   verl: any pipeline producing <512 filtered entries hits
   `assert len(train_dataloader) >= 1` AssertionError. Override to 64
   (or smaller) lets small datasets train, at cost of noisier gradients.

4. **DEO's `extract_challenger_output` had a nested-`<question>` bug**
   (mutator sometimes outputs two open tags before one close → outer
   regex captures literal tag inside). Fixed with `rfind("<question>")`
   on the captured group; affects ~2% of MCMC pool entries pre-filter.

5. **Both DEO and R-Zero fail to address pseudo-label noise**, which is
   the real ceiling. Any future self-training scheme on this base would
   benefit from a per-question external grader (e.g. GPT-4o on solver
   M-vote answer correctness) before training; that's outside both
   schemes' "zero external supervision" charter though.

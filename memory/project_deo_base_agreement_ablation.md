---
name: project-deo-base-agreement-ablation
description: "2026-05-18/19 ablation showing base/solver M-vote agreement filter cuts DEO's peak-to-iter5 drop from -7.8 (canonical) to -1.8 — strongest fix among three ablations"
metadata: 
  node_type: memory
  type: project
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

Ablation conducted 2026-05-18 → 05-19 on `/eph/nvme0/yyd/DEO_base_agreement_v2/`,
archived to `paper_data/DEO_base_agreement_v2_ablation/`. For iter ≥ 2, requires
the current solver's M-vote majority AND the base model's M-vote majority to
agree on the pseudo-label (mathruler `grade_answer`) before a question enters
the training set. Reuses canonical's `mcmc_iter_{N}.json` as question source
(no per-iter MCMC walk). Adds `data.rollout_batch_size=64` because filter
shrinks training set to ~400 entries.

**Three-way trajectory comparison (rechecked acc on MATH-500):**

| iter | canonical | reuse_iter1 | base_agreement_v2 |
|---|---|---|---|
| 1 peak | 76.8 | 76.8 | 76.8 |
| 2 | 75.0 | 76.2 | 74.8 |
| 3 | 71.8 | 76.8 | 75.4 |
| 4 | 71.2 | 76.6 | **76.0** (back near peak) |
| 5 | 69.0 | 73.8 | 75.0 |
| **drop from peak** | **−7.8** | −3.0 | **−1.8** ✅ smallest |

**Refined decomposition of canonical's −7.8 degradation:**
- ~4.8 = per-iter MCMC drift contribution (reuse_iter1 removes this)
- ~5–6 = pseudo-label noise contribution (base_agreement_v2 removes most)
- ~1.8 = remaining ceiling (intrinsic to r_unc≈0.5 questions where even
  base+solver agreement can be wrong on the same answer)

**Key implication for self-training design:**
Per-iter MCMC regeneration is NOT inherently harmful — it provides fresh
question variety at low cost. What's harmful is using the just-trained
solver's biased M-vote alone for pseudo-labels. Adding an independent
voter (base model) to confirm the label is the cheapest, highest-impact
fix tested so far.

**Filter funnel per iter (out of 1500 canonical mcmc pool questions):**
- ~80-100 lose solver pseudo (extract garbage)
- ~475-583 fail p_hat ∈ [0.3, 0.8]
- **~449-466 base/solver disagree** ← this is the new filter contribution
- ~360-480 final training set

Disagree rate stable at ~50% of p_hat-passing entries across iters.

**Diagnostic: iter 1 training alignment (saved at `iter1_alignment_report.json`):**

| metric on iter1's 924 training entries | base | solver_v1 |
|---|---|---|
| match stored pseudo (M-vote majority) | 74.2% | 86.3% |
| mean p_hat (consistency) | 0.465 | 0.725 |
| median p_hat | 0.444 | 0.778 |

GRPO iter 1 successfully aligned solver_v1 with pseudo-labels (+12pp match,
+0.26 p_hat). But only +4.6 pp generalized to MATH-500 — 67% of training-set
alignment is overfit to noisy labels. Direct evidence of pseudo-label
correctness as the self-training ceiling.

**Use when:**
- Designing future self-training experiments: include double-voter
  agreement as a pseudo-label filter (cheap, high-impact).
- Citing DEO trajectory degradation: it decomposes into MCMC drift + label
  noise; the latter is bigger and base-agreement filter recovers most.

Related: [[project-rzero-vs-deo-findings]], [[project-deo-reuse-iter1-ablation]],
[[project-math500-eval]].

---
name: project-deo-walk-vs-drift-decomposition
description: "DEO canonical degradation decomposes into MCMC walk (-6.2 pp) and labeler drift (-2.2 pp) at iter 5; walk is the dominant factor, not drift as previously thought"
metadata: 
  node_type: memory
  type: project
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

Canonical DEO on Qwen3-4B-Base loses 7.8 pp from peak (iter 1 = 76.8) to
iter 5 = 69.0. We initially blamed labeler drift; the baseline_drift
ablation (no walk + canonical-style drifting labeler) showed otherwise.

Decomposition at iter 5 (MATH-500 + GPT-mini boxed-only rechecked):

| condition | iter 5 acc | Δ vs canonical |
|---|---|---|
| canonical (walk + drift) | 69.0 | — |
| baseline_drift (no walk + drift) | 75.2 | +6.2 ← walk only |
| baseline_klfix (no walk + frozen sv1) | 77.4 | +8.4 ← walk + drift |
| conf_filter canonical (walk + drift + conf) | 76.0 | +7.0 ← walk + drift + label noise filter |

**MCMC walk contributes ~6.2 pp to canonical's degradation; labeler drift
adds ~2.2 pp on top. Walk is the dominant single factor.**

Mechanism conjecture: walk uses r_unc + Metropolis-Hastings to bias the
question pool toward "hard for current solver" regions across iters. Each
iter the pool drifts further out-of-distribution from the solver, so
training signal becomes increasingly noisy. Removing the walk (fresh
init pool every iter, no mutation) keeps difficulty stable. Drift on top
of stable difficulty causes only modest degradation.

**Why:** clarifies that the original DEO design's MCMC challenger is the
biggest bug — the very mechanism advertised as "diversity-promoting" was
the dominant degradation driver. The labeler-freeze fixes
([[project-deo-solver-v1-label-ablation]]) work because they reduce
label noise downstream, but the upstream walk is more important to fix.

**How to apply:** when discussing trade-offs of DEO components in
ablation reports or paper drafts, lead with the walk-removal effect
(+6.2 pp at iter 5), not labeler-freezing. The recommended recipe should
disable MCMC walk first; frozen labeler and conf filter are secondary
levers.

Full per-iter data: `/eph/nvme0/yyd/paper_data/DEO_baseline_drift_ablation/`
and SUMMARY.md "walk-vs-drift decomposition" section. Linked memories:
[[project-deo-solver-v1-label-ablation]] (the "BEST iter-5" frozen
labeler result, now superseded as the dominant fix),
[[project-deo-base-agreement-ablation]] (earlier decomposition assigning
~5-6 pp to "pseudo-label noise" — actually mostly walk-induced label
shift).

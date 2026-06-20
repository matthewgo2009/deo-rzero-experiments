---
name: project-deo-conf-filter-ablation
description: Top-50% confidence (per-token logprob) filter on canonical recovers +7.0 pp at iter 5 without changing labeler; stacks weakly with frozen labeler
metadata: 
  node_type: memory
  type: project
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

Stage-5 filter added to canonical DEO pipeline: after the existing
4-stage filter (regex + p_hat + pseudo + LLM-judge), run one more
solver M-vote with `logprobs=1`, compute per-token mean logprob of
trajectories matching the majority answer, keep top-50% by confidence.

**Two configurations tested:**

| pipeline | iter 1 | iter 2 | iter 3 | iter 4 | iter 5 | peak | peak → i5 |
|---|---|---|---|---|---|---|---|
| conf_filter canonical (walk + drift + conf) | 75.8 | 76.4 | 76.6 | 75.8 | 76.0 | 76.6 | -0.6 |
| baseline_klfix + conf (no walk + frozen sv1 + conf) | 76.6 | 78.4 | 75.8 | 78.0 | 75.8 | 78.4 | -2.6 |

Key results:
- conf_filter canonical iter 5 = 76.0 (vs canonical 69.0 → **+7.0 pp**).
  Recovers almost all degradation **without changing the labeler**.
- baseline_klfix+conf peaks at 78.4 (highest single iter across all
  ablations) but drops to 75.8 at iter 5, slightly worse than
  baseline_klfix alone (77.4). Conf filter cuts training data in half
  on top of already-small no-walk pool (~225 entries); reduced training
  signal partly offsets cleaner labels.

**Confidence definition:** for each candidate that survived the 4-stage
filter, run M-vote with logprobs; for trajectories producing the
majority answer, take per-token mean logprob; confidence = mean across
matching trajectories. Higher (less negative) = more confident.

Empirically (iter1 dataset, 924 entries graded by GPT-5 oracle):
- correctness ROC-AUC for per-token logprob alone: 0.65
- correctness ROC-AUC for M-vote consistency p_hat: 0.75
- correct-rate jumps from 35% (lowest-conf quartile) to 68% (highest)

**Why:** conf filter is the only "no labeler change" lever that recovers
most of canonical's loss, and it's orthogonal to frozen-labeler fixes.
For paper narrative, it provides an alternative path to stability that
doesn't require designing a labeler-swap protocol.

**How to apply:** when canonical-style pipelines must keep a drifting
labeler (e.g., for honest "self-evolving" claim), recommend conf filter
as the noise-reduction lever. Avoid stacking with frozen labeler on
small (<500-entry) datasets — training-data reduction dominates.

Full code: `/home/azureuser/yyd/DEO/conf_filter_canonical_main.py` and
`baseline_klfix_conf_main.py`. Archived data at
`/eph/nvme0/yyd/paper_data/DEO_baseline_klfix_conf_ablation/`.

Linked: [[project-deo-walk-vs-drift-decomposition]] (the broader
attribution context), [[project-deo-solver-v1-label-ablation]] (the
labeler-freeze fix that conf filter is orthogonal to).

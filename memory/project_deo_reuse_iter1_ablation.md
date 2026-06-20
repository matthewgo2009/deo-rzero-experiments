---
name: project-deo-reuse-iter1-ablation
description: "2026-05-17/18 ablation showing per-iter MCMC pool regeneration accounts for ~60% of DEO's iter 2-5 degradation; fixed iter1 dataset + multi-iter GRPO holds peak acc 3 more iters"
metadata: 
  node_type: memory
  type: project
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

Ablation conducted 2026-05-17 → 05-18 on `/eph/nvme0/yyd/DEO_reuse_iter1/`
(archived to `paper_data/DEO_reuse_iter1_ablation/`). Setup: start from
canonical DEO solver_v1 ckpt, train solver_v2..v5 with verl GRPO on the
**fixed** iter 1 HF dataset (`yuyang322/deo_qwen3_4b_base_solver_v1@train`,
924 entries) instead of regenerating MCMC each iter. Same eval + recheck.

| iter | canonical (per-iter MCMC) | reuse_iter1 (fixed) | Δ |
|---|---|---|---|
| 1 peak | 76.8 | 76.8 | 0 |
| 2 | 75.0 | 76.2 | +1.2 |
| 3 | 71.8 | 76.8 | **+5.0** |
| 4 | 71.2 | 76.6 | +5.4 |
| 5 | 69.0 | 73.8 | +4.8 |
| **drop from peak** | −7.8 | −3.0 | — |

**Decomposition of canonical's −7.8 drop:**
- **~−4.8 = per-iter MCMC drift** (the gap reuse_iter1 avoids). Each iter's
  MCMC pool is M-voted by the just-trained solver, so solver_v(N−1)'s biases
  cumulatively contaminate solver_vN's training labels. Removing this loop
  preserves peak acc for ~3 more iters.
- **~−3.0 = over-training on fixed noisy labels** (still in reuse_iter1's
  iter-5 drop). Even with stable labels, repeated GRPO over ~50%-wrong
  pseudo-labels eventually makes the solver confidently wrong.

**Use this finding when:**
- Designing next-gen self-training experiments: a simpler "generate pseudo-labels
  ONCE with base model + multi-iter GRPO on fixed data" beats canonical
  co-evolution. Less compute (~no per-iter MCMC), better acc.
- Reviewing R-Zero / DEO architectural claims: the "co-evolution" premise that
  generation and learning should iterate together is **not** strictly beneficial
  on Qwen3-4B-Base. Generation iteration is a noise amplifier here.
- The real bottleneck remaining is **pseudo-label correctness at r_unc≈0.5**
  band — neither approach addresses it without external supervision.

Related: [[project-rzero-vs-deo-findings]] (R-Zero collapse mode is different,
its challenger format-fidelity drops, but solver-side dynamics share this
ceiling), [[project-math500-eval]].

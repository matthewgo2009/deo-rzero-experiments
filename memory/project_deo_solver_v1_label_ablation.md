---
name: project-deo-solver-v1-label-ablation
description: "2026-05-19/20 ablation — frozen solver_v1 as labeler across all iters; +0.4 above iter-1 peak, only configuration to net-improve trajectory"
metadata: 
  node_type: memory
  type: project
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

Ablation conducted 2026-05-19 → 05-20, archived at
`paper_data/DEO_solver_v1_label_ablation/`. For iter ≥ 2, M-vote the
canonical mcmc_iter_{N}.json pool with the FROZEN solver_v1 ckpt (eval
76.8) — never reloaded as solver evolves. Trainer is still updated each
iter (solver_v(N-1) → solver_vN); only the labeler is frozen.

**Complete 5-way ablation table (MATH-500 acc, GPT-mini boxed-only rechecked):**

| iter | canonical | reuse_iter1 | base_agreement_v2 | base_only_pseudo | **solver_v1_label** |
|---|---|---|---|---|---|
| 1 peak | 76.8 | 76.8 | 76.8 | 76.8 | 76.8 |
| 2 | 75.0 | 76.2 | 74.8 | 76.0 | **76.6** |
| 3 | 71.8 | 76.8 | 75.4 | 76.0 | **77.2 ⭐** |
| 4 | 71.2 | 76.6 | 76.0 | 75.4 | 75.6 |
| 5 | 69.0 | 73.8 | 75.0 | **76.8** | **77.2 ⭐** |
| **drop** | **−7.8** | −3.0 | −1.8 | 0.0 | **+0.4** ✅ |

**Why this wins:** combines two beneficial design choices:
- Frozen labeler (no per-iter drift loop) — same protection as reuse_iter1
  and base_only_pseudo
- Higher-quality labeler than base: solver_v1 has 76.8 acc vs base's 72.2,
  so its pseudo-labels are ~4-5 pp more correct in absolute terms, AND
  ~20 pp better aligned with stored canonical iter 2 labels (per
  `iter2_relabel_comparison.json`).

**Self-training recipe (validated across the full ablation suite):**
1. Fresh MCMC question pool per iter (diversity helps)
2. Frozen labeler (NOT current solver — drift kills you)
3. Use a HIGHER-QUALITY labeler if possible (solver_v1 > base)
4. KL ref pinned to base
5. `data.rollout_batch_size=64` override (small datasets crash on default 512)
6. Filter: regex BAD_PATTERNS + p_hat∈[0.3,0.8] + non-null pseudo

**Implementation detail to remember:**
Symlinks to /storage paths inside the launcher script must be RELATIVE
(or use `/storage/...`), NOT absolute /eph/nvme0/yyd/DEO/... — the host
path is not resolvable inside the docker container.
Use `(cd /eph/nvme0/yyd/DEO/models && ln -s SRC LINK)` to create relative.

Related: [[project-rzero-vs-deo-findings]], [[project-deo-reuse-iter1-ablation]],
[[project-deo-base-agreement-ablation]], [[project-math500-eval]].

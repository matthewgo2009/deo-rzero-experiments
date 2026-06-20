---
name: feedback-data-driven-recommendations
description: "Before recommending one option over another in this project, measure or A/B-test it; do not argue from abstractions"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

When advising on an engineering trade-off (GPU allocation, prompt versions, filter
thresholds, parallel strategies), measure or A/B-test first and lead with the numbers.
Abstract argumentation alone is insufficient and the user pushes back on it.

**Why:** Two clear examples from 2026-05-13:
1. User initially asked "+1 GPU MCMC, +1 GPU verl"; I gave back a table with measured
   timings + Qwen3-4B head count (32, not divisible by 3) + FSDP/TP integer constraint
   showing verl +1 would give ≈0 speedup. User immediately switched to "+2 GPU to MCMC,
   verl 不动" because the numbers were concrete.
2. For the LLM-judge prompt v2 vs v3, I A/B'd on 943 candidates and showed v3 freed 49
   entries, then hand-classified 8 random ones (5/8 wrongly accepted, including
   `[your problem here]` placeholder). User immediately picked v2.

**How to apply:**
- For GPU allocation: compute power-of-2 TP constraints, FSDP-divisor constraints, and
  rough per-stage timing per layout before recommending.
- For prompt / classifier changes: A/B on existing data and report both aggregate
  counts AND a sampled set of disagreement cases.
- For "this might be slower / faster": measure or estimate from a similar known
  benchmark before claiming it. Avoid "could be" / "might".

Related: [[feedback-smoke-test-first]].

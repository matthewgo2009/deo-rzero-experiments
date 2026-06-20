---
name: project-math500-eval
description: "MATH-500 evaluation methodology for DEO; mathruler alone undercounts ~13-14%; GPT-mini boxed-only recheck reproduces R-Zero's GPT-4o post-processing"
metadata: 
  node_type: memory
  type: project
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

DEO's MATH-500 numbers must be reported *after* a GPT-4o-mini secondary
verification stage, otherwise they are systematically ~13-14% lower than the
true accuracy and not comparable to R-Zero paper numbers.

**Why:** Discovered 2026-05-14 while reconciling DEO's plateau at ~62% vs
R-Zero's reported 69.8% on MATH-500. Two root causes:

1. **mathruler is too strict.** It misses common equivalent forms like
   `\frac{1}{2}` vs `0.5`, `\dfrac` vs `\frac`, `(2,3)` vs `(2, 3)`, matrix
   spacing, etc. On Qwen3-4B-Base output this miscounts ~13-14% of MATH-500
   entries.

2. **R-Zero's reported 69.8 was on `Qwen3-4B-Instruct-2507`, not Base.** Their
   `final_results.jsonl` (in /home/azureuser/yyd/R-Zero/) shows the path
   `qwen3-4b-instruct-2507_solver_v1`. The Base-model number is in the
   README table as MATH AVG=49.07 (avg over 7 benchmarks, not MATH-500 alone).
   Direct apples-to-apples DEO-on-Base vs R-Zero-on-Base is not what their
   headline number measures.

3. **R-Zero adds GPT-4o post-processing.** `evaluation/results_recheck.py`
   reruns mathruler-failed entries through GPT-4o, bumps them to score=1.0
   if GPT confirms equivalence. DEO had no such step until we added it.

**How to apply:**

- When citing DEO's MATH-500 acc, use `results_summary_rechecked.json` (built
  from per-entry rechecked `results_math.json` files via `recheck_all.sh`),
  NOT `results_summary.json` (which the in-flight run keeps overwriting with
  mathruler-only numbers).
- GPT-judge prompt must use **boxed-only** extraction (see
  `mcmc_deo_vllm.py:_recheck_user_msg`). Full-text prompts give ~33% false
  positive rate because GPT-mini gets distracted by reasoning traces and
  hallucinates equivalence — e.g. accepted `\boxed{-330}` as equivalent to
  GT 331 when both appeared in the long response. Boxed-only got 0/10 FP
  in eyeball check.
- Cost is trivial (~$0.02 per full 5-iter DEO run with gpt-4o-mini). Don't
  bother optimizing; just always include the recheck.
- When comparing DEO to R-Zero paper, remember the base-model gap: their
  Qwen3-4B-Base trajectory peaks at MATH AVG ~49% (over 7 benchmarks);
  comparing only on MATH-500 needs a Base-only R-Zero run, which we don't
  have (their local run on this cluster used Instruct-2507).

Related: [[deploy-md-pointer]], [[feedback-data-driven-recommendations]].

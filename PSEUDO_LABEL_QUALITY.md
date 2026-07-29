# Pseudo-label quality of self-evolving training data (Qwen3-8B-Base)

**Question.** DEO/R-Zero train the solver on *self-generated* questions labeled by the solver's own
majority vote (pseudo-label ȳ(x)). How **correct** are those pseudo-labels, and how does label quality
relate to downstream accuracy?

**Method.** For each run we take its per-iteration training pool (DEO: the MCMC dataset `mcmc_iter_N`;
R-Zero: the HF training parquet `..._solver_vN@train`, filtered to score∈[0.3,0.8]), sample **300
questions per iteration**, and use **Claude Sonnet-5 as an independent judge**: Claude solves each
problem from scratch and returns `VERDICT: YES/NO` on whether the solver's pseudo-label is correct.
"acc" = fraction YES over the *judged* subset; "unjudged" = Claude emitted no parseable verdict within a
3000-token budget (the hardest / ill-posed / ambiguous questions). Grader is a *different* model family
from the solver, so this is an external check. 3 runs × 5 iters × 300 = 4500 judgements.

## Per-iteration pseudo-label accuracy (%, Claude-judged)

| run | v1 | v2 | v3 | v4 | v5 | mean unjudged |
|---|--|--|--|--|--|--|
| baseline_drift (no walk) | 66.7 | 70.8 | 77.3 | 70.3 | 73.2 | 16% |
| adaptive strong-control (β→0.02) | 47.9 | 52.7 | 46.8 | 72.3 | 71.2 | 23% |
| **R-Zero** | 72.2 | 58.0 | 46.4 | 33.8 | **28.7** | **46%** |

(mean r_unc of the sampled pool per iter — baseline ~0.55 flat; strong 0.51–0.56 then **0.76** at v4–v5
once β hits its floor 0.02; R-Zero ~0.52 flat. Judged counts /300: baseline ~250, strong ~230, R-Zero
falls 223→129.)

Pooled over all 5 iters (150-sample earlier run, consistent): baseline **68.7%**, strong **64.3%**,
R-Zero **47.6%**.

## Findings

1. **R-Zero's pseudo-label quality COLLAPSES across iterations: 72% → 29%**, and the unverifiable
   fraction climbs from 26% (v1) to 57% (v5). The *trained* questioner progressively reward-hacks the
   solver-uncertainty objective by emitting increasingly **ambiguous / ill-posed / unanswerable**
   questions — so much so that even Claude cannot assign a verdict to over half of v5. Its self-generated
   training data becomes largely noise.

2. **Yet R-Zero achieves the BEST downstream accuracy** (8B MATH AVG peak 53.7 ours / 54.6 Claude-graded;
   best hard-set incl. AIME). So the intuitive chain **"more-correct pseudo-labels → better solver" does
   NOT hold** for these self-evolving methods. R-Zero's gains come *despite* collapsing label quality —
   plausibly from (a) the questioner's adaptive *difficulty* (keeps producing "solver half-right" items
   whose GRPO group-advantage is non-zero), and (b) GRPO's group normalization tolerating label noise.

3. **baseline_drift (no MCMC walk) has the most stable, accurate labels (~70–77%, ↑ slightly as the
   solver strengthens)** — base-model questions aren't reward-hacked.

4. **Strong-β control:** labels are noisy early (~48–53% at v1–v3) then rise to ~72% at v4–v5, coinciding
   with β hitting its floor (0.02) and the pool shifting to r_unc≈0.76 (p̂≈0.62, moderately-confident)
   rather than the p̂≈0.5 max-uncertainty region. Even so, downstream accuracy was unchanged vs mild-β
   adaptive — i.e. neither label quality nor question hardness in this range moved final MATH accuracy.

## Caveats

- "acc" is over the *judged* subset; unjudged (hardest/ill-posed) are excluded, so the true correctness
  including them is **lower** — most severely for R-Zero (46% unjudged), so its real label quality is
  even worse than the numbers above.
- Claude Sonnet-5 is a strong but imperfect judge; the 3000-token budget truncates some hard solves
  (counted as unjudged, not wrong).
- All on Qwen3-8B-Base. Data: `paper_data/pseudo_label_periter.jsonl`.

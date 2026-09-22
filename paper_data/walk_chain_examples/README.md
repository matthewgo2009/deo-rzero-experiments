# Successful DEO mutation chains (paper examples)

Source: the instrumented canonical DEO run `mem-4b-deo` (placid_crayon) — Qwen3-4B,
mutV1 recipe (fixed beta=0.1, no CD, V1 mutation prompt), iteration 1, full walk log
(3181 proposals over 2000 chains x 5 sweeps). Walk logs from the ORIGINAL mutV1 run
(icy_sprout) predate the log-persistence fix and were not recoverable; this run uses
the identical recipe.

`successful_chains_mem4b_iter1.json`: the 62 chains with >=2 ACCEPTED mutations and
a seed->final tent-uncertainty gain >= 0.5. Per chain:
- `seed` / `final` question, `r_unc_seed` -> `r_unc_final` (tent uncertainty
  1-2|p_hat-0.5|; 0.89 corresponds to p_hat ~ 4/9 or 5/9 with m=9 rollouts)
- `final_p_hat`, `final_pseudo_label`, `final_in_band` (p_hat in [0.3, 0.8]) joined
  from the end-of-iteration pool
- `events`: EVERY proposal on the chain (accepted and rejected) with step, mutation
  strategy tag (A-E; '?' = tag not echoed), old/new question text, r_unc transition.

Selection criteria are mechanical (accept count + r_unc gain), NOT hand-curated for
question quality — some accepted intermediates/finals contain flawed statements
(e.g. over-determined constraints). For the paper, pick chains where the final
question is also mathematically clean; candidates spotted during export:
- chain 660: complex-modulus seed -> B -> A -> C, r_unc 0.22 -> 0.89, ends as a
  Vieta-style system with a modulus constraint
- chain 394: incircle/circumcircle triangle seed -> inscribed equilateral ->
  inscribed tetrahedron (dimension lift), r_unc 0.22 -> 0.89
- chain 1407: circle geometry seed -> hyperbola family with parameters,
  r_unc 0.00 -> 0.89
Distribution: accepts per chain over the full walk: 4x accepts: 5 chains,
3x: 59, 2x: 361, 1x: 868 (of 2000).

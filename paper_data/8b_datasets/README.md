# 8B per-iteration training datasets (baseline DEO / strong-control DEO / R-Zero)

Per-iteration training data for the three Qwen3-8B-Base self-evolving runs, for external
(e.g. GPT) analysis of question difficulty / label quality / drift across iterations.

Source runs (AzureML):
- **baseline** = `qwen8b-baseline-v2` (DEO, no β control) — output `yyd_8b2_baseline`
- **strong**   = `qwen8b-adaptive-strongctrl` (DEO, adaptive β, band[0.4,1.0] δ=0.1 η_λ=2 λ0=10) — `yyd_8b_adaptive_strong`
- **rzero**    = `qwen8b-rzero-v2` (R-Zero) — HF Hub `yuyang322/qwen3-8b-base-rzero_solver_v{1..5}`

Files: `{run}/{run}_iter_{1..5}.json`, one JSON array per iteration.

## ⚠️ Stage caveat — these are NOT the same stage

- **DEO (baseline / strong)** = the **full MCMC pool, PRE-filter**, N=1500/iter. Training only
  uses the subset with `p_hat ∈ [0.3, 0.8]` (~700–1180/iter); the rest is included here so you can
  see the whole difficulty distribution the sampler produced.
- **R-Zero** = the **POST-filter training set** actually pushed to the solver, already restricted to
  `score ∈ [0.3, 0.8]`, N≈3946–4564/iter. R-Zero's raw pre-filter pool is not persisted.

So to compare like-for-like, first restrict the DEO files to `p_hat ∈ [0.3, 0.8]`.

## Schemas

**DEO (baseline / strong)** — one record per candidate question:
| field | meaning |
|--|--|
| `question` | the question text (MCMC-sampled / mutated from the base model) |
| `gt` | the challenger's proposed answer for the question |
| `p_hat` | solver self-consistency = modal-answer count / m over m=9 samples (temp 1.0); the difficulty measure. p̂≈0.5 = maximally uncertain, low p̂ = solver all-over-the-place |
| `pseudo_label` | the solver's modal answer (the pseudo-label used for GRPO if kept) |
| `r_unc` | uncertainty reward = 1 − 2·|p̂ − 0.5| ∈ [0,1] |

**R-Zero** — one record per (already-filtered) training question:
| field | meaning |
|--|--|
| `problem` | the question text (from the trained questioner) |
| `answer` | the questioner's boxed answer (used as the label) |
| `score` | = p̂ (majority-vote fraction, `max_count/len`); same quantity as DEO `p_hat` |

Field correspondence: DEO `question` ↔ R-Zero `problem`; DEO `p_hat` ↔ R-Zero `score`.
Label note: DEO `pseudo_label` is the **solver's** modal answer; R-Zero `answer` is the
**questioner's** self-proposed answer — the pseudo-label is generated differently in the two methods.

## Known summary stats (see ../difficulty_phat_stats.txt, ../../DIFFICULTY_DISTRIBUTION.md)

Mean p̂ within the training band [0.3,0.8], per iter v1..v5:
- baseline: 0.514 0.517 0.511 0.517 0.514
- strong:   0.459 0.473 0.464 0.463 0.463  (hardest in-band)
- rzero:    0.524 0.515 0.515 0.517 0.527  (easiest by p̂, but worst label-correctness — see ../../PSEUDO_LABEL_QUALITY.md)

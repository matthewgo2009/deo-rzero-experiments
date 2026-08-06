# Question samples: R-Zero vs DEO-MCMC (for external analysis)

Readable samples of the self-generated training questions from R-Zero and DEO, for GPT/qualitative
analysis of question style, difficulty, correctness, and drift across iterations.

## Files (each = 5 iterations, ~40 evenly-spaced questions/iter)

| file | method | model | notes |
|--|--|--|--|
| `DEO_8B_strong.md` | DEO MCMC walk | Qwen3-8B | strong-β, m=9, **no CD**, 1500-q pool |
| `DEO_4B_heroic_eye.md` | DEO MCMC walk | Qwen3-4B | full stack: CD n=12 + strong-β + KL-prev, 2000-q pool |
| `RZERO_8B.md` | R-Zero trained questioner | Qwen3-8B | post-filter training set |
| `RZERO_4B.md` | R-Zero trained questioner | Qwen3-4B | post-filter training set |

## Field meaning

- **q** — the question text.
- **label** — the pseudo-label used for solver training. DEO: the solver's modal answer over m samples.
  R-Zero: the questioner's own boxed answer.
- **gt (challenger)** — DEO only: the challenger/proposer's proposed answer for the question.
- **p̂** — self-consistency = modal-answer count / m (m=9 for 8B/no-CD, m=12 for 4B/CD). R-Zero's `score`
  is exactly this quantity. Difficulty proxy: p̂≈0.5 = maximally uncertain; low p̂ = solver all over the place.
- **r_unc** — DEO only: uncertainty reward = 1 − 2·|p̂ − 0.5| ∈ [0,1] (the MCMC energy term).

## Important sampling / stage caveats

- **DEO files = the full pre-filter MCMC pool** (N=1500 or 2000), so they include questions outside the
  p̂∈[0.3,0.8] training band. R-Zero files = the **post-filter training set** (already p̂∈[0.3,0.8]),
  N≈3900–6300. To compare like-for-like, restrict the DEO samples to p̂∈[0.3,0.8].
- Samples are evenly spaced by index (deterministic), not random — they reflect generation order.
- Full pools (all questions, all fields) are in `paper_data/8b_datasets/` (8B) and reproducible from the
  job outputs / HF Hub `yuyang322/qwen3-{4,8}b-base-rzero_solver_v{1..5}` for R-Zero.

## Context (accuracy, same Claude grader)

- 4B: DEO (heroic_eye) 7-set peak 48.94 ≳ R-Zero 48.13; R-Zero stalls at iter2.
- 8B: R-Zero 54.57 > best DEO 53.01; R-Zero keeps climbing to iter5.
- See `../../RESULTS_4B_FULLSTACK.md`, `../../PSEUDO_LABEL_QUALITY.md`, `../../DIFFICULTY_DISTRIBUTION.md`.

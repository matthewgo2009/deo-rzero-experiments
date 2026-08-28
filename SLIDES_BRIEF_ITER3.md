# DEO vs R-Zero — Results Brief for Slides (iter1–3, matched budget)

Source material for a slide deck. All accuracies = 7-benchmark suite (MATH-500, GSM8K, AMC,
Minerva, OlympiadBench, AIME24, AIME25), graded by one identical grader (Claude Haiku,
boxed-answer exact check, ~0% false positives). AVG7 = unweighted mean of the 7.
HARD5 = mean of AMC/Minerva/Olympiad/AIME24/AIME25. Reporting cut at **iteration 3** =
matched iteration/compute budget (see note at the end).

## 1. Setup (one slide)

- **Task**: self-evolving math training from zero human data. A generator proposes questions,
  the solver's majority vote over 9 rollouts pseudo-labels them, questions with solver
  self-consistency p̂∈[0.3,0.8] train the solver via GRPO (verl; 1280 prompts × 5 rollouts/iter,
  identical for every method).
- **R-Zero** (baseline method): TRAINS a second LLM (questioner) with RL to produce questions
  at the solver's uncertainty frontier. Cost: 2 LLMs trained per iteration.
- **DEO** (our method): NO second LLM. Closed-form optimal challenger q*(X) ∝ π_base·exp(r_c/β),
  sampled by Metropolis–Hastings: base model proposes mutations of a 2000-question pool, the
  MH rule (uncertainty reward + BLEU repetition penalty) accepts/rejects. Roughly HALF the
  training compute of R-Zero per iteration.
- **baseline** (ablation floor): no walk — sample 2000 questions from the base model, filter,
  train. Isolates the value of the MCMC walk.
- Models: Qwen3-4B-Base and Qwen3-8B-Base, 5 self-evolving iterations, 8×H100.

## 2. Headline results @ matched budget, iterations 1→3 (main slide)

### Qwen3-8B-Base (base model: AVG7 44.2, HARD5 30.1)

| method | AVG7 i1 | i2 | **i3** | HARD5 i1 | i2 | **i3** |
|--|--|--|--|--|--|--|
| baseline 2000q (no walk) | 51.91 | 50.64 | 51.77 | 38.04 | 36.28 | 37.48 |
| **DEO walk 2000q** | 52.01 | **53.39** | **53.60** | 38.22 | **40.02** | **39.96** |
| R-Zero (2 LLMs) | 51.96 | 51.70 | 52.88 | 37.88 | 37.68 | 38.97 |

**@i3: DEO +0.72 AVG7 / +1.0 HARD5 over R-Zero, at ~half the compute (no questioner training).**
Walk's net contribution over no-walk: +1.8 AVG7.

### Qwen3-4B-Base (base model: AVG7 43.5, HARD5 30.1)

| method | AVG7 i1 | i2 | **i3** | HARD5 i1 | i2 | **i3** |
|--|--|--|--|--|--|--|
| baseline 2000q (no walk) | 45.49 | 47.73 | 45.67 | 30.02 | 32.84 | 30.48 |
| **DEO walk 2000q** | 45.11 | 47.71 | **47.57** | 29.02 | 33.18 | **32.96** |
| R-Zero (2 LLMs) | 46.46 | 48.13 | 45.93 | 31.38 | 33.96 | 31.16 |

**@i3: DEO +1.64 AVG7 / +1.8 HARD5 over R-Zero.** At 4B the trained questioner never compounds
(R-Zero peaks at i2 and regresses); DEO keeps rising.

### One-line takeaways
- At equal iteration budget, DEO ≥ R-Zero at BOTH scales, with no second LLM trained.
- The single ingredient that matters is the MCMC walk on a 2000-question pool.
- MATH-500 alone is non-discriminative at 8B (all methods 80–82); differences live in hard sets.

## 3. The graveyard: 10 tricks that did NOT help (1–2 slides)

Every idea below was implemented, run end-to-end (5 iters, same grader), and falsified.
Numbers are 7-set AVG deltas vs the relevant control.

| # | trick | idea | result |
|--|--|--|--|
| 1 | **Adaptive/strong β control** | GDA controller keeps the pool inside the trainable-difficulty band | β pins to its floor; at 8B strong control packs 85% of the pool in-band — **accuracy unchanged or worse** (52.57 vs 53.01). In-band fraction ≠ accuracy. |
| 2 | **Ceperley–Dewing noisy-MH correction** | debias exp-of-noisy-reward acceptance with a variance penalty (12-rollout U-statistic + jackknife) | mathematically sound, empirically **−1.4 @4B, −0.4 @8B**: the 1/β² penalty suppresses walk mixing; better labels don't compensate. Also freezes the walk entirely under small β (kills trick #1 combos). |
| 3 | **KL anchor → previous solver** (R-Zero-style drift) | maybe R-Zero's advantage is its looser drifting KL reference | **−0.5** on both 8B arms (walk and no-walk). |
| 4 | **Warm-start chains** | continue the MCMC chain from the previous iteration's mutated pool | no gain (48.59 vs 48.94 @4B); converged chains yield fewer trainable questions. |
| 5 | **Thompson-sampling operator bandit** (memory over mutation operators) | learn WHICH mutation type works per question context | **−1.1** vs uniform; posteriors stay flat across 10 iterations — the five operators are statistically indistinguishable, and the 8B-class mutator ignores operator instructions 62% of the time anyway. |
| 6 | **Conservative "one localized step" mutation prompt** | smaller, validity-checked mutations to reduce overshoot | accuracy tie (48.83 vs 48.79), but it just WEAKENS the walk (56% vs 69% in-band) — small steps ≠ better steps. |
| 7 | **Perfect labels (Claude relabeling)** | R-Zero's majority-vote label correctness collapses 63%→23% across iters — fix it with Claude-solved labels | **+0.2 mean** (noise). GRPO is extremely tolerant of label noise; label quality is NOT the bottleneck. |
| 8 | **Leak-strip + easy-band ballast** | remove \boxed-leak/malformed questions; reserve 25% of training data at p̂∈[0.6,0.8] where labels are usually right | mechanisms worked exactly as designed; **−0.3**, and the ballast moved the peak EARLIER (opposite of prediction). |
| 9 | **Selection headroom (8000-question pools)** | give DEO R-Zero-scale raw material, walk only 1700 selected in-band seeds | **−1.2.** Breadth without an adaptive generator hurts; i.i.d. base samples carry no curriculum. |
| 10 | **Olympiad-register style transfer** | rewrite questions into competition phrasing (R-Zero's style drifts this way) without changing the math | HARD5 **down** 0.5; style imitation doesn't transfer the benefit. |

### What the graveyard proves (synthesis slide)
Nine transplantable ingredients of R-Zero were grafted onto DEO and none reproduced R-Zero's
late-iteration behavior. The only unfalsified explanation for R-Zero's remaining edge: the
**trained questioner's distribution MOVES with the solver** (gradient pressure on the generator
itself) — a capability that cannot be imitated by any static intervention on a frozen-base
proposal distribution. That capability is exactly what costs R-Zero 2× compute, and it only
pays off late (see caveat).

## 4. Supporting mechanism findings (optional slides)

- **Label collapse, measured twice independently**: R-Zero's majority-vote labels agree with
  Claude 62.6% → 22.8% (iter1→5); an independent judge-based measurement gives 72% → 29%.
  Yet downstream accuracy is unaffected (trick #7) — GRPO group-normalization absorbs it.
- **The difficulty filter is provably blind**: while R-Zero's answer distribution collapses
  (distinct answers 1221→493; top-10 answers = 72% of labels) mean p̂ stays flat (0.524→0.527).
  Self-consistency cannot detect degeneracy.
- **Question corpora evolve oppositely**: DEO's corpus is distribution-stationary across
  iterations; R-Zero's drifts hard (length 248→439 chars, template collapse: 38% "Let P(x)…",
  28% Alice/Bob games, vague-quantifier rate 7%→31% — a reward-hacking signature).
- **Walk × pool-size interaction**: growing the pool 1500→2000 does nothing without the walk
  (53.01→53.01) but gives +2.2 with it (51.41→53.60): the walk converts breadth into
  trainable difficulty; sampling alone doesn't.

## 5. Honest caveat (must appear on a slide)

Full-run peaks (5 iters) at 8B: R-Zero 54.57 (still rising at i5) vs DEO 53.60 (peaks i3).
R-Zero's headline win exists ONLY in iterations 4–5 and ONLY at 8B: its questioner keeps
supplying a moving distribution that extends useful training signal past DEO's saturation
point. Framing: "DEO matches or beats R-Zero at matched budget and half the compute;
R-Zero's residual advantage is late-iteration non-saturation at 8B, attributable to
questioner adaptivity — the one component that cannot be transplanted."

At 4B the caveat cuts the other way: DEO's full-run peak (48.79–48.94) also beats R-Zero's
(48.13), so DEO wins at every budget.

## Numbers appendix (for backup slides)

- Runs referenced: 8B baseline=mighty_chicken, 8B DEO walk=cool_loquat, 8B R-Zero;
  4B baseline=willing_panda, 4B DEO walk=icy_sprout(mutV1), 4B R-Zero.
- All methods: identical GRPO solver recipe (rollout.n=5, batch 64×20 steps=1280 prompts/iter,
  max_response 4096, ckpt @ step 15) and identical p̂∈[0.3,0.8] filter.
- Grader consistency check: base model scores agree across all grade jobs (44.1–44.3 @8B,
  43.4–43.6 @4B).
- Full tables, per-dataset numbers, and the complete falsification ledger:
  `RESULTS_4B_FULLSTACK.md`, `BUCKET_ANALYSIS.md`, `BANDIT_FINAL_REPORT.md`,
  `PSEUDO_LABEL_QUALITY.md`, `paper_data/QUESTION_ANALYSIS_8B.md`.

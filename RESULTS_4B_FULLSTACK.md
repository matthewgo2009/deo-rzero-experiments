# 4B full-stack DEO: walk / CD / strong-β / KL-prev / 2000q / warm-start

Qwen3-4B-Base. Three matched 2000-question/iter runs isolating the contribution of the
MCMC machinery over a plain no-walk baseline, plus a warm-start variant of the full stack.

| run | job | walk | CD | β control | KL anchor | init pool |
|--|--|--|--|--|--|--|
| **heroic_eye** | `heroic_eye_27t7rwjp20` | yes | yes (U-stat n=12) | strong (band[0.4,1.0], δ=0.1, η_λ=2, λ0=10) | prev-iter solver | fresh from base each iter (cold) |
| **witty_soca** | `witty_soca_n1qkh3yjpp` | yes | yes | strong | prev-iter solver | **warm start** = prev-iter mutated pool (re-scored) |
| **willing_panda** | `willing_panda_k1zb5m1s59` | **no** (`MCMC_STEPS=0`) | no | none | base (fixed) | fresh from base each iter |

All three: 2000-q pool/iter, same p̂∈[0.3,0.8]+judge filter, verl GRPO consumes 1280 prompt-instances/iter.

## MATH-500 (in-run eval, raw — OpenAI quota dead so +0 GPT bump)

| run | base | iter1 | iter2 | iter3 | iter4 | iter5 | peak |
|--|--|--|--|--|--|--|--|
| heroic_eye (full, cold) | 58.0 | 62.6 | 62.8 | 64.2 | **64.6** | 62.4 | **64.6** @i4 |
| witty_soca (full, warm) | 58.0 | 61.8 | 62.4 | 63.6 | 62.0 | 64.2 | 64.2 @i5 |
| willing_panda (baseline, no walk) | 58.0 | 62.2 | 63.6 | 61.4 | 62.2 | 61.4 | 63.6 @i2 |

- **Full stack > no-walk baseline by ~1pt** (64.6 vs 63.6): the MCMC walk + CD + strong-β together add a small but consistent gain over a plain filtered base-sample pool.
- **Warm-start does not help** (64.2 ≤ 64.6): reusing the prev-iter mutated pool matches/slightly-trails a cold restart; peak just shifts later (i5).

## 7-set accuracy (Claude Haiku boxed grader = "ours")

### heroic_eye — DONE (job `gifted_collar_hfb2bbjpb0`)

| ckpt | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | **AVG** |
|--|--|--|--|--|--|--|--|--|
| base | 72.4 | 81.8 | 48.1 | 36.4 | 39.1 | 10.0 | 16.6 | 43.49 |
| iter1 | 77.6 | 91.7 | 55.2 | 44.5 | 41.8 | 10.0 | 10.0 | 47.26 |
| iter2 | 78.0 | 92.0 | 49.8 | 46.3 | 40.6 | 16.7 | 10.0 | 47.63 |
| iter3 | 79.0 | 92.1 | 50.4 | 50.0 | 41.3 | 16.7 | 3.4 | 47.56 |
| **iter4** | 79.4 | 92.3 | 64.6 | 47.4 | 42.2 | 10.0 | 6.7 | **48.94** |
| iter5 | 75.8 | 91.7 | 57.5 | 47.8 | 41.9 | 6.7 | 6.6 | 46.86 |

### witty_soca (warm-start) — DONE (job `helpful_watch_7qyr66x33w`)

| ckpt | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | **AVG** |
|--|--|--|--|--|--|--|--|--|
| iter1 | 76.8 | 91.7 | 52.9 | 47.1 | 42.7 | 6.7 | 3.3 | 45.89 |
| iter2 | 76.6 | 91.6 | 52.6 | 48.5 | 40.6 | 6.7 | 6.8 | 46.20 |
| **iter3** | 78.4 | 92.2 | 55.0 | 51.1 | 40.0 | 16.7 | 6.7 | **48.59** |
| iter4 | 76.4 | 92.0 | 54.9 | 48.5 | 40.6 | 13.3 | 10.0 | 47.96 |
| iter5 | 78.2 | 91.7 | 54.9 | 51.5 | 41.6 | 10.0 | 3.3 | 47.31 |

### willing_panda (baseline 2000q, no walk) — DONE (job `gentle_hook_k8g0sxb6s2`)

| ckpt | math | gsm8k | amc | minerva | olympiad | aime24 | aime25 | **AVG** |
|--|--|--|--|--|--|--|--|--|
| iter1 | 76.2 | 92.2 | 50.1 | 43.4 | 42.7 | 6.7 | 7.1 | 45.49 |
| iter2 | 78.0 | 91.9 | 56.6 | 48.5 | 42.2 | 10.2 | 6.7 | 47.73 |
| iter3 | 75.8 | 91.5 | 52.6 | 47.4 | 39.1 | 10.0 | 3.3 | 45.67 |
| **iter4** | 75.6 | 91.7 | 52.5 | 45.2 | 41.9 | 10.1 | 17.3 | **47.76** |
| iter5 | 75.0 | 91.9 | 50.2 | 48.2 | 42.7 | 10.0 | 3.3 | 45.90 |

### Four-way, same grader (Claude Haiku boxed) — the clean comparison (incl. R-Zero 4B)

| run | config | 7-set AVG peak | @iter | HARD5 pk | COMP4 pk |
|--|--|--|--|--|--|
| **heroic_eye** | full stack, cold | **48.94** | i4 | **34.18** | **30.88** |
| witty_soca | full stack, **warm-start** | 48.59 | i3 | 33.90 | 29.60 |
| **R-Zero 4B** | trained questioner | 48.13 | **i2** | 33.96 | 29.68 |
| willing_panda | **baseline, no walk** | 47.76 | i4 | 33.40 | 30.45 |

(base ≈ 43.6 for all four → grader consistent. HARD5 = amc/minerva/olympiad/aime24/aime25 ; COMP4 drops minerva.)

- **DEO full stack ≳ R-Zero at 4B** (48.94 vs 48.13 AVG, +0.8): hard sets are ~tied (HARD5 34.18 vs 33.96),
  and on competition-only COMP4 R-Zero is actually the lowest (29.68 < full-stack 30.88 < no-walk 30.45).
- **R-Zero 4B peaks early (i2) then plateaus** ~47–48; DEO variants peak i3–i4.

### Cross-scale (same Claude grader both ends) — the ranking FLIPS

| | 4B best DEO | 4B R-Zero | 8B best DEO | 8B R-Zero |
|--|--|--|--|--|
| 7-set AVG peak | **48.94** (full stack) | 48.13 | 53.01 (baseline) | **54.57** |
| HARD5 peak | **34.18** | 33.96 | 39.62 | **41.36** |
| R-Zero peak iter | — | i2 (early) | — | i5 (still rising) |

At 4B, DEO (esp. full stack) edges R-Zero and R-Zero stalls at i2; at 8B, R-Zero clearly wins (+1.5 AVG,
+1.7 HARD5) and keeps climbing to i5. The trained questioner only starts compounding at 8B scale.
(Caveat: the 8B runs are 1500-q, no-CD; 8B has no full-stack run yet.)

- **Full stack beats the no-walk baseline by ~1.2pt** (48.94 vs 47.76; mean 47.65 vs 46.51) — under the
  same Claude grader, so this is the cleanest statement of the MCMC machinery's contribution.
- **Warm-start gives no benefit** (48.59 < 48.94; mean 47.19 < 47.65); peak just shifts to i3.
- Baseline (no walk) is the jaggiest (45.5–47.8 sawtooth); the full stack is more stable across iters.

## 7-set peak vs prior 4B runs

| run | config | grader | 7-set AVG peak | HARD5 peak | COMP4 peak |
|--|--|--|--|--|--|
| **heroic_eye** | full stack, 2000q | Claude-box | **48.94** (i4) | **34.2** (i4) | 30.9 (i4) |
| baseline (old) | no walk, 1500q | GPT-box | 48.18 (i1) | 33.7 (i1) | **31.3** (i1) |
| adaptive (old) | weak β, 1500q | GPT-box | 48.15 (i5) | 33.8 (i5) | 30.6 (i5) |
| curriculum (old) | 1500q | GPT-box | 47.59 (i4) | 33.2 (i4) | 29.7 (i4) |
| fixedbeta (old) | 1500q | GPT-box | 47.08 (i4) | 32.3 (i4) | 29.3 (i4) |

HARD5 = amc/minerva/olympiad/aime24/aime25 ; COMP4 = amc/olympiad/aime24/aime25 (no minerva).
Grader note: heroic_eye = Claude Haiku boxed, others = GPT-4o-mini boxed; both are boxed-only ~0% FP
and agree on the base (~43 AVG / ~29.5 HARD5), so the cross-grader comparison is fair.

**Read:** full stack is best on total AVG (+0.7 over old baseline) and HARD5 (+0.4), but the hard-set
edge is entirely AMC + Minerva (heroic_eye's amc i4=64.6 and minerva i3=50.0 are the single highest cells
of any run). On competition-only COMP4 it's a wash (old baseline i1=31.3 ahead). Olympiad flat ~40-42
everywhere; AIME is noise at 4B (3–17%, ~1 problem per AVG point). Pushing questions harder (strong-β)
did not convert into competition-problem accuracy.

## Per-iter questions entering solver training (pool=2000)

| iter | willing_panda (no walk) | witty_soca (warm) | heroic_eye (cold) |
|--|--|--|--|
| 1 | 977 | 793 | 737 |
| 2 | 913 | 741 | 904 |
| 3 | 1002 | 785 | 899 |
| 4 | 941 | 741 | 894 |
| 5 | 939 | 781 | 857 |
| mean | **954** | **768** | 858 |

- **No-walk baseline yields the most (~950)** — base-sample pool has the highest in-band fraction, i.i.d. each iter.
- **Warm-start yields the fewest & flattest (~770)** — the chain has converged near the hard-region edge, so re-scored in-band fraction is lower and barely moves iter-to-iter.
- **Cold full-stack jumps i1→i2** (737→904) as β hits its 0.02 floor and packs more in-band.
- Training-set size (768–954) has **no monotone relation** to MATH-500 peak (64.2/64.6/63.6): the run with the most questions (baseline) is the lowest. Consistent with the earlier "quantity ≠ performance" finding.

### R-Zero 4B, for reference (HF Hub filtered training sets)

| iter | v1 | v2 | v3 | v4 | v5 |
|--|--|--|--|--|--|
| N | 3939 | 4312 | 4376 | 5387 | 6270 |
| mean p̂ | 0.529 | 0.520 | 0.509 | 0.510 | 0.528 |

R-Zero generates 4–8× more questions and grows each iter, but its solver GRPO also consumes only
1280 prompt-instances/iter → it trains on ~20–33% of its pool (<1 epoch), while DEO cycles its
~800–950 questions ~1.4–1.7 epochs. The larger pool is not converted into more gradient updates.

## Adaptive-β controller trajectory (heroic_eye / witty_soca, identical)

β pinned to floor 0.02 from iter1; λ climbs monotonically 10→23→35→47→59→71; in-band ~0.57–0.61;
violation ~0.39–0.43 (never reaches cap δ=0.1 — controller keeps pushing but the pool can't get more
in-band). CD U-statistic acceptance (m=9, n=12) ran every iter.

## Takeaway

The full MCMC stack (walk + CD debias + strong-β + KL-prev) is the best 4B run to date (MATH-500 64.6,
7-set 48.94, HARD5 34.2). Under the same Claude grader it beats a plain no-walk 2000q baseline by
~1.2pt 7-set (48.94 vs 47.76; mean 47.65 vs 46.51) — real but modest. Warm-starting the chain from the
previous iteration's pool gives no benefit (48.59). Gains trace to cleaner labels + the walk finding
useful (mostly AMC/Minerva-type) questions, not to difficulty control or pool size — consistent with
`BUCKET_ANALYSIS.md`, where CD's 1/β² variance penalty actually freezes the strong-β walk at the β floor,
so the "strong control" concentration never materializes yet accuracy is still highest.

## Question-distribution & label interventions — both falsified (Aug 2026)

Two intervention experiments testing the QUESTION_ANALYSIS_8B.md hypotheses (Claude grader):

**R-Zero 4B + Claude labels** (`frank_fork`, RZ_LABEL=claude, 2000/iter relabeled by Sonnet):
Claude-vs-majority-vote agreement fell 62.6%→22.8% across iters (independent reproduction of the
label collapse). Downstream: AVG7 peak 48.90 @i3 vs original 48.13 @i2 (+0.77 peak, +0.18 mean),
HARD5 34.64. Ties best-4B heroic_eye (48.94) but the mean gain is noise-level → **fixing labels
buys almost nothing; GRPO's label-noise tolerance confirmed.**

**DEO 8B fix12** (`polite_neck`: walk 2000q + CD + β=0.1 + KL base + leak-strip + 25% easy-band
ballast): mechanisms worked exactly as designed (upstream gates kept the pool leak-free; ballast
share 0.25 every iter; training sets 812–972 after reshaping). Accuracy:

| run | AVG7 per iter | peak |
|--|--|--|
| fix12 | 53.29 53.07 52.09 52.19 51.50 | 53.29 @i1 |
| cool_loquat (no CD, no fixes) | 52.01 53.39 53.60 51.60 51.17 | **53.60** @i3 |
| coral_sail (CD+KLprev, no fixes) | 53.13 52.30 50.77 52.16 – | 53.13 @i1 |

Prediction A (leak-strip +0.3–0.7 AVG): **failed** (−0.31 vs cool_loquat; +0.16 vs the CD-matched
control = noise; the deficit matches the known −0.5 CD effect). Prediction B (ballast moves the
peak past iter3): **failed in the opposite direction** (peak at i1, monotone decline; possibly
worsened by the ballast shrinking the training set ~40% → more repetition).

**Hypothesis ledger for R-Zero's 8B lead (54.57 vs best DEO 53.60):** labels ✗ (claudelabel),
mechanical leaks ✗, easy-band ballast ✗, KL anchor ✗, β control ✗, warm-start ✗ (4B), operator
bandit ✗ (4B), conservative mutations ✗. Remaining untested: (i) olympiad-register style transfer
(QUESTION_ANALYSIS rec #3), (ii) selection headroom / pool scale (8000-q + more verl steps).

## Hypothesis ledger CLOSED (Aug 2026): headroom and style falsified — all nine down

**Selection headroom** (`icy_arch`: generate 8000/iter, walk 1700 random in-band seeds; raw
material matches R-Zero's ~4000 in-band): AVG7 peak **52.43** @i1 declining, HARD5 38.54 —
**worse than the plain 2000-pool walk (53.60/40.02) by −1.2**. Random breadth without an adaptive
generator hurts: pre-in-band seeds neutralize the walk's job and i.i.d. base samples carry no
curriculum.

**Olympiad-register style transfer** (`upbeat_turnip`: DEO_STYLE_P=0.5, [F] rewrite operator,
math preserved, fresh labels): AVG7 peak 53.27 @i1, HARD5 peak 39.50 — prediction (HARD5 ≥40.5)
**failed**; slightly below control. (In-run MATH-500 i4/i5 "drop" was an OpenAI-quota grader
artifact; raw flat 64.4–65.8.)

**Final conclusion of the ablation program.** R-Zero's 8B edge (54.57 vs 53.60 AVG7, 41.36 vs
40.02 HARD5) survives the falsification of ALL nine transplantable ingredients: label quality,
mechanical leaks, easy-band ballast, KL anchor, β control, warm-start, operator bandit,
conservative mutations, selection headroom, and style register. The only unfalsified explanation
is the adaptive questioner itself — a generator whose distribution MOVES with the solver via its
own gradient updates, which no static intervention on a frozen-base proposal distribution
reproduced. Conversely DEO reaches 98% of R-Zero's 8B AVG at roughly half the compute (no second
LLM trained) and beats it outright at 4B (48.94 vs 48.13), where the questioner fails to compound.

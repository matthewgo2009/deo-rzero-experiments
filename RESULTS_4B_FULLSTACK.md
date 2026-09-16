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

## 8B fixed-iteration comparison @ iter3 (Claude grader) — the lead is about saturation, not strength

| 8B run | AVG7 @i3 | HARD5 @i3 |
|--|--|--|
| **DEO walk 2000q** (cool_loquat) | **53.60** | **39.96** |
| baseline 1500q | 53.01 | 39.62 |
| style (upbeat_turnip) | 52.90 | 39.46 |
| **R-Zero** | 52.88 | 38.97 |
| strong-β 1500q | 52.57 | 38.69 |
| fix12 (polite_neck) | 52.09 | 38.24 |
| adaptive 1500q | 52.07 | 38.15 |
| baseline 2000q (mighty_chicken) | 51.77 | 37.48 |
| fixedbeta-walk 1500q | 51.41 | 36.96 |
| baseline+KLprev+CD (olive_pizza) | 51.37 | 37.00 |
| biginit (icy_arch) | 50.79 | 36.34 |
| walk+KLprev+CD (coral_sail) | 50.77 | 36.16 |

At equal iteration budget (3 iters) DEO walk LEADS R-Zero by +0.72 AVG7 / +1.0 HARD5. R-Zero's
headline lead comes entirely from iters 4–5 (52.88 → 53.21 → 54.57) while every DEO variant
saturates around iter3 and regresses. The precise statement is therefore not "R-Zero is stronger"
but "R-Zero does not saturate": its trained questioner keeps supplying a moving distribution that
extends the useful training signal past iter3, which none of the nine transplanted ingredients
could reproduce on DEO's frozen-base proposal distribution. Under a matched-compute lens the gap
widens further in DEO's favor: at iter3 DEO has spent roughly half of R-Zero's compute (no
questioner training) and is already ahead.

## 4B fixed-iteration comparison @ iter3 (Claude grader)

| 4B run | AVG7 @i3 | HARD5 @i3 | AVG7 peak | @iter |
|--|--|--|--|--|
| R-Zero + Claude labels | **48.90** | **34.64** | 48.90 | i3 |
| witty_soca (full stack + warm) | 48.59 | 33.90 | 48.59 | i3 |
| mutV2 (conservative walk) | 47.77 | 33.04 | 48.83 | i5 |
| mutV1 (walk) | 47.57 | 32.96 | 48.79 | i5 |
| heroic_eye (full stack) | 47.56 | 32.36 | **48.94** | i4 |
| bandit r2 | 45.94 | 31.00 | 47.73 | i5 |
| R-Zero (original) | 45.93 | 31.16 | 48.13 | i2 |
| willing_panda (no walk) | 45.67 | 30.48 | 47.76 | i4 |

Unlike 8B, the 4B iter3 slice has no saturation story: peak iterations scatter i2–i5 and the
slice ranking disagrees with the peak ranking — per-iter fluctuation (±1–1.5) is noise-dominated
at this scale, so any single-slice ordering should be read cautiously. Two facts survive any
slice: original R-Zero never leads at 4B (i3 45.93; peak 48.13 < full-stack 48.94), and at equal
iteration budget DEO is ≥ R-Zero at BOTH scales (4B +1.6 @i3, 8B +0.72 @i3) — R-Zero's advantage
exists only in 8B late iterations.

## 4B CD × pool grid closed (nice_nail, Aug 2026)

`nice_nail` (4B, 2000q + CD n=12 + fixed β=0.1, KL base) — Claude 7-set: 46.96 / 47.13 / 46.51 /
46.40 / **47.40** (peak i5), HARD5 pk 32.42, i3 slice 46.51/31.24.

| pool | no CD | CD |
|--|--|--|
| 1500 | (old fixedbeta, GPT grader: 47.08) | deo_cd **48.18** |
| 2000 | mutV1 **48.79** | nice_nail **47.40** |

- **CD on an active fixed-β walk is clearly negative at 4B: −1.39** (48.79 → 47.40 at 2000q) —
  same sign as 8B (−0.3/−0.5), larger magnitude. CD's variance penalty suppresses walk mixing
  even at β=0.1, and its better 12-sample labels do not compensate.
- Pool 1500→2000 on the CD arm: 48.18 → 47.40 (−0.78, within 4B noise but not positive) — the
  walk×pool synergy seen at 8B (+2.2, no-CD) does not appear under CD.
- Re-attribution for heroic_eye (48.94 = 2000q+CD+strong-β+KL-prev): with CD now shown to be
  −1.4 on an active walk, heroic_eye's edge cannot come from CD-on-walk; its strong-β froze the
  walk (BUCKET_ANALYSIS), making it effectively a no-walk run with 12-sample labels + KL-prev —
  vs willing_panda (no walk, 9-sample, KL base) 47.76, i.e. +1.2 from that combination. 4B
  per-run noise is ±1, so treat single-cell readings cautiously; the robust statement is that
  **no CD configuration beats the plain V1 walk (48.79) at 4B**.

## Graveyard #11 — longer chains (MCMC_STEPS 5→50): falsified at iter1, stopped early

`icy_kitchen` (4B, 2000q, fixed β=0.1, no CD, KL base — identical to mutV1 except
DEO_MCMC_STEPS=50, i.e. 100k proposals/iter). Iter1 outcome vs the 5-step control:

| | walk-50 | walk-5 (mutV1) |
|--|--|--|
| in-band fraction (pre-judge) | **92%** (1847/2000) | ~69% |
| LLM-judge drop rate | **15%** (275) | 1–3% |
| MATH-500 iter1 (raw) | **61.2** | 64.0 |

Deeper mixing "succeeds" on the MCMC objective (in-band beyond even the 8B strong-β collapse's
85%) yet accuracy DROPS 2.8 raw and judge rejections spike 5× — long mutation chains accumulate
malformed/degenerate questions. Run cancelled after iter1 (saving ~7 days of H100); the point
completes the frozen-distribution ceiling argument: walking FARTHER on the frozen base
distribution over-concentrates and degrades quality, same as every other static intervention.
(In-band ≠ accuracy, now demonstrated on the chain-length axis too.)

## Graveyard #12 — weakness memory (two attempts, mechanism-verified null) (Sep 2026)

Hypothesis: give the MCMC walk a cross-iteration memory of SOLVER WEAKNESSES (one
LLM-written note per evaluated proposal from answer-cluster disagreement → ≤10 global
weaknesses per iter → next iter 80% of chains guided toward one sampled weakness),
so the frozen-base proposal distribution "moves with the solver" without training a
questioner. Spec: WEAKNESS_MEMORY_IMPLEMENTATION.md (+ two GPT review rounds:
WEAKNESS_MEMORY_FIXES.md, WEAKNESS_MEMORY_REVIEW_3BBE09D.md).

**Attempt 1 (cool_fig, deo_wm_fb):** AVG7 peak 48.80 @i5 ≈ control 48.79 — but the
audit showed the mechanism never really ran: writer notes topic-level & parroted,
summarizer degraded by a context bug + silent member loss, and guided mutations
mostly ignored their target (seed dominates the mutation prompt). An invalid null.
Data: paper_data/weakness_memory_4b/ + _reaudit/.

**Attempt 2 (willing_gyro, deo_wm2_fb, v2.1.2 after R1-R8 fixes + passed smoke):**
mechanism VERIFIED working at scale — real semantic memory every iter (10 items,
top supports 17-25), ~40% of chains guided (domain-compatible routing), guided
acceptance ≈ unguided, smoke sample 5/6 in-domain / 2/6 exact-operation. Result:

| arm | AVG7 per iter (Claude) | peak | mean |
|--|--|--|--|
| mutV1 control | 45.09 47.71 47.57 46.08 48.79 | 48.79 | 47.05 |
| WM v2 | 45.44 47.24 48.14 47.67 45.41 | **48.14** | 46.78 |

Peak −0.65, mean −0.27, below control at every matched iter. **Targeting the
solver's measured weaknesses with a frozen generator buys nothing** — the tenth
falsified guidance-on-frozen-base intervention. Together with SGLD (soft-prefix =
pure diffusion ≈ baseline 49.06; Planning-SGLD pending length-gate redesign) the
picture is unchanged: base distribution + band filter sets the score; only a
generator whose DISTRIBUTION moves (R-Zero's trained questioner) has ever beaten it
at 8B, and nothing static reproduces that.

Full GPT-analyzable data (pools with per-chain target/guidance fields, memories,
audits, provenance events): paper_data/weakness_memory_v2_4b/; per-set grades:
paper_data/claude_grade/4b_wm2_claude.jsonl.

## General-domain transfer (R-Zero paper Table-2 counterpart) — DEO transfers like R-Zero (Sep 2026)

Project harness (NOT the paper's): MMLU-Pro / SuperGPQA / BBEH, frozen stratified
3000-sample subsets (seed 0), greedy, boxed extraction; identical for every
checkpoint, so cross-checkpoint deltas are comparable. 4B, step-15 ckpts.

| model | MMLU-Pro | SuperGPQA | BBEH | AVG3 |
|--|--|--|--|--|
| base | 55.13 | 27.07 | 8.20 | 30.13 |
| DEO(mutV1) i1 | 58.87 | 28.67 | 9.40 | **32.31** |
| DEO i2 | 58.37 | 28.10 | 9.60 | 32.02 |
| DEO i3 | 58.10 | 28.50 | 9.90 | 32.17 |
| DEO i4 | 58.73 | 28.50 | 9.13 | 32.12 |
| DEO i5 | 58.63 | 28.30 | 9.13 | 32.02 |
| R-Zero i1 | 59.03 | 28.60 | 8.87 | 32.17 |
| R-Zero i2 | 58.67 | 28.43 | 8.80 | 31.97 |
| R-Zero i3 | 59.37 | 29.53 | 8.87 | **32.59** |
| R-Zero i4 | 59.37 | 29.00 | 8.73 | 32.37 |
| R-Zero i5 | 58.43 | 28.73 | 9.17 | 32.11 |

Math-only self-evolution transfers to general reasoning for BOTH methods
(+1.9-2.5 AVG3, materializing at i1 then plateauing — same dynamics as math).
Peak-to-peak DEO 32.31 vs R-Zero 32.59: parity within noise; DEO leads BBEH every
iteration, R-Zero leads MMLU-Pro/SuperGPQA slightly. The "98% of R-Zero at ~half
compute" conclusion extends to the general domain. Per-item outputs:
blob yyd_geval_4b/geval/*.json; harness DEO/general_eval{,_main}.py.
Baseline no-walk 2000q arm (orange_ghost):

| model | MMLU-Pro | SuperGPQA | BBEH | AVG3 |
|--|--|--|--|--|
| baseline i1 | 58.40 | 28.87 | 8.33 | 31.87 |
| baseline i2 | 58.67 | 28.57 | 8.40 | 31.88 |
| baseline i3 | 58.73 | 28.67 | 8.27 | 31.89 |
| baseline i4 | 58.93 | 29.70 | 9.17 | **32.60** |
| baseline i5 | 58.07 | 28.40 | 8.47 | 31.65 |

**The general-domain transfer belongs to math RL itself, not to any curriculum
mechanism**: the plain no-walk baseline peaks at 32.60 — statistically identical
to R-Zero's 32.59 and DEO's 32.31. All three curricula buy the same ~+2 AVG3;
neither the MCMC walk nor the trained questioner adds general-domain transfer
beyond what training on ANY self-generated in-band math set provides. (This
reframes R-Zero's Table-2 claim: the transfer is real but not attributable to
their challenger.)

## Runtime & memory: DEO vs R-Zero (measured wallclock, Sep 2026)

End-to-end job wallclock (5 iterations + final 7-set eval), identical node type
(1x Standard_ND96isr_H100_v5, 8x H100), from job log timestamps:

| scale | R-Zero | DEO (walk 2000q) | ratio |
|--|--|--|--|
| 4B | shy_brake: **51.6 h** (413 GPU-h) | icy_sprout/mutV1: **24.9 h** (199 GPU-h) | **0.48x** |
| 8B | plum_plane (qwen8b-rzero-v2): **89.2 h** (714 GPU-h) | cool_loquat: **34.9 h** (279 GPU-h) | **0.39x** |

- The "~half compute" claim is exact at 4B and conservative at 8B: R-Zero's
  questioner GRPO training grows with model size, while DEO's walk scoring is
  pure inference and scales flatter.
- Cost composition: solver GRPO is IDENTICAL on both sides; the entire gap is
  "train a second model every iteration" (R-Zero: questioner GRPO + generating
  ~4-5k questions/iter) vs "sample a frozen generator" (DEO: 2000 questions +
  5-sweep proposal scoring, inference only).
- Caveat: these are end-to-end recipe costs, not per-question-matched costs —
  R-Zero generates ~2x more questions per iteration by design.

Memory (structural; peak telemetry was not logged): the solver-training footprint
is identical. On the generator side R-Zero holds a full TRAINING instance of the
second model — bf16 weights + grads + fp32 Adam moments + master ~ 16 bytes/param
(8B: ~128 GB training state, FSDP-sharded) plus its rollout KV — while DEO serves
a FROZEN generator at ~2 bytes/param (8B: ~16 GB weights + KV cache, one GPU at
0.85 util, no gradients/optimizer). DEO's peak is always the solver trainer; the
generation side stays constant and small. The same 16-vs-2-bytes/param structure
is what lets DEO use closed-source API models as the generator (see the claudegen
arm), which is impossible for a trainable challenger.

## Closed-source generator: Claude writes the questions — best 4B result of the program (Sep 2026)

`tender_cartoon` (deo_claudegen_fb): **Haiku 4.5 generates the initial pool AND all
MCMC mutations** (raw-HTTP messages API, system-prompt caching, mutation temp
clamped 1.1->1.0 by the API); the Qwen3-4B solver, majority-vote pseudo-labels,
judge, band filter, GRPO training and eval are UNCHANGED from the mutV1 recipe.

| iter | AVG7 (Claude grader) |
|--|--|
| i1 | 45.63 |
| i2 | 47.66 |
| **i3** | **49.60** |
| i4 | 47.17 |
| i5 | 47.63 |

- **Peak 49.60 @i3 is the highest 4B number in the program**: mutV1 48.79, SGLD
  49.06, heroic_eye 48.94, claudelabel 48.90, R-Zero 48.13. At the iter-3 budget
  the lead over mutV1 is +2.03 (49.60 vs 47.57), driven by Minerva (50.4 vs 46.0)
  and AIME24 (20.0).
- Mechanistic correlate: Claude's questions pass the band filter at **~82%/iter
  (1612-1665/2000)** vs ~65% for local-base generation — a stronger generator
  yields a denser trainable curriculum, and it shows downstream.
- The structural point: DEO only requires a SAMPLABLE proposal distribution, so a
  closed-source API model slots in directly. R-Zero cannot use such a generator at
  all (its challenger must be trainable). Single-run caveat applies as everywhere.
- Grades: paper_data/claude_grade/4b_claudegen_claude.jsonl; pools in blob
  yyd_claudegen_4b.

## Non-Qwen base: OctoThinker-3B three arms (Claude grader) (Sep 2026)

OctoThinker-3B-Hybrid-Base (Llama lineage; paper does not name the variant —
Hybrid assumed and recorded). Base AVG7 = 23.42. In-band is sparse on this model
(~350-440/2000 per iter for all arms).

| arm | i1 | i2 | i3 | i4 | i5 | peak |
|--|--|--|--|--|--|--|
| DEO (frosty_yuca) | 26.67 | 27.10 | **28.30** | 26.98 | 27.32 | 28.30 @i3 |
| baseline no-walk (honest_needle) | 27.81 | 27.34 | 27.50 | 28.42 | **28.65** | 28.65 @i5 |
| R-Zero (patient_turtle) | **27.75** | 26.46 | 25.94 | 25.62 | 27.12 | 27.75 @i1 |

- **R-Zero is clearly the weakest on the weak base**: peaks at i1 then declines
  through i4 (its AMC drops BELOW base at i3: 12.5 vs 17.4) — the trained
  questioner fails to compound on a Llama-lineage base, consistent with the
  paper's small OctoThinker gains.
- DEO vs baseline is an early-peak vs late-peak noise-level split (28.30 @i3 vs
  28.65 @i5); at the iter-3 budget DEO leads baseline +0.80 and R-Zero +2.36.
  The in-run MATH-500 gap (DEO 52.8 vs baseline 51.8 peaks) does not fully
  survive 7-set grading.
- All three arms transfer: +4.3 to +5.2 AVG7 over base. Grades:
  paper_data/claude_grade/octo3b_{deo,baseline,rzero}_claude.jsonl.
- OctoThinker-8B: baseline done (base 31.95 -> peak 36.56 @i1,
  octo8b_baseline_claude.jsonl); DEO and R-Zero arms in flight.

### OctoThinker per-dataset detail (Claude grader)

**OctoThinker-3B DEO (frosty_yuca)**

| model | MATH-500 | GSM8K | AMC | Minerva | Olympiad | AIME24 | AIME25 | AVG7 |
|--|--|--|--|--|--|--|--|--|
| base | 44.60 | 61.49 | 17.42 | 19.12 | 14.67 | 6.67 | 0.00 | 23.42 |
| i1 | 49.80 | 74.00 | 22.58 | 24.63 | 15.70 | 0.00 | 0.00 | 26.67 |
| i2 | 53.00 | 75.82 | 17.50 | 23.90 | 16.15 | 3.33 | 0.00 | 27.10 |
| i3 | 52.60 | 74.75 | 25.08 | 26.47 | 15.85 | 3.33 | 0.00 | 28.30 |
| i4 | 51.40 | 73.84 | 15.08 | 26.10 | 19.11 | 3.33 | 0.00 | 26.98 |
| i5 | 50.00 | 74.15 | 20.00 | 22.06 | 18.37 | 3.33 | 3.33 | 27.32 |

**OctoThinker-3B baseline no-walk (honest_needle)**

| model | MATH-500 | GSM8K | AMC | Minerva | Olympiad | AIME24 | AIME25 | AVG7 |
|--|--|--|--|--|--|--|--|--|
| base | 44.60 | 61.49 | 17.42 | 19.12 | 14.67 | 6.67 | 0.00 | 23.42 |
| i1 | 50.60 | 72.55 | 27.50 | 26.84 | 17.19 | 0.00 | 0.00 | 27.81 |
| i2 | 48.20 | 73.54 | 25.00 | 25.37 | 19.26 | 0.00 | 0.00 | 27.34 |
| i3 | 48.40 | 73.24 | 25.00 | 27.21 | 18.67 | 0.00 | 0.00 | 27.50 |
| i4 | 50.00 | 73.24 | 27.50 | 23.90 | 17.63 | 6.67 | 0.00 | 28.42 |
| i5 | 52.20 | 74.68 | 27.50 | 26.10 | 16.74 | 3.33 | 0.00 | 28.65 |

**OctoThinker-3B R-Zero (patient_turtle)**

| model | MATH-500 | GSM8K | AMC | Minerva | Olympiad | AIME24 | AIME25 | AVG7 |
|--|--|--|--|--|--|--|--|--|
| base | 44.60 | 61.49 | 17.42 | 19.12 | 14.67 | 6.67 | 0.00 | 23.42 |
| i1 | 50.20 | 73.39 | 24.92 | 25.00 | 17.33 | 3.33 | 0.10 | 27.75 |
| i2 | 51.00 | 72.71 | 17.58 | 26.47 | 17.48 | 0.00 | 0.00 | 26.46 |
| i3 | 50.20 | 73.09 | 12.50 | 27.21 | 15.26 | 3.33 | 0.00 | 25.94 |
| i4 | 48.40 | 72.93 | 15.00 | 24.26 | 15.41 | 3.33 | 0.00 | 25.62 |
| i5 | 48.60 | 73.09 | 27.50 | 26.10 | 14.52 | 0.00 | 0.00 | 27.12 |

**OctoThinker-8B baseline no-walk (purple_dolphin)**

| model | MATH-500 | GSM8K | AMC | Minerva | Olympiad | AIME24 | AIME25 | AVG7 |
|--|--|--|--|--|--|--|--|--|
| base | 54.60 | 80.74 | 27.50 | 26.84 | 23.85 | 6.77 | 3.33 | 31.95 |
| i1 | 59.00 | 85.75 | 35.08 | 38.97 | 27.11 | 6.67 | 3.33 | 36.56 |
| i2 | 61.40 | 86.73 | 22.50 | 39.34 | 26.22 | 0.00 | 0.00 | 33.74 |
| i3 | 62.20 | 87.34 | 30.00 | 37.87 | 25.63 | 10.00 | 0.94 | 36.28 |
| i4 | 61.60 | 87.64 | 32.50 | 38.60 | 26.07 | 6.67 | 0.00 | 36.15 |
| i5 | 60.40 | 88.32 | 30.00 | 39.34 | 25.93 | 6.67 | 0.10 | 35.82 |

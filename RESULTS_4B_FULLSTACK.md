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

### witty_soca (warm-start) — PENDING (`helpful_watch_7qyr66x33w`)
### willing_panda (baseline 2000q) — PENDING (`gentle_hook_k8g0sxb6s2`)

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
7-set 48.94, HARD5 34.2), but the margin over a plain no-walk 2000q baseline is small (~1pt MATH-500),
and warm-starting the chain from the previous iteration's pool gives no benefit. Gains trace to cleaner
labels + the walk finding useful (mostly AMC/Minerva-type) questions, not to difficulty control or pool
size. Pending: 7-set Claude grades for witty_soca and willing_panda to confirm this on effective accuracy.

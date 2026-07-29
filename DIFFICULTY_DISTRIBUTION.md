# Question difficulty (p̂) of self-evolving training data (Qwen3-8B-Base)

**Difficulty measure.** `p̂(x) = (modal-answer count) / m` = the solver's self-consistency over m=9
samples. Low p̂ ⇒ solver disagrees with itself ⇒ harder / more uncertain (p̂≈0.5 is maximal uncertainty
for a binary-ish answer; p̂ near the number of distinct answers' floor is "all over the place"). Both DEO
and R-Zero filter training questions to **p̂ ∈ [0.3, 0.8]**. R-Zero's stored `score` is exactly this p̂
(`max_count/len`), so the two are the same quantity.

**Stage caveat.** DEO's `mcmc_iter_N` is the **full MCMC pool** (pre-filter, N=1500). R-Zero's parquet
is the **post-filter training set** (already p̂∈[0.3,0.8]). So the first table mixes stages; the second
table restricts everyone to the p̂∈[0.3,0.8] training band for an apples-to-apples comparison.

## p̂ distribution per iteration (fraction of pool in each band)

| run | iter | N | mean p̂ | p̂<0.3 (too hard) | p̂∈[0.3,0.8] (kept) | p̂>0.8 (too easy) |
|---|--|--|--|--|--|--|
| baseline (full pool) | v1 | 1500 | 0.395 | 0.425 | 0.513 | 0.062 |
| baseline | v2 | 1500 | 0.398 | 0.411 | 0.529 | 0.060 |
| baseline | v3 | 1500 | 0.397 | 0.409 | 0.531 | 0.059 |
| baseline | v4 | 1500 | 0.400 | 0.411 | 0.526 | 0.063 |
| baseline | v5 | 1500 | 0.402 | 0.413 | 0.515 | 0.072 |
| strong (full pool) | v1 | 1500 | 0.327 | 0.476 | 0.515 | 0.009 |
| strong | v2 | 1500 | 0.321 | 0.537 | 0.445 | 0.019 |
| strong | v3 | 1500 | 0.305 | 0.547 | 0.444 | 0.009 |
| strong | v4 | 1500 | **0.427** | **0.145** | **0.854** | 0.001 |
| strong | v5 | 1500 | **0.426** | **0.147** | **0.853** | 0.000 |
| R-Zero (filtered) | v1 | 3946 | 0.524 | 0 | 1.000 | 0 |
| R-Zero | v2 | 3977 | 0.515 | 0 | 1.000 | 0 |
| R-Zero | v3 | 4129 | 0.515 | 0 | 1.000 | 0 |
| R-Zero | v4 | 4401 | 0.517 | 0 | 1.000 | 0 |
| R-Zero | v5 | 4564 | 0.527 | 0 | 1.000 | 0 |

## Same stage: mean p̂ within the training band [0.3,0.8]

| run | v1 | v2 | v3 | v4 | v5 |
|---|--|--|--|--|--|
| baseline | 0.514 | 0.517 | 0.511 | 0.517 | 0.514 |
| **strong** | **0.459** | **0.473** | **0.464** | **0.463** | **0.463** |
| R-Zero | 0.524 | 0.515 | 0.515 | 0.517 | 0.527 |

(lower mean p̂ = harder within the band; strong is hardest ~0.46, R-Zero easiest ~0.52, baseline ~0.51.)

## Findings

1. **baseline (no walk): stable, ~52% of the base-model pool lands in-band**, mean p̂≈0.40; ~41% too
   hard (p̂<0.3), ~6% too easy. No drift across iters.

2. **strong-β control: two regimes.** v1–v3 (β descending 1→1.6) sit at mean p̂≈0.31 with **~48–55%
   of the pool *below* 0.3** (too hard/ambiguous — the greedy walk overshoots into the disagreement
   region). Once β hits its floor 0.02 (v4–v5), the pool **concentrates toward p̂≈0.5**: mean p̂ rises to
   0.43, too-hard drops to ~15%, and the in-band (trainable) fraction **jumps to ~85%**. So strong β
   control does pack far more questions into the band — but at p̂≈0.46 they are the *hardest* in-band set.

3. **R-Zero: trained-questioner questions sit at mean p̂≈0.52** (slightly *above* 0.5 → marginally the
   *easiest* by this metric), and the filtered training set grows each iter (3946→4564). Combined with
   the pseudo-label analysis (`PSEUDO_LABEL_QUALITY.md`), R-Zero's questions are not the hardest by p̂,
   yet its labels are the least *correct* — i.e. low p̂-consistency and low actual-correctness are
   distinct: R-Zero produces questions the solver is *confidently wrong* on (moderate p̂, wrong modal).

4. **p̂-difficulty does not track downstream accuracy** either: strong (hardest in-band, ~0.46) ties the
   milder adaptive run; R-Zero (easiest in-band p̂ but worst labels) wins. Consistent with
   `PSEUDO_LABEL_QUALITY.md`: neither p̂-hardness nor label-correctness explains final MATH accuracy.

Data: `paper_data/difficulty_phat_stats.txt`.

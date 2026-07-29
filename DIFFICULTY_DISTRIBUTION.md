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

## How many questions actually enter solver training (per iter)

Training-set size = # questions passing the p̂∈[0.3,0.8] filter, pushed to verl (8B runs, pool=1500):

| iter | baseline (DEO) | strong (DEO) | R-Zero |
|--|--|--|--|
| v1 | 731 | 681 | 3946 |
| v2 | 757 | 604 | 3977 |
| v3 | 762 | 585 | 4129 |
| v4 | 746 | 1179 | 4401 |
| v5 | 730 | 1164 | 4564 |

- **Dataset size differs ~5–7×** (DEO ~600–1180 vs R-Zero ~4000–4600): DEO's MCMC pool is fixed at
  TOTAL_QUESTIONS/iter (1500 here) → ~700 pass the filter; R-Zero's questioner emits a far larger raw
  pool → ~4000+ pass. (strong's v4–v5 jump to ~1180 is the β→0.02 in-band spike to ~85%.)
- **But verl consumes the SAME ~1280 prompt-instances/iter for both** — solver GRPO runs
  `max_steps=20 × rollout_batch_size=64 = 1280` prompts (each ×5 rollouts). So DEO (~700 < 1280) cycles
  its set ~1.8 epochs, while R-Zero (~4000 > 1280) touches only ~1280 (~0.3 epoch, ~32%) of its set.
  **→ Despite generating 5–7× more questions, R-Zero trains on the same # of prompt-instances/iter as
  DEO; the larger pool is not converted into more training steps.**
- (DEO pool raised to TOTAL_QUESTIONS=2000 for subsequent runs → ~1000–1300 pass the filter, closer to
  the 1280 verl consumes, i.e. ~1 epoch with less repetition.)

## KL anchor (reference model) — DEO vs R-Zero

Same KL strength (`kl_coef=0.01, use_kl_loss=true, low_var_kl`) but **different reference each iter**:
- **DEO**: `worker.ref.model = BASE_MODEL` → KL anchored to the **original base model every iter** (fixed).
- **R-Zero**: ref unset → verl default = `actor.model` = that iter's init = the **previous-iter solver**
  → the KL anchor **drifts** iteration to iteration.

Data: `paper_data/difficulty_phat_stats.txt`.

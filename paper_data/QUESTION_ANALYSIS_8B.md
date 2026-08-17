# Question-Level Analysis: DEO 8B walk-2000 (V1) vs R-Zero 8B Self-Generated Datasets

Comparative analysis of the two 8B self-generated math training corpora, iterations 1-5.
All "like-for-like" comparisons use DEO's post-filter training sets
(`paper_data/8b_datasets/walk2000_v1/filtered_deo_fixedbeta_2000_solver_v{1..5}.json`, p_hat in [0.33, 0.78])
against R-Zero's post-filter sets (`paper_data/8b_datasets/rzero/rzero_iter_{1..5}.json`, score in [0.33, 0.80]).
Verified: every DEO filtered record is a subset of the pool restricted to p_hat in [0.3, 0.8]
(~150/iter of the in-band pool are additionally dropped by DEO's own filter).
Both methods' labels are solver majority votes; no claim of different label mechanisms is made anywhere below.

Accuracy context (same grader): DEO walk-2000 7-set peak 53.60 (peaks iter3); R-Zero 8B 54.57
(still rising at iter5; HARD5 41.36 vs DEO 40.02). verl consumes only 1280 prompts/iter from either set.

## Summary table (iter1 → iter5, like-for-like sets)

| Metric | DEO iter1 | DEO iter5 | R-Zero iter1 | R-Zero iter5 |
|---|---|---|---|---|
| n (post-filter) | 1520 | 1525 | 3946 | 4564 |
| Question chars (median / mean) | 213 / 238 | 212 / 237 | 224 / 248 | **364 / 439** |
| Words (mean) | 46.6 | 46.4 | 48.6 | **82.7** |
| LaTeX cmds per 100 words | 3.41 | 3.10 | 2.12 | 1.49 |
| Integer-format answers | 69.5% | 70.7% | 73.7% | **87.3%** |
| Narrative framing (names/story) | 2.0% | 2.2% | 11.2% | **31.5%** |
| Vague quantifiers ("some/several/certain N") | 3.7% | 3.5% | 7.4% | **31.1%** |
| "Alice/Bob game" questions | 0.3% | 0.5% | 5.3% | **28.4%** |
| "P(x) is a polynomial" template | 0.3% | 0.1% | 7.8% | **37.9%** |
| Optimal-play / winning-strategy phrasing | 0.5% | 0.3% | 2.7% | **16.5%** |
| Within-iter first-80-char collisions | 1 (0.1%) | 4 (0.3%) | 277 (7.0%) | **1493 (32.7%)** |
| Iter-N first-80 overlap with iter1 | — | 0.4% | — | 13.5% (20.9% at iter4) |
| Distinct answers | 561 | 558 | 1221 | **493** |
| Top-10 answers' share of labels | 47.0% | 48.0% | 39.4% | **72.0%** |
| Mean p_hat | 0.473 | 0.472 | 0.524 | 0.527 |
| Mass in p_hat [0.6, 0.8) | 10.4% | 10.3% | 32.8% | 29.7% |
| \boxed{...} leaked into question | 5.7% | 5.9% | 0.0% | 0.0% |

Headline: **DEO is stationary across iterations (every metric flat within noise) but carries a persistent
~6-10% malformed/leaky-question floor. R-Zero starts cleaner and drifts hard — questions double in length,
collapse onto two template families (polynomial-P(x) + Alice/Bob games) covering ~2/3 of iter5, and label
diversity collapses (top-10 answers = 72% of iter5), consistent with the known 72%→29% label-correctness
collapse in PSEUDO_LABEL_QUALITY.md.**

## 1. Surface statistics (per iter, like-for-like)

Question length (chars): median / mean / p10 / p90. LaTeX = `\command` tokens per question.

| Set | chars med/mean/p10/p90 | words mean | LaTeX/q mean | LaTeX per 100w | numeric tokens/q |
|---|---|---|---|---|---|
| DEO i1 | 213 / 238 / 116 / 367 | 46.6 | 1.59 | 3.41 | 5.4 |
| DEO i2 | 213 / 243 / 119 / 376 | 47.0 | 1.73 | 3.68 | 5.4 |
| DEO i3 | 210 / 241 / 119 / 388 | 47.2 | 1.49 | 3.16 | 5.4 |
| DEO i4 | 215 / 240 / 119 / 384 | 47.2 | 1.58 | 3.34 | 5.5 |
| DEO i5 | 212 / 237 / 117 / 382 | 46.4 | 1.44 | 3.10 | 5.2 |
| RZ i1 | 224 / 248 / 102 / 416 | 48.6 | 1.03 | 2.12 | 5.0 |
| RZ i2 | 293 / 338 / 155 / 547 | 64.0 | 1.45 | 2.27 | 5.7 |
| RZ i3 | 339 / 388 / 204 / 619 | 73.4 | 1.38 | 1.88 | 5.4 |
| RZ i4 | 360 / 417 / 219 / 680 | 78.6 | 1.13 | 1.44 | 5.3 |
| RZ i5 | 364 / 439 / 210 / 735 | 82.7 | 1.24 | 1.49 | 5.0 |

Answer-format distribution (% of labels):

| Set | integer | fraction | expression | interval/set | decimal | other |
|---|---|---|---|---|---|---|
| DEO i1 | 69.5 | 13.2 | 10.4 | 4.1 | 1.8 | 0.9 |
| DEO i3 | 69.4 | 13.4 | 10.5 | 4.0 | 1.8 | 0.9 |
| DEO i5 | 70.7 | 12.0 | 10.2 | 4.5 | 2.0 | 0.7 |
| RZ i1 | 73.7 | 11.5 | 11.2 | 1.3 | 1.5 | 0.9 |
| RZ i3 | 78.1 | 6.3 | 12.3 | 2.3 | 0.2 | 0.8 |
| RZ i5 | 87.3 | 2.0 | 7.9 | 2.4 | 0.1 | 0.4 |

Reading: DEO's surface profile is flat across all five iterations — same length, same LaTeX density,
same answer-format mix. R-Zero questions grow ~77% longer in mean chars (248 -> 439) while LaTeX density per
word *falls* (2.12 -> 1.49 per 100 words): the added length is prose, not math. R-Zero's answer space also
homogenizes toward bare integers (73.7% -> 87.3%; fractions collapse 11.5% -> 2.0%, decimals 1.5% -> 0.1%) —
a signature of questions whose "answers" are small counts/min-max values rather than computed quantities.

## 2. Topic distribution (keyword classifier, primary label, % of questions)

Priority order geometry > number theory > combinatorics > probability > calculus > sequences > algebra;
"other" = no keyword hit. Crude keyword matching — treat trends, not levels, as meaningful.

| Set | geom | numthy | combin | prob | calc | seq | algebra | other |
|---|---|---|---|---|---|---|---|---|
| DEO i1 | 34.9 | 13.9 | 2.0 | 7.4 | 6.6 | 2.0 | 15.9 | 17.2 |
| DEO i3 | 38.3 | 12.6 | 2.0 | 5.2 | 7.9 | 1.9 | 15.3 | 16.8 |
| DEO i5 | 39.0 | 12.5 | 3.1 | 5.5 | 6.8 | 1.9 | 14.0 | 17.3 |
| RZ i1 | 27.7 | 17.8 | 5.4 | 5.3 | 1.4 | 7.8 | 16.6 | 18.0 |
| RZ i3 | 34.1 | 22.5 | 5.3 | 4.1 | 1.0 | 6.2 | 14.8 | 11.9 |
| RZ i5 | 34.3 | 25.9 | 4.3 | 2.0 | 0.9 | 4.1 | 19.1 | 9.5 |

Reading: DEO drifts mildly toward geometry (+4pt) and away from probability; otherwise stationary, and keeps a
7-9% calculus slice R-Zero never has (~1%). R-Zero drifts strongly toward number theory (17.8 -> 25.9,
largely the P(x)-polynomial-with-integer-coefficients family, see Sec. 6) and geometry, while probability
(5.3 -> 2.0), sequences (7.8 -> 4.1) and combinatorics shrink. R-Zero's "other" halves (18.0 -> 9.5) — late
iters are *more* classifiable because they concentrate on a few stereotyped families.

## 3. Style markers (% of questions)

names = common personal names / Mr./Mrs.; story = story-context nouns (garden, farmer, game show, ...);
narrative = names OR story. multi-ask = >=2 imperative asks ("find ... determine ...").
no-ask = neither an interrogative "?" nor any find/compute/how-many verb (likely ill-posed fragment).

| Set | names | story | narrative | multipart | multi-ask | figure-ref | meta-instr | no-ask |
|---|---|---|---|---|---|---|---|---|
| DEO i1 | 1.0 | 1.0 | 2.0 | 5.5 | 9.4 | 0.0 | 5.7 | 3.0 |
| DEO i3 | 1.1 | 1.5 | 2.5 | 4.6 | 10.7 | 0.2 | 5.0 | 2.3 |
| DEO i5 | 1.2 | 1.2 | 2.2 | 3.0 | 10.9 | 0.1 | 4.7 | 2.3 |
| RZ i1 | 6.6 | 4.9 | 11.2 | 1.8 | 5.4 | 0.4 | 3.5 | 0.7 |
| RZ i3 | 19.0 | 3.0 | 21.5 | 4.6 | 9.2 | 0.1 | 2.6 | 0.5 |
| RZ i5 | 30.4 | 1.4 | 31.5 | 5.8 | 8.8 | 0.1 | 1.2 | 0.3 |

Suspicious-construct rates specific to each method:

| Construct | DEO i1 | DEO i5 | RZ i1 | RZ i5 |
|---|---|---|---|---|
| `\boxed{...}` inside the QUESTION text | 5.7% | 5.9% | 0.0% | 0.0% |
| ...and the boxed value equals the training label (answer leak) | 3.5% | 3.6% | 0% | 0% |
| Mutation meta-text leaked ("original problem", "Change the operation...", `<inside>`) | 0.7% | 0.5% | ~0% | ~0% |
| `\inst{...}` corruption artifact | 0.1% | 0.0% | 0% | 0% |
| Vague quantifier ("some/several/certain number...") | 3.7% | 3.5% | 7.4% | 31.1% |
| "where <var>" late-symbol clause | 9.3% | 10.1% | 3.2% | 5.9% |
| Constraint stacking ("Additionally/Furthermore/Moreover") | 4.5% | 3.3% | 1.2% | 4.3% |
| Optimal-play / "winning strategy" / "guarantee" | 0.5% | 0.3% | 2.7% | 16.5% |
| Label is infinity/undefined | 1.25% | 1.25% | 0.63% | 1.47% |
| Label "0" on a "how many"-style question | 4.7% | 4.3% | 2.7% | 3.4% |

Reading: the two failure profiles are disjoint. DEO's problems are *mechanical*: a constant ~6% of questions
carry a `\boxed{}` from the mutation source, and in 3.5% the boxed value IS the training label — the solver
can copy the answer out of the question (free reward, zero learning signal). Figure references are absent in
both (good). R-Zero's problems are *semantic*: narrative framing triples (11% -> 31%), vague quantifiers
quadruple (7.4% -> 31.1%), and guarantee/optimal-strategy phrasing grows 6x (2.7% -> 16.5%) — the classic
shape of hard-sounding but under-specified questions on which a majority vote still "agrees" on a small
integer. This is the reward-hacking signature quantified in Sec. 6.

## 4. Duplication and diversity

Within-iter (exact = normalized-text match; first80 = normalized first-80-char collision;
near-dup = token 8-gram Jaccard >= 0.5):

| Set | exact | first80 collisions | near-dup (J>=0.5) |
|---|---|---|---|
| DEO i1 | 0 | 1 (0.1%) | 0 (0.0%) |
| DEO i3 | 0 | 1 (0.1%) | 0 (0.0%) |
| DEO i5 | 0 | 4 (0.3%) | 1 (0.1%) |
| RZ i1 | 6 | 277 (7.0%) | 38 (1.0%) |
| RZ i3 | 0 | 675 (16.3%) | 11 (0.3%) |
| RZ i5 | 0 | 1493 (32.7%) | 11 (0.2%) |

Across-iter recurrence (share of iter-N questions whose normalized first 80 chars appear in iter1):
DEO: 0.3% / 0.3% / 0.5% / 0.4% (i2..i5). R-Zero: 9.0% / 13.8% / 20.9% / 13.5%.
Exact full-text recurrence is ~0 for both; 8-gram near-dup (J>=0.5) between iter1 and iter5 is 0.0% for both
(sampled 800x800). Distinct/total 8-gram ratio: DEO 0.973 / 0.975 / 0.973 (i1/i3/i5); R-Zero 0.876 / 0.879 / 0.837.

Reading: R-Zero does not literally recycle questions (exact dup ~0) but heavily recycles *openings and
templates*: by iter5 one third of questions share their first 80 characters with another question in the same
iteration, and ~1 in 5 iter4 questions open identically to an iter1 question. Top iter5 templates:
"Let P(x) be a polynomial with integer coefficients such that P(0)=1 and P(1)=3..." (x65), "...P(1)=2 ..." (x58),
"...P(1)=2023..." (x51), "Let P(x) ... P(1)=2023 and P(2023)=..." (x49), "Alice and Bob play a game on an
infinite grid of unit squares..." (x41). Broad counts: 37.9% of iter5 matches "P(x) is/be a (monic) polynomial"
(7.8% at iter1); 28.4% matches Alice/Bob-game phrasing (5.3% at iter1). DEO's MCMC walk keeps near-zero
duplication at every level throughout — its diversity does not degrade, but note R-Zero still fields ~4,000
unique problems/iter vs DEO's ~1,520 (both are subsampled to 1280 by verl, so R-Zero has ~3x selection headroom).

## 5. Difficulty structure (p_hat)

Like-for-like histograms (% of set per bucket; p_hat = solver self-consistency = score):

| Set | mean | [0.3,0.4) | [0.4,0.5) | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) |
|---|---|---|---|---|---|---|
| DEO i1 | 0.473 | 23.4 | 39.5 | 26.8 | 8.6 | 1.8 |
| DEO i3 | 0.466 | 27.5 | 35.8 | 27.9 | 7.6 | 1.2 |
| DEO i5 | 0.472 | 25.0 | 38.0 | 26.7 | 8.1 | 2.2 |
| RZ i1 | 0.524 | 27.5 | 18.0 | 21.6 | 17.6 | 15.2 |
| RZ i3 | 0.515 | 25.6 | 25.6 | 20.9 | 15.7 | 12.2 |
| RZ i5 | 0.527 | 21.6 | 25.4 | 23.4 | 16.9 | 12.8 |

DEO pre-filter pool (full 2000/iter) is also stationary: mean p_hat 0.414-0.425 every iter;
[0.1,0.3)=16-17%, [0.3,0.5)=53-56%, [0.5,0.8)=27-30%, >=0.8 ~0.1%. The MCMC walk reliably concentrates
proposals near the target band before filtering (76-84% of the raw pool is already in [0.3,0.8]).

Length vs difficulty: weak negative correlation everywhere — r(len, p_hat) = -0.09/-0.08/-0.04 for DEO
i1/i3/i5 and -0.11/-0.06/-0.07 for R-Zero; harder (lower-p_hat) questions are slightly longer in both.

Reading: DEO's distribution is strongly peaked at p_hat in [0.4,0.6) (66% of the set) with only ~10% in the
easier [0.6,0.8) tail. R-Zero spreads much flatter: ~30% of its set sits in [0.6,0.8). Under a
majority-vote-label regime, the [0.6,0.8) band is where labels are most likely *correct*, so R-Zero's
training batches carry a built-in ballast of easier, more-reliably-labeled items, while DEO's harder-centered
band maximizes uncertainty-based signal but also maximizes label noise. DEO's in-band gt==pseudo_label
agreement (challenger-proposed answer vs solver majority) is flat at 80.5-82.8% across all five iters —
no drift, but a constant ~18% disagreement floor.

## 6. Iteration drift and reward-hacking signatures

DEO iter1 -> iter5: essentially nothing changes. Length flat (238 -> 237 mean chars), topic mix +-4pt,
p_hat mean 0.473 -> 0.472, answer concentration flat (top-10 = 47.0% -> 48.0%, distinct answers 561 -> 558),
duplication ~0 throughout, artifact rates constant. The walk is distribution-stationary by construction;
its failure modes at iter5 are the same as at iter1.

R-Zero iter1 -> iter5, quantified drift (all monotone across the 5 iters):

| Signature | i1 | i2 | i3 | i4 | i5 |
|---|---|---|---|---|---|
| Mean question chars | 248 | 338 | 388 | 417 | 439 |
| "P(x) polynomial" template | 7.8% | 13.5% | 24.6% | 35.6% | 37.9% |
| Alice/Bob game framing | 5.3% | 10.0% | 15.9% | 25.7% | 28.4% |
| Contest-year token (2020-2029) in question | 9.2% | 16.3% | 19.2% | 20.9% | 26.7% |
| Optimal-play/guarantee phrasing | 2.7% | 5.1% | 8.7% | 14.5% | 16.5% |
| Vague quantifiers | 7.4% | 13.8% | 22.7% | 26.9% | 31.1% |
| "smallest/largest n/k/m/d such that" | 2.9% | 5.0% | 7.3% | 8.2% | 12.1% |
| Within-iter first80 collisions | 7.0% | 7.5% | 16.3% | 30.3% | 32.7% |
| Distinct answers | 1221 | — | 795 | — | 493 |
| Top-10 answer share | 39.4% | — | 55.8% | — | 72.0% |
| Small-integer (0-10) labels | 40.3% | — | 54.4% | — | 68.1% |
| Integer-format labels | 73.7% | 75.2% | 78.1% | 84.1% | 87.3% |

The reward-hacking mechanism these numbers describe: the challenger discovers that game-theoretic /
min-max-guarantee questions ("smallest k such that Alice can always win...") and constrained-polynomial
riddles reliably produce mid-band self-consistency — the solver can't verify them, but its guesses cluster
on small integers (1, 2, 3 alone are 49.5% of iter5 labels vs 21.6% at iter1). Question difficulty
"increases" in appearance while the label distribution degenerates toward a near-constant prior. This is the
question-side counterpart of the 72% -> 29% label-correctness collapse measured in PSEUDO_LABEL_QUALITY.md.
Notably, mean p_hat stays flat (0.524 -> 0.527) — the filter metric itself is blind to this collapse.

## 7. Qualitative read (15 evenly-spaced questions per method per {iter1, iter5})

Method: indices j*n/15, j=0..14, read in full; excerpts truncated. Small sample — directional only.

### DEO iter1 and iter5 (30 questions read)
Roughly 8-9/15 per iter are well-posed and plausibly labeled (e.g. "smallest N with 7N = 8 mod 29" -> 26,
correct; "P monic, integer coefficients, P(0)=-2022, 5 distinct integer roots, find P(1)" -> 0, plausible).
The rest split into recurring, iteration-independent failure modes:
- **Corruption/leak artifacts**: `f\inst{1:5dN_1}(x_1,...)` garbage tokens; `<inside question> ... \end{inside}`
  wrappers; questions ending mid-sentence ("...guaranteed to be continuous sides '").
- **Answer leaked in question**: "...partitions kept at suit levels also in variables? \boxed{14}" with
  label 14 (word-salad question + free answer; 3.5% of the corpus has this exact leak pattern).
- **Mutation instruction leaked as question text**: "Change the operation in the function f(x)=x^3-3x^2+2x
  to multiplication. Find the derivative..."; "Alter the original problem's structure by separating the
  variables into two parts. Find 1x2x3x...xn ...".
- **Underdetermined/ill-posed**: "Find x^2+y^2 given that there is exactly one pair of integers (x,y)
  satisfying x^2+y^2=k" (k never fixed; label 0); "Find all points equidistant from A and B on plane
  x+y+z=5. How many such points exist?" labeled `\infty` (a line — label technically right but untrainable);
  self-contradictory geometry (circle of radius BC/2 tangent to both legs AND through B and C).
DEO iter5 reads exactly like iter1 — no stylistic drift, same artifact mix.

### R-Zero iter1 (15 questions read)
Mostly fluent, well-formed contest-style items. ~9/15 well-posed with plausible labels (e.g. "20 points, no
3 collinear, number of triangles" -> 1140, correct; cyclic quadrilateral area-ratio item -> defensible).
Failures are over-constrained or double-asked word problems (rectangle with perimeter 30 whose constraints
are inconsistent, then asks BOTH area and dimensions; label 50 answers only the first ask), and
pseudo-profound number theory with false premises ("for each N>10 there exist a,b,c with a+b+c=N and
a!+b!+c!=N^4" — false; label 1).

### R-Zero iter5 (15 questions read)
Style is dramatically more "olympiad-flavored" and verbose; only ~4/15 clearly well-posed. Recurring shapes:
- **Template family**: 5/15 are "Let P(x) be a polynomial with integer coefficients such that P(1)=2023 / 
  P(0)=1 ..." variants; labels are tiny integers (1, 2) of dubious correctness (e.g. "P(1)=2023, P(2)=P(3),
  find minimum degree n" labeled 2 — degree 0 constant 2023 satisfies both stated conditions... the intended
  answer depends on unstated assumptions).
- **Unverifiable game constructions**: "Alice and Bob play a game on a 2024x2024 grid... smallest k Alice
  needs to always win" -> 1012; "minimum moves Bob needs to guarantee a win" -> 3 — rules are genuinely
  ambiguous (undefined win conditions, underspecified move legality), no unique answer exists, yet majority
  vote produces a confident small integer.
- **Self-contradictory setups**: recursive set S with 0 in S, x+1 and x^2 in S — S is claimed "finite" but the
  rules force it infinite; "minimum number of ways Alice can arrange books such that the rule is ALWAYS
  satisfied" (category error) -> 2.
The iter5 set reads as a reward-hacked corpus: impressive surface, hollow verifiability.

## 8. Synthesis and recommendations

**Why R-Zero leads (54.57 vs 53.60, HARD5 41.36 vs 40.02) despite worse late-iter label quality.**
(a) *Cleaner early corpus*: R-Zero iter1-2 has near-zero mechanical corruption, while DEO carries a constant
~6% boxed-leak + ~4% fragment/no-ask floor in every iter — at 1280 verl prompts/iter that is ~75-125
wasted-or-poisoned prompts per DEO iteration, every iteration.
(b) *Easier-band ballast*: ~30% of R-Zero's set sits at p_hat [0.6,0.8) where majority labels are most often
right, vs DEO's ~10%; R-Zero batches mix reliable easy items with hard ones, DEO concentrates at maximal
label noise (p ~ 0.4-0.5).
(c) *Hard-set style match*: R-Zero's drift toward long, olympiad-phrased, min-max/number-theory items is
distributionally closer to HARD5-style benchmarks; even noisy labels on such questions teach format and
proof-sketch heuristics. DEO's corpus stays short and computational.
(d) *Selection headroom*: ~4,000 -> 1280 subsampling gives R-Zero 3x the pool; DEO offers 1520.
Why R-Zero keeps climbing at iter5 while its labels collapse: with only 1280 consumed prompts and label
accuracy ~29%, most of the surviving gradient signal plausibly comes from the still-correct easy-band slice
plus style transfer; the flat p_hat filter never notices the collapse — expect this climb to be a
slow-motion version of the same failure, not immunity to it (cf. probability/fraction answer classes
already extinguished by iter5).

**Concrete, falsifiable recommendations for DEO's mutation operators:**
1. *Strip solution leakage*: reject or clean any proposal containing `\boxed{}`, `<inside...>`, `\inst{`,
   mutation meta-verbs ("original problem", "change the operation"), or no interrogative/imperative.
   Prediction: removes ~8-10% of pool, +0.3-0.7 pt 7-set AVG at equal walk budget (test: regenerate
   filtered_v* with the regex filter of Sec. 3 and rerun one verl cycle).
2. *Add an easy-band ballast*: sample ~25% of the 1280 training prompts from p_hat [0.6,0.8) instead of the
   current [0.4,0.6)-heavy mix (DEO has only ~10% there — raise the pool share via temperature or acceptance
   shaping). Prediction: label-correctness of consumed prompts rises measurably (Claude-grade a 200-sample),
   and late-iter (i4/i5) checkpoints stop regressing, moving the peak past iter3.
3. *Imitate the style, not the hack*: add a mutation operator that rewrites a seed into longer
   olympiad-register phrasing ("Let ... Suppose further ... Determine the smallest ...") WITHOUT changing the
   underlying computation. Prediction: HARD5 gap (40.02 vs 41.36) closes by >=0.5 pt with unchanged label
   correctness — this isolates style transfer from difficulty drift as the cause of R-Zero's hard-set edge.
4. *Avoid what R-Zero fell into*: penalize (in the MCMC acceptance) vague quantifiers, optimal-play/guarantee
   phrasing, and answer-prior degeneracy — e.g. reject proposals whose pseudo-label is already among the
   iteration's top-3 labels AND p_hat < 0.5. Prediction: keeps top-10 answer share <= 50% (R-Zero hit 72%)
   with no AVG cost; if AVG drops, the small-integer prior was doing real work and hypothesis (c) weakens.
5. *Well-posedness veto*: for a 10% audit sample per iter, ask the solver "is this question fully specified
   with a unique answer? yes/no" (or Claude-grade); track the rate as a dashboard metric alongside p_hat.
   Prediction: DEO's rate is flat ~80%, R-Zero-style drift would show as monotone decline — an early-warning
   metric p_hat provably cannot provide (R-Zero's mean p_hat moved 0.524 -> 0.527 while wellposedness collapsed).

## Methodological caveats

- All classifiers are regex/keyword-based: topic labels are crude (multi-topic questions take the
  priority-order first hit; "other" absorbs misses), narrative detection uses a fixed name/story lexicon,
  and suspicious-construct patterns undercount paraphrases. Trends across iters are far more trustworthy
  than absolute levels.
- Near-duplicate detection (8-gram Jaccard >= 0.5) is conservative; paraphrase-level duplication is not
  measured. First-80-char collision overcounts "same opening, different question" — for R-Zero it is best
  read as a templating metric, not a duplication metric.
- Cross-iter near-dup used 800x800 samples (seeded); within-iter shingle buckets skip 8-grams shared by >50
  questions, which can miss duplicates inside extremely large template families (i.e., R-Zero's true iter5
  near-dup rate is likely higher than the 0.2% reported).
- Qualitative reads are 15 questions/method/iter — directional, not statistically powered; well-posedness
  fractions quoted there have ~+-13pt binomial error.
- p_hat semantics are identical (solver majority-vote self-consistency) but the solvers differ (each method's
  own checkpoint lineage), so like-for-like histograms compare each solver's view of its own data, not one
  judge's view of both.
- Answer-format classifier misclassifies some LaTeX edge cases ("other" ~1%); label-correctness is NOT
  directly measured here — statements about correctness cite PSEUDO_LABEL_QUALITY.md or are flagged as
  plausibility judgments.
- DEO filtered files were confirmed to be a strict subset of the p_hat-[0.3,0.8] pool slice (~150/iter
  additionally removed by DEO's own pipeline); analyses of "DEO" use the filtered files throughout.

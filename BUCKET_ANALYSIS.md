# Controller band vs filter band: the p̂∈[0.2,0.3) gap (4B)

Breaks each iter's 2000-q MCMC pool into cumulative counts to test whether the strong-β
controller is "fooled" — reporting in-band success on questions the p̂ filter then deletes.

## Definitions

- `r_unc = 1 − 2·|p̂ − 0.5|`  → **r_unc ∈ [0.4,1] ⟺ p̂ ∈ [0.2,0.8]** (symmetric around 0.5)
- controller optimizes in-band on **r_unc ∈ [0.4,1]** (band = DEO_RMIN/RMAX = 0.4/1.0)
- training filter keeps **p̂ ∈ [0.3,0.8]** (asymmetric: low cut 0.3, high cut 0.8)
- **GAP = p̂ ∈ [0.2,0.3)** = questions the controller counts as in-band but the filter deletes
  (= r_unc-band minus filter-band, since both share the 0.8 high cut)

## Per-iter breakdown (N=2000/iter)

| run | iter | r_unc∈[.4,1] | p̂∈[.3,.8] | +pseudo | GAP[.2,.3) | p̂=0.25 | gap %N | gap %r_unc |
|--|--|--|--|--|--|--|--|--|
| **heroic_eye** (full, CD n=12) | 1 | 1146 | 822 | 802 | 324 | 324 | 16.2% | 28.3% |
| | 2 | 1213 | 957 | 925 | 256 | 256 | 12.8% | 21.1% |
| | 3 | 1211 | 956 | 924 | 255 | 255 | 12.8% | 21.1% |
| | 4 | 1193 | 941 | 908 | 252 | 252 | 12.6% | 21.1% |
| | 5 | 1188 | 914 | 883 | 274 | 274 | 13.7% | 23.1% |
| **witty_soca** (warm, CD n=12) | 1 | 1183 | 905 | 875 | 278 | 278 | 13.9% | 23.5% |
| | 2 | 1157 | 853 | 822 | 304 | 304 | 15.2% | 26.3% |
| | 3 | 1171 | 881 | 859 | 290 | 290 | 14.5% | 24.8% |
| | 4 | 1188 | 854 | 827 | 334 | 334 | 16.7% | 28.1% |
| | 5 | 1210 | 884 | 858 | 326 | 326 | 16.3% | 26.9% |
| **willing_panda** (baseline, no walk, m=9) | 1 | 1419 | 1029 | 989 | 390 | 0 | 19.5% | 27.5% |
| | 2 | 1362 | 970 | 933 | 392 | 0 | 19.6% | 28.8% |
| | 3 | 1432 | 1044 | 1014 | 388 | 0 | 19.4% | 27.1% |
| | 4 | 1449 | 1006 | 957 | 443 | 0 | 22.1% | 30.6% |
| | 5 | 1384 | 987 | 951 | 397 | 0 | 19.9% | 28.7% |

(controller in-band matches the logs: e.g. heroic_eye iter1 r_unc-band 1146/2000 = 0.573 = logged in-band.)

p̂ histogram <0.35 (iter1):
- heroic_eye (CD n=12): 0.083:360, 0.167:403, **0.25:324**, 0.333:218
- willing_panda (baseline m=9): 0.111:379, **0.222:390**, 0.333:278

## Findings

**1. The GAP is real and large, but baseline has MORE of it, not full-stack.**
Full-stack gap ≈13–16% of N; baseline ≈19–22%. The hypothesis that the full stack concentrates
extra quality into the deleted p̂∈[0.2,0.3) bucket (vs baseline) is **not supported** — directionally
the reverse. So the gap does not explain any full-stack shortfall relative to baseline.

**2. "p̂ = 0.25" is a CD-discretization artifact, not a signal.**
CD uses n=12 samples → p̂ = k/12; the interval [0.2,0.3) contains exactly one grid point, 3/12 = 0.25,
so the whole gap collapses onto 0.25 (that's why GAP == p̂=0.25 for the CD runs). Baseline uses m=9 →
p̂ = k/9; its gap sits at 2/9 = 0.222, so "p̂=0.25" is literally 0. Same phenomenon (modal count one
notch below the 0.3 cutoff), different sampling grid — the 0.25 count is **not comparable** across CD
vs non-CD runs.

**3. The controller IS systematically fooled — root cause: r_unc is symmetric, the filter is not.**
r_unc treats p̂=0.25 and p̂=0.75 identically (both = 0.5), but the filter deletes 0.25 and keeps 0.75.
So the controller optimizing r_unc∈[0.4,1] cannot distinguish the doomed "too-hard" p̂∈[0.2,0.3) from
the kept "good-hard" p̂∈(0.7,0.8]; driving β→0.02 to raise r_unc-in-band inevitably drags the doomed
mass along. Concretely for full-stack: logged in-band = 0.57–0.61, but filter-usable (p̂∈[0.3,0.8]/N)
is only **41–48%** — the ~13–16% difference (≈21–28% of the r_unc-band) is counted as success and then
filtered out. The controller over-reports success by ≈1/4.

## Implication

The fix is not to tune the gap but to **make the controller's constraint band match the filter**:
constrain the fraction with p̂∈[0.3,0.8] directly (an asymmetric band on p̂), rather than in-band on the
symmetric r_unc. As-is, the r_unc band structurally wastes ~1/4 of the controller's effort on the
p̂<0.3 region the filter always discards.

Data: `azureml/collect_4b_pools.sh` (mounts the 3 job outputs) → `tmp/analyze_buckets.py`.
Source runs: heroic_eye_27t7rwjp20, witty_soca_n1qkh3yjpp, willing_panda_k1zb5m1s59.

#!/usr/bin/env python3
"""Break each iter's 2000-q pool into the four cumulative counts + the p_hat=0.25 gap bucket.
r_unc = 1-2|p_hat-0.5|; controller band r_unc in [0.4,1] <=> p_hat in [0.2,0.8];
filter band p_hat in [0.3,0.8].  GAP = p_hat in [0.2,0.3): controller-in-band but filter-deleted."""
import json, glob, os, re
from collections import Counter

RUNS = {"heroic_eye (full-stack, CD n=12)": "tmp/collect_4b_dl/named-outputs/out/heroic_eye",
        "witty_soca (warm-start)":          "tmp/collect_4b_dl/named-outputs/out/witty_soca",
        "willing_panda (baseline, no walk)":"tmp/collect_4b_dl/named-outputs/out/willing_panda"}

def phat_of(d):
    return float(d.get("p_hat"))
def runc_of(d):
    r = d.get("r_unc")
    return float(r) if r is not None else 1 - 2*abs(phat_of(d)-0.5)

for name, folder in RUNS.items():
    files = sorted(glob.glob(f"{folder}/mcmc_iter_*.json"),
                   key=lambda p: int(re.search(r"iter_(\d+)", p).group(1)))
    if not files:
        print(f"\n### {name}: NO FILES in {folder}"); continue
    print(f"\n### {name}")
    print(f"{'iter':4s} {'N':>4s} | {'runc[.4,1]':>10s} {'phat[.3,.8]':>11s} {'+pseudo':>7s} | "
          f"{'GAP[.2,.3)':>10s} {'phat=.25':>8s} | {'gap%ofN':>7s} {'gap%ofRunc':>10s}")
    for f in files:
        data = json.load(open(f))
        N = len(data)
        n_runc = n_filt = n_pseudo = n_gap = n_025 = 0
        for d in data:
            ph = phat_of(d); ru = runc_of(d)
            in_runc = 0.4 <= ru <= 1.0
            in_filt = 0.3 <= ph <= 0.8
            if in_runc: n_runc += 1
            if in_filt: n_filt += 1
            if in_filt and d.get("pseudo_label") not in (None,"","None"): n_pseudo += 1
            if 0.2 <= ph < 0.3: n_gap += 1
            if abs(ph-0.25) < 1e-6: n_025 += 1
        it = re.search(r"iter_(\d+)", f).group(1)
        gpN  = 100*n_gap/N if N else 0
        gpR  = 100*n_gap/n_runc if n_runc else 0
        print(f"{it:>4s} {N:>4d} | {n_runc:>10d} {n_filt:>11d} {n_pseudo:>7d} | "
              f"{n_gap:>10d} {n_025:>8d} | {gpN:>6.1f}% {gpR:>9.1f}%")
    # phat value histogram (iter1) to see where CD's discrete p_hat lands
    d1 = json.load(open(files[0]))
    hist = Counter(round(phat_of(x),3) for x in d1)
    lo = sorted(v for v in hist if v < 0.35)
    print(f"     p_hat hist (iter1, values<0.35): " +
          ", ".join(f"{v}:{hist[v]}" for v in lo))

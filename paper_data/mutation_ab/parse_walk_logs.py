#!/usr/bin/env python3
"""Parse DEO MCMC walk logs (mutation-prompt A/B arms) into per-proposal JSONL.

Walk-log block format (mcmc_deo_vllm.py):
  [Step S | Batch B] Result: ACCEPTED|REJECTED | Strategy: X
  --- [OLD QUESTION] ---
  <text>
  --- [NEW QUESTION] ---
  <text>
  r_unc:        0.1234 -> 0.5678
  r_rep[k]:     0.0010 -> 0.0010  (cluster_size 2 -> 2)
  energy total: 123.4 -> 124.0  (dE=+0.6)
  Alpha:        0.9876
  ------------------------------------------------------------

Fields NOT persisted by these runs (emitted as null): question_id/pool index, mutator_gt,
old/new p_hat (only r_unc logged; p_hat = 0.5 ± r_unc/2 is ambiguous), per-proposal
pseudo_label, per-question solver sampled answers.

Final-state flags (passed_phat_filter / passed_validity_judge / entered_training) are joined
onto the LAST accepted proposal of each pool slot via exact question-text match against the
final pool json + filtered_*.json.
"""
import json, re, glob, os, sys

BLOCK_RE = re.compile(
    r"\[Step (\d+) \| Batch (\d+)\] Result: (ACCEPTED|REJECTED) \| Strategy: (\S+)\n"
    r"--- \[OLD QUESTION\] ---\n(.*?)\n--- \[NEW QUESTION\] ---\n(.*?)\n"
    r"r_unc:\s+([\d.]+) -> ([\d.]+)\n"
    r"r_rep\[k\]:\s+([\d.]+) -> ([\d.]+)\s+\(cluster_size (\d+) -> (\d+)\)\n"
    r"(?:sigma2D=[\d.]+\s+)?energy total:\s+(-?[\d.]+) -> (-?[\d.]+)\s+\(dE=([+-][\d.]+)\)\n"
    r"Alpha:\s+([\d.]+)\n",
    re.DOTALL)

def parse_run(run, folder, out_path):
    pool = {}      # final pool: question -> record
    filt = set()   # questions that entered training
    for f in glob.glob(f"{folder}/mcmc_iter_*.json"):
        pass  # handled per-iter below
    rows = []
    for logf in sorted(glob.glob(f"{folder}/mcmc_iter_*.log"),
                       key=lambda p: int(re.search(r"iter_(\d+)", p).group(1))):
        it = int(re.search(r"iter_(\d+)", logf).group(1))
        base = re.sub(r"\.log$", "", os.path.basename(logf)).replace("mcmc_", "")
        pool_f = glob.glob(f"{folder}/mcmc_iter_{it}_*.json")
        filt_f = glob.glob(f"{folder}/filtered_*_v{it}.json")
        pool_map = {}
        if pool_f:
            for d in json.load(open(pool_f[0])):
                pool_map[d["question"].strip()] = d
        train_set = set()
        if filt_f:
            for d in json.load(open(filt_f[0])):
                train_set.add(d["problem"].strip())
        txt = open(logf, encoding="utf-8", errors="ignore").read()
        for m in BLOCK_RE.finditer(txt):
            step, batch, res, strat, oldq, newq = m.group(1, 2, 3, 4, 5, 6)
            newq_s = newq.strip()
            fin = pool_map.get(newq_s)  # in final pool == survived to end of walk
            rows.append({
                "run": run, "iter": it, "mcmc_step": int(step), "batch": int(batch),
                "question_id": None,                       # pool index not logged
                "old_question": oldq.strip(), "new_question": newq_s,
                "strategy": strat if strat in "ABCDE" else None,
                "mutator_gt": None,                        # not logged in these runs
                "old_p_hat": None, "new_p_hat": None,      # only r_unc logged (p_hat=0.5±r_unc/2 ambiguous)
                "old_r_unc": float(m.group(7)), "new_r_unc": float(m.group(8)),
                "old_r_rep": float(m.group(9)), "new_r_rep": float(m.group(10)),
                "delta_energy": float(m.group(15)), "alpha": float(m.group(16)),
                "accepted": res == "ACCEPTED",
                # final-state joins (only meaningful if this proposal survived to end of iter)
                "in_final_pool": fin is not None,
                "final_p_hat": (float(fin["p_hat"]) if fin else None),
                "pseudo_label": (fin.get("pseudo_label") if fin else None),
                "passed_phat_filter": (0.3 <= float(fin["p_hat"]) <= 0.8 and
                                       fin.get("pseudo_label") not in (None, "", "None")) if fin else None,
                "entered_training": (newq_s in train_set) if fin else None,
            })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = sum(1 for r in rows if r["accepted"])
    print(f"{run}: {len(rows)} proposals, {acc} accepted ({100*acc/len(rows):.1f}%), -> {out_path}")
    return rows

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "tmp/mutab_logs_dl/named-outputs/out"
    os.makedirs("paper_data/mutation_ab", exist_ok=True)
    parse_run("V1", f"{src}/mutv1", "paper_data/mutation_ab/V1_proposals.jsonl")
    parse_run("V2", f"{src}/mutv2", "paper_data/mutation_ab/V2_proposals.jsonl")

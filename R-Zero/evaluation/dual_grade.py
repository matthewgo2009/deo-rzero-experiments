#!/usr/bin/env python3
"""Read-only DUAL grader: score each results_{dataset}.json under BOTH
   - ours  = gpt-4o-mini, boxed-only (recheck_mini.py logic), ~0% false-positive
   - paper = gpt-4o, full-text answer-vs-response (results_recheck.py logic), lenient
without mutating the source file. Both graders start from the SAME raw math_verify
scores and only bump entries with score < 0.5 (never lower a score), exactly like
the two original scripts. Averages drop the trailing {average_score} summary row.

Emits one line per (model, dataset) to regrade_compare.jsonl (cwd):
   {"model","dataset","n","raw","ours","paper"}   (accuracies in %)

Reads OpenAI key from tokens.json (cwd). STORAGE_PATH from env.
Usage: python evaluation/dual_grade.py --model_name <name_or_path> [--workers 16]
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

STORAGE_PATH = os.getenv("STORAGE_PATH")
DATASETS = ["math", "gsm8k", "amc", "minerva", "olympiad", "aime2024", "aime2025"]
_BOXED_OPEN_RE = re.compile(r"\\boxed\{")


def extract_last_boxed(text):
    if text is None:
        return None
    text = str(text)   # amc/aime ground-truth answers can be numeric, not str
    if not text:
        return None
    starts = [m.end() for m in _BOXED_OPEN_RE.finditer(text)]
    if not starts:
        return None
    pos, depth, out = starts[-1], 1, []
    while pos < len(text) and depth > 0:
        c = text[pos]
        if c == "{":
            depth += 1; out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
        pos += 1
    if depth != 0:
        return None
    return ("".join(out).strip()) or None


def ours_msg(model_response, ground_truth):
    """gpt-4o-mini boxed-only (recheck_mini.make_user_msg)."""
    m_boxed = extract_last_boxed(model_response)
    gt_boxed = extract_last_boxed(ground_truth) or ground_truth
    m_part = m_boxed if m_boxed is not None else "(NO BOXED ANSWER FOUND IN MODEL RESPONSE)"
    return (f"Hi, there is an answer: {m_part},and the ground truth answer is: {gt_boxed},"
            "please check whether the answer is correct or not, and return the **only**Yes or No.")


def paper_msg(answer, response):
    """gpt-4o full-text (results_recheck.process_example: answer=GT soln, response=model gen)."""
    return (f"Hi, there is a answer: {str(answer)}\n\n, and the ground truth answer is: {str(response)}\n\n, "
            "please check whether the answer is correct or not, and return the **only** Yes or No.")


def load_key():
    with open("tokens.json") as f:
        tok = json.load(f).get("openai")
    if not tok:
        raise SystemExit("ERROR: openai key not set in tokens.json")
    return tok


def grade_dataset(client, eval_dir, label, dataset, workers, retries=4):
    path = f"{eval_dir}/results_{dataset}.json"
    if not os.path.exists(path):
        print(f"[dual {dataset}] MISSING {path}"); return None
    model_name = label
    with open(path) as f:
        results = json.load(f)
    entries = results[:-1] if (results and isinstance(results[-1], dict)
                               and "average_score" in results[-1]) else results
    n = len(entries)
    if n == 0:
        return None
    raw = sum(e["score"] for e in entries) / n
    fail_idx = [i for i, e in enumerate(entries) if e["score"] < 0.5]

    def _call(grader, idx):
        e = entries[idx]
        msg = (ours_msg(e["response"], e["answer"]) if grader == "ours"
               else paper_msg(e["answer"], e["response"]))
        model = "gpt-4o-mini" if grader == "ours" else "gpt-4o"
        for attempt in range(retries):
            try:
                r = client.chat.completions.create(
                    model=model, max_tokens=4, temperature=0.1,
                    messages=[{"role": "system", "content": "You are a math answer checker."},
                              {"role": "user", "content": msg}])
                return grader, idx, ("yes" in r.choices[0].message.content.strip().lower())
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return grader, idx, False

    ours_bump = paper_bump = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_call, g, i) for g in ("ours", "paper") for i in fail_idx]
        for fut in as_completed(futs):
            g, idx, ok = fut.result()
            if ok:
                if g == "ours":
                    ours_bump += 1
                else:
                    paper_bump += 1
    base_correct = n * raw
    ours = (base_correct + ours_bump) / n
    paper = (base_correct + paper_bump) / n
    rec = {"model": model_name, "dataset": dataset, "n": n,
           "raw": round(raw * 100, 2), "ours": round(ours * 100, 2),
           "paper": round(paper * 100, 2)}
    with open("regrade_compare.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[dual {dataset:9s}] n={n:4d} raw={rec['raw']:5.1f} ours={rec['ours']:5.1f} "
          f"paper={rec['paper']:5.1f}  (fails={len(fail_idx)}, +ours{ours_bump}/+paper{paper_bump})")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", help="reconstruct eval dir as $STORAGE_PATH/evaluation/<name_/>")
    ap.add_argument("--eval_dir", help="grade this dir of results_<ds>.json directly (robust to path renaming)")
    ap.add_argument("--label", help="model label for output when --eval_dir is used")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    client = openai.OpenAI(api_key=load_key())
    if args.eval_dir:
        eval_dir = args.eval_dir.rstrip("/")
        label = args.label or os.path.basename(eval_dir)
    else:
        eval_dir = f"{STORAGE_PATH}/evaluation/{args.model_name.replace('/', '_')}"
        label = args.model_name
    print(f"=== dual_grade {label} (dir={eval_dir}) ===")
    for ds in DATASETS:
        grade_dataset(client, eval_dir, label, ds, args.workers)


if __name__ == "__main__":
    main()

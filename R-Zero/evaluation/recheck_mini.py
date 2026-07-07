#!/usr/bin/env python3
"""GPT-4o-mini boxed-only secondary verification, generalized to ANY dataset.

Generalization of results_recheck_math500_mini.py: instead of hardcoding
results_math.json, takes --dataset and rechecks results_{dataset}.json.
Same grader as the DEO/R-Zero fair comparison:
  - gpt-4o-mini (cheap)
  - extract \\boxed{...} from BOTH model response and ground truth, send only
    those to GPT (boxed-only -> ~0% false-positive on Qwen3-4B-Base outputs)
  - bumps mathruler/math_verify-failed entries (score < 0.5) to 1.0 if GPT says
    the boxed answers are equivalent
  - rewrites results_{dataset}.json in place with bumped scores + new average

Reads OpenAI key from tokens.json (cwd, same as generate.py).
Intended for the 7 math benchmarks (math gsm8k amc minerva olympiad
aime2024 aime2025) whose answers are all \\boxed-style. NOT for multiple-choice
sets (mmlu_pro/super_gpqa), which use exact-match, not GPT.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

STORAGE_PATH = os.getenv("STORAGE_PATH")
_BOXED_OPEN_RE = re.compile(r"\\boxed\{")


def extract_last_boxed(text):
    if not text:
        return None
    starts = [m.end() for m in _BOXED_OPEN_RE.finditer(text)]
    if not starts:
        return None
    pos = starts[-1]
    depth = 1
    out = []
    while pos < len(text) and depth > 0:
        c = text[pos]
        if c == "{":
            depth += 1
            out.append(c)
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


def make_user_msg(model_response, ground_truth):
    m_boxed = extract_last_boxed(model_response)
    gt_boxed = extract_last_boxed(ground_truth) or ground_truth
    m_part = m_boxed if m_boxed is not None else "(NO BOXED ANSWER FOUND IN MODEL RESPONSE)"
    return (
        f"Hi, there is an answer: {m_part},"
        f"and the ground truth answer is: {gt_boxed},"
        "please check whether the answer is correct or not, "
        "and return the **only**Yes or No."
    )


def load_openai_key():
    with open("tokens.json") as f:
        tok = json.load(f).get("openai")
    if not tok or tok.startswith("your") or tok == "":
        raise SystemExit("ERROR: openai key not set in tokens.json")
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    results_path = (
        f"{STORAGE_PATH}/evaluation/{args.model_name.replace('/', '_')}/results_{args.dataset}.json"
    )
    if not os.path.exists(results_path):
        sys.exit(f"ERROR: {results_path} not found")

    client = openai.OpenAI(api_key=load_openai_key())
    with open(results_path) as f:
        results = json.load(f)
    eval_entries = results[:-1]
    n = len(eval_entries)
    old_avg = results[-1]["average_score"]
    fail_idx = [i for i, e in enumerate(eval_entries) if e["score"] < 0.5]

    def _check(idx):
        e = eval_entries[idx]
        last_exc = None
        for attempt in range(args.retries):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a math answer checker."},
                        {"role": "user", "content": make_user_msg(e["response"], e["answer"])},
                    ],
                    max_tokens=4,
                    temperature=0.1,
                )
                text = resp.choices[0].message.content.strip().lower()
                return idx, ("yes" in text), None
            except Exception as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        return idx, False, type(last_exc).__name__

    t0 = time.time()
    bumped = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_check, i) for i in fail_idx]
        for fut in as_completed(futs):
            idx, is_correct, err = fut.result()
            if err is not None:
                n_err += 1
            if is_correct:
                eval_entries[idx]["score"] = 1.0
                bumped += 1

    new_avg = (sum(e["score"] for e in eval_entries) / n) if n else 0.0
    out = eval_entries + [{"average_score": new_avg}]
    with open(results_path, "w") as f:
        json.dump(out, f, indent=4)

    with open("final_results.jsonl", "a") as f:
        json.dump({"model": args.model_name, "dataset": args.dataset,
                   "score": round(new_avg * 100, 2)}, f)
        f.write("\n")

    print(f"[recheck-mini {args.dataset}] bumped {bumped}/{len(fail_idx)}  "
          f"{old_avg:.4f} -> {new_avg:.4f} (d{new_avg - old_avg:+.4f}, "
          f"{time.time() - t0:.0f}s, api-errs={n_err})")


if __name__ == "__main__":
    main()

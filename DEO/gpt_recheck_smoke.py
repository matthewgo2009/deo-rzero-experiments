#!/usr/bin/env python3
"""Smoke test for GPT-4o-mini secondary verification on MATH-500 eval results.

Mirrors R-Zero's results_recheck.py logic but cleaner:
  - Iterates entries where mathruler gave score < 0.5
  - Asks gpt-4o-mini if model_answer ≡ ground_truth
  - If "yes", bumps score to 1.0
  - Reports old vs new average_score

Run against archived eval files to see how much GPT can recover before
we bolt this into eval_math500() in mcmc_deo_vllm.py.
"""
import json
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import re

import openai

SYSTEM_PROMPT = "You are a math answer checker."

_BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(text):
    """Pull the LAST \\boxed{...} content with balanced braces.
    Returns the inner string, or None if no well-formed \\boxed{...} exists.
    """
    if not text:
        return None
    starts = [m.end() for m in _BOXED_RE.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    out = []
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
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
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip() or None


def make_user_msg(model_response, ground_truth):
    """Build the GPT-judge prompt.

    Strategy: extract the LAST \\boxed{...} from BOTH model response and
    ground-truth text, then ask GPT to compare just those short snippets.
    This cuts confabulation from long reasoning traces (~33% FP in our
    smoke test with full-text prompts → expected ~10% FP with boxed-only).

    Falls back to full text if either side lacks a \\boxed{...} — in that
    case, GPT should usually say NO (no final answer = invalid).
    """
    model_boxed = extract_boxed(model_response)
    gt_boxed = extract_boxed(ground_truth) or ground_truth  # GT is sometimes the raw answer

    if model_boxed is None:
        # Model didn't produce a boxed final answer → it's almost certainly wrong
        # for MATH-500. Surface this clearly to GPT.
        model_part = "(NO BOXED ANSWER FOUND IN MODEL RESPONSE)"
    else:
        model_part = model_boxed

    # User-specified template (matches R-Zero's results_recheck.py verbatim).
    # {answer}   slot <- model-generated solution (boxed only)
    # {response} slot <- ground-truth answer from the benchmark (boxed only)
    return (
        f"Hi, there is an answer: {model_part},"
        f"and the ground truth answer is: {gt_boxed},"
        "please check whether the answer is correct or not, "
        "and return the **only**Yes or No."
    )


def check_one(client, entry, retries=3):
    """Return (is_correct, status) where status is 'ok' / 'err:<reason>'."""
    last_exc = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": make_user_msg(entry["response"], entry["answer"])},
                ],
                max_tokens=4,
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip().lower()
            return ("yes" in text), "ok"
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
    return False, f"err:{type(last_exc).__name__}:{str(last_exc)[:80]}"


def recheck(path, api_key, max_workers=16, dry=False):
    with open(path) as f:
        results = json.load(f)
    if not results or "average_score" not in results[-1]:
        raise ValueError(f"unexpected results format: {path}")
    eval_entries = results[:-1]
    n = len(eval_entries)
    old_avg = results[-1]["average_score"]

    fail_idx = [i for i, e in enumerate(eval_entries) if e["score"] < 0.5]
    if dry:
        print(f"  {path}\n    n={n} old_avg={old_avg:.4f} score<0.5: {len(fail_idx)}  (dry, no API calls)")
        return None

    client = openai.OpenAI(api_key=api_key)
    bumped = 0
    err_counts = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_i = {ex.submit(check_one, client, eval_entries[i]): i for i in fail_idx}
        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            is_correct, status = fut.result()
            if status != "ok":
                err_counts[status] = err_counts.get(status, 0) + 1
            if is_correct:
                eval_entries[i]["score"] = 1.0
                bumped += 1
    elapsed = time.time() - t0
    new_avg = sum(e["score"] for e in eval_entries) / n
    delta = new_avg - old_avg
    # Persist the bumped scores + new average back to the file (idempotent —
    # re-running on already-rechecked file is a no-op since all bumped entries
    # already have score=1.0). Skipped if file is read-only or dry mode.
    try:
        results_out = eval_entries + [{"average_score": new_avg}]
        with open(path, "w") as f:
            json.dump(results_out, f, indent=4)
        wrote = "wrote"
    except OSError as exc:
        wrote = f"NO-WRITE ({type(exc).__name__})"
    print(f"  {path}")
    print(f"    n={n} old_avg={old_avg:.4f} -> new_avg={new_avg:.4f}  "
          f"(Δ={delta:+.4f}, bumped {bumped}/{len(fail_idx)}, {elapsed:.0f}s) [{wrote}]")
    if err_counts:
        print(f"    ERRORS: {err_counts}")
    return {"path": path, "n": n, "old": old_avg, "new": new_avg,
            "bumped": bumped, "n_fail": len(fail_idx), "errors": err_counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry", action="store_true",
                    help="Report counts only, no API calls")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not args.dry and not api_key:
        sys.exit("ERROR: OPENAI_API_KEY env var not set")

    summary = []
    for p in args.paths:
        s = recheck(p, api_key, max_workers=args.workers, dry=args.dry)
        if s:
            summary.append(s)
    if summary:
        print("\n=== summary ===")
        for s in summary:
            print(f"  {s['old']:.4f} -> {s['new']:.4f}  (Δ={s['new']-s['old']:+.4f}, "
                  f"{s['bumped']} bumped)  {s['path'].split('/')[-2]}")


if __name__ == "__main__":
    main()

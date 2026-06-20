"""GPT-5 as independent oracle: solve each of iter 1's 924 problems and
compare its boxed answer with solver_v1's M-vote majority. Then correlate
correctness with confidence (mean per-token log prob of trajectories
that produced the majority).
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

INPUT_PATH = "/eph/nvme0/yyd/DEO/iter1_solverv1_confidence.json"
OUTPUT_PATH = "/eph/nvme0/yyd/DEO/iter1_gpt5_graded.json"

assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY env var"
client = openai.OpenAI(timeout=180.0)

SYSTEM = (
    r"You are a competition math problem solver. Solve the problem step by step "
    r"and put your final answer in \boxed{}. Output the boxed answer always."
)

_BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(text):
    if not text:
        return None
    starts = [m.end() for m in _BOXED_RE.finditer(text)]
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


def norm(s):
    """Loose normalization for math-answer string comparison."""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    s = (s.replace("\\dfrac", "\\frac")
           .replace("\\tfrac", "\\frac")
           .replace("\\,", "").replace("\\!", "")
           .replace("\\left", "").replace("\\right", "")
           .replace("$", ""))
    return s.lower()


def solve_with_gpt5(problem, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": problem},
                ],
                max_completion_tokens=10000,
                reasoning_effort="low",
            )
            return r.choices[0].message.content or "", r.usage.completion_tokens
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    return f"<ERR:{type(last_err).__name__}:{str(last_err)[:120]}>", 0


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")

    # Skip extremely long trajectories field — only carry summary into output
    entries_lean = [
        {
            "problem": e["problem"],
            "stored_answer": e["stored_answer"],
            "majority_answer": e["majority_answer"],
            "majority_count": e["majority_count"],
            "p_hat_now": e["p_hat_now"],
            "confidence_sum_logprob_mean": e["confidence_sum_logprob_mean"],
            "confidence_per_token_logprob_mean": e["confidence_per_token_logprob_mean"],
        }
        for e in data
    ]

    def task(idx_entry):
        idx, entry = idx_entry
        text, tokens = solve_with_gpt5(entry["problem"])
        return idx, text, tokens

    results = [None] * len(entries_lean)
    t0 = time.time()
    n_err = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(task, (i, e)) for i, e in enumerate(entries_lean)]
        done = 0
        for fut in as_completed(futs):
            idx, text, tokens = fut.result()
            results[idx] = (text, tokens)
            done += 1
            if isinstance(text, str) and text.startswith("<ERR:"):
                n_err += 1
            if done % 50 == 0:
                print(f"  {done}/{len(entries_lean)}  ({time.time() - t0:.0f}s, errs={n_err})")

    out = []
    for i, e in enumerate(entries_lean):
        text, tokens = results[i]
        gpt5_boxed = extract_boxed(text)
        majority = e["majority_answer"]
        is_correct = (
            majority is not None
            and gpt5_boxed is not None
            and norm(majority) == norm(gpt5_boxed)
        )
        out.append({
            **e,
            "gpt5_answer": gpt5_boxed,
            "gpt5_response_tail": text[-400:] if isinstance(text, str) else None,
            "gpt5_tokens": tokens,
            "is_correct_vs_gpt5": is_correct,
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUTPUT_PATH}")
    print()

    # Correlation analysis
    valid = [e for e in out if e["majority_answer"] is not None
             and e["confidence_per_token_logprob_mean"] is not None
             and e["gpt5_answer"] is not None]
    print(f"=== analysis on {len(valid)}/{len(out)} entries ===")
    n_correct = sum(1 for e in valid if e["is_correct_vs_gpt5"])
    print(f"  solver_v1 majority correct (vs gpt-5):  {n_correct}/{len(valid)}  ({n_correct/len(valid)*100:.1f}%)")
    print()

    correct = [e for e in valid if e["is_correct_vs_gpt5"]]
    incorrect = [e for e in valid if not e["is_correct_vs_gpt5"]]
    print(f"  mean per_token_logprob (correct):   {sum(e['confidence_per_token_logprob_mean'] for e in correct)/len(correct):.4f}")
    print(f"  mean per_token_logprob (incorrect): {sum(e['confidence_per_token_logprob_mean'] for e in incorrect)/len(incorrect):.4f}")
    print(f"  mean sum_logprob       (correct):   {sum(e['confidence_sum_logprob_mean'] for e in correct)/len(correct):.2f}")
    print(f"  mean sum_logprob       (incorrect): {sum(e['confidence_sum_logprob_mean'] for e in incorrect)/len(incorrect):.2f}")
    print()

    print("=== correctness rate by confidence quintile (sorted high → low) ===")
    sorted_valid = sorted(valid, key=lambda x: x["confidence_per_token_logprob_mean"], reverse=True)
    nq = 5
    bs = len(sorted_valid) // nq
    print(f"  {'quintile':>10s} {'lp range':>22s} {'n':>5s} {'#correct':>10s} {'%correct':>10s}")
    for q in range(nq):
        s, e_ = q * bs, (q + 1) * bs if q < nq - 1 else len(sorted_valid)
        bucket = sorted_valid[s:e_]
        nc = sum(1 for x in bucket if x["is_correct_vs_gpt5"])
        lp_max = bucket[0]["confidence_per_token_logprob_mean"]
        lp_min = bucket[-1]["confidence_per_token_logprob_mean"]
        print(f"  {q+1}/{nq}        [{lp_min:.3f},{lp_max:.3f}] {len(bucket):>5d} {nc:>10d} {nc/len(bucket)*100:>9.1f}%")

    print()
    print("=== correctness rate by majority_count ===")
    for mc in sorted(set(e["majority_count"] for e in valid)):
        bucket = [e for e in valid if e["majority_count"] == mc]
        nc = sum(1 for x in bucket if x["is_correct_vs_gpt5"])
        print(f"  maj_count={mc}: {nc}/{len(bucket)} correct ({nc/len(bucket)*100:.1f}%), "
              f"mean per_token_logprob={sum(e['confidence_per_token_logprob_mean'] for e in bucket)/len(bucket):.4f}")


if __name__ == "__main__":
    main()

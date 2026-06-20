#!/usr/bin/env python3
"""Smoke test for two-stage validity filtering on MCMC datasets.

Stage 1: regex/substring BAD_PATTERNS (cheap, deterministic).
Stage 2: LLM judge via current solver vllm at http://localhost:8001 (chat completion).

For each dataset, reports:
  - regex drops (out of 1500 total)
  - p_hat ∈ [0.3, 0.8] survivors (what would normally enter training)
  - judge drops on those survivors
  - final clean count
  - sample dropped entries (so user can eyeball the filter calls)

Run:
  python3 validity_smoke_test.py 1          # iter 1 only
  python3 validity_smoke_test.py all        # all four iters
  python3 validity_smoke_test.py 1 --quick  # 100-entry subsample of iter 1
"""
import json
import re
import sys
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Stage 1: regex / keyword filter
# ============================================================
BAD_PATTERNS = [
    "[the NEW mutated problem statement]",
    "[Write the full problem statement",
    "[Insert the challenging",
    "final_answer",
    "Assistant:",
    "User:",
    "<strategy>",
    "<question>",
    "</question>",
    "```html",
    "```python",
    "```math",
    "MongoClient",
    "disable_shortcode",
    "GuestRemoved",
    "\\[最终答案",
    "primitive_helper",
    "primitive-item",
    "mid-translate",
    "my response was",
    "\\boxed{}",
    "prove that",
    "show that",
]
BAD_RE = re.compile("|".join(re.escape(p) for p in BAD_PATTERNS), re.IGNORECASE)


def regex_invalid_marker(q):
    """Return matched bad pattern, or None if clean."""
    m = BAD_RE.search(q)
    return m.group(0) if m else None


# ============================================================
# Stage 2: LLM judge via vllm_solver completion API
#
# We use raw /v1/completions with a fill-in-the-blank template rather than
# chat.completions because:
#   - solver_v3 is GRPO-trained: chat-mode triggers verbose "reason then answer"
#     traces and rarely emits a clean one-word verdict.
#   - completion-mode lets us constrain max_tokens to ~6 and read whichever of
#     "VALID" / "INVALID" appears first in the continuation.
# ============================================================
JUDGE_TEMPLATE = """You are a strict validity filter for competition-math problems.

Mark a problem as INVALID ONLY IF you are CONFIDENT one of these specific issues applies:
- Contains placeholder/template text like "[the NEW mutated problem statement]", "final_answer", or "[Write the full problem statement]".
- Contains visible prompt leakage, role tags (Assistant:, User:), HTML/XML tags (<question>, <strategy>), or markdown code fences (```).
- Contains corrupted Unicode garbage, randomly-mixed unrelated languages, or web/chat artifacts.
- Is clearly NOT a math problem.
- Asks for a proof, justification, true/false, yes/no, or open-ended explanation.
- The proposed boxed answer is missing or empty.

When in doubt, the problem is VALID. Do NOT solve the problem. Do NOT reject for being hard, unusual, or having minor formatting quirks.

Problem:
<<<
{q}
>>>

Proposed boxed answer: {ans}

Verdict ("VALID" or "INVALID"):"""


SOLVER_URL = "http://localhost:8001/v1/completions"
SOLVER_MODEL = "Qwen/Qwen3-4B-Base"


def judge_one(question, answer, timeout=60):
    """Returns (is_valid: bool, raw_text: str)."""
    ans = answer if answer not in (None, "") else "(none)"
    prompt = JUDGE_TEMPLATE.format(q=question, ans=ans)
    payload = {
        "model": SOLVER_MODEL,
        "prompt": prompt,
        "max_tokens": 6,
        "temperature": 0,
    }
    r = requests.post(SOLVER_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    text = r.json()["choices"][0]["text"].strip().upper()
    # The model may emit " VALID" / " INVALID" / "INVALID\n\nReason..." — look at first word.
    # NOTE: check "INVALID" before "VALID" because "VALID" is a substring of "INVALID".
    if "INVALID" in text.split()[0] if text.split() else "":
        return False, text
    if text.split() and text.split()[0].startswith("VALID"):
        return True, text
    # Fallback: scan first 12 chars for whichever appears.
    head = text[:12]
    if "INVALID" in head:
        return False, text
    if "VALID" in head:
        return True, text
    # Default: treat unparseable as VALID so we don't drop on noise.
    return True, f"<UNPARSEABLE:{text!r}>"


# ============================================================
# Pipeline simulation (matches what would go into training)
# ============================================================
def simulate_pipeline(entries, max_workers=24):
    """Mirror production order:
    1) regex on all entries (drops everything early)
    2) p_hat filter [0.3, 0.8] on regex survivors
    3) LLM judge on p_hat survivors → final training set
    Returns dict with per-stage counts and dropped-entry samples.
    """
    n_total = len(entries)

    # Stage 1
    regex_drops = []
    after_regex_idx = []
    for i, e in enumerate(entries):
        m = regex_invalid_marker(e["question"])
        if m is not None:
            regex_drops.append((i, e, m))
        else:
            after_regex_idx.append(i)

    # Stage 1.5: p_hat filter (would happen in filter_and_push)
    after_phat_idx = []
    for i in after_regex_idx:
        ph = entries[i].get("p_hat")
        if ph is not None and 0.3 <= ph <= 0.8:
            after_phat_idx.append(i)

    # Stage 2: LLM judge
    judge_results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                judge_one,
                entries[i]["question"],
                entries[i].get("pseudo_label") or entries[i].get("gt"),
            ): i
            for i in after_phat_idx
        }
        done = 0
        for f in as_completed(futures):
            idx = futures[f]
            try:
                judge_results[idx] = f.result()
            except Exception as exc:
                # Treat judge errors as VALID so we don't kill data on network blip.
                judge_results[idx] = (True, f"<ERR:{type(exc).__name__}>")
            done += 1
            if done % 100 == 0:
                print(f"    judge progress {done}/{len(after_phat_idx)} "
                      f"({time.time() - t0:.0f}s elapsed)")

    judge_drops = [
        (i, entries[i], judge_results[i][1])
        for i in after_phat_idx
        if not judge_results[i][0]
    ]
    survivors_idx = [i for i in after_phat_idx if judge_results[i][0]]

    return {
        "n_total": n_total,
        "regex_drops": regex_drops,
        "after_regex": len(after_regex_idx),
        "after_phat": len(after_phat_idx),
        "judge_drops": judge_drops,
        "survivors": survivors_idx,
        "judge_seconds": time.time() - t0,
    }


def show_samples(label, drops, k=6):
    if not drops:
        print(f"\n  [{label}] no samples")
        return
    print(f"\n  [{label}] showing {min(k, len(drops))} of {len(drops)}:")
    for idx, e, info in drops[:k]:
        q = e["question"]
        ph = e.get("p_hat")
        pseudo = e.get("pseudo_label")
        prev = q[:240].replace("\n", " ⏎ ") + ("…" if len(q) > 240 else "")
        print(f"    [idx={idx} p_hat={ph} pseudo={pseudo!r}] marker={info!r}")
        print(f"      Q: {prev}")


def report_one(path, quick=False):
    print(f"\n========== {path} ==========")
    with open(path) as f:
        entries = json.load(f)
    if quick:
        entries = entries[:100]
        print(f"  (QUICK MODE: first 100 entries only)")
    r = simulate_pipeline(entries)

    n = r["n_total"]
    n_regex_dropped = len(r["regex_drops"])
    n_after_regex = r["after_regex"]
    n_after_phat = r["after_phat"]
    n_judge_dropped = len(r["judge_drops"])
    n_survivors = len(r["survivors"])

    print()
    print(f"  total entries:               {n}")
    print(f"  regex-dropped:               {n_regex_dropped:>5}  ({n_regex_dropped/n:.1%})")
    print(f"  after regex:                 {n_after_regex:>5}  ({n_after_regex/n:.1%})")
    print(f"  after p_hat ∈ [0.3, 0.8]:    {n_after_phat:>5}  ({n_after_phat/n:.1%})")
    print(f"  judge-dropped:               {n_judge_dropped:>5}  ({n_judge_dropped/max(n_after_phat,1):.1%} of post-phat)")
    print(f"  FINAL training set:          {n_survivors:>5}  ({n_survivors/n:.1%})")
    print(f"  (judge stage: {r['judge_seconds']:.0f}s for {n_after_phat} calls)")

    show_samples("REGEX drops", r["regex_drops"], k=6)
    show_samples("JUDGE drops", r["judge_drops"], k=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="1",
                    help="iter index (1-4) or 'all'")
    ap.add_argument("--quick", action="store_true",
                    help="only first 100 entries (sanity check)")
    args = ap.parse_args()

    datasets = [
        f"/eph/nvme0/yyd/DEO/datasets/mcmc_iter_{i}.json" for i in range(1, 5)
    ]
    if args.which == "all":
        paths = datasets
    else:
        paths = [datasets[int(args.which) - 1]]

    for path in paths:
        report_one(path, quick=args.quick)


if __name__ == "__main__":
    main()

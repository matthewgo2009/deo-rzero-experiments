"""Diagnostic: did GRPO iter 1 training actually align solver_v1 with the
iter 1 pseudo-labels?

For each question in iter 1's filtered training set (924 entries), we run
M-vote (M=9) with BOTH base (Qwen3-4B-Base) and solver_v1, then compare
the M-vote majority to the stored pseudo_label using mathruler.grade_answer.

If GRPO is working, expect:
  - Solver_v1 majority matches pseudo_label much more often than base does.
  - Solver_v1's p_hat (consistency rate) is much higher than original (which
    was in [0.3, 0.8] by filter design).

Uses existing vllm endpoints:
  - vllm_base   on http://localhost:8000  (Qwen3-4B-Base, always)
  - vllm_solver on http://localhost:8001  (currently loaded with solver_v1)

Run from the host as `python3 iter1_training_alignment.py`. Doesn't need
docker because we're just hitting HTTP endpoints + reading a JSON file.
"""
import json
import os
import re
import sys
import time
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set OPENAI_API_KEY (not needed here actually) — we just talk to local vllm.
TRAINING_SET = "/eph/nvme0/yyd/paper_data/DEO/filtered_datasets/filtered_deo_qwen3_4b_base_solver_v1.json"
BASE_URL = "http://localhost:8000/v1/completions"
SOLVER_URLS = [
    "http://localhost:8001/v1/completions",
    "http://localhost:8004/v1/completions",
    "http://localhost:8005/v1/completions",
]
MODEL_NAME = "Qwen/Qwen3-4B-Base"
M_SAMPLES = 9
SOLVER_TEMP = 1.0
SOLVER_TOP_P = 1.0
SOLVER_TOP_K = 40
SOLVER_MAX_TOKENS = 4096
SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


# ---- mathruler-style grade_answer + boxed extractor ----
_BOXED_OPEN_RE = re.compile(r"\\boxed\{")


def extract_boxed(text):
    if not text:
        return None
    starts = [m.end() for m in _BOXED_OPEN_RE.finditer(text)]
    if not starts:
        return None
    pos = starts[-1]
    depth, out = 1, []
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
    return ("".join(out).strip()) or None if depth == 0 else None


def grade_match(predicted_text, gt_answer):
    """Returns True if extracted boxed answer of predicted_text matches gt_answer
    by exact string after whitespace strip, or fallback simple normalization.
    For richer math equivalence we'd use mathruler.grade_answer, but it's not
    available in host Python — use string match as proxy (acceptable since
    pseudo_label was also extracted via the same boxed regex during generation)."""
    p = extract_boxed(predicted_text)
    if p is None:
        return False
    return p.strip().replace(" ", "") == str(gt_answer).strip().replace(" ", "")


def build_chat_prompt(question, system=SYSTEM_PROMPT):
    """Qwen3 chat template (matches what apply_chat_template would produce).
    Hard-coded since we run from host without transformers loaded."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def m_vote(endpoint_url, prompts, label):
    """Call vllm /v1/completions with `prompts` list. Returns list of texts."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompts,
        "max_tokens": SOLVER_MAX_TOKENS,
        "temperature": SOLVER_TEMP,
        "top_p": SOLVER_TOP_P,
        "top_k": SOLVER_TOP_K,
    }
    t0 = time.time()
    r = requests.post(endpoint_url, json=payload, timeout=600)
    r.raise_for_status()
    out = r.json()
    elapsed = time.time() - t0
    texts = [c["text"] for c in out["choices"]]
    print(f"  [{label}] {len(prompts)} prompts in {elapsed:.0f}s "
          f"({len(prompts)/elapsed:.1f} prompts/s)")
    return texts


def analyze(entries, model_name, endpoint_url=None, dp_urls=None, max_dispatch=2000):
    """For each entry, send M=9 prompts and compute (majority, p_hat, matches_pseudo)."""
    questions = [e["problem"] for e in entries]
    pseudo_labels = [e["answer"] for e in entries]
    prompts = [build_chat_prompt(q) for q in questions for _ in range(M_SAMPLES)]
    n_prompts = len(prompts)

    if endpoint_url and not dp_urls:
        # Single endpoint, chunked to avoid huge single POSTs
        all_texts = []
        for i in range(0, n_prompts, max_dispatch):
            chunk = prompts[i:i + max_dispatch]
            all_texts.extend(m_vote(endpoint_url, chunk, f"{model_name} chunk {i // max_dispatch + 1}"))
        texts = all_texts
    elif dp_urls:
        # DP across multiple endpoints
        n_dp = len(dp_urls)
        base, rem = divmod(n_prompts, n_dp)
        slices = []
        start = 0
        for i in range(n_dp):
            size = base + (1 if i < rem else 0)
            slices.append(prompts[start:start + size])
            start += size

        results = [None] * n_dp
        with ThreadPoolExecutor(max_workers=n_dp) as ex:
            futs = {ex.submit(m_vote, dp_urls[i], slices[i], f"{model_name} DP{i}"): i
                    for i in range(n_dp) if slices[i]}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        texts = [t for sub in results if sub for t in sub]
    else:
        raise ValueError("need endpoint_url or dp_urls")

    # Extract per-prompt boxed answers
    extracted = [extract_boxed(t) for t in texts]
    n_total = len(entries)
    n_match_pseudo = 0
    p_hats = []
    no_extract = 0
    for i in range(n_total):
        chunk = extracted[i * M_SAMPLES: (i + 1) * M_SAMPLES]
        valid = [a for a in chunk if a is not None]
        if not valid:
            no_extract += 1
            p_hats.append(0.0)
            continue
        major, count = Counter(valid).most_common(1)[0]
        p_hat = count / M_SAMPLES
        p_hats.append(p_hat)
        # Match against stored pseudo_label
        if major.strip().replace(" ", "") == str(pseudo_labels[i]).strip().replace(" ", ""):
            n_match_pseudo += 1

    return {
        "model": model_name,
        "n": n_total,
        "n_match_pseudo": n_match_pseudo,
        "match_rate": n_match_pseudo / n_total,
        "no_extract": no_extract,
        "mean_p_hat": sum(p_hats) / len(p_hats),
        "median_p_hat": sorted(p_hats)[len(p_hats) // 2],
        "p_hats": p_hats,
    }


def main():
    with open(TRAINING_SET) as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} iter-1 filtered training entries from {TRAINING_SET}")

    # Score distribution of stored pseudo_labels (these came from BASE at generation time)
    stored_scores = [e["score"] for e in entries]
    print(f"\nStored score stats (base M-vote consistency at iter 1 generation time):")
    print(f"  mean p_hat: {sum(stored_scores)/len(stored_scores):.3f}")
    print(f"  by bucket:")
    for k, c in sorted(Counter(round(s, 2) for s in stored_scores).items()):
        print(f"    {k:.2f}: {c}")

    print("\n=== Running base model M-vote on training set ===")
    base_result = analyze(entries, "base", endpoint_url=BASE_URL)
    print(f"\n  base match rate (now vs stored pseudo): "
          f"{base_result['n_match_pseudo']}/{base_result['n']} "
          f"({base_result['match_rate']*100:.1f}%)")
    print(f"  base mean p_hat now: {base_result['mean_p_hat']:.3f} "
          f"(was {sum(stored_scores)/len(stored_scores):.3f})")

    print("\n=== Running solver_v1 M-vote on training set (DP across 3 endpoints) ===")
    solver_result = analyze(entries, "solver_v1", dp_urls=SOLVER_URLS)
    print(f"\n  solver_v1 match rate (vs stored pseudo): "
          f"{solver_result['n_match_pseudo']}/{solver_result['n']} "
          f"({solver_result['match_rate']*100:.1f}%)")
    print(f"  solver_v1 mean p_hat: {solver_result['mean_p_hat']:.3f}")
    print(f"  solver_v1 no-extract: {solver_result['no_extract']}/{solver_result['n']}")

    # Summary
    print("\n" + "=" * 60)
    print("ITER 1 TRAINING ALIGNMENT SUMMARY")
    print("=" * 60)
    print(f"{'metric':<35s} {'base now':>10s} {'solver_v1':>10s}")
    print(f"{'match rate (vs stored pseudo)':<35s} "
          f"{base_result['match_rate']*100:>9.1f}% "
          f"{solver_result['match_rate']*100:>9.1f}%")
    print(f"{'mean p_hat (self-consistency)':<35s} "
          f"{base_result['mean_p_hat']:>10.3f} "
          f"{solver_result['mean_p_hat']:>10.3f}")
    print(f"{'median p_hat':<35s} "
          f"{base_result['median_p_hat']:>10.3f} "
          f"{solver_result['median_p_hat']:>10.3f}")
    print(f"{'no-extract count':<35s} "
          f"{base_result['no_extract']:>10d} "
          f"{solver_result['no_extract']:>10d}")

    out = {
        "training_set": TRAINING_SET,
        "n_entries": len(entries),
        "stored_pseudo_p_hat_mean": sum(stored_scores) / len(stored_scores),
        "base_now": {k: v for k, v in base_result.items() if k != "p_hats"},
        "solver_v1": {k: v for k, v in solver_result.items() if k != "p_hats"},
    }
    out_path = "/eph/nvme0/yyd/DEO/iter1_alignment_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull report saved: {out_path}")


if __name__ == "__main__":
    main()

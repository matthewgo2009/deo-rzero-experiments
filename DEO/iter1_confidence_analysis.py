"""
For each of iter 1's 924 filtered training questions, compute solver_v1's
M-vote majority answer AND a confidence measure for that answer.

Confidence = average log-probability of the trajectories (samples) that
produced the majority answer. We report both:
  - total_log_prob mean (sum over tokens, length-sensitive)
  - per_token_log_prob mean (length-normalized; -log perplexity per token)

Output: JSON list, one entry per question, including all 9 trajectory
metadata (extracted answer, num tokens, sum logprob) so the user can
inspect / re-analyze.
"""
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

# Sit alongside deo helpers
sys.path.insert(0, "/home/azureuser/yyd/DEO")
# Avoid importing mcmc_deo_vllm (loads transformers, slow). Re-implement
# minimal pieces inline.

ITER1_DATASET = "/eph/nvme0/yyd/paper_data/DEO/filtered_datasets/filtered_deo_qwen3_4b_base_solver_v1.json"
OUTPUT_PATH = "/eph/nvme0/yyd/DEO/iter1_solverv1_confidence.json"

SOLVER_URLS = [
    "http://localhost:8001/v1",
    "http://localhost:8004/v1",
    "http://localhost:8005/v1",
]
MODEL_NAME = "Qwen/Qwen3-4B-Base"
M_SAMPLES = 9
MAX_TOKENS = 4096
TEMP = 1.0
TOP_P = 1.0
TOP_K = 40
SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."

# Chunk to keep each HTTP call modest
CHUNK_QUESTIONS = 100  # 100 q × 9 = 900 prompts per shard call

_BOXED_OPEN_RE = re.compile(r"\\boxed\{")


def extract_boxed(text):
    """Last \\boxed{...} body with balanced braces."""
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


def extract_solver_answer(text):
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    fb = re.search(
        r"(?:answer is|equals|value is|is exactly)[\s:]*([0-9a-zA-Z\.\-\\\/]+)",
        text[-200:], re.IGNORECASE,
    )
    if fb:
        return fb.group(1).strip()
    return None


def chat_prompt(question):
    """Plain Qwen chat template assembled by hand to avoid loading tokenizer."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def score_one_shard(client, prompts):
    """Send `prompts` list to one solver, return list of dicts with
    {text, num_tokens, sum_logprob, mean_logprob_per_token}.
    """
    resp = client.completions.create(
        model=MODEL_NAME,
        prompt=prompts,
        max_tokens=MAX_TOKENS,
        temperature=TEMP,
        top_p=TOP_P,
        extra_body={"top_k": TOP_K},
        logprobs=1,  # return logprob of each emitted token
    )
    out = []
    for c in resp.choices:
        tokens = c.logprobs.token_logprobs if c.logprobs else []
        # token_logprobs[0] can be None in some setups; filter
        token_logprobs = [lp for lp in tokens if lp is not None]
        n = len(token_logprobs)
        sum_lp = float(sum(token_logprobs)) if n else 0.0
        out.append({
            "text": c.text,
            "num_tokens": n,
            "sum_logprob": sum_lp,
            "mean_logprob_per_token": (sum_lp / n) if n else 0.0,
        })
    return out


def run_chunked_dp(all_prompts, clients):
    """Split all_prompts across DP solvers, chunk inside each shard."""
    n_dp = len(clients)
    # Shard prompts by question (each question contributes M consecutive prompts;
    # we shard at the question level to keep all M samples per question on the
    # same shard, simplifying re-aggregation — but actually it doesn't matter
    # since we just need to recover per-question slices). For simplicity, split
    # contiguously and reconstruct by index.
    base_sz, rem = divmod(len(all_prompts), n_dp)
    slices = []
    start = 0
    for i in range(n_dp):
        size = base_sz + (1 if i < rem else 0)
        slices.append((start, start + size))
        start += size

    results_by_idx = [None] * len(all_prompts)

    def shard_worker(shard_idx):
        s, e = slices[shard_idx]
        if s == e:
            return shard_idx, []
        shard_prompts = all_prompts[s:e]
        shard_results = []
        client = clients[shard_idx]
        for c in range(0, len(shard_prompts), CHUNK_QUESTIONS * M_SAMPLES):
            sub = shard_prompts[c:c + CHUNK_QUESTIONS * M_SAMPLES]
            shard_results.extend(score_one_shard(client, sub))
        return shard_idx, shard_results

    with ThreadPoolExecutor(max_workers=n_dp) as ex:
        futs = [ex.submit(shard_worker, i) for i in range(n_dp)]
        for fut in as_completed(futs):
            shard_idx, shard_results = fut.result()
            s, e = slices[shard_idx]
            for j, r in enumerate(shard_results):
                results_by_idx[s + j] = r
    return results_by_idx


def main():
    print(f"Loading dataset: {ITER1_DATASET}")
    with open(ITER1_DATASET) as f:
        entries = json.load(f)
    print(f"  {len(entries)} questions")

    clients = [openai.OpenAI(api_key="EMPTY", base_url=url, timeout=1800.0)
               for url in SOLVER_URLS]

    # Build M prompts per question
    print(f"Building {len(entries) * M_SAMPLES} prompts (M={M_SAMPLES} samples per question)")
    all_prompts = []
    for e in entries:
        p = chat_prompt(e["problem"])
        for _ in range(M_SAMPLES):
            all_prompts.append(p)

    t0 = time.time()
    print(f"Running M-vote generation on {len(SOLVER_URLS)}-way DP solver_v1...")
    all_results = run_chunked_dp(all_prompts, clients)
    print(f"  done in {time.time() - t0:.0f}s")

    # Aggregate per question
    out = []
    n_no_majority = 0
    for i, entry in enumerate(entries):
        traj = all_results[i * M_SAMPLES:(i + 1) * M_SAMPLES]
        # Extract answer per trajectory
        for t in traj:
            t["extracted_answer"] = extract_solver_answer(t["text"])
        # Majority answer (among non-None extracts)
        valid = [t for t in traj if t["extracted_answer"] is not None]
        if not valid:
            n_no_majority += 1
            out.append({
                "problem": entry["problem"],
                "stored_answer": entry["answer"],
                "stored_p_hat": entry["score"],
                "majority_answer": None,
                "majority_count": 0,
                "p_hat_now": 0.0,
                "matching_traj_count": 0,
                "confidence_sum_logprob_mean": None,
                "confidence_per_token_logprob_mean": None,
                "trajectories": traj,
            })
            continue
        counter = Counter(t["extracted_answer"] for t in valid)
        major_ans, major_count = counter.most_common(1)[0]
        matching = [t for t in valid if t["extracted_answer"] == major_ans]
        sum_lp_mean = sum(t["sum_logprob"] for t in matching) / len(matching)
        per_tok_mean = sum(t["mean_logprob_per_token"] for t in matching) / len(matching)
        out.append({
            "problem": entry["problem"],
            "stored_answer": entry["answer"],
            "stored_p_hat": entry["score"],
            "majority_answer": major_ans,
            "majority_count": major_count,
            "p_hat_now": major_count / M_SAMPLES,
            "matching_traj_count": len(matching),
            "confidence_sum_logprob_mean": sum_lp_mean,
            "confidence_per_token_logprob_mean": per_tok_mean,
            "trajectories": traj,
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved per-question results to: {OUTPUT_PATH}")

    # Quick summary statistics
    valid_entries = [e for e in out if e["majority_answer"] is not None]
    print(f"\nSummary on {len(valid_entries)}/{len(out)} questions with valid majority:")
    print(f"  no-majority entries: {n_no_majority}")
    print(f"  mean confidence (sum_logprob_mean over majority traj): "
          f"{sum(e['confidence_sum_logprob_mean'] for e in valid_entries) / len(valid_entries):.2f}")
    print(f"  mean confidence (per_token_logprob_mean): "
          f"{sum(e['confidence_per_token_logprob_mean'] for e in valid_entries) / len(valid_entries):.4f}")
    print(f"  mean num_tokens per majority trajectory: "
          f"{sum(sum(t['num_tokens'] for t in e['trajectories']) / len(e['trajectories']) for e in valid_entries) / len(valid_entries):.1f}")


if __name__ == "__main__":
    main()

"""General-domain eval worker: ONE model x ONE dataset (MMLU-Pro / SuperGPQA / BBEH).

Project protocol (NOT a reproduction of R-Zero's exact Table-2 harness — fixed
subsets, greedy decoding, boxed-letter extraction; identical for every checkpoint,
so cross-checkpoint deltas are comparable):
  - MMLU-Pro: TIGER-Lab/MMLU-Pro test, stratified sample 3000 by category, seed 0
  - SuperGPQA: m-a-p/SuperGPQA, stratified sample 3000 by discipline, seed 0
  - BBEH: MrLight/bbeh-eval (BIG-Bench Extra Hard mirror), stratified 3000 by task
  - greedy (temp 0), max 4096 new tokens, R-Zero solver system prompt, answer in
    \\boxed{}; MC graded by option letter, BBEH by normalized exact match.
"""
import argparse
import json
import os
import random
import re


SYSTEM = r"Please reason step by step, and put your final answer within \boxed{}."
LETTERS = "ABCDEFGHIJ"


def _get(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    raise KeyError(f"none of {keys} in {list(d)[:12]}")


def _stratified(rows, key_fn, n, seed=0):
    from collections import defaultdict
    if len(rows) <= n:
        return rows
    by = defaultdict(list)
    for r in rows:
        by[key_fn(r)].append(r)
    rng = random.Random(seed)
    out, cats = [], sorted(by)
    quota = {c: max(1, int(round(n * len(by[c]) / len(rows)))) for c in cats}
    for c in cats:
        pool = by[c]
        rng.shuffle(pool)
        out.extend(pool[:quota[c]])
    rng.shuffle(out)
    return out[:n]


def load_items(dataset):
    """-> list of {prompt, target, kind} with a frozen subset."""
    from huggingface_hub import hf_hub_download
    items = []
    if dataset == "mmlu_pro":
        from datasets import load_dataset
        rows = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
        rows = _stratified(rows, lambda r: r["category"], 3000)
        for r in rows:
            opts = r["options"]
            items.append({"prompt": _mc_prompt(r["question"], opts),
                          "target": LETTERS[r["answer_index"]]
                          if "answer_index" in r else str(r["answer"]).strip(),
                          "kind": "mc"})
    elif dataset == "supergpqa":
        p = hf_hub_download("m-a-p/SuperGPQA", "SuperGPQA-all.jsonl",
                            repo_type="dataset")
        rows = [json.loads(l) for l in open(p)]
        rows = _stratified(rows, lambda r: r.get("discipline", r.get("field", "x")), 3000)
        for r in rows:
            items.append({"prompt": _mc_prompt(_get(r, "question"),
                                               _get(r, "options")),
                          "target": str(_get(r, "answer_letter", "answer")).strip(),
                          "kind": "mc"})
    elif dataset == "bbeh":
        p = hf_hub_download("MrLight/bbeh-eval", "train.jsonl", repo_type="dataset")
        rows = [json.loads(l) for l in open(p)]
        rows = _stratified(rows, lambda r: r.get("task", "x"), 3000)
        for r in rows:
            q = _get(r, "input", "question")
            items.append({"prompt": f"{q}\n\nPut your final answer within \\boxed{{}}.",
                          "target": str(_get(r, "target", "answer")).strip(),
                          "kind": "em"})
    else:
        raise ValueError(dataset)
    return items


def _mc_prompt(question, options):
    lines = [question, "", "Options:"]
    for i, o in enumerate(options):
        lines.append(f"{LETTERS[i]}. {o}")
    lines.append("")
    lines.append("Answer with the letter of the single correct option, "
                 "and put that letter within \\boxed{}.")
    return "\n".join(lines)


_BOXED = re.compile(r"\\boxed\{")


def last_boxed(text):
    starts = [m.end() for m in _BOXED.finditer(text or "")]
    for st in reversed(starts):
        depth = 1
        for i in range(st, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[st:i].strip()
    return None


def _norm(s):
    s = str(s).strip().strip(".").strip()
    s = re.sub(r"^\((.)\)$", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip("'\"` ")


def grade(response, target, kind):
    ans = last_boxed(response)
    if ans is None:
        m = re.search(r"answer is[:\s]*\(?([A-Ja-j0-9][^\n\.]{0,40})", (response or "")[-300:])
        ans = m.group(1) if m else None
    if ans is None:
        return 0
    if kind == "mc":
        m = re.match(r"^\(?([A-Ja-j])\)?\b", ans.strip())
        return int(bool(m) and m.group(1).upper() == target.upper())
    return int(_norm(ans) == _norm(target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = load_items(args.dataset)
    import vllm
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = vllm.LLM(model=args.model, tokenizer=args.model,
                     gpu_memory_utilization=0.85, max_model_len=16384,
                     dtype="bfloat16")
    sp = vllm.SamplingParams(max_tokens=4096, temperature=0.0,
                             stop_token_ids=[tokenizer.eos_token_id])
    if tokenizer.chat_template:
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": it["prompt"]}],
            tokenize=False, add_generation_prompt=True) for it in items]
    else:
        prompts = [f"system: {SYSTEM}\nuser: {it['prompt']}\n" for it in items]
    outs = model.generate(prompts, sampling_params=sp, use_tqdm=True)
    texts = [o.outputs[0].text for o in outs]
    scores = [grade(t, it["target"], it["kind"]) for t, it in zip(texts, items)]
    acc = 100.0 * sum(scores) / len(scores)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"label": args.label, "model": args.model, "dataset": args.dataset,
                   "n": len(items), "acc": round(acc, 2),
                   "results": [{"target": it["target"], "score": s,
                                "response_tail": t[-400:]}
                               for it, s, t in zip(items, scores, texts)]}, f)
    print(f"GEVAL_RESULT {json.dumps({'label': args.label, 'dataset': args.dataset, 'n': len(items), 'acc': round(acc, 2)})}",
          flush=True)


if __name__ == "__main__":
    main()

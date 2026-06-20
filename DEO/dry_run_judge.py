"""Dry-run the JUDGE_PROMPT on the 30 manually-verified questions from iter 1 + iter 2.

Compares vllm_base's verdict with my hand-graded ground truth.
Reports per-type accuracy + overall confusion matrix.
"""
import json, re, random, sys
import openai
from transformers import AutoTokenizer

# We use main run's vllm_base on port 8000 (Qwen3-4B-Base).
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME    = "Qwen/Qwen3-4B-Base"

JUDGE_PROMPT = """You are a strict reviewer of competition-math problems. Read the problem and decide if it is well-formed.

A problem is INVALID if it has any of:

(1) TRUNCATED — sentence cuts off mid-clause, ends with a preposition, or refers to "the above problem / the previous problem" without context.
(2) NONSENSE — contains random non-math words, gibberish phrases, undefined symbols, or template placeholders like "[Write the full problem...]".
(3) PROVE-TYPE — asks to "prove", "show", "explain why", "justify" rather than compute a single answer. (If it asks BOTH to prove AND to compute a number, treat as VALID.)
(4) YES/NO — asks "does there exist", "is it true", "determine whether" with yes/no answer.

Otherwise, the problem is VALID.

Reply with exactly one word: VALID or INVALID.

Examples:

Problem: Find the smallest positive integer n such that n^2 \\equiv 1 \\pmod{105}.
Verdict: VALID

Problem: Compute the sum of all real x satisfying x^4 - 5x^2 + 6 = 0.
Verdict: VALID

Problem: Let a, b, n be positive integers with n \\geq 3 and (1+a+b)^n = 1 + a^n + b^n. Find n.
Verdict: VALID

Problem: In a triangle with sides 13, 14, 15, find AE where the incircle touches CA, and prove BD = CF if isosceles.
Verdict: VALID

Problem: A right circular cone has base radius r and height h. An inscribed cylinder has height k*h. Find the cylinder's radius as a function of k.
Verdict: VALID

Problem: [the NEW mutated problem statement]
Verdict: INVALID

Problem: In an equilateral triangle inscribed in a circle, find AO if all angles are formed through neurological pulses such that no complex activities created before birth fundamentals.
Verdict: INVALID

Problem: Prove that for any prime p > 3, p^2 - 1 is divisible by 24.
Verdict: INVALID

Problem: In the above mutated problem: find how many such cubes have an odd number of diagonals.
Verdict: INVALID

Problem: A rectangular prism with coordinates (0,0,0), (a,0,0), (0,b,0), (0,0,c) and another identical prism centered at (a/2,b/2,c/2) creates a shape such that the distance between any point on the first prism and any point on the second is always less than the min(a,b,c). How many distinct integer values are
Verdict: INVALID

Problem: A geometric series composed of three interior anglespace has a countable sum. Calculate the joint degrees sum for the smallest inner anglespace given by x, 2x, 3x. \\boxed{janus}
Verdict: INVALID

Problem: Determine whether there exist positive integers (x, y) for which x^2 - xy + y^2 = 3.
Verdict: INVALID

Problem: {question}
Verdict:"""


# === Ground truth: manually graded ===
# (q_id, expected, type, comment)
# expected: True = VALID, False = INVALID
GT = {
    "1.1":  (True,  "valid",       "n,n^2+1 coprime → 0"),
    "1.2":  (True,  "valid",       "18π"),
    "1.3":  (True,  "valid",       "10 (interpret n in [1,10])"),
    "1.4":  (True,  "valid-double","6"),
    "1.5":  (True,  "valid-borderline","ambiguous 'sum of terms'"),
    "1.6":  (True,  "valid",       "y=2x"),
    "1.7":  (True,  "valid",       "9/4"),
    "1.8":  (True,  "valid",       "well-posed; pseudo wrong"),
    "1.9":  (True,  "valid",       "4"),
    "1.10": (False, "A-contradict","a>0 vs derived a=0"),
    "1.11": (False, "E-undefined", "monochromatic matching ill-defined"),
    "1.12": (True,  "valid",       "0"),
    "1.13": (True,  "valid-mixed", "prove + find, my rule says VALID"),
    "1.14": (False, "B-truncated", "in the above..."),
    "1.15": (False, "A-contradict","non-repeating but repeated"),

    "2.1":  (True,  "valid",       "60"),
    "2.2":  (True,  "valid",       "irrelevant noise but math is OK"),
    "2.3":  (True,  "valid",       "0"),
    "2.4":  (False, "C-nonsense",  "boundary where imaginary part transitions..."),
    "2.5":  (True,  "valid-mixed", "find+prove"),
    "2.6":  (False, "B-truncated", "cut off 'How many distinct integer values are'"),
    "2.7":  (True,  "valid",       "m=1-k"),
    "2.8":  (False, "A-contradict","BD!=BE violates tangent equality"),
    "2.9":  (True,  "valid",       "√7"),
    "2.10": (True,  "valid",       "∅"),
    "2.11": (False, "B-truncated", "ends with 'if...'"),
    "2.12": (True,  "valid",       "n≡1,3,5,7 mod 8"),
    "2.13": (True,  "valid",       "x=0"),
    "2.14": (True,  "valid-weird", "1"),
    "2.15": (False, "B-missing-R", "R never given"),
}


def load_questions():
    """Reproduce the same 15-per-iter random sample from earlier inspection."""
    out = {}
    for it in [1, 2]:
        with open(f'/storage/datasets/mcmc_iter_{it}.json') as f:
            data = json.load(f)
        filtered = [d for d in data if d.get('pseudo_label') not in (None, '', 'None') and 0.3 <= d['p_hat'] <= 0.8]
        random.seed(it * 17)
        sample = random.sample(filtered, 15)
        for i, d in enumerate(sample, 1):
            qid = f"{it}.{i}"
            out[qid] = d['question']
    return out


def parse_verdict(text):
    text = text.strip().upper()
    if re.search(r"\bINVALID\b", text):
        return False
    if re.search(r"\bVALID\b", text):
        return True
    return None  # ambiguous


def main():
    qs = load_questions()
    client = openai.OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)

    qids = list(qs.keys())
    prompts = [JUDGE_PROMPT.replace("{question}", qs[qid]) for qid in qids]
    print(f"Sending {len(prompts)} questions to {VLLM_BASE_URL} ({MODEL_NAME})")

    resp = client.completions.create(
        model=MODEL_NAME,
        prompt=prompts,
        max_tokens=8,
        temperature=0.0,
    )

    rows = []
    for qid, c in zip(qids, resp.choices):
        gt_valid, gt_type, _ = GT[qid]
        verdict_raw = c.text.strip()
        pred = parse_verdict(verdict_raw)
        if pred is None:
            pred_label = "?"
        else:
            pred_label = "VALID" if pred else "INVALID"
        gt_label = "VALID" if gt_valid else "INVALID"
        match = "✓" if pred == gt_valid else "✗"
        rows.append((qid, gt_type, gt_label, pred_label, match, verdict_raw))

    # Print
    print()
    print(f"{'qid':<6}{'type':<22}{'GT':<10}{'pred':<10}{'match':<7}{'raw'}")
    print("-" * 90)
    for r in rows:
        print(f"{r[0]:<6}{r[1]:<22}{r[2]:<10}{r[3]:<10}{r[4]:<7}{r[5][:30]!r}")

    # Confusion matrix + per-type
    tp = sum(1 for r in rows if r[2] == 'INVALID' and r[3] == 'INVALID')
    fn = sum(1 for r in rows if r[2] == 'INVALID' and r[3] != 'INVALID')
    tn = sum(1 for r in rows if r[2] == 'VALID' and r[3] == 'VALID')
    fp = sum(1 for r in rows if r[2] == 'VALID' and r[3] != 'VALID')
    n = len(rows)

    print()
    print("=== Confusion matrix (positive = INVALID) ===")
    print(f"  True INVALID, predicted INVALID (TP):  {tp}")
    print(f"  True INVALID, predicted VALID   (FN):  {fn}    ← missed bad questions")
    print(f"  True VALID,   predicted VALID   (TN):  {tn}")
    print(f"  True VALID,   predicted INVALID (FP):  {fp}    ← false alarms")
    print()
    print(f"  Recall (catch rate of bad): {tp}/{tp+fn} = {tp/max(tp+fn,1):.0%}")
    print(f"  Precision (when pred=BAD, % truly bad): {tp}/{tp+fp} = {tp/max(tp+fp,1):.0%}")
    print(f"  Overall accuracy: {(tp+tn)}/{n} = {(tp+tn)/n:.0%}")

    # Per-type breakdown for INVALIDs
    print()
    print("=== Recall per ill-posed type ===")
    types = {}
    for r in rows:
        if r[2] == 'INVALID':
            t = r[1]
            types.setdefault(t, []).append(r[3] == 'INVALID')
    for t, hits in types.items():
        print(f"  {t:<22}: {sum(hits)}/{len(hits)} caught")


if __name__ == "__main__":
    main()

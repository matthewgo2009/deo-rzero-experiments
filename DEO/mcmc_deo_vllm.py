"""
DEO + R-Zero verl trainer (fair comparison setup).

Pipeline per outer iteration:
  1. MCMC sample N questions from base model (vllm-base, port 8000),
     using current solver (vllm-solver, port 8001) to score r_unc.
  2. Filter by majority-vote rate p_hat in [0.3, 0.8] -> push HF dataset.
  3. Run R-Zero's verl GRPO trainer (subprocess) on that dataset.
  4. Reload vllm-solver with merged ckpt; loop.
"""
import os
import re
import sys
import json
import time
import random
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import openai
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from huggingface_hub import login as hf_login
from mathruler.grader import grade_answer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Optional sympy fallback for LaTeX-equivalence grading.
try:
    from sympy import simplify, N
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    _SYMPY_TRANSFORMATIONS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    _SYMPY_AVAILABLE = True
except Exception:
    _SYMPY_AVAILABLE = False


# ==========================================
# 1. Config
# ==========================================
class Config:
    # --- Model ---
    MODEL_NAME = "Qwen/Qwen3-4B-Base"          # base model: questioner + initial solver
    MODEL_ABBR = "deo_qwen3_4b_base"

    # --- Storage ---
    STORAGE_ROOT = "/storage"                  # mounted from /eph/nvme0/yyd/DEO inside container
    RZERO_DIR    = "/workspace"                # R-Zero repo (verl lives here)
    HF_USER      = os.getenv("HUGGINGFACENAME", "yuyang322")
    HF_TOKEN     = None                        # filled from RZERO_DIR/tokens.json

    # --- vLLM endpoints ---
    VLLM_BASE_URL = "http://localhost:8000/v1"     # Qwen3-4B-Base, GPU 0 (questioner / mutator)

    # Solver scoring runs as data-parallel across N vllm instances on separate GPUs.
    # Each entry: (gpu_id, port, docker_container_name).
    SOLVER_INSTANCES = [
        ("1", "8001", "vllm_solver"),
        ("4", "8004", "vllm_solver_dp1"),
        ("5", "8005", "vllm_solver_dp2"),
    ]
    VLLM_SOLVER_URLS   = [f"http://localhost:{port}/v1" for _, port, _ in SOLVER_INSTANCES]
    # Legacy single-endpoint aliases (kept so any external diagnostics still resolve).
    VLLM_SOLVER_URL    = VLLM_SOLVER_URLS[0]
    SOLVER_DOCKER_NAME = SOLVER_INSTANCES[0][2]

    # LLM-judge validity filter uses ONLY the first solver instance, to avoid stealing
    # DP throughput from r_unc scoring. Judge cost is ~3s/1000 entries so single endpoint
    # is plenty.
    JUDGE_URL = f"http://localhost:{SOLVER_INSTANCES[0][1]}/v1/completions"

    # --- Sampling for r_unc (matches R-Zero question_evaluate.py defaults) ---
    M_SAMPLES   = 9
    SOLVER_TEMP = 1.0
    SOLVER_TOP_P = 1.0
    SOLVER_TOP_K = 40
    SOLVER_MAX_TOKENS = 4096

    # --- MCMC ---
    BETA           = 0.1
    TAU_BLEU       = 0.5
    LAMBDA_REP     = 10.0
    TOTAL_QUESTIONS = 1500           # ensures filtered count >= 512 even with ~65% pass rate
    MCMC_STEPS     = 5
    MUTATE_BATCH_SIZE = 20
    INIT_BATCH_SIZE = 64

    # --- Outer loop ---
    NUM_ITERATIONS = 5               # match R-Zero's 5 iterations

    # --- HF filter (must match R-Zero upload.py) ---
    MIN_SCORE = 0.3
    MAX_SCORE = 0.8


config = Config()


# ==========================================
# 1b. Validity filtering (regex + LLM-judge cascade)
# ==========================================
# Stage 1: cheap substring/regex check. Drops obvious prompt-template echoes,
# XML-tag leakage, mutation-strategy keywords, proof-class problems. Applied
# inside extract_challenger_output, so polluted questions never enter the
# MCMC pool (saves their r_unc scoring cost and keeps the BLEU graph clean).
BAD_PATTERNS = [
    # prompt-template echoes
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
    # specific hallucination residue we observed in iter 1-4 datasets
    "MongoClient",
    "disable_shortcode",
    "GuestRemoved",
    "\\[最终答案",
    "primitive_helper",
    "primitive-item",
    "mid-translate",
    "my response was",
    "\\boxed{}",
    # math-task-shape rejects (R-Zero training expects single-answer questions)
    "prove that",
    "show that",
    # mutation-strategy keywords leaking into the generated question
    "DUALIZE the problem",
    "GENERALIZE the problem",
    "COMPOSE the problem",
    "INVERT the problem",
    "CHANGE_OBJECTIVE the problem",
    "the seed problem",
    "the spirit problem",
    "original seed",
    "[ADD]",
    "[INSERT]",
    "your problem here",
    "[ANSWER should be",
]
BAD_RE = re.compile("|".join(re.escape(p) for p in BAD_PATTERNS), re.IGNORECASE)


# Stage 2: LLM-judge prompt. Verdict-fill-in-the-blank template; we read only
# the first 6 tokens of the completion and look for VALID / INVALID.
# Empirically (validity_smoke_test.py) this drops ~9-22% of post-phat candidates
# with ~80% true-positive rate (and ~20% false positives we accept as cost).
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


def judge_one_validity(question, answer, timeout=60):
    """Return True if the LLM judge considers the (question, answer) pair valid.

    Implementation note: we use the raw /v1/completions endpoint rather than
    chat.completions because the solver (post-GRPO) treats chat-mode prompts
    as "math problems to reason through", emitting verbose traces instead of
    a clean one-word verdict. Completions + fill-in-the-blank + max_tokens=6
    is the most constrained form.
    """
    ans = answer if answer not in (None, "") else "(none)"
    prompt = JUDGE_TEMPLATE.format(q=question, ans=ans)
    payload = {
        "model": config.MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 6,
        "temperature": 0,
    }
    r = requests.post(config.JUDGE_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    text = r.json()["choices"][0]["text"].strip().upper()
    parts = text.split()
    # NOTE: check INVALID before VALID because "VALID" is a substring of "INVALID".
    if parts and "INVALID" in parts[0]:
        return False
    if parts and parts[0].startswith("VALID"):
        return True
    head = text[:12]
    if "INVALID" in head:
        return False
    if "VALID" in head:
        return True
    # Unparseable output → default VALID so a noisy judge doesn't kill data.
    return True


# ==========================================
# 2. Clients (lazy-init helpers)
# ==========================================
_client_base = None
_clients_solver = None


def base_client():
    """vLLM endpoint serving Qwen3-4B-Base for question generation/mutation."""
    global _client_base
    if _client_base is None:
        _client_base = openai.OpenAI(api_key="EMPTY", base_url=config.VLLM_BASE_URL)
    return _client_base


def solver_clients():
    """List of OpenAI clients, one per data-parallel solver instance."""
    global _clients_solver
    if _clients_solver is None:
        _clients_solver = [
            openai.OpenAI(api_key="EMPTY", base_url=url)
            for url in config.VLLM_SOLVER_URLS
        ]
    return _clients_solver


def solver_client():
    """Back-compat: first solver instance only (kept for ad-hoc callers)."""
    return solver_clients()[0]


# ==========================================
# 3. Prompts
# ==========================================
MATH_TOPICS = [
    "Algebra: Polynomial roots and coefficients",
    "Geometry: Triangle centers and circles",
    "Geometry: 3D spatial geometry and volume",
    "Geometry: Coordinate geometry and loci",
    "Number Theory: Diophantine equations",
    "Number Theory: Modular arithmetic and congruences",
    "Combinatorics: Probability and expected value",
    "Calculus: Limits and derivatives",
]

CHALLENGER_SYSTEM_PROMPT = """You are an expert competition-math problem setter. FIRST, in your private scratch-pad, think
step-by-step to design a brand-new, non-trivial problem. Aim for a medium-to-hard competition level.

CRITICAL RULES:
1. The final answer MUST be a SPECIFIC NUMBER, ALGEBRAIC EXPRESSION, or FINITE SET.
2. DO NOT generate "Prove that", "Show that", "Justify", or "Explain why" questions.
3. DO NOT generate questions that ask for True/False or Yes/No answers.
4. Ensure the problem has exactly one unambiguous final answer.
5. LIMIT YOUR SCRATCH-PAD THINKING TO UNDER 50 WORDS! Do not write out the full proof.

THEN, output the problem and answer exactly in this format:
<question>
[Write the full problem statement here on one or more lines]
</question>
\\boxed{final_answer}"""

MUTATOR_SYSTEM_PROMPT = """You are an expert competition-math problem setter. I will provide a seed problem.
Your task is to generate a STRUCTURALLY DIFFERENT problem by applying ONE of the following mutation strategies.

MUTATION STRATEGIES (pick exactly ONE that is NOT already exemplified by the seed):
[A] GENERALIZE — Lift the structure: replace a specific constant with a parameter, OR raise the dimension
    (2D → 3D, single-variable → multivariable, single equation → system of equations).
[B] COMPOSE   — Add a NEW non-trivial second condition that interacts with the existing structure
    (NOT just a range bound like "with x > 0"). The two conditions together must create a real interaction.
[C] INVERT    — Given the original answer or output, ask the reader to recover an input or precondition.
[D] CHANGE_OBJECTIVE — Change WHAT is being asked: e.g. "find x" → "count integer solutions",
    "compute" → "find the smallest n such that ...", "find value" → "find sum of all such values".
[E] DUALIZE   — Swap to a dual concept: sum↔product, max↔min, area↔perimeter,
    gcd↔lcm, addition↔multiplication, distance↔angle.

CRITICAL RULES:
1. DO NOT just swap numbers. If the only edit is a digit change, you have FAILED — restart with another strategy.
2. DO NOT pick the strategy already exemplified by the seed. (e.g. if seed is already 3D, don't pick GENERALIZE→add dimension.)
3. The final answer MUST be a SPECIFIC NUMBER, ALGEBRAIC EXPRESSION, or FINITE SET.
4. NO "Prove that", "Show that", "Justify", "True/False", or "Yes/No" questions.
5. LIMIT scratch-pad reasoning to UNDER 50 WORDS.

Output format (STRICT — all three tags required):
<strategy>{A|B|C|D|E}</strategy>
<question>
[the NEW mutated problem statement]
</question>
\\boxed{final_answer}"""

MUTATOR_USER_TEMPLATE = (
    "Here is the seed problem:\n{seed}\n\n"
    "Pick ONE mutation strategy from {{A,B,C,D,E}} and apply it now. "
    "Remember: number-swapping is FAILURE."
)

# Solver chat template: must match R-Zero's verl/utils/dataset.py:196 exactly,
# so the pseudo-labels we generate match the distribution verl will train against.
RZERO_SOLVER_SYSTEM = r"Please reason step by step, and put your final answer within \boxed{}."


def apply_chat_template(tokenizer, system, user):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Base model fallback (matches R-Zero question_evaluate.py:118-119)
    return f"system: {system}\nuser: {user}\n"


# ==========================================
# 4. Extractors + grader (preserved from original; well-tested)
# ==========================================
def extract_last_boxed(text):
    """Extract the LAST \\boxed{...} content with balanced-brace matching."""
    if not text:
        return None
    needle = "\\boxed{"
    idx = text.rfind(needle)
    if idx == -1:
        return None
    i = idx + len(needle)
    depth = 1
    start = i
    n = len(text)
    while i < n and depth > 0:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def extract_challenger_output(text):
    q_matches = list(re.finditer(r"<question>(.*?)</question>", text, re.DOTALL))
    if not q_matches:
        return None, None
    question = q_matches[-1].group(1)
    # Mutator sometimes hallucinates a nested <question> open tag inside the
    # captured region (no matching close). The non-greedy outer regex then
    # captures from the FIRST <question> through the only </question>, which
    # leaves a literal <question> mid-string. Keep only text after the last
    # nested open tag in the capture.
    last_nested = question.rfind("<question>")
    if last_nested != -1:
        question = question[last_nested + len("<question>"):]
    question = question.strip()
    if not question:
        return None, None
    # Cheap substring filter: drops obvious prompt-template echoes and
    # mutation-strategy leakage before they ever enter the MCMC pool. See
    # BAD_PATTERNS list for the full set.
    if BAD_RE.search(question):
        return None, None
    gt_answer = extract_last_boxed(text)
    if not gt_answer:
        fb = re.search(
            r"(?:answer is|equals|value is)[\s:]*([0-9a-zA-Z\.\-\\\/]+)",
            text[-200:], re.IGNORECASE,
        )
        if fb:
            gt_answer = fb.group(1).strip()
    if gt_answer and len(gt_answer) > 100:
        gt_answer = None
    return question, gt_answer


def extract_solver_answer(text):
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed
    fb = re.search(
        r"(?:answer is|equals|value is|is exactly)[\s:]*([0-9a-zA-Z\.\-\\\/]+)",
        text[-200:], re.IGNORECASE,
    )
    if fb:
        return fb.group(1).strip()
    if len(text.split()) > 50:
        return "GUESSED_FAIL_FORMAT"
    return None


_STRATEGY_RE = re.compile(r"<strategy>\s*([A-E])\s*</strategy>", re.IGNORECASE)


def extract_mutation_strategy(text):
    if not text:
        return None
    m = _STRATEGY_RE.search(text)
    return m.group(1).upper() if m else None


# LaTeX → sympy normalization for robust answer-equivalence checks.
_LATEX_CMD_STRIP = re.compile(r"\\(?:left|right|,|!|;|:|\\)")
_LATEX_FRAC_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_LATEX_SQRT_BRACE_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_LATEX_SQRT_BARE_RE = re.compile(r"\\sqrt\s*([A-Za-z0-9])")
_LATEX_POW_BRACE_RE = re.compile(r"\^\s*\{([^{}]+)\}")
_LATEX_POW_BARE_RE = re.compile(r"\^\s*([A-Za-z0-9])")
_LATEX_GENERIC_CMD = re.compile(r"\\[a-zA-Z]+")


def _latex_to_sympy_str(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    inner = extract_last_boxed(s)
    if inner is not None:
        s = inner
    s = _LATEX_CMD_STRIP.sub("", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi")
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", s)
    s = s.replace("\\circ", "").replace("°", "")
    s = s.replace("\\%", "/100").replace("%", "/100")
    s = s.replace("\\$", "").replace("$", "")
    for _ in range(6):
        new = _LATEX_FRAC_RE.sub(r"((\1)/(\2))", s)
        new = _LATEX_SQRT_BRACE_RE.sub(r"sqrt(\1)", new)
        new = _LATEX_SQRT_BARE_RE.sub(r"sqrt(\1)", new)
        new = _LATEX_POW_BRACE_RE.sub(r"**(\1)", new)
        new = _LATEX_POW_BARE_RE.sub(r"**\1", new)
        if new == s:
            break
        s = new
    s = _LATEX_GENERIC_CMD.sub(lambda m: m.group(0)[1:], s)
    s = re.sub(r"\^\s*$", "", s).strip()
    return s or None


def robust_grade(student_answer, gt_answer):
    if student_answer is None or gt_answer is None:
        return False
    s_raw = str(student_answer).strip()
    g_raw = str(gt_answer).strip()
    if not s_raw or not g_raw:
        return False
    if s_raw == g_raw:
        return True
    try:
        if grade_answer(s_raw, g_raw):
            return True
    except Exception:
        pass
    if not _SYMPY_AVAILABLE:
        return False
    try:
        s_clean = _latex_to_sympy_str(s_raw)
        g_clean = _latex_to_sympy_str(g_raw)
        if not s_clean or not g_clean:
            return False
        a = parse_expr(s_clean, transformations=_SYMPY_TRANSFORMATIONS)
        b = parse_expr(g_clean, transformations=_SYMPY_TRANSFORMATIONS)
        diff = simplify(a - b)
        if diff == 0:
            return True
        try:
            if abs(complex(N(diff))) < 1e-9:
                return True
        except Exception:
            pass
    except Exception:
        pass
    return False


# ==========================================
# 5. r_unc evaluation (uses solver vllm, R-Zero-aligned sampling)
# ==========================================
def evaluate_r_unc_vllm(tokenizer, questions):
    """Score each question by majority-vote consistency on the current solver.

    Sampling settings + chat template match R-Zero's question_evaluate.py and
    verl/utils/dataset.py:196 so pseudo-labels are drawn from the same
    distribution verl will train on.

    Returns: (r_unc_list, p_hat_list, pseudo_label_list)
    """
    if not questions:
        return [], [], []
    m = config.M_SAMPLES

    prompts = [
        apply_chat_template(tokenizer, RZERO_SOLVER_SYSTEM, q)
        for q in questions for _ in range(m)
    ]

    # R-Zero question_evaluate.py uses temp=1.0, top_p=1.0, top_k=40, n=9.
    # Data-parallel across len(solver_clients()) endpoints; preserve prompt order.
    clients = solver_clients()
    n_dp = len(clients)
    n_prompts = len(prompts)
    base, rem = divmod(n_prompts, n_dp)
    slices = []
    start = 0
    for i in range(n_dp):
        size = base + (1 if i < rem else 0)
        slices.append(prompts[start:start + size])
        start += size

    def _run_shard(idx):
        chunk = slices[idx]
        if not chunk:
            return idx, []
        resp = clients[idx].completions.create(
            model=config.MODEL_NAME,                 # served name on solver endpoint
            prompt=chunk,
            max_tokens=config.SOLVER_MAX_TOKENS,
            temperature=config.SOLVER_TEMP,
            top_p=config.SOLVER_TOP_P,
            extra_body={"top_k": config.SOLVER_TOP_K},
        )
        return idx, [c.text for c in resp.choices]

    texts_by_shard = [None] * n_dp
    with ThreadPoolExecutor(max_workers=n_dp) as ex:
        for fut in as_completed([ex.submit(_run_shard, i) for i in range(n_dp)]):
            idx, txt = fut.result()
            texts_by_shard[idx] = txt
    texts = [t for shard in texts_by_shard for t in shard]
    answers = [extract_solver_answer(t) for t in texts]

    r_unc_list, p_hat_list, pseudo_list = [], [], []
    for i in range(len(questions)):
        chunk = answers[i * m: (i + 1) * m]
        valid = [a for a in chunk if a is not None and a != "GUESSED_FAIL_FORMAT"]
        if not valid:
            r_unc_list.append(0.0)
            p_hat_list.append(0.0)
            pseudo_list.append(None)
            continue
        major_ans, count = Counter(valid).most_common(1)[0]
        p_hat = count / m   # absolute consistency rate, matches R-Zero
        r_unc = max(0.0, 1.0 - 2.0 * abs(p_hat - 0.5))   # peak at p_hat=0.5

        # R-Zero's filter rejects malformed mathruler outputs (long \text{...})
        is_garbage = (
            "text" in major_ans.lower()
            or "\\text" in major_ans
            or len(major_ans) > 100
        )
        r_unc_list.append(r_unc)
        p_hat_list.append(p_hat)
        pseudo_list.append(None if is_garbage else major_ans)

    return r_unc_list, p_hat_list, pseudo_list


# ==========================================
# 6. Energy = sum_i max(0, r_unc_i - lambda * r_rep_i)
#
# r_rep[i] = cluster_size[i] / N  (BLEU-based "neighbors of i" count, including self)
# We keep the bool neighbor matrix + cluster_size around so MCMC accept/reject is O(N)
# instead of O(N^2).  At N=1000 this is the difference between 33s and 33ms per proposal.
# ==========================================
_BLEU_SMOOTH = SmoothingFunction().method1


def _is_close(tok_a, tok_b):
    bleu = sentence_bleu([tok_b], tok_a, smoothing_function=_BLEU_SMOOTH)
    return (1.0 - bleu) < config.TAU_BLEU


def build_neighbor_matrix(pool_tokens):
    """Initial O(N^2) BLEU sweep. Run once at MCMC init."""
    n = len(pool_tokens)
    M = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(M, True)
    for i in range(n):
        for j in range(i + 1, n):
            if _is_close(pool_tokens[i], pool_tokens[j]):
                M[i, j] = M[j, i] = True
    return M


def neighbor_row_for(q_tokens, pool_tokens, self_idx):
    """Bool[N] where out[j] = is q_tokens close to pool_tokens[j]. self_idx -> True."""
    n = len(pool_tokens)
    out = np.zeros(n, dtype=bool)
    for j in range(n):
        if j == self_idx:
            out[j] = True
        elif _is_close(q_tokens, pool_tokens[j]):
            out[j] = True
    return out


def energy_from_state(r_unc_arr, cluster_size, n):
    """Compute V(X) = sum_i max(0, r_unc[i] - lambda * cluster_size[i] / N)."""
    r_rep = cluster_size / n
    r_c = np.maximum(0.0, r_unc_arr - config.LAMBDA_REP * r_rep)
    return float(r_c.sum())


def calculate_batch_energy(questions, r_unc_list):
    """One-shot initial energy. Builds matrix and returns (energy, neighbor, cluster_size)."""
    n = len(questions)
    if n == 0:
        return 0.0, np.zeros((0, 0), dtype=bool), np.zeros(0)
    pool_tokens = [q.split() for q in questions]
    neighbor = build_neighbor_matrix(pool_tokens)
    cluster_size = neighbor.sum(axis=1).astype(float)
    r_unc_arr = np.asarray(r_unc_list, dtype=float)
    return energy_from_state(r_unc_arr, cluster_size, n), neighbor, cluster_size


# ==========================================
# 7. MCMC (single-process; large vllm batches)
# ==========================================
def generate_batch_mcmc(tokenizer, num_questions, log_path):
    """Returns: list of dicts {question, gt, p_hat, pseudo_label, r_unc}."""
    print(f"\n[MCMC] Initializing pool of {num_questions} questions via base model...")
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("=" * 50 + "\nMCMC Phase\n" + "=" * 50 + "\n")

    pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo = [], [], [], [], []
    pbar = tqdm(total=num_questions, desc="Init")

    forbidden = ["prove that", "show that", "justify", "explain", "true or false", "yes or no"]

    while len(pool_q) < num_questions:
        needed = num_questions - len(pool_q)
        bs = min(config.INIT_BATCH_SIZE, needed)
        c_prompts = []
        for _ in range(bs):
            topic = random.choice(MATH_TOPICS)
            user_p = (
                "Generate one new, challenging reasoning question now. "
                f"YOU MUST STRICTLY FOCUS ON: **{topic}**."
            )
            c_prompts.append(apply_chat_template(tokenizer, CHALLENGER_SYSTEM_PROMPT, user_p))

        resp = base_client().completions.create(
            model=config.MODEL_NAME,
            prompt=c_prompts,
            max_tokens=1536,
            temperature=1.0,
            top_p=0.95,
        )
        valid_qs, valid_gts = [], []
        for c in resp.choices:
            q, gt = extract_challenger_output(c.text)
            if q and len(q) > 30 and not any(w in q.lower() for w in forbidden):
                valid_qs.append(q)
                valid_gts.append(gt)
        if not valid_qs:
            continue

        r_uncs, p_hats, pseudos = evaluate_r_unc_vllm(tokenizer, valid_qs)
        for q, gt, ru, ph, ps in zip(valid_qs, valid_gts, r_uncs, p_hats, pseudos):
            if len(pool_q) >= num_questions:
                break
            pool_q.append(q)
            pool_gt.append(gt)
            pool_runc.append(ru)
            pool_phat.append(ph)
            pool_pseudo.append(ps)
            pbar.update(1)
    pbar.close()

    # Build initial state (one-shot O(N^2) BLEU sweep, ~30s at N=1000)
    print("[MCMC] Computing initial neighbor matrix (one-shot O(N^2))...")
    energy, neighbor, cluster_size = calculate_batch_energy(pool_q, pool_runc)
    pool_tokens = [q.split() for q in pool_q]
    r_unc_arr = np.asarray(pool_runc, dtype=float)
    n = num_questions
    log_file.write(f"[Init] V(X_0) = {energy:.4f}\n")

    # MCMC mutation walk — every accept/reject is O(N) thanks to incremental update
    for step in range(config.MCMC_STEPS):
        idx_perm = list(range(num_questions))
        random.shuffle(idx_perm)
        pbar_step = tqdm(total=num_questions, desc=f"MCMC step {step+1}/{config.MCMC_STEPS}")
        for i in range(0, num_questions, config.MUTATE_BATCH_SIZE):
            batch_idx = idx_perm[i: i + config.MUTATE_BATCH_SIZE]
            m_prompts = [
                apply_chat_template(
                    tokenizer, MUTATOR_SYSTEM_PROMPT,
                    MUTATOR_USER_TEMPLATE.format(seed=pool_q[k]),
                )
                for k in batch_idx
            ]
            resp = base_client().completions.create(
                model=config.MODEL_NAME,
                prompt=m_prompts,
                max_tokens=1536,
                temperature=1.1,
            )
            proposals = []
            for j, k in enumerate(batch_idx):
                t = resp.choices[j].text
                qp, gtp = extract_challenger_output(t)
                strat = extract_mutation_strategy(t) or "?"
                if qp and len(qp) > 30 and qp != pool_q[k]:
                    proposals.append({"k": k, "q": qp, "gt": gtp, "strat": strat,
                                      "old": pool_q[k]})
            if proposals:
                qs_new = [p["q"] for p in proposals]
                rus, phs, pls = evaluate_r_unc_vllm(tokenizer, qs_new)
                for j, p in enumerate(proposals):
                    k = p["k"]
                    q_prime_tokens = p["q"].split()

                    # ---- O(N) incremental energy ----
                    new_row = neighbor_row_for(q_prime_tokens, pool_tokens, self_idx=k)
                    old_row = neighbor[k]
                    # Off-diagonal cluster delta: M[k][j] flipping changes cluster_size[j] by +/-1
                    delta = new_row.astype(int) - old_row.astype(int)
                    delta[k] = 0
                    new_cluster = cluster_size + delta
                    # Diagonal: cluster_size[k] = sum of new row (which already counts self)
                    new_cluster[k] = float(new_row.sum())

                    new_runc = r_unc_arr.copy()
                    new_runc[k] = rus[j]
                    new_e = energy_from_state(new_runc, new_cluster, n)
                    # ----------------------------------

                    alpha = min(1.0, np.exp((new_e - energy) / config.BETA))
                    accept = random.random() < alpha

                    # Per-question r_c contribution (the only term that changed is index k)
                    r_rep_k_old = float(cluster_size[k]) / n
                    r_rep_k_new = float(new_cluster[k]) / n
                    r_c_k_old = max(0.0, float(r_unc_arr[k]) - config.LAMBDA_REP * r_rep_k_old)
                    r_c_k_new = max(0.0, rus[j] - config.LAMBDA_REP * r_rep_k_new)

                    log_file.write(
                        f"\n[Step {step+1} | Batch {i // config.MUTATE_BATCH_SIZE}] "
                        f"Result: {'ACCEPTED' if accept else 'REJECTED'} | Strategy: {p['strat']}\n"
                        f"--- [OLD QUESTION] ---\n{p['old']}\n"
                        f"--- [NEW QUESTION] ---\n{p['q']}\n"
                        f"r_unc:        {float(r_unc_arr[k]):.4f} -> {rus[j]:.4f}\n"
                        f"r_rep[k]:     {r_rep_k_old:.4f} -> {r_rep_k_new:.4f}  "
                        f"(cluster_size {int(cluster_size[k])} -> {int(new_cluster[k])})\n"
                        f"r_c[k]:       {r_c_k_old:.4f} -> {r_c_k_new:.4f}\n"
                        f"V(X) total:   {energy:.4f} -> {new_e:.4f}  (dE={new_e - energy:+.4f})\n"
                        f"Alpha:        {alpha:.4f}\n"
                        f"{'-' * 60}\n"
                    )
                    log_file.flush()
                    if accept:
                        pool_q[k] = p["q"]
                        pool_tokens[k] = q_prime_tokens
                        pool_gt[k] = pls[j] or p["gt"]
                        pool_runc[k] = rus[j]
                        pool_phat[k] = phs[j]
                        pool_pseudo[k] = pls[j]
                        # Commit incremental state: row k <-> col k symmetric
                        neighbor[k, :] = new_row
                        neighbor[:, k] = new_row
                        cluster_size = new_cluster
                        r_unc_arr = new_runc
                        energy = new_e
            pbar_step.update(len(batch_idx))
            pbar_step.set_postfix({"V": f"{energy:.3f}"})
        pbar_step.close()

    log_file.close()
    return [
        {"question": pool_q[i], "gt": pool_gt[i], "p_hat": pool_phat[i],
         "pseudo_label": pool_pseudo[i], "r_unc": pool_runc[i]}
        for i in range(num_questions)
    ]


# ==========================================
# 8. Filter + push to HuggingFace (R-Zero upload.py format)
# ==========================================
def filter_and_push(train_data, repo_name, config_name):
    """Three-stage filter before HF upload, then push:

      Stage 1 (cheap):  p_hat ∈ [MIN_SCORE, MAX_SCORE] AND pseudo_label non-null.
                        (BAD_PATTERNS regex is already applied earlier, at
                        extract_challenger_output time, so polluted questions
                        never even reach this function.)
      Stage 2 (LLM):    judge_one_validity per entry via solver completion API.
                        Single endpoint (config.JUDGE_URL = port 8001) so DP
                        scoring on 8004/8005 keeps its full throughput.
    """
    n_total = len(train_data)
    stage1 = [
        d for d in train_data
        if d["pseudo_label"] not in (None, "", "None")
        and config.MIN_SCORE <= d["p_hat"] <= config.MAX_SCORE
    ]
    n_phat = len(stage1)
    print(f"[filter] phat∈[{config.MIN_SCORE},{config.MAX_SCORE}]+pseudo: "
          f"{n_phat}/{n_total} passed")

    if stage1:
        t0 = time.time()
        judge_ok = [True] * n_phat
        with ThreadPoolExecutor(max_workers=24) as ex:
            futs = {
                ex.submit(judge_one_validity, d["question"], d["pseudo_label"]): i
                for i, d in enumerate(stage1)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    judge_ok[i] = fut.result()
                except Exception:
                    # Default to VALID on judge error so a transient API blip
                    # doesn't shred the training set.
                    judge_ok[i] = True
        n_judge_drop = sum(1 for v in judge_ok if not v)
        print(f"[filter] LLM-judge dropped {n_judge_drop}/{n_phat} "
              f"({time.time() - t0:.1f}s)")
        stage2 = [d for d, ok in zip(stage1, judge_ok) if ok]
    else:
        stage2 = []

    filtered = [
        {"problem": d["question"], "answer": d["pseudo_label"], "score": d["p_hat"]}
        for d in stage2
    ]
    print(f"[upload] {len(filtered)}/{n_total} final after regex+phat+pseudo+judge")
    if not filtered:
        raise SystemExit("ERROR: 0 questions passed filter — cannot train empty dataset.")

    # Persist the post-filter training set locally too — the HF push is the
    # source verl reads, but a local JSON makes offline error analysis (e.g.
    # checking pseudo-label correctness) trivial without needing HF auth or
    # the arrow cache. File is keyed by repo_name so it sorts by iter.
    local_path = f"{config.STORAGE_ROOT}/datasets/filtered_{repo_name}.json"
    with open(local_path, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"[upload] also wrote local JSON: {local_path}")

    repo_full = f"{config.HF_USER}/{repo_name}"
    ds = DatasetDict({"train": Dataset.from_list(filtered)})
    ds.push_to_hub(repo_full, private=True, config_name=config_name)
    print(f"[upload] pushed to https://huggingface.co/datasets/{repo_full}")
    return len(filtered)


# ==========================================
# 9. Run R-Zero verl trainer (subprocess)
# ==========================================
def run_verl_solver(solver_ckpt, dataset_repo, exp_name, extra_verl_overrides=None):
    """Spawn verl in subprocess; matches R-Zero's solver_train.sh exactly.

    extra_verl_overrides: optional list of additional Hydra overrides (e.g.
      ["data.rollout_batch_size=64"]) appended to cmd, useful when the
      filtered dataset is below the default rollout_batch_size=512 (e.g.
      base-agreement-filter ablation).
    """
    storage = config.STORAGE_ROOT
    rzero = config.RZERO_DIR

    # Bust local HF cache for this dataset to avoid stale empty-dataset shadow
    cache_root = os.path.join(os.environ.get("HF_HOME", "/root/.cache/huggingface"), "datasets")
    cache_dir = os.path.join(cache_root, f"{config.HF_USER}___{exp_name}")
    if os.path.exists(cache_dir):
        subprocess.run(["rm", "-rf", cache_dir], check=False)

    print(f"\n[verl] training {exp_name} on dataset {dataset_repo}")
    # rollout_batch_size left at config.yaml default (512) for fair compare with R-Zero.
    # Requires filtered dataset >= 512 — TOTAL_QUESTIONS=1500 ensures this in expectation.
    cmd = [
        "python3", "-m", "verl.trainer.main",
        "config=examples/config.yaml",
        "data.max_response_length=4096",
        f"worker.actor.model.model_path={solver_ckpt}",
        # Pin KL ref to the original base across all iters (no cumulative drift).
        # Requires verl patch: RefConfig.model + fsdp_workers ref branch.
        f"worker.ref.model.model_path={config.MODEL_NAME}",
        f"trainer.experiment_name={exp_name}",
        f"trainer.save_checkpoint_path={storage}/models/{exp_name}/",
        f"data.train_files={dataset_repo}@train",
        "trainer.total_epochs=100",
        "trainer.max_steps=20",
        "trainer.n_gpus_per_node=2",
        "worker.actor.global_batch_size=16",
        "worker.rollout.tensor_parallel_size=2",
        "data.format_prompt=./examples/format_prompt/solver.jinja",
        "trainer.val_freq=4",
        "worker.actor.micro_batch_size_per_device_for_update=1",
        "worker.actor.micro_batch_size_per_device_for_experience=1",
    ]
    if extra_verl_overrides:
        cmd.extend(extra_verl_overrides)
    # GPU isolation handled by docker `--gpus device=2,3` flag at the outer launch,
    # so verl picks up the 2 visible cards via its default behavior.
    env = os.environ.copy()
    env["STORAGE_PATH"] = storage

    # Defensive Ray cleanup: any prior verl/eval may have left a stale GCS at /tmp/ray*
    # that gets reused on next ray.init() and reports 0 available GPUs (the iter-4 crash).
    # Force a fresh Ray cluster every iteration.
    def cleanup_ray():
        subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
        subprocess.run("rm -rf /tmp/ray /tmp/ray_* 2>/dev/null", shell=True, check=False)
        time.sleep(3)
    cleanup_ray()
    try:
        subprocess.run(cmd, cwd=rzero, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[verl] FIRST attempt failed (rc={e.returncode}). Cleaning Ray + retrying once...")
        cleanup_ray()
        subprocess.run(cmd, cwd=rzero, check=True, env=env)

    # Merge FSDP shards into a single HF-format checkpoint at .../huggingface
    actor_dir = f"{storage}/models/{exp_name}/global_step_15/actor"
    subprocess.run(
        ["python", "scripts/model_merger.py", "--local_dir", actor_dir],
        cwd=rzero, check=True, env=env,
    )
    return f"{actor_dir}/huggingface"


# ==========================================
# 9b. GPT-4o-mini secondary verification on MATH-500 eval results
#
# R-Zero's results_recheck.py uses GPT-4o to upgrade mathruler-failed entries
# whose answers are mathematically equivalent (e.g. \frac{1}{2} vs 0.5).
# Empirically (gpt_recheck_smoke.py): mathruler-only undercounts MATH-500 acc
# by ~13-14% on Qwen3-4B-Base. We reproduce R-Zero's logic but:
#   - use gpt-4o-mini (cheaper; ~$0.02 / full DEO run)
#   - extract \boxed{...} from BOTH model response and GT and send only those
#     to GPT (full-text prompts gave ~33% FP rate; boxed-only gave ~0% FP)
# ==========================================
_BOXED_OPEN_RE = re.compile(r"\\boxed\{")


def _extract_last_boxed(text):
    """Last \\boxed{...} body with balanced braces. None if absent or malformed."""
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


def _recheck_user_msg(model_response, ground_truth):
    """Build the GPT-judge prompt with boxed-only extraction."""
    m_boxed = _extract_last_boxed(model_response)
    gt_boxed = _extract_last_boxed(ground_truth) or ground_truth
    m_part = m_boxed if m_boxed is not None else "(NO BOXED ANSWER FOUND IN MODEL RESPONSE)"
    return (
        f"Hi, there is an answer: {m_part},"
        f"and the ground truth answer is: {gt_boxed},"
        "please check whether the answer is correct or not, "
        "and return the **only**Yes or No."
    )


def _load_openai_key():
    """Return openai key from tokens.json, or None if missing/placeholder."""
    p = os.path.join(config.RZERO_DIR, "tokens.json")
    try:
        with open(p) as f:
            tok = json.load(f).get("openai")
    except Exception:
        return None
    if not tok or tok.startswith("your") or tok == "":
        return None
    return tok


def gpt_recheck_math500(results_path, max_workers=12, retries=3):
    """Apply GPT-4o-mini secondary verification to a MATH-500 results_math.json.

    Iterates entries where mathruler gave score < 0.5; asks GPT-4o-mini if the
    model's boxed answer is mathematically equivalent to the GT's boxed answer.
    If yes, bumps score to 1.0. Rewrites results_path in place with adjusted
    scores + recomputed average_score.

    Returns: (old_avg, new_avg, n_bumped). On missing API key, returns the
    original average unchanged so the pipeline still works.
    """
    api_key = _load_openai_key()
    if not api_key:
        with open(results_path) as f:
            d = json.load(f)
        print("[recheck] openai key not set in tokens.json; skipping GPT recheck")
        a = d[-1]["average_score"]
        return a, a, 0

    client = openai.OpenAI(api_key=api_key)
    with open(results_path) as f:
        results = json.load(f)
    eval_entries = results[:-1]
    n = len(eval_entries)
    old_avg = results[-1]["average_score"]
    fail_idx = [i for i, e in enumerate(eval_entries) if e["score"] < 0.5]

    def _check(idx):
        e = eval_entries[idx]
        last_exc = None
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a math answer checker."},
                        {"role": "user", "content": _recheck_user_msg(e["response"], e["answer"])},
                    ],
                    max_tokens=4,
                    temperature=0.1,
                )
                text = resp.choices[0].message.content.strip().lower()
                return idx, ("yes" in text), None
            except Exception as exc:
                last_exc = exc
                time.sleep(1.0 * (attempt + 1))
        return idx, False, type(last_exc).__name__

    t0 = time.time()
    bumped = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_check, i) for i in fail_idx]
        for fut in as_completed(futs):
            idx, is_correct, err = fut.result()
            if err is not None:
                n_err += 1
            if is_correct:
                eval_entries[idx]["score"] = 1.0
                bumped += 1

    new_avg = (sum(e["score"] for e in eval_entries) / n) if n else 0.0
    results = eval_entries + [{"average_score": new_avg}]
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    err_str = f" (api-errs: {n_err})" if n_err else ""
    print(f"[recheck] bumped {bumped}/{len(fail_idx)} entries  "
          f"{old_avg:.4f} → {new_avg:.4f} (Δ{new_avg - old_avg:+.4f}, "
          f"{time.time() - t0:.0f}s){err_str}")
    return old_avg, new_avg, bumped


# ==========================================
# 10. Reload vllm-solver docker with new model
# ==========================================
def eval_math500(model_path, label):
    """Run R-Zero's evaluation/generate.py on MATH-500 against `model_path`,
    then run GPT-4o-mini secondary verification on mathruler-failed entries.

    Returns: GPT-recheck-adjusted average accuracy (float in [0, 1]).
    """
    print(f"\n=== MATH-500 eval: {label}  (model={model_path}) ===")
    env = os.environ.copy()
    env["STORAGE_PATH"] = config.STORAGE_ROOT
    # Inside deo_runner, container exposes host GPUs 2,3 as logical 0,1.
    # Pin eval to logical GPU 0 (= host GPU 2). Verl is done; nothing else uses it.
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    subprocess.run(
        ["python3", "evaluation/generate.py",
         "--model", model_path, "--dataset", "math"],
        cwd=config.RZERO_DIR, check=True, env=env,
    )
    # generate.py writes to: {STORAGE_PATH}/evaluation/{model.replace('/', '_')}/results_math.json
    results_path = (
        f"{config.STORAGE_ROOT}/evaluation/"
        f"{model_path.replace('/', '_')}/results_math.json"
    )
    old_avg, new_avg, n_bumped = gpt_recheck_math500(results_path)
    print(f"=== {label}: MATH-500 acc = {new_avg:.4f} ({new_avg * 100:.2f}%) "
          f"[raw mathruler {old_avg:.4f}, +{n_bumped} GPT-bumped] ===\n")
    return new_avg


def reload_vllm_solver(new_model_path):
    """Stop+restart ALL solver vllm containers with new --model. Full FT can't hot-swap."""
    n = len(config.SOLVER_INSTANCES)
    print(f"\n[vllm-solver] reloading {n} instances with {new_model_path}")
    # Tear down everything first so GPU memory is free before any new container starts.
    for _, _, name in config.SOLVER_INSTANCES:
        subprocess.run(["docker", "stop", name], check=False)
        subprocess.run(["docker", "rm",   name], check=False)

    for gpu_id, port, name in config.SOLVER_INSTANCES:
        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "--gpus", f'"device={gpu_id}"',
            "--network", "host",
            "--shm-size", "16g",
            "-v", "/eph/nvme0/yyd/hf_cache:/root/.cache/huggingface",
            "-v", "/eph/nvme0/yyd/DEO:/storage",
            "-v", "/eph/nvme0/yyd/R-Zero:/storage_rzero",
            "vllm/vllm-openai:v0.9.1",
            "--model", new_model_path,
            "--served-model-name", config.MODEL_NAME,
            "--dtype", "bfloat16",
            "--max-model-len", "6144",
            "--tensor-parallel-size", "1",
            "--port", port,
        ]
        subprocess.run(cmd, check=True)
    for url in config.VLLM_SOLVER_URLS:
        wait_for_vllm_ready(url, label=f"vllm-solver@{url}")
    # Reset the lazy clients so they pick up the new endpoints cleanly.
    global _clients_solver
    _clients_solver = None


def wait_for_vllm_ready(url, label="vllm", timeout=600):
    """Poll /v1/models until the server returns 200 or we time out."""
    health = url.rstrip("/").replace("/v1", "") + "/v1/models"
    print(f"[{label}] waiting on {health} ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(health, timeout=5)
            if r.status_code == 200:
                print(f"[{label}] up after {time.time() - t0:.1f}s")
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"[{label}] did not become ready in {timeout}s")


# ==========================================
# 11. Main loop
# ==========================================
def load_hf_token():
    """Read RZERO_DIR/tokens.json; fail loudly if missing/placeholder."""
    p = os.path.join(config.RZERO_DIR, "tokens.json")
    with open(p) as f:
        tok = json.load(f)["huggingface"]
    if not tok or tok.startswith("your") or tok == "":
        raise SystemExit(f"ERROR: HF token not set in {p}")
    return tok


def main():
    os.makedirs(f"{config.STORAGE_ROOT}/datasets", exist_ok=True)
    os.makedirs(f"{config.STORAGE_ROOT}/logs", exist_ok=True)
    os.makedirs(f"{config.STORAGE_ROOT}/models", exist_ok=True)
    # R-Zero verl expects these too
    os.makedirs(f"{config.STORAGE_ROOT}/evaluation", exist_ok=True)
    os.makedirs(f"{config.STORAGE_ROOT}/generated_question", exist_ok=True)
    os.makedirs(f"{config.STORAGE_ROOT}/temp_results", exist_ok=True)

    # HF login (push_to_hub needs it)
    config.HF_TOKEN = load_hf_token()
    hf_login(token=config.HF_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify all vllm endpoints are up before we start spending compute.
    wait_for_vllm_ready(config.VLLM_BASE_URL, label="vllm-base")
    for url in config.VLLM_SOLVER_URLS:
        wait_for_vllm_ready(url, label=f"vllm-solver@{url}")

    # Track MATH-500 accuracy at every milestone and dump to JSON at the end
    eval_history = {}
    summary_path = f"{config.STORAGE_ROOT}/results_summary.json"

    def save_summary():
        with open(summary_path, "w") as f:
            json.dump(eval_history, f, indent=2)

    # --- Baseline (iter 0): MATH-500 acc on the untouched Qwen3-4B-Base ---
    eval_history["iter_0_baseline"] = eval_math500(config.MODEL_NAME, "iter 0 baseline")
    save_summary()

    current_solver = config.MODEL_NAME           # iter 1 starts from base
    for it in range(1, config.NUM_ITERATIONS + 1):
        print(f"\n{'='*60}\n=== Iteration {it}/{config.NUM_ITERATIONS} ===\n{'='*60}")
        exp_name  = f"{config.MODEL_ABBR}_solver_v{it}"
        repo_name = exp_name
        log_path  = f"{config.STORAGE_ROOT}/logs/mcmc_iter_{it}.log"

        # 1. MCMC sampling (uses base for proposals, current solver for r_unc)
        train_data = generate_batch_mcmc(tokenizer, config.TOTAL_QUESTIONS, log_path)
        with open(f"{config.STORAGE_ROOT}/datasets/mcmc_iter_{it}.json", "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        # 2. Filter + push HF dataset
        n_pushed = filter_and_push(train_data, repo_name, exp_name)

        # 3. verl GRPO (subprocess)
        merged_ckpt = run_verl_solver(current_solver, f"{config.HF_USER}/{repo_name}", exp_name)

        # 4. MATH-500 eval on the freshly trained ckpt
        eval_history[f"iter_{it}"] = eval_math500(merged_ckpt, f"iter {it}")
        save_summary()

        # 5. Reload vllm-solver with the new merged checkpoint for next iter's MCMC
        if it < config.NUM_ITERATIONS:
            reload_vllm_solver(merged_ckpt)
        current_solver = merged_ckpt

    # --- Done. Print summary and run R-Zero's full eval suite at the end ---
    print("\n" + "=" * 60)
    print("MATH-500 progression:")
    for label, acc in eval_history.items():
        print(f"  {label:25s}  {acc:.4f}  ({acc * 100:.2f}%)")
    print("=" * 60)

    print("\n=== Final full eval suite (math/gsm8k/amc/minerva/olympiad/aime2024/aime2025) ===")
    subprocess.run(
        ["bash", "evaluation/evaluate.bash", current_solver],
        cwd=config.RZERO_DIR, check=True,
        env={**os.environ, "STORAGE_PATH": config.STORAGE_ROOT},
    )


if __name__ == "__main__":
    main()

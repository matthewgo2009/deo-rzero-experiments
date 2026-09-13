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
import gzip
import hashlib
import sys
import json
import time
import random
import subprocess
import itertools
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
    MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B-Base")  # env-overridable (e.g. Qwen/Qwen3-8B-Base)
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

    # --- Generation backend for challenger init + mutations (closed-source demo) ---
    # vllm      = frozen local base model (default, current behavior)
    # anthropic = Claude generates and mutates the questions; the solver, labels,
    #             judge and eval stay LOCAL and unchanged. Demonstrates that DEO's
    #             frozen proposal distribution can be ANY model, including API-only
    #             ones — impossible for R-Zero, whose challenger must be trainable.
    GEN_BACKEND = os.getenv("DEO_GEN_BACKEND", "vllm")
    GEN_MODEL_ANTHROPIC = os.getenv("DEO_GEN_MODEL", "claude-haiku-4-5-20251001")
    GEN_API_CONCURRENCY = int(os.getenv("DEO_GEN_CONCURRENCY", "8"))

    # --- Sampling for r_unc (matches R-Zero question_evaluate.py defaults) ---
    M_SAMPLES   = 9
    # --- Ceperley-Dewing U-statistic penalty acceptance (paper §1.4) ---
    CD_ENABLE    = os.getenv("DEO_CD_ACCEPT", "0") == "1"   # gate: use noise-debiased CD acceptance
    USTAT_M      = int(os.getenv("DEO_USTAT_M", "9"))       # subset size for order-m U-statistic
    USTAT_N      = int(os.getenv("DEO_USTAT_N", "12"))      # responses generated per question when CD on
    CD_FRESH_OLD = os.getenv("DEO_CD_FRESH_OLD", "0") == "1"  # re-score current state each step (else cache labels)

    # --- Memory-augmented mutation kernel (deo_with_memory.pdf §1.4): contextual
    #     Thompson sampling over strategies A-E. Bandit picks the strategy, the LLM
    #     realizes it; MH acceptance unchanged. Frozen within an outer iteration,
    #     discounted update afterwards (rho<1 forgets stale experience).
    BANDIT_ENABLE = os.getenv("DEO_BANDIT", "0") == "1"
    BANDIT_RHO    = float(os.getenv("DEO_BANDIT_RHO", "0.9"))
    BANDIT_EPS    = float(os.getenv("DEO_BANDIT_EPS", "0.1"))  # Trainable-context: require r_unc(x') >= r_unc(x)-eps

    # --- QUESTION_ANALYSIS_8B.md recommendations 1+2 ---
    # 1) STRIP_LEAKS: reject questions containing solution leakage (\boxed in the text,
    #    mutation meta-text) or lacking any interrogative/imperative — ~6-10% of DEO pools
    #    are wasted/poisoned prompts otherwise.
    # 2) EASY_BALLAST: reshape the uploaded training set so ~this share comes from
    #    p_hat in [0.6, MAX_SCORE] (labels most likely correct); rest is subsampled.
    STRIP_LEAKS  = os.getenv("DEO_STRIP_LEAKS", "0") == "1"
    EASY_BALLAST = float(os.getenv("DEO_EASY_BALLAST", "0"))

    # Big-init subset walk (pool-scale / selection-headroom test): generate INIT_POOL
    # questions per iter, then walk only TOTAL_QUESTIONS of them — in-band seeds chosen
    # at random (mirrors verl's uniform sampling of R-Zero's ~4000-question sets).
    # 0 = off (generate exactly TOTAL_QUESTIONS, current behavior).
    INIT_POOL = int(os.getenv("DEO_INIT_POOL", "0"))

    # Olympiad-register style operator (QUESTION_ANALYSIS rec #3): with this probability
    # a walk proposal uses the [F] OLYMPIAD_REWRITE operator (rephrase into formal
    # competition register, mathematics preserved) instead of the standard mutator. 0 = off.
    STYLE_P = float(os.getenv("DEO_STYLE_P", "0"))

    # --- Weakness memory (WEAKNESS_MEMORY_IMPLEMENTATION.md): minimal cross-iteration
    #     memory loop. Each evaluated proposal gets one short weakness note (written by
    #     the frozen base model from the solver's answer-cluster disagreement); each
    #     chain keeps the note of its final accepted state; post-filter notes are
    #     map-reduced into <=MEMORY_TOP_K global weaknesses; next iteration each chain
    #     samples one weakness w.p. MEMORY_GUIDED_PROB to guide its mutations.
    #     MH energy/acceptance, p_hat/r_unc/pseudo-labels, filtering and the training
    #     budget are untouched. Disabled (default) reproduces current behavior exactly.
    #     Implemented for the canonical fixed-beta non-CD walk only.
    WEAKNESS_MEMORY_ENABLED = os.getenv("DEO_WEAKNESS_MEMORY", "0") == "1"
    # LOCAL_WEAKNESS_FEEDBACK_IMPLEMENTATION.md: guidance source per proposal.
    #   global_fixed   = current behavior (one fixed global target per chain per iter)
    #   local_feedback = the accepted state's OWN note guides the next mutation;
    #                    missing note falls back to the assigned global target
    WM_GUIDANCE_MODE = os.getenv("DEO_WM_GUIDANCE_MODE", "global_fixed")
    MEMORY_GUIDED_PROB = float(os.getenv("DEO_MEMORY_GUIDED_PROB", "0.8"))
    MEMORY_TOP_K = int(os.getenv("DEO_MEMORY_TOP_K", "10"))
    MEMORY_MIN_SUPPORT = int(os.getenv("DEO_MEMORY_MIN_SUPPORT", "3"))
    MEMORY_SUMMARY_CHUNK_SIZE = int(os.getenv("DEO_MEMORY_SUMMARY_CHUNK_SIZE", "100"))
    MEMORY_TRACE_MAX_CHARS = int(os.getenv("DEO_MEMORY_TRACE_MAX_CHARS", "1500"))
    SOLVER_TEMP = 1.0
    SOLVER_TOP_P = 1.0
    SOLVER_TOP_K = 40
    SOLVER_MAX_TOKENS = 4096

    # --- MCMC ---
    BETA           = 0.1
    TAU_BLEU       = 0.5
    LAMBDA_REP     = 10.0
    TOTAL_QUESTIONS = 2000           # MCMC pool per iter (env DEO_TOTAL_Q overrides); ~1000-1300 pass the p_hat[0.3,0.8] filter
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
# Leak filter (QUESTION_ANALYSIS_8B.md rec #1, gated by DEO_STRIP_LEAKS): rejects
# questions with solution leakage or mutation meta-text, and questions with no
# interrogative/imperative (fragments that can't be answered).
_LEAK_MARKERS = ["\\boxed{", "original problem", "seed problem", "change the operation",
                 "<inside", "\\inst{", "mutated problem", "mutation strategy"]
_ASK_RE = re.compile(
    r"\?|find|determine|compute|calculate|evaluate|how many|how much|what is|what are|solve for|solve",
    re.IGNORECASE)


def question_is_leaky(q):
    ql = q.lower()
    if any(m in ql for m in _LEAK_MARKERS):
        return True
    if not _ASK_RE.search(q):
        return True   # no ask: statement fragment, cannot be a training prompt
    return False


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


def _anthropic_msg(system, user, max_tokens, temperature, top_p=1.0):
    """One Anthropic messages call via raw HTTP (the job image has requests; the
    pip-resolved SDK proved version-unstable on the node). The constant system
    prompt carries cache_control. The API discourages temperature AND top_p
    together, so when top_p<1 we send top_p only (recorded protocol choice)."""
    import requests
    body = {"model": config.GEN_MODEL_ANTHROPIC, "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}]}
    if top_p < 1.0:
        body["top_p"] = top_p
    else:
        body["temperature"] = min(1.0, temperature)
    r = requests.post(
        "https://api.anthropic.com/v1/messages", json=body, timeout=180,
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"]
                   if b.get("type") == "text")


def gen_texts_pairs(tokenizer, pairs, max_tokens, temperature, top_p=1.0):
    """Generate one text per (system, user) pair via the configured backend.

    vllm: chat-template rendered batch completions, choice.index attribution.
    anthropic: threaded messages API (3 retries, backoff); the constant system
    prompt gets cache_control to amortize input cost; temperature clamped to the
    API max of 1.0 (mutations use 1.1 locally — recorded protocol difference).
    Returns texts aligned with pairs; None marks a failed slot (never shifts)."""
    if config.GEN_BACKEND == "vllm":
        prompts = [apply_chat_template(tokenizer, sy, us) for sy, us in pairs]
        resp = base_client().completions.create(
            model=config.MODEL_NAME, prompt=prompts,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return _ordered_completion_texts(resp, len(prompts))
    if config.GEN_BACKEND == "anthropic":
        def one(pair):
            sy, us = pair
            for attempt in range(3):
                try:
                    return _anthropic_msg(sy, us, max_tokens, temperature, top_p)
                except Exception as e:
                    if attempt == 2:
                        print(f"[gen] anthropic failed after retries: {str(e)[:120]}",
                              flush=True)
                    time.sleep(2 * (attempt + 1))
            return None
        with ThreadPoolExecutor(max_workers=config.GEN_API_CONCURRENCY) as ex:
            return list(ex.map(one, pairs))
    raise ValueError(f"unknown DEO_GEN_BACKEND={config.GEN_BACKEND!r}")


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

# --- V1 (default): aggressive "structurally different" mutator ---
_MUTATOR_SYSTEM_PROMPT_V1 = """You are an expert competition-math problem setter. I will provide a seed problem.
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

_MUTATOR_USER_TEMPLATE_V1 = (
    "Here is the seed problem:\n{seed}\n\n"
    "Pick ONE mutation strategy from {{A,B,C,D,E}} and apply it now. "
    "Remember: number-swapping is FAILURE."
)

# --- V2 (env DEO_MUT_PROMPT=v2): conservative "one localized step, stay close, validity-checked" mutator ---
_MUTATOR_SYSTEM_PROMPT_V2 = r"""You are an expert competition-math problem setter. I will provide a seed problem.
Your task is to generate a valid, self-contained, moderately harder problem by applying exactly ONE
localized mutation strategy.

The mutated problem should remain close to the seed problem. Preserve its main mathematical topic,
objects, notation, and core solution structure. Increase the required reasoning by approximately one
meaningful step, rather than making the problem arbitrarily complicated.

MUTATION STRATEGIES (pick exactly ONE):

[A] GENERALIZE — Replace one fixed constant with a simple parameter, or extend one existing relationship
to a closely related case. Do NOT raise the dimension, introduce unrelated objects, or convert a
single-variable problem into a multivariable problem unless this is a direct and natural extension.

[B] COMPOSE — Add exactly ONE mathematically compatible condition that interacts with the original
structure and adds one meaningful inference step. Do NOT combine unrelated topics or attach an
arbitrary second problem.

[C] INVERT — Exchange one given quantity with the requested quantity, so that the reader recovers an
input or precondition. Ensure the new conditions determine a unique answer.

[D] CHANGE_OBJECTIVE — Change only what is being requested while preserving the original setting, such
as changing "find x" to "count the valid values of x", "find a value" to "find the sum of all valid
values", or "compute" to "find the smallest value satisfying the same structure".

[E] DUALIZE — Replace one concept by a closely related dual concept only when the resulting problem is
mathematically natural and fully determined, such as gcd↔lcm, max↔min, or area↔perimeter.
Do NOT mechanically swap concepts when the new statement would be ambiguous or false.

CRITICAL VALIDITY RULES:

1. Apply exactly ONE localized mutation. Do not make multiple independent changes.
2. Preserve the seed problem's mathematical domain, main objects, and general solution method.
3. Every variable, symbol, function, point, and mathematical object must be explicitly defined.
4. All conditions must be mutually consistent and sufficient to determine the answer.
5. The problem must have exactly one unambiguous final answer, unless it explicitly asks for a finite
   set or for all valid values.
6. The final answer MUST be a SPECIFIC NUMBER, ALGEBRAIC EXPRESSION, or FINITE SET.
7. The problem must be understandable without a missing figure, diagram, source problem, or external
   context.
8. Do NOT include the answer, a boxed expression, solution steps, intermediate conclusions, or hints
   inside the question.
9. Do NOT refer to the "seed problem", "original problem", "previous question", mutation strategy,
   prompt, or generation process inside the question.
10. Do NOT introduce higher-dimensional geometry unless the seed is already in that same dimension.
11. Do NOT combine unrelated mathematical topics, add arbitrary advanced terminology, or invent
    unnecessary objects.
12. Do NOT output malformed LaTeX, placeholders, template text, code, XML/HTML inside the question,
    corrupted symbols, or meta-commentary.
13. NO "Prove that", "Show that", "Justify", "Explain why", True/False, Yes/No, or open-ended questions.
14. Do NOT merely make the problem longer. Difficulty must come from one valid additional inference.
15. Do NOT make a mutation whose validity depends on an unstated theorem-specific exceptional case.

Before producing the output, privately verify:

* every symbol is defined;
* the conditions are consistent;
* the requested value exists;
* the answer is unique or the requested finite set is well-defined;
* the proposed boxed answer actually follows from the question;
* the answer is not revealed anywhere in the question.

If the selected strategy cannot produce a valid localized mutation, choose another strategy. Prefer a
small, mathematically safe mutation over a creative but ambiguous one.

LIMIT scratch-pad reasoning to UNDER 50 WORDS.

Output format (STRICT — all three components required, with no extra text): <strategy>{A|B|C|D|E}</strategy> <question>
[the complete NEW mutated problem statement] </question>
\boxed{final_answer}"""

_MUTATOR_USER_TEMPLATE_V2 = (
    "Here is the seed problem:\n{seed}\n\n"
    "Apply exactly ONE localized mutation strategy from {{A,B,C,D,E}}. "
    "Keep the same mathematical topic and core structure. "
    "The new problem must be self-contained, well-posed, uniquely answerable, and only moderately harder. "
    "Do not leak the answer or include solution steps."
)

# Select mutator prompt via env (default = V1). DEO_MUT_PROMPT=v2 -> conservative localized mutator.
if os.getenv("DEO_MUT_PROMPT", "v1").lower() == "v2":
    MUTATOR_SYSTEM_PROMPT = _MUTATOR_SYSTEM_PROMPT_V2
    MUTATOR_USER_TEMPLATE = _MUTATOR_USER_TEMPLATE_V2
    print("[mutator] using V2 (conservative localized) mutation prompt", flush=True)
else:
    MUTATOR_SYSTEM_PROMPT = _MUTATOR_SYSTEM_PROMPT_V1
    MUTATOR_USER_TEMPLATE = _MUTATOR_USER_TEMPLATE_V1

# --- Bandit-forced mutation (deo_with_memory.pdf §1.4): the bandit picks strategy a,
#     the LLM must realize K_a. Dedicated system prompt: the operator is selected
#     EXTERNALLY (no "pick one" / "switch strategy" language that could make the model
#     execute a different operator than the one the bandit gets credit for).
#     Style: V1 large structural steps + V2 validity constraints, minus V2 locality.
from mutation_bandit import MutationBandit, ACTION_NAMES  # noqa: E402

# Per-operator descriptions (V1 wording). The bandit prompt shows ONLY the selected
# operator — the model is never offered a menu, so there is nothing to "switch" to.
STRATEGY_DESCRIPTIONS = {
    "A": "GENERALIZE — Lift the structure: replace a specific constant with a parameter, OR raise the dimension\n"
         "(2D → 3D, single-variable → multivariable, single equation → system of equations).",
    "B": "COMPOSE — Add a NEW non-trivial second condition that interacts with the existing structure\n"
         "(NOT just a range bound like \"with x > 0\"). The two conditions together must create a real interaction.",
    "C": "INVERT — Given the original answer or output, ask the reader to recover an input or precondition.",
    "D": "CHANGE_OBJECTIVE — Change WHAT is being asked: e.g. \"find x\" → \"count integer solutions\",\n"
         "\"compute\" → \"find the smallest n such that ...\", \"find value\" → \"find sum of all such values\".",
    "E": "DUALIZE — Swap to a dual concept: sum↔product, max↔min, area↔perimeter,\n"
         "gcd↔lcm, addition↔multiplication, distance↔angle.",
}

# Single-operator system prompt (placeholders replaced via .replace to avoid brace escaping).
_MUTATOR_SYSTEM_PROMPT_BANDIT_TMPL = """You are an expert competition-math problem setter. I will provide a seed problem.
Your task is to generate a NEW problem by applying the following mutation operation to the seed:

[__LETTER__] __STRATEGY_BLOCK__

CRITICAL VALIDITY RULES:
1. DO NOT just swap numbers. If the only edit is a digit change, you have FAILED.
2. Every variable, symbol, function, and mathematical object must be explicitly defined.
3. All conditions must be mutually consistent and sufficient to determine the answer.
4. The final answer MUST be a SPECIFIC NUMBER, ALGEBRAIC EXPRESSION, or FINITE SET, and must be unique
   (unless the question explicitly asks for a finite set or all valid values).
5. Do NOT include the answer, solution steps, or hints inside the question; do NOT refer to the seed
   problem, the mutation operation, or the generation process inside the question.
6. NO "Prove that", "Show that", "Justify", True/False, or Yes/No questions.
7. Do NOT output malformed LaTeX, placeholders, code, or XML/HTML inside the question text.
8. LIMIT scratch-pad reasoning to UNDER 50 WORDS.

Output format (STRICT — all three tags required):
<strategy>__LETTER__</strategy>
<question>
[the NEW mutated problem statement]
</question>
\\boxed{final_answer}"""


def mutator_system_prompt_bandit(action):
    return (_MUTATOR_SYSTEM_PROMPT_BANDIT_TMPL
            .replace("__STRATEGY_BLOCK__", STRATEGY_DESCRIPTIONS[action])
            .replace("__LETTER__", action))


MUTATOR_USER_TEMPLATE_FORCED = (
    "Here is the seed problem:\n{seed}\n\n"
    "Apply the [{action}] ({action_name}) mutation to it now. "
    "Remember: number-swapping is FAILURE."
)

# [F] OLYMPIAD_REWRITE (QUESTION_ANALYSIS rec #3): style-transfer operator — rewrite the
# seed in formal olympiad register, identical mathematics. Proposals are re-scored with
# fresh solver rollouts like any mutation, so an unfaithful rewrite still gets a
# consistent fresh label.
MUTATOR_SYSTEM_PROMPT_STYLE = """You are an expert competition-math problem editor. I will provide a seed problem.
Your task is to REWRITE it in formal olympiad register while preserving the underlying mathematics EXACTLY.

REWRITE RULES:
1. Keep every quantity, constraint, mathematical object, and the final answer IDENTICAL to the seed.
   Do NOT change numbers, conditions, or what is being asked for mathematically.
2. Rephrase into formal competition style: introduce objects with "Let ..." / "Suppose further that ...",
   state conditions as separate formal clauses, and phrase the goal as "Determine ..." / "Find the value of ...".
3. Prefer precise quantifiers ("for every positive integer n", "there exists a unique") over casual wording;
   remove story/narrative framing (names, everyday objects) in favor of abstract mathematical statement.
4. The rewrite should read like an AMC/AIME/olympiad problem statement. Length may grow moderately,
   but do NOT add new conditions, hints, or solution steps.
5. Do NOT include the answer or any \\boxed expression inside the question text.
6. LIMIT scratch-pad reasoning to UNDER 30 WORDS.

Output format (STRICT — all three tags required):
<strategy>F</strategy>
<question>
[the rewritten problem statement]
</question>
\\boxed{final_answer}"""

MUTATOR_USER_TEMPLATE_STYLE = (
    "Here is the seed problem:\n{seed}\n\n"
    "Rewrite it in formal olympiad register now. Preserve the mathematics and the final answer exactly; "
    "change only the phrasing and presentation."
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
def evaluate_r_unc_vllm(tokenizer, questions, return_labels=False, return_details=False):
    """Score each question by majority-vote consistency on the current solver.

    Sampling settings + chat template match R-Zero's question_evaluate.py and
    verl/utils/dataset.py:196 so pseudo-labels are drawn from the same
    distribution verl will train on.

    Samples n=USTAT_N responses when CD acceptance is enabled (so the MCMC step
    can form the order-USTAT_M U-statistic), else M_SAMPLES.

    Returns: (r_unc_list, p_hat_list, pseudo_label_list[, labels_list][, details_list])
    where labels_list[i] is the list of n canonical answers for question i and
    details_list[i] (weakness memory) holds answer-cluster counts plus one truncated
    representative trace per top cluster. Full rollout texts are not retained.
    """
    if not questions:
        empty = [[], [], []] + ([[]] if return_labels else []) + ([[]] if return_details else [])
        return tuple(empty)
    m = config.USTAT_N if config.CD_ENABLE else config.M_SAMPLES

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
        # R6: per-shard index attribution — a missing/duplicate choice keeps its
        # slot as None (counted as an invalid answer) instead of shifting every
        # later rollout into the wrong question's m-slice
        st = {}
        texts_shard = _ordered_completion_texts(resp, len(chunk), st)
        if st.get("missing") or st.get("dup_index") or st.get("invalid_index"):
            print(f"[eval] WARNING shard {idx} response anomalies: {st} "
                  f"(missing slots stay None; p_hat denominators unchanged)", flush=True)
        return idx, texts_shard

    texts_by_shard = [None] * n_dp
    with ThreadPoolExecutor(max_workers=n_dp) as ex:
        for fut in as_completed([ex.submit(_run_shard, i) for i in range(n_dp)]):
            idx, txt = fut.result()
            texts_by_shard[idx] = txt
    texts = [t for shard in texts_by_shard for t in shard]
    answers = [extract_solver_answer(t) if t is not None else None for t in texts]

    r_unc_list, p_hat_list, pseudo_list, labels_list, details_list = [], [], [], [], []
    for i in range(len(questions)):
        chunk = answers[i * m: (i + 1) * m]
        labels_list.append(chunk)
        if return_details:
            details_list.append(_build_cluster_details(chunk, texts[i * m: (i + 1) * m]))
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

    out = [r_unc_list, p_hat_list, pseudo_list]
    if return_labels:
        out.append(labels_list)
    if return_details:
        out.append(details_list)
    return tuple(out)


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


# ==========================================
# Ceperley-Dewing U-statistic penalty acceptance helpers (paper §1.4.1)
# ==========================================
_USTAT_FULL = {}   # n -> list of size-m subset index tuples
_USTAT_JK = {}     # n -> [ per deleted-j: list of size-m subsets excluding j ]


def _ustat_index_cache(n, m):
    if n not in _USTAT_FULL:
        full = list(itertools.combinations(range(n), m))
        _USTAT_FULL[n] = full
        _USTAT_JK[n] = [[s for s in full if j not in s] for j in range(n)]
    return _USTAT_FULL[n], _USTAT_JK[n]


def ustat_jackknife(labels, r_rep_k, m=None):
    """Order-m U-statistic estimate of the per-question utility
        u = mean_{|S|=m} [ (1 - 2|p_hat(S) - 1/2|) - lambda_rep * r_rep_k ]_+
    over all size-m subsets S of the n canonical answer labels, plus its
    delete-one jackknife variance v (paper eq 23, 26). Invalid answers
    (None / GUESSED_FAIL_FORMAT) never join a modal group, matching
    evaluate_r_unc_vllm's p_hat = count/m convention. Returns (u, v)."""
    lam = config.LAMBDA_REP
    n = len(labels)
    if m is None:
        m = config.USTAT_M
    if n == 0 or n < m:
        return 0.0, 0.0
    idmap, ids = {}, []
    for a in labels:
        if a is None or a == "" or a == "GUESSED_FAIL_FORMAT":
            ids.append(-1)
        else:
            ids.append(idmap.setdefault(a, len(idmap)))
    full, jk = _ustat_index_cache(n, m)

    def kernel(subidx):
        c, best = {}, 0
        for t in subidx:
            v = ids[t]
            if v < 0:
                continue
            nc = c.get(v, 0) + 1
            c[v] = nc
            if nc > best:
                best = nc
        p = best / m
        return max(0.0, (1.0 - 2.0 * abs(p - 0.5)) - lam * r_rep_k)

    u = sum(kernel(s) for s in full) / len(full)
    ujk = [(sum(kernel(s) for s in jk[j]) / len(jk[j]) if jk[j] else 0.0) for j in range(n)]
    ubar = sum(ujk) / n
    v = (n - 1) / n * sum((x - ubar) ** 2 for x in ujk)
    return float(u), float(v)


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
# ==========================================
# 7b. Weakness memory (WEAKNESS_MEMORY_IMPLEMENTATION.md + WEAKNESS_MEMORY_FIXES.md)
# ==========================================
WEAKNESS_DOMAINS = {"algebra", "geometry", "number_theory", "combinatorics",
                    "probability", "calculus", "other"}
WM_WRITER_PROMPT_VERSION = "wm_writer_v3"
WM_SUMMARY_PROMPT_VERSION = "wm_summary_v2"
WM_CONTEXT_LIMIT = 6144          # base vLLM --max-model-len
WM_WRITER_MAX_TOKENS = 300
WM_SUMMARY_MAX_TOKENS = 1536
WM_MERGE_MAX_ITEMS = 16   # verbose 4B outputs: ~60-90 tokens/group; cap items so JSON never truncates

WEAKNESS_WRITER_SYSTEM = """You analyze how a math solver's sampled responses disagree.

Given one problem and up to three clusters of solver responses (with counts), return
exactly one JSON object with keys: status, domain, weakness, evidence, cluster_id,
excerpt.

status must be one of:
- "weakness": you can point to a specific reasoning step on which the clusters
  disagree.
- "problem_issue": the problem itself looks contradictory, underspecified, broken,
  or asks about an undefined object.
- "insufficient_evidence": you can only tell the topic, or the traces do not reveal
  where the reasoning diverges.

domain must be one of algebra, geometry, number_theory, combinatorics, probability,
calculus, other.

weakness: one sentence, IN YOUR OWN WORDS, naming the transferable reasoning
operation these clusters disagree on. Name the specific operation this problem's
solutions diverge at, not merely the problem topic, and do not copy problem-specific
constants into the weakness label.

evidence: one sentence saying WHICH clusters diverge at WHICH step. Quoting the
problem's numbers and the clusters' answers here IS allowed.

cluster_id: the integer id (1-3) of the cluster your excerpt comes from.
excerpt: a short quote (at most 120 characters) copied VERBATIM from that cluster's
representative reasoning shown to you.

Never treat the majority answer as verified; only describe visible disagreement.
First consider whether the disagreement comes from a defect of the problem itself.
If you cannot locate the diverging step in the traces, return insufficient_evidence.
Do not invent problems, numbers, reasoning or answers that are not in the input."""

WEAKNESS_GUIDANCE_TMPL = """

KNOWN SOLVER WEAKNESS ({domain}):
{weakness}
Observed disagreement: {evidence}

Within the seed problem's own mathematical structure, mutate it so that solving the
new problem genuinely requires the reasoning operation above. If the seed cannot
reasonably be steered toward it, apply a normal high-quality mutation instead. Do
not force keywords, introduce contradictory conditions, copy an old question, or
leak the answer. Preserve a unique, verifiable answer and apply exactly one
strategy A-E."""

# Few-shot completion-style classification: a BASE model given a bare "classify
# this" instruction just starts SOLVING the problem (observed in smoke: completions
# began "SOLUTION: Step 1..."). A two-shot PROBLEM/DOMAIN pattern + stop at newline
# pins the continuation to a single domain word.
WM_DOMAIN_CLASSIFY_SYSTEM = (
    "Classify each math problem into exactly one domain from this list: algebra, "
    "geometry, number_theory, combinatorics, probability, calculus, other.")
WM_DOMAIN_CLASSIFY_FEWSHOT = (
    "PROBLEM: If 3x + 2 = 8, find x.\nDOMAIN: algebra\n\n"
    "PROBLEM: In how many ways can 5 people sit in a row?\nDOMAIN: combinatorics\n\n"
    "PROBLEM: {q}\nDOMAIN:")


def _tok_len(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _norm_ws(text):
    return " ".join((text or "").split()).lower()


def _norm_cite(text):
    """Citation normalization (R2): collapse whitespace ONLY. Case, symbols,
    digits and math notation are preserved — `A` and `a` are different symbols."""
    return " ".join((text or "").split())


def _truncate_trace(text, max_chars=None):
    """Keep the beginning and the ending of a long trace (reasoning setup + final
    answer). Deterministic, so the writer's view can be reconstructed from the raw
    trace + max_chars."""
    max_chars = max_chars or config.MEMORY_TRACE_MAX_CHARS
    if text is None or len(text) <= max_chars:
        return text or ""
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def _build_cluster_details(answers_chunk, texts_chunk, top_n=3):
    """Cluster the m rollout answers (same Counter grouping p_hat uses).

    clusters_all keeps EVERY valid cluster's answer/count (fixes doc 3.1.2);
    clusters carries representative traces for the top-N only, each tagged with
    cluster_id / rollout_id / truncation flag so the writer's citations are
    verifiable. One question = one observation."""
    valid_ix = [j for j, a in enumerate(answers_chunk)
                if a is not None and a != "GUESSED_FAIL_FORMAT"]
    counts = Counter(answers_chunk[j] for j in valid_ix)
    clusters_all = [{"answer": a, "count": c} for a, c in counts.most_common()]
    clusters = []
    for cid, (ans, cnt) in enumerate(counts.most_common(top_n), 1):
        rid = next(j for j in valid_ix if answers_chunk[j] == ans)
        raw = texts_chunk[rid] or ""
        clusters.append({"cluster_id": cid, "answer": ans, "count": cnt,
                         "rollout_id": rid,
                         "representative_trace": _truncate_trace(raw),
                         "truncated": len(raw) > config.MEMORY_TRACE_MAX_CHARS})
    return {"rollout_count": len(answers_chunk),
            "valid_answer_count": len(valid_ix),
            "invalid_answer_count": len(answers_chunk) - len(valid_ix),
            "clusters_all": clusters_all,
            "clusters": clusters,
            # transient: raw rollouts for provenance logging (doc 3.1.4); the
            # logger gzips them, then callers strip this key — never kept long-term
            "_raw": {"answers": list(answers_chunk), "texts": list(texts_chunk)}}


def _ordered_completion_texts(resp, n, stats=None):
    """Map an OpenAI-style batched completion onto request order by choice.index
    (doc 3.1.5 + R6): a missing response yields None at its slot instead of
    shifting every later item; duplicate indices keep the first and are counted.
    Falls back to list order ONLY when no choice exposes an integer index at all
    AND the count matches exactly — indices that exist but are all invalid never
    silently degrade to positional matching."""
    out = [None] * n
    saw_attr = False
    dup = bad = 0
    for c in resp.choices:
        i = getattr(c, "index", None)
        if isinstance(i, int) and not isinstance(i, bool):
            saw_attr = True
            if 0 <= i < n:
                if out[i] is None:
                    out[i] = c.text
                else:
                    dup += 1
            else:
                bad += 1
    if not saw_attr and len(resp.choices) == n:
        out = [c.text for c in resp.choices]
    if stats is not None:
        stats["dup_index"] = stats.get("dup_index", 0) + dup
        stats["invalid_index"] = stats.get("invalid_index", 0) + bad
        stats["missing"] = stats.get("missing", 0) + sum(1 for o in out if o is None)
    return out


class WmLogger:
    """Provenance sink (doc 3.3 + R7): per-iteration notes jsonl, structured event
    jsonl, and gzipped raw solver rollouts for every writer-eligible question.

    Rerun-safe: each logger instance claims a fresh attempt number, so a restart
    of the same iteration writes to NEW `.a{k}` files (no overwrite/append mixing)
    and event ids embed the attempt — every event stays uniquely attributable.
    Raw rollouts and writer outputs are stored in full (compressed), never
    truncated."""

    def __init__(self, iteration, run_id=None):
        self.iter = iteration
        self.run_id = run_id or config.MODEL_ABBR
        d = _wm_dir()
        attempt = 1
        while os.path.exists(f"{d}/wm_events_iter_{iteration}.a{attempt}.jsonl"):
            attempt += 1
        self.attempt = attempt
        self.notes_f = open(f"{d}/weakness_notes_iter_{iteration}.a{attempt}.jsonl",
                            "w", encoding="utf-8")
        self.events_f = open(f"{d}/wm_events_iter_{iteration}.a{attempt}.jsonl",
                             "w", encoding="utf-8")
        self.traces_f = gzip.open(f"{d}/raw_rollouts_iter_{iteration}.a{attempt}.jsonl.gz",
                                  "wt", encoding="utf-8")
        self._seq = 0
        self.event(kind="meta", model=config.MODEL_NAME, model_abbr=config.MODEL_ABBR,
                   writer_prompt_version=WM_WRITER_PROMPT_VERSION,
                   summary_prompt_version=WM_SUMMARY_PROMPT_VERSION,
                   trace_max_chars=config.MEMORY_TRACE_MAX_CHARS,
                   azureml_run=os.environ.get("AZUREML_RUN_ID") or os.environ.get("MLFLOW_RUN_ID"),
                   code_version=os.environ.get("DEO_CODE_VERSION"))

    def new_event_id(self, kind):
        self._seq += 1
        return f"{self.run_id}_it{self.iter}a{self.attempt}_{kind}_{self._seq}"

    def event(self, **kw):
        kw.setdefault("run_id", self.run_id)
        kw.setdefault("iter", self.iter)
        kw.setdefault("attempt", getattr(self, "attempt", None))
        self.events_f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.events_f.flush()

    def traces(self, event_id, question, answers, texts):
        # FULL texts (R7): compression is fine, silent truncation is not
        self.traces_f.write(json.dumps(
            {"event_id": event_id, "question": question, "answers": answers,
             "texts": texts}, ensure_ascii=False) + "\n")

    def note_row(self, **kw):
        self.notes_f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.notes_f.flush()

    def close(self):
        for f in (self.notes_f, self.events_f, self.traces_f):
            try:
                f.close()
            except Exception:
                pass


def _extract_json_value(text, open_ch="{", close_ch="}"):
    """First parseable JSON value starting at an `open_ch` in the completion.

    Uses a real JSON decoder (raw_decode), so braces/brackets INSIDE string
    literals and escape sequences are handled correctly (R5) — a plain
    bracket-counter misparses e.g. an excerpt containing "the set {"."""
    dec = json.JSONDecoder()

    def _scan(t):
        pos = t.find(open_ch)
        while pos != -1:
            try:
                val, _end = dec.raw_decode(t, pos)
                return val
            except Exception:
                pos = t.find(open_ch, pos + 1)
        return None

    val = _scan(text)
    if val is not None:
        return val
    # LaTeX repair: base models emit raw \frac / \boxed / \( inside JSON strings —
    # invalid escapes that kill the whole parse. Re-escape every backslash that is
    # not opening a string-closing escape, so \boxed survives VERBATIM as \boxed
    # (a naive invalid-escape-only repair would eat \b -> backspace and break
    # citation matching). Repair applies only after the strict parse failed.
    return _scan(re.sub(r'\\(?!["\\])', r"\\\\", text))


def _parse_weakness_note(text, details=None):
    """Parse + structurally validate a v2 writer output against the details the
    writer actually saw. Returns (status, note_or_None, reason).

    status="weakness" requires a verifiable citation: cluster_id must belong to the
    request and the excerpt must appear verbatim (whitespace-normalized) in that
    cluster's representative trace. Validation proves the citation exists — it does
    NOT prove the diagnosis is semantically right or the question valid."""
    obj = _extract_json_value(text)
    if not isinstance(obj, dict):
        return "parse_failed", None, "no JSON object"
    status = str(obj.get("status") or "").strip().lower()
    if status not in ("weakness", "problem_issue", "insufficient_evidence"):
        return "parse_failed", None, f"bad status {status!r}"
    domain = str(obj.get("domain") or "").strip().lower().replace(" ", "_")
    if domain not in WEAKNESS_DOMAINS:
        domain = "other"
    weakness = str(obj.get("weakness") or "").strip()[:300]
    evidence = str(obj.get("evidence") or "").strip()[:400]
    if status != "weakness":
        return status, None, weakness or evidence or status
    if not weakness:
        return "parse_failed", None, "weakness status without weakness text"
    try:
        cid = int(obj.get("cluster_id"))
    except Exception:
        return "parse_failed", None, "missing/invalid cluster_id"
    excerpt = str(obj.get("excerpt") or "").strip()[:200]
    if details is not None:
        cl = next((c for c in details["clusters"] if c["cluster_id"] == cid), None)
        if cl is None:
            return "parse_failed", None, f"cluster_id {cid} not in request"
        # R2: match against the trace ACTUALLY SHOWN to the writer this attempt,
        # whitespace-collapsed but case/symbol-preserving (A != a)
        if not excerpt or _norm_cite(excerpt) not in _norm_cite(cl["representative_trace"]):
            return "parse_failed", None, "excerpt not found in the trace shown to the writer"
    return "weakness", {"domain": domain, "weakness": weakness, "evidence": evidence,
                        "cluster_id": cid, "excerpt": excerpt}, "ok"


def _shown_details(details, trace_chars):
    """The details as ACTUALLY rendered for one writer attempt (R2): traces
    re-truncated to this attempt's budget. Citation validation must use this,
    never the full details."""
    shown = dict(details)
    shown["clusters"] = [dict(cl, representative_trace=_truncate_trace(
        cl["representative_trace"], trace_chars)) for cl in details["clusters"]]
    return shown


def _writer_user_prompt(question, p_hat, details, trace_chars):
    m = details["rollout_count"]
    counts_all = ", ".join(f"{c['answer']}:{c['count']}" for c in details["clusters_all"])
    lines = [f"PROBLEM:\n{question}\n",
             f"SELF-CONSISTENCY:\np_hat={p_hat:.3f}; "
             f"valid_answers={details['valid_answer_count']}/{m}; "
             f"invalid={details['invalid_answer_count']}\n",
             f"ALL ANSWER COUNTS: {counts_all}\n",
             "TOP CLUSTERS WITH REPRESENTATIVE REASONING:"]
    for cl in details["clusters"]:
        tr = _truncate_trace(cl["representative_trace"], trace_chars)
        lines.append(f"cluster_id={cl['cluster_id']}: answer={cl['answer']}, "
                     f"count={cl['count']}\n   representative reasoning={tr}")
    lines.append("\nReturn the JSON object now.")
    return "\n".join(lines)


def generate_weakness_notes_batch(tokenizer, questions, p_hats, pseudos, details_list,
                                  logger=None, stage="", chains=None):
    """One outcome per question. Returns (notes, statuses, event_ids):
    notes[i] is a dict ONLY when status=weakness (problem_issue /
    insufficient_evidence are audit-only and never enter memory); statuses[i] in
    {not_eligible, request_failed, parse_failed, weakness, problem_issue,
    insufficient_evidence}. Reuses the existing m rollouts — no extra solver calls.
    Never raises into the MCMC run."""
    n = len(questions)
    notes = [None] * n
    statuses = ["not_eligible"] * n
    event_ids = [None] * n
    chains = chains if chains is not None else list(range(n))
    elig = [i for i in range(n)
            if pseudos[i] not in (None, "", "None")
            and config.MIN_SCORE <= float(p_hats[i]) <= config.MAX_SCORE
            and details_list[i] and details_list[i].get("clusters")]
    for i in elig:
        statuses[i] = "request_failed"       # until proven otherwise
        if logger:
            eid = logger.new_event_id("eval")
            event_ids[i] = eid
            det = details_list[i]
            logger.event(kind="eval", event_id=eid, chain=chains[i], stage=stage,
                         question_sha1=_sha1(questions[i]), m=det["rollout_count"],
                         p_hat=float(p_hats[i]), counts=det["clusters_all"],
                         invalid=det["invalid_answer_count"])
            raw = det.get("_raw")
            if raw:
                logger.traces(eid, questions[i], raw["answers"], raw["texts"])

    def _render(i):
        """Token-budget fit for the FINAL full prompt incl. chat template + output
        reserve (doc 3.1.6 + R4). Returns (prompt, chars, shown_details), or
        (None, chars, None) when even the floor-truncated request cannot fit —
        such an item is excluded from the batch as input_too_long instead of
        poisoning it."""
        chars = config.MEMORY_TRACE_MAX_CHARS
        while True:
            prompt = apply_chat_template(
                tokenizer, WEAKNESS_WRITER_SYSTEM,
                _writer_user_prompt(questions[i], float(p_hats[i]), details_list[i], chars))
            if _tok_len(tokenizer, prompt) + WM_WRITER_MAX_TOKENS <= WM_CONTEXT_LIMIT:
                return prompt, chars, _shown_details(details_list[i], chars)
            if chars <= 300:
                return None, chars, None
            chars = max(300, chars // 2)

    pending = []
    for i in elig:
        p_, c_, sh_ = _render(i)
        if p_ is None:
            statuses[i] = "input_too_long"    # R4: never enters a batch
            if logger:
                logger.event(kind="writer", event_id=event_ids[i], attempt=0,
                             prompt_version=WM_WRITER_PROMPT_VERSION, ok=False,
                             status="input_too_long", trace_chars=c_,
                             reason="prompt exceeds context at floor truncation")
        else:
            pending.append(i)
    for attempt in range(2):
        if not pending:
            break
        rendered = {i: _render(i) for i in pending}   # (prompt, chars, shown)
        try:
            resp = base_client().completions.create(
                model=config.MODEL_NAME, prompt=[rendered[i][0] for i in pending],
                max_tokens=WM_WRITER_MAX_TOKENS, temperature=0.2, top_p=0.9)
        except Exception as e:
            print(f"[wm] writer batch failed (attempt {attempt + 1}): {e}", flush=True)
            if logger:
                for i in pending:
                    logger.event(kind="writer", event_id=event_ids[i], attempt=attempt + 1,
                                 prompt_version=WM_WRITER_PROMPT_VERSION,
                                 ok=False, status="request_failed", reason=str(e)[:200])
            continue
        texts = _ordered_completion_texts(resp, len(pending))
        still = []
        for i, t in zip(pending, texts):
            if t is None:
                statuses[i] = "request_failed"
                still.append(i)
                if logger:
                    logger.event(kind="writer", event_id=event_ids[i], attempt=attempt + 1,
                                 prompt_version=WM_WRITER_PROMPT_VERSION, ok=False,
                                 status="request_failed", reason="missing choice index")
                continue
            # R2: validate the citation against what THIS attempt actually showed
            status, note, reason = _parse_weakness_note(t, rendered[i][2])
            if logger:
                logger.event(kind="writer", event_id=event_ids[i], attempt=attempt + 1,
                             prompt_version=WM_WRITER_PROMPT_VERSION,
                             ok=status != "parse_failed", status=status,
                             reason=reason[:200], trace_chars=rendered[i][1],
                             raw_output=t)
            if status == "parse_failed":
                statuses[i] = "parse_failed"
                still.append(i)
            else:
                statuses[i] = status
                notes[i] = note   # None unless status == "weakness"
        pending = still
    if elig:
        cnt = Counter(statuses[i] for i in elig)
        print(f"[wm] writer[{stage}]: {len(elig)} eligible -> {dict(cnt)}", flush=True)
    return notes, statuses, event_ids


def _wm_dir():
    d = f"{config.STORAGE_ROOT}/weakness_memory"
    os.makedirs(d, exist_ok=True)
    return d


def load_global_weakness_memory(iteration):
    """Global memory written after iteration `iteration` (frozen for the whole walk).
    Items carry source_iter so a target is never attributed to the wrong memory."""
    path = f"{_wm_dir()}/global_weakness_memory_iter_{iteration}.json"
    if not os.path.exists(path):
        print(f"[wm] no global memory at {path}; running unguided", flush=True)
        return []
    try:
        with open(path) as f:
            mem = json.load(f)
        if not isinstance(mem, list):
            return []
        for it_ in mem:
            it_.setdefault("source_iter", iteration)
        return mem
    except Exception as e:
        print(f"[wm] failed to load {path}: {e}; running unguided", flush=True)
        return []


def _weighted_pick(rng, items):
    ws = [max(1e-6, float(it.get("support", 1)) *
              (1.0 - abs(float(it.get("avg_p_hat", 0.5)) - 0.5))) for it in items]
    r = rng.random() * sum(ws)
    for it, w in zip(items, ws):
        r -= w
        if r <= 0:
            return it
    return items[-1]


def classify_domains_batch(tokenizer, questions, logger=None):
    """One light batched call: coarse domain per question, None on any failure.
    Matches the EARLIEST domain keyword anywhere in the completion (space or
    underscore form) — base-model answers rarely start with the bare word."""
    if not questions:
        return []
    out = [None] * len(questions)
    raw = [None] * len(questions)
    try:
        prompts = [apply_chat_template(tokenizer, WM_DOMAIN_CLASSIFY_SYSTEM,
                                       WM_DOMAIN_CLASSIFY_FEWSHOT.format(q=q[:2000]))
                   for q in questions]
        resp = base_client().completions.create(
            model=config.MODEL_NAME, prompt=prompts, max_tokens=8, temperature=0.0,
            stop=["\n"])
        texts = _ordered_completion_texts(resp, len(questions))
        for i, t in enumerate(texts):
            raw[i] = t
            if not t:
                continue
            tl = t.lower()
            best, best_pos = None, len(tl) + 1
            for d in WEAKNESS_DOMAINS:
                for form in (d, d.replace("_", " ")):
                    p = tl.find(form)
                    if p != -1 and p < best_pos:
                        best, best_pos = d, p
            out[i] = best
    except Exception as e:
        print(f"[wm] domain classify failed (non-fatal): {e}", flush=True)
    matched = sum(1 for o in out if o)
    print(f"[wm] domain classifier: {matched}/{len(questions)} matched", flush=True)
    if logger:
        logger.event(kind="classify", n=len(questions), matched=matched,
                     sample_outputs=[(raw[i] or "")[:80] for i in range(min(5, len(raw)))])
    return out


def assign_targets(memory, seed_domains, rng, guided_prob=None):
    """Domain-compatible target routing (doc §5): a chain may only be guided by a
    memory item whose domain matches its seed's domain; anything else degrades to
    unguided WITH a recorded reason. Selection uses a dedicated RNG so enabling
    memory never perturbs the MH random stream."""
    guided_prob = config.MEMORY_GUIDED_PROB if guided_prob is None else guided_prob
    by_dom = {}
    for m in memory:
        by_dom.setdefault(m.get("domain"), []).append(m)
    targets, reasons = [], []
    for d in seed_domains:
        if not memory:
            targets.append(None); reasons.append("no_memory"); continue
        if d is None:
            targets.append(None); reasons.append("no_seed_domain"); continue
        if rng.random() >= guided_prob:
            targets.append(None); reasons.append("unguided_draw"); continue
        cands = by_dom.get(d, [])
        if not cands:
            targets.append(None); reasons.append("no_compatible_target"); continue
        targets.append(_weighted_pick(rng, cands)); reasons.append("guided")
    return targets, reasons


def select_guidance(mode, local_note, global_target):
    """One guidance per proposal (local-feedback doc §2). local_feedback: the
    current accepted state's own trusted note wins; missing note falls back to the
    chain's fixed global target; never both. global_fixed reproduces the previous
    behavior exactly."""
    if mode == "local_feedback":
        if local_note:
            return local_note, "local"
        if global_target:
            return global_target, "global_fallback"
        return None, "none"
    if global_target:
        return global_target, "global_fixed"
    return None, "none"


def _guidance_evidence(item):
    # global items carry representative_evidence, local notes carry evidence —
    # never silently drop the local one (doc §4)
    return str(item.get("representative_evidence") or item.get("evidence") or "")[:200]


def weakness_guidance_block(item):
    return WEAKNESS_GUIDANCE_TMPL.format(domain=item.get("domain", "?"),
                                         weakness=item["weakness"],
                                         evidence=_guidance_evidence(item))


# ---- global summary (doc §4 + R1/R3/R4/R5): budget-aware conserving map-reduce ----
_WM_SUMMARY_SYSTEM = """You merge duplicate descriptions of solver weaknesses.

Given a numbered list of weakness notes (all from the SAME math domain), group the
notes that describe the same specific reasoning capability. Return exactly one JSON
object with keys "groups" and "unassigned":
- groups: a list of objects {"weakness": <UNDER 15 words naming the shared
  capability — do NOT restate any note's evidence or copy problem constants>,
  "indices": [note numbers in the group],
  "representative": <the ONE note number whose evidence best supports the merged
  label>, "reason": <under 8 words>}
- unassigned: the note numbers you could not confidently group.

Keep the output SHORT: terse labels, no prose outside the JSON object.

Every input note number must appear exactly once, either in some group's indices or
in unassigned. The representative must be one of that group's indices. Only merge
notes describing the same specific capability; do not merge notes merely because
they share a broad word. Do not invent notes."""


def _valid_index(x):
    """R5: an id must be a genuine integer (or a pure digit string). bools,
    floats and anything else are rejected — never truncation-converted."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return None


def _merge_notes_llm(tokenizer, note_items, stats=None, log_call=None, _depth=0):
    """One conservation-checked LLM merge over same-domain note_items
    ({"weakness","evidence"}). Returns (groups, unassigned):
    groups = [{"weakness", "indices", "rep"}] with LOCAL indices, disjoint;
    groups+unassigned always partition range(n). `rep` is the model-selected
    representative member (validated to belong to the group); a group whose
    representative cannot be validated is NOT force-merged — its members go to
    unassigned (R3). Oversized inputs are bisected AFTER final-prompt budget
    measurement (R4). Falls back to exact-string grouping (same domain, no
    invented labels; rep = first member, method recorded by the caller)."""
    stats = stats if stats is not None else {}
    n = len(note_items)
    if n == 0:
        return [], []
    if n == 1:
        return [{"weakness": note_items[0]["weakness"], "indices": [0], "rep": 0}], []
    listing = "\n".join(
        f"{j}. {nt['weakness']} || evidence: {str(nt.get('evidence') or '')[:100]}"
        for j, nt in enumerate(note_items))
    user = f"WEAKNESS NOTES:\n{listing}\n\nReturn the JSON object now."
    prompt = apply_chat_template(tokenizer, _WM_SUMMARY_SYSTEM, user)
    # R4: measure the FINAL rendered prompt; bisect instead of sending over budget
    if _tok_len(tokenizer, prompt) + WM_SUMMARY_MAX_TOKENS > WM_CONTEXT_LIMIT:
        stats["split"] = stats.get("split", 0) + 1
        mid = n // 2
        g1, u1 = _merge_notes_llm(tokenizer, note_items[:mid], stats, log_call, _depth + 1)
        g2, u2 = _merge_notes_llm(tokenizer, note_items[mid:], stats, log_call, _depth + 1)
        for g in g2:
            g["indices"] = [j + mid for j in g["indices"]]
            g["rep"] += mid
        return g1 + g2, u1 + [j + mid for j in u2]
    raw_out = None
    for attempt in range(2):
        try:
            resp = base_client().completions.create(
                model=config.MODEL_NAME, prompt=[prompt],
                max_tokens=WM_SUMMARY_MAX_TOKENS, temperature=0.2, top_p=0.9)
            raw_out = resp.choices[0].text
        except Exception as e:
            stats["request_failure"] = stats.get("request_failure", 0) + 1
            if log_call:
                log_call(attempt=attempt + 1, n_input=n, ok=False,
                         outcome="request_failed", reason=str(e)[:200])
            print(f"[wm] summarizer call failed: {e}", flush=True)
            continue
        obj = _extract_json_value(raw_out)
        if not isinstance(obj, dict) or not isinstance(obj.get("groups"), list):
            stats["parse_failure"] = stats.get("parse_failure", 0) + 1
            if log_call:
                log_call(attempt=attempt + 1, n_input=n, ok=False,
                         outcome="parse_failed", raw_output=raw_out)
            continue
        groups, seen, repaired = [], set(), False
        for cl in obj["groups"]:
            if not isinstance(cl, dict):
                repaired = True
                continue
            raw_ix = cl.get("indices")
            if not isinstance(raw_ix, list):        # R5: scalar/None never TypeErrors
                repaired = True
                continue
            ixs = []
            for x in raw_ix:
                v = _valid_index(x)
                if v is None or not (0 <= v < n) or v in seen:
                    repaired = True                 # rejected/dup/out-of-range: dropped
                    continue
                ixs.append(v); seen.add(v)
            w = str(cl.get("weakness") or "").strip()[:300]
            if not ixs or not w:
                if ixs:
                    seen.difference_update(ixs)
                repaired = True
                continue
            rep = _valid_index(cl.get("representative"))
            if rep is None or rep not in ixs:
                # R3: no validated representative -> do NOT force the merge
                seen.difference_update(ixs)
                repaired = True
                continue
            groups.append({"weakness": w, "indices": ixs, "rep": rep})
        unassigned = [j for j in range(n) if j not in seen]   # conservation
        if groups:
            # 'repaired' only counts CODE repairs; model-declared unassigned is legal
            key = "llm_repaired" if repaired else "llm_success"
            stats[key] = stats.get(key, 0) + 1
            if log_call:
                log_call(attempt=attempt + 1, n_input=n, ok=True, outcome=key,
                         raw_output=raw_out,
                         groups=[{"weakness": g["weakness"], "indices": g["indices"],
                                  "rep": g["rep"]} for g in groups],
                         unassigned=unassigned)
            return groups, unassigned
        stats["parse_failure"] = stats.get("parse_failure", 0) + 1
    # fallback: exact-string grouping within this same-domain input
    stats["fallback"] = stats.get("fallback", 0) + 1
    if log_call:
        log_call(n_input=n, ok=False, outcome="fallback", raw_output=raw_out)
    exact = {}
    for j, nt in enumerate(note_items):
        exact.setdefault(_norm_ws(nt["weakness"]), []).append(j)
    groups = [{"weakness": note_items[ixs[0]]["weakness"], "indices": ixs, "rep": ixs[0]}
              for ixs in exact.values()]
    return groups, []


def _budget_chunks(tokenizer, items, render, max_items):
    """Split items into chunks whose rendered listing fits the summary budget
    (estimate; _merge_notes_llm re-measures the FINAL prompt and bisects — R4)."""
    budget = WM_CONTEXT_LIMIT - WM_SUMMARY_MAX_TOKENS - _tok_len(
        tokenizer, _WM_SUMMARY_SYSTEM) - 128
    chunks, cur, cur_tok = [], [], 0
    for it in items:
        t = _tok_len(tokenizer, render(it)) + 4
        if cur and (cur_tok + t > budget or len(cur) >= max_items):
            chunks.append(cur); cur, cur_tok = [], 0
        cur.append(it); cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks


def _exact_merge_groups(groups):
    """R1: deterministic GLOBAL same-name merge within a domain — sources union,
    representative taken from the largest constituent. Chunk boundaries and input
    order cannot split identical labels."""
    merged = {}
    for g in groups:
        key = _norm_ws(g["weakness"])
        if key in merged:
            m = merged[key]
            m["src"] = sorted(set(m["src"]) | set(g["src"]))
            if len(g["src"]) > m["_repsize"]:
                m["rep"], m["_repsize"] = g["rep"], len(g["src"])
                m["rep_method"] = g.get("rep_method", "exact_name")
                m["weakness"] = g["weakness"]
        else:
            merged[key] = dict(g, _repsize=len(g["src"]))
    out = []
    for m in merged.values():
        m.pop("_repsize", None)
        out.append(m)
    return out


def summarize_global_weakness_memory(tokenizer, records, iteration, logger=None):
    """Domain-bucketed, budget-aware, conservation-checked map-reduce over the
    POST-complete-filter final-chain notes.

    Order of operations (R1): global exact-name pre-merge per domain -> LLM map
    over budget chunks -> LLM reduce passes -> global exact-name post-merge.
    support = distinct (iter, chain); avg_p_hat recomputed in Python from members;
    every item carries source_event_ids and a representative whose membership is
    program-verified (semantic supportiveness still needs human sampling — the
    membership check is NOT a semantic guarantee). Full group list incl. trims in
    the audit file; per-merge-call logs go to the provided logger. Never raises."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    out_path = f"{_wm_dir()}/global_weakness_memory_iter_{iteration}.json"
    audit_path = f"{_wm_dir()}/global_weakness_memory_iter_{iteration}_audit.json"
    stats, call_seq = {}, [0]

    def _log_call(stage, dom, member_gis):
        if logger is None:
            return None
        def log_call(**kw):
            call_seq[0] += 1
            logger.event(kind="merge", call_id=f"merge_{call_seq[0]}", stage=stage,
                         domain=dom, member_gis=member_gis,
                         prompt_version=WM_SUMMARY_PROMPT_VERSION, **kw)
        return log_call

    members = []
    for d in records:
        nt = d.get("_weakness_note")
        if isinstance(nt, dict) and nt.get("weakness"):
            members.append({"chain": d.get("_chain_id"), "p_hat": float(d["p_hat"]),
                            "event_id": d.get("_wm_event_id"),
                            "domain": nt.get("domain", "other"),
                            "weakness": nt["weakness"],
                            "evidence": nt.get("evidence", "")})
    if not members:
        with open(out_path, "w") as f:
            json.dump([], f)
        print(f"[wm] iter {iteration}: no usable notes -> empty global memory", flush=True)
        return []

    by_dom = {}
    for gi, m in enumerate(members):
        by_dom.setdefault(m["domain"], []).append(gi)

    provisional = []
    for dom, gixs in by_dom.items():
        # R1 step 1: domain-global exact-name pre-merge (chunk-independent)
        base = {}
        for gi in gixs:
            base.setdefault(_norm_ws(members[gi]["weakness"]), []).append(gi)
        base_groups = [{"domain": dom, "weakness": members[g[0]]["weakness"],
                        "src": sorted(g), "rep": g[0], "rep_method": "exact_name"}
                       for g in base.values()]
        # step 2: LLM map over budget chunks of BASE GROUPS
        chunks = _budget_chunks(
            tokenizer, base_groups,
            lambda bg: f"0. {bg['weakness']} || evidence: "
                       f"{members[bg['rep']]['evidence'][:100]}",
            min(config.MEMORY_SUMMARY_CHUNK_SIZE, WM_MERGE_MAX_ITEMS))
        for chunk in chunks:
            items = [{"weakness": bg["weakness"],
                      "evidence": members[bg["rep"]]["evidence"]} for bg in chunk]
            groups, unassigned = _merge_notes_llm(
                tokenizer, items, stats,
                _log_call("map", dom, [bg["src"] for bg in chunk]))
            for g in groups:
                src = sorted({gi for j in g["indices"] for gi in chunk[j]["src"]})
                rep_bg = chunk[g["rep"]]
                provisional.append({"domain": dom, "weakness": g["weakness"],
                                    "src": src, "rep": rep_bg["rep"],
                                    "rep_method": "llm_selected"})
            for j in unassigned:
                provisional.append(chunk[j])
    # step 3: reduce passes (same-domain), representative evidence carried forward
    for _pass in range(2):
        by_dom_p = {}
        for p in provisional:
            by_dom_p.setdefault(p["domain"], []).append(p)
        nxt, changed = [], False
        for dom, plist in by_dom_p.items():
            if len(plist) <= 1:
                nxt.extend(plist); continue
            chunks = _budget_chunks(
                tokenizer, plist,
                lambda p: f"0. {p['weakness']} || evidence: "
                          f"{members[p['rep']]['evidence'][:100]}",
                min(config.MEMORY_SUMMARY_CHUNK_SIZE, WM_MERGE_MAX_ITEMS))
            for chunk in chunks:
                if len(chunk) == 1:
                    nxt.append(chunk[0]); continue
                items = [{"weakness": p["weakness"],
                          "evidence": members[p["rep"]]["evidence"]} for p in chunk]
                groups, unassigned = _merge_notes_llm(
                    tokenizer, items, stats,
                    _log_call(f"reduce{_pass + 1}", dom, [p["src"] for p in chunk]))
                if len(groups) + len(unassigned) < len(chunk):
                    changed = True
                for g in groups:
                    src = sorted({gi for j in g["indices"] for gi in chunk[j]["src"]})
                    nxt.append({"domain": dom, "weakness": g["weakness"], "src": src,
                                "rep": chunk[g["rep"]]["rep"],
                                "rep_method": "llm_selected"})
                for j in unassigned:
                    nxt.append(chunk[j])
        provisional = nxt
        if not changed:
            break
    # R1 step 4: domain-global exact-name POST-merge (cross-chunk duplicates meet)
    by_dom_p = {}
    for p in provisional:
        by_dom_p.setdefault(p["domain"], []).append(p)
    provisional = [g for plist in by_dom_p.values() for g in _exact_merge_groups(plist)]

    audit, items_out = [], []
    for cl in provisional:
        src = sorted(set(cl["src"]))
        support = len({(iteration, members[gi]["chain"]) for gi in src})
        avg_p = sum(members[gi]["p_hat"] for gi in src) / len(src)
        rep_gi = cl["rep"]
        assert rep_gi in src, "representative must belong to its group"
        entry = {"domain": cl["domain"], "weakness": cl["weakness"],
                 "support": support, "avg_p_hat": round(avg_p, 4),
                 "representative_evidence": members[rep_gi]["evidence"],
                 "representative_event_id": members[rep_gi]["event_id"],
                 "rep_method": cl.get("rep_method", "exact_name"),
                 "source_event_ids": [members[gi]["event_id"] for gi in src],
                 "source_chains": [members[gi]["chain"] for gi in src],
                 "source_iter": iteration,
                 "_score": support * (1.0 - abs(avg_p - 0.5))}
        audit.append(dict(entry))
        if support >= config.MEMORY_MIN_SUPPORT:
            items_out.append(entry)
    items_out.sort(key=lambda it: -it["_score"])
    kept, _cut = items_out[:config.MEMORY_TOP_K], items_out[config.MEMORY_TOP_K:]
    kept_keys = {tuple(k["source_event_ids"]) + (k["weakness"],) for k in kept}
    for a in audit:
        a["trim_reason"] = ("kept" if tuple(a["source_event_ids"]) + (a["weakness"],)
                            in kept_keys
                            else ("support<min" if a["support"] < config.MEMORY_MIN_SUPPORT
                                  else "below top-K"))
    for r, it_ in enumerate(kept, 1):
        it_["id"] = f"memory_{r}"
        del it_["_score"]
    with open(out_path, "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    with open(audit_path, "w") as f:
        json.dump({"stats": stats, "n_input_notes": len(members),
                   "groups": audit}, f, indent=2, ensure_ascii=False)
    print(f"[wm] iter {iteration}: {len(members)} notes -> {len(provisional)} groups "
          f"-> {len(kept)} kept (supports {[k['support'] for k in kept]}); "
          f"merge stats {stats} -> {out_path}", flush=True)
    return kept


def generate_batch_mcmc(tokenizer, num_questions, log_path, init_pool=None):
    """Returns: list of dicts {question, gt, p_hat, pseudo_label, r_unc}.

    init_pool: if given (list of prev-iter question dicts), WARM-START the MCMC chain from
    those questions (re-scored with the current solver) instead of sampling fresh from base.
    """
    print(f"\n[MCMC] Initializing pool of {num_questions} questions via base model...")
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("=" * 50 + "\nMCMC Phase\n" + "=" * 50 + "\n")

    pool_q, pool_gt, pool_runc, pool_phat, pool_pseudo = [], [], [], [], []
    pool_labels = []   # per-question list of n canonical answers (CD U-statistic)
    pool_topic = []    # SEED topic per slot: assigned at initial generation and inherited
                       # through mutations (context is topic(x_0), not re-classified topic(x_t))
    pool_details = []  # weakness memory: per-question answer-cluster details (None when off)

    # --- Weakness memory: active only on the canonical non-CD walk; the previous
    #     iteration's global memory is loaded once and FROZEN for this whole walk.
    wm_active = config.WEAKNESS_MEMORY_ENABLED and not config.CD_ENABLE
    if config.WEAKNESS_MEMORY_ENABLED and config.CD_ENABLE:
        print("[wm] WARNING: weakness memory not implemented for the CD walk; disabled", flush=True)
    if wm_active:
        if config.WM_GUIDANCE_MODE not in ("global_fixed", "local_feedback"):
            raise ValueError(f"[wm] unknown DEO_WM_GUIDANCE_MODE={config.WM_GUIDANCE_MODE!r}")
        if config.BANDIT_ENABLE:
            raise RuntimeError("[wm] weakness memory + bandit is an unsupported combination")
    wm_iter = None
    wm_memory = []
    if wm_active:
        m_it = re.search(r"mcmc_iter_(\d+)", os.path.basename(log_path or ""))
        wm_iter = int(m_it.group(1)) if m_it else None
        if wm_iter is None:
            print("[wm] WARNING: cannot parse iteration from log_path; disabled", flush=True)
            wm_active = False
        elif wm_iter > 1:
            wm_memory = load_global_weakness_memory(wm_iter - 1)
    gen_target = (config.INIT_POOL
                  if (init_pool is None and config.INIT_POOL > num_questions)
                  else num_questions)
    if gen_target > num_questions:
        print(f"[MCMC] big-init: generating {gen_target}, walking a {num_questions}-seed subset")
    pbar = tqdm(total=gen_target, desc="Init")

    forbidden = ["prove that", "show that", "justify", "explain", "true or false", "yes or no"]

    if init_pool is not None:
        # WARM START: reuse prev-iter (mutated) questions as X_0, re-scored with current solver.
        qs = [d["question"] for d in init_pool]
        gts = [d.get("gt", "") for d in init_pool]
        print(f"[MCMC] WARM START from {len(qs)} prev-iter questions (re-scoring with current solver)...")
        CH = 256
        tps = [d.get("topic", "unknown") for d in init_pool]
        for s in range(0, len(qs), CH):
            ch_q, ch_g, ch_t = qs[s:s + CH], gts[s:s + CH], tps[s:s + CH]
            if wm_active:
                r_uncs, p_hats, pseudos, labs, dets = evaluate_r_unc_vllm(
                    tokenizer, ch_q, return_labels=True, return_details=True)
            else:
                r_uncs, p_hats, pseudos, labs = evaluate_r_unc_vllm(tokenizer, ch_q, return_labels=True)
                dets = [None] * len(ch_q)
            for q, gt, tp, ru, ph, ps, lb, de in zip(ch_q, ch_g, ch_t, r_uncs, p_hats, pseudos, labs, dets):
                pool_q.append(q); pool_gt.append(gt); pool_runc.append(ru)
                pool_phat.append(ph); pool_pseudo.append(ps); pool_labels.append(lb)
                pool_topic.append(tp); pool_details.append(de); pbar.update(1)
        num_questions = len(pool_q)

    while init_pool is None and len(pool_q) < gen_target:
        needed = gen_target - len(pool_q)
        bs = min(config.INIT_BATCH_SIZE, needed)
        c_pairs, c_topics = [], []
        for _ in range(bs):
            topic = random.choice(MATH_TOPICS)
            user_p = (
                "Generate one new, challenging reasoning question now. "
                f"YOU MUST STRICTLY FOCUS ON: **{topic}**."
            )
            c_pairs.append((CHALLENGER_SYSTEM_PROMPT, user_p))
            c_topics.append(topic)

        init_texts = gen_texts_pairs(tokenizer, c_pairs, max_tokens=1536,
                                     temperature=1.0, top_p=0.95)
        valid_qs, valid_gts, valid_tps = [], [], []
        for j, t in enumerate(init_texts):
            if t is None:
                continue          # missing choice: skip THIS slot; topics stay aligned
            q, gt = extract_challenger_output(t)
            if (q and len(q) > 30 and not any(w in q.lower() for w in forbidden)
                    and not (config.STRIP_LEAKS and question_is_leaky(q))):
                valid_qs.append(q)
                valid_gts.append(gt)
                valid_tps.append(c_topics[j])
        if not valid_qs:
            continue

        if wm_active:
            r_uncs, p_hats, pseudos, labs, dets = evaluate_r_unc_vllm(
                tokenizer, valid_qs, return_labels=True, return_details=True)
        else:
            r_uncs, p_hats, pseudos, labs = evaluate_r_unc_vllm(tokenizer, valid_qs, return_labels=True)
            dets = [None] * len(valid_qs)
        for q, gt, tp, ru, ph, ps, lb, de in zip(valid_qs, valid_gts, valid_tps, r_uncs, p_hats, pseudos, labs, dets):
            if len(pool_q) >= gen_target:
                break
            pool_q.append(q)
            pool_gt.append(gt)
            pool_runc.append(ru)
            pool_phat.append(ph)
            pool_pseudo.append(ps)
            pool_labels.append(lb)
            pool_topic.append(tp)
            pool_details.append(de)
            pbar.update(1)
    pbar.close()

    if len(pool_q) > num_questions:
        # Select the walk subset: random among in-band seeds (selection headroom),
        # topped up with the out-of-band questions closest to p_hat=0.5 if short.
        in_band = [i for i in range(len(pool_q))
                   if config.MIN_SCORE <= float(pool_phat[i]) <= config.MAX_SCORE]
        in_band_set = set(in_band)
        random.shuffle(in_band)
        sel = in_band[:num_questions]
        if len(sel) < num_questions:
            rest = sorted((i for i in range(len(pool_q)) if i not in in_band_set),
                          key=lambda i: abs(float(pool_phat[i]) - 0.5))
            sel += rest[:num_questions - len(sel)]
        sel.sort()
        n_in = sum(1 for i in sel if i in in_band_set)
        print(f"[MCMC] big-init selection: {len(in_band_set)}/{len(pool_q)} in-band -> "
              f"walking {len(sel)} seeds ({n_in} in-band)")
        log_file.write(f"[Init] big-init: {len(pool_q)} generated, {len(in_band)} in-band, "
                       f"{len(sel)} selected for walk\n")
        pool_q      = [pool_q[i] for i in sel]
        pool_gt     = [pool_gt[i] for i in sel]
        pool_runc   = [pool_runc[i] for i in sel]
        pool_phat   = [pool_phat[i] for i in sel]
        pool_pseudo = [pool_pseudo[i] for i in sel]
        pool_labels = [pool_labels[i] for i in sel]
        pool_topic  = [pool_topic[i] for i in sel]
        pool_details = [pool_details[i] for i in sel]
        num_questions = len(pool_q)

    # Build initial state (one-shot O(N^2) BLEU sweep, ~30s at N=1000)
    print("[MCMC] Computing initial neighbor matrix (one-shot O(N^2))...")
    energy, neighbor, cluster_size = calculate_batch_energy(pool_q, pool_runc)
    pool_tokens = [q.split() for q in pool_q]
    r_unc_arr = np.asarray(pool_runc, dtype=float)
    n = num_questions
    log_file.write(f"[Init] V(X_0) = {energy:.4f}\n")

    # --- CD: per-question U-statistic utility u_i (energy becomes sum_i u_i) ---
    pool_ustat = None
    if config.CD_ENABLE:
        print(f"[MCMC][CD] precomputing U-statistic utilities (m={config.USTAT_M}, n={config.USTAT_N})...")
        pool_ustat = [ustat_jackknife(pool_labels[i], float(cluster_size[i]) / n)[0]
                      for i in range(num_questions)]
        energy = float(np.sum(pool_ustat))
        log_file.write(f"[Init][CD] U(X_0) = {energy:.4f}\n")

    # --- Weakness memory: one FIXED target and one current note per chain. Initial
    #     pool states get notes too (a chain that rejects all 5 proposals contributes
    #     its seed's note). Targets stay fixed across all MCMC_STEPS; routing is
    #     seed-domain-compatible (doc §5) with a dedicated RNG so the MH random
    #     stream is untouched.
    pool_target = [None] * num_questions
    pool_note = [None] * num_questions
    pool_event = [None] * num_questions
    pool_seed_domain = [None] * num_questions
    pool_guide_reason = ["memory_off"] * num_questions
    pool_n_accepted = [0] * num_questions
    wm_logger = None
    wm_src_prop, wm_src_acc = Counter(), Counter()
    wm_status_counts = Counter()
    if wm_active:
        wm_logger = WmLogger(wm_iter)
        pool_note, init_statuses, pool_event = generate_weakness_notes_batch(
            tokenizer, pool_q, pool_phat, pool_pseudo, pool_details,
            logger=wm_logger, stage="init")
        wm_status_counts.update(init_statuses)
        for d in pool_details:
            if d:
                d.pop("_raw", None)
        for i, nt in enumerate(pool_note):
            wm_logger.note_row(event_id=pool_event[i], stage="init", chain=i,
                               accepted=True, target_id=None, target_source_iter=None,
                               note_status=init_statuses[i],
                               **(nt or {"domain": None, "weakness": None,
                                         "evidence": None}))
        # coarse seed domain: trusted init note first, light batch classify otherwise
        pool_seed_domain = [nt["domain"] if nt else None for nt in pool_note]
        need = [i for i in range(num_questions) if pool_seed_domain[i] is None]
        if need and wm_memory:
            doms = classify_domains_batch(tokenizer, [pool_q[i] for i in need],
                                          logger=wm_logger)
            for i, dm in zip(need, doms):
                pool_seed_domain[i] = dm
        wm_rng = random.Random(f"{config.MODEL_ABBR}_wm_iter{wm_iter}")
        pool_target, pool_guide_reason = assign_targets(
            wm_memory, pool_seed_domain, wm_rng)
        reason_counts = Counter(pool_guide_reason)
        msg = (f"[wm] ON: iter={wm_iter}, loaded {len(wm_memory)} global items "
               f"(source_iter={wm_memory[0].get('source_iter') if wm_memory else None}); "
               f"init statuses {dict(wm_status_counts)}; routing {dict(reason_counts)}")
        print(msg, flush=True)
        log_file.write(msg + "\n")

    # --- Bandit: contextual Thompson-sampling memory over mutation strategies.
    #     Loaded from persisted state (cross-iteration memory), FROZEN during this
    #     walk; discounted update + save happen after the walk completes (Eq 17).
    bandit, bandit_path, bandit_run_id = None, None, None
    bd_mismatch = bd_invalid = 0
    if config.BANDIT_ENABLE:
        # per-run state file: independent runs must NOT share learned memory
        bandit_run_id = os.getenv("DEO_BANDIT_RUN_ID", config.MODEL_ABBR)
        bandit_path = f"{config.STORAGE_ROOT}/datasets/bandit_state_{bandit_run_id}.json"
        bandit = MutationBandit.load(bandit_path, pmin=config.MIN_SCORE,
                                     pmax=config.MAX_SCORE, rho=config.BANDIT_RHO)
        print(f"[MCMC][bandit] TS mutation memory ON (run={bandit_run_id}, rho={config.BANDIT_RHO}, "
              f"eps={config.BANDIT_EPS}, {len(bandit.state)} learned contexts)", flush=True)
        log_file.write(f"[Init][bandit] run={bandit_run_id} rho={config.BANDIT_RHO} "
                       f"eps={config.BANDIT_EPS} contexts={len(bandit.state)}\n")

    # MCMC mutation walk — every accept/reject is O(N) thanks to incremental update
    for step in range(config.MCMC_STEPS):
        idx_perm = list(range(num_questions))
        random.shuffle(idx_perm)
        pbar_step = tqdm(total=num_questions, desc=f"MCMC step {step+1}/{config.MCMC_STEPS}")
        for i in range(0, num_questions, config.MUTATE_BATCH_SIZE):
            batch_idx = idx_perm[i: i + config.MUTATE_BATCH_SIZE]
            chosen = {}   # k -> (action, context_key); bandit-selected strategy per slot
            style_ks = set()  # slots mutated via the [F] olympiad-rewrite operator
            gsnap = {}    # k -> guidance snapshot FROZEN at prompt-build time (doc §6)
            if bandit is not None:
                m_pairs = []
                for k in batch_idx:
                    a, ctx = bandit.select(pool_topic[k], pool_phat[k])
                    chosen[k] = (a, ctx)
                    # single-operator prompt: only the selected operator is described,
                    # so the model has no menu to deviate to.
                    m_pairs.append((mutator_system_prompt_bandit(a),
                                    MUTATOR_USER_TEMPLATE_FORCED.format(
                                        seed=pool_q[k], action=a,
                                        action_name=ACTION_NAMES[a])))
            else:
                m_pairs = []
                for k in batch_idx:
                    if config.STYLE_P > 0 and random.random() < config.STYLE_P:
                        style_ks.add(k)
                        if wm_active:
                            gsnap[k] = {"guidance_source": "style", "old_q": pool_q[k]}
                        m_pairs.append((MUTATOR_SYSTEM_PROMPT_STYLE,
                                        MUTATOR_USER_TEMPLATE_STYLE.format(seed=pool_q[k])))
                    else:
                        user_msg = MUTATOR_USER_TEMPLATE.format(seed=pool_q[k])
                        guide, gsrc = select_guidance(
                            config.WM_GUIDANCE_MODE,
                            pool_note[k] if wm_active else None,
                            pool_target[k])
                        if guide is not None:
                            # guided mutation: append ONLY the selected guidance
                            user_msg += weakness_guidance_block(guide)
                        if wm_active:
                            gsnap[k] = {
                                "guidance_source": gsrc,
                                "guidance_domain": guide.get("domain") if guide else None,
                                "guidance_weakness": guide.get("weakness") if guide else None,
                                "guidance_evidence": _guidance_evidence(guide) if guide else None,
                                "guidance_ref": (pool_event[k] if gsrc == "local" else
                                                 (f"{guide.get('id')}@iter{guide.get('source_iter')}"
                                                  if guide else None)),
                                "old_q": pool_q[k],
                            }
                        m_pairs.append((MUTATOR_SYSTEM_PROMPT, user_msg))
            proposals = []
            mut_texts = gen_texts_pairs(tokenizer, m_pairs, max_tokens=1536,
                                        temperature=1.1)
            for j, k in enumerate(batch_idx):
                t = mut_texts[j]
                if t is None:
                    # R6: missing mutation response = malformed proposal for THIS
                    # slot only; later slots keep their own responses
                    if bandit is not None:
                        a, ctx = chosen[k]
                        bandit.record(ctx, a, 0)
                    continue
                qp, gtp = extract_challenger_output(t)
                actual = extract_mutation_strategy(t) or "?"
                if bandit is not None:
                    a, ctx = chosen[k]
                    # AUDIT ONLY: with a single-operator prompt the tag is expected to echo
                    # the assigned letter; a different tag is counted but NOT a rejection —
                    # the proposal was generated under the selected operator's instructions,
                    # so the outcome is credited to the chosen action regardless.
                    if actual != a:
                        bd_mismatch += 1
                if (not (qp and len(qp) > 30 and qp != pool_q[k])
                        or (config.STRIP_LEAKS and question_is_leaky(qp))):
                    if bandit is not None:
                        a, ctx = chosen[k]
                        bandit.record(ctx, a, 0)   # malformed/unchanged/leaky: failure (Eq 16, Valid=0)
                    continue
                proposals.append({"k": k, "q": qp, "gt": gtp,
                                  "strat": ("F" if k in style_ks else
                                            (chosen[k][0] if k in chosen else actual)),
                                  "old": pool_q[k]})

            # pre-MH surface-validity gate (bandit path): a proposal the LLM judge flags
            # as INVALID is a bandit failure and never gets solver-scored or enters the
            # chain. (The judge is a cheap local-vLLM format/surface filter that defaults
            # to VALID — it does not certify mathematical correctness.)
            if bandit is not None and proposals:
                with ThreadPoolExecutor(max_workers=8) as ex:
                    verdicts = list(ex.map(lambda p: judge_one_validity(p["q"], p["gt"]),
                                           proposals))
                kept = []
                for p, ok in zip(proposals, verdicts):
                    if ok:
                        kept.append(p)
                    else:
                        a, ctx = chosen[p["k"]]
                        bandit.record(ctx, a, 0)
                        bd_invalid += 1
                        log_file.write(f"[bandit] surface-INVALID k={p['k']} (act={a}) "
                                       f"-> dropped pre-MH\n")
                proposals = kept
            if proposals:
                qs_new = [p["q"] for p in proposals]
                labs_new, labs_old_map, notes_new = None, {}, None
                if config.CD_ENABLE:
                    rus, phs, pls, labs_new = evaluate_r_unc_vllm(tokenizer, qs_new, return_labels=True)
                    if config.CD_FRESH_OLD:   # re-score current state fresh (else reuse cached labels)
                        _r, _p, _l, labs_old = evaluate_r_unc_vllm(
                            tokenizer, [p["old"] for p in proposals], return_labels=True)
                        labs_old_map = {p["k"]: labs_old[jj] for jj, p in enumerate(proposals)}
                elif wm_active:
                    prop_ids = [wm_logger.new_event_id("prop") for _ in proposals]
                    rus, phs, pls, dets_new = evaluate_r_unc_vllm(
                        tokenizer, qs_new, return_details=True)
                    # every proposal is diagnosed (accepted or not); the chain's note is
                    # replaced only on acceptance
                    notes_new, statuses_new, eids_new = generate_weakness_notes_batch(
                        tokenizer, qs_new, phs, pls, dets_new, logger=wm_logger,
                        stage=f"step{step + 1}", chains=[p["k"] for p in proposals])
                    wm_status_counts.update(statuses_new)
                    for d in dets_new:
                        if d:
                            d.pop("_raw", None)
                else:
                    rus, phs, pls = evaluate_r_unc_vllm(tokenizer, qs_new)
                for j, p in enumerate(proposals):
                    k = p["k"]
                    q_prime_tokens = p["q"].split()

                    # ---- O(N) incremental cluster update (BLEU neighbor row for k) ----
                    new_row = neighbor_row_for(q_prime_tokens, pool_tokens, self_idx=k)
                    old_row = neighbor[k]
                    delta = new_row.astype(int) - old_row.astype(int)
                    delta[k] = 0
                    new_cluster = cluster_size + delta
                    new_cluster[k] = float(new_row.sum())

                    # --- bandit feedback (Eq 16): success = Valid * in-band * Novel.
                    #     Judged on the PROPOSAL, independent of the MH accept event.
                    #     (Surface validity was already gated pre-scoring; pseudo-label
                    #     existence is the residual cheap Valid component here.)
                    #     For an already-Trainable state, additionally require the
                    #     uncertainty utility not to degrade by more than eps, so the
                    #     bandit can't score "success" for drifting a good state to the
                    #     band edge.
                    bd_note = ""
                    if bandit is not None:
                        a_b, ctx_b = chosen[k]
                        s_ok = (pls[j] not in (None, "", "None")
                                and config.MIN_SCORE <= float(phs[j]) <= config.MAX_SCORE
                                and int(new_cluster[k]) <= 1)
                        if s_ok and bandit.bucket(pool_phat[k]) == "Trainable":
                            s_ok = float(rus[j]) >= float(r_unc_arr[k]) - config.BANDIT_EPS
                        bandit.record(ctx_b, a_b, s_ok)
                        bd_note = f"Bandit: ctx={ctx_b} act={a_b} succ={int(bool(s_ok))}  "

                    sigma2D = 0.0
                    nbr_u_new = {}
                    if config.CD_ENABLE:
                        # Ceperley-Dewing U-statistic penalty acceptance (coordinate-wise):
                        # only question k's rollouts carry noise; changed-neighbor terms are
                        # deterministic (cached labels, only r_rep moves) so add to D~ not sigma2D.
                        old_labels_k = labs_old_map.get(k, pool_labels[k])
                        u_k_old, v_k_old = ustat_jackknife(old_labels_k, float(cluster_size[k]) / n)
                        u_k_new, v_k_new = ustat_jackknife(labs_new[j], float(new_cluster[k]) / n)
                        Dtilde = u_k_new - u_k_old
                        for jj in np.nonzero(delta)[0]:
                            jj = int(jj)
                            uu, _ = ustat_jackknife(pool_labels[jj], float(new_cluster[jj]) / n)
                            nbr_u_new[jj] = uu
                            Dtilde += uu - pool_ustat[jj]
                        sigma2D = v_k_old + v_k_new
                        a = Dtilde / config.BETA - sigma2D / (2.0 * config.BETA * config.BETA)
                        alpha = float(np.exp(min(0.0, a)))
                        new_e = energy + Dtilde
                    else:
                        new_runc = r_unc_arr.copy()
                        new_runc[k] = rus[j]
                        new_e = energy_from_state(new_runc, new_cluster, n)
                        alpha = min(1.0, np.exp((new_e - energy) / config.BETA))
                    accept = random.random() < alpha

                    if wm_active:
                        tgt = pool_target[k]
                        snap = gsnap.get(k, {"guidance_source": "none", "old_q": p["old"]})
                        src = snap.get("guidance_source", "none")
                        wm_src_prop[src] += 1
                        wm_src_acc[src] += int(accept)
                        nt = notes_new[j] if notes_new else None
                        wm_logger.note_row(
                            event_id=eids_new[j], proposal_id=prop_ids[j],
                            stage=f"step{step + 1}", chain=k,
                            accepted=bool(accept),
                            guidance_mode=config.WM_GUIDANCE_MODE,
                            **snap,
                            new_q=p["q"],
                            p_hat_new=float(phs[j]),
                            target_id=tgt.get("id") if tgt else None,
                            target_source_iter=tgt.get("source_iter") if tgt else None,
                            guide_reason=pool_guide_reason[k],
                            note_status=statuses_new[j],
                            **(nt or {"domain": None, "weakness": None, "evidence": None}))

                    r_rep_k_old = float(cluster_size[k]) / n
                    r_rep_k_new = float(new_cluster[k]) / n
                    cd_note = f"sigma2D={sigma2D:.4f}  " if config.CD_ENABLE else ""
                    log_file.write(
                        f"\n[Step {step+1} | Batch {i // config.MUTATE_BATCH_SIZE}] "
                        f"Result: {'ACCEPTED' if accept else 'REJECTED'} | Strategy: {p['strat']} | "
                        f"{bd_note}k={k}\n"
                        f"--- [OLD QUESTION] ---\n{p['old']}\n"
                        f"--- [NEW QUESTION] ---\n{p['q']}\n"
                        f"r_unc:        {float(r_unc_arr[k]):.4f} -> {rus[j]:.4f}\n"
                        f"r_rep[k]:     {r_rep_k_old:.4f} -> {r_rep_k_new:.4f}  "
                        f"(cluster_size {int(cluster_size[k])} -> {int(new_cluster[k])})\n"
                        f"{cd_note}energy total: {energy:.4f} -> {new_e:.4f}  (dE={new_e - energy:+.4f})\n"
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
                        if wm_active:
                            # rejected proposals keep the old state AND its old note;
                            # an accepted proposal without a trusted note clears the
                            # note but keeps its own evaluation event id
                            pool_note[k] = notes_new[j] if notes_new else None
                            pool_event[k] = eids_new[j]
                            pool_n_accepted[k] += 1
                        neighbor[k, :] = new_row
                        neighbor[:, k] = new_row
                        cluster_size = new_cluster
                        if config.CD_ENABLE:
                            pool_labels[k] = labs_new[j]
                            pool_ustat[k] = u_k_new
                            for jj, uu in nbr_u_new.items():
                                pool_ustat[jj] = uu
                            r_unc_arr[k] = rus[j]
                        else:
                            r_unc_arr = new_runc
                        energy = new_e
            pbar_step.update(len(batch_idx))
            pbar_step.set_postfix({"V": f"{energy:.3f}"})
        pbar_step.close()

    # --- bandit: discounted end-of-iteration update (Eq 17) + persist + iter snapshot ---
    if bandit is not None:
        n_prop, n_succ = bandit.end_iteration()
        bandit.save(bandit_path)
        m_it = re.search(r"iter_(\d+)", os.path.basename(log_path or ""))
        if m_it:  # per-iteration snapshot for posterior-evolution analysis
            bandit.save(f"{config.STORAGE_ROOT}/datasets/"
                        f"bandit_state_{bandit_run_id}_iter{m_it.group(1)}.json")
        means = bandit.action_means()
        msg = (f"[bandit] update: {n_succ}/{n_prop} proposal-successes this iter "
               f"(tag-mismatch-audit={bd_mismatch}, surface-invalid={bd_invalid}) | "
               "action means: " + ", ".join(f"{a}={m:.2f}" for a, m in sorted(means.items())))
        print(msg, flush=True)
        log_file.write(msg + "\n")
        for line in bandit.summary():
            log_file.write(f"[bandit] {line}\n")

    if wm_active:
        by_src = ", ".join(f"{sr}={wm_src_acc[sr]}/{wm_src_prop[sr]}"
                           for sr in sorted(wm_src_prop))
        msg = (f"[wm] mode={config.WM_GUIDANCE_MODE}; proposal acceptance by "
               f"guidance_source: {by_src}; "
               f"final notes: {sum(1 for nt in pool_note if nt is not None)}/{num_questions}; "
               f"writer statuses (all stages): {dict(wm_status_counts)}")
        print(msg, flush=True)
        log_file.write(msg + "\n")
        wm_logger.close()

    log_file.close()
    records = [
        {"question": pool_q[i], "gt": pool_gt[i], "p_hat": pool_phat[i],
         "pseudo_label": pool_pseudo[i], "r_unc": pool_runc[i],
         "topic": pool_topic[i] if i < len(pool_topic) else "unknown"}
        for i in range(num_questions)
    ]
    if wm_active:
        # private metadata for memory building/audit; underscore-prefixed fields are
        # never uploaded (filter_and_push maps only problem/answer/score to HF)
        for i, d in enumerate(records):
            d["_chain_id"] = i
            d["_target_memory_id"] = pool_target[i].get("id") if pool_target[i] else None
            d["_target_source_iter"] = (pool_target[i].get("source_iter")
                                        if pool_target[i] else None)
            d["_seed_domain"] = pool_seed_domain[i]
            d["_guidance_reason"] = pool_guide_reason[i]
            d["_weakness_note"] = pool_note[i]
            d["_wm_event_id"] = pool_event[i]
            d["_n_accepted"] = pool_n_accepted[i]
            d["_guidance_mode"] = config.WM_GUIDANCE_MODE
    return records


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
    if config.STRIP_LEAKS:  # defense-in-depth: also scrub anything already in the pool
        n_before = len(stage1)
        stage1 = [d for d in stage1 if not question_is_leaky(d["question"])]
        print(f"[filter] leak-strip dropped {n_before - len(stage1)}/{n_before}")
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

    # Weakness memory: build this iteration's global memory from the records that
    # passed the COMPLETE filter (regex + pseudo + p_hat band + judge), so invalid
    # questions never become "solver weaknesses". Non-fatal on any failure.
    if config.WEAKNESS_MEMORY_ENABLED:
        m_it = re.search(r"_v(\d+)$", repo_name)
        if m_it:
            try:
                _lg = WmLogger(int(m_it.group(1)))   # fresh attempt for the summary phase
                try:
                    summarize_global_weakness_memory(None, stage2, int(m_it.group(1)),
                                                     logger=_lg)
                finally:
                    _lg.close()
            except Exception as e:
                print(f"[wm] global-memory summarization failed (non-fatal): {e}", flush=True)
        else:
            print(f"[wm] WARNING: cannot parse iteration from repo_name={repo_name}; "
                  "skipping global-memory build", flush=True)

    filtered = [
        {"problem": d["question"], "answer": d["pseudo_label"], "score": d["p_hat"]}
        for d in stage2
    ]

    # Easy-band ballast (QUESTION_ANALYSIS_8B.md rec #2): reshape the uploaded set so
    # ~EASY_BALLAST of it sits at p_hat in [0.6, MAX_SCORE], where majority-vote labels
    # are most often correct. verl samples its 1280 prompts uniformly from this set, so
    # the dataset share IS the expected consumed share. Rest is subsampled (never below
    # verl's ~512-row floor).
    if config.EASY_BALLAST > 0 and filtered:
        easy = [d for d in filtered if float(d["score"]) >= 0.6]
        rest = [d for d in filtered if float(d["score"]) < 0.6]
        share = len(easy) / len(filtered)
        if not easy:
            print("[ballast] WARNING: no easy-band questions available; skipping reshape")
        elif share >= config.EASY_BALLAST:
            print(f"[ballast] easy share already {share:.2f} >= target {config.EASY_BALLAST}; no reshape")
        else:
            max_rest = int(len(easy) * (1 - config.EASY_BALLAST) / config.EASY_BALLAST)
            keep_rest = max(max_rest, 512 - len(easy))
            if keep_rest < len(rest):
                random.seed(0)
                rest = random.sample(rest, keep_rest)
            filtered = easy + rest
            random.shuffle(filtered)
            print(f"[ballast] reshaped: easy={len(easy)} rest={len(rest)} "
                  f"-> share={len(easy)/len(filtered):.2f} (target {config.EASY_BALLAST}), "
                  f"n={len(filtered)}")

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
        # KL reference: DEO_KL_REF=base -> pin to original base every iter (no drift, DEO default);
        #               DEO_KL_REF=prev -> anchor to this iter's init solver (= prev-iter solver, R-Zero-style, drifts).
        f"worker.ref.model.model_path={solver_ckpt if os.getenv('DEO_KL_REF','base')=='prev' else config.MODEL_NAME}",
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


def wait_for_vllm_ready(url, label="vllm", timeout=int(os.getenv("VLLM_READY_TIMEOUT", "600"))):
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

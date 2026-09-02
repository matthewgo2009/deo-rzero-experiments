"""Categorical Planning SGLD (DEO_SGLD.pdf v2, §2) — interpretable plan variable.

State: per-particle planning logits A ∈ R^{K×L} (K semantic axes, L values, masked
padding). Hard one-hot plans Z ~ row-wise Categorical(softmax(A/γ)) (Eq 12); a
deterministic renderer c(Z) turns Z into a natural-language instruction; the frozen
base model generates questions from ordinary TEXT prompts (Eq 14) — no embeddings,
no backprop through the LM.

Gradient (Eq 18-19), fully analytic in the reward term:
    ĝ = ((Û(Z) − b)/γ) (Z − P(A;γ))  −  τ_plan ∇_A KL(P(A;γ)‖ρ0)  −  λ_A A
SGLD with annealing + row centering (Eq 20):
    A ← A + η ĝ + √(2ηT) ε ;  A_k ← A_k − mean(A_k)
Quantization (Eq 21): row-wise argmax → hard plan for the training-set generation.
M particles (Eq 22) with independent EMA baselines guard against mode collapse.

Dynamics note: KL(P‖ρ0) is bounded by log L, so its restoring force vanishes once the
softmax saturates; far from the origin only λ_A·A pulls back. At high Langevin T with
tiny λ_A the chain therefore wanders among saturated (near-one-hot) plans with
relaxation time ~1/(η·λ_A). In production this is acceptable — the reward term
dominates the drift and saturation toward high-reward plans is the desired endpoint —
but λ_A is the knob to raise if exploration dies too early.

Temperature scale note: the stationary density is ∝ exp(J/T) and U lives in [0,1]
with per-axis utility differences of only ~0.1-0.2, so T must sit WELL BELOW that
scale for the sampler to prefer good plans at all (T0=0.2 → mild exploration;
T1=0.02 → ~e^10 preference per 0.2 utility gap). T0=1 would be pure diffusion.

Reward note: Û uses the project-standard TENT uncertainty surrogate (documented
deviation from TeX Eq 3; see sgld_soft_prefix.py for the rationale).
"""
import json
import math
import os

import torch
import torch.nn.functional as F

# ---- planning axes (Eq 10). L = max row length; shorter rows are masked. ----
AXES = [
    ("topic", ["algebra", "geometry", "number theory", "combinatorics"]),
    ("length", ["80-150", "150-250", "250-400", "400-600"]),   # target statement tokens
    ("depth", ["short", "medium", "long", "very long"]),
    ("composition", ["a single concept", "two interacting concepts", "multiple concepts"]),
    ("format", ["a direct calculation", "a counting problem", "an optimization problem",
                "a find-all-solutions problem"]),
]
K = len(AXES)
L = max(len(v) for _, v in AXES)
LENGTH_BINS = {0: (80, 150), 1: (150, 250), 2: (250, 400), 3: (400, 600)}
LENGTH_AXIS = 1


def render_plan(Z_rows):
    """Deterministic renderer c(Z): row value indices -> instruction text."""
    topic = AXES[0][1][Z_rows[0]]
    length = AXES[1][1][Z_rows[1]]
    depth = AXES[2][1][Z_rows[2]]
    comp = AXES[3][1][Z_rows[3]]
    fmt = AXES[4][1][Z_rows[4]]
    return (f"Generate one new, challenging {topic} question now. "
            f"Its problem statement should be about {length} tokens long, it should require "
            f"{depth} reasoning involving {comp}, and it should be {fmt} "
            f"with a single unambiguous final answer.")


def length_ok(n_tokens, rows, tol=0.10):
    """Strict per-bin adherence (review P0-2): the accepted interval is the bin
    itself with ±tol slack — bins must NOT overlap into each other's cores."""
    lo, hi = LENGTH_BINS[rows[LENGTH_AXIS]]
    return (1.0 - tol) * lo <= n_tokens <= (1.0 + tol) * hi


def gated_utility(r_unc, rep, pseudo, p_hat, judge_ok, len_ok,
                  lambda_rep, pmin, pmax):
    """Review P0-3: TENT utility gated by usability so the planner cannot reward
    ambiguous/unsolvable/mislength questions (the R-Zero reward-hack path):
      u = 1{pseudo}·1{p_hat in band}·1{judge}·1{length} · [r_unc − λ_rep·rep]_+
    The band gate ALIGNS the planner objective with the training filter
    (BUCKET_ANALYSIS lesson: controller band must match filter band)."""
    if pseudo in (None, "", "None"):
        return 0.0
    if not (pmin <= float(p_hat) <= pmax):
        return 0.0
    if not judge_ok or not len_ok:
        return 0.0
    return max(0.0, float(r_unc) - lambda_rep * float(rep))


def coordinate_refine(rows, eval_fn):
    """One coordinate-ascent sweep over the plan (post-quantization): for each axis,
    evaluate every valid value with a fresh probe batch and keep the argmax.
    Deterministically repairs the single-stuck-axis failure mode that annealed
    SGLD exhibits at short production horizons (K·L probe batches, run once/iter).
    eval_fn(list_of_plans) -> list of utilities."""
    from plan_sgld import AXES as _AXES
    best = list(rows)
    for k in range(len(_AXES)):
        variants = []
        for l in range(len(_AXES[k][1])):
            v = list(best)
            v[k] = l
            variants.append(v)
        utils = eval_fn(variants)
        best[k] = int(max(range(len(utils)), key=lambda i: utils[i]))
    return best


def allocate_quota(plans, utils, n, tau_mix=0.05):
    """Review P1-4: dedup quantized plans, weight by validated utility.
    Returns list of (rows, quota) with sum quota == n; zero-utility plans get 0."""
    uniq = {}
    for rows, u in zip(plans, utils):
        key = tuple(rows)
        uniq[key] = max(uniq.get(key, 0.0), float(u))
    keys = [k for k in uniq if uniq[k] > 0.0]
    if not keys:                      # everything failed validation: fall back uniform
        keys = list(uniq.keys())
        w = [1.0 / len(keys)] * len(keys)
    else:
        import math as _m
        mx = max(uniq[k] for k in keys)
        e = [_m.exp((uniq[k] - mx) / max(tau_mix, 1e-6)) for k in keys]
        Zs = sum(e)
        w = [x / Zs for x in e]
    quota = [int(n * x) for x in w]
    i = 0
    while sum(quota) < n:
        quota[i % len(quota)] += 1
        i += 1
    return [(list(k), q) for k, q in zip(keys, quota)]


class PlanSGLD:
    def __init__(self, M=8, eta=0.3, T0=0.2, T1=0.02, gamma0=1.0, gamma1=0.3,
                 tau_plan=0.05, lambda_A=1e-3, score_clip=10.0, seed=0):
        self.M = M
        self.eta = float(eta)
        self.T0, self.T1 = float(T0), float(T1)
        self.g0, self.g1 = float(gamma0), float(gamma1)
        self.tau_plan, self.lambda_A = float(tau_plan), float(lambda_A)
        self.score_clip = float(score_clip)
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(M, K, L, generator=g) * 0.1
        self.mask = torch.zeros(K, L, dtype=torch.bool)      # True = valid value
        for k, (_n, vals) in enumerate(AXES):
            self.mask[k, :len(vals)] = True
        self.A = self.A.masked_fill(~self.mask, -1e9)
        self._center()
        self.baseline = [0.5] * M                            # per-particle EMA (Eq 19 b)
        self.baseline_m = 0.9
        # uniform prior over VALID values per axis (rho0, Eq 17)
        self.rho0 = self.mask.float() / self.mask.float().sum(dim=1, keepdim=True)

    # ---- schedules (annealing with floors, review-of-paper §2.3) ----
    def temps(self, s, S):
        f = s / max(1, S - 1)
        T = self.T0 + (self.T1 - self.T0) * f
        gamma = self.g0 + (self.g1 - self.g0) * f
        return max(T, self.T1), max(gamma, self.g1)

    def probs(self, p, gamma):
        logits = (self.A[p] / gamma).masked_fill(~self.mask, -1e9)
        return F.softmax(logits, dim=-1)                     # (K, L)

    # ---- sampling (Eq 12): hard one-hot rows ----
    def sample_plan(self, p, gamma, generator=None):
        P = self.probs(p, gamma)
        rows = [int(torch.multinomial(P[k], 1, generator=generator)) for k in range(K)]
        return rows, P

    # ---- analytic reward-score + autograd regularizer gradient (Eq 19) ----
    def grad(self, p, rows, P, U_hat, gamma):
        Z = torch.zeros(K, L)
        for k, l in enumerate(rows):
            Z[k, l] = 1.0
        adv = float(U_hat) - self.baseline[p]
        coeff = max(-self.score_clip, min(self.score_clip, adv / gamma))  # clip 1/γ score
        g_reward = coeff * (Z - P)
        # regularizer: τ_plan Σ_k KL(P_k(A;γ) || ρ0_k) + λ_A/2 ||A||² via autograd (tiny)
        A = self.A[p].clone().requires_grad_(True)
        logits = (A / gamma).masked_fill(~self.mask, -1e9)
        Psoft = F.softmax(logits, dim=-1)
        kl = (Psoft * (torch.log(Psoft + 1e-12) - torch.log(self.rho0 + 1e-12))
              ).masked_select(self.mask).sum()
        reg = self.tau_plan * kl + 0.5 * self.lambda_A * (A.masked_select(self.mask) ** 2).sum()
        reg.backward()
        return g_reward - A.grad

    def step(self, p, rows, P, U_hat, s, S, generator=None):
        T, gamma = self.temps(s, S)
        g = self.grad(p, rows, P, U_hat, gamma)
        eta = self.eta * (gamma / self.g0)                   # shrink step as γ sharpens
        noise = torch.randn(K, L, generator=generator) * math.sqrt(2.0 * eta * T)
        upd = eta * g + noise
        self.A[p] = torch.where(self.mask, self.A[p] + upd, self.A[p])
        self._center()
        self.baseline[p] = self.baseline_m * self.baseline[p] + (1 - self.baseline_m) * float(U_hat)
        return {"T": T, "gamma": gamma, "adv": float(U_hat) - self.baseline[p],
                "gnorm": float(g.masked_select(self.mask).norm())}

    def _center(self):
        # remove row-wise softmax shift non-identifiability over VALID entries (Eq 20)
        for k in range(K):
            m = self.mask[k]
            self.A[:, k, :][:, m] -= self.A[:, k, :][:, m].mean(dim=1, keepdim=True)

    # ---- quantization (Eq 21) ----
    def quantize(self, p):
        return [int(self.A[p, k].masked_fill(~self.mask[k], -1e9).argmax()) for k in range(K)]

    def describe(self, rows):
        return {AXES[k][0]: AXES[k][1][rows[k]] for k in range(K)}

    def row_entropy(self, p, gamma=1.0):
        """Mean per-axis entropy (nats) of P(A;γ) — saturation monitor (review P2-4)."""
        P = self.probs(p, gamma)
        ent = -(P * torch.log(P + 1e-12)).masked_select(self.mask)
        # sum within rows then average across axes
        Pm = self.probs(p, gamma)
        rows = [float(-(Pm[k][self.mask[k]] * torch.log(Pm[k][self.mask[k]] + 1e-12)).sum())
                for k in range(K)]
        return sum(rows) / K

    def add_refresh_noise(self, scale=0.05, min_entropy=0.3):
        """Warm-start refresh (Alg 1 step 3c). Saturated particles (mean row entropy
        < min_entropy nats) get CONTRACTED logits + larger noise — plain small noise
        cannot meaningfully refresh near-one-hot logits (review P2-4)."""
        for p in range(self.M):
            if self.row_entropy(p) < min_entropy:
                self.A[p] = torch.where(self.mask, self.A[p] * 0.5, self.A[p])
                sc = scale * 4
            else:
                sc = scale
            self.A[p] += torch.where(self.mask[None].squeeze(0),
                                     torch.randn(K, L) * sc, torch.zeros(K, L))
        self._center()

    def save(self, path):
        torch.save({"A": self.A, "baseline": self.baseline}, path)

    def load(self, path):
        if os.path.exists(path):
            d = torch.load(path, map_location="cpu")
            if d["A"].shape == self.A.shape:
                self.A = d["A"]
                self.baseline = list(d.get("baseline", self.baseline))
                print(f"[plan] warm-started A from {path}", flush=True)

    def logits_snapshot(self):
        return {f"particle{p}": {AXES[k][0]: [round(float(v), 3) for v, ok in
                                              zip(self.A[p, k], self.mask[k]) if ok]
                                 for k in range(K)} for p in range(self.M)}

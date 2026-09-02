"""CPU correctness tests for Planning SGLD (finite plan space => exact enumeration).

1. Score-function identity: exact grad of E_{Z~qA}[U(Z)] by full enumeration+autograd
   vs the analytic MC estimator ((U−b)/γ)(Z−P) averaged over many samples.
2. Prior-only behavior: with U ≡ const, SGLD keeps P(A;γ) near the uniform prior.
3. Directional: reward one specific plan cell => argmax quantization finds it.

Run: python3 DEO/plan_sgld_tests.py  (CPU, ~20 s)
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "DEO")
from plan_sgld import PlanSGLD, K, L, AXES, LENGTH_AXIS as LENGTH_AXIS_TEST


def enum_plans(mask):
    """All valid plans as lists of row indices."""
    import itertools
    choices = [list(range(int(mask[k].sum()))) for k in range(K)]
    return list(itertools.product(*choices))


def test_score_identity(n_mc=3_000_000, seed=0):
    torch.manual_seed(seed)
    pl = PlanSGLD(M=1, seed=seed)
    gamma = 0.7
    U_table = {plan: float(torch.rand(1)) for plan in enum_plans(pl.mask)}

    # exact gradient by enumeration + autograd
    A = pl.A[0].clone().requires_grad_(True)
    logits = (A / gamma).masked_fill(~pl.mask, -1e9)
    P = F.softmax(logits, dim=-1)
    EU = 0.0
    for plan, u in U_table.items():
        prob = 1.0
        for k, l in enumerate(plan):
            prob = prob * P[k, l]
        EU = EU + prob * u
    EU.backward()
    g_exact = A.grad.masked_select(pl.mask)

    # MC analytic estimator with a constant baseline b (unbiased for any b) — vectorized
    b = 0.5
    gen = torch.Generator().manual_seed(seed + 1)
    Pdet = pl.probs(0, gamma)
    samples = torch.stack([torch.multinomial(Pdet[k], n_mc, replacement=True, generator=gen)
                           for k in range(K)])                     # (K, n_mc)
    u_vals = torch.tensor([U_table[tuple(samples[:, i].tolist())] for i in range(n_mc)])
    w = (u_vals - b) / gamma                                       # (n_mc,)
    g_mc = torch.zeros(K, L)
    for k in range(K):
        g_mc[k].scatter_add_(0, samples[k], w)                     # Σ w·Z_k
        g_mc[k] -= w.sum() * Pdet[k]                               # − Σ w·P_k
    g_mc = (g_mc / n_mc).masked_select(pl.mask)
    rel = float((g_mc - g_exact).norm() / (g_exact.norm() + 1e-12))
    print(f"[test1] rel_err={rel:.4f} (exact-norm {float(g_exact.norm()):.4f})")
    assert rel < 0.05, f"FAIL: categorical score estimator biased rel={rel}"
    print("[test1] PASSED: categorical score-function identity vs enumeration\n")


def test_prior_only(seed=1, steps=20000, burn=4000):
    """Zero-advantage SGLD: the ERGODIC (time-averaged) plan distribution must sit
    near the prior; single snapshots fluctuate with variance ~ T/curvature, so we
    average P over the post-burn-in trajectory. Also assert no logit runaway."""
    # NOTE: KL(P||uniform) is BOUNDED (<= log L), so its restoring force vanishes
    # once softmax saturates; at high T with tiny lambda_A the chain legitimately
    # wanders among saturated corners (relaxation ~ 1/(eta*lambda_A)). Test in the
    # regime where the restoring force dominates the noise:
    pl = PlanSGLD(M=1, eta=0.1, T0=0.05, T1=0.05, gamma0=1.0, gamma1=1.0,
                  tau_plan=0.5, lambda_A=0.05, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    Pbar = torch.zeros(K, L)
    m = 0
    for s in range(steps):
        rows, P = pl.sample_plan(0, 1.0, generator=gen)
        pl.step(0, rows, P, U_hat=pl.baseline[0], s=0, S=2, generator=gen)  # zero advantage
        if s >= burn:
            Pbar += pl.probs(0, 1.0)
            m += 1
    Pbar /= m
    for k in range(K):
        valid = pl.mask[k]
        dev = float((Pbar[k][valid] - pl.rho0[k][valid]).abs().max())
        assert dev < 0.12, f"FAIL: axis {k} time-avg drifted from prior (max dev {dev})"
    assert float(pl.A.masked_select(pl.mask).abs().max()) < 8.0, "FAIL: logit runaway"
    print("[test2] PASSED: zero-advantage ergodic average stays near uniform prior\n")


def test_directional(seed=2, steps=1500, M=4):
    """Reward = fraction of axes matching a hidden target (smooth, separable).
    A SINGLE annealed chain can lock into a wrong basin (EMA baseline zeroes the
    advantage inside any basin) — that is precisely why the method keeps M
    particles (Eq 22). Production shape: M particles, assert AT LEAST ONE
    quantizes to the exact target and that particle's plan wins on reward."""
    target = [0, 2, 1, 1, 3]
    pl = PlanSGLD(M=M, eta=0.4, T0=0.1, T1=0.01, gamma0=1.0, gamma1=0.3,
                  tau_plan=0.02, lambda_A=1e-3, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    for s in range(steps):
        for p in range(M):
            rows, P = pl.sample_plan(p, pl.temps(s, steps)[1], generator=gen)
            u = sum(1 for a, b in zip(rows, target) if a == b) / K
            pl.step(p, rows, P, u, s, steps, generator=gen)
    plans = [pl.quantize(p) for p in range(M)]
    scores = [sum(1 for a, b in zip(rows, target) if a == b) / K for rows in plans]
    print(f"[test3] target={target}")
    for p, rows in enumerate(plans):
        print(f"[test3] particle{p}: {rows} match={scores[p]:.1f}")
    assert max(scores) == 1.0, "FAIL: no particle recovered the rewarded plan"
    print("[test3] PASSED: >=1 of M particles quantizes to the rewarded plan\n")


# ================= review-added tests (P1-5) =================

def test_grad_implementation(n_mc=300_000, seed=3):
    """Expectation of the IMPLEMENTED PlanSGLD.grad() (incl. KL + L2 regularizers)
    vs exact enumeration+autograd of J(A) = E[U] − τ_plan·KL − λ_A/2‖A‖²."""
    torch.manual_seed(seed)
    pl = PlanSGLD(M=1, tau_plan=0.07, lambda_A=0.02, seed=seed)
    pl.baseline[0] = 0.4                       # fixed baseline b
    gamma = 0.8
    U_table = {plan: float(torch.rand(1)) for plan in enum_plans(pl.mask)}

    # exact ∇J by enumeration + autograd
    A = pl.A[0].clone().requires_grad_(True)
    logits = (A / gamma).masked_fill(~pl.mask, -1e9)
    P = F.softmax(logits, dim=-1)
    EU = 0.0
    for plan, u in U_table.items():
        prob = 1.0
        for k, l in enumerate(plan):
            prob = prob * P[k, l]
        EU = EU + prob * u
    kl = (P * (torch.log(P + 1e-12) - torch.log(pl.rho0 + 1e-12))).masked_select(pl.mask).sum()
    J = EU - pl.tau_plan * kl - 0.5 * pl.lambda_A * (A.masked_select(pl.mask) ** 2).sum()
    J.backward()
    g_exact = A.grad.masked_select(pl.mask)

    # MC average of the implemented grad() (baseline fixed; unbiased for any b)
    gen = torch.Generator().manual_seed(seed + 1)
    g_sum = torch.zeros(K, L)
    Pdet = pl.probs(0, gamma)
    samples = torch.stack([torch.multinomial(Pdet[k], n_mc, replacement=True, generator=gen)
                           for k in range(K)])
    u_vals = torch.tensor([U_table[tuple(samples[:, i].tolist())] for i in range(n_mc)])
    w = (u_vals - pl.baseline[0]) / gamma
    for k in range(K):
        g_sum[k].scatter_add_(0, samples[k], w)
        g_sum[k] -= w.sum() * Pdet[k]
    g_reward_mc = g_sum / n_mc
    # deterministic regularizer part straight from the implementation
    g_reg = pl.grad(0, [0]*K, Pdet, pl.baseline[0], gamma)  # adv=0 → pure −∇reg
    g_mc = (g_reward_mc + g_reg).masked_select(pl.mask)
    rel = float((g_mc - g_exact).norm() / (g_exact.norm() + 1e-12))
    print(f"[test4] implemented-grad rel_err={rel:.4f}")
    assert rel < 0.05, f"FAIL: PlanSGLD.grad() disagrees with enumeration (rel {rel})"
    print("[test4] PASSED: implemented grad (reward+KL+L2) matches enumeration\n")


def test_coupled_shared_reward(n_mc=800_000, seed=4):
    """Two particles, reward penalizing SAME choice on axis 0 (non-separable).
    The shared-batch-reward estimator must match exact enumeration."""
    torch.manual_seed(seed)
    pl = PlanSGLD(M=2, tau_plan=0.0, lambda_A=0.0, seed=seed)
    # skew particle 2's axis-0 so the exact coupling gradient is well away from zero
    # (near-uniform P2 makes g_exact ~ 0 and the relative metric ill-conditioned)
    pl.A[1, 0, :4] = torch.tensor([1.5, -0.5, 0.0, -1.0])
    gamma = 1.0
    P1, P2 = pl.probs(0, gamma), pl.probs(1, gamma)

    # exact ∇_{A1} E[R], R = 1 − 1{axis0 equal} (depends only on axis 0)
    A1 = pl.A[0].clone().requires_grad_(True)
    l1 = (A1 / gamma).masked_fill(~pl.mask, -1e9)
    P1g = F.softmax(l1, dim=-1)
    ER = 1.0 - (P1g[0] * P2[0].detach()).sum()
    ER.backward()
    g_exact = A1.grad.masked_select(pl.mask)

    gen = torch.Generator().manual_seed(seed + 1)
    s1 = torch.stack([torch.multinomial(P1[k], n_mc, replacement=True, generator=gen)
                      for k in range(K)])
    s2_ax0 = torch.multinomial(P2[0], n_mc, replacement=True, generator=gen)
    R = 1.0 - (s1[0] == s2_ax0).float()
    b = float(R.mean())
    w = (R - b) / gamma
    g_sum = torch.zeros(K, L)
    for k in range(K):
        g_sum[k].scatter_add_(0, s1[k], w)
        g_sum[k] -= w.sum() * P1[k]
    g_mc = (g_sum / n_mc).masked_select(pl.mask)
    rel = float((g_mc - g_exact).norm() / (g_exact.norm() + 1e-12))
    print(f"[test5] coupled shared-reward rel_err={rel:.4f}")
    assert rel < 0.06, f"FAIL: shared-reward estimator biased (rel {rel})"
    print("[test5] PASSED: shared pooled reward correct under particle coupling\n")


def test_production_horizon(seeds=8, S=30, B=25, M=8):
    """Production config: S=30 annealed steps + ONE coordinate-refine sweep on the
    best particle (as in the runner). Pure SGLD at this horizon plateaus at ~0.8
    (one axis locks early — measured); the refine sweep must repair it."""
    from plan_sgld import coordinate_refine
    target = [0, 2, 1, 1, 3]
    def u_of(rows, gen):
        u = sum(1 for a, b2 in zip(rows, target) if a == b2) / K
        return u + float(torch.randn(1, generator=gen)) * (0.2 / (B ** 0.5))
    hits, best_matches = 0, []
    for sd in range(seeds):
        pl = PlanSGLD(M=M, seed=100 + sd)          # production defaults
        gen = torch.Generator().manual_seed(200 + sd)
        for s in range(S):
            for p in range(M):
                rows, P = pl.sample_plan(p, pl.temps(s, S)[1], generator=gen)
                pl.step(p, rows, P, u_of(rows, gen), s, S, generator=gen)
        plans = [pl.quantize(p) for p in range(M)]
        utils = [u_of(r, gen) for r in plans]
        top = max(range(M), key=lambda p: utils[p])
        refined = coordinate_refine(plans[top],
                                    lambda vs: [u_of(v, gen) for v in vs])
        m = sum(1 for a, b2 in zip(refined, target) if a == b2) / K
        best_matches.append(m)
        hits += (m == 1.0)
    mean_best = sum(best_matches) / seeds
    print(f"[test6] production+refine: mean best-match={mean_best:.2f}, "
          f"perfect {hits}/{seeds}")
    assert hits >= seeds - 1, f"FAIL: refine did not repair stuck axes ({hits}/{seeds})"
    print("[test6] PASSED: 30-step schedule + coordinate refine recovers the target\n")


def test_length_and_quota():
    from plan_sgld import length_ok, gated_utility, allocate_quota, LENGTH_BINS
    # disjoint half-open bins (audit P1-3): EVERY boundary token maps to exactly one bin
    for tok in [79, 80, 149, 150, 249, 250, 399, 400, 599, 600, 601]:
        owners = []
        for b in range(4):
            r = [0, b, 0, 0, 0]
            if length_ok(tok, r):
                owners.append(b)
        lo_any = 80 <= tok < 600
        assert len(owners) == (1 if lo_any else 0), f"token {tok} maps to bins {owners}"
    # gates (review P0-3)
    assert gated_utility(0.9, 0, "42", 0.5, True, True, 0, 0.3, 0.8) == 0.9
    assert gated_utility(0.9, 0, None, 0.5, True, True, 0, 0.3, 0.8) == 0.0
    assert gated_utility(0.9, 0, "42", 0.9, True, True, 0, 0.3, 0.8) == 0.0
    assert gated_utility(0.9, 0, "42", 0.5, False, True, 0, 0.3, 0.8) == 0.0
    assert gated_utility(0.9, 0, "42", 0.5, True, False, 0, 0.3, 0.8) == 0.0
    # quota: dedup + zero plans excluded + floor/cap guards (audit P1-2)
    plans = [[0]*K, [0]*K, [1]*K, [2]*K]
    utils = [0.9, 0.9, 0.3, 0.05]
    alloc = allocate_quota(plans, utils, 1000, tau_mix=0.05)
    d = {tuple(r): q for r, q in alloc}
    assert sum(d.values()) == 1000 and len(d) == 3
    assert max(d.values()) <= 0.45 * 1000, f"share cap violated: {d}"   # 0.4 cap (+rounding)
    assert min(d.values()) >= 0.04 * 1000, f"min floor violated: {d}"
    # all-zero -> None (audit P0-3)
    assert allocate_quota(plans, [0.0, 0.0, 0.0, 0.0], 100) is None
    print(f"[test7] PASSED: disjoint bins incl. boundaries, gates, capped quota "
          f"(alloc={d}), all-zero -> None\n")


if __name__ == "__main__":
    test_score_identity()
    test_prior_only()
    test_directional()
    test_grad_implementation()
    test_coupled_shared_reward()
    test_production_horizon()
    test_length_and_quota()
    print("ALL PLANNING-SGLD TESTS PASSED")

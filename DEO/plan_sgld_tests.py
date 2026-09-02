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
from plan_sgld import PlanSGLD, K, L, AXES


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


if __name__ == "__main__":
    test_score_identity()
    test_prior_only()
    test_directional()
    print("ALL PLANNING-SGLD TESTS PASSED")

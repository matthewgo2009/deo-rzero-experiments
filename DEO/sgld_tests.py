"""CPU math-correctness tests for the SGLD machinery (review §Required Tests 2-3).

1. EXACT-ENUMERATION score-function test: a tiny categorical autoregressive model
   whose sequence space can be enumerated. Verifies
       ∇_z E_{x~π(·|z)}[R(x)]  ==  E[R(x) ∇_z log π(x|z)]
   by comparing the Monte-Carlo score-function estimate (many samples) against the
   EXACT enumerated gradient (autograd through the exact expectation).
2. PRIOR-ONLY SGLD stationarity: with zero advantage the chain must converge to the
   Gaussian prior N(0, σ²).

Run:  python3 DEO/sgld_tests.py   (CPU, ~30 s)
"""
import math
import sys

import torch


def test_score_function_identity(seed=0, n_mc=200_000):
    torch.manual_seed(seed)
    V, T, dz = 4, 2, 3                      # vocab 4, length 2, latent dim 3
    W = torch.randn(T, V, dz) * 0.7         # position-wise logits = W_t @ tanh(z)
    R = torch.rand(V, V) * 2 - 0.5          # arbitrary reward per sequence (x1,x2)

    def logits(z):                          # (T, V)
        return torch.einsum("tvd,d->tv", W, torch.tanh(z))

    # ---- exact gradient by enumeration + autograd ----
    z = torch.zeros(dz, requires_grad=True)
    lg = torch.log_softmax(logits(z), dim=-1)
    # E[R] = Σ_{a,b} p(a)p(b|·) R[a,b]  (factorized toy: position-independent given z)
    p = lg.exp()
    ER = torch.einsum("a,b,ab->", p[0], p[1], R)
    ER.backward()
    g_exact = z.grad.clone()

    # ---- Monte-Carlo score-function estimate ----
    with torch.no_grad():
        pdet = torch.log_softmax(logits(torch.zeros(dz)), dim=-1).exp()
        xs1 = torch.multinomial(pdet[0], n_mc, replacement=True)
        xs2 = torch.multinomial(pdet[1], n_mc, replacement=True)
        r = R[xs1, xs2]
        b = r.mean()                        # baseline (independent of each sample as n→∞)
    z2 = torch.zeros(dz, requires_grad=True)
    lg2 = torch.log_softmax(logits(z2), dim=-1)
    seq_lp = lg2[0, xs1] + lg2[1, xs2]
    ((r - b).detach() * seq_lp).sum().backward()
    g_mc = z2.grad / n_mc

    rel = float((g_mc - g_exact).norm() / (g_exact.norm() + 1e-12))
    print(f"[test1] exact grad {g_exact.tolist()}")
    print(f"[test1] MC score-function grad {g_mc.tolist()}  rel_err={rel:.4f}")
    assert rel < 0.05, f"FAIL: score-function estimator biased (rel {rel})"
    print("[test1] PASSED: score-function identity verified against enumeration\n")


def test_prior_only_stationarity(seed=1, sigma=1.0, eta=5e-3, steps=20_000):
    sys.path.insert(0, "DEO")
    from sgld_soft_prefix import SoftPrefixSGLD
    s = object.__new__(SoftPrefixSGLD)
    s.sigma, s.tau, s.eta = sigma, 0.1, eta
    s.K, s.d, s.n = 2, 4, 64
    torch.manual_seed(seed)
    s.Z = torch.randn(s.n, s.K, s.d) * 3.0     # start far from stationarity
    s.baseline, s.baseline_m = 0.5, 0.9
    s.alpha = 1.0
    def diagnostics(idxs=None):
        return {"prompt_rms": 1.0, "latent_rms": 0.0, "delta_rms": 0.0, "delta_to_prompt": 0.0}
    s.diagnostics = diagnostics
    idxs = list(range(s.n))
    zero_g = {k: torch.zeros(s.K, s.d) for k in idxs}
    const_r = {k: s.baseline for k in idxs}    # zero advantage → prior-only dynamics
    for _ in range(steps):
        s.step(idxs, const_r, zero_g)
    mean = float(s.Z.mean()); std = float(s.Z.std())
    # discretized OU stationary variance = σ²/(1 − ηh/σ²/2) ≈ σ² for small η
    print(f"[test2] after {steps} prior-only steps: mean={mean:+.4f} std={std:.4f} "
          f"(target 0 / {sigma:.2f})")
    assert abs(mean) < 0.05 and abs(std - sigma) < 0.1 * sigma, "FAIL: prior stationarity"
    print("[test2] PASSED: prior-only SGLD converges to N(0, σ²)\n")


if __name__ == "__main__":
    test_score_function_identity()
    test_prior_only_stationarity()
    print("ALL SGLD MATH-CORRECTNESS TESTS PASSED")

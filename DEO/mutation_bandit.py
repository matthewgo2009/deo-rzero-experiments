"""Contextual Thompson-sampling memory over mutation operators (deo_with_memory.pdf §1.4).

The bandit learns WHICH mutation strategy (A–E) tends to produce valid, in-band, novel
questions for each question context; Metropolis–Hastings remains responsible for
accepting or rejecting the resulting proposals.

- Context c(x) = (SEED topic, difficulty bucket of p_hat): buckets Hard (p<pmin),
  Trainable (pmin<=p<=pmax), Easy (p>pmax)                            [Eq (10)-(11)]
  The topic is assigned at initial generation and inherited through mutations
  (i.e. c uses topic(x_0), not a re-classified topic(x_t)).
- Per (context, action): Beta(alpha, beta) posterior, uniform init    [Eq (12)]
- Selection: Thompson sampling, z_a ~ Beta, a* = argmax               [Eq (14)]
- Success s(x') = Valid(x') * 1{pmin<=p_hat(x')<=pmax} * Novel(x')    [Eq (16)]
  Valid = strategy compliance + parse + SURFACE validity (a cheap LLM format
  filter that defaults to VALID; it does not certify mathematical correctness).
  For an already-Trainable state the caller additionally requires
  r_unc(x') >= r_unc(x) - eps so "success" can't mean drifting to the band edge.
- Frozen within an outer iteration; discounted update afterwards:
  alpha <- 1 + rho*(alpha-1) + S,  beta <- 1 + rho*(beta-1) + F       [Eq (17)]

Note: the TS mixture proposal is state-dependent (context of x vs x' differ) and
the LLM kernels K_a are not symmetric, so with the l~=0 approximation the sampler
is an APPROXIMATE MH / energy-based accept-reject heuristic, not exact MH — same
approximation the non-bandit walk already makes, made explicit here.

State is a plain-JSON dict, safe to persist/reload across outer iterations and
job restarts. Pure stdlib (random.betavariate); no numpy needed.
"""
import json
import os
import random

ACTIONS = ["A", "B", "C", "D", "E"]
ACTION_NAMES = {
    "A": "GENERALIZE", "B": "COMPOSE", "C": "INVERT",
    "D": "CHANGE_OBJECTIVE", "E": "DUALIZE",
}


class MutationBandit:
    def __init__(self, pmin=0.3, pmax=0.8, rho=0.9):
        self.pmin, self.pmax, self.rho = pmin, pmax, rho
        self.state = {}     # {context_key: {action: [alpha, beta]}}
        self.pending = []   # frozen-iteration history H: list of (context_key, action, success)

    # ---- context (Eq 10-11) ----
    def bucket(self, p_hat):
        if p_hat is None:
            return "Unknown"
        p = float(p_hat)
        if p < self.pmin:
            return "Hard"
        if p > self.pmax:
            return "Easy"
        return "Trainable"

    def context_key(self, topic, p_hat):
        return f"{topic or 'unknown'}|{self.bucket(p_hat)}"

    def _cell(self, ctx, a):
        return self.state.setdefault(ctx, {}).setdefault(a, [1.0, 1.0])

    # ---- Thompson sampling selection (Eq 14); does NOT mutate state ----
    def select(self, topic, p_hat):
        ctx = self.context_key(topic, p_hat)
        draws = {a: random.betavariate(*self._cell(ctx, a)) for a in ACTIONS}
        return max(draws, key=draws.get), ctx

    # ---- frozen-iteration feedback buffer (success per Eq 16, judged by caller) ----
    def record(self, ctx, action, success):
        self.pending.append((ctx, action, 1 if success else 0))

    # ---- discounted end-of-iteration update (Eq 17) ----
    def end_iteration(self):
        S, F = {}, {}
        for ctx, a, s in self.pending:
            d = S if s else F
            d[(ctx, a)] = d.get((ctx, a), 0) + 1
        touched = set(S) | set(F)
        # discount every existing cell (stale experience decays even if untouched this iter)
        for ctx, acts in self.state.items():
            for a, ab in acts.items():
                ab[0] = 1.0 + self.rho * (ab[0] - 1.0)
                ab[1] = 1.0 + self.rho * (ab[1] - 1.0)
        for (ctx, a) in touched:
            ab = self._cell(ctx, a)
            ab[0] += S.get((ctx, a), 0)
            ab[1] += F.get((ctx, a), 0)
        n_prop = len(self.pending)
        n_succ = sum(s for _, _, s in self.pending)
        self.pending = []
        return n_prop, n_succ

    # ---- summary for logs ----
    def summary(self, top=12):
        rows = []
        for ctx, acts in self.state.items():
            for a, (al, be) in acts.items():
                n = al + be - 2.0
                if n > 0:
                    rows.append((ctx, a, al / (al + be), n))
        rows.sort(key=lambda r: -r[3])
        return [f"{ctx} {a}({ACTION_NAMES[a][:3]}): mean={m:.2f} n={n:.0f}"
                for ctx, a, m, n in rows[:top]]

    def action_means(self):
        """Aggregate per-action success mean across all contexts (for quick logging)."""
        agg = {a: [0.0, 0.0] for a in ACTIONS}
        for acts in self.state.values():
            for a, (al, be) in acts.items():
                agg[a][0] += al - 1.0
                agg[a][1] += be - 1.0
        return {a: (s / (s + f) if s + f > 0 else 0.5) for a, (s, f) in agg.items()}

    # ---- persistence ----
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"pmin": self.pmin, "pmax": self.pmax, "rho": self.rho,
                       "state": self.state}, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, pmin=0.3, pmax=0.8, rho=0.9):
        b = cls(pmin=pmin, pmax=pmax, rho=rho)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            b.state = d.get("state", {})
            print(f"[bandit] loaded state from {path} "
                  f"({len(b.state)} contexts)", flush=True)
        return b

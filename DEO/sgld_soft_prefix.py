"""Latent SGLD over soft-prefix variables (DEO_SGLD.pdf §2).

Per question i, a continuous soft prefix z_i ∈ R^{K×d} (K = prompt token count,
d = hidden dim) is ADDED to the prompt token embeddings of the frozen base model,
defining x_i ~ π0(·|p0, z_i)  (Eq 10-11; z=0 recovers the base distribution).

Latent Gibbs target (Eq 13):  q(Z) ∝ N(Z; 0, σ²I) · exp(r̄_c(Z,θ)/τ).
Score-function gradient (Eq 14-17): ĝ_i = (r̂_i − b) · ∇_{z_i} log π0(x_i|p0,z_i),
computed by teacher-forcing the SAMPLED tokens through the frozen model and
backpropagating only to z_i (Eq 15-16).
SGLD update (Eq 19):  z ← z + η(−z/σ² + ĝ/τ) + √(2η)·ε.

IMPLEMENTATION NOTES (deviations to flag for review):
1. REWARD CREDIT. Eq 17 multiplies every score s_i by the single BATCH-level
   scalar r̂_c(X,θ) (mean over n questions). With n≈2000 that coefficient is
   nearly constant across i and the estimator is almost pure noise. We instead
   use the PER-QUESTION utility u_i = r_unc(x_i) − λ_rep·r_rep(x_i;X) (the
   i-th summand of Eq 9, un-normalized) with a running-mean baseline. For the
   separable part this is the standard local-credit REINFORCE with strictly
   lower variance and the same expected direction; r_rep still couples through
   the current batch. Env SGLD_GLOBAL_REWARD=1 restores the paper-literal
   batch-scalar coefficient.
2. TEMPERATURE SCALE. The paper's τ = β/n applies to the batch-MEAN reward
   (scale 1/n). With per-question u_i (scale 1) the matching temperature is β
   itself. Env SGLD_TAU is therefore interpreted on the per-question scale
   (default 0.1, matching the fixed-β walk runs).
3. Generation AND teacher-forced scoring run on a single HF copy of the frozen
   base model (vLLM cannot take per-sample inputs_embeds); solver rollouts for
   rewards use the existing vLLM DP endpoints via the caller.
"""
import math
import os

import torch
import torch.nn.functional as F


class SoftPrefixSGLD:
    def __init__(self, model, tokenizer, prompt_text, n,
                 sigma=1.0, tau=0.1, eta=1e-3, device="cuda:7",
                 gen_bs=16, max_new_tokens=1024, seed=0):
        """model: frozen HF causal LM (bf16 ok). prompt_text: the rendered
        question-generation prompt p0 (chat template already applied)."""
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = tokenizer
        self.device = device
        self.sigma, self.tau, self.eta = float(sigma), float(tau), float(eta)
        self.gen_bs, self.max_new_tokens = int(gen_bs), int(max_new_tokens)

        ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        self.prompt_ids = ids.to(device)                      # (K,)
        emb = self.model.get_input_embeddings()(self.prompt_ids)  # (K, d)
        self.prompt_emb = emb.detach().to(torch.float32)      # fp32 master copy
        self.K, self.d = self.prompt_emb.shape

        g = torch.Generator().manual_seed(seed)
        # z lives on CPU fp32 (n×K×d can be GBs); minibatches move to GPU.
        self.Z = torch.randn(n, self.K, self.d, generator=g) * self.sigma
        self.n = n
        self.baseline = 0.5   # running-mean baseline b (u_i ∈ [~-λ, 1])
        self.baseline_m = 0.9

    # ---------- generation: x_i ~ π0(·|p0, z_i) ----------
    @torch.no_grad()
    def generate(self, idxs, temperature=1.0, top_p=0.95):
        """Sample one question per index. Returns list[str] (raw completions)."""
        outs = [None] * len(idxs)
        mdtype = next(self.model.parameters()).dtype
        for s in range(0, len(idxs), self.gen_bs):
            chunk = idxs[s:s + self.gen_bs]
            z = self.Z[chunk].to(self.device)                       # (b,K,d) fp32
            inp = (self.prompt_emb.unsqueeze(0) + z).to(mdtype)     # (b,K,d)
            attn = torch.ones(inp.shape[:2], dtype=torch.long, device=self.device)
            gen = self.model.generate(
                inputs_embeds=inp, attention_mask=attn,
                do_sample=True, temperature=temperature, top_p=top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
            # with inputs_embeds, HF returns ONLY the generated ids
            for j, row in enumerate(gen):
                outs[s + j] = self.tok.decode(row, skip_special_tokens=True)
        return outs

    # ---------- score: ∇_z log π0(x|p0,z) by teacher forcing (Eq 15-16) ----------
    def score_grads(self, idxs, texts, score_bs=4, max_len=1024):
        """Returns dict idx -> fp32 CPU grad tensor (K,d). texts are the RAW
        completions sampled from the same z (teacher forcing holds them fixed)."""
        grads = {}
        mdtype = next(self.model.parameters()).dtype
        for s in range(0, len(idxs), score_bs):
            chunk = idxs[s:s + score_bs]
            ctexts = texts[s:s + score_bs]
            tok = self.tok(ctexts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len, add_special_tokens=False)
            x_ids = tok.input_ids.to(self.device)               # (b, L)
            x_mask = tok.attention_mask.to(self.device)
            z = self.Z[chunk].to(self.device).requires_grad_(True)   # leaf, fp32
            prefix = (self.prompt_emb.unsqueeze(0) + z).to(mdtype)   # grad flows to z
            x_emb = self.model.get_input_embeddings()(x_ids)         # (b, L, d)
            full = torch.cat([prefix, x_emb], dim=1)
            attn = torch.cat([torch.ones(prefix.shape[:2], dtype=torch.long,
                                         device=self.device), x_mask], dim=1)
            logits = self.model(inputs_embeds=full, attention_mask=attn).logits
            # positions predicting x tokens: prefix ends at K-1; logits[K-1+t] -> x[t]
            lp = F.log_softmax(logits[:, self.K - 1:self.K - 1 + x_ids.shape[1], :].float(), dim=-1)
            tok_lp = lp.gather(-1, x_ids.unsqueeze(-1)).squeeze(-1)   # (b, L)
            seq_lp = (tok_lp * x_mask).sum(dim=1)                     # Σ_ℓ log π0 (Eq 15)
            seq_lp.sum().backward()
            for j, k in enumerate(chunk):
                grads[k] = z.grad[j].detach().to("cpu", torch.float32)
            del z, prefix, x_emb, full, logits, lp
            torch.cuda.empty_cache()
        return grads

    # ---------- SGLD update (Eq 19) ----------
    def step(self, idxs, rewards, grads, eta=None):
        """rewards: dict idx -> scalar u_i (per-question credit; see note 1).
        Advantage = u_i − running baseline. Updates Z in place; returns stats."""
        eta = self.eta if eta is None else eta
        adv_sum, gnorm_sum, m = 0.0, 0.0, 0
        for k in idxs:
            if k not in grads:
                continue
            adv = float(rewards[k]) - self.baseline
            g = grads[k] * (adv / self.tau)                    # (1/τ)·ĝ  (Eq 17)
            drift = -self.Z[k] / (self.sigma ** 2) + g
            noise = torch.randn_like(self.Z[k]) * math.sqrt(2.0 * eta)
            self.Z[k] += eta * drift + noise
            adv_sum += adv; gnorm_sum += float(g.norm()); m += 1
        # running baseline update from this minibatch's rewards
        if idxs:
            rbar = sum(float(rewards[k]) for k in idxs if k in rewards) / max(
                1, sum(1 for k in idxs if k in rewards))
            self.baseline = self.baseline_m * self.baseline + (1 - self.baseline_m) * rbar
        return {"mean_adv": adv_sum / max(1, m), "mean_gnorm": gnorm_sum / max(1, m),
                "z_rms": float(self.Z[idxs].pow(2).mean().sqrt()) if idxs else 0.0,
                "baseline": self.baseline}

    def save(self, path):
        torch.save({"Z": self.Z, "baseline": self.baseline}, path)

    def load(self, path):
        if os.path.exists(path):
            d = torch.load(path, map_location="cpu")
            if d["Z"].shape == self.Z.shape:
                self.Z = d["Z"]
                self.baseline = float(d.get("baseline", 0.5))
                print(f"[sgld] warm-started Z from {path}", flush=True)
            else:
                print(f"[sgld] shape mismatch in {path}, cold start", flush=True)

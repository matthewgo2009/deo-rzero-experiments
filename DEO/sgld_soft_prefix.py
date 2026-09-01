"""Latent SGLD over soft-prefix variables (DEO_SGLD.pdf §2) — post-review revision.

Per question i, a dimensionless latent z_i ∈ R^{K×d}, z ~ N(0, σ²I), steers the frozen
base model through a SEPARATELY SCALED additive perturbation of the prompt embeddings:

    E_input(i) = E(p0) + α · z_i          (α ≪ prompt-embedding RMS; z=0 ⇒ base dist.)

Sampling:   x_i ~ π0(·|p0, z_i) with the RAW model policy (temperature=1, top_p=1,
            top_k=0, no logits warpers) so that the sampling distribution EQUALS the
            distribution whose score is used below (REINFORCE validity).
Score:      ∇_z log π0(x|p0,z) by teacher-forcing the EXACT sampled token ids
            (incl. the terminating EOS when present) — no decode/re-encode round trip.
SGLD:       z ← z + η(−z/σ² + ĝ/τ) + √(2η)·ε.

OBJECTIVE NOTE (deliberate deviation from the TeX, documented): the reward used by the
caller is the TENT uncertainty r_unc = 1−2|p̂−0.5| — the surrogate every prior DEO run
optimizes — NOT the TeX Eq-3 disagreement 1−p̂, which is maximized at p̂→0, i.e. outside
the p̂∈[0.3,0.8] training band the pipeline keeps. The TeX should be updated to match.
Per-question credit with τ=β is EXACT for this separable term (review §P0/P1); the
repetition penalty is disabled by default (λ_rep=0) pending an unbiased batch estimator.
"""
import math
import os

import torch
import torch.nn.functional as F


class SoftPrefixSGLD:
    def __init__(self, model, tokenizer, prompt_text, n,
                 alpha=None, sigma=1.0, tau=0.1, eta=1e-3, device="cuda:7",
                 gen_bs=16, max_new_tokens=1024, seed=0):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = tokenizer
        self.device = device
        self.sigma, self.tau, self.eta = float(sigma), float(tau), float(eta)
        self.gen_bs, self.max_new_tokens = int(gen_bs), int(max_new_tokens)

        ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        self.prompt_ids = ids.to(device)
        emb = self.model.get_input_embeddings()(self.prompt_ids)
        self.prompt_emb = emb.detach().to(torch.float32)           # (K, d) fp32 master
        self.K, self.d = self.prompt_emb.shape
        self.prompt_rms = float(self.prompt_emb.pow(2).mean().sqrt())

        # perturbation scale α: default 5% of prompt-embedding RMS (review §P1) —
        # a unit-RMS z would otherwise swamp the prompt entirely.
        self.alpha = float(alpha) if alpha is not None else 0.05 * self.prompt_rms

        # RAW sampling policy == scored policy (review §P0-1). Fresh GenerationConfig so
        # the model's own generation_config defaults (Qwen ships top_k=20/top_p=0.8!)
        # cannot silently truncate.
        from transformers import GenerationConfig  # lazy: step-math tests run w/o transformers
        self.gen_cfg = GenerationConfig(
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            num_beams=1, max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        print(f"[sgld] prompt K={self.K} d={self.d} prompt_rms={self.prompt_rms:.4f} "
              f"alpha={self.alpha:.5f} (delta/prompt RMS target "
              f"{self.alpha/self.prompt_rms:.3f}) | gen_cfg: temp=1 top_p=1 top_k=0 "
              f"max_new={self.max_new_tokens}", flush=True)

        g = torch.Generator().manual_seed(seed)
        self.Z = torch.randn(n, self.K, self.d, generator=g) * self.sigma   # CPU fp32
        self.n = n
        self.baseline = 0.5
        self.baseline_m = 0.9

    # ---------- diagnostics (review §P1) ----------
    def diagnostics(self, idxs=None):
        Z = self.Z if idxs is None else self.Z[idxs]
        z_rms = float(Z.pow(2).mean().sqrt())
        return {"prompt_rms": self.prompt_rms, "latent_rms": z_rms,
                "delta_rms": self.alpha * z_rms,
                "delta_to_prompt": self.alpha * z_rms / max(self.prompt_rms, 1e-9)}

    def _prefix(self, chunk_idxs, requires_grad=False):
        z = self.Z[chunk_idxs].to(self.device)
        if requires_grad:
            z.requires_grad_(True)
        mdtype = next(self.model.parameters()).dtype
        prefix = (self.prompt_emb.unsqueeze(0) + self.alpha * z).to(mdtype)
        return z, prefix

    # ---------- generation: raw policy, exact ids returned (review §P0-2) ----------
    @torch.no_grad()
    def generate(self, idxs):
        """Returns (texts, ids_list): texts for reward/parsing, EXACT generated token
        id tensors (cpu, incl. terminating EOS if emitted) for scoring."""
        texts, ids_list = [None] * len(idxs), [None] * len(idxs)
        eos = self.tok.eos_token_id
        pad = self.gen_cfg.pad_token_id
        for s in range(0, len(idxs), self.gen_bs):
            chunk = idxs[s:s + self.gen_bs]
            _z, prefix = self._prefix(chunk)
            attn = torch.ones(prefix.shape[:2], dtype=torch.long, device=self.device)
            gen = self.model.generate(inputs_embeds=prefix, attention_mask=attn,
                                      generation_config=self.gen_cfg)
            for j, row in enumerate(gen):        # with inputs_embeds HF returns gen ids only
                row = row.detach().cpu()
                # strip trailing PAD (batch padding), keep the first EOS (it was sampled)
                keep = len(row)
                if eos is not None and (row == eos).any():
                    keep = int((row == eos).nonzero()[0]) + 1     # include sampled EOS
                elif pad is not None and pad != eos and (row == pad).any():
                    keep = int((row == pad).nonzero()[0])
                ids = row[:keep]
                ids_list[s + j] = ids
                texts[s + j] = self.tok.decode(ids, skip_special_tokens=True)
        return texts, ids_list

    # ---------- score: exact sampled ids, teacher forcing (Eq 15-16) ----------
    def score_grads(self, idxs, ids_list, score_bs=2):
        """dict idx -> fp32 CPU grad (K,d) of Σ_ℓ log π0(x_ℓ | x_<ℓ, p0, z).
        ids_list must be the EXACT tensors returned by generate().
        Memory: fused cross-entropy (no full-vocab fp32 log_softmax map),
        use_cache=False, optional gradient checkpointing (SGLD_GRAD_CKPT=1)."""
        grads, logps = {}, {}
        pad = self.gen_cfg.pad_token_id
        ckpt = os.getenv("SGLD_GRAD_CKPT", "1") == "1"
        if ckpt and not getattr(self, "_ckpt_on", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self._ckpt_on = True
        for s in range(0, len(idxs), score_bs):
            chunk = idxs[s:s + score_bs]
            seqs = ids_list[s:s + score_bs]
            L = max(len(t) for t in seqs)
            x_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
            x_mask = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, t in enumerate(seqs):
                x_ids[j, :len(t)] = t
                x_mask[j, :len(t)] = 1
            x_ids, x_mask = x_ids.to(self.device), x_mask.to(self.device)
            z, prefix = self._prefix(chunk, requires_grad=True)
            x_emb = self.model.get_input_embeddings()(x_ids)
            full = torch.cat([prefix, x_emb], dim=1)
            attn = torch.cat([torch.ones(prefix.shape[:2], dtype=torch.long,
                                         device=self.device), x_mask], dim=1)
            logits = self.model(inputs_embeds=full, attention_mask=attn,
                                use_cache=False).logits
            sl = logits[:, self.K - 1:self.K - 1 + L, :]           # (b, L, V) bf16
            V = sl.shape[-1]
            # fused CE = -log softmax at the sampled token; avoids (b,L,V) fp32 map
            # CE directly on bf16 logits (internal fp32 math, no (b·L,V) fp32 copy);
            # only the tiny (b,L) output is upcast.
            tok_lp = -F.cross_entropy(sl.reshape(-1, V), x_ids.reshape(-1),
                                      reduction="none").view(x_ids.shape).float()
            seq_lp = (tok_lp * x_mask).sum(dim=1)                  # padding → zero
            seq_lp.sum().backward()
            zg = z.grad * self.alpha                                # ∂E_input/∂z = α
            for j, k in enumerate(chunk):
                grads[k] = zg[j].detach().to("cpu", torch.float32)
                logps[k] = float(seq_lp[j])
            del z, prefix, x_emb, full, logits, sl, tok_lp, seq_lp
            torch.cuda.empty_cache()   # needed: fragmentation OOMs GPU7 otherwise
        return grads, logps

    # ---------- SGLD update (Eq 19) ----------
    def step(self, idxs, rewards, grads, eta=None):
        eta = self.eta if eta is None else eta
        adv_sum = gnorm_sum = drift_sum = 0.0
        m = 0
        for k in idxs:
            if k not in grads:
                continue
            adv = float(rewards[k]) - self.baseline
            g = grads[k] * (adv / self.tau)
            drift = -self.Z[k] / (self.sigma ** 2) + g
            noise = torch.randn_like(self.Z[k]) * math.sqrt(2.0 * eta)
            self.Z[k] += eta * drift + noise
            adv_sum += adv; gnorm_sum += float(g.norm()); drift_sum += float(drift.norm()); m += 1
        if idxs:
            got = [k for k in idxs if k in rewards]
            if got:
                rbar = sum(float(rewards[k]) for k in got) / len(got)
                self.baseline = self.baseline_m * self.baseline + (1 - self.baseline_m) * rbar
        d = self.diagnostics(idxs)
        d.update({"mean_adv": adv_sum / max(1, m), "mean_gnorm": gnorm_sum / max(1, m),
                  "mean_driftnorm": drift_sum / max(1, m), "baseline": self.baseline})
        return d

    # ---------- sweep partition (review §P1: every latent exactly once per sweep) ----------
    def sweep_minibatches(self, minibatch, generator=None):
        perm = torch.randperm(self.n, generator=generator).tolist()
        return [perm[i:i + minibatch] for i in range(0, self.n, minibatch)]

    def save(self, path):
        torch.save({"Z": self.Z, "baseline": self.baseline, "alpha": self.alpha}, path)

    def load(self, path):
        if os.path.exists(path):
            d = torch.load(path, map_location="cpu")
            if d["Z"].shape == self.Z.shape:
                self.Z = d["Z"]
                self.baseline = float(d.get("baseline", 0.5))
                if "alpha" in d:
                    self.alpha = float(d["alpha"])
                print(f"[sgld] warm-started Z from {path}", flush=True)
            else:
                print(f"[sgld] shape mismatch in {path}, cold start", flush=True)

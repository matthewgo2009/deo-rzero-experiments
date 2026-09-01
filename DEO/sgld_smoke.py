"""Standalone single-GPU smoke test for the soft-prefix SGLD machinery.

Validates, WITHOUT the full pipeline (no vLLM, no verl):
  1. generation through inputs_embeds works and z=init produces parseable questions;
  2. score_grads returns finite, non-zero gradients w.r.t. z;
  3. a few SGLD steps with a synthetic reward VISIBLY move the sampled distribution
     (we reward LONGER questions as a cheap differentiable-free signal and check
     mean generated length rises).

Usage (any 1 GPU, ~10 min at 4B):
  CUDA_VISIBLE_DEVICES=0 python3 DEO/sgld_smoke.py [--model Qwen/Qwen3-4B-Base]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Base"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--gen_bs", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sgld_soft_prefix import SoftPrefixSGLD
    import mcmc_deo_vllm as deo_mod  # only for the prompt + parser (no servers touched)

    device = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    p0 = deo_mod.apply_chat_template(
        tok, deo_mod.CHALLENGER_SYSTEM_PROMPT,
        "Generate one new, challenging reasoning question now.")
    s = SoftPrefixSGLD(model, tok, p0, n=args.n, device=device,
                       gen_bs=args.gen_bs, max_new_tokens=512, eta=5e-3, tau=0.1)
    print(f"[smoke] K={s.K} d={s.d} n={args.n}")

    idxs = list(range(args.n))

    # --- check 1: generation + parse rate at init ---
    raw = s.generate(idxs)
    parsed = [deo_mod.extract_challenger_output(t)[0] for t in raw]
    ok = sum(1 for q in parsed if q and len(q) > 30)
    print(f"[smoke] init parse rate: {ok}/{args.n}")
    print("[smoke] sample question:", (next((q for q in parsed if q), "NONE"))[:160])
    assert ok >= args.n // 4, "FAIL: too few parseable questions from soft-prefix generation"

    # --- check 2: gradients finite/nonzero ---
    grads = s.score_grads(idxs[:4], raw[:4])
    for k, g in grads.items():
        assert torch.isfinite(g).all(), f"FAIL: non-finite grad for {k}"
        assert float(g.norm()) > 0, f"FAIL: zero grad for {k}"
    print(f"[smoke] grad norms: {[round(float(g.norm()),3) for g in grads.values()]}")

    # --- check 3: synthetic reward (longer question => higher u) moves lengths ---
    def mean_len(texts):
        return sum(len(t.split()) for t in texts) / len(texts)
    len0 = mean_len(raw)
    for step in range(args.steps):
        raw = s.generate(idxs)
        rewards = {i: min(1.0, len(raw[j].split()) / 200.0) for j, i in enumerate(idxs)}
        grads = s.score_grads(idxs, raw)
        st = s.step(idxs, rewards, grads)
        print(f"[smoke] step {step+1}: mean_len={mean_len(raw):.1f} "
              f"adv={st['mean_adv']:+.3f} |g/τ|={st['mean_gnorm']:.2e} z_rms={st['z_rms']:.3f}")
    raw = s.generate(idxs)
    len1 = mean_len(raw)
    print(f"[smoke] mean generated length: {len0:.1f} -> {len1:.1f} "
          f"({'MOVED' if len1 > len0 * 1.05 else 'no clear movement — inspect eta/tau'})")
    print("[smoke] PASSED (structural checks); distribution movement is advisory.")


if __name__ == "__main__":
    main()

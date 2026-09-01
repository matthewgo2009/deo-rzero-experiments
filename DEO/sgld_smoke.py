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

    # --- check 0: raw-policy generation config (review §P0-1) ---
    gc = s.gen_cfg
    assert gc.temperature == 1.0 and gc.top_p == 1.0 and gc.top_k == 0 and gc.num_beams == 1, \
        f"FAIL: gen config not raw policy: {gc}"
    print("[smoke] gen config is raw policy (temp=1, top_p=1, top_k=0)")

    # --- check 1: generation + parse rate at init ---
    raw, ids_list = s.generate(idxs)
    parsed = [deo_mod.extract_challenger_output(t)[0] for t in raw]
    ok = sum(1 for q in parsed if q and len(q) > 30)
    print(f"[smoke] init parse rate: {ok}/{args.n}")
    print("[smoke] sample question:", (next((q for q in parsed if q), "NONE"))[:160])
    assert ok >= args.n // 4, "FAIL: too few parseable questions from soft-prefix generation"

    # --- check 2: exact-token scoring, batched == single, finite/nonzero grads ---
    grads_b, logps_b = s.score_grads(idxs[:4], ids_list[:4], score_bs=4)   # one batch
    grads_1, logps_1 = s.score_grads(idxs[:4], ids_list[:4], score_bs=1)   # singles
    for k in grads_b:
        assert torch.isfinite(grads_b[k]).all() and float(grads_b[k].norm()) > 0
        assert abs(logps_b[k] - logps_1[k]) < 0.5, \
            f"FAIL: batched vs single logp mismatch {logps_b[k]} vs {logps_1[k]}"
        rel = float((grads_b[k]-grads_1[k]).norm()) / (float(grads_1[k].norm())+1e-9)
        assert rel < 0.05, f"FAIL: batched vs single grad mismatch rel={rel}"
    print(f"[smoke] exact-id scoring OK; batched==single (logp diff < 0.5); "
          f"grad norms {[round(float(g.norm()),3) for g in grads_b.values()]}")

    # --- check 3: synthetic reward (longer question => higher u) moves lengths ---
    def mean_len(texts):
        return sum(len(t.split()) for t in texts) / len(texts)
    len0 = mean_len(raw)
    for step in range(args.steps):
        raw, ids_list = s.generate(idxs)
        rewards = {i: min(1.0, len(raw[j].split()) / 200.0) for j, i in enumerate(idxs)}
        grads, _lp = s.score_grads(idxs, ids_list)
        st = s.step(idxs, rewards, grads)
        print(f"[smoke] step {step+1}: mean_len={mean_len(raw):.1f} "
              f"adv={st['mean_adv']:+.3f} |g/τ|={st['mean_gnorm']:.2e} z_rms={st['latent_rms']:.3f}")
    raw, _ = s.generate(idxs)
    len1 = mean_len(raw)
    print(f"[smoke] mean generated length: {len0:.1f} -> {len1:.1f} "
          f"({'MOVED' if len1 > len0 * 1.05 else 'no clear movement — inspect eta/tau'})")
    print("[smoke] PASSED (structural checks); distribution movement is advisory.")


if __name__ == "__main__":
    main()

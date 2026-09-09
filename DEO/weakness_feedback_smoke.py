"""Local-weakness-feedback smoke (LOCAL_WEAKNESS_FEEDBACK_IMPLEMENTATION.md §7).

Two walks over the SAME 100 initial questions with a FIXED solver and NO training:
  arm A: DEO_WM_GUIDANCE_MODE=global_fixed   (iter1, no prior memory -> unguided)
  arm B: DEO_WM_GUIDANCE_MODE=local_feedback (local notes guide from step 1)
Each arm re-scores the shared questions with its own fresh rollouts (the warm-start
path's semantics); texts/trajectories are NOT claimed identical — the comparison is
mechanism stats, not paired outcomes. Produces smoke_report_feedback.json with
per-source proposal/acceptance stats, machine checks, and up to 50 seeded audit
samples of LOCAL-guided ACCEPTED steps (old question -> frozen guidance -> new
question) for the human semantic review.
"""
import json
import glob
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcmc_deo_vllm as deo  # noqa: E402

deo.config.STORAGE_ROOT = os.environ["STORAGE_PATH"]
deo.config.TOTAL_QUESTIONS = int(os.environ.get("DEO_TOTAL_Q", "100"))
assert deo.config.WEAKNESS_MEMORY_ENABLED, "set DEO_WEAKNESS_MEMORY=1"
assert not deo.config.CD_ENABLE and not deo.config.BANDIT_ENABLE


def main():
    for sub in ["datasets", "logs"]:
        os.makedirs(f"{deo.config.STORAGE_ROOT}/{sub}", exist_ok=True)
    tokenizer = deo.AutoTokenizer.from_pretrained(deo.config.MODEL_NAME,
                                                  trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    deo.wait_for_vllm_ready(deo.config.VLLM_BASE_URL, label="vllm-base")
    for url in deo.config.VLLM_SOLVER_URLS:
        deo.wait_for_vllm_ready(url, label=f"vllm-solver@{url}")

    # ---- shared initial questions: one scored init pool, zero walk steps ----
    steps = deo.config.MCMC_STEPS
    deo.config.MCMC_STEPS = 0
    deo.config.MODEL_ABBR = "wmfb_init"
    init_records = deo.generate_batch_mcmc(
        tokenizer, deo.config.TOTAL_QUESTIONS,
        f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_1_wmfb_init.log")
    deo.config.MCMC_STEPS = steps
    shared_pool = [{"question": d["question"], "gt": d["gt"], "topic": d["topic"]}
                   for d in init_records]
    print(f"[wmfb-smoke] shared initial pool: {len(shared_pool)} questions, "
          f"{steps} steps per arm", flush=True)

    report = {"n_questions": len(shared_pool), "steps": steps, "arms": {}}
    for mode in ("global_fixed", "local_feedback"):
        print(f"\n===== [wmfb-smoke] arm: {mode} =====", flush=True)
        deo.config.WM_GUIDANCE_MODE = mode
        deo.config.MODEL_ABBR = f"wmfb_{mode}"
        records = deo.generate_batch_mcmc(
            tokenizer, len(shared_pool),
            f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_1_wmfb_{mode}.log",
            init_pool=[dict(d) for d in shared_pool])
        with open(f"{deo.config.STORAGE_ROOT}/datasets/wmfb_{mode}_pool.json", "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        # per-source stats from the note rows of the NEWEST attempt files
        rows = []
        for fp in sorted(glob.glob(
                f"{deo.config.STORAGE_ROOT}/weakness_memory/weakness_notes_iter_1.a*.jsonl")):
            rows += [json.loads(l) for l in open(fp)]
        prows = [r for r in rows if r.get("guidance_mode") == mode
                 and r.get("stage", "").startswith("step")]
        by_src = Counter(r["guidance_source"] for r in prows)
        acc_src = Counter(r["guidance_source"] for r in prows if r["accepted"])
        arm = {
            "proposals_by_source": dict(by_src),
            "accepted_by_source": dict(acc_src),
            "final_notes": sum(1 for d in records if d.get("_weakness_note")),
            "in_band": sum(1 for d in records
                           if deo.config.MIN_SCORE <= d["p_hat"] <= deo.config.MAX_SCORE),
            "mean_p_hat": round(sum(d["p_hat"] for d in records) / len(records), 4),
        }
        if mode == "local_feedback":
            smp = random.Random(0)
            cand = [r for r in prows if r["guidance_source"] == "local" and r["accepted"]]
            arm["audit_samples"] = [
                {k: r.get(k) for k in ("chain", "stage", "old_q", "guidance_weakness",
                                       "guidance_evidence", "guidance_ref", "new_q",
                                       "p_hat_new", "note_status")}
                for r in smp.sample(cand, min(50, len(cand)))]
            arm["n_local_guided_accepted"] = len(cand)
        report["arms"][mode] = arm
        print(f"[wmfb-smoke] {mode}: "
              f"{json.dumps({k: v for k, v in arm.items() if k != 'audit_samples'}, ensure_ascii=False)}",
              flush=True)

    a, b = report["arms"]["global_fixed"], report["arms"]["local_feedback"]
    checks = {
        "global_arm_has_no_local_rows": a["proposals_by_source"].get("local", 0) == 0,
        "local_arm_used_local_guidance": b["proposals_by_source"].get("local", 0) > 0,
        "local_arm_has_accepted_local_steps": b.get("n_local_guided_accepted", 0) > 0,
        "notes_written_both_arms": a["final_notes"] > 0 and b["final_notes"] > 0,
        "audit_samples_have_snapshots": all(
            s["old_q"] and s["new_q"] and s["guidance_weakness"]
            for s in b.get("audit_samples", [])),
    }
    report["checks"] = checks
    out = f"{deo.config.STORAGE_ROOT}/weakness_memory/smoke_report_feedback.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[wmfb-smoke] checks: {checks}\n[wmfb-smoke] report -> {out}", flush=True)
    if not all(checks.values()):
        raise SystemExit("[wmfb-smoke] FAILED machine checks")
    print("[wmfb-smoke] machine checks PASSED — human review of audit_samples "
          "(old_q -> guidance -> new_q; exact_operation/domain_only/miss + "
          "unjudgeable_target + did-old-q-already-require-it) decides the full run",
          flush=True)


if __name__ == "__main__":
    main()

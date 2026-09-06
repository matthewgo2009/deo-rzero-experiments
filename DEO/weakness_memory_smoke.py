"""Weakness-memory v2 smoke run (WEAKNESS_MEMORY_FIXES.md §6.2).

~100 questions, the normal 5-step walk, TWO iterations with a FIXED solver and NO
training / NO HF push. Closes the loop: iter1 writes global memory -> iter2 loads it
and runs domain-routed guided mutations. Produces machine-checkable artifacts under
{STORAGE_PATH}/weakness_memory/ (notes, events, raw rollouts, global memory + audit)
plus smoke_report.json with the §6.2 acceptance numbers.
"""
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcmc_deo_vllm as deo  # noqa: E402

deo.config.STORAGE_ROOT = os.environ["STORAGE_PATH"]
deo.config.MODEL_ABBR = os.environ.get("DEO_ABBR", "deo_wm_smoke")
deo.config.TOTAL_QUESTIONS = int(os.environ.get("DEO_TOTAL_Q", "100"))
assert deo.config.WEAKNESS_MEMORY_ENABLED, "set DEO_WEAKNESS_MEMORY=1 for the smoke run"
assert not deo.config.CD_ENABLE, "weakness memory smoke covers the non-CD walk only"


def stage_filter(train_data):
    """The complete filter (pseudo + p_hat band + judge) WITHOUT the HF push."""
    stage1 = [d for d in train_data
              if d["pseudo_label"] not in (None, "", "None")
              and deo.config.MIN_SCORE <= d["p_hat"] <= deo.config.MAX_SCORE]
    if not stage1:
        return []
    with ThreadPoolExecutor(max_workers=16) as ex:
        ok = list(ex.map(
            lambda d: bool(deo.judge_one_validity(d["question"], d["pseudo_label"])),
            stage1))
    return [d for d, o in zip(stage1, ok) if o]


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

    report = {"n_questions": deo.config.TOTAL_QUESTIONS, "iters": {}}
    for it in (1, 2):
        print(f"\n===== [wm-smoke] iteration {it} =====", flush=True)
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.log"
        records = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.json",
                  "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        filtered = stage_filter(records)
        mem = deo.summarize_global_weakness_memory(tokenizer, filtered, it)

        guided = [d for d in records if d.get("_target_memory_id")]
        rep = {
            "pool": len(records),
            "post_filter": len(filtered),
            "final_notes": sum(1 for d in records if d.get("_weakness_note")),
            "guidance_reasons": dict(Counter(d.get("_guidance_reason") for d in records)),
            "seed_domains": dict(Counter(d.get("_seed_domain") for d in records)),
            "global_items": [{"id": m["id"], "domain": m["domain"],
                              "support": m["support"], "weakness": m["weakness"]}
                             for m in mem],
            "guided_examples": [
                {"chain": d["_chain_id"], "target": d["_target_memory_id"],
                 "target_source_iter": d["_target_source_iter"],
                 "seed_domain": d["_seed_domain"], "p_hat": d["p_hat"],
                 "question": d["question"][:400]}
                for d in guided[:12]],
        }
        report["iters"][it] = rep
        print(f"[wm-smoke] iter{it}: {json.dumps({k: v for k, v in rep.items() if k != 'guided_examples'}, ensure_ascii=False)}",
              flush=True)

    # §6.2 acceptance gates (machine-checkable half; on-target question quality is a
    # human read of guided_examples + the notes/audit files)
    it2 = report["iters"][2]
    checks = {
        "iter1_wrote_memory": len(report["iters"][1]["global_items"]) > 0,
        "iter2_loaded_and_guided": it2["guidance_reasons"].get("guided", 0) > 0,
        "iter2_targets_from_iter1": all(
            g["target_source_iter"] == 1 for g in it2["guided_examples"]),
        "notes_present_both_iters": all(
            report["iters"][t]["final_notes"] > 0 for t in (1, 2)),
    }
    report["checks"] = checks
    out = f"{deo.config.STORAGE_ROOT}/weakness_memory/smoke_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[wm-smoke] checks: {checks}", flush=True)
    print(f"[wm-smoke] report -> {out}", flush=True)
    if not all(checks.values()):
        raise SystemExit("[wm-smoke] FAILED machine checks — see smoke_report.json")
    print("[wm-smoke] machine checks PASSED (human review of guided_examples + audit "
          "files still required before a full run)", flush=True)


if __name__ == "__main__":
    main()

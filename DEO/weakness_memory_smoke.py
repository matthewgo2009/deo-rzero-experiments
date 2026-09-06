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
    prev_mem = []
    for it in (1, 2):
        print(f"\n===== [wm-smoke] iteration {it} =====", flush=True)
        log_path = f"{deo.config.STORAGE_ROOT}/logs/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.log"
        records = deo.generate_batch_mcmc(tokenizer, deo.config.TOTAL_QUESTIONS, log_path)
        with open(f"{deo.config.STORAGE_ROOT}/datasets/mcmc_iter_{it}_{deo.config.MODEL_ABBR}.json",
                  "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        filtered = stage_filter(records)
        _lg = deo.WmLogger(it)   # capture per-merge-call raw outputs in the smoke too
        try:
            mem = deo.summarize_global_weakness_memory(tokenizer, filtered, it, logger=_lg)
        finally:
            _lg.close()

        guided = [d for d in records if d.get("_target_memory_id")]
        mem_by_id = {m["id"]: m for m in (prev_mem or [])}
        import random as _rnd
        smp = _rnd.Random(0)   # fixed sampling seed (review §6 requirement)
        mutated = [d for d in guided if d.get("_n_accepted", 0) > 0]
        pristine = [d for d in guided if d.get("_n_accepted", 0) == 0]

        def ex(d):
            tgt = mem_by_id.get(d["_target_memory_id"], {})
            return {"chain": d["_chain_id"], "target": d["_target_memory_id"],
                    "target_weakness": tgt.get("weakness"),
                    "target_source_iter": d["_target_source_iter"],
                    "seed_domain": d["_seed_domain"], "p_hat": d["p_hat"],
                    "n_accepted": d.get("_n_accepted", 0),
                    "question": d["question"][:400]}

        rep = {
            "pool": len(records),
            "post_filter": len(filtered),
            "final_notes": sum(1 for d in records if d.get("_weakness_note")),
            "guidance_reasons": dict(Counter(d.get("_guidance_reason") for d in records)),
            "seed_domains": dict(Counter(d.get("_seed_domain") for d in records)),
            "n_guided_mutated": len(mutated),
            "n_guided_pristine_seed": len(pristine),
            "global_items": [{"id": m["id"], "domain": m["domain"],
                              "support": m["support"], "rep_method": m.get("rep_method"),
                              "weakness": m["weakness"]}
                             for m in mem],
            "guided_examples_mutated": [ex(d) for d in smp.sample(mutated, min(10, len(mutated)))],
            "guided_examples_pristine": [ex(d) for d in smp.sample(pristine, min(4, len(pristine)))],
        }
        # ---- full-record machine checks (not just the displayed examples) ----
        rep["all_guided_target_iter_ok"] = all(
            d["_target_source_iter"] == it - 1 for d in guided)
        eids = [d.get("_wm_event_id") for d in records if d.get("_wm_event_id")]
        rep["event_ids_unique"] = len(eids) == len(set(eids))
        # recompute every kept item's support/avg_p_hat from the filtered records
        audit_ok = True
        p_by_chain = {d["_chain_id"]: float(d["p_hat"]) for d in filtered}
        for m in mem:
            chains = m.get("source_chains", [])
            if m["support"] != len(set(chains)):
                audit_ok = False
            known = [p_by_chain[c] for c in chains if c in p_by_chain]
            if len(known) == len(chains) and chains:
                if abs(sum(known) / len(known) - m["avg_p_hat"]) > 1e-3:
                    audit_ok = False
            if m.get("representative_event_id") not in (m.get("source_event_ids") or [None]):
                audit_ok = False
        rep["memory_stats_recompute_ok"] = audit_ok
        report["iters"][it] = rep
        print(f"[wm-smoke] iter{it}: "
              f"{json.dumps({k: v for k, v in rep.items() if not k.startswith('guided_examples')}, ensure_ascii=False)}",
              flush=True)
        prev_mem = mem

    # §6.2 acceptance gates (machine-checkable half; on-target question quality is a
    # human read of guided_examples + the notes/audit files)
    it2 = report["iters"][2]
    checks = {
        "iter1_wrote_memory": len(report["iters"][1]["global_items"]) > 0,
        "iter2_loaded_and_guided": it2["guidance_reasons"].get("guided", 0) > 0,
        "all_guided_targets_from_iter1": it2["all_guided_target_iter_ok"],
        "event_ids_unique_both_iters": all(
            report["iters"][t]["event_ids_unique"] for t in (1, 2)),
        "memory_stats_recompute_ok": all(
            report["iters"][t]["memory_stats_recompute_ok"] for t in (1, 2)),
        "notes_present_both_iters": all(
            report["iters"][t]["final_notes"] > 0 for t in (1, 2)),
        "some_guided_chain_mutated": it2["n_guided_mutated"] > 0,
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

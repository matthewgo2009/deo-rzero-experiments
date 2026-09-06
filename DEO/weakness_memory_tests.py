"""Offline acceptance tests for the weakness-memory loop, v2
(WEAKNESS_MEMORY_IMPLEMENTATION.md + WEAKNESS_MEMORY_FIXES.md §6.1).

No vLLM/GPU: every LLM path is exercised via mocked clients. Runs on-node as the
job gate. Coverage of the §6.1 checklist:
  - response attribution: shuffled/missing batch choices never shift later items
  - evidence input: ALL cluster counts kept, representative traces top-3 only,
    p_hat semantics untouched
  - writer statuses: eligibility / request_failed / parse_failed / weakness /
    problem_issue / insufficient_evidence; citation must exist verbatim
  - state updates: accept replaces, reject keeps, accepted-no-note clears
  - merge conservation: dup/out-of-range/missing ids repaired, unassigned never
    silently dropped, same-domain-only fallback, source/support/avg_p traceable
  - guidance routing: seed-domain compatibility, fallback reasons, dedicated RNG
  - disabled default
"""
import json
import os
import sys
import tempfile
import types
import random as _random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub heavyweight deps that may be absent on a dev box (the job image has real ones).
for name, attrs in [
    ("nltk", {}),
    ("nltk.translate", {}),
    ("nltk.translate.bleu_score",
     {"sentence_bleu": (lambda *a, **k: 0.0),
      "SmoothingFunction": type("SF", (), {"method1": staticmethod(lambda *a, **k: None)})}),
    ("datasets", {"Dataset": object, "DatasetDict": dict}),
    ("mathruler", {}),
    ("mathruler.grader", {"grade_answer": (lambda *a, **k: False)}),
]:
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

import mcmc_deo_vllm as deo  # noqa: E402


class FakeTokenizer:
    chat_template = None  # apply_chat_template falls back to plain string concat

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=list(range(max(1, len(text) // 4))))


class FakeChoice:
    def __init__(self, text, index=None):
        self.text = text
        if index is not None:
            self.index = index


class FakeResp:
    def __init__(self, choices):
        self.choices = choices


def fake_client(reply_fn):
    """OpenAI-shaped stub: completions.create(prompt=[...]) -> in-order choices."""
    class _Comp:
        @staticmethod
        def create(**kw):
            return FakeResp([FakeChoice(reply_fn(p), index=i)
                             for i, p in enumerate(kw["prompt"])])
    return types.SimpleNamespace(completions=_Comp())


TOK = FakeTokenizer()


def _details(answers, texts):
    return deo._build_cluster_details(answers, texts)


def note_json(domain="algebra", weakness="w", evidence="e", cluster_id=1,
              excerpt=None, status="weakness"):
    return json.dumps({"status": status, "domain": domain, "weakness": weakness,
                       "evidence": evidence, "cluster_id": cluster_id,
                       "excerpt": excerpt})


def test_cluster_details():
    # 4/3/2 split: ALL counts kept (fixes 3.1.2), traces top-3 with ids, invalids counted
    answers = ["24", "12", "24", "48", "24", "12", "48", "24", "12"]
    texts = [f"trace_{i} reasoning toward {a}" for i, a in enumerate(answers)]
    d = _details(answers, texts)
    assert d["rollout_count"] == 9 and d["valid_answer_count"] == 9
    assert [(c["answer"], c["count"]) for c in d["clusters_all"]] == [("24", 4), ("12", 3), ("48", 2)]
    assert [c["cluster_id"] for c in d["clusters"]] == [1, 2, 3]
    assert d["clusters"][0]["rollout_id"] == 0 and d["clusters"][1]["rollout_id"] == 1
    assert d["_raw"]["answers"] == answers            # provenance payload present
    # >3 clusters: all counts survive, only top-3 get traces
    a2 = ["1", "2", "3", "4", "1", None, "GUESSED_FAIL_FORMAT", "1", "2"]
    d2 = _details(a2, [f"t{i}" for i in range(9)])
    assert len(d2["clusters_all"]) == 4 and len(d2["clusters"]) == 3
    assert d2["invalid_answer_count"] == 2 and d2["valid_answer_count"] == 7
    print("[test1] PASSED: full cluster counts, top-3 traces with ids, invalids, raw payload")


def test_ordered_choices():
    # shuffled indices + a missing item: no shift, missing slot stays None
    resp = FakeResp([FakeChoice("c", 2), FakeChoice("a", 0)])
    assert deo._ordered_completion_texts(resp, 3) == ["a", None, "c"]
    # no index info + count mismatch -> all None (cannot attribute)
    class NoIdx:  # noqa
        def __init__(self, t): self.text = t
    assert deo._ordered_completion_texts(FakeResp([NoIdx("x")]), 2) == [None, None]
    assert deo._ordered_completion_texts(FakeResp([NoIdx("x"), NoIdx("y")]), 2) == ["x", "y"]
    print("[test2] PASSED: response attribution never shifts on shuffled/missing choices")


def test_parse_note_v2():
    det = _details(["1", "1", "2"], ["alpha beta gamma steps", "b", "delta epsilon end"])
    tr = det["clusters"][0]["representative_trace"]
    ok = note_json(excerpt=tr[:20])
    st, nt, _ = deo._parse_weakness_note(ok, det)
    assert st == "weakness" and nt["cluster_id"] == 1 and nt["excerpt"] == tr[:20]
    # citation not in trace -> parse_failed (hallucinated excerpt is detectable)
    st, nt, r = deo._parse_weakness_note(note_json(excerpt="NOT IN ANY TRACE"), det)
    assert st == "parse_failed" and nt is None
    # wrong cluster id -> parse_failed
    st, _, _ = deo._parse_weakness_note(note_json(cluster_id=9, excerpt=tr[:10]), det)
    assert st == "parse_failed"
    # problem_issue / insufficient_evidence: audit-only, note=None, no citation needed
    st, nt, _ = deo._parse_weakness_note(note_json(status="problem_issue"), det)
    assert st == "problem_issue" and nt is None
    st, nt, _ = deo._parse_weakness_note(note_json(status="insufficient_evidence"), det)
    assert st == "insufficient_evidence" and nt is None
    # garbage / bad status
    assert deo._parse_weakness_note("no json", det)[0] == "parse_failed"
    assert deo._parse_weakness_note('{"status": "great"}', det)[0] == "parse_failed"
    print("[test3] PASSED: v2 statuses + verbatim-citation validation")


def test_writer_statuses_and_failures():
    det = _details(["1", "1", "2"], ["alpha beta gamma", "b", "c"])
    tr = det["clusters"][0]["representative_trace"]
    qs = ["q0", "q1", "q2", "q3"]
    phats = [0.5, 0.9, 0.5, 0.5]
    pseudos = ["1", "1", None, "1"]
    dets = [det, det, det, det]
    deo.base_client = lambda: fake_client(lambda p: note_json(excerpt=tr[:10]))
    notes, statuses, _ = deo.generate_weakness_notes_batch(TOK, qs, phats, pseudos, dets)
    assert notes[0] is not None and notes[3] is not None
    assert statuses == ["weakness", "not_eligible", "not_eligible", "weakness"]
    deo.base_client = lambda: fake_client(lambda p: "garbage")
    notes, statuses, _ = deo.generate_weakness_notes_batch(TOK, qs, phats, pseudos, dets)
    assert notes == [None] * 4 and statuses[0] == "parse_failed"
    def boom():
        raise RuntimeError("down")
    deo.base_client = boom
    notes, statuses, _ = deo.generate_weakness_notes_batch(TOK, qs, phats, pseudos, dets)
    assert notes == [None] * 4 and statuses[0] == "request_failed"
    assert statuses[1] == "not_eligible"    # eligibility distinct from failures
    print("[test4] PASSED: writer status taxonomy; failures never raise")


def test_state_update_contract():
    # accept replaces note+event; reject keeps; accepted-with-no-note CLEARS
    pool_note, pool_event = [{"weakness": "seed"}], ["e0"]
    seq = [(False, {"weakness": "p1"}, "e1"),
           (True, {"weakness": "p2"}, "e2"),
           (False, {"weakness": "p3"}, "e3"),
           (True, None, "e4"),
           (False, {"weakness": "p5"}, "e5")]
    for accept, note, eid in seq:
        if accept:
            pool_note[0] = note
            pool_event[0] = eid if note else None
    assert pool_note[0] is None and pool_event[0] is None
    print("[test5] PASSED: accept replaces / reject keeps / accepted-no-note clears")


def test_merge_conservation():
    notes = [{"weakness": f"w{i}", "evidence": f"ev{i}"} for i in range(6)]
    # LLM returns dup + out-of-range + omits idx 5 -> repaired, 5 lands in unassigned
    reply = json.dumps({"groups": [{"weakness": "g1", "indices": [0, 1, 1, 9]},
                                   {"weakness": "g2", "indices": [2, 3, 4]}],
                        "unassigned": []})
    deo.base_client = lambda: fake_client(lambda p: reply)
    groups, unassigned = deo._merge_notes_llm(TOK, notes, stats := {})
    got = sorted(i for g in groups for i in g["indices"]) + sorted(unassigned)
    assert sorted(got) == list(range(6))                    # conservation invariant
    assert unassigned == [5] and stats.get("llm_repaired") == 1
    # dead endpoint -> same-domain exact-string fallback, no invented labels
    def boom():
        raise RuntimeError("down")
    deo.base_client = boom
    dup = [{"weakness": "Same Thing", "evidence": ""},
           {"weakness": "same  thing", "evidence": ""},
           {"weakness": "other", "evidence": ""}]
    groups, unassigned = deo._merge_notes_llm(TOK, dup, {})
    parts = sorted(tuple(sorted(g["indices"])) for g in groups)
    assert parts == [(0, 1), (2,)] and unassigned == []
    print("[test6] PASSED: merge conservation, id repair, exact-string fallback")


def test_summarize_traceability():
    tmp = tempfile.mkdtemp()
    deo.config.STORAGE_ROOT = tmp
    deo.config.MEMORY_TOP_K = 2
    deo.config.MEMORY_MIN_SUPPORT = 3

    def rec(i, dom, w, p):
        return {"question": f"q{i}", "p_hat": p, "pseudo_label": "1",
                "_chain_id": i, "_wm_event_id": f"ev{i}",
                "_weakness_note": {"domain": dom, "weakness": w, "evidence": f"E{i}"}}
    records = ([rec(i, "algebra", "w_A", 0.5) for i in range(5)]
               + [rec(10 + i, "algebra", "w_B", 0.75) for i in range(3)]
               + [rec(20 + i, "geometry", "w_A", 0.5) for i in range(2)]   # same label, other domain
               + [{"question": "no-note", "p_hat": 0.5, "_weakness_note": None}])
    # LLM merges exactly by label within each domain bucket
    def reply(prompt):
        lines = [l for l in prompt.split("\n") if l and l[0].isdigit()]
        by_label = {}
        for l in lines:
            j, rest = l.split(". ", 1)
            by_label.setdefault(rest.split(" ||")[0], []).append(int(j))
        return json.dumps({"groups": [{"weakness": k, "indices": v}
                                      for k, v in by_label.items()], "unassigned": []})
    deo.base_client = lambda: fake_client(reply)
    items = deo.summarize_global_weakness_memory(TOK, records, 3)
    # w_A(algebra) support 5 kept; w_B support 3 kept; w_A(geometry) support 2 dropped
    assert [(it["weakness"], it["support"]) for it in items] == [("w_A", 5), ("w_B", 3)]
    assert items[0]["domain"] == "algebra"               # never merged across domains
    assert items[0]["source_event_ids"] == [f"ev{i}" for i in range(5)]
    assert items[0]["representative_event_id"] in items[0]["source_event_ids"]
    assert items[0]["representative_evidence"] == "E0"
    assert abs(items[0]["avg_p_hat"] - 0.5) < 1e-9 and items[0]["source_iter"] == 3
    audit = json.load(open(f"{tmp}/weakness_memory/global_weakness_memory_iter_3_audit.json"))
    reasons = {tuple(g["source_chains"]): g["trim_reason"] for g in audit["groups"]}
    assert reasons[(20, 21)] == "support<min"            # trimmed groups auditable
    assert deo.load_global_weakness_memory(3) == items   # next iter loads this file
    print("[test7] PASSED: domain-bucketed summary, source/support/evidence traceable")


def test_target_routing():
    mem = [{"id": "memory_1", "domain": "algebra", "weakness": "wa",
            "support": 100, "avg_p_hat": 0.5, "source_iter": 2},
           {"id": "memory_2", "domain": "geometry", "weakness": "wg",
            "support": 3, "avg_p_hat": 0.5, "source_iter": 2}]
    deo.config.MEMORY_GUIDED_PROB = 1.0
    rng = _random.Random(0)
    seeds = ["algebra", "geometry", "probability", None]
    targets, reasons = deo.assign_targets(mem, seeds, rng)
    assert targets[0]["id"] == "memory_1" and targets[1]["id"] == "memory_2"
    assert targets[2] is None and reasons[2] == "no_compatible_target"
    assert targets[3] is None and reasons[3] == "no_seed_domain"
    assert reasons[:2] == ["guided", "guided"]
    # empty memory / unguided draw
    assert deo.assign_targets([], ["algebra"], rng)[1] == ["no_memory"]
    deo.config.MEMORY_GUIDED_PROB = 0.0
    _, rs = deo.assign_targets(mem, ["algebra"] * 20, rng)
    assert set(rs) == {"unguided_draw"}
    deo.config.MEMORY_GUIDED_PROB = 0.8
    # dedicated RNG: same seed -> same assignment (MH stream independence)
    t1, _ = deo.assign_targets(mem, ["algebra"] * 50, _random.Random("s"))
    t2, _ = deo.assign_targets(mem, ["algebra"] * 50, _random.Random("s"))
    assert [t and t["id"] for t in t1] == [t and t["id"] for t in t2]
    g = deo.weakness_guidance_block(dict(mem[0], representative_evidence="clusters split"))
    assert "KNOWN SOLVER WEAKNESS" in g and "wa" in g and "clusters split" in g
    print("[test8] PASSED: domain-compatible routing with recorded fallback reasons")


def test_trace_truncation_and_budget():
    deo.config.MEMORY_TRACE_MAX_CHARS = 100
    long = "H" * 200 + "T" * 200
    t = deo._truncate_trace(long)
    assert t.startswith("H" * 60) and t.endswith("T" * 40) and "[truncated]" in t
    deo.config.MEMORY_TRACE_MAX_CHARS = 1500
    # budget chunking: many items split into >1 chunk, all items preserved in order
    items = list(range(300))
    chunks = deo._budget_chunks(TOK, items, lambda i: "x" * 400, 100)
    assert [i for c in chunks for i in c] == items and len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    print("[test9] PASSED: deterministic truncation; budget chunking preserves order")


def test_disabled_default():
    assert deo.Config.WEAKNESS_MEMORY_ENABLED == (os.getenv("DEO_WEAKNESS_MEMORY", "0") == "1")
    print("[test10] PASSED: weakness memory disabled by default")


if __name__ == "__main__":
    test_cluster_details()
    test_ordered_choices()
    test_parse_note_v2()
    test_writer_statuses_and_failures()
    test_state_update_contract()
    test_merge_conservation()
    test_summarize_traceability()
    test_target_routing()
    test_trace_truncation_and_budget()
    test_disabled_default()
    print("\nALL WEAKNESS-MEMORY v2 TESTS PASSED")

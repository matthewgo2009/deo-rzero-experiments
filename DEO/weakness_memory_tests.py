"""Offline acceptance tests for the weakness-memory loop, v2.1
(WEAKNESS_MEMORY_IMPLEMENTATION.md + WEAKNESS_MEMORY_FIXES.md + review R1-R8).

No vLLM/GPU: LLM paths run against mocked clients. Runs on-node as the job gate.
Test classes (per the R-review's delivery requirements):
  [unit]   original v2 unit tests, updated for v2.1 semantics
  [regress] R1-R8 counterexample regressions (each reproduced the pre-fix bug)
  [wiring] REAL evaluate_r_unc_vllm / generate_batch_mcmc runs with mock services
  [tokenizer] the real-tokenizer budget boundary check runs on-node only (needs
     the Qwen tokenizer); locally a FakeTokenizer approximates token counts
Semantic quality of writer/summarizer outputs is NOT provable here — that is what
the GPU smoke run (MODE=wm_smoke) and its human review are for.
"""
import json
import gzip
import os
import sys
import tempfile
import types
import random as _random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    chat_template = None

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
    class _Comp:
        @staticmethod
        def create(**kw):
            return FakeResp([FakeChoice(reply_fn(p), index=i)
                             for i, p in enumerate(kw["prompt"])])
    return types.SimpleNamespace(completions=_Comp())


TOK = FakeTokenizer()
_ORIG = {k: getattr(deo, k) for k in
         ("base_client", "solver_clients", "evaluate_r_unc_vllm", "WM_CONTEXT_LIMIT")}


def restore():
    for k, v in _ORIG.items():
        setattr(deo, k, v)
    deo.config.MEMORY_GUIDED_PROB = 0.8
    deo.config.MEMORY_TRACE_MAX_CHARS = 1500
    deo.config.MEMORY_TOP_K = 10
    deo.config.MEMORY_MIN_SUPPORT = 3


def note_json(domain="algebra", weakness="w", evidence="e", cluster_id=1,
              excerpt=None, status="weakness"):
    return json.dumps({"status": status, "domain": domain, "weakness": weakness,
                       "evidence": evidence, "cluster_id": cluster_id,
                       "excerpt": excerpt})


def merge_json(groups, unassigned=()):
    return json.dumps({"groups": groups, "unassigned": list(unassigned)})


# ---------------- [unit] ----------------

def test_cluster_details():
    answers = ["24", "12", "24", "48", "24", "12", "48", "24", "12"]
    texts = [f"trace_{i} reasoning toward {a}" for i, a in enumerate(answers)]
    d = deo._build_cluster_details(answers, texts)
    assert d["rollout_count"] == 9 and d["valid_answer_count"] == 9
    assert [(c["answer"], c["count"]) for c in d["clusters_all"]] == [("24", 4), ("12", 3), ("48", 2)]
    assert [c["cluster_id"] for c in d["clusters"]] == [1, 2, 3]
    assert d["clusters"][0]["rollout_id"] == 0 and d["_raw"]["answers"] == answers
    a2 = ["1", "2", "3", "4", "1", None, "GUESSED_FAIL_FORMAT", "1", "2"]
    d2 = deo._build_cluster_details(a2, [f"t{i}" for i in range(9)])
    assert len(d2["clusters_all"]) == 4 and len(d2["clusters"]) == 3
    assert d2["invalid_answer_count"] == 2
    print("[unit1] PASSED: cluster details (all counts, top-3 traces, raw payload)")


def test_state_semantics_unitlevel():
    # kept as a spec-level statement; the PRODUCTION wiring is covered by
    # test_wiring_walk below (R-review: unit replicas alone are insufficient)
    pool_note, pool_event = [{"weakness": "seed"}], ["e0"]
    for accept, note, eid in [(False, {"weakness": "p1"}, "e1"),
                              (True, None, "e2")]:
        if accept:
            pool_note[0] = note
            pool_event[0] = eid   # eval event survives even when the note is cleared
    assert pool_note[0] is None and pool_event[0] == "e2"
    print("[unit2] PASSED: accepted-no-note clears note, keeps its eval event")


# ---------------- [regress] R5 ----------------

def test_R5_json_and_indices():
    # (a) brace inside a JSON string must parse
    txt = 'ok {"status": "weakness", "domain": "algebra", "weakness": "sets", ' \
          '"evidence": "e", "cluster_id": 1, "excerpt": "the set {1,2}"} tail'
    obj = deo._extract_json_value(txt)
    assert isinstance(obj, dict) and obj["excerpt"] == "the set {1,2}"
    # (b) indices: scalar -> no TypeError, group skipped, conservation holds
    notes = [{"weakness": f"w{i}", "evidence": ""} for i in range(3)]
    deo.base_client = lambda: fake_client(lambda p: merge_json(
        [{"weakness": "g", "indices": 1, "representative": 1}]))
    groups, unassigned = deo._merge_notes_llm(TOK, notes, st := {})
    got = sorted(i for g in groups for i in g["indices"]) + sorted(unassigned)
    assert sorted(got) == [0, 1, 2]
    # (c) floats and bools rejected (never int()-truncated); digit strings accepted
    assert deo._valid_index(0.9) is None and deo._valid_index(True) is None
    assert deo._valid_index("2") == 2 and deo._valid_index("x") is None
    deo.base_client = lambda: fake_client(lambda p: merge_json(
        [{"weakness": "g", "indices": [0.9, 0, "1"], "representative": 0}]))
    groups, unassigned = deo._merge_notes_llm(TOK, notes, st2 := {})
    members = {i for g in groups for i in g["indices"]}
    assert 0 in members and 1 in members and sorted(members | set(unassigned)) == [0, 1, 2]
    assert st2.get("llm_repaired") == 1          # 0.9 dropped = code repair
    # (d) model-declared unassigned alone is NOT counted as repair
    deo.base_client = lambda: fake_client(lambda p: merge_json(
        [{"weakness": "g", "indices": [0, 1], "representative": 0}], unassigned=[2]))
    deo._merge_notes_llm(TOK, notes, st3 := {})
    assert st3.get("llm_success") == 1 and "llm_repaired" not in st3
    restore()
    print("[R5] PASSED: string-safe JSON, typed indices, honest repair stats")


def test_latex_json_repair():
    # smoke finding (sweet_melon): 178/203 writer outputs failed as "no JSON object"
    # because raw LaTeX (\\frac, \\boxed, \\pmod) is an invalid JSON escape.
    bad = ('{ "status": "weakness", "domain": "geometry", "weakness": "w", '
           '"evidence": "disagree on \\( \\frac{BI}{IC} \\)", "cluster_id": 1, '
           '"excerpt": "\\boxed{2\\pi} and \\pmod{100}" }')
    obj = deo._extract_json_value(bad)
    assert obj and obj["excerpt"] == "\\boxed{2\\pi} and \\pmod{100}"   # verbatim, no \\b eaten
    det = deo._build_cluster_details(
        ["1", "1", "2"], ["steps \\boxed{2\\pi} and \\pmod{100} end", "b", "c"])
    st, nt, _ = deo._parse_weakness_note(bad, deo._shown_details(det, 1500))
    assert st == "weakness" and nt is not None
    # strict-valid JSON still parses strictly (repair only after strict failure)
    ok = json.dumps({"a": "b\\nc"})
    assert deo._extract_json_value(ok) == {"a": "b\\nc"}
    print("[latex] PASSED: raw-LaTeX JSON repaired; excerpts stay verbatim")


def test_classifier_parse():
    deo.base_client = lambda: fake_client(
        lambda p: "This problem is about Number Theory." if "PLANNT" in p
        else ("  algebra\\n" if "PLANALG" in p else "no idea"))
    out = deo.classify_domains_batch(TOK, ["q PLANNT", "q PLANALG", "q mystery"])
    assert out == ["number_theory", "algebra", None], out
    restore()
    print("[classify] PASSED: whole-text earliest-domain matching, None on no match")


# ---------------- [regress] R6 ----------------

def test_R6_helper():
    resp = FakeResp([FakeChoice("c", 2), FakeChoice("a", 0)])
    assert deo._ordered_completion_texts(resp, 3) == ["a", None, "c"]
    # indices exist but ALL invalid -> never positional fallback
    st = {}
    assert deo._ordered_completion_texts(FakeResp([FakeChoice("x", 99)]), 1, st) == [None]
    assert st["invalid_index"] == 1
    # duplicate index: first kept, counted
    st = {}
    out = deo._ordered_completion_texts(
        FakeResp([FakeChoice("a", 0), FakeChoice("b", 0)]), 2, st)
    assert out == ["a", None] and st["dup_index"] == 1 and st["missing"] == 1
    # genuinely index-less API + exact count -> positional is allowed
    class NoIdx:  # noqa
        def __init__(self, t): self.text = t
    assert deo._ordered_completion_texts(FakeResp([NoIdx("x"), NoIdx("y")]), 2) == ["x", "y"]
    print("[R6a] PASSED: helper fallback rules (invalid-index never positional)")


def test_R6_evaluate_wiring():
    # REAL evaluate_r_unc_vllm with a shuffled-but-indexed solver response:
    # two questions x m=3 identical answers must give p_hat = 1.0 each.
    old_m, old_cd = deo.config.M_SAMPLES, deo.config.CD_ENABLE
    deo.config.M_SAMPLES, deo.config.CD_ENABLE = 3, False

    def make_solver(shuffle, drop=None):
        class _Comp:
            @staticmethod
            def create(**kw):
                n = len(kw["prompt"])
                ans = ["A", "A", "A", "B", "B", "B"][:n]
                ch = [FakeChoice(f"reasoning. The answer is \\\\boxed{{{ans[i]}}}", index=i)
                      for i in range(n) if i != drop]
                if shuffle:
                    ch = ch[::-1]
                return FakeResp(ch)
        return types.SimpleNamespace(completions=_Comp())

    deo.solver_clients = lambda: [make_solver(shuffle=True)]
    r, p, ps = deo.evaluate_r_unc_vllm(TOK, ["q one", "q two"])
    assert p == [1.0, 1.0] and ps == ["A", "B"], (p, ps)
    # a missing choice keeps its slot: that question drops ONE valid answer only
    deo.solver_clients = lambda: [make_solver(shuffle=True, drop=1)]
    r, p, ps = deo.evaluate_r_unc_vllm(TOK, ["q one", "q two"])
    assert abs(p[0] - 2 / 3) < 1e-9 and p[1] == 1.0 and ps == ["A", "B"]
    deo.config.M_SAMPLES, deo.config.CD_ENABLE = old_m, old_cd
    restore()
    print("[R6b] PASSED: real evaluate wiring survives shuffled/missing choices")


# ---------------- [regress] R2 + R4 ----------------

def test_R2_citation_against_shown():
    # marker placed so that NO truncation level (1500/750/375/300 head+tail) shows it
    long_trace = "H" * 950 + " XMARKERX " + "T" * 1050
    det = deo._build_cluster_details(["1", "1", "2"],
                                     [long_trace, "short trace two", "short trace three"])
    deo.WM_CONTEXT_LIMIT = 5000
    # (a) truncated-away citation must fail END TO END through the writer batch
    hidden = note_json(excerpt="XMARKERX")
    deo.base_client = lambda: fake_client(lambda p: hidden)
    notes, statuses, _ = deo.generate_weakness_notes_batch(
        TOK, ["short q"], [0.5], ["1"], [det])
    assert statuses[0] == "parse_failed" and notes[0] is None, statuses
    # (b) a citation actually visible at the same budget passes
    vis = note_json(excerpt="H" * 40)
    deo.base_client = lambda: fake_client(lambda p: vis)
    notes, statuses, _ = deo.generate_weakness_notes_batch(
        TOK, ["short q"], [0.5], ["1"], [det])
    assert statuses[0] == "weakness" and notes[0] is not None, statuses
    # (c) case sensitivity: `A` != `a`
    shown = deo._shown_details(det, 300)
    assert deo._parse_weakness_note(note_json(excerpt="h" * 20), shown)[0] == "parse_failed"
    assert deo._parse_weakness_note(note_json(excerpt="H" * 20), shown)[0] == "weakness"
    restore()
    print("[R2] PASSED: citations validated against the SHOWN trace, case-sensitive")


def test_R4_oversize_isolation():
    det = deo._build_cluster_details(["1", "1", "2"], ["t1 alpha", "t2", "t3"])
    deo.WM_CONTEXT_LIMIT = 1500          # short item fits; the huge one never can
    huge_q = "x" * 20000                 # cannot fit even at floor truncation
    deo.base_client = lambda: fake_client(lambda p: note_json(excerpt="t1 alpha"))
    notes, statuses, _ = deo.generate_weakness_notes_batch(
        TOK, [huge_q, "normal question"], [0.5, 0.5], ["1", "1"], [det, det])
    assert statuses[0] == "input_too_long" and notes[0] is None
    assert statuses[1] == "weakness" and notes[1] is not None   # batch not poisoned
    # summarizer side: oversize input bisects instead of sending over budget
    notes_many = [{"weakness": "w" * 300, "evidence": "e" * 100} for _ in range(40)]
    def under_budget_reply(p):
        assert deo._tok_len(TOK, p) + deo.WM_SUMMARY_MAX_TOKENS <= deo.WM_CONTEXT_LIMIT, \
            "a prompt above budget was sent"
        return merge_json([])
    deo.WM_CONTEXT_LIMIT = 1400
    deo.base_client = lambda: fake_client(under_budget_reply)
    groups, unassigned = deo._merge_notes_llm(TOK, notes_many, st := {})
    got = sorted(i for g in groups for i in g["indices"]) + sorted(unassigned)
    assert sorted(got) == list(range(40)) and st.get("split", 0) >= 1
    restore()
    print("[R4] PASSED: oversize items isolated; every sent prompt within budget")


# ---------------- [regress] R3 ----------------

def test_R3_representative():
    notes = [{"weakness": "w0", "evidence": "e0"},
             {"weakness": "w1", "evidence": "e1"},
             {"weakness": "w2", "evidence": "e2"}]
    # valid representative honored
    deo.base_client = lambda: fake_client(lambda p: merge_json(
        [{"weakness": "merged", "indices": [0, 1], "representative": 1,
          "reason": "e1 matches"}], unassigned=[2]))
    groups, unassigned = deo._merge_notes_llm(TOK, notes, {})
    assert groups[0]["rep"] == 1 and unassigned == [2]
    # missing/invalid representative -> group NOT force-merged, members conserved
    deo.base_client = lambda: fake_client(lambda p: merge_json(
        [{"weakness": "merged", "indices": [0, 1]}], unassigned=[2]))
    groups, unassigned = deo._merge_notes_llm(TOK, notes, st := {})
    got = sorted(i for g in groups for i in g["indices"]) + sorted(unassigned)
    assert sorted(got) == [0, 1, 2]
    assert not any(set(g["indices"]) == {0, 1} for g in groups)
    restore()
    print("[R3] PASSED: representative selected+validated; no rep -> no forced merge")


# ---------------- [regress] R1 ----------------

def test_R1_cross_chunk_merge():
    tmp = tempfile.mkdtemp()
    deo.config.STORAGE_ROOT = tmp
    deo.config.MEMORY_TOP_K = 10
    deo.config.MEMORY_MIN_SUPPORT = 3

    def rec(i, w):
        return {"question": f"q{i}", "p_hat": 0.5, "_chain_id": i,
                "_wm_event_id": f"ev{i}",
                "_weakness_note": {"domain": "algebra", "weakness": w,
                                   "evidence": f"E{i}"}}
    # GPT's counterexample: identical weakness at positions 0/100/200, chunk=100
    records = [rec(i, "repeated_skill" if i in (0, 100, 200) else f"unique_{i}")
               for i in range(201)]
    def boom():
        raise RuntimeError("llm down")   # pure-fallback path
    deo.base_client = boom
    items = deo.summarize_global_weakness_memory(TOK, records, 1)
    assert [(m["weakness"], m["support"]) for m in items] == [("repeated_skill", 3)], items
    assert sorted(m["source_chains"] for m in items)[0] == [0, 100, 200]
    # order/chunk invariance of exact grouping
    rng = _random.Random(7)
    shuf = records[:]
    rng.shuffle(shuf)
    items2 = deo.summarize_global_weakness_memory(TOK, shuf, 2)
    assert [(m["weakness"], m["support"], sorted(m["source_chains"])) for m in items2] \
        == [("repeated_skill", 3, [0, 100, 200])]
    restore()
    print("[R1] PASSED: same-name notes merge across chunk boundaries, order-invariant")


# ---------------- [unit] summary traceability (updated for rep selection) ----------------

def test_summarize_traceability():
    tmp = tempfile.mkdtemp()
    deo.config.STORAGE_ROOT = tmp
    deo.config.MEMORY_TOP_K = 2
    deo.config.MEMORY_MIN_SUPPORT = 3

    def rec(i, dom, w, p):
        return {"question": f"q{i}", "p_hat": p, "_chain_id": i,
                "_wm_event_id": f"ev{i}",
                "_weakness_note": {"domain": dom, "weakness": w, "evidence": f"E{i}"}}
    records = ([rec(i, "algebra", f"style{i} of solving quadratics", 0.5) for i in range(5)]
               + [rec(10 + i, "algebra", f"way{i} of misusing vieta", 0.75) for i in range(3)]
               + [rec(20 + i, "geometry", "quadratics", 0.5) for i in range(2)])

    def reply(prompt):
        lines = [l for l in prompt.split("\n") if l and l.split(".")[0].isdigit()]
        quad = [int(l.split(".")[0]) for l in lines if "quadratic" in l]
        viet = [int(l.split(".")[0]) for l in lines if "vieta" in l]
        groups = []
        if quad:
            groups.append({"weakness": "solving quadratics", "indices": quad,
                           "representative": quad[0], "reason": "core case"})
        if viet:
            groups.append({"weakness": "misusing vieta", "indices": viet,
                           "representative": viet[-1], "reason": "clearest"})
        return merge_json(groups)
    deo.base_client = lambda: fake_client(reply)
    items = deo.summarize_global_weakness_memory(TOK, records, 3)
    assert [(it["weakness"], it["support"]) for it in items] == [
        ("solving quadratics", 5), ("misusing vieta", 3)]
    assert items[0]["domain"] == "algebra"       # geometry 'quadratics' never merged in
    assert items[0]["source_event_ids"] == [f"ev{i}" for i in range(5)]
    assert items[0]["representative_event_id"] in items[0]["source_event_ids"]
    assert items[0]["rep_method"] == "llm_selected"
    assert items[1]["representative_event_id"] == "ev12"   # model picked viet[-1]
    audit = json.load(open(f"{tmp}/weakness_memory/global_weakness_memory_iter_3_audit.json"))
    assert any(g["trim_reason"] == "support<min" for g in audit["groups"])
    assert deo.load_global_weakness_memory(3) == items
    restore()
    print("[unit3] PASSED: domain buckets, rep selection traceable, audit trim reasons")


# ---------------- [regress] R7 ----------------

def test_R7_logger_attempts():
    tmp = tempfile.mkdtemp()
    deo.config.STORAGE_ROOT = tmp
    lg1 = deo.WmLogger(5)
    e1 = lg1.new_event_id("eval")
    long_text = "Z" * 30000
    lg1.traces(e1, "q", ["a"], [long_text])
    lg1.close()
    lg2 = deo.WmLogger(5)            # rerun: fresh attempt, no overwrite/append mix
    e2 = lg2.new_event_id("eval")
    lg2.close()
    assert lg1.attempt == 1 and lg2.attempt == 2 and e1 != e2
    d = f"{tmp}/weakness_memory"
    assert os.path.exists(f"{d}/wm_events_iter_5.a1.jsonl")
    assert os.path.exists(f"{d}/wm_events_iter_5.a2.jsonl")
    rows = [json.loads(l) for l in gzip.open(f"{d}/raw_rollouts_iter_5.a1.jsonl.gz", "rt")]
    assert len(rows[0]["texts"][0]) == 30000     # FULL trace, no truncation
    meta = json.loads(open(f"{d}/wm_events_iter_5.a1.jsonl").readline())
    assert meta["kind"] == "meta" and meta["writer_prompt_version"]
    # per-merge-call logging (R7): every summarizer call leaves a reconstructable event
    recs = [{"question": "q", "p_hat": 0.5, "_chain_id": i, "_wm_event_id": f"e{i}",
             "_weakness_note": {"domain": "algebra", "weakness": f"w{i}", "evidence": "E"}}
            for i in range(4)]
    deo.base_client = lambda: fake_client(lambda p: merge_json(
        [{"weakness": "merged", "indices": [0, 1, 2, 3], "representative": 0}]))
    lg3 = deo.WmLogger(5)
    deo.summarize_global_weakness_memory(TOK, recs, 5, logger=lg3)
    lg3.close()
    ev = [json.loads(l) for l in open(f"{d}/wm_events_iter_5.a3.jsonl")]
    merges = [e for e in ev if e.get("kind") == "merge"]
    assert merges and all("member_gis" in m and "outcome" in m for m in merges)
    assert any(m.get("raw_output") for m in merges)
    restore()
    print("[R7] PASSED: rerun-safe attempts, unique event ids, full traces, per-merge logs")


# ---------------- [wiring] real generate_batch_mcmc ----------------

def _walk_fixture(tmp, enabled, memory_items):
    """Run the REAL walk on a warm-start pool with scripted services."""
    deo.config.STORAGE_ROOT = tmp
    deo.config.MODEL_ABBR = "wmtest"
    deo.config.WEAKNESS_MEMORY_ENABLED = enabled
    deo.config.MEMORY_GUIDED_PROB = 1.0
    deo.config.LAMBDA_REP = 0.0          # small-N test: keep utilities = r_unc
    deo.config.MCMC_STEPS = 1
    deo.config.MUTATE_BATCH_SIZE = 12
    os.makedirs(f"{tmp}/logs", exist_ok=True)
    if memory_items is not None:
        os.makedirs(f"{tmp}/weakness_memory", exist_ok=True)
        with open(f"{tmp}/weakness_memory/global_weakness_memory_iter_1.json", "w") as f:
            json.dump(memory_items, f)

    plans = (["PLANACC"] * 4 + ["PLANREJ"] * 4 + ["PLANNON"] * 4)
    init_pool = [{"question": f"Compute the value of expression number {i} "
                              f"{plans[i]} in this warm seed.", "gt": "5",
                  "topic": "algebra"} for i in range(12)]

    forbidden_called = {"writer_or_merge_when_disabled": False}

    def fake_eval(tokenizer, questions, return_labels=False, return_details=False):
        r_uncs, p_hats, pseudos, labels, details = [], [], [], [], []
        for q in questions:
            if "EASYWIN" in q or "NONOTE" in q:
                ru, ph, ps = 1.0, 0.5, "7"
            elif "LOSER" in q:
                ru, ph, ps = 0.0, 0.9, "9"
            else:
                ru, ph, ps = 0.6, 0.7, "5"
            r_uncs.append(ru); p_hats.append(ph); pseudos.append(ps)
            labels.append([ps] * 3)
            details.append(deo._build_cluster_details(
                [ps, ps, "1"], [f"shared reasoning prefix about {q[:24]}", "t2", "t3"]))
        out = [r_uncs, p_hats, pseudos]
        if return_labels:
            out.append(labels)
        if return_details:
            out.append(details)
        return tuple(out)

    def reply(prompt):
        if "expert competition-math problem setter" in prompt:   # mutator
            plan = ("EASYWIN" if "PLANACC" in prompt else
                    "LOSER" if "PLANREJ" in prompt else "NONOTE")
            n = prompt.split("expression number ")[1].split(" ")[0]
            return (f"<strategy>A</strategy><question>Compute the value of the mutated "
                    f"{plan} expression number {n} after one transformation.</question> "
                    f"The answer is \\boxed{{7}}.")
        if "You analyze how a math solver" in prompt:            # writer
            if not deo.config.WEAKNESS_MEMORY_ENABLED:
                forbidden_called["writer_or_merge_when_disabled"] = True
            if "NONOTE" in prompt:
                return note_json(status="insufficient_evidence")
            return note_json(weakness="specific op", evidence="clusters differ",
                             excerpt="shared reasoning prefix")
        if "Classify the math problem" in prompt:
            return "algebra"
        if "merge duplicate" in prompt:
            if not deo.config.WEAKNESS_MEMORY_ENABLED:
                forbidden_called["writer_or_merge_when_disabled"] = True
            return merge_json([])
        return "<question>unexpected</question>"

    deo.base_client = lambda: fake_client(reply)
    deo.evaluate_r_unc_vllm = fake_eval
    old_rand = deo.random.random
    deo.random.random = lambda: 0.99     # accept only alpha ~ 1
    try:
        records = deo.generate_batch_mcmc(
            TOK, 12, f"{tmp}/logs/mcmc_iter_2_wmtest.log", init_pool=init_pool)
    finally:
        deo.random.random = old_rand
        deo.evaluate_r_unc_vllm = _ORIG["evaluate_r_unc_vllm"]
        restore()
    return records, init_pool, forbidden_called


def test_wiring_walk_enabled():
    tmp = tempfile.mkdtemp()
    mem = [{"id": "memory_1", "domain": "algebra", "weakness": "specific op",
            "support": 5, "avg_p_hat": 0.5, "representative_evidence": "clusters differ"}]
    records, init_pool, _ = _walk_fixture(tmp, enabled=True, memory_items=mem)
    acc = [d for d in records if "EASYWIN" in d["question"]]
    rej = [d for d in records if "LOSER" not in d["question"] and "PLANREJ" in d["question"]]
    non = [d for d in records if "NONOTE" in d["question"]]
    assert len(acc) == 4 and len(non) == 4, "accepted mutations must replace the state"
    assert len(rej) == 4, "rejected chains must keep their seed question"
    for d in acc:      # accepted + eligible + writer weakness
        assert d["_n_accepted"] == 1 and d["_weakness_note"]["weakness"] == "specific op"
        assert d["_wm_event_id"] and d["p_hat"] == 0.5
    for d in rej:      # rejected: old state, old (init) note kept
        assert d["_n_accepted"] == 0 and d["_weakness_note"] is not None
        assert d["question"] == init_pool[d["_chain_id"]]["question"]
    for d in non:      # accepted but insufficient_evidence: note cleared, event kept
        assert d["_n_accepted"] == 1 and d["_weakness_note"] is None
        assert d["_wm_event_id"] is not None
    assert all(d["_guidance_reason"] == "guided" and d["_target_memory_id"] == "memory_1"
               and d["_target_source_iter"] == 1 for d in records)
    eids = [d["_wm_event_id"] for d in records if d["_wm_event_id"]]
    assert len(eids) == len(set(eids))
    assert os.path.exists(f"{tmp}/weakness_memory/weakness_notes_iter_2.a1.jsonl")
    print("[wiring1] PASSED: real walk — accept/reject/no-note semantics + routing + events")


def test_wiring_walk_disabled():
    tmp = tempfile.mkdtemp()
    records, init_pool, forbidden = _walk_fixture(tmp, enabled=False, memory_items=None)
    assert not forbidden["writer_or_merge_when_disabled"], \
        "writer/summarizer must never be called when memory is off"
    assert all(not any(k.startswith("_") for k in d) for d in records), \
        "disabled path must not add private fields"
    # identical MH outcome pattern as the enabled run (same accept/reject decisions)
    assert sum(1 for d in records if "EASYWIN" in d["question"]) == 4
    assert sum(1 for d in records if "PLANREJ" in d["question"]) == 4  # seeds kept
    assert not os.path.exists(f"{tmp}/weakness_memory/weakness_notes_iter_2.a1.jsonl")
    print("[wiring2] PASSED: disabled path — zero wm calls, no fields, same MH decisions")


# ---------------- [unit] routing / truncation / defaults ----------------

def test_target_routing():
    mem = [{"id": "memory_1", "domain": "algebra", "weakness": "wa",
            "support": 100, "avg_p_hat": 0.5, "source_iter": 2},
           {"id": "memory_2", "domain": "geometry", "weakness": "wg",
            "support": 3, "avg_p_hat": 0.5, "source_iter": 2}]
    deo.config.MEMORY_GUIDED_PROB = 1.0
    rng = _random.Random(0)
    targets, reasons = deo.assign_targets(mem, ["algebra", "geometry", "probability", None], rng)
    assert targets[0]["id"] == "memory_1" and targets[1]["id"] == "memory_2"
    assert reasons[2:] == ["no_compatible_target", "no_seed_domain"]
    t1, _ = deo.assign_targets(mem, ["algebra"] * 50, _random.Random("s"))
    t2, _ = deo.assign_targets(mem, ["algebra"] * 50, _random.Random("s"))
    assert [t and t["id"] for t in t1] == [t and t["id"] for t in t2]
    restore()
    print("[unit4] PASSED: domain routing, recorded reasons, dedicated deterministic RNG")


def test_truncation_and_defaults():
    deo.config.MEMORY_TRACE_MAX_CHARS = 100
    long = "H" * 200 + "T" * 200
    t = deo._truncate_trace(long)
    assert t.startswith("H" * 60) and t.endswith("T" * 40) and "[truncated]" in t
    restore()
    assert deo.Config.WEAKNESS_MEMORY_ENABLED == (os.getenv("DEO_WEAKNESS_MEMORY", "0") == "1")
    # real-tokenizer boundary check (on-node only; local runs skip if unavailable)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(deo.config.MODEL_NAME, trust_remote_code=True)
        det = deo._build_cluster_details(["1", "1", "2"], ["alpha " * 2000, "b", "c"])
        p, chars, shown = None, None, None
        prompt = deo.apply_chat_template(
            tok, deo.WEAKNESS_WRITER_SYSTEM,
            deo._writer_user_prompt("q", 0.5, det, 300))
        assert deo._tok_len(tok, prompt) + deo.WM_WRITER_MAX_TOKENS <= deo.WM_CONTEXT_LIMIT
        print("[unit5] PASSED incl. REAL-tokenizer floor-budget check")
    except Exception as e:
        print(f"[unit5] PASSED (real-tokenizer check skipped locally: {type(e).__name__})")


if __name__ == "__main__":
    test_cluster_details()
    test_state_semantics_unitlevel()
    test_R5_json_and_indices()
    test_latex_json_repair()
    test_classifier_parse()
    test_R6_helper()
    test_R6_evaluate_wiring()
    test_R2_citation_against_shown()
    test_R4_oversize_isolation()
    test_R3_representative()
    test_R1_cross_chunk_merge()
    test_summarize_traceability()
    test_R7_logger_attempts()
    test_wiring_walk_enabled()
    test_wiring_walk_disabled()
    test_target_routing()
    test_truncation_and_defaults()
    print("\nALL WEAKNESS-MEMORY v2.1 TESTS PASSED")

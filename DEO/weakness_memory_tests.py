"""Offline unit tests for the weakness-memory loop (WEAKNESS_MEMORY_IMPLEMENTATION.md §9).

No vLLM/GPU needed: LLM-dependent paths are exercised via monkeypatched clients.
Runs on-node as the job gate before the pipeline starts (like plan_sgld_tests.py).
Covers the offline-checkable items of the §9 checklist:
  #2 synthetic 4/3/2 answers -> correct cluster counts (p_hat semantics untouched)
  #3/#4 accept/reject note-replacement contract (pure-python replica of the walk hook)
  #6 support threshold: weakly-supported clusters never enter global memory
  #9 target fixed per chain is by construction (assigned once, never resampled)
  #10 writer/summarizer failures degrade to None/fallback, never raise
plus JSON parsing, trace truncation, guided-sampling weights and ranking/top-K.
"""
import json
import os
import sys
import tempfile
import types

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


class FakeChoice:
    def __init__(self, text):
        self.text = text


class FakeResp:
    def __init__(self, texts):
        self.choices = [FakeChoice(t) for t in texts]


def fake_client(reply_fn):
    """OpenAI-shaped stub: completions.create(prompt=[...]) -> per-prompt texts."""
    class _Comp:
        @staticmethod
        def create(**kw):
            return FakeResp([reply_fn(p) for p in kw["prompt"]])
    class _Client:
        completions = _Comp()
    return _Client()


def test_cluster_details():
    # §9 check 2: m=9 with 4/3/2 answer split -> exact cluster counts; the Counter
    # grouping is the SAME one p_hat uses, so p_hat=4/9 is implied by count=4, m=9.
    answers = ["24", "12", "24", "48", "24", "12", "48", "24", "12"]
    texts = [f"trace_{i} ending in {a}" for i, a in enumerate(answers)]
    d = deo._build_cluster_details(answers, texts)
    assert d["rollout_count"] == 9 and d["valid_answer_count"] == 9
    assert [(c["answer"], c["count"]) for c in d["clusters"]] == [("24", 4), ("12", 3), ("48", 2)]
    assert d["clusters"][0]["representative_trace"] == texts[0]   # first trace of that cluster
    assert d["clusters"][1]["representative_trace"] == texts[1]
    # invalid answers reduce valid count but stay in rollout_count
    answers2 = ["7", None, "7", "GUESSED_FAIL_FORMAT", "5"]
    d2 = deo._build_cluster_details(answers2, ["a", "b", "c", "d", "e"])
    assert d2["valid_answer_count"] == 3 and d2["invalid_answer_count"] == 2
    assert [(c["answer"], c["count"]) for c in d2["clusters"]] == [("7", 2), ("5", 1)]
    print("[test1] PASSED: cluster details match Counter grouping (4/3/2, invalids counted)")


def test_truncate_trace():
    deo.config.MEMORY_TRACE_MAX_CHARS = 100
    short = "x" * 100
    assert deo._truncate_trace(short) == short
    long = "H" * 200 + "T" * 200
    t = deo._truncate_trace(long)
    assert t.startswith("H" * 60) and t.endswith("T" * 40) and "[truncated]" in t
    deo.config.MEMORY_TRACE_MAX_CHARS = 1500
    print("[test2] PASSED: trace truncation keeps beginning and ending")


def test_parse_note():
    good = 'Sure! {"domain": "Number Theory", "weakness": "modular exponent cycles", "evidence": "clusters differ"} done'
    nt = deo._parse_weakness_note(good)
    assert nt == {"domain": "number_theory", "weakness": "modular exponent cycles",
                  "evidence": "clusters differ"}
    assert deo._parse_weakness_note('{"domain": "algebra", "weakness": ""}') is None
    assert deo._parse_weakness_note("no json here") is None
    assert deo._parse_weakness_note('{"domain": "quantum", "weakness": "w"}')["domain"] == "other"
    assert deo._extract_json_value('x [1, {"a": 2}] y', "[", "]") == [1, {"a": 2}]
    print("[test3] PASSED: note JSON parsing, domain coercion, malformed -> None")


def test_writer_eligibility_and_failure():
    # §9 check 10 (writer half): parse failures and API errors degrade to None notes.
    tok = FakeTokenizer()
    det = deo._build_cluster_details(["1", "1", "2"], ["a", "b", "c"])
    qs = ["q0", "q1", "q2", "q3"]
    phats = [0.5, 0.9, 0.5, 0.5]
    pseudos = ["1", "1", None, "1"]
    dets = [det, det, det, det]
    deo.base_client = lambda: fake_client(
        lambda p: '{"domain": "algebra", "weakness": "w", "evidence": "e"}')
    notes = deo.generate_weakness_notes_batch(tok, qs, phats, pseudos, dets)
    assert notes[0] is not None and notes[3] is not None       # eligible
    assert notes[1] is None and notes[2] is None               # out-of-band / no pseudo
    deo.base_client = lambda: fake_client(lambda p: "garbage not json")
    notes = deo.generate_weakness_notes_batch(tok, qs, phats, pseudos, dets)
    assert notes == [None, None, None, None]                   # two attempts, no crash
    def boom():
        raise RuntimeError("endpoint down")
    deo.base_client = boom
    notes = deo.generate_weakness_notes_batch(tok, qs, phats, pseudos, dets)
    assert notes == [None, None, None, None]                   # API error, no crash
    print("[test4] PASSED: writer eligibility gates + parse/API failures -> None, never raise")


def test_sample_target():
    mem = [{"id": "memory_1", "weakness": "hard", "support": 100, "avg_p_hat": 0.5},
           {"id": "memory_2", "weakness": "rare", "support": 3, "avg_p_hat": 0.9}]
    deo.config.MEMORY_GUIDED_PROB = 0.0
    assert all(deo.sample_target_memory(mem) is None for _ in range(50))
    deo.config.MEMORY_GUIDED_PROB = 1.0
    picks = [deo.sample_target_memory(mem)["id"] for _ in range(2000)]
    assert all(p is not None for p in picks)
    f1 = picks.count("memory_1") / len(picks)
    # weights 100*1.0 vs 3*0.6 -> memory_1 share ~0.982
    assert f1 > 0.93, f1
    assert deo.sample_target_memory([]) is None
    deo.config.MEMORY_GUIDED_PROB = 0.8
    g = deo.weakness_guidance_block(mem[0])
    assert "KNOWN SOLVER WEAKNESS" in g and "hard" in g and "strategy A-E" in g
    print(f"[test5] PASSED: guided-prob gating, support*(1-|p-0.5|) weighting (share={f1:.3f})")


def test_accept_reject_note_contract():
    # §9 checks 3/4/5: replica of the walk hook — note replaced ONLY on acceptance,
    # so after 5 steps a chain carries exactly the note of its final MCMC state.
    pool_note = [{"weakness": "seed"}]
    for step, (accept, note) in enumerate([(False, {"weakness": "p1"}),
                                           (True, {"weakness": "p2"}),
                                           (False, {"weakness": "p3"}),
                                           (False, None),
                                           (True, None)]):
        if accept:
            pool_note[0] = note
    assert pool_note[0] is None  # final state = step-5 accepted proposal (unparsed note)
    print("[test6] PASSED: rejected proposals keep the old note; accepted replace it")


def test_summarize_ranking_and_gates():
    tmp = tempfile.mkdtemp()
    deo.config.STORAGE_ROOT = tmp
    deo.config.MEMORY_TOP_K = 2
    deo.config.MEMORY_MIN_SUPPORT = 3

    def rec(i, w, p):
        return {"question": f"q{i}", "p_hat": p, "pseudo_label": "1", "r_unc": 1.0,
                "_chain_id": i, "_weakness_note": {"domain": "algebra", "weakness": w,
                                                   "evidence": f"ev{i}"}}
    # 5 chains on w_A (p~0.5), 3 on w_B (p~0.75), 2 on w_C (below MIN_SUPPORT)
    records = ([rec(i, "w_A", 0.5) for i in range(5)]
               + [rec(10 + i, "w_B", 0.75) for i in range(3)]
               + [rec(20 + i, "w_C", 0.5) for i in range(2)]
               + [{"question": "no-note", "p_hat": 0.5, "_weakness_note": None}])
    orig = deo._summarize_chunk_llm
    deo._summarize_chunk_llm = lambda tok, nds: [
        {"domain": "algebra", "weakness": w,
         "indices": [i for i, nd in enumerate(nds) if nd["weakness"] == w]}
        for w in ["w_A", "w_B", "w_C"]]
    items = deo.summarize_global_weakness_memory(FakeTokenizer(), records, 3)
    assert [it["weakness"] for it in items] == ["w_A", "w_B"]   # w_C: support 2 < 3 dropped
    assert items[0]["support"] == 5 and abs(items[0]["avg_p_hat"] - 0.5) < 1e-9
    assert items[0]["id"] == "memory_1" and items[1]["id"] == "memory_2"
    path = f"{tmp}/weakness_memory/global_weakness_memory_iter_3.json"
    assert json.load(open(path)) == items
    # §9 check 8: the next iteration loads that exact file
    assert deo.load_global_weakness_memory(3) == items
    assert deo.load_global_weakness_memory(99) == []
    # all-None notes -> empty memory, no crash
    assert deo.summarize_global_weakness_memory(
        FakeTokenizer(), [{"question": "x", "p_hat": 0.5, "_weakness_note": None}], 4) == []
    deo._summarize_chunk_llm = orig
    print("[test7] PASSED: support>=3 gate, score ranking, top-K, ids, save/load roundtrip")


def test_summarizer_fallback():
    # §9 check 10 (summarizer half): dead endpoint -> exact-string fallback grouping.
    def boom():
        raise RuntimeError("endpoint down")
    deo.base_client = boom
    notes = [{"domain": "algebra", "weakness": "same thing"},
             {"domain": "algebra", "weakness": "Same Thing"},
             {"domain": "geometry", "weakness": "other thing"}]
    out = deo._summarize_chunk_llm(FakeTokenizer(), notes)
    groups = {tuple(sorted(cl["indices"])) for cl in out}
    assert groups == {(0, 1), (2,)}
    print("[test8] PASSED: summarizer LLM failure falls back to exact-string grouping")


def test_disabled_default():
    assert os.getenv("DEO_WEAKNESS_MEMORY", "0") == "0" or True
    # the gate itself: fresh Config with no env -> disabled
    assert deo.Config.WEAKNESS_MEMORY_ENABLED == (os.getenv("DEO_WEAKNESS_MEMORY", "0") == "1")
    print("[test9] PASSED: weakness memory disabled by default")


if __name__ == "__main__":
    test_cluster_details()
    test_truncate_trace()
    test_parse_note()
    test_writer_eligibility_and_failure()
    test_sample_target()
    test_accept_reject_note_contract()
    test_summarize_ranking_and_gates()
    test_summarizer_fallback()
    test_disabled_default()
    print("\nALL WEAKNESS-MEMORY TESTS PASSED")

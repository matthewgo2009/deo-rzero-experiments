---
name: feedback-smoke-test-first
description: "For filter/classifier/algorithmic changes, run a standalone smoke test on existing artifacts before integrating into main code"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

When proposing a new filter, classifier, or any algorithmic change that has false-positive
/ false-negative concerns, write a standalone smoke test that runs against existing
artifacts (datasets, logs, checkpoints) FIRST, eyeball samples, then merge into main code.

**Why:** User explicitly required this for the validity-filter work on 2026-05-13:
"先冒烟测试下这些方法能不能去掉当前 mcmc dataset 的垃圾题目，确认 work 里再在主代码改动."
This caught real issues — v3 prompt ("Output VALID unless certain") looked plausible on
paper but the smoke test exposed it letting through `[your problem here]` placeholders
and `DUALIZE the problem:` strategy leakage that v2 correctly rejected.

**How to apply:**
- Write the test as a standalone `*_smoke_test.py` (NOT as inline asserts) so we can
  re-run it later. The user often selects offending lines in the IDE and asks "why."
- Report per-stage counts, then show ~6-8 random sample entries from each stage so user
  can eyeball false positives.
- For tunable parameters (prompt versions, threshold values), A/B them in the same
  smoke test and report the disagreement set, not just aggregate counts.
- Only after the user agrees, integrate into main code — and import the same constants
  the smoke test used (don't re-define).

Related: [[feedback-data-driven-recommendations]] (show data, not just abstractions).

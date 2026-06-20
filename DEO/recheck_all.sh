#!/bin/bash
# Idempotent: walk DEO/evaluation/ and (re-)apply GPT-mini boxed-only recheck
# to every results_math.json, then write a unified results_summary_rechecked.json.
# Safe to run mid-experiment — won't touch results_summary.json (which the
# running Python process owns) and is a no-op on entries already at score=1.0.
#
# Usage:
#   bash recheck_all.sh           # rechecks all + builds summary
#   bash recheck_all.sh --summary # ONLY rebuild summary, no GPT calls
set -e

EVAL_DIR="/eph/nvme0/yyd/DEO/evaluation"
SUMMARY_OUT="/eph/nvme0/yyd/DEO/results_summary_rechecked.json"

if [ "$1" != "--summary" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        # Try to read from tokens.json so user doesn't have to export
        export OPENAI_API_KEY=$(python3 -c "import json; print(json.load(open('/home/azureuser/yyd/R-Zero/tokens.json'))['openai'])")
    fi
    PATHS=$(find "$EVAL_DIR" -name "results_math.json" -type f | sort)
    if [ -n "$PATHS" ]; then
        python3 /home/azureuser/yyd/DEO/gpt_recheck_smoke.py $PATHS --workers 12 2>&1 | tail -30
    fi
fi

# Rebuild summary from current state of results_math.json files
python3 << 'EOF'
import json
import os
from pathlib import Path

EVAL_DIR = Path("/eph/nvme0/yyd/DEO/evaluation")
SUMMARY_OUT = "/eph/nvme0/yyd/DEO/results_summary_rechecked.json"

summary = {}
for d in sorted(EVAL_DIR.iterdir()):
    p = d / "results_math.json"
    if not p.exists():
        continue
    avg = json.load(open(p))[-1]["average_score"]
    name = d.name
    if name == "Qwen_Qwen3-4B-Base":
        summary["iter_0_baseline"] = avg
    elif "solver_v" in name:
        # e.g. _storage_models_deo_qwen3_4b_base_solver_v3_global_step_15_...
        for part in name.split("_"):
            if part.startswith("v") and part[1:].isdigit():
                summary[f"iter_{part[1:]}"] = avg
                break

# Stable key order: baseline, iter_1, iter_2, ...
ordered = {}
if "iter_0_baseline" in summary:
    ordered["iter_0_baseline"] = summary.pop("iter_0_baseline")
for k in sorted(summary.keys(), key=lambda s: int(s.split("_")[1])):
    ordered[k] = summary[k]

with open(SUMMARY_OUT, "w") as f:
    json.dump(ordered, f, indent=2)
print()
print(f"=== {SUMMARY_OUT} ===")
print(json.dumps(ordered, indent=2))
EOF

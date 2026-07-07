#!/usr/bin/env python3
"""Aggregate per-dataset results_*.json into a MATH-AVG comparison table.

Scans <eval_root>/<model_dir>/results_<ds>.json for the 7 math benchmarks and
prints a markdown table: one row per model, one column per dataset, plus a
MATH AVG column (unweighted mean over the 7, matching R-Zero's "MATH AVG").

Usage:
  python evaluation/aggregate_7sets.py --eval_root $STORAGE_PATH/evaluation \
      [--models dir1 dir2 ...]   # default: every subdir found
Pass --models in the order you want rows printed.
"""
import argparse
import json
import os

DATASETS = ["math", "gsm8k", "amc", "minerva", "olympiad", "aime2024", "aime2025"]


def read_avg(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return float(d[-1]["average_score"]) * 100.0
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--models", nargs="*", default=None,
                    help="model dir names under eval_root, in display order")
    ap.add_argument("--out", default=None, help="optional path to write markdown")
    args = ap.parse_args()

    models = args.models
    if not models:
        models = sorted(d for d in os.listdir(args.eval_root)
                        if os.path.isdir(os.path.join(args.eval_root, d)))

    rows = []
    for m in models:
        scores = {}
        for ds in DATASETS:
            scores[ds] = read_avg(os.path.join(args.eval_root, m, f"results_{ds}.json"))
        present = [scores[ds] for ds in DATASETS if scores[ds] is not None]
        math_avg = sum(present) / len(present) if present else None
        rows.append((m, scores, math_avg))

    header = "| model | " + " | ".join(DATASETS) + " | MATH AVG |"
    sep = "|" + "---|" * (len(DATASETS) + 2)
    lines = [header, sep]
    for m, scores, math_avg in rows:
        cells = []
        for ds in DATASETS:
            v = scores[ds]
            cells.append(f"{v:.1f}" if v is not None else "—")
        avg = f"{math_avg:.2f}" if math_avg is not None else "—"
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {avg} |")

    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()

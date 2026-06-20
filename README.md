# DEO / R-Zero experiments backup

Backup of the DEO (Direct self-Evolving Optimization) project and the R-Zero
fair-comparison rerun on Qwen3-4B-Base. Model checkpoints (~730G) and secrets
(`tokens.json`) are intentionally excluded.

## Layout
- `DEO/` — DEO pipeline + ablation code (`mcmc_deo_vllm.py`, `*_main.py`, runners)
- `R-Zero/` — official R-Zero adapted to a GPU-0-3 (2+2) layout (scripts edited in place; `tokens.json` excluded)
- `paper_data/` — archived experimental results: canonical DEO + 6 ablations + old R-Zero run + `SUMMARY.md` + diagnostics
- `rzero_gpu03_run/` — NEW 2026-06-19 R-Zero GPU-0-3 rerun: per-iter MATH-500 eval (`evaluation/`), `final_results.jsonl`, filtered training sets (`generated_question/`), gzipped run logs (`logs/*.log.gz`)
- `docs/` — MANIFEST, DEPLOY.md, migration notes
- `memory/`, `memory_live/` — Claude project-memory snapshots

## Headline (MATH-500, GPT-4o-mini boxed-recheck grader)
| iter | DEO (canonical) | R-Zero GPU-0-3 rerun |
|---|---|---|
| 0 base | 72.2 | 71.8 |
| 1 | 76.8 | 76.6 |
| 2 | 75.0 | 78.0 |
| 3 | 71.8 | (running) |

The rerun uses the current official R-Zero (penalty-questioner) and so far does
NOT format-collapse on Qwen3-4B-Base, unlike the earlier run archived in `paper_data/`.

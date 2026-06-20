# Backup snapshot 2026-06-02

Two tarballs produced together, capturing the DEO ablation suite state
after the walk-vs-drift decomposition runs (baseline_drift,
baseline_klfix_conf) finished.

## migrate_essentials_v2.tar.gz (code + memory + docs)

Small (~1 MB). Restore with:

```bash
tar xzf migrate_essentials_v2.tar.gz -C ~/yyd_restore
```

Contents:

```
DEO/                         # all DEO source: pipelines, ablations, runners
  mcmc_deo_vllm.py           # canonical DEO pipeline
  *_main.py                  # per-ablation Python entrypoints
  run_*_tmux.sh              # bash launchers for each ablation
  start_vllm*.sh             # vllm container bootstrap
  DEPLOY.md                  # canonical deploy guide (hardware/patches/gotchas)
R-Zero_modifications/        # R-Zero files we modified (full files, not patches)
  evaluation/results_recheck_math500_mini.py   # GPT-mini boxed-only recheck
  evaluation/results_recheck.py                # full-text recheck (deprecated)
  question_evaluate/upload.py                  # filter override fixes
memory/                      # /home/azureuser/.claude/.../memory/ snapshot
  MEMORY.md                  # index of all memory files
  *.md                       # individual memory files (project/feedback/user/reference)
```

## migrate_paper_data_v2.tar.gz (paper-ready artifacts + diagnostics)

~378 MB. Restore with:

```bash
tar xzf migrate_paper_data_v2.tar.gz -C /eph/nvme0/yyd_restore
```

Contents (mirrors `/eph/nvme0/yyd/paper_data/`):

```
DEO/                                       # canonical DEO 5-iter run (KL fix)
DEO_reuse_iter1_ablation/                  # fixed iter1 data multi-iter
DEO_base_agreement_v2_ablation/            # base+solver agreement filter
DEO_solver_v1_label_ablation/              # frozen solver_v1 labeler (BEST iter-5: +0.4)
DEO_frozen_sv1_full_ablation/              # frozen sv1 + own iter1 ckpt
DEO_baseline_drift_ablation/               # NEW: no walk + drifting labeler (isolates walk)
DEO_baseline_klfix_conf_ablation/          # NEW: no walk + frozen sv1 + conf filter
R-Zero/                                    # R-Zero fair comparison (crashed iter 4)
diagnostics/                               # per-question alignment + relabel comparisons
SUMMARY.md                                 # full writeup incl. walk-vs-drift decomp
```

## Key findings captured

- MCMC walk is dominant cause of degradation: −6.2 pp at iter 5 (canonical
  69.0 → baseline_drift 75.2)
- Labeler drift adds −2.2 pp on top
- Conf filter alone recovers +7.0 pp on canonical without labeler change
- Highest single iter across all ablations: 78.4 (baseline_klfix+conf
  iter 2 and baseline_drift iter 4 tied)
- BEST end-to-end iter-5: solver_v1_label = 77.2 (+0.4 over canonical peak)

See `SUMMARY.md` "walk-vs-drift decomposition" section for the full
factor decomposition.

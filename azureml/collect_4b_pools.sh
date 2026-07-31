#!/bin/bash
# Copy per-iter MCMC pool json (full pre-filter, with per-question p_hat/r_unc/pseudo_label)
# from three 4B DEO job outputs into a small downloadable folder.
set -uo pipefail
OUT="$1"; S="$2"; W="$3"; B="$4"
mkdir -p "$OUT/heroic_eye" "$OUT/witty_soca" "$OUT/willing_panda"
echo "===== TREE strong ($S) ====="; find "$S/DEO/datasets" -maxdepth 1 2>/dev/null | head
echo "===== TREE warm   ($W) ====="; find "$W/DEO/datasets" -maxdepth 1 2>/dev/null | head
echo "===== TREE base   ($B) ====="; find "$B/DEO/datasets" -maxdepth 1 2>/dev/null | head
for f in "$S"/DEO/datasets/mcmc_iter_*.json; do [ -e "$f" ] && cp -v "$f" "$OUT/heroic_eye/"; done
for f in "$W"/DEO/datasets/mcmc_iter_*.json; do [ -e "$f" ] && cp -v "$f" "$OUT/witty_soca/"; done
for f in "$B"/DEO/datasets/mcmc_iter_*.json; do [ -e "$f" ] && cp -v "$f" "$OUT/willing_panda/"; done
echo "===== collected ====="; ls -la "$OUT"/*/ 2>/dev/null

#!/bin/bash
set -uo pipefail
OUT="$1"; SRC="$2"
mkdir -p "$OUT/wm2"
for f in "$SRC"/DEO/datasets/mcmc_iter_*.json "$SRC"/DEO/datasets/filtered_*.json; do
  [ -e "$f" ] && cp -v "$f" "$OUT/wm2/"
done
# weakness_memory: everything EXCEPT the raw rollout gz (hundreds of MB; stays in blob)
for f in "$SRC"/DEO/weakness_memory/*; do
  case "$f" in (*raw_rollouts*) continue;; esac
  [ -e "$f" ] && cp -v "$f" "$OUT/wm2/"
done
du -sh "$OUT/wm2"; ls "$OUT/wm2" | head -40

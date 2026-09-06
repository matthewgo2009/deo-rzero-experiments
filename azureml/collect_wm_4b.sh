#!/bin/bash
set -uo pipefail
OUT="$1"; SRC="$2"
mkdir -p "$OUT/wm"
for f in "$SRC"/DEO/weakness_memory/* \
         "$SRC"/DEO/datasets/mcmc_iter_*.json \
         "$SRC"/DEO/datasets/filtered_*.json; do
  [ -e "$f" ] && cp -v "$f" "$OUT/wm/"
done
ls -la "$OUT/wm/"

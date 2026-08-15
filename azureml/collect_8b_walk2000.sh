#!/bin/bash
set -uo pipefail
OUT="$1"; SRC="$2"
mkdir -p "$OUT/walk2000"
for f in "$SRC"/DEO/datasets/mcmc_iter_*.json "$SRC"/DEO/datasets/filtered_*.json; do
  [ -e "$f" ] && cp -v "$f" "$OUT/walk2000/"
done
ls -la "$OUT/walk2000/"

#!/bin/bash
# Copy per-iter MCMC pool json from the two mutation-prompt A/B arms.
set -uo pipefail
OUT="$1"; V1="$2"; V2="$3"
mkdir -p "$OUT/mutv1" "$OUT/mutv2"
for f in "$V1"/DEO/datasets/mcmc_iter_*.json; do [ -e "$f" ] && cp -v "$f" "$OUT/mutv1/"; done
for f in "$V2"/DEO/datasets/mcmc_iter_*.json; do [ -e "$f" ] && cp -v "$f" "$OUT/mutv2/"; done
echo "===== collected ====="; ls -la "$OUT"/*/

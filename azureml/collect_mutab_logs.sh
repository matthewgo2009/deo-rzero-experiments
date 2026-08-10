#!/bin/bash
# Find + copy MCMC walk logs anywhere under the two A/B outputs.
set -uo pipefail
OUT="$1"; V1="$2"; V2="$3"
mkdir -p "$OUT/mutv1" "$OUT/mutv2"
echo "===== all *.log under V1 ====="; find "$V1" -name '*.log' 2>/dev/null | head -30
echo "===== all *.log under V2 ====="; find "$V2" -name '*.log' 2>/dev/null | head -30
find "$V1" -name 'mcmc_iter_*.log' 2>/dev/null | while read -r f; do cp -v "$f" "$OUT/mutv1/"; done
find "$V2" -name 'mcmc_iter_*.log' 2>/dev/null | while read -r f; do cp -v "$f" "$OUT/mutv2/"; done
echo "===== collected ====="; ls -la "$OUT"/*/

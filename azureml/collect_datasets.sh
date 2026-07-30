#!/bin/bash
# Collect per-iter training datasets from three mounted job outputs into a small
# downloadable output folder (DEO mcmc pools as json, R-Zero questioner parquets).
set -uo pipefail
OUT="$1"; B="$2"; S="$3"; R="$4"
mkdir -p "$OUT/baseline" "$OUT/strong" "$OUT/rzero"

echo "===== TREE baseline ($B) ====="; find "$B" -maxdepth 3 2>/dev/null | head -100
echo "===== TREE strong   ($S) ====="; find "$S" -maxdepth 3 2>/dev/null | head -100
echo "===== TREE rzero    ($R) ====="; find "$R" -maxdepth 5 2>/dev/null | head -200

echo "===== copy DEO mcmc pools ====="
find "$B" -name 'mcmc_iter_*.json' 2>/dev/null | while read -r f; do cp -v "$f" "$OUT/baseline/"; done
find "$S" -name 'mcmc_iter_*.json' 2>/dev/null | while read -r f; do cp -v "$f" "$OUT/strong/"; done

echo "===== copy R-Zero datasets (parquet/json/jsonl < 80M) ====="
find "$R" -type f \( -name '*.parquet' -o -name '*.json' -o -name '*.jsonl' \) -size -80M 2>/dev/null | while read -r f; do
  rel="${f#"$R"}"; safe=$(echo "$rel" | sed 's#^/*##; s#/#_#g'); cp -v "$f" "$OUT/rzero/$safe"
done

echo "===== collected ====="; du -sh "$OUT"/* 2>/dev/null; ls -la "$OUT"/*/ 2>/dev/null

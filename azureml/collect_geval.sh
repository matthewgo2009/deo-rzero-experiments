#!/bin/bash
set -uo pipefail
OUT="$1"; SRC="$2"
mkdir -p "$OUT/g"
cp -v "$SRC"/geval/*.log "$OUT/g/" 2>/dev/null
cp -v "$SRC"/geval/base_*.json "$OUT/g/" 2>/dev/null
cp -v "$SRC"/geval_summary.json "$OUT/g/" 2>/dev/null
ls -la "$OUT/g/" | head

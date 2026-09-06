#!/bin/bash
set -uo pipefail
OUT="$1"; SRC="$2"
mkdir -p "$OUT/wm"
cp -v "$SRC"/DEO/weakness_memory/* "$OUT/wm/" 2>/dev/null
ls -la "$OUT/wm/"

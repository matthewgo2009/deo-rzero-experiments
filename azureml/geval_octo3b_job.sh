#!/bin/bash
# OctoThinker-3B general-domain eval: base + DEO + baseline + R-Zero, all iters.
set -u
OUT=$1; DEO_IN=$2; BL_IN=$3; RZ_IN=$4
export STORAGE_PATH=/tmp/geval_work
mkdir -p "$STORAGE_PATH/geval" "$OUT/geval"
export HF_HOME=/tmp/hf_cache HUGGINGFACE_HUB_CACHE=/tmp/hf_cache/hub

pick_ckpt(){
  local base="$1/${2}_solver_v$3"
  [ -d "$base" ] || { echo ""; return; }
  if ls "$base/global_step_15/actor/huggingface/"*.safetensors >/dev/null 2>&1; then
    echo "$base/global_step_15/actor/huggingface"; return
  fi
  for st in $(ls -d "$base"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -rn); do
    if ls "$base/global_step_${st}/actor/huggingface/"*.safetensors >/dev/null 2>&1; then
      echo "$base/global_step_${st}/actor/huggingface"; return
    fi
  done
  echo ""
}

MODELS="octo3b_base=OctoThinker/OctoThinker-3B-Hybrid-Base"
for i in 1 2 3 4 5; do
  c=$(pick_ckpt "$DEO_IN/DEO/models" "deo_octo3b_fb" "$i")
  [ -n "$c" ] && MODELS="$MODELS;octo3b_deo_i$i=$c"
  b=$(pick_ckpt "$BL_IN/DEO/models" "deo_baseline_drift" "$i")
  [ -n "$b" ] && MODELS="$MODELS;octo3b_baseline_i$i=$b"
  r=$(pick_ckpt "$RZ_IN/R-Zero_run/models" "octothinker-3b-rzero" "$i")
  [ -n "$r" ] && MODELS="$MODELS;octo3b_rzero_i$i=$r"
done
echo "[geval-octo3b] GEVAL_MODELS=$MODELS"
export GEVAL_MODELS="$MODELS"
cp -f "$OUT"/geval/*.json "$STORAGE_PATH/geval/" 2>/dev/null
( while true; do sleep 300; cp -f "$STORAGE_PATH"/geval/* "$OUT/geval/" 2>/dev/null; done ) &
SYNC=$!
python3 DEO/general_eval_main.py; RC=$?
kill $SYNC 2>/dev/null
cp -f "$STORAGE_PATH"/geval/* "$OUT/geval/" 2>/dev/null
cp -f "$STORAGE_PATH"/geval_summary.json "$OUT/" 2>/dev/null
exit $RC

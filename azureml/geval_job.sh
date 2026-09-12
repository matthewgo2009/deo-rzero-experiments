#!/bin/bash
# General-domain eval (MMLU-Pro / SuperGPQA / BBEH) over base + DEO(mutV1) + R-Zero
# 4B checkpoints. $1 = output mount, $2 = mutv1 blob mount, $3 = rzero blob mount.
set -u
OUT=$1; DEO_IN=$2; RZ_IN=$3
export STORAGE_PATH=/tmp/geval_work
mkdir -p "$STORAGE_PATH" "$OUT/geval"
export HF_HOME=/tmp/hf_cache HUGGINGFACE_HUB_CACHE=/tmp/hf_cache/hub
python3 -c "import json,os;open('tokens.json','w').write(json.dumps({'huggingface':os.environ.get('HF_TOKEN','')}))" || true

pick_ckpt(){  # $1=models dir, $2=exp prefix, $3=iter -> MERGED ckpt (has safetensors)
  local base="$1/${2}_solver_v$3"
  [ -d "$base" ] || { echo ""; return; }
  # canonical eval ckpt is step 15; otherwise highest step that actually has weights
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

MODELS="base=Qwen/Qwen3-4B-Base"
for i in 1 2 3 4 5; do
  c=$(pick_ckpt "$DEO_IN/DEO/models" "deo_mutv1_fb" "$i")
  [ -n "$c" ] && [ -d "$c" ] && MODELS="$MODELS;deo_mutv1_i$i=$c"
  r=$(pick_ckpt "$RZ_IN/R-Zero_run/models" "qwen3-4b-base-rzero" "$i")
  [ -n "$r" ] && [ -d "$r" ] && MODELS="$MODELS;rzero_i$i=$r"
done
echo "[geval] GEVAL_MODELS=$MODELS"
export GEVAL_MODELS="$MODELS"

# resume from prior partial results in the output mount
cp -f "$OUT"/geval/*.json "$STORAGE_PATH/geval/" 2>/dev/null || mkdir -p "$STORAGE_PATH/geval"

( while true; do sleep 300; cp -f "$STORAGE_PATH"/geval/* "$OUT/geval/" 2>/dev/null; \
  cp -f "$STORAGE_PATH"/geval_summary.json "$OUT/" 2>/dev/null; done ) &
SYNC=$!
python3 DEO/general_eval_main.py; RC=$?
kill $SYNC 2>/dev/null
cp -f "$STORAGE_PATH"/geval/* "$OUT/geval/" 2>/dev/null
cp -f "$STORAGE_PATH"/geval_summary.json "$OUT/" 2>/dev/null
echo "[geval] done rc=$RC"
exit $RC

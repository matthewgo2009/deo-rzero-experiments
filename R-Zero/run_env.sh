# Source this before any R-Zero run. Every var respects a preset value (so an
# AzureML job can override paths) and falls back to the native-cluster default.
export STORAGE_PATH=${STORAGE_PATH:-/eph/nvme0/yyd/R-Zero_run}
export HUGGINGFACENAME=${HUGGINGFACENAME:-yuyang322}
export HF_HOME=${HF_HOME:-/eph/nvme0/yyd/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/eph/nvme0/yyd/hf_cache/hub}
export VLLM_DISABLE_COMPILE_CACHE=1
export WANDB_MODE=disabled
# venv only on the native cluster; in a job container deps are baked into the image
export VENV=${VENV:-/eph/nvme0/yyd/rzero_venv}
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
# generate.py / question_generate.py do `import evaluation.datasets_loader` (package-style),
# which needs the repo root on PYTHONPATH (the script's own dir alone is not enough).
export PYTHONPATH=${RZERO_CODE_DIR:-/home/azureuser/yyd/R-Zero}:${PYTHONPATH:-}

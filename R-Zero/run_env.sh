# Source this before any R-Zero run. Big artifacts live on the 27TB nvme0 disk.
export STORAGE_PATH=/eph/nvme0/yyd/R-Zero_run
export HUGGINGFACENAME=yuyang322
export HF_HOME=/eph/nvme0/yyd/hf_cache
export HUGGINGFACE_HUB_CACHE=/eph/nvme0/yyd/hf_cache/hub
export VLLM_DISABLE_COMPILE_CACHE=1
export WANDB_MODE=disabled
export VENV=/eph/nvme0/yyd/rzero_venv
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
# generate.py / question_generate.py do `import evaluation.datasets_loader` (package-style),
# which needs the repo root on PYTHONPATH (the script's own dir alone is not enough).
export PYTHONPATH=/home/azureuser/yyd/R-Zero:${PYTHONPATH:-}

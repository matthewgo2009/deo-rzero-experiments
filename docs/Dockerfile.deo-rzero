# deo-rzero verl-training runner image — reproduces the proven native env
# (torch 2.7.0+cu126, vllm 0.9.1, flash_attn 2.7.4.post1). R-Zero repo (patched verl)
# and DEO code are MOUNTED at runtime, not copied, so edits need no rebuild.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN python -m pip install --upgrade pip "setuptools<81" wheel

# staged install: torch first, then matching flash-attn wheel, then the rest
RUN pip install --prefer-binary torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 numpy==2.2.6
RUN pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
COPY requirements_gpu03.txt /tmp/requirements_gpu03.txt
RUN pip install --prefer-binary -r /tmp/requirements_gpu03.txt && pip install stopit
ENV VLLM_DISABLE_COMPILE_CACHE=1
WORKDIR /workspace

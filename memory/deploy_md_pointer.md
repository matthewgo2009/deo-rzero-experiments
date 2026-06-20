---
name: deploy-md-pointer
description: DEPLOY.md at /home/azureuser/yyd/DEO/DEPLOY.md is the canonical project doc; read it first when onboarding
metadata: 
  node_type: memory
  type: reference
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

`/home/azureuser/yyd/DEO/DEPLOY.md` is the authoritative DEO deployment + operations
guide. Read it before answering any structural question about the project — it covers
hardware layout (8× H100), the 6-container GPU plan, the 4 critical verl patches we
applied (KL ref pinned to base), known gotchas (filter rate < 512, Ray "0 GPUs", HF
dataset cache poisoning, vllm_solver hot-reload), and the broken-KL vs KL-fixed paper
data table.

When the user asks about deployment, config knobs, or troubleshooting, derive answers
from DEPLOY.md rather than guessing — it's the single source of truth and the user
maintains it. If something in DEPLOY.md is stale relative to the code (paths, GPU
layout, container names), tell the user so they can update both in sync.

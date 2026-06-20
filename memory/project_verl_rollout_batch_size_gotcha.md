---
name: project-verl-rollout-batch-size-gotcha
description: verl default rollout_batch_size=512 silently kills any pipeline with <512 filtered training entries; override to 64 lets small datasets train at cost of noisier gradients
metadata: 
  node_type: memory
  type: project
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

verl's `data.rollout_batch_size` defaults to **512** in R-Zero's
`examples/config.yaml`. Any solver_train.sh / verl invocation that hands it
a HuggingFace dataset with fewer than 512 examples will crash at startup
with:

```
File "/workspace/verl/trainer/data_loader.py", line 83, in create_dataloader
    assert len(train_dataloader) >= 1
AssertionError
```

because `len(train_dataloader) = floor(N_entries / rollout_batch_size)`,
which is 0 when N < 512.

This is DEPLOY.md gotcha #1, hit by:
- Broken-KL DEO run on iter 4 baseline (filter pass 476/1500).
- R-Zero on Qwen3-4B-Base at iter 2 (filter pass 126) and iter 4 (17).

**Fix:** add `data.rollout_batch_size=64` (or smaller) to the verl command-line:

```bash
python3 -m verl.trainer.main config=examples/config.yaml \
    data.rollout_batch_size=64 \
    ...
```

Lower batch means each step samples fewer distinct prompts → noisier gradient.
On a 74-entry dataset (R-Zero iter 3) with batch=64 and `worker.rollout.n=5`
and `max_steps=20`, each prompt is sampled ~17× across training — heavy
over-sampling, training is effectively memorizing the tiny dataset. Eval acc
in this regime is dominated by ±2.2% MATH-500 noise, not by training signal.

**When to use:**
- Always include this in solver_train.sh when adding new filters that might
  drop the dataset below 512.
- For DEO, the regex+phat+judge filter cascade currently leaves ~800-1000
  per iter so the 512 default is fine. But if filters get stricter, override.

Related: [[project-rzero-vs-deo-findings]] (where this gotcha surfaced).

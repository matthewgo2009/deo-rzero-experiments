"""Orchestrator: schedule (model x dataset) general-eval workers across 8 GPUs.

Env:
  GEVAL_MODELS   semicolon list of label=path (path may be an HF id or local dir)
  GEVAL_DATASETS comma list (default mmlu_pro,supergpqa,bbeh)
  STORAGE_PATH   output root; per-task json under geval/, summary geval_summary.json

Each worker is a subprocess pinned to one GPU (offline vLLM, TP=1). One retry per
failed task; a task that fails twice is reported as missing, never fabricated.
"""
import json
import os
import subprocess
import time

STORAGE = os.environ["STORAGE_PATH"]
DATASETS = os.getenv("GEVAL_DATASETS", "mmlu_pro,supergpqa,bbeh").split(",")
N_GPU = int(os.getenv("GEVAL_GPUS", "8"))


def parse_models():
    out = []
    for spec in os.environ["GEVAL_MODELS"].split(";"):
        spec = spec.strip()
        if not spec:
            continue
        label, path = spec.split("=", 1)
        out.append((label, path))
    return out


def main():
    models = parse_models()
    tasks = [(lb, p, ds) for lb, p in models for ds in DATASETS]
    outdir = f"{STORAGE}/geval"
    os.makedirs(outdir, exist_ok=True)
    pending = [(lb, p, ds, 0) for lb, p, ds in tasks
               if not os.path.exists(f"{outdir}/{lb}_{ds}.json")]   # resume-safe
    print(f"[geval] {len(tasks)} tasks total, {len(pending)} to run", flush=True)
    running = {}          # gpu -> (proc, task)
    failed = []
    while pending or running:
        for gpu in list(running):
            proc, (lb, p, ds, tries) = running[gpu]
            if proc.poll() is None:
                continue
            del running[gpu]
            if proc.returncode != 0:
                if tries < 1:
                    print(f"[geval] retry {lb}/{ds} (rc={proc.returncode})", flush=True)
                    pending.append((lb, p, ds, tries + 1))
                else:
                    print(f"[geval] FAILED twice: {lb}/{ds}", flush=True)
                    failed.append((lb, ds))
        while pending and len(running) < N_GPU:
            gpu = next(g for g in range(N_GPU) if g not in running)
            lb, p, ds, tries = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
            log = open(f"{outdir}/{lb}_{ds}.log", "a")
            proc = subprocess.Popen(
                ["python3", os.path.join(os.path.dirname(__file__), "general_eval.py"),
                 "--model", p, "--dataset", ds, "--label", lb,
                 "--out", f"{outdir}/{lb}_{ds}.json"],
                env=env, stdout=log, stderr=log)
            running[gpu] = (proc, (lb, p, ds, tries))
            print(f"[geval] gpu{gpu} <- {lb}/{ds}", flush=True)
        time.sleep(20)
    # summary
    summary = []
    for lb, p, ds in tasks:
        f = f"{outdir}/{lb}_{ds}.json"
        if os.path.exists(f):
            r = json.load(open(f))
            summary.append({"label": lb, "dataset": ds, "n": r["n"], "acc": r["acc"]})
    with open(f"{STORAGE}/geval_summary.json", "w") as f:
        json.dump({"summary": summary, "failed": failed}, f, indent=2)
    print("[geval] SUMMARY", flush=True)
    for row in summary:
        print(f"  {row['label']:28s} {row['dataset']:10s} {row['acc']:6.2f} (n={row['n']})",
              flush=True)
    if failed:
        print(f"[geval] missing after retries: {failed}", flush=True)


if __name__ == "__main__":
    main()

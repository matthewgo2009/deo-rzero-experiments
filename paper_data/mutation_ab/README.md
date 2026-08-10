# Mutation-prompt A/B (V1 aggressive vs V2 conservative) — question data for external analysis

Runs: `icy_sprout_k8hzk5hmtq` (V1) / `boring_soca_dq3t4952ry` (V2). Qwen3-4B, MODE=curriculum,
fixed β=0.1×5, walk on (MCMC_STEPS=5), 2000-q pool, no CD, KL→base. The ONLY difference between
the arms is the mutation prompt (`DEO_MUT_PROMPT=v2`); prompts live in `DEO/mcmc_deo_vllm.py`
(V1 = "STRUCTURALLY DIFFERENT" aggressive; V2 = "one localized step, stay close, validity rules").

## Files

`V1_final_pool.jsonl`, `V2_final_pool.jsonl` — one line per pool slot per iteration
(2000 × 5 iters = 10,000 rows per run). This is the **end-of-walk state** of every question.

| field | meaning |
|--|--|
| run | V1 / V2 |
| iter | 1–5 |
| pool_index | slot index in the 2000-question pool |
| question | final question text of that slot (post-walk) |
| mutator_gt | the mutator/challenger's proposed answer for that question |
| p_hat | solver self-consistency (modal count / 9, temp 1.0) |
| r_unc | 1 − 2·\|p̂ − 0.5\| |
| pseudo_label | solver's modal answer (training label if kept) |
| passed_phat_filter | p̂∈[0.3,0.8] AND pseudo_label non-null |
| entered_training | passed filter AND survived the LLM validity judge (present in filtered_*.json) |

Totals entering training: V1 = 6137/10000, V2 = 5165/10000 (matches per-iter upload logs).

## ⚠️ Per-proposal walk records are NOT available for these two runs

The requested per-proposal fields (old_question → new_question, strategy A–E, accepted, alpha /
delta_energy, old/new p̂, per-proposal solver samples) were written to per-iter MCMC walk logs
(`$DEO_STORAGE/logs/mcmc_iter_*.log`), but a persistence bug meant `DEO/logs` was never synced to
the job output — the logs died with the compute nodes. The bug is fixed
(azureml/run_pipeline_job.sh now persists `DEO/logs`), and `parse_walk_logs.py` in this folder
converts walk logs into per-proposal JSONL for any FUTURE run. Fields that additionally need a
logging extension (question index k, mutator_gt per proposal, p̂ old/new, pseudo_label per
proposal, the 9 sampled solver answers) require a small patch + re-run of the two arms.

## Analysis context (see BUCKET_ANALYSIS.md tail section)

- Accuracy (Claude 7-set): V1 48.79 @i5, V2 48.83 @i5 — tie; V2 mean higher (47.41 vs 47.05), stabler early.
- Pools: V1 in-band[.3,.8] ~69%, V2 ~56% (V2's small localized steps = weaker walk mixing; fatter tails both sides).

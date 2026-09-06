# Weakness Memory v2.1 — R1–R8 逐项回复(对 WEAKNESS_MEMORY_REVIEW_3BBE09D.md)

审查基线 `3bbe09d`;本轮修复 commit 见 git log(本文件同 commit)。全部改动在
`DEO/mcmc_deo_vllm.py`、`DEO/weakness_memory_tests.py`、`DEO/weakness_memory_smoke.py`、
`azureml/run_pipeline_job.sh`。测试分级:**[unit]** 单元、**[regress]** R1–R8 反例回归、
**[wiring]** 真实生产函数接线(mock 服务)、**[tokenizer]** 真实 Qwen tokenizer 预算边界
(本地已跑通,节点上再跑)。语义质量不在离线测试的证明范围内——由 GPU smoke + 人工抽样负责。

| # | 状态 | 实现 | 测试 |
|---|---|---|---|
| R1 | 已修 | `summarize_global_weakness_memory`:domain 内先做**全域**同名 pre-merge(与 chunk 无关)→ LLM map/reduce → **全域**同名 post-merge(`_exact_merge_groups`,来源并集、代表取最大子组);fallback 不再受 chunk 限制,因为同名合并发生在 chunk 化之前/之后的全域步骤 | `test_R1_cross_chunk_merge`:201 条反例(0/100/200 同名)在纯 fallback 下得到 support=3 的组;打乱输入顺序后成员集合与 support 不变 |
| R2 | 已修 | `_render` 返回 `(prompt, chars, shown_details)`,`_shown_details` 按本次预算重截 trace;`_parse_weakness_note` 只对 **shown** 校验;新 `_norm_cite` 仅折叠空白、保留大小写/符号/数字(与分组用的小写 `_norm_ws` 分离);实际展示可由 raw trace + 记录的 `trace_chars` 确定性重建 | `test_R2_citation_against_shown`:被截掉的引用端到端 parse_failed;同预算下可见引用通过;`H`/`h` 互换不通过 |
| R3 | 已修 | merge prompt 要求 `representative`+`reason`;程序校验 rep ∈ 该组 indices,无有效 rep 的组**不强行合并**(成员守恒地进 unassigned);reduce 渲染与最终 `representative_evidence` 都用选出的 rep(`rep_method`: `llm_selected` / `exact_name`,同名合并取最大子组的 rep);membership 校验≠语义正确,报告如此表述 | `test_R3_representative`、`test_summarize_traceability`(rep=模型选的 viet[-1],非 src[0]) |
| R4 | 已修 | writer:`_render` 对**最终完整 prompt**(含 chat template+输出预留)测 token;floor 后仍超限 → `input_too_long`,不进 batch;重试只针对失败项(pending 机制,index 对应保持);summarizer:`_merge_notes_llm` 在最终渲染后再测,超限**二分**而不发送;不改 counts/p̂/伪标签定义 | `test_R4_oversize_isolation`:同批超长项+正常项,正常项成功;reply 内断言每条实际发送 prompt 都在预算内;`[tokenizer]` 真实 tokenizer floor 检查(unit5) |
| R5 | 已修 | `_extract_json_value` 改用 `JSONDecoder.raw_decode`(字符串内 `{`/转义安全);`_valid_index`:只收 int 与纯数字串,显式拒绝 bool/float(不截断转换);`indices` 非 list → 组级结构异常,成员守恒进 unassigned,无 TypeError;统计口径:仅代码修复计 `llm_repaired`,模型自报 unassigned 计 `llm_success` | `test_R5_json_and_indices` 覆盖审查表格三例 + 统计口径 |
| R6 | 已修 | `_ordered_completion_texts` 接入 **solver shard**(`_run_shard` 内,缺/重/越界告警,缺槽记 None→invalid answer,p̂ 分母不变)、**init 出题**(topic 对齐保留)、**mutation**(缺槽=该槽 malformed,后续槽不前移)、writer、classifier;helper:有 index 但全非法**绝不**回退列表顺序,重复 index 保首个并计数 | `test_R6_helper`(index=99/n=1、重复);`test_R6_evaluate_wiring`:**真实** `evaluate_r_unc_vllm`,乱序 choices → p̂=[1,1],缺一条 → 该题 p̂=2/3 且不串题;mutation 接线由 `test_wiring_walk_enabled` 覆盖 |
| R7 | 已修 | `WmLogger` 每实例领新 attempt(`.a{k}` 文件,不混覆盖/追加),event_id 含 attempt 全局唯一;raw rollouts 与 writer 原始输出**全量**保存(仅压缩,不截断);meta 事件记录 model/abbr/prompt 版本/AZUREML_RUN_ID/DEO_CODE_VERSION;summarize 接收 logger,每次 merge 调用记 call_id/stage/domain/成员来源映射/原始响应/修复结果;filter_and_push 为汇总阶段开新 attempt logger | `test_R7_logger_attempts`:同 iter 两次启动 → a1/a2 文件、event id 不冲突、30000 字符 trace 全量回读、merge 事件含 member_gis/outcome/raw_output |
| R8 | 已修 | `run_wm_smoke`:离线测试**在 pipeline 内执行**且失败即 return 非零;smoke 退出码存 `WM_RC`,清理与 sync 照常执行但不覆盖状态;顶层 `case` 后 `MODE_RC` 非零 → 打印 FAILED 并 `exit $MODE_RC`(EXIT trap 的最终持久化仍运行) | bash 语法检查 + rc 传播形状验证(`(exit 7)` → 顶层 7);smoke 三态(0/非零/启动失败)在节点上由本次 smoke job 实证 |

## 测试报告分级

- 原有测试(更新语义后保留):unit1–unit5
- 新增反例回归:R1/R2/R3/R4/R5/R6a/R7
- 生产函数接线:`test_R6_evaluate_wiring`(真实 evaluate)、`test_wiring_walk_enabled` /
  `test_wiring_walk_disabled`(真实 `generate_batch_mcmc`,覆盖 accept 替换、reject 保留、
  accepted-no-note 清 note 留 event、`_n_accepted`、路由 reason、event 唯一性、关闭路径零
  wm 调用 + 无私有字段 + MH 判定相同)
- 实际 tokenizer 检查:unit5(本地已用真实 Qwen tokenizer 跑通 floor 预算)
- 真实 GPU smoke:MODE=wm_smoke(修复后重提交),机器检查覆盖**全部** records 的
  target source-iter、event 唯一性、support/avg_p̂ 重算、representative 归属;
  guided 样例按固定种子抽样并区分"接受过 mutation"与"仍是原 seed"

## 措辞修正

- `RECOMPUTE_REPORT.md` 中"events 已记录逐次 merge"的表述在 v2 时不成立,本轮已实现并
  测试(见 R7);报告已加注。
- 所有"验证"均指来源归属/守恒等**程序可证**性质;语义正确性(note 是否真describe分歧、
  合并是否同一能力、题目是否考目标)只能由真实样本人工抽查,mock 通过不构成语义验收。
- mock 反例证明"条件触发时会出错",不证明旧实验的串题根因;旧数据归因仍待新 run 的
  raw rollouts。

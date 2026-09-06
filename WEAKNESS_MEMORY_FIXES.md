# DEO Weakness Memory：修复任务与验收要求

给 Claude 的实现交接文档。目标是修复现有最简 memory 方案的证据质量、汇总和 mutation 引导，不扩展成复杂的 agent/memory 架构。

## 1. 审查基线与实现范围

- 仓库：`matthewgo2009/deo-rzero-experiments`
- 分支：`curriculum-deo-claude-experiments`
- 本次审查的数据与代码版本：`8cd1a815571a3d5241e47d9f12b90c983e0b2732`
- 数据目录：`paper_data/weakness_memory_4b/`
- 主要实现：`DEO/mcmc_deo_vllm.py`
- 实验：Qwen3-4B-Base，2000 题/iter，m=9，5 步 mutation，5 个训练 iter，fixed beta=0.1，no CD，V1 mutation。

先比较当前分支和上述审查版本，确认哪些问题已修。不要为了复现旧版本覆盖后续工作。

保留原有流程：本轮 frozen solver 的现有 m 个 rollouts → 每题一条 note → 每条链只保留最终状态的 note → 完整过滤后汇总 → 下一轮冻结 global memory。

本次修复保持 reward、p_hat 定义、beta、MH acceptance、walk 步数、训练预算不变。Memory 仍只影响 mutation proposal，不把“是否符合 target”加入 reward 或强制拒绝规则。不要因此声称恢复了精确 MH/detailed balance。

## 2. 已确认的问题与证据

### 2.1 Local notes 有串题，也有把坏题当作弱点的情况

下列例子都来自最终 pool，且对应题目已进入 filtered training set。

| iter / chain | 实际题目 | note 中的问题 |
| --- | --- | --- |
| 1 / 530 | 三次多项式求 f(2) | evidence 讲 quarts 到 cups 的换算 |
| 1 / 973 | 6×4×3 长方体内接球 | evidence 讲宽度 5 的花园及内接圆 |
| 4 / 119 | n² ≡ 1 (mod 15) | evidence 讲向量张成空间 |
| 4 / 549 | gcd(a,b)·lcm(a,b)=ab+k，k>0 | 条件与恒等式矛盾，却被记作 solver 不理解 gcd/lcm |

也有可用例子：iter4 / chain247 的同余题，note 指出答案分成 23、38、224，分歧是确定满足多个同余条件的最小正整数。这个描述与题目相关，但仍需要原始 rollouts 才能核验这些答案是否真实出现。

五轮过滤后共有 4444 条非空 notes，其中 1063 条 evidence 含 correct/incorrect/correctly/incorrectly/wrong/erroneous 等词，约 23.9%。这只是判断性用语的词面统计，不是错误率，也不应直接成为关键词删除规则。

**目前没有导出 writer 实际看到的 solver traces，不能确认串题来自 writer 幻觉、solver 跑题还是上游批量响应对应关系。先补证据，不要直接把某一种原因当作已查明的 bug。**

### 2.2 Global summary 存在错误合并与异常丢失

iter1 的例子：

- `memory_1`：global label 为 Calculating expected value；代表 evidence 对应 chain10，其原始 note 是 algebra / determining variable values。
- `memory_6`：global label 为 multiple congruences；代表 evidence 对应 chain9，其原始 note 是 geometry / calculating exterior angles。

重算口径：先将完整 pool 与 filtered dataset 对齐，再统计其中非空的最终 note。

| iter | 过滤后有 note 的题数 | global 项数 | global support 合计 |
| --- | ---: | ---: | ---: |
| 1 | 879 | 10 | 275 |
| 2 | 885 | 10 | 162 |
| 3 | 890 | 10 | 74 |
| 4 | 889 | 10 | 103 |
| 5 | 901 | 3 | 14 |

Top-10 本来就可以只覆盖部分 notes，不能仅凭覆盖率低断言有 bug。但 iter5 仍有 18 条完全同名的 modular arithmetic、14 条 solving quadratic equations，global 却只剩三项代数描述。因此不能把 global 中某技能消失解释成 solver 已掌握它。

README 的“所有 LLM merge 调用失败，全部走 exact-string fallback”与导出数据不一致：纯 fallback 不会创造新的跨领域标签；对同一批输入重算纯 fallback，iter5 应得到 10 项，而非当前 3 项。需要核对运行代码版本和真实调用日志。

代码中已确认的风险：

- `_summarize_chunk_llm()` 只要解析出部分合法组就返回，未归组的输入可静默消失。
- 每个中间 chunk 被限制为最多 10 组，容易提前丢弃或过度合并不同能力。
- reduce 将所有 provisional groups 一次送入模型，没有实际 token 预算控制。
- 合并模型只看到 domain 和 weakness，没有 evidence。
- `representative_evidence` 直接取组内第一条；输出没有保留成员 chain IDs，无法完整追溯。

### 2.3 Mutation guidance 的执行不稳定

抽查的是已经接受过至少一次 mutation 的 guided chains，不把原封不动的 seed 算作 mutator 忽略指令的证据。

- iter2 / chain1795：target=expected value，最终题确实是骰子收益期望。
- iter3 / chain1275：target=3D distance，最终题是导数定义的极限。
- iter2 / chain104：把抽彩球期望与 3^n ≡ 1 (mod 7) 生硬拼接，过程与停止条件不清楚。

因此应描述为“有时遵循、有时忽略或生硬拼接”，不能写成“guidance 完全没有进入题目”。接受率相近也不等于题目分布相同。

## 3. P0：让每条 local note 可核查

涉及函数：`evaluate_r_unc_vllm()`、`_build_cluster_details()`、`generate_weakness_notes_batch()`、`_parse_weakness_note()` 和 proposal 日志。

### 3.1 保留现有 rollout 预算，修正证据输入

1. 继续复用本次评价该题的 m=9 个 solver rollouts，不为了 memory 额外调用 solver。
2. 保留全部有效答案 cluster 的 answer/count，以及 invalid count。当前代码只保留 top-3 的 counts，需要补全；不要改变既有答案归一化和 p_hat 计算。
3. writer 仍只看 top-3 clusters 各一条代表 trace，保持输入紧凑；每条 trace 带 cluster_id、rollout_id 和截断标记。
4. 本地日志保存这 m 个原始 response 文本和抽取答案，writer 实际看到的截断版本可另存或通过确定性截断重建。无需把全部原文加入 prompt。
5. 核对 batch response 的数量、index 与 request 的对应关系；不能仅假定列表顺序。缺失响应必须单独记为失败/重试，不能令后续题目的 note 顺移。
6. 在每个 writer 请求前按实际 tokenizer 检查输入+输出预算；超限时缩短 traces，保留题目与完整 counts，并记录实际截断。

### 3.2 Writer 允许不诊断；记录“分歧”，不臆断对错

给现有 note 增加最小状态信息：

- `status=weakness`：能指出不同回答在某个具体推理步骤上的分歧。
- `status=problem_issue`：题目疑似矛盾、信息不足、对象未定义或表述破损。
- `status=insufficient_evidence`：只能识别题型，或 traces 不足以定位推理分歧。

后两种状态保留审计记录，但不进入 global weakness memory。若为兼容旧接口仍返回 `note=None`，必须在旁路日志保留 status/reason。

`weakness` 用一句话描述可迁移的推理操作。例如“区分有序和无序计数”，而不只是“combinatorics”或“solving equations”。无需预定义细粒度 skill taxonomy。

`evidence` 说明哪些 clusters 在哪个步骤分歧，并带来源引用，例如 `cluster_id + rollout_id + 短原文摘录`。摘录必须出现在 writer 实际输入的 trace 中；允许引用题目数值与答案。把“不要复制具体常数”的约束仅用于 weakness 标签，不能妨碍 evidence 可核查。

Writer 指令必须明确：

> 不把多数答案当作 verified answer；只描述可见分歧。先检查分歧是否可能来自题目本身的缺陷。不能从 traces 定位分歧时返回 insufficient_evidence。不得补写输入中没有的题目、数字、推理或答案。

结构校验负责验证引用确实存在、ID 属于当前请求、必要字段完整；这不能替代语义审查，也不能证明题目有效。不要只靠禁止 correct/wrong 等单词来保证质量。

### 3.3 补充足够的溯源字段

每次评价/写 note 使用稳定的 `event_id`，并保存：

| 数据 | 必要字段 |
| --- | --- |
| 评价事件 | run_id、iter、chain_id、step、event_id、question/hash、solver checkpoint、m、p_hat、答案 counts、response 文件引用 |
| Writer 事件 | event_id、writer model、prompt version、输入引用、原始输出、重试/异常、解析结果、status |
| Walk 事件 | event_id、target 的来源 iter/id、evaluated、accepted、最终采用的 state/event_id |

区分 `not_evaluated`、`not_eligible`、`request_failed`、`parse_failed` 和模型给出的诊断状态。现有 `domain=null` 同时可能表示不满足 eligibility 和 writer 失败，不能继续把它统计成 parse failure。

本次审查未发现记录层面的 accept/reject note 状态不一致，保留其语义：接受后 question/p/note 一起更新；拒绝后保留旧状态；接受了没有可信 note 的新题，应清空旧 note。每条链最终只贡献一条观察。

对疑似坏题，先禁止其进入 weakness summary 并记录原因。本轮不要静默改变训练过滤器；如要同步增强训练题 validity filter，请作为独立改动/实验报告，避免混淆 memory 效果。Writer 的 problem_issue 也不能当作数学 oracle。

## 4. P0：修复 global summary，保留每次合并的来源

涉及函数：`_summarize_chunk_llm()`、`summarize_global_weakness_memory()`。

### 4.1 按上下文预算处理，中间层不强制 top-10

- chunk 大小由真实 token 数决定，满足 input_tokens + reserved_output_tokens <= context_limit；不要假设固定 100 条一定超限或一定安全。
- map 和 reduce 都使用预算控制；reduce 过长就分批继续合并，不把全部 provisional groups 硬塞入一次请求。
- 同一具体能力才合并。中间层允许超过 10 组，或显式留下未分组项；只在最终排序时取最多 10 项。
- 按现有粗 domain 分桶有助于防止跨领域误合并。不同 domain 默认不合并；`other` 不作为任意合并的通道。
- 合并输入包含简短 evidence/source excerpt，避免只凭“congruence”等词误合并角度与同余方程。

### 4.2 输入成员不能静默消失

每一级输出同时记录 `groups` 与 `unassigned`。实现以下不变量：

1. 组成员 ID 必须来自本级输入；同一输入成员不能重复归组。
2. 分组成员与 unassigned 不重叠；两者并集必须等于本级全部输入。
3. 未分配项作为单独 provisional group 继续传递，不能静默丢弃；只在最终 support 筛选和 top-10 排序时裁剪，并保留理由。
4. 缺失、重复、越界 ID 的返回不可作为完整成功；确定性地修复为 unassigned 或重试，再走保守 fallback。
5. fallback 只合并相同 domain 下规范化后完全一致的 weakness；不发明标签。

对日志分别统计 LLM success、partial/repaired、request failure、parse failure、fallback。保留每次输入/输出、token 数及调用代码版本，查明旧 README 归因为何与文件不符。

### 4.3 support 与 evidence 都必须可追溯

每个最终 global item 增加 `source_event_ids`（可通过它们定位 iter/chain/question）。

- `support` = 该组不同 `(iter, chain_id)` 的最终合格 notes 数，不是 proposal 数、rollout 数或重复出现次数。
- `avg_p_hat` 从上述同一成员集合的最终 p_hat 用 Python 重算。
- 从确实支持该组标签的成员中选 representative evidence，保存它的 source_event_id；先验证 source 属于该组。不要固定取第一条，也不要重新编造 evidence。
- 排序后保留 top-10、support>=3；其余组的来源和被裁剪原因留在审计输出中。
- 保留原有每轮重建、下轮冻结语义；某类消失只表示本轮未进入输出，不能解释为能力已掌握。

旧数据缺少 rollouts，能够重做的是对“已保存 notes”的聚类和溯源，不能把重汇总结果标记为 rollout-verified memory。保留旧导出，另存修复后的汇总以便对比。

## 5. P1：让 memory guidance 与 seed 兼容

涉及函数：`sample_target_memory()`、`weakness_guidance_block()` 及 V1 mutation prompt。

1. 在每条链初始题上确定一个粗 domain，可复用可信初始 note 的 domain；其他情况可以复用现有逐题检查或一次批量轻量分类。不要把保存的初始 `topic` 字段当作最终变异题的分类。
2. 仍以 0.8 概率尝试 guidance，但只从与 seed domain 兼容的 global items 中，按现有 support/avg_p_hat 权重选一项；没有可靠 domain 或兼容项则 unguided，记录 fallback reason。
3. 目标在该链 5 步内固定；global memory 在本 iter 内固定。`memory_1` 等 ID 每轮复用，必须连同 memory_source_iter 一起存，避免用错上一轮/本轮 memory。
4. Prompt 要求在原题的合理数学结构内，使解题确实需要目标推理操作。若无法合理结合，允许正常变异并记录不匹配；不要硬塞关键词、引入矛盾条件或泄漏答案。
5. 除 weakness 标签外，可提供一条已核验、简短的分歧描述，帮助模型理解要针对的操作；不喂未经核验的旧 evidence。
6. 兼容筛选只改 proposal 的上下文分配。不要新增 target-match reward 或为了达标强行接受/拒绝 proposal。

这里的兼容路由是对原始最简设计的修正，不应全部归因于实现偏离规范。继续使用粗 domain 即可，不需要技能树、向量数据库、跨题在线复杂状态或额外 solver rollouts。

Memory 的随机选择使用独立且显式设种子的 RNG，避免仅采样一个 target 就额外消耗现有 MH/策略随机数。关闭 memory 时应保留原有执行路径。

## 6. 验收与复现实验

### 6.1 先做离线验收

- **响应归属**：用乱序、有缺项的 mock batch responses 验证 question、rollout、note 不串位；缺项不会静默消失。
- **证据输入**：超过 3 个答案 clusters 的例子仍保留全部 counts，代表 traces 只取 top-3；验证其余/无效答案计数及 p_hat 未变。
- **状态更新**：accept 更新全部状态，reject 保留旧状态，accepted/no-note 清空旧 note；最终 summary 每链只计一次。
- **合并守恒**：覆盖重复/越界/遗漏 IDs、部分合法 JSON、长输入、fallback；检查 source 集合、support、avg_p_hat、代表 evidence 来源。
- **真实回归例子**：将上述串题、矛盾题、跨领域合并作为人工验收 fixtures；若缺原始 trace，仅测试“记录与题目不一致”，不能伪造缺失的 rollout 证据。
- **归因核对**：用旧数据重算纯 exact-string fallback，输出与旧 global 的差异；不要以写死“iter5 必须 10 项”的测试替代对一般算法的验证。
- **关闭开关**：memory disabled 时不调用 writer/summarizer，原有 prompts、reward、p_hat、acceptance 路径及随机数使用不变。

### 6.2 小规模检查通过后再跑完整训练

先做约 100 题、沿用 5 步的 smoke run，保存可复核的完整日志。至少验证一次“global 写出 → 下一轮加载 → guided mutation”的闭环；可以固定 solver 跑两轮小规模出题，暂不进行训练。检查 local note 是否有证据、summary 来源是否守恒、guided mutation 是否产生合理题目，再决定完整 2000 题实验。不要只用 JSON parse rate 判定修复成功。

人工抽样规则写入报告：固定随机种子；local notes 分层抽查各状态；guided chains 将“至少接受过一次 mutation”和“仍是原始 seed”分开。旧审查的 30 条 local notes、40 条 guided chains 只是定性抽查，不是全量正确率。

建议报告：

| 层次 | 必报指标 |
| --- | --- |
| Writer | eligible 数、调用/解析失败数、三种 status 数；人工题目相关性、trace 支持性、具体操作可识别性；疑似坏题数 |
| Summary | 输入 notes、归组/unassigned/裁剪数、LLM/fallback 次数、各组 distinct-chain support、来源与标签一致性 |
| Mutation | 实际 guided 比例、无兼容 target 比例、接受过 mutation 的链数、目标匹配/部分匹配/不匹配，以及独立的题目有效性判断 |
| 结果 | guided/unguided 的 p_hat、带内率、训练存活率、题长；完整训练时再报每 iter 的 MATH500 和 AVG7 |

不得仅以 acceptance 相近推断分布没有移动。目标匹配也不等于题目有效或已带来训练收益。Guided/unguided 的描述性对比存在题目/领域构成差异，不直接解释为因果效应。

README 还应修正：单次 iter1 的差值不能建立“±0.85 噪声范围”；原表 iter2 差值为 -1.85，iter3 为 -0.91，也不满足“所有差值都在 ±0.85 内”。没有多 seed 结果时，只报告观察到的分数与不确定性，不作显著性结论。

## 7. 请交付

1. 修复代码与上述必要测试，列出相对审查版本的修改。
2. 旧数据离线重算报告：复现已确认案例；明确哪些原因仍因缺日志无法确认。
3. 新日志 schema、prompt 版本与一组可端到端追踪的 smoke-run 示例。
4. 更新 README，区分事实、代码风险、待验证假设；不要覆盖原始实验数据。
5. 简短说明是否已满足完整实验条件，以及仍存在的具体问题。

优先完成证据溯源和汇总修复，再验证兼容 guidance。不要在这些检查之前用新一轮完整训练替代排错。

## 参考文件

- [实验数据与 README](https://github.com/matthewgo2009/deo-rzero-experiments/tree/8cd1a81/paper_data/weakness_memory_4b)
- [审查版本实现](https://github.com/matthewgo2009/deo-rzero-experiments/blob/8cd1a81/DEO/mcmc_deo_vllm.py)
- [iter1 原始 pool](https://github.com/matthewgo2009/deo-rzero-experiments/blob/8cd1a81/paper_data/weakness_memory_4b/mcmc_iter_1_deo_wm_fb.json)
- [iter4 原始 pool](https://github.com/matthewgo2009/deo-rzero-experiments/blob/8cd1a81/paper_data/weakness_memory_4b/mcmc_iter_4_deo_wm_fb.json)
- [iter5 global memory](https://github.com/matthewgo2009/deo-rzero-experiments/blob/8cd1a81/paper_data/weakness_memory_4b/global_weakness_memory_iter_5.json)

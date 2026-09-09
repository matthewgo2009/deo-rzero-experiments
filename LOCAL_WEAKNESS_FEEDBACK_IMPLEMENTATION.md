# Local weakness feedback：给 Claude 的实现说明

日期：2026-09-09  
仓库：`matthewgo2009/deo-rzero-experiments`  
分支：`curriculum-deo-claude-experiments`  
本说明核对基线：`648dbecd9d7496970490752d6b0d32e2d81d990d`

## 1. 要实现的改动

**在一道题的 MCMC walk 中，用当前被接受题目的 solver rollout 所揭示的弱点，指导下一步 mutation。** 每条 chain 只保留一个当前 note；一轮结束后，仍汇总各条 chain 的最终 note，生成下一轮 global memory。

当前实现已经保存并更新 `pool_note[k]`，但 mutation prompt 只读取固定的 `pool_target[k]`。这次核心改动是让 prompt 优先读取当前 `pool_note[k]`。

新增可切换模式，保留原有路径：

```bash
# 默认：复现当前固定 global target 的行为
DEO_WM_GUIDANCE_MODE=global_fixed

# 新实验：当前 local note 优先，缺失时回退到已有 global target
DEO_WM_GUIDANCE_MODE=local_feedback
```

上述变量是本次建议新增的配置，不是已经存在的接口。总开关继续使用 `DEO_WEAKNESS_MEMORY`。本次支持范围是已经启用 weakness memory 的 plain-MH 路径；CD、bandit 等未支持组合应明确报错或沿用既有显式禁用行为，不能默默忽略所选模式。

## 2. 最小状态和 guidance 选择

每条 chain 的状态：

| 状态 | 含义 | 更新方式 |
|---|---|---|
| `q` | 当前被接受的题目 | proposal 被接受时替换 |
| `w` | 当前题目的可信 weakness note，允许 `None` | 与 `q` 同步替换；拒绝时保留 |
| `g` | 从上一轮 global memory 分配的固定 fallback target，允许 `None` | 本轮内固定 |

`q`、`w` 分别复用 `pool_q[k]`、`pool_note[k]`；`g` 复用 `pool_target[k]`。沿用现有 source event id，不新增历史 note 列表、memory bank 或动态 skill taxonomy。

新模式每次生成 proposal 前选择一次 guidance：

```python
if current_note is not None:
    guidance = current_note
    guidance_source = "local"
elif assigned_global_target is not None:
    guidance = assigned_global_target
    guidance_source = "global_fallback"
else:
    guidance = None
    guidance_source = "none"
```

只附加所选的一条 guidance，不同时拼接 local 和 global。local note 的来源就是当前题目，无需再经过 seed-domain 路由。

**覆盖率口径：**本次最简版本中，可信 local note 可用于任意 chain，不以是否分配到 global target 为前提。原 `MEMORY_GUIDED_PROB` 和 domain 路由继续控制 global fallback 的分配，不额外随机丢弃 local note。因此旧日志中的 `unguided_draw` 仅描述 global target 分配，不能再当作新模式的实际未引导标记。新模式没有固定的 20% 完全未引导 holdout；统计必须以本次实际 `guidance_source` 为准。

## 3. 每一步用哪些 rollout，何时更新

### 初始化

1. 生成初始题目 `q0`，沿用现有 solver 评估，取得本题的 `m` 个 rollout。
2. 用这些 rollout 计算 `p_hat`、pseudo-label 和原 reward。
3. writer 复用**同一组** rollout 的答案簇、簇计数和代表 trace，得到 `w0`。
4. 如果 writer 不产出可信 weakness，令 `w0 = None`。

不为写 note 追加 solver 采样。沿用现有 writer eligibility：例如带外、空 pseudo-label 等情况可得到 `not_eligible`，此时使用 fallback 或普通 mutation。不要在本次实验中同时放宽 writer eligibility 或修改 band。

iter1 没有上一轮 global memory，但仍可以通过初始题目的 local note 引导第一步。因此新模式从 iter1 就可能影响出题。

### 一次 mutation

1. 从当前 `(q, w)` 选择 guidance，保存本次使用的 guidance 快照。
2. 输入 `q + guidance`，生成一个候选题 `q_prime`。
3. solver 为 `q_prime` 生成 **m 个新的 rollout**。它们仅属于 `q_prime`；不能混入旧题、其他 chain 或上个 proposal 的回答。
4. 用这组 rollout 计算候选题的 `p_hat`、pseudo-label 和 reward，执行现有 MH 接受/拒绝。
5. 按下表处理状态。

| 结果 | 下一状态题目 | 下一状态 note | 下一步 guidance |
|---|---|---|---|
| 拒绝 proposal | 原 `q` | 原 `w` | 继续按原状态选择 |
| 接受且得到可信 `w_prime` | `q_prime` | `w_prime` | 优先使用新 note |
| 接受但没有可信 note | `q_prime` | **`None`，清空旧 note** | global fallback 或普通 mutation |

“没有可信 note”包括 `problem_issue`、`insufficient_evidence`、解析/请求失败、上下文超限和 `not_eligible`。这些状态保留在审计日志中，不转换成 weakness。

**“上一步的弱点”准确地指当前 accepted state 的弱点。** 如果上一步 proposal 被拒绝，不能把它的诊断带回旧题。也不能接受了新题却继续使用旧题的 note。

### 参考伪代码

```python
q = initial_question
ev = score_with_m_solver_rollouts(q)       # 已有评估，同一组 m 个回答
w = write_note_if_eligible(q, ev)          # 不增加 solver 调用；可能为 None
g = assigned_global_target_or_none

for step in range(L):
    guide = w if w is not None else g
    q_prime = mutate(q, guide)
    if q_prime is None:                   # 沿用原解析失败处理
        continue

    ev_prime = score_with_m_solver_rollouts(q_prime)
    accepted = existing_mh_accept(q, ev, q_prime, ev_prime)

    if accepted:
        w_prime = write_note_if_eligible(q_prime, ev_prime)
        q, ev, w = q_prime, ev_prime, w_prime
    # 拒绝：保留 q、ev、w，绝不重采样旧题以写 note

return q, ev, w
```

`existing_mh_accept` 是现有完整题池 energy 增量更新的占位符，不是建议把 batch-level energy 改成单题 reward。邻居、repetition 和题池状态仍按原实现更新。

参考伪代码展示只在接受后调用 writer 的省成本实现。**为缩小第一版 diff，可以保留当前“所有 proposal 先写 note，再做 MH”的顺序**；只要拒绝的 note 不更新状态、不进入 global summary，语义相同。日志仍可保存 rejected note 供审计。

若实现 accepted-only writer：先完成当前批次原有顺序的 MH 更新，按 proposal/chain id 收集接受项，再批量写 note；下一步 mutation 开始前必须完成 note 更新。保留这些接受项的原 rollout details 和 raw payload，不能重新调用 solver，也不能提前清除 `_raw` 导致引用和原始回答无法核对。

## 4. Prompt 和 writer

第一版复用现有 `WEAKNESS_GUIDANCE_TMPL`，仅添加 local note 适配：local note 的证据字段是 `evidence`，global item 的字段是 `representative_evidence`。不要因为字段不同而把 local evidence 静默丢掉。

意图是：

> 在当前题目的数学结构内变异，使新题的求解必须用到指定的推理操作。保持条件充分、对象定义清楚、答案明确；不要只抄关键词、常数或答案，也不要给出解题提示。

本次先不同时更换 writer 模型、重写 summary、改变数学答案归并器或增加 target-hit reward。保留现有三态 writer、completion index 对位、引用验证、token budget 和失败处理。

质量判断必须区分：

| 可用于定向训练的描述 | 不足以证明存在具体弱点 |
|---|---|
| 从计数中扣除重复排列时遗漏对称因子 | 答案不一样 |
| 计算首次成功的等待次数时混淆是否计入成功那次 | 期望有分歧 |
| 平方变形后没有检查增根 | 二次方程求解错误 |

这些示例说明诊断应有的具体程度，不表示可以在证据不足时让 writer 推断错误。多数答案不等于已验证答案；缺条件的题目、等价表达式和不同但都正确的推导不能自动视为 solver 弱点。当前 writer 的语义质量仍是限制，应在审计中单列，不能仅以 JSON/引用验证通过宣称诊断正确。

## 5. 一轮结束后的 global memory

保留当前流程：

1. 对最终 pool 执行原 band、pseudo-label 和 validity 过滤。
2. 只从最终通过过滤的题目收集其**当前最终 note**。
3. 沿用现有 domain 分桶、merge、support、top-K 和 source event 守恒规则。
4. 输出 `global_weakness_memory_iter_t.json`，供 iter `t+1` 的 global fallback 使用。

不能把一条 chain 中已经被替换的历史 note 或 rejected proposal 的 note 加入 summary，也不能把本轮 summary 提前反馈给本轮尚未结束的 walk。每条 chain 在最终 summary 中至多贡献其最终题目的一条 note。

## 6. 代码落点和最小日志

主要修改 `DEO/mcmc_deo_vllm.py`：

- 配置：新增 guidance mode，默认保留 `global_fixed`。
- `generate_batch_mcmc` 的 prompt 构造：用当前 `pool_note[k]` 优先选择 guidance。
- `weakness_guidance_block` 或小型适配函数：统一 local/global evidence 字段。
- accepted-state 更新：保留接受时同步换 note、accepted-no-note 清空、拒绝不变的规则。
- `WmLogger`：记录真正使用的 guidance，不再根据 `pool_target[k] != None` 推断实际 guided 状态。

每次 proposal 至少记录以下信息，以便直接回答“这一步是否沿弱点变异”：

| 信息 | 要求 |
|---|---|
| `iter / chain / step / proposal_id` | 能唯一关联本次生成、评估、接受和 note |
| `guidance_mode / guidance_source` | source 为 `local / global_fallback / none`；旧模式可标记 `global_fixed` |
| 生成前的题目、候选题 | 保存原文或可解析到原文的引用；只有 hash 不足以人工审计 |
| 本次 guidance 的 domain、weakness、evidence | 在生成前冻结快照，不能事后用新 note 覆盖 |
| guidance 来源 | local 的源 eval event id；global 的 memory id 和 source iter |
| candidate 的 eval event、p_hat、accept | 与打分用的 m 个 rollout 对应 |
| candidate note status、最终采用的 note event | 能检查拒绝保留、接受替换和清空行为 |

完整题目与 raw rollout 可以保存在压缩审计文件；轻量事件保存引用即可。每个 proposal 均需可追踪，包括不符合 writer eligibility 的 proposal，不要要求 writer 成功才有 proposal id。

沿用旧 `_target_memory_id` 字段时，明确它表示固定 global fallback，而不是每一步实际使用的目标。新增逐步 source 日志后，下游报表应据此计算 local/global/none 的 proposal 数、接受数，以及接受后无 note 的次数。

## 7. 实验和验收

### 固定的实验条件

对照当前 `global_fixed` 模式与新 `local_feedback` 模式。full run 沿用 Qwen3-4B-Base、2000q/iter、m=9、L=5、fixed beta=0.1、无 CD、无 bandit，以及原 mutation prompt、reward、过滤和训练设置。记录实际 solver checkpoint、writer/proposer checkpoint、配置和 seed。

新模式包含 local 优先及覆盖率变化，所以结果首先评价这整个改动，不能把差异全部解释为“逐步更新”的独立因果效应。若后续需要隔离更新本身，再增加“固定初始 local note”的消融；本次不要求先实现该消融。

不要给 MH 添加命中弱点的 reward 或硬约束。本次仅改变 proposal 的输入，保留原 uncertainty/repetition energy 和 beta。新的反馈依赖并没有解决已有 proposal-ratio 缺失问题；结果描述应为近似 MCMC 的实验改动，不宣称获得精确 Gibbs 不变分布。

### 离线验收：覆盖状态流转，不调用真实模型

1. local 与 global 同时存在：prompt 只包含当前 local guidance。
2. local 缺失：使用已有 global fallback；两者均缺失：普通 mutation。
3. proposal 被拒绝：下一步仍使用旧题及旧 note，rejected note 不进入 summary。
4. 接受且得到新 note：下一步使用新题与新 note，source event 正确切换。
5. 接受但 writer 无可信结果：旧 note 被清空，下一步 fallback/none；覆盖 `not_eligible` 和请求失败。
6. writer 复用候选题原有的 m 个 rollout：不增加 solver 调用，打乱 completion 返回顺序也不能串题。
7. 最终 summary 只收集通过过滤的最终 state note；保留既有来源守恒检查。
8. `global_fixed` 默认路径与原行为兼容；日志快照不能被后续 note 更新覆盖。

### 小规模机制验收

先用固定 solver 和相同初始题池做 100q × 5 步的两个模式对照，暂不训练。共享初始题目及初始评估结果；相同 m 和相同 proposal 尝试上限，记录实际 solver/writer 调用数，不宣称生成文本或整条随机轨迹会完全相同。

输出机器状态检查结果，以及按预先固定随机种子抽取的最多 50 个 **local-guided 且 accepted** 的 step 样本。每个样本展示“旧题 → 实际使用的弱点及证据 → 新题”，单独标注：

- 目标是否具体、是否有 trace 证据；泛泛目标标记 `unjudgeable_target`，不能自动算命中。
- 新题是否有效、条件是否充分。
- 相对目标是 `exact_operation / domain_only / miss`。
- 旧题是否已经需要同一操作；新题是保留、强化还是丢失该操作。不能仅因新题命中就宣称 mutation 新增了针对性。

这是人工语义审计口径，不新增在线 judge 到接受率。报告抽样方法和分母，不能把 prompt 注入成功、MH 接受或过滤通过直接当作 target-hit 成功。

## 8. Claude 交付内容

交付最小代码 diff、上述离线状态测试结果、配置说明和机制审计格式。具备运行环境后按上述固定 solver 小实验验证，再沿用项目已有流程安排完整训练。

实现成功的判据是：**当前题目的诊断实际进入下一步 prompt；接受/拒绝正确维护题目与 note 的对应关系；同一组 m 个 rollout 完成打分和诊断；日志足以验证真实的变异方向。** 准确率是否提升由后续实验判断。

## 参考

- [核对基线的主实现](https://github.com/matthewgo2009/deo-rzero-experiments/blob/648dbecd9d7496970490752d6b0d32e2d81d990d/DEO/mcmc_deo_vllm.py)
- [上一版完整实验数据与设置](https://github.com/matthewgo2009/deo-rzero-experiments/tree/648dbecd9d7496970490752d6b0d32e2d81d990d/paper_data/weakness_memory_v2_4b)

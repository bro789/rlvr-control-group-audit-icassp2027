# 预注册附录：CodeTAT-QA 作为第二任务（配置①）

> 冻结时刻：2026-07-24 00:2x，**在任何 CodeTAT-QA 训练/评测跑之前**。
> 本附录只声明"任务实例化"的部分；Gate 0/1/2 的全部判据、Δ 定义、判决词表
> **原样沿用** `PREREG_v2_DRAFT.md` 与 DEVIATIONS_v2.md D16 记录的冻结协议，一字不改。

## 0. 为什么需要这份附录

冻结协议 §8 把"第二个未触碰任务复现方向"列为 PAPER-LEVEL GO 的判据来源，§10 要求任何
变更留 diff。配置①实例化了五处判决输入：新数据集、verifier 的 df 注入语义、新 prompt
模板、M_GRID 的 FULL 值、全新 solver。这些都会影响 Δ，**必须在看到任何结果之前冻结**，
否则事后无法自证"判据不是看了结果再定的"。

## 1. 任务与数据

- 数据集：`data/CodeTAT-QA__{train,test}.jsonl`（train 2856 / test 288）
- 字段与 CodeFinQA 同构：question/answer/task/context/context_type/options/program
- `context_type` 恒为 `json`：context 是一个 **JSON 字符串**，内容为 dict-of-dicts 的表格
- gold program 通过 `df["列标签"]["行标签"]` 取数后做算术

## 2. verifier 的 df 注入（已实现并实测）

执行前在程序头部注入一行 `df = <context 原文>`（裸文本拼接，不做 json.dumps —— context
本身就是合法 Python dict 字面量；dict-of-dicts 上 `df[a][b]` 与 pandas 同语法，无需 pandas）。

**守卫边界（重要）**：`_static_check` 与 `_has_computation`（G1 反常量）**只检查模型生成的
code，不含注入的数据**。否则大 dict 字面量里的数字会让"常数程序"蒙混过 G1。

**实测（2026-07-24，冻结前）**：
- CodeFinQA 回归：verifier 自检 7/7 PASS，`context=None` 时行为与注入前逐位相同
- CodeTAT-QA 全量 gold 自洽：**2846/2856 = 99.65%**
- 10 条失败经逐条查明为**数据集自身缺陷**，非适配问题：
  - 7 条 gold 末行不赋值给 `answer`（如 idx 492 末行是 `total_stock_based_compensation = ...`）
  - 3 条用 `sum([...])`/列表推导，无裸 BinOp，被 G1 反常量守卫拒（与 CodeFinQA 的 `+=` 长尾同源）
  - 失败 idx = [492, 688, 1137, 1206, 1274, 1653, 1657, 1839, 2583, 2821]
- 对照：CodeFinQA 的 gold 自洽率是 4668→4035 = **86.4%**，CodeTAT-QA 干净得多
- `cmd_split` 本就用 verifier 过滤不自洽 gold，这 10 条会被自动剔除，不进任何子集

## 3. M_GRID 与 FULL 值

沿用 `[0, 16, 64, 256, 1024, FULL]` 的档位结构。FULL = clean 数 − dev 400，**由 split 实测
回填**（回填后必须重跑 split，否则 subsets 无该键）。回填值：**FULL=2446**（split 实测 n_clean=2846，dev=400，pool=2446）

## 4. solver 设计原则（**本附录最关键的一节**）

### 4.1 为什么要预先声明

solver 是 Δ 的 max 集合成员，它的强度直接决定 ①的门槛，进而决定 Δ 的符号。写强则 GRPO
难赢，写弱则容易复现"RL 输给 solver"。**这是一个不该由实现者在看到结果后掌握的自由度**，
故在此冻结设计原则。

### 4.2 冻结的原则：good-faith 最强努力

**承诺：solver 按"能力最大化"实现，不因任何中间结果调整其强度。**

1. **不从零写**：全量移植已有的 4037 行 solver 资产（`solver_v2_table_align.py` 的表格
   列/行标签对齐、`solver_v2_rulelib.py` 的运算模板库、`solver_v2_combo.py` 的组合策略）。
   CodeTAT-QA 的 `df["列"]["行"]` 结构比 CodeFinQA 的文本表格更规整，移植方向是**增强**不是削弱。
2. **标签对齐用最强匹配**：精确匹配 → 归一化匹配（大小写/空白/标点）→ 模糊匹配 →
   年份与数字标签识别 → 同义词。不满足于精确匹配。
3. **运算模板全量保留**：差值、增长率、占比、求和、平均、比值等，从 rulelib 全量移植，
   不做裁剪。
4. **scarce 变体在约束内做到最好**：检索库 = 该档位的 m 条 gold（依据公平性约束"同一 m 下
   所有分支看到完全相同的 gold"），在这个限制内尽力（从 m 条里学模板/标签分布）。
   m=0 时退化为纯规则。
5. **禁止的做法**：为了让某个结论成立而削弱 solver、裁剪模板、放宽/收紧匹配阈值。

### 4.3 两个变体（沿用 solver_arm.py 结构）

- **scarce**（进 Δ 的 max 集合，协议合规臂）：检索库 = 该档位 m 条 gold
- **full**（诊断上界，**不进** Δ）：检索库 = 整个 demo pool

泄漏防御沿用 `solver_arm.py:90-97`：运行前断言检索库 ∩ dev = ∅，有交集立即中止。

### 4.4 预先声明的结果解释规则

- 若 solver 强而 GRPO 在稀缺档输 → 支持 CodeFinQA 上的发现（外部效度成立）
- 若 solver 强而 **GRPO 反而赢** → **如实报告**。这不是失败，而是"结论依赖任务的符号化
  程度"这一更细致的发现，论文叙事相应调整，**不得通过削弱 solver 来挽回原叙事**
- 两种情形都在本附录冻结时刻**事先接受**

## 5. prompt 模板

模板本体在 `gate2_passk.py:24`（PROMPT_TMPL）。few-shot 两例换成 CodeTAT-QA 表格示例，
并说明 `df["列"]["行"]` 用法。**长度已实测安全**：现行模板 + Qwen tokenizer 下 prompt token
mean=669 / p95=1177 / max=2041，0% 超 MAX_PROMPT=3072（context 字符长 p95=1790 / max=3667，
0% 触及 make_prompt 的 5000 字符截断点）。

## 6. 不变的部分（原样沿用冻结协议）

- Δ(m) = Acc(GRPO|W_m) − max{Acc(W_m), Acc(continued SFT), Acc(RS-SFT), Acc(solver)}
- Gate 0 八项、Gate 1 四条判据与档位选择规则（5-20% / 25-45%）
- Gate 2 Go 四重合取、Task No-Go 三条件、INCONCLUSIVE 带（2-5pp）
- 判决词表：CANNOT-EVALUATE / BENCHMARK-NO-GO / TASK-NO-GO / INCONCLUSIVE / PROVISIONAL-GO
- 3 训练种子；两层 bootstrap；配对 McNemar；分层报告
- BENCHMARK-NO-GO 三分支出口（含 D9 裁决的"区间空缺→补档位"分支）

### 4.5 solver 实现修正（2026-07-24，实测后）

首版 build_program 让 copy_match（检索复用）优先于规则路径。实测 solver scarce 剂量曲线
**随检索库增大而下降**（m=0 纯规则 72.0% → m=2446 全库 60.5%），因为大检索库更容易匹配到
"表面相似、实际不同"的 gold program，覆盖了规则本可算对的答案。纯规则(72%)强于任何
带检索版本(≤60.5%)。

**修正**：规则路径优先，copy_match 仅在规则解不出时兜底 → 保证 solver ≥ 纯规则，检索只增不减。

**方向性声明**：此修正**提高**了 solver（最强非 RL 基线）的门槛，使 GRPO 更难在配置① 上判 Go
——即对"RL 在演示稀缺时输给 solver"这一预期结论**更不利**。故它是 §4.2「good-faith 最强努力」
承诺的应有实现，而非为迎合结论的调参（若为迎合结论，反而应保留弱 solver）。

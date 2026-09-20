# PREREG v2（草案）：demonstration-scarce RLVR —— 演示数量剂量曲线协议

- 日期：2026-07-20
- 状态：**实验设计 / 尚未执行 / 尚未冻结**（含 §7 未决问题，解决前不得开跑）
- 上游：`GATE4_RESULT.md`（CA-RLVR 终结性 No-Go）+ `GATE4_RAW_STATS.md`
- 与 v1 的关系：**这是新选题，不是 CA-RLVR 的补丁**。v1 论题"verifier 可靠 ∧ solver 构造不出 = RL 机会"已被证伪；v2 检验补上的第三条件"演示数据稀缺"是否才是 RLVR 的真实生态位。

## 0. 核心检验量（冻结）

```
Δ(m) = Acc(GRPO | W_m) − max{ Acc(W_m),
                              Acc(continued SFT),
                              Acc(RS-SFT),
                              Acc(solver) }
```

其中 m 为 gold 演示数量，W_m 为仅用这 m 条演示训练出的**共同 warm start**。

真正的 RL 机会必须表现为**交叉结构**：演示少时 Δ(m) > 0，演示多时优势消失。

设计要点：所有方法从**完全相同的信息起点**分叉——这直接封堵了 v1 "GRPO from base vs 见过全量 gold 的 SFT"的信息量不对等问题。

## 1. 实验矩阵

演示数量嵌套、分层抽样：

```
m ∈ {0, 16, 64, 256, 1024, 4668}
```

- 每个大集合必须包含对应小集合（嵌套）
- 按程序长度、运算类型、难度、来源分层并报告
- 3 套独立演示子集种子

每个 m 先训练共同起点 W_m = SFT(m)，然后从**同一 checkpoint** 分叉：

| 分支 | 用途 |
|---|---|
| W_m 不继续训练 | warm-start 基线 |
| Continued SFT | 继续反复训练相同 m 条演示，不获新标注 |
| Iterative RS-SFT | 当前模型采样 → verifier 筛选正确轨迹 → SFT → 重复 |
| GRPO | 当前模型采样 → 同一 verifier 给奖励 → 策略更新 |
| 确定性 solver | 最强非训练基线 |

m=0 时无 continued SFT，其他分支照常。

**RS-SFT 必须保留**：若 GRPO 只胜普通 SFT 却胜不过"采样—验证—再训练"，结果只说明 verifier 生成的数据有价值，不能说明 policy gradient 有独立价值。

## 2. 公平性约束（运行前冻结）

- 同一 base model、tokenizer、prompt、训练/测试划分、硬件
- 同一 m 下所有分支看到完全相同的 gold 演示
- 除指定 m 条外，gold 程序对 GRPO 和 RS-SFT 不可见
- GRPO 与 RS-SFT 使用相同的采样题目、rollout 数、verifier 调用预算
- 主预算：等 wall-clock，每个继续训练分支约 5,700 秒（沿用上轮 matched SFT 尺度）—— **但见 §7.1，此条待修正**
- 同时记录：生成 token、优化 token、verifier 调用数、真实 GPU 时间
- 超参数、checkpoint、演示档位**只用 dev set 选**；test 冻结后只跑一次
- 主指标 greedy execution accuracy；prompt、解码、最大输出长度完全一致

**禁止**将"GRPO 从 base 开始"与"已看过 m 条演示的 SFT"直接比较——那会重新混入信息量差异。

## 3. Gate 0：实现与 verifier 审计

任何一项失败，结论只能是 `CANNOT-EVALUATE`，**绝不能判想法 No-Go**。

- gold 程序接受率 100%
- 语义攻击/常数程序 false accept ≤ 1%
- 固定种子后数据、评测结果、样本顺序可复现
- 训练 reward 与离线 verifier 逐样本一致
- 检查答案泄漏、测试集进入训练、数值容差、格式解析
- 分开统计：语法错误、执行错误、超时、空输出、verifier 拒绝
- 人工构造 canary batch，验证正负 reward、优势方向、参数确实更新
- GRPO 至少出现合理 reward 上升；报告全零奖励组、全一奖励组比例

## 4. Gate 1：演示剂量是否操纵成功

只训练各 W_m，暂不跑昂贵的 GRPO。

- 各档位形成明显梯度（不要求严格单调）
- 低档位至少一个落在 dev accuracy 5%–20%
- 中档位至少一个落在 25%–45%
- 完整演示明显优于低演示档位
- 3 个种子的异常逆序须检查抽样难度，不得直接解释为算法现象

若 m=16 或 64 已接近全量 SFT：

> `BENCHMARK-NO-GO：CodeFinQA 无法提供有效的演示稀缺区间。`

这**不是**对总假设的 No-Go。

## 5. Gate 2：决定性冒烟

由 dev set 预先选定两档：

- m_L：使 W_m dev accuracy 首次落入 5%–20% 的最小 m
- m_H：使 W_m dev accuracy 首次落入 25%–45% 的最小 m

在 m_L, m_H 上运行 Continued SFT / Iterative RS-SFT / GRPO，各 3 个训练种子，预算相同。

### Gate 2 Go
至少一个低演示档位同时满足：
- GRPO 相对最强非 RL 方法平均领先 ≥ 3pp
- 3 个种子中至少 2 个为正
- 无 verifier hacking、格式捷径；增益不由少数题型独占
- **GRPO 优于 RS-SFT**，而不只是优于普通 SFT

### Gate 2 Task No-Go
两个低演示档位均满足任意一项：
- Δ(m) 的 90% CI 上界仍 < +3pp
- RS-SFT 与 GRPO 差距在 ±2pp 内但成本更低
- GRPO 稳定弱于最强非 RL 方法

结论只能写作：

> `CodeFinQA 上 demonstration-scarce RLVR No-Go`

**不能**写成"demonstration scarcity 假设被否定"。

### Inconclusive
平均效应 2–5pp、种子方向冲突、或 CI 跨过门槛 → 追加第 4、5 个训练种子；追加前保持 `INCONCLUSIVE`，不得抢判。

规模：2 档位 × 3 分支 × 3 种子 = 18 次继续训练，按每次约 1.6 GPU 小时估 ≈ 29 GPU 小时，另加较便宜的 warm-start 训练。

## 6. Gate 3：完整假设确认

仅 Gate 2 Go 才运行全部 6 档位，关键档位 5 个训练种子。

最终必须看到交叉结构：

```
gold 演示少：GRPO > 最强非 RL 前沿
gold 演示多：SFT/RS-SFT ≥ GRPO
```

Paper-level Go 需同时满足：

1. 两个相邻低演示档位上，平均 Δ(m) ≥ +5pp
2. 两档的 95% 分层 bootstrap CI 下界均 > 0
3. 至少 4/5 训练种子效应同号
4. GRPO 胜过 iterative RS-SFT
5. `方法 × log2(m+1)` 交互项为负，95% CI 低于 0
6. **m=1024 或全量时 GRPO 优势消失或转负**
7. 在第二个未使用过的可验证程序构造任务上复现方向；≥ +3pp，或跨任务汇总交互项成立

第 6 条关键：若 GRPO 在所有档位都领先，说明发现的是"GRPO 整体更强"，不支持"优势由演示稀缺导致"。

## 7. 统计方法

不再只报单次训练的题目级 CI。

- 每个训练种子的 paired accuracy difference
- 两层 bootstrap：先重采样训练种子，再重采样测试题目
- 每档位 Δ(m) 及 95% CI
- 方法、演示数量及其交互项
- McNemar discordants —— 观察 GRPO 是否解决了非 RL 方法不会的题，而非只看总体准确率
  （上轮此项最具杀伤力：GRPO 独对仅 37/664）
- 按程序长度、操作类型、难度、报告来源分层
- 每个正确百分点所需 GPU 小时

## 8. 最终判决规则

| 状况 | 判决 |
|---|---|
| verifier、数据或实现异常 | `CANNOT-EVALUATE` |
| CodeFinQA 未形成稀缺区间 | `BENCHMARK-NO-GO` |
| CodeFinQA 低演示档位无 RL 优势 | `TASK-NO-GO` |
| 效应接近门槛或种子冲突 | `INCONCLUSIVE`，追加种子 |
| CodeFinQA 出现稳定交叉效应 | `PROVISIONAL-GO` |
| 第二个未触碰任务也复现 | `PAPER-LEVEL GO` |
| 两个有效任务低演示区间效应上界均 < +3pp | `HYPOTHESIS NO-GO` |

CodeFinQA 的 664 题已用于上一轮决策，**只能作为探索性/冒烟证据**。最终确认必须增加未触碰测试集或第二任务，否则不能诚实声称 paper-level Go。

---

# 9. 冻结前必须解决的未决问题

以下 5 条在解决前本协议**不得冻结、不得开跑**。前两条会直接决定结论真伪。

## 9.1 【致命】等 wall-clock 预算在低 m 档位会人为制造 GRPO 优势

m=16 时让 continued SFT 跑满 5,700 秒，等于在 16 条样本上训数百个 epoch——必然灾难性过拟合，dev accuracy 会先升后崩。此时 Δ(m_L) > 0 **是预算协议的伪影，不是 RL 的胜利**。

这会让整个 Gate 2 Go 变成自我实现的预言：低 m 档位的"最强非 RL 方法"被预算规则打残了。

**修正方案**：等 wall-clock 仍作为**算力上限**，但每个分支必须允许 **dev-based early stopping / best-checkpoint 选择**，且明确记录每分支实际用时。报告 Δ(m) 时用各分支的 best-dev checkpoint，而非终点 checkpoint。上一轮 matched SFT 恰好没触发这个问题（4,668 条 × 20 epochs 未过拟合），所以是新引入的风险。

## 9.2 【致命】缺 test-time verifier selection 基线，会高估 RL 的独立贡献

当前"最强非 RL 方法"里没有**不训练、只在推理时用 verifier 的 best-of-n**（W_m 采样 n 条 → verifier 过滤 → 选第一条通过的）。

这是最便宜也最强的对照：上一轮 base 的 pass@64 就有 14.5%，而 GRPO 训 300 步 greedy 才 19.6%。warm start 之后 best-of-n 很可能直接超过 GRPO。若如此，结论就从"RL 有价值"变成"**verifier 有价值，而推理时用它比训练时用它更划算**"——这是完全不同的论文，而且更可能是真的。

**修正方案**：把 `Acc(best-of-n(W_m) + verifier)` 加进 Δ(m) 的 max 集合，n 取与 GRPO 训练等 verifier 调用预算下的可比值，并单独报告 n∈{1,4,16,64}。这条几乎不花额外算力，但可能是整个协议里性价比最高的杀点，**建议提到 Gate 1 就跑**。

## 9.3 【重要】Gate 3 的交互项检验统计功效不足

第 5 条要求 `方法 × log2(m+1)` 交互项 95% CI 低于 0。但：
- 6 档位 × 5 种子，交互项的有效样本量是**种子数**级别，不是题目数级别
- 真实结构是**交叉**（低 m 正、高 m 负），线性 log 交互项对交叉形状的捕捉力弱，可能给出不显著结果而误杀真效应

**修正方案**：主检验改为**预注册的对比对**——直接检验 `Δ(m_L) − Δ(m_full) > 0` 的两层 bootstrap CI（这直接对应交叉结构，功效高得多）；线性交互项降为辅助描述性指标，不作 Go 判据。

## 9.4 【重要】模型规模与家族未指定，Qwen-only 结果按上轮标准即不可信

`GATE4_RESULT.md` 已记录"Qwen-only 结果不可信，须 Llama3.1-8B/OLMo2-7B 复现方向"。本协议未指定 base model。

若沿用 Qwen2.5-0.5B：优点是与上轮可比、便宜；缺点是 0.5B 在低 m 档位的 warm start 极弱，剂量曲线可能整体压在地板上（Gate 1 的 25%–45% 中档位可能根本达不到）。

**修正方案**：Gate 0/1 用 0.5B 探路确认剂量区间存在，Gate 2 起至少加一个 1.5B 或 3B。第二模型家族推迟到 Gate 3，但必须在协议里写明它是 Paper-level Go 的必要条件。

## 9.5 【重要】总算力与四个月周期不匹配

粗算：
- Gate 1：6 档位 × 3 演示种子 = 18 次 warm start（便宜，但非零）
- Gate 2：≈ 29 GPU 小时（若按 9.4 加大模型，×3–4）
- Gate 3：6 档位 × 3 分支 × 5 种子 = 90 次继续训练 ≈ 145 GPU 小时，**再加第二任务全套**

合计乐观估计 250–400 GPU 小时，且这还没算 Gate 0 的实现与审计工时（上轮 verifier + solver 就花了大量人日）。这与"4 个月 + 同时推进选题 7 和选题 1"不相容。

**修正方案**：明确本协议的**冒烟边界只到 Gate 2**（≈ 40–60 GPU 小时，2–3 周）。Gate 3 不是本轮承诺，而是 Gate 2 Go 之后的**重新立项决策点**——届时须重新评估是否值得把 CA-RLVR 那次"升为主项"的资源冲突再走一遍。写进协议可防止 Gate 2 出现弱阳性时被沉没成本拖进 Gate 3。

---

## 10. Material Passport

- **当前阶段**：实验设计 / 尚未执行 / 尚未冻结
- **输入**：上一轮 Gate 4 结果（`GATE4_RESULT.md`、`GATE4_RAW_STATS.md`）、CodeFinQA 演示集、现有 verifier
- **本轮产物**：本文件（含 §9 未决问题的冻结版 Go/No-Go 协议**草案**）
- **执行范围**：Gate 0–2 为冒烟必做；Gate 3 仅在前面通过后**重新立项**启动（见 §9.5）
- **冻结条件**：§9 五条全部裁定后，另存为 `PREREG_v2_FROZEN.md` 并记录 diff

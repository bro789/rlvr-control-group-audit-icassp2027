# CA-RLVR 效应量冒烟测试 —— 预注册文档（冻结版）

- 冻结时间：2026-07-19（本文档冻结后判据不得修改；任何偏离须在 DEVIATIONS.md 中记录原因，且偏离后的结果只能作探索性证据）
- 检验对象：**生死题 #2 —— matched 条件下 GRPO-over-SFT 增益是否 ≥5pp 存在**（已发表金融先验：Fin-R1 +3.0/+4.0、Fin-o1 +1.37、DianJin-R1 −2.56~+4.84，全部 <5pp）
- 执行环境：RTX 5090 32GB / torch 2.10+cu130 / Windows；主模型 Qwen2.5-0.5B-Instruct，复核档 Qwen2.5-1.5B-Instruct

## 任务与数据

- 主候选任务：**CodeFinQA**（kensho/bizbench，train 4,669 / test 844，金融文本+表格 → 可执行 Python 程序）
- 备选：CodeTAT-QA（同源）、FinanceMath（无 train split，仅在主候选被杀时启用且只做 eval 侧分析）
- 评测集：CodeFinQA test。若 FinanceReasoning 修正版（2,238 题）中含本任务修正子集则优先用修正版；否则用原 test 并在结论中报告 14.21% 标注噪声敞口（FinanceReasoning arXiv:2506.05828 实测值）

## 前置门（顺序执行，任何一门失败即效应量测试无意义，全程终止）

### Gate 1 —— CC@budget 审计（纯 CPU）
- **Solver 白名单**（预注册，测 CC 只允许这些家族）：
  1. 正则/规则抽取 + 直接算术；
  2. TF-IDF / 数值匹配检索操作数 + 模板程序库（≤12 类：growth rate、ratio、diff、sum、avg、percent-of、margin、CAGR、变化率、加权和、除法链、恒等式回填）；
  3. LP/符号求解器；
  4. 确定性单位/缩放规范化器；
  5. 时间/元数据过滤器。
- **显式排除**：任何学习模型（含 frozen LLM + prompt、embedding 检索）。
- **预算**：本轮自动化审计 timebox = 1 个会话日；所有 solver 尝试记录在 solver_log.md（构造过程即论文 closure 测量协议的数据）。
- **Kill**：白名单 solver 在 test 集成功率（1% 容差）**CC > 0.40** → CodeFinQA 判死，换 CodeTAT-QA 重测；两者皆死 → 效应量冒烟终止，报告"低 closure 前提不成立"。

### Gate 2 —— 探索可行性（GPU，约 3-5 GPU-h）
- Qwen2.5-0.5B-Instruct 与 1.5B-Instruct，test 随机子集 n≥200，temperature 1.0、top-p 0.95 采样 64 次 + greedy 1 次。
- **Kill**：两档模型 **pass@64 均 <5%**（稀疏奖励无法启动）或 **pass@1 均 >60%**（无 5pp 增益空间）→ 终止。
- 顺带记录（只记录不分析）：正确 rollout 的解模板多样性、pass@k 方差。

### Gate 3 —— verifier 对抗自检（纯 CPU）
- 攻击电池：纯常量程序、percent-vs-decimal、×1000/×1e6 缩放撞容差窗口、gold 数值回抄。
- **Kill**：加守卫后误接受率 >10% → 修 verifier，修不到 10% 以下则终止。

## Gate 4 —— 四臂效应量测试（主检验）

| 臂 | 设定 |
|---|---|
| A. GRPO | 0.5B LoRA(r=16, α=32)，G=8 rollouts/prompt，~300 优化步，lr 1e-5，KL β=0.04，奖励=verifier 通过(1)/失败(0)，**不接触 gold 程序** |
| B. Matched SFT | 同 LoRA 配置，训练数据=train split gold 程序；**matched-compute 为主定义**（GPU 小时与 A 臂相等），matched-sample 为敏感性分析 |
| C. 后处理对照 | base 模型 + Gate 1 的白名单 solver 做约束解码/答案回填（增益必须不可被它消除） |
| D. 奖励对照 | 与 A 完全同配置，奖励替换为 (i) 随机 0/1，(ii) 翻转（错误得 1） |

- 全臂共用：seed=42（单 seed 试点）、greedy 解码评测、同一 test 集。
- **主指标**：Δ = acc(A) − acc(B)，单位 pp。

### 预注册判据（冻结）
1. **Δ < 3pp**（单 seed SE≈3-5pp，故另要求 A 臂训练曲线单调上升方向一致）→ **效应量 No-Go**；
2. Δ ≥ 3pp 但 acc(A) − acc(C) < 2pp（增益被确定性后处理消除）→ **No-Go**（LedgerEquiv/PBPO 同款死法）；
3. D 臂任一奖励变体的增益 ≥ A 臂增益的 50% → **Qwen 家族结果不可信**，效应量问题在本机上无法回答（需 Llama3.1-8B/OLMo2-7B，超出本冒烟范围，如实报告）；
4. Δ ≥ 5pp 且通过 2、3 → **效应量前提成立**，生死题 #2 过关，进入完整 Stage-0。
5. 3pp ≤ Δ < 5pp 且通过 2、3 → **边缘区**：不判死，但记录"5pp Go 判据需下调或需更强任务"，交由 3-seed 复测裁决。

## 诚实条款
- GPU 当前被用户自己的 train_lora.py（PID 40784，seed 1337）占用：**排队等待，绝不抢占**。
- 单 seed 结果只作冒烟证据，不作论文证据；任何阳性结果须 3 seeds 复测后方可写入论文。
- 本文档由 Claude 起草并执行，所有原始输出（训练日志、solver 尝试、评测 JSON）保留在 ca_rlvr_smoke/ 下备查。

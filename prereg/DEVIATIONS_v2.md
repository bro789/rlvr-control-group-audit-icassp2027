# PREREG v2 偏离记录

原则：任何与 `PREREG_v2_DRAFT.md` 不一致的实际做法，当场记录，不事后追认。
每条注明**方向性影响**——偏差是让假设更容易成立（危险）还是更难成立（保守）。

---

## D1【口径修正】演示总数 4668 → 4035

- **PREREG 写法**：m ∈ {0,16,64,256,1024,4668}，称"CodeFinQA 有 4,668 条 gold 程序"
- **实际**：4668 是 `CodeFinQA__train.jsonl` 的**原始行数**；用 verifier 逐条自检后，gold 程序能自洽复现答案的只有 **4035 条**（86.4%）
- **来源**：`gate_v2.py split` 实测，`data/v2_split.json: n_clean=4035`
- **影响**：上轮 Gate 4 的训练本来就只用 clean 子集，**结论不受影响**；但引用"演示数量"时必须用 4035，`GATE4_RESULT.md` 中"4,668 条 gold 程序"一句需按此口径理解
- **方向**：中性（口径澄清）

## D2【档位调整】最大档位 4668 → 3635

- **原因**：dev 400 条从 train clean 中留出（见 D3），demo pool = 4035 − 400 = **3635**
- **实际网格**：m ∈ {0, 16, 64, 256, 1024, **3635**}
- **方向**：中性；最大档位少 400 条演示，对"高 m 端 SFT 应该很强"的预期影响可忽略

## D3【设计选择】dev 从 train 留出，test 664 全程冻结

- **PREREG 要求**：超参/checkpoint/档位只用 dev 选，test 冻结后只跑一次；且"CodeFinQA 664 题已用于上一轮决策，只能作探索性证据"
- **实际**：从 train clean 中随机留出 **400 条作 dev**，checkpoint 选择用其前 **200 条**（`CKPT_DEV_N`）以省算力；`CodeFinQA__test.jsonl` 在 Gate 0/1 全程**未被读取用于任何选择**（仅在 Gate 0 泄漏审计中用作指纹比对）
- **验证**：Gate 0 检查 3/5/6 确认 dev∩pool=0、dev∩test=0、pool∩test=0
- **方向**：保守（比复用 test 更严格）

## D4【实现偏离】step-capped 而非 epoch-matched

- **实际**：`MAX_STEPS_CAP=800, MAX_EPOCHS=60`，有效 batch 固定 64。各档位实际训练量：

  | m | steps/epoch | 总步数 | 实际 epochs |
  |---|---|---|---|
  | 16 | 1 | 60 | 60 |
  | 64 | 1 | 60 | 60 |
  | 256 | 4 | 240 | 60 |
  | 1024 | 16 | 800（封顶） | 50 |
  | 3635 | 56 | 800（封顶） | **14** |

- **问题**：低档位训 60 epochs，最高档位只训 14 epochs。上轮 Gate 4 全量 SFT 训了 1460 步/20 epochs 才到 50.2%，**m=3635 存在欠训风险**
- **方向**：**保守**。欠训压低曲线右端（W_full 偏弱），使"完整演示明显优于低演示档位"更难成立，也使"高 m 时 RL 优势消失"更难观测——偏差方向不利于假设成立，不会伪造交叉结构
- **处置**：若 m=3635 的 best-dev 明显低于上轮 50.2%，补训至 epoch 可比后重测；结论报告中必须声明本轮为 step-capped

## D5【已知天花板】max_new_tokens=256 截断约 0.1% 最长程序

- **实测程序长度分布**（1500 条 train，Qwen2.5 tokenizer）：中位 **50**，p95 98，p99 180，p999 **340**，max **391**
- **设置**：`MAX_COMP=256`，覆盖 p99，但截断 p999 及以上
- **保持原因**：与上轮 Gate 4 完全一致，跨轮可比性优先
- **方向**：中性（对所有臂一视同仁），但报告绝对准确率时须声明此天花板

## D6【编排偏离】串行驱动 → 并行编排（显存令牌调度）

- **起因**：用户指示"前期发现可并行必须重来"。原 `run_gate1.sh` 为 16 步串行长龙
- **实际**：改为 `orchestrate.py`，每个 m 一条链（warm→evalckpt→bon），链间并行，**按显存配额发令牌**而非固定并发数（`COST={warm:10, evalckpt:4, bon:6}` GB，预算 28.1 GB）
- **触发的连锁修正**：
  - 用户已在 NVIDIA 面板关闭 CUDA Sysmem Fallback → 超显存直接 OOM 而非降速，故必须显存感知调度
  - PyTorch caching allocator 不归还显存，账面 18 GB 时实测占用达 30.8 GB → 增加 `torch.cuda.set_per_process_memory_fraction()` 每进程硬上限
- **方向**：中性（只影响墙钟，不影响任何数值结果）
- **注意**：并行会改变各任务的实际墙钟，故**本轮墙钟数据不可用于"等 wall-clock 预算"的配平**。Gate 2 的等预算配平必须在独占 GPU 的条件下重新测定

## D7【故障与重跑】HuggingFace SSL 故障导致 4 条链失败

- **现象**：2026-07-20 10:05–10:10，`evalckpt` m=256/64/16/0 连续失败，均为 `huggingface.co` SSL EOF
- **根因**：`from_pretrained` 每次联网校验模型元数据；并行多进程并发请求触发失败（串行时未暴露）
- **修复**：强制 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`（模型已在 `E:\hf_cache`）
- **附带加固**：`evalckpt` 改为逐 ckpt 增量落盘（`bestdev_partial_*.json`），中断可续跑
- **数据影响**：**无**。失败链未产出任何结果，全部重跑；已完成的 warm checkpoint 未受影响
- **方向**：中性（纯工程故障）

## D8【范围限定】Gate 1 探路仅用 seed 0

- **PREREG 要求**：3 套独立演示子集种子
- **实际**：Gate 1 先用 **seed 0** 单种子确认剂量区间是否存在，再决定是否补 seed 1/2
- **理由**：Gate 1 的判据是"区间归属"而非效应量显著性，单种子足以判断网格是否可用；省下的算力留给 Gate 2
- **方向**：中性；但 Gate 1 的"3 个种子异常逆序须检查抽样难度"一条本轮无法执行，若补种子后出现逆序需回头处理

## D9【待定】5%–20% 区间可能为空，或需补档位 m=128

- **观察**（滚动值，非最终 best-dev）：m=64 ≈ 1.0%、m=256 ≈ 29.2%，**5%–20% 区间内无档位**
- **含义**：若 best-dev 最终确认此断层，则按 PREREG §5 无法选出 m_L
- **处置**：补 m=128（64 与 256 的几何中点），必要时再加 m=192。嵌套性天然满足（子集按前缀切分）
- **注意**：这**不构成** `BENCHMARK-NO-GO`——该判决针对的是"m=16/64 已接近全量演示"（稀缺区间不存在），本例是相反情况（低档位过弱，网格不够密）
- **状态**：等 best-dev 最终值确认后决定

## D10【已裁决】Gate 1 结论汇总（覆盖 D9 的待定项）

- Gate 1 判决 **PASS**，四条判据全过，详见 `GATE1_v2_RESULT.md`
- **D9 取消**：m=16 最终 best-dev = 11.50%，落在 5–20% 区间，**不需要补 m=128**。此前观察到的"断层"是评测未完成时的滚动值假象
- **D4 欠训担心解除**：m=3635 的 best 落在 checkpoint-532（非终点），其后回落至 42.5%，说明 800 步内已过峰，不是训练不足
- Gate 2 选点：**m_L=16（11.50%）、m_H=256（32.50%）**

## D11【方法学纠正】§9.1 的过拟合担心未被证实，且作者曾据不完整数据误判

- **§9.1 原论证**："等 wall-clock 会让低档位 continued SFT 灾难性过拟合，人为制造 GRPO 优势"
- **实测不支持**：5 个档位中 4 个的 best-dev 就是终点 checkpoint，唯一落差在 m=3635（+2.0pp，200 题上接近噪声）。m=16 用 60 epochs 训 16 条样本，dev acc 单调上升至终点，未见崩塌
- **推测原因**：LoRA + lr 1e-5 + cosine + completion-only loss 的组合足够温和
- **处置**：best-dev checkpoint 选择继续保留（成本低、无害），但 §9.1 中"等 wall-clock 制造 GRPO 优势"的论证**失去实证基础**
- **附带记录的推断错误**：作者（Claude）在评测仅完成 1–3 个 checkpoint 时，观察到"m=64 的 best 位于 checkpoint-10"，据此宣称"第 10 步即到顶、其后全在过拟合、坐实 §9.1"，并重复三次。该推断错误——best 只能落在已评的少数 checkpoint 中，属评测进度假象。`gate1_verdict.py` 内置了"未完成档位不参与判决"的保护，但口头解读绕过了它
- **方向**：中性（纠正后不影响 Gate 1 判决）

## D12【已裁决·不可换】vLLM 等价性 gate 失败 —— Gate 2 继续用 HF generate

### 结论

在同一 checkpoint（`merged_m1024_s0`，即 W_1024 的 best-dev）、同一批 dev 200 题、prompt 指纹一致（`b96d876c7e37e562`）的条件下对拍：

| 指标 | 值 |
|---|---|
| HF generate acc | **41.00%** |
| vLLM acc | **38.00%** |
| **准确率差** | **−3.00pp** |
| 逐题 reward 一致率 | **91.00%**（18/200 题翻转） |
| **生成文本完全相同率** | **38.50%** |
| 只有 HF 对 / 只有 vLLM 对 | 12 / 6 |
| McNemar χ² | 1.39（**不显著**，p>0.05） |

**判决：❌ 不等价，不可换引擎。** 证据存于 `vllm_equiv_result.json`（含 18 条逐题翻转样本）。

### 差异性质：路径分叉，非浮点末位噪声

翻转样本显示两引擎在同一题上走了完全不同的推理路径：

```
题 0  gold=36.10
  HF  : restricted_stock_units_outstanding = 3188   （正确分母）
  vLLM: restrictions_total = 3972                   （取了另一个数）

题 4  gold=6308.30
  HF  : (6197 + 6305) / 2      （求平均）
  vLLM: (6197 - 6305) / 3      （求差再除以 3）
```

机制：0.5B 模型 + ~3000 token 长 prompt + bf16，logits 间距极小；HF 的 left-padding 批量解码与 vLLM 的 continuous batching + FLASH_ATTN 后端在数值上的微小差异，足以让某个位置的 argmax 翻转，其后整条程序分叉。**文本完全相同率仅 38.5%**，说明这种分叉是普遍现象而非个例。

### 为什么这 3pp 是致命的

**Gate 2 的 Go 判据是 Δ(m) ≥ 3pp，而引擎差异恰好也是 3.0pp——两者同量级。**

若拿 vLLM 跑 Gate 2 去比 Gate 1 的 HF 基线，引擎噪声与待测效应量无法区分，且这种污染在结果中不可见，只会表现为"GRPO 好像差了/好了 3 个点"。

McNemar χ²=1.39 不显著意味着差异**没有系统性方向**（12 vs 6 随机翻转），这比有系统偏移**更糟**：它不是可校正的常数偏置，而是不可预测的噪声。

### 残留可用性

vLLM 性能优势极大（200 题：HF ≈ 5 分钟 vs vLLM **4.5 秒**生成 / 26 秒含加载，约 60×）。仅在**全部环节统一使用同一引擎**时可用——但这意味着 Gate 1 的 6 个档位需用 vLLM 全部重跑才能自洽，成本高于收益，故放弃。

### 配置过程踩的坑（供复现/后续参考）

环境：Windows 11 + WSL2（Ubuntu，内核 6.6.87.2）+ RTX 5090（sm_120 Blackwell）+ 无系统级 CUDA toolkit。

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | **WSL 默认 Python 3.14 过新** | vLLM 不支持 | 用 `uv venv --python 3.12` 建独立环境 |
| 2 | **pypi 直连慢** | 安装耗时 | `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` |
| 3 | **`RuntimeError: UVA is not available`** | EngineCore 启动即失败 | 见下方详述 |
| 4 | **`VLLM_USE_V1=0` 无效** | 警告 "Unknown vLLM environment variable" | vLLM 0.25.1 已移除 V0 引擎，无法回退 |
| 5 | **`Could not find nvcc`** | EngineCore 启动失败 | `uv pip install nvidia-cuda-nvcc-cu13`，设 `CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13` |
| 6 | **`enforce_eager=True` 不能绕过 nvcc** | 仍报缺 nvcc | nvcc 需求来自 attention/sampling 后端的 JIT，非 inductor 编译 |
| 7 | **FlashInfer JIT 编译失败** | `CUDA compiler and CUDA toolkit headers are incompatible` | nvcc 13.2 与 flashinfer 0.6.13 自带 cccl 头文件版本冲突 → `VLLM_USE_FLASHINFER_SAMPLER=0` 绕开 |
| 8 | **Windows 侧 JSON 写入崩溃** | `UnicodeEncodeError: 'gbk' codec` | `open(..., "w", encoding="utf-8")`（判决串含 emoji） |

**坑 3 详述（最有价值的一条）**：vLLM 报 `UVA is not available` 并非硬件/驱动限制。追查发现：

```python
# vllm/utils/platform_utils.py
def is_uva_available() -> bool:
    return is_pin_memory_available() or current_platform.is_cpu()

# vllm/platforms/cuda.py
def is_pin_memory_available(cls) -> bool:
    if in_wsl():
        version = _get_wsl_kernel_version()
        if version is None or version < (4, 19, 121):
            return False
        # On compatible WSL2 kernels, pinned memory is supported but
        # disabled by default. Enable it via VLLM_WSL2_ENABLE_PIN_MEMORY=1.
        return envs.VLLM_WSL2_ENABLE_PIN_MEMORY
    return True
```

即 WSL2 内核 ≥4.19.121 时 pinned memory 实际可用，只是**出于保守默认关闭**。本机内核 6.6.87.2 远超门槛。开关前先做了实测验证（非盲目绕过安全检查）：

```
pin_memory OK / is_pinned True / H2D 传输 OK
get_accelerator_view_from_cpu_tensor 成功
CPU 写入 123 → GPU 侧零拷贝读到 123  ✓
```

确认功能真实可用后启用 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`。

**最终可运行配置**（记录备查，虽然本轮不采用）：

```bash
# WSL2
cd ~/vllm_env && source .venv/bin/activate
export CUDA_HOME=$HOME/vllm_env/.venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 python <script>
# 实际选中的后端：FLASH_ATTN（候选 FLASH_ATTN/FLASHINFER/TRITON_ATTN/FLEX_ATTENTION）
# 版本：vllm 0.25.1 / torch 2.11.0+cu130 / RTX 5090 capability (12,0)
```

### 方法学价值

本 gate 的设计要点是**生成与验证分离**：`gen` 阶段两引擎各自只产生文本，`cmp` 阶段回到同一个 verifier 验证。否则 WSL/Windows 的 Python 与依赖差异会混入结果，测到的就不是引擎差异。同时校验 prompt SHA 指纹，确保两侧输入完全一致。

**若跳过此 gate 直接换引擎，Gate 2 将产出一批被 3pp 不可见噪声污染、且事后无从追溯的结果。**

---

## D13【工具修正】`gate2_eval.py` 逐题结果增量落盘（2026-07-20 20:1x）

**改动**：`eval_branch()` 原本每次调用都用空的 `per_item = {}`，而续跑时已评过的 checkpoint 走 `continue` 跳过、不重算逐题结果。若评测中断后续跑、且最终 `best` 点恰好是中断前评的，`per_item.get(best)` 返回 `None`，写出的 `bestdev_{branch}` 里 `per_item_best` 就是 `null`。

新增 `runs_g2/peritem_{branch}_m{m}_s{seed}.json`，与 `bestdev_partial_*` 同步增量落盘，续跑时先读回。

**动机**：Gate 2 v2 要做逐题配对 McNemar（对标 v1 Gate 4 最硬的那条证据：GRPO 独对 37 / SFT 独对 240，χ²=147.3）。`per_item_best` 一旦为 `null`，配对分析就只能重烧那个 checkpoint 才能补，属于可预防的返工。

**影响**：只增不改——准确率、best 点选择、`bestdev_*` 既有字段全部不变，纯粹是把已经算出来的逐题结果多存一份。改动时 `gate2_eval.py` 未在运行。

**配对可行性核查（同时完成，无需改动）**：
- `bondump_m{m}_s{seed}.json` 每题存 `{gold, rewards[n], vals[n]}`（`gate_v2.py:403-410`），`vals` 即 `majority_at_k` 过完 execution guard 的数值列表 → **任意 k 的逐题 pass@k / maj@k 可离线重算，无需 GPU**。
- bon 用 `--dev-n 200`，`gate2_eval.py` 用 `CKPT_DEV_N = 200`，两者都是 `dev_items()` 的同一确定性前缀 → **下标对齐，可直接配对**。

---

## D14【效率缺陷·本轮不修】`cmd_bon` 同一段代码重复执行 3–6 遍（2026-07-20 21:5x 实测）

**现象**：`gate_v2.py` 的 `cmd_bon` 内层循环里，每条采样的代码被 `run_program` 执行多次。以桩函数计数实测（n=43, ns=[1,4,16,43]）：

| 采样位置 | 执行次数 |
|---|---|
| 第 0 条 | **6** |
| 第 1–3 条 | 5 |
| 第 4–15 条 | 4 |
| 第 16–42 条 | 3 |
| **每题合计** | **150**（必要的只有 43，冗余 **71%**）|

三处来源：
1. `verify(c, gold)` 执行一遍 —— **必要**
2. `vals` 循环（`gate_v2.py:405-409`）再执行一遍 —— 冗余，`verify()` 返回值里已有 `pred`
3. `majority_at_k(codes, k)` 对 `ns` 里**每个 k** 各执行一遍前缀 —— 冗余，`vals` 就是它要算的数值；前缀嵌套导致靠前的采样被重复最多

**代价**：`run_program` 每次 `subprocess.run([sys.executable, "-I", path])` 起一个新解释器，实测单次中位 **35 ms**（极简程序，纯开销下界）→ 冗余部分约 **12.5 min/档**。且 `TIMEOUT = 5`：一个死循环程序若落在第 0 位要烧 6×5s=30s。另外 `cmd_bon` 的 `verify` 是**串行**列表推导，而 `cmd_grpo` / `cmd_rssft` 都用了 `ThreadPoolExecutor`（后者 `max_workers=12`）—— 故 bon 只跑满 98% 单核（24 核机器）。

**范围（重要）**：**此缺陷仅存在于 `cmd_bon`。** `cmd_csft` 完全不调 verifier（`verifier_calls: 0`）；`cmd_grpo`、`cmd_rssft` 均为每条候选一次 `verify` 且并行。**三训练分支的等预算配平未被污染。**

**对数据正确性的影响：无。** 冗余执行浪费的是时间不是正确性，且已交叉验证：由 `bondump` 的 `vals` 离线重算 maj@k，与报告值在 k=1/4/16/43 **四点全部精确吻合**（重算时的 gold 取自 `dev_items(200)` 而非 dump 内存的字符串，故此检验同时坐实了 `bondump[i]` ↔ `dev_items(200)[i]` 的下标对齐）。

**本轮不修的理由**：m=16 是决战档位且已跑完，中途改实现会让后续档位与它不可比；`verify` 里 `compare` 收原始字符串 `val` 而 `vals` 存 `parse_number(v)`，两条路径处理有细微差别，改动需配单测。

**下一轮修法**：`vals` 取 `verify()` 的 `pred`；`majority_at_k` 改为接收数值列表而非源码。150 → 43 次，降 71%，且超时程序只执行一次。

**附带结论（可入论文）**：`maj@k` 的计算成本被此实现严重高估。若论文要论证「maj@k 是可部署的 test-time 策略」，成本论证须基于修复后的 43 次执行。

---

## D15【v1 文档瑕疵·不影响结论】`GATE4_RAW_STATS.md` 两条 CI 非 Wilson（2026-07-20 23:0x）

`GATE4_RAW_STATS.md` §1 表头标注「95% Wilson CI」，但复核发现 GRPO@200 与 GRPO@300 两行的 CI 既非普通 Wilson 也非连续性校正 Wilson：

| 臂 | 文档 | 普通 Wilson | CC Wilson |
|---|---|---|---|
| GRPO@200 (85/664) | [10.44, 15.60] | [10.47, 15.56] | [10.40, 15.64] |
| GRPO@300 (130/664) | [16.68, 22.77] | [16.74, 22.77] | [16.67, 22.85] |
| base (10/664) | [0.82, 2.75] | [0.82, 2.75] ✔ | — |
| matched SFT (333/664) | [46.36, 53.94] | [46.36, 53.94] ✔ | — |

四条里两条精确吻合、两条差 0.03–0.06pp。差异**不是**由 `reward` 计数(85/130) 与 `reason=='match'` 计数(84/129) 的口径差引起（按 match 算差得更远）。

**影响：无。** 量级 ≤0.06pp，而 v1 的结论建立在 37pp 的差距上；且 v1 Gate 4 整体已被 v2 取代（不同评测集、不同起点）。

**本轮处置**：v2 的所有 CI 统一由 `gate4_v2.py` 计算，其 `wilson()` 与仓库 `gate2_verdict.py:21` 逐位一致（8 组测试验证）。`gate4_v2.py` 的 McNemar 亦已用 v1 三组已发表数字对账通过（χ²=207.51 / 147.31 / 25.14，全部精确吻合）。

---

## D16【合规返工】按冻结协议全面审计并修正判决层（2026-07-21 01:0x–01:4x）

用户于 2026-07-21 00:3x 贴出冻结协议，要求逐条核对。6 名审计员 + 28 条对抗核验（**0 条被推翻**）共查出 **49 条不合规**：blocker 12、major 22、minor 15。

### 最严重的一条：`Acc(solver)` 被静默漏出 Δ 的 max 集合

协议 §0 冻结的检验量是 `Δ(m) = Acc(GRPO|W_m) − max{Acc(W_m), Acc(continued SFT), Acc(RS-SFT), Acc(solver)}`。`gate2_verdict.py:4-7` 的文档串照抄了这个公式，但 rows 组装（64-85 行）从不加入 solver 臂，缺失断言也不检查它——**正好踩中该文件 60-61 行自己警告的「max 集合悄悄塌缩，门槛降低，打印出假 Go」**。`gate4_v2.py` 全文 0 处 "solver"。

新增 `solver_arm.py`，在 dev n=200 上评测。两个变体：
- **scarce**（协议合规臂，进 Δ）：检索库 = 该档位的 m 条 gold，依据 §二「同一 m 下所有分支看到完全相同的 gold 演示」；
- **full**（诊断上界，不进 Δ）：检索库 = 全 pool 3635 条。

实测（dev 200，泄漏防御断言：检索库 ∩ dev = ∅）：

| m | 0 | 16 | 64 | 256 | 1024 | 3635 |
|---|---|---|---|---|---|---|
| solver(scarce) | 34.50% | **36.00%** | 35.00% | 35.00% | 36.00% | 38.00% |

**solver 几乎与 m 无关**（m=0 纯规则已 34.5%），是五分支中唯一不随演示量缩放的——其实力来自规则模板与表格解析，检索库贡献仅约 3.5pp。

**后果：Δ(m=16) 由 +3.00pp 翻转为 −14.00pp。** 先前那个「刀刃上的 INCONCLUSIVE」是这条遗漏制造出来的假象。

顺带修复 `solver_arm.py` 里一个会毁数据的坑：`S.load_sc_flags(子集)` 按 `len(rows)` 校验缓存，长度不符会重算并**覆盖** `train_sc_flags.json`（写成子集长度）。改为读一次全量再按原始下标切片。

### `gate4_v2.py` 判决层重写

| # | 协议条款 | 原状 | 现状 |
|---|---|---|---|
| 1 | Δ max 集合含 solver | 漏 | 四臂齐全；maj@k/pass@k 降为**附加诊断臂**，明示不进 max |
| 2 | 缺臂不得裁决 | 护栏被拆（实证：缺 csft 仍打印 GO） | 移植 `gate2_verdict.py:45-62`，缺任一必需臂 → `CANNOT-EVALUATE` + `sys.exit(1)` |
| 3 | INCONCLUSIVE 带（2–5pp、种子冲突、CI 跨门槛） | 只有二元 GO/NO-GO → 灰区抢判 | 完整判决树，含种子数不足触发 |
| 4 | 判决词表 | 自造 GO/NO-GO | `CANNOT-EVALUATE`/`BENCHMARK-NO-GO`/`TASK-NO-GO`/`INCONCLUSIVE`/`PROVISIONAL-GO` |
| 5 | Go 四重合取 | 只实现第 1 条 | 四条全部编码并逐条打印 |
| 6 | ≥3pp 比较 | 浮点（`0.19+0.03` 边界不可靠） | **整数计数**：`(k_grpo − k_best) ≥ ceil(0.03·n)` |
| 7 | Δ 的 CI | 用两个**独立** Wilson CI 重叠冒充 | 逐题**配对** bootstrap（B=10000），出 90%/95% CI |
| 8 | 两层 bootstrap | 无（`gate2_verdict.py:133` 自认） | 多种子时启用；单种子明示为**退化版**、CI 偏窄 |
| 9 | Task No-Go 三条件 | 无 | 三条全部编码（90% CI 上界、±2pp+成本、多种子一致为负） |
| 10 | 分层报告 | 无 | 程序长度、运算类型两轴（AST 提取）；**难度/来源两轴数据无字段，不可构造**，随报告声明 |
| 11 | 失败模式分解 | 无 | 从 `reason` 字段分解 greedy 侧 |
| 12 | GPU-h/正确百分点 | 无 | 实现（附三条口径声明） |
| 13 | 多重比较 | 8 个裸 p 值 | Holm 校正 + 明示唯一 confirmatory 对比 |
| 14 | 连续性校正 | `b==c` 时统计量为 `1/n_disc` | 下限截零 |

**Go 合取项 3「无 verifier hacking / 格式捷径」判定为「不可评估」**：审计需要 GRPO 生成的**程序文本**，而 `gate2_eval.py` 的 `per_item_best` 只存 `{reward, reason}`。协议中该条是 Go 的必要条件，**未经检验 ≠ 已满足**，故实现为硬性置否（Go 不可达）。增益集中度部分可算，已随分层输出。

### runner 脚本修正

| 文件 | 条款 | 修正 |
|---|---|---|
| `gate_v2.py` `cmd_bon` | Gate 0「固定种子后可复现」 | 采样前 `set_seed(SEED*10000+m*10+seed)`，种子写入 `bon_*.json` 的 `gen_seed`。**此前 maj@43 是判决输入却不可复现**（±2–3pp 采样噪声） |
| `gate_v2.py` `cmd_bon` | Gate 0「分开统计语法/执行/超时/空输出/拒绝」 | `bondump` 增 `reasons` 字段。原先五类失败全压成 `None`（4863/8600 条），采样侧失败模式事后不可恢复 |
| `gate2.py` `cmd_rssft` | 同上 | 第 1 轮采样在任何 Trainer 实例化之前，torch 全局 RNG 未定种 → 开头 `set_seed(SEED+TS)` |
| `gate2.py` `cmd_grpo` | §二「记录生成/优化 token」+ Gate 0「全零/全一奖励组比例」 | meta 增 `gen_tokens`/`opt_tokens`/`groups`/`groups_all_zero`/`groups_all_one`/`group_zero_var_frac` |
| `gate2.py` `cmd_grpo` | Gate 0「训练 reward 与离线 verifier 逐样本一致」 | 新增 `reward_dump.jsonl` 落盘 `(code, gold, reward)`，供事后逐样本比对 |
| `gate2.py` `cmd_rssft` | §二 token 计量 | 累加 `gen_tokens`；meta 增 `budget_binding`/`wall_frac_used`（显式标注是 vcalls 还是 wall 先到） |
| `gate2.py` | §一「3 套演示子集种子」vs Gate 2「3 个训练种子」是**两条轴** | 拆出 `--train-seed`；`out_dir` 在 `tseed != seed` 时加 `_t{N}` 后缀（既有产物路径不变）；rssft 对齐 GRPO 预算时按同训练种子查找 |
| `gate2.py` `cmd_csft` | §9.1 预测的 SFT 峰值区间整段漏采 | 新增 `--dense-early`（30/60/120/240/480s）。根因：eff batch 64 > m=16 样本数 ⇒ **1 gstep = 1 epoch**，均匀 6 点最早落在 950s |

### 尚未修复（需 GPU 或需重切分，已立项）

- **m_H=256 三分支全缺** —— Task No-Go 要求「两个低演示档位均满足」，缺此档使完整裁决结构上不可能
- **训练种子只有 1 个** —— Go 合取项 2 逻辑上不可评估
- **rssft 只吃到 33.6% 墙钟预算**（1914s/5700s，verifier 轴先耗尽）—— 需 wall-matched 敏感性臂
- **三分支 GPU 争用条件不同** —— grpo 期间 filler 并行占 37% 窗口；csft 全程被 bon 抢卡；rssft 独占。`filler.sh:7` 声称「代价已记入 D13」但 D13 是 gate2_eval 落盘修正，**指针失效**；且 D6 自己写明「Gate 2 等预算配平必须在独占 GPU 下重新测定」，被 csft 直接违反
- **演示子集为纯随机前缀，非分层抽样**（`gate_v2.py:62-72`）—— 完全合规需重切分+重跑全部 warm，会作废现有 Gate 1/2 产物
- **Gate 0 八项只覆盖三项半** —— 第 4（reward 逐样本一致）、6（失败模式分解）、7（canary batch）、8（全零/全一组比例）此前不在审计脚本内，却已判 `verdict=PASS`。代码侧已补齐落盘能力，但**审计本身需重跑才能声称通过**
- **`extract_code` 零审计零单测**
- **`GATE1_v2_RESULT.md` §4/§6 的 sel@k 表已与产物不符**且门槛叙述被 RESUME.md 自行推翻，无勘误标注

### 方向性总评

49 条里，方向性**利于「判 Go」**的偏离占压倒多数：漏 solver（门槛降 21pp）、拆护栏（缺臂仍裁决）、无 INCONCLUSIVE 带（灰区直接判 GO）、rssft 墙钟饥饿（削弱最强学习类对手）、Go 合取项 3 从未检查。这不是随机疏漏，而是**系统性偏向假设成立**。补上 solver 一条即由 +3.00pp 变为 −14.00pp。

### 当前判决

`gate4_v2.py report --m 16 --seed 0` 输出 **INCONCLUSIVE**（训练种子仅 1/3，协议禁止抢判），并附注：Task No-Go 判据 1 已在本档位满足（Δ 90% CI 上界 −7.00pp < +3pp），方向明确为负；缺第二个低演示档位故不能判 TASK-NO-GO。

---

## 待补充

- Gate 2 执行中的偏离（部分已入 D16）
- 若补 seed 1/2：种子间一致性检查结果（D8）

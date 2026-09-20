# Gate 1 v2 solver "combo" — 迭代日志

评测协议：dev = train 的 idx%4==0 且 gold 自洽子集（n=1013），检索库 LOO（排除自身）；
快速内嵌执行复刻 verifier 语义（answer 变量 + 1% 相对容差 + 反常量守卫）。
test 只在最后评一次（真 verifier）。

## 基线（solver_gate1.py, 上一版）
- test CC = 2.9%。

## 预分析（train gold 挖掘，见 scratchpad/analyze1-3.py）
- train 自洽率 86.4%（4035/4668）；操作数 2 个占 76%；操作数 96% 在 context 数字集中，67% 全在表格。
- 签名分布：(N/N)*100 31%、((N-N)/N)*100 19%、N-N 13%、N/N 10%。
- 约定：growth = (new-old)/abs(old)*100，decrease 措辞时 = (old-new)/abs(old)*100；
  diff = new-old（dec 时反向）；portion num<den 98%，分母 63% 是 total 行。
- 同 context 近重复问题（jaccard≥0.9）复制 gold 程序：LOO 命中 67%，覆盖 ~4%。

## Round 1 — 表格解析 + 模板 + 检索复制/kNN 签名投票
dev CC = 0.2428。分型：growth 38.9%、portion 21.2%、ratio 24.8%、diff 21.1%、copy 71.4%。
失败模式：portion 角色切分动词不足；'total' 被当停用词导致 total 行匹配失效；
表头短行未右对齐 → 年份列错位；文本中 "december 31" 的 31 被当候选数；
mojibake（2019s/2013xxx）伪年份；kNN 把 sum 误改写成 growth。

## Round 2 — 大修
- 清洗：日期日数过滤、粘连 mojibake 年份过滤、表头右对齐；'total/net/rate' 恢复为实义 token；
- portion 重写（part 细胞级打分 + whole 候选族：wt行/total行/total列）、margin 模板、
  ratio 双年模式与角色绑定、avg 转置/文本、sum 双年、prod rate×principal、every-dollar；
- kNN 覆盖限制在 other/sum_weak。
dev CC = 0.2942。growth 42.9%、portion 29.8%、margin 53.3%。

## Round 3 — 特殊表结构
- 滚动余额表（Beginning/Ending Balance、January 1/December 31）→ 单年/跨年 change 对；
- base-100 股价表现表 + cumulative/five-year return → old=100 基准列；
- 表头粘连年份（September2014）解析；sum "next N years"（连续 N 年行/列求和）；
- portion whole 候选强化（total 列/行加权、wt=part实体判定、实体在列的 wtcol 模式）、abs() 输出。
dev CC = 0.3169。growth 48.9%、portion 31.2%、diff 25.9%。
剩余大桶：portion 258 miss、growth 94、ratio 80、diff 83、extract 41、avg 37。

## Round 4 — portion 结构化大修（对债务/租约义务型表）
诊断：portion 失败中 158/262 gold 操作数全在表内。gold 惯例：无年份限定时读 Total 列；
"total <X> 的百分比" 且 X=part 实体 → whole=同行 Total 列；"due in <year>" 的年份优先于句首 as-of 年份；
"after <year>" → Thereafter 行（禁年份加分）；"percent of A to B" 按 ratio 切分角色。
dev CC = 0.3445（portion 31.2→38.7%）。

## Round 5 — ratio 大修 + range/extreme 模板
ratio 角色在列的模式（shared row）、无年份时 Total 列默认、"X-to-Y ratio" 连字符模式、
角色 token 不过滤 TYPE_WORDS；新增 mathematical range（max-min）与 highest/lowest（max/min）模板。
dev CC = 0.3524。

## Round 6 — 清洗与文本侧
camelCase 表头拆分（UnitedStates）、MM/DD/YY 列年份（12/07）、respectively 配对放宽
（数字多于年份时取年份前最后 k 个）、average X per Y → 除法、avg 文本序列模式、
"increased to A from B" 趋势句抽取（弱表分时兜底）。
dev CC = 0.3583。

## Round 7 — 同 context 骨架迁移 + kNN 兜底
skeleton transfer：同 context、同题型、jaccard≥0.45 的 train 近邻，把其 gold 程序两操作数
定位到表内 (行,列)，按本题年份/实体重绑定。结果：transfer 桶 59.5% 命中但净效应 -5
（growth/diff 迁移把模板本可解对的题抢走答错）。extract 兜底改为 kNN 签名驱动的双操作数模板。
dev CC = 0.3534（负迭代，回撤 growth/diff 迁移）。

## Round 8 — 迁移仅保留 portion（jac≥0.55）
transfer_portion 76 题 67.1% 命中，portion 总体 40.3%。**dev CC = 0.3653（最终版）**。

## Round 9 — portion 无 Total 表的 sum-of-parts 分母（尝试后回撤）
whole = part+sibling 候选压过了更好的文本 whole 兜底，dev CC 0.3593 → 回撤，恢复 0.3653。

## 机制自检
- 真 verifier 与快速内嵌判定在 46/46 抽样题上一致；
- 确定性：所有排序均带稳定 tie-break；同进程 hash() 仅作分组键；每题恰好一个程序。

## 最终 dev 分型（n=1013, CC=0.3653）
growth 48.9% (90/184) | portion 33.3%+transfer 67.1% → 合计 40.3% (149/370) | copy 71.4% (30/42)
ratio 31.8% | diff 27.2% | margin 53.3% | extreme 50% | avg 23.9% | sum/sum_weak ~16% | extract 4.5%

## 最终 test 评测（一次性，真 verifier）
- gold 自洽子集：664/795（与预期一致）
- **test CC = 0.3343（222/664）** —— 低于 Kill 线 0.40，CodeFinQA 在 Gate 1 存活
- 分型：portion 34.6% (89/257) | growth 46.9% (61/130) | diff 29.9% | ratio 36.7% |
  avg 22.2% | margin 50% | sum 26.7% | sum_weak 11.1% | extract 7.1% | prod 0%
- 重要观察：copy / transfer 检索桶在 test 上零触发 → test 与 train 无共享 context，
  dev(0.365) 与 test(0.334) 的差主要来自 dev 的同 context 检索收益；模板部分迁移良好。
- 剩余上行空间（诚实评估）：文本操作数抽取（portion/growth 失败中 ~40-50% gold 操作数不在表内）、
  复合式 gold（sum-of-parts 分母、rate×principal、单位换算）。再投入数人日的规则工程，
  估计上限 ~0.40-0.45 —— 与 Kill 线的余量不大，Gate 1 结论应标注该敏感性。

# Gate 1 v2 solver 迭代日志 — 策略：表格结构解析 + 行列对齐选操作数（table_align）

评测：train 前 1200 条 gold 自洽子集（进程内快评，与 verifier 同容差/同守卫语义）；
test 仅在最终评一次（真 verifier 子进程）。

## Round 1 — 基线结构化实现（train CC = 13.1%）
- markdown 表格重建（首行 header、跳过 :--- 分隔行）→ (行标签, 列头, 单元格) 候选；
- 正文数字候选（±90 字符词窗、句子年份）；
- 问题分类 growth/portion/diff/sum/avg/ratio/other；growth+portion ×100；
- TF-IDF(纯词频统计) kNN 检索 train 问题 → 公式骨架迁移（用于 other 桶）；
- 失败模式：gold 公式拟合大量 None（正则把变量名里的年份当字面量）。

## Round 2 — AST 字面量抽取 + 宽容公式拟合（train CC = 14.8%）
- gold 程序数值改用 AST Constant（源码顺序），公式库扩到 sum5/avg5/pct_of_diff 等，
  任意操作数位置枚举拟合；
- 分类修正：percentage point→diff、diff 优先于 sum、ratio 优先于 diff、
  "by what percent did X increase"→growth；
- 模板：多年份元组对齐 pick_year_tuple、期初/期末 balance 行、年份区间展开、月份词表。
- 诊断结论：公式家族已对齐（portion→pct_ratio 155、growth→growth_pct 107、diff→diff 82
  都是公式对、操作数错），瓶颈在操作数选择；
- 发现结构性 bug：正文数字用句级年份集合 → "26.7,21.3 … 2011,2010 respectively"
  把同一个数字配到两个年份；multi-table context 全被并进第一张表的 header。

## Round 3 — 结构修复（train CC = 20.6%）
- multi-table 切块解析（:--- 分隔行的前一行为该表 header）；
- 正文 数字↔年份 强关联：respectively 位置配对 + 最近年份(±110 字符)；
  表格对齐元组加结构先验（同行 +2.5 / 同列 +2.0 / 正文 -1.0 且要求不同 token）；
- balance 滚动表（期初/期末行）优先于泛化行列对齐；月份对齐（share repurchase 表）。
- growth 命中 45%；portion 仍 14%（date 数字 "december 31" 的 31 被当候选、
  正文词窗过肥压过表格、den=Total 对齐缺失）。

## Round 4 — portion 重构 + 评分再平衡（train CC = 23.6%）
- 日期"日"数字剔除（月份名后 / ", YYYY" 前的 1-31 整数）；
- 评分：表格列头词权重 1.0→2.0，无空格复合表头子串匹配（半权），正文词窗降权 1.2/-1.5；
- portion 角色解析："of A to B"、"of total"（num 排除 total 行、den=同列 total 行/同行 Total 列）、
  whole 短语扩终止词；swap 保护仅当 den 无 total。
- 分类型：growth 101/192(53%)、portion 48/285(17%)、diff 41/169、sum 30/173、
  avg 29/108、ratio 20/62、other_knn 14/205。

## Round 5 — total 行修复 + portion 结构对齐（train CC = 26.5%）
- 发现 'total' 在 STOP 词表里 → Total 行/列全程不可见；移出；
- total_aligned 结构偏好：num 是 total 行 → 分母取同行 Total 列，否则同列 total 行；
- 列线索缺省时 num 切到同行 Total 列；"due in YYYY" 的 part 年份优先于句首报告年；
- 正文候选降权 (1.0/-2.0)。portion 42→80。

## Round 6 — diff/sum/avg 模板扩展（train CC = 27.5%）
- 滚动表 begin/end 行正则放宽（beginning/ending/end of/january 1）+ 选中期初行自动换期末行；
- "difference between A and B" 实体角色拆分（|大-小|）、季度→月份行、
  百分点/rate 类允许 % 候选进入年份对；
- sum: "due in five years" 年份行求和、季度三月份求和、跨句逐年选取、无 total 行的列合计；
- avg: "last N years" 最近 N 个年份列平均；"YYYY from YYYY" 年份解析。

## Round 7 — kNN 近邻操作数描述符迁移（train CC = 30.5%）
- fit_formula 返回操作数角色值；全 train 预计算 gold 操作数的 (词集, 年份角色, 表/文) 描述符
  （缓存 ta_lib_cache.json，3795/4668 条可用）；
- 近邻迁移：按描述符词集×IDF + 年份角色对齐在当前表选单元格，emit 对应公式；
- train 扫参（transfer vs rule 双跑）：growth/avg/diff 规则更强（迁移禁用），
  portion T=0.90 / ratio T=0.85 / sum T=0.95 / other T=0.50；
- 公式家族准确率：规则 growth 0.91、portion 0.83、ratio 1.00；kNN sim≥0.95 时 0.79。

## Round 8-9 — 正文模式 + 结构关系迁移（train CC = 31.0%）
- 正文同值"假年份对"过滤（"increased 14%" 复述）；
- "increased to X from Y"/"from Y to X" 直接抽取 (new, old)；
- portion part 词集变小时表格候选不被正文覆盖；delta 语义 den 优先正文增减句；
- 迁移描述符记录操作数间同行/同列关系，迁移时 +2.5 结构一致分。

## Hold-out 验证（train offset 1420 起 1200 条 clean，从未用于开发）
- CC = 38.3%（高于 dev 片 31.0%——dev 片更难，无过拟合迹象）。
- 分类型：growth 61%、portion 35%、knn_xfer 50%、diff 39%、ratio 41%、sum 15%。

## 协议合规
- 白名单内：正则/规则、markdown 表结构重建、TF-IDF 纯词频检索（无任何学习模型）、
  train 近邻程序骨架+操作数描述符迁移（"检索+模板"条款）、确定性规范化；
- 推理只读当前题 question+context；IDF/检索库/阈值常数全部由 train 构建/调优；
- 开发迭代只用 train（前 1200 clean 快循环 + offset1420 hold-out 验证一次）；
  test 只在最终用真 verifier 评一次（ta_test_final.out）；
- 开发用进程内快评（同容差/同反常量守卫语义），最终数字以真 verifier 为准；
- 确定性：无随机源，平手用固定排序键，每题恰好一个程序。

## 最终 test（gold 自洽子集，真 verifier，单次）
- n_eval = 664（与预期 664/795 一致），**CC = 0.3102**（206 hit / 458 miss）。
- 分类型命中率：growth 55/94=59%、portion 53/186=28%、knn_xfer 36/106=34%、
  ratio 14/37=38%、diff 16/65=25%、avg 12/49=24%、sum 15/83=18%、other 5/44=11%。
- 对比：dev 片 31.0%、hold-out 片 38.3% → test 31.0%，无 test 过拟合（test 只跑了这一次）。
- 结论：v2 solver 把 CC 从 gate1 的 2.9% 推到 31.0%，未越过 0.40 kill 线，
  但余量不大；剩余最大桶是 portion 操作数消歧（133 miss）、近邻迁移选格（70）、
  sum 加数集合确定（68）。若再投入（Total 行的列和一致性校验、portion 角色解析
  细化、迁移覆盖率提升），估计还能到 ~0.35-0.40，越线与否不确定。

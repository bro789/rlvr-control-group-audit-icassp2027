# solver_v2_rulelib 迭代日志（Gate 1 closure 审计）

评测协议：train gold 自洽子集前 1200 条为开发集（in-process fast verifier，与沙箱 verifier 判定逻辑一致）；test 仅最后评一次（用真 verifier 子进程沙箱）。

## 预分析（train 全量 gold 自洽子集 n=4035）
- AST 归一化 gold 程序模式：(N/N)*100 占 30.5%，((N-N)/N)*100 占 16.6%，N-N 占 12.5%，N/N 占 9.5%，N+N 4.3%，N*N 2.6%，N+N+N 2.4%，(N-100) 报酬表 2.2%，avg 家族 ~3%。top-8 覆盖 ~80%。
- 操作数 60% 全在表格、29% 表格+文本、11% 有算出的中间量。
- 规约：growth 答案 672/672 为正；diff 95% 为正 → 模板取 abs；portion 分子<分母 96%；portion 分母行标签含 total 54%；"by how much did X increase" 75% 是百分比而非绝对差。

## Round 1（基础版：词窗打分 + 9 类模板）
- dev CC = 20.3%（243/1200）。breakdown：growth 40%，portion 19%，diff 23%，sum 12%，avg 19%，extract 1%。
- 失败模式：日期噪声数字（"december 31" 的 31 被当操作数）；行标签匹配精度差；"what percent of the increase...due to" 误分类为 growth；年份行表格（rows=年份）没有列序列抽象。

## Round 2（Series 抽象：行序列+列序列、精确率罚项、junk 过滤）
- dev CC = 23.6%（283/1200）。
- oracle 分析（同一候选池下任意组合能否命中）：portion 172 个纯选择失误、growth 120、diff 89、sum 67 —— 选择器是瓶颈，headroom 大。
- 失败模式：行列序列重复候选导致 part==whole；"long-term" vs "long term" 分词不一致；表头实为数据行（'Beginning of year | 546'）；"of X to Y" 双实体切分缺失。

## Round 3（分词拆连字符/驼峰、表头数据行并入、label_year 过滤 change 行、pos 去重、portion 切分模式扩展）
- 初版反而回退到 20.2%：bisect 发现裸年份表头（'2010'）被误判为"表头即数据"，整表结构被破坏。修复后 dev CC = 24.9%。
- 继续 bisect：PORTION_OF_TO（"of X to Y" 切分）过度触发（"due to/related to" 也被切）使 portion -17；限制为 " to the " 且排除 due/related/attributable 后修复。

## Round 4（return 重写 + 表现表处理）
- 股价表现表：表头全空时提升首行为表头；实体序列（公司/指数）匹配；期末列默认取最后一列；"difference between A and B" → a-b；"ratio of A to B" → base-100 时 (a-100)/(b-100) 否则 a/b。
- dev CC = 27.0%（324/1200）。breakdown：growth 97/218，portion 75/272，diff 58/175，return 19/28，avg 32/110，sum 21/154，ratio 18/59，extract 4/183。

## Round 5（sum 修复 + extract 子规则 + max/min 类）
- sum：want-年份分支去词面门槛（年份行表格标签无内容词）、"five years" 规则、列求和规则（total 行该列为空 → 对列求和）、单年列跨行求和。avg 的 cover/逐年回退反而伤 avg（year-col 表里所有行都覆盖年份，纯覆盖排序会选错行），回退掉。
- extract 子规则：interest expense = 本金×问题内利率；margin/tax rate → 转 portion；"without Y" → 费用加回/收益扣减；highest/lowest → max/min 模板（"minimum lease payments" 陷阱要排除）。
- dev CC = 27.8%。

## Round 6（train 检索 + 程序骨架迁移，白名单"检索+模板"）
- 检索库：train gold 自洽子集问题 TF-IDF 倒排索引；查询排除同 context 的 train 条目（实测 test→train 同 context 率 0/150，dev 上必须同样排除才无偏；train 内部同 context 近重复率 ~17%）。
- 迁移：NN gold 程序 AST 中非常数字面量按"角色"（NN 表内行标签词集+年份）重定位到当前 context 候选，年份按问题年份双射映射，保留骨架与常数（×100、/2 等）。
- 调参（dev）：迁移抢先只对 portion（cos>=0.90）和 extract/times（cos>=0.60）有增益；growth/return/ratio/avg 规则更强，迁移抢先 -3~-12。solver 失败时以 cos>=0.60 迁移兜底。
- dev CC = 29.7%（356/1200）。

## Round 7（文本年份对齐关键 bug 修复）
- 重大 bug：NUMTOK 尾随空格使"年份后跟字母"守卫（防 FinQA 撇号转义 2019s）误杀所有句中年份 → 文本操作数年份错配严重。修复后文本两操作数类大涨。
- respectively 并列列表（"$109, $83 and $59 for 2016, 2015 and 2014, respectively"）成员共享全句词窗，只由年份区分——此前词窗不对称导致首个数字总赢。
- dev CC = 32.2%（386/1200）。growth 110/218，sum 30/152。

## Round 8（否定词失配罚项 + 日期日数过滤加强）
- 'approved' vs 'not approved' 行标签否定失配罚 -3；"31 , 2013" 型日期日数过滤；文本 "( x )" 负数试验（FinQA 里 "( x )" 多为重复标注而非负数，回退）。
- dev CC = 32.3%（387/1200）。
- 保留验证：train 后续 800 条 clean（未用于任何调参）CC = 37.5%（300/800）——无过拟合迹象。
- 冻结代码，test 全量只评一次（真沙箱 verifier）。

## 最终 test 结果（只评一次，真 verifier 子进程沙箱）
- **test CC = 0.3163（210/664）**，n_eval = 664（gold 自洽子集，与预注册的 664/795 一致），skipped_noisy_gold = 131。
- 分类型：growth 55/107（51%）、return 10/15（67%）、diff 27/67（40%）、portion 76/209（36%）、ratio 11/45（24%）、avg 8/52（15%）、sum 10/76（13%）、extract 11/91（12%）、mmax 2/2。
- 与 dev 32.3%、train 保留段 37.5% 一致，无过拟合迹象。
- 审计结论：CC = 0.32 < 0.40 判死线，任务在 Gate 1 存活。但注意这是构造性下界：oracle 分析显示候选池内可命中组合远多于当前命中（选择器仍是瓶颈），portion 的复合分母（列求和/差额）、单位换算（×1000、/1e6）、乘法类（价×量）等桶还有规则空间；再投入约一人周规则工程，估计可到 0.36-0.42。0.40 的红线余量不算舒适，建议在论文中如实报告该下界与其增长曲线（8 轮迭代 0.03→0.32）。

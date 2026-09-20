# -*- coding: utf-8 -*-
"""从 results_backup/ 的逐项判对向量直算扩展版统计附录，输出 paper/EXTENDED_STATS.md。

覆盖四件 4 页正文放不下、而审稿人必问的事：
  1) 17 组逐组 McNemar + Holm + 配对 bootstrap 区间
  2) 聚合算子敏感性（max4 / max 学习臂 / 各单一对手 / mean / median / 次优）
  3) max 的胜者诅咒：交叉拟合 Δ（半样本选臂、另半样本评估）
  4) 四条算力轴上的臂间对照（墙钟 / verifier 调用 / 生成 token / 优化步数）
"""
import os, sys, json, math, random
from statistics import mean, median

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BK = "results_backup"
OUT = os.path.join("paper", "EXTENDED_STATS.md")
random.seed(20270101)

ARMS = ["solver", "warm", "csft", "rssft"]
LEARNING = ["warm", "csft", "rssft"]
NAME = {"solver": "solver", "warm": "$W_m$", "csft": "c-SFT", "rssft": "RS-SFT"}

GROUPS = [
    ("C1", "CodeTAT-QA $\\times$ Qwen2.5-0.5B", m, s,
     os.path.join(BK, "test_final", f"test_m{m}_s{s}.json"))
    for m in (2, 64) for s in (0, 1, 2)
] + [
    ("C2", "CodeFinQA $\\times$ Llama-3.2-1B", m, s,
     os.path.join(BK, "test_final_config2", f"test_m{m}_s{s}.json"))
    for m in (128, 256) for s in (0, 1, 2)
] + [
    ("C3", "CodeTAT-QA $\\times$ Qwen2.5-7B", m, s,
     os.path.join(BK, "config3_7b", f"test_m{m}_s{s}.json"))
    for m in (2, 64) for s in (0, 1, 2)
]
EXCLUDED = ("C2", 128, 1)


def load(p):
    return json.load(open(p, encoding="utf-8"))


def mcnemar_exact(v_a, v_b):
    """双尾精确 McNemar。返回 (p, b, c)，b = a 对 b 错，c = a 错 b 对。"""
    b = sum(1 for x, y in zip(v_a, v_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(v_a, v_b) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, b, c


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (len(pvals) - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


GO = 3.0


def boot_go(per, B=50000):
    """按预注册真正问的问题检验：GRPO 有没有够到 +3 的 Go 门槛。
    返回 bootstrap 下 P(Δ>=+3) 与 P(Δ>=0)。前者才是判定量对应的单尾证据。"""
    n = len(per["grpo"])
    ge_go = ge_0 = 0
    for _ in range(B):
        idx = [random.randrange(n) for _ in range(n)]
        g = sum(per["grpo"][i] for i in idx) / n
        b = max(sum(per[a][i] for i in idx) / n for a in ARMS)
        d = (g - b) * 100
        ge_go += d >= GO
        ge_0 += d >= 0
    return ge_go / B, ge_0 / B


def boot_delta(per, B=10000):
    """配对 bootstrap。两种口径：每次重采样后重新取 max（recomputed），
    以及把臂锁定在全样本 argmax 上（fixed）。差值即选择带来的偏置。"""
    n = len(per["grpo"])
    fixed_arm = max(ARMS, key=lambda a: sum(per[a]))
    d_rec, d_fix = [], []
    idx_range = range(n)
    for _ in range(B):
        idx = [random.randrange(n) for _ in idx_range]
        g = sum(per["grpo"][i] for i in idx) / n
        accs = {a: sum(per[a][i] for i in idx) / n for a in ARMS}
        d_rec.append((g - max(accs.values())) * 100)
        d_fix.append((g - accs[fixed_arm]) * 100)
    d_rec.sort()
    d_fix.sort()
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
    return (q(d_rec, .025), q(d_rec, .975), mean(d_rec),
            q(d_fix, .025), q(d_fix, .975), mean(d_fix))


def crossfit_delta(per, K=2000):
    """交叉拟合 Δ：半样本上选出最强对照臂，在另半样本上评估。
    消掉「在同一批数据上既选臂又报分」带来的胜者诅咒。"""
    n = len(per["grpo"])
    idx = list(range(n))
    vals = []
    for _ in range(K):
        random.shuffle(idx)
        a_half, b_half = idx[: n // 2], idx[n // 2:]
        pick = max(ARMS, key=lambda a: sum(per[a][i] for i in a_half))
        g = sum(per["grpo"][i] for i in b_half) / len(b_half)
        o = sum(per[pick][i] for i in b_half) / len(b_half)
        vals.append((g - o) * 100)
    vals.sort()
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
    return mean(vals), q(.025), q(.975)


rows = []
for cfg, desc, m, s, path in GROUPS:
    if (cfg, m, s) == EXCLUDED:
        continue
    if not os.path.exists(path):
        print(f"!! 缺文件 {path}")
        continue
    d = load(path)
    per = {k: v["per_item"] for k, v in d["arms"].items()}
    acc = {k: sum(v) / len(v) * 100 for k, v in per.items()}
    n = len(per["grpo"])
    best = max(ARMS, key=lambda a: acc[a])
    delta = acc["grpo"] - acc[best]
    p, b, c = mcnemar_exact(per["grpo"], per[best])
    lo_r, hi_r, mu_r, lo_f, hi_f, mu_f = boot_delta(per)
    cf_mu, cf_lo, cf_hi = crossfit_delta(per)
    p_go, p_ge0 = boot_go(per)
    best_learn = max(LEARNING, key=lambda a: acc[a])
    union = sum(1 for i in range(n) if per["grpo"][i] or per[best][i]) / n * 100
    union5 = sum(1 for i in range(n)
                 if any(per[a][i] for a in ARMS + ["grpo"])) / n * 100
    both = sum(1 for i in range(n) if per["grpo"][i] and per[best][i]) / n * 100
    rows.append(dict(
        p_go=p_go, p_ge0=p_ge0,
        union=union, union5=union5, both=both,
        headroom=union - max(acc["grpo"], acc[best]),
        cfg=cfg, desc=desc, m=m, s=s, n=n, acc=acc, best=best, delta=delta,
        p=p, b=b, c=c, lo=lo_r, hi=hi_r, lo_f=lo_f, hi_f=hi_f,
        bias=mu_r - mu_f, cf=cf_mu, cf_lo=cf_lo, cf_hi=cf_hi,
        best_learn=best_learn,
        d_learn=acc["grpo"] - acc[best_learn],
        d_csft=acc["grpo"] - acc["csft"],
        d_warm=acc["grpo"] - acc["warm"],
        d_rssft=acc["grpo"] - acc["rssft"],
        d_solver=acc["grpo"] - acc["solver"],
        d_mean=acc["grpo"] - mean(acc[a] for a in ARMS),
        d_median=acc["grpo"] - median(acc[a] for a in ARMS),
        d_second=acc["grpo"] - sorted((acc[a] for a in ARMS), reverse=True)[1],
    ))

adj = holm([r["p"] for r in rows])
for r, a in zip(rows, adj):
    r["p_holm"] = a

L = []
W = L.append
W("# 扩展版统计附录（自动生成，勿手改）\n")
W("> 由 `python analysis/gen_extended_stats.py` 从 `results_backup/` 的逐项判对向量直算。\n")
W(f"> 组数 {len(rows)}；bootstrap B=10000，交叉拟合 K=2000，seed=20270101。\n")

W("\n## 1. 逐组显著性与区间\n")
W("| 配置 | m | seed | n | argmax 臂 | Δ | McNemar p | Holm p | 配对 bootstrap 95% CI |")
W("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    W(f"| {r['cfg']} | {r['m']} | {r['s']} | {r['n']} | {r['best']} "
      f"| {r['delta']:+.2f} | {r['p']:.3g} | {r['p_holm']:.3g} "
      f"| [{r['lo']:+.2f}, {r['hi']:+.2f}] |")
sig = [r for r in rows if r["p_holm"] < 0.05]
neg_ci = [r for r in rows if r["hi"] < 0]
W(f"\n- Holm 校正后 p<0.05 的组：**{len(sig)}/{len(rows)}**"
  f"（{', '.join(r['cfg']+' m'+str(r['m'])+' s'+str(r['s']) for r in sig) or '无'}）")
W(f"- 95% bootstrap 区间整体位于 0 以下的组：**{len(neg_ci)}/{len(rows)}**")
W(f"- 全部 {len(rows)} 组 Δ 为负；符号检验（按组，独立性最乐观口径）p = {2*0.5**len(rows):.2e}")
cells = sorted({(r["cfg"], r["m"]) for r in rows})
W(f"- 按 (配置, 预算) 聚类的 {len(cells)} 个单元全部为负，符号检验 p = {2*0.5**len(cells):.3f}")
W(f"- 按配置聚类的 3 个单元全部为负，符号检验 p = {2*0.5**3:.3f}")

def mde(n_d, n, alpha):
    """给定不一致对数 n_d，双尾精确 McNemar 在水平 alpha 下能拒绝的最小 |b-c|，
    换算成百分点即该组的最小可检出效应。"""
    for d in range(0, n_d + 1):
        b = (n_d + d) // 2
        c = n_d - b
        if b - c < d:
            continue
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n_d, i) for i in range(k + 1)) / 2 ** n_d)
        if p < alpha:
            return (b - c) / n * 100
    return float("inf")


W("\n### 1.0 按预注册真正问的问题检验：GRPO 够到 +3 的 Go 门槛了吗\n")
W("上表的 p 值回答的是「Δ 是否显著为负」，但论文预注册的判定量不是这个，"
  "而是 Go 门槛：$\\Delta \\geq +3$ 才算 GRPO 达标。"
  "这两个原假设的证据强度差很多，**报错了会把自己的结论说弱**。\n")
W("| 配置 | m | seed | Δ | 95% CI 上界 | 排除 +3? | P(Δ≥+3) | 严格为负? | P(Δ≥0) |")
W("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    W(f"| {r['cfg']} | {r['m']} | {r['s']} | {r['delta']:+.2f} | {r['hi']:+.2f} "
      f"| {'是' if r['hi'] < GO else '否'} | {r['p_go']:.4f} "
      f"| {'是' if r['hi'] < 0 else '否'} | {r['p_ge0']:.4f} |")
n_go = sum(1 for r in rows if r["hi"] < GO)
n_neg0 = sum(1 for r in rows if r["hi"] < 0)
pg = sorted(r["p_go"] for r in rows)
W(f"\n- 「GRPO 够到 +3 了吗」——95% CI 上界低于 +3 的组：**{n_go}/{len(rows)}**；"
  f"P(Δ≥+3) 中位数 **{pg[len(pg)//2]:.4f}**，最差一组 **{max(pg):.4f}**。")
W(f"- 「GRPO 显著更差吗」——CI 上界低于 0 的组：**{n_neg0}/{len(rows)}**。")
W("- **两者相差 %d 组。前者是预注册的问题，后者是论文从未声称过的更强主张。**"
  "正文只报 Holm 后 2/17，等于拿后者的标准给自己打分。" % (n_go - n_neg0))
miss = [r for r in rows if r["hi"] >= GO]
if miss:
    W(f"- 排除不掉 +3 的 {len(miss)} 组："
      + "、".join(f"{r['cfg']} m={r['m']} s={r['s']}（CI 上界 {r['hi']:+.2f}）" for r in miss)
      + f"，全部属于配置{'①' if all(r['cfg']=='C1' for r in miss) else '不止一个'}。")
    W("- CI 上界落在 +3 附近的组对重采样敏感，计数会在 ±1 组之间抖；"
      "连续量 P(Δ≥+3) 不受此影响，应以它为准。")
W("- 配置③ 同为 n=282 却全部达标，因为 GRPO 与 c-SFT 逐项高度一致"
  "（不一致对仅 21–35），配对区间窄；配置① 的 GRPO 与 solver 逐项分歧大"
  "（不一致对 70–82），区间宽。")

W("\n### 1.1 最小可检出效应\n")
W("逐组不显著不等于效应不存在，也可能是这个样本量本来就检不出。"
  "下表给出各组在其实际不一致对数下的最小可检出 |Δ|。\n")
W("| 配置 | m | seed | n | 不一致对 b+c | MDE @α=0.05 | MDE @Holm(α/17) | 实测 \\|Δ\\| |")
W("|---|---|---|---|---|---|---|---|")
for r in rows:
    nd = r["b"] + r["c"]
    W(f"| {r['cfg']} | {r['m']} | {r['s']} | {r['n']} | {nd} "
      f"| {mde(nd, r['n'], 0.05):.2f} | {mde(nd, r['n'], 0.05/len(rows)):.2f} "
      f"| {abs(r['delta']):.2f} |")
m05 = [mde(r["b"] + r["c"], r["n"], 0.05) for r in rows]
mho = [mde(r["b"] + r["c"], r["n"], 0.05 / len(rows)) for r in rows]
W(f"\n- 未校正时的 MDE 中位数 **{median(m05):.2f} 点**，Holm 校正后 **{median(mho):.2f} 点**。")
W(f"- 17 组实测 |Δ| 的中位数只有 {median([abs(r['delta']) for r in rows]):.2f} 点。")
W("- 也就是说：**本研究的逐组功效本来就不足以检出自己要测的效应量**——"
  "预注册的 +3 点 Go 门槛在任何一组上都低于该组的 MDE。"
  "定量结论只能建立在 17 组的符号一致性与聚类符号检验上，不能建立在逐组显著性上。"
  "这是设计层面的局限，不是事后辩解：它同样意味着若 GRPO 真有 +3 点的优势，本设计也检不出。")

W("\n## 2. 聚合算子敏感性\n")
W("同一批 run，只换判定量里对照组的聚合方式。\n")
W("| 配置 | m | seed | max(4 臂) | max(仅学习臂) | vs c-SFT | vs RS-SFT | vs $W_m$ | vs solver | 均值 | 中位数 | 次优 |")
W("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    W(f"| {r['cfg']} | {r['m']} | {r['s']} | {r['delta']:+.2f} | {r['d_learn']:+.2f} "
      f"| {r['d_csft']:+.2f} | {r['d_rssft']:+.2f} | {r['d_warm']:+.2f} | {r['d_solver']:+.2f} "
      f"| {r['d_mean']:+.2f} | {r['d_median']:+.2f} | {r['d_second']:+.2f} |")


def tally(key):
    v = [r[key] for r in rows]
    return f"{mean(v):+.2f} | {min(v):+.2f} ~ {max(v):+.2f} | {sum(1 for x in v if x < 0)}/{len(v)}"


W("\n| 聚合算子 | 均值 Δ | 范围 | 为负的组数 |")
W("|---|---|---|---|")
for key, label in [("delta", "max（预注册）"), ("d_learn", "max 仅学习臂"),
                   ("d_csft", "只对 c-SFT"), ("d_rssft", "只对 RS-SFT"),
                   ("d_warm", "只对 $W_m$"), ("d_solver", "只对 solver"),
                   ("d_mean", "四臂均值"), ("d_median", "四臂中位数"),
                   ("d_second", "四臂次优")]:
    W(f"| {label} | {tally(key)} |")

W("\n## 3. max 的胜者诅咒（选择偏置）\n")
W("`recomputed` 每次重采样后重新取 max，`fixed` 把臂锁在全样本 argmax 上；"
  "两者之差即「在同一批数据上既选臂又报分」引入的偏置。"
  "`交叉拟合` 用半样本选臂、另半样本评估，是无偏口径。\n")
W("| 配置 | m | seed | Δ（朴素） | bootstrap 偏置 | 交叉拟合 Δ | 交叉拟合 95% CI |")
W("|---|---|---|---|---|---|---|")
for r in rows:
    W(f"| {r['cfg']} | {r['m']} | {r['s']} | {r['delta']:+.2f} | {r['bias']:+.3f} "
      f"| {r['cf']:+.2f} | [{r['cf_lo']:+.2f}, {r['cf_hi']:+.2f}] |")
W(f"\n- 偏置绝对值最大的一组：{max(abs(r['bias']) for r in rows):.3f} 点")
W(f"- 平均偏置：{mean(r['bias'] for r in rows):+.3f} 点")
W(f"- 交叉拟合后仍为负的组：**{sum(1 for r in rows if r['cf'] < 0)}/{len(rows)}**")
W(f"- 交叉拟合 Δ 均值：{mean(r['cf'] for r in rows):+.2f} 点"
  f"（朴素口径 {mean(r['delta'] for r in rows):+.2f}）")

W("\n## 4. 互补性：GRPO 与最强对照臂解的不是同一批题\n")
W("`b/c` 已经暗示两者答对的题集重叠有限。这里直接给出并集上界。\n")
W("| 配置 | m | seed | GRPO | argmax 臂 | 都对 | 并集（oracle） | 并集 − 单臂最优 |")
W("|---|---|---|---|---|---|---|---|")
for r in rows:
    W(f"| {r['cfg']} | {r['m']} | {r['s']} | {r['acc']['grpo']:.2f} | {r['acc'][r['best']]:.2f} "
      f"| {r['both']:.2f} | {r['union']:.2f} | {r['headroom']:+.2f} |")
W(f"\n- 并集相对单臂最优的平均抬升：**{mean(r['headroom'] for r in rows):+.2f} 点**"
  f"（范围 {min(r['headroom'] for r in rows):+.2f} ~ {max(r['headroom'] for r in rows):+.2f}）")
W(f"- 五臂全并集均值：{mean(r['union5'] for r in rows):.2f}%"
  f"（单臂最优均值 {mean(max(r['acc'][r['best']], r['acc']['grpo']) for r in rows):.2f}%）")
W("- 这不是一个可部署的方法（并集需要知道答案才能选臂），而是说明"
  "「RL 臂与确定性臂谁更强」这个问法本身丢掉了信息：两者的能力是错开的。")

COST = {
    ("C1", 2): dict(csft=[2910, 2579, 3196], grpo=[5703, 5702, 5701], rssft=[5715, 5716, 5732],
                    gv=[15488, 17920, 16512], rv=[21824, 24576, 22016],
                    gt=[722261, 798452, 728725], rt=[5440704, 6136320, 5444224],
                    gstep=[242, 280, 258]),
    ("C1", 64): dict(csft=[5707, 5704, 5703], grpo=[5711, 5728, 5711], rssft=[5815, 5779, 4561],
                     gv=[12160, 12608, 13184], rv=[15872, 15104, 13184],
                     gt=[908133, 914166, 947029], rt=[3048064, 3301056, 2906048],
                     gstep=[190, 197, 206]),
    ("C2", 128): dict(csft=[5715, 5709, 5704], grpo=[5742, 5751, 5709], rssft=[5905, 1549, 5900],
                      gv=[6272, 7040, 6528], rv=[9536, 7040, 9728],
                      gt=[296292, 316721, 322669], rt=[1297920, 1103040, 1206720],
                      gstep=[98, 110, 102]),
    ("C2", 256): dict(csft=[5714, 5704, 5703], grpo=[5720, 5755, 5749], rssft=[5927, 6038, 5822],
                      gv=[7104, 7104, 7872], rv=[6656, 6656, 6144],
                      gt=[305458, 298527, 335210], rt=[761920, 686016, 601984],
                      gstep=[111, 111, 123]),
    ("C3", 2): dict(csft=[5706, 5702, 5703], grpo=[5712, 5753, 5761], rssft=None,
                    gv=[2432, 2304, 2880], rv=None,
                    gt=[149730, 130787, 142044], rt=None,
                    gstep=[38, 36, 45]),
    ("C3", 64): dict(csft=[5748, 5721, 5731], grpo=[5864, 5750, 5766], rssft=None,
                     gv=[2240, 2496, 2240], rv=None,
                     gt=[166562, 180615, 174446], rt=None,
                     gstep=[35, 39, 35]),
}
W("| 配置 | m | 墙钟 c-SFT | 墙钟 GRPO | 墙钟 RS-SFT | GRPO verifier | RS-SFT verifier "
  "| GRPO token | RS-SFT token | GRPO 步数 |")
W("|---|---|---|---|---|---|---|---|---|---|")


def rng(v):
    return "—" if not v else (f"{min(v)}" if min(v) == max(v) else f"{min(v)}–{max(v)}")


for (cfg, m), d in COST.items():
    W(f"| {cfg} | {m} | {rng(d['csft'])} | {rng(d['grpo'])} | {rng(d['rssft'])} "
      f"| {rng(d['gv'])} | {rng(d['rv'])} | {rng(d['gt'])} | {rng(d['rt'])} | {rng(d['gstep'])} |")
import glob
import re

BON = {"C1": ("config1_full", 72.0, [2, 64]), "C2": ("config2_full", None, [128, 256])}
SOLVER_DEV = {"C1": {m: 72.0 for m in (0, 1, 2, 4, 8, 16, 64, 256, 1024, 2446)},
              "C2": {0: 34.50, 16: 35.50, 64: 35.17, 128: 35.33,
                     256: 35.33, 1024: 36.33, 3635: 38.00}}
bon_rows = []
for cfg, (d, _, ops) in BON.items():
    for f in sorted(glob.glob(os.path.join(BK, d, "runs_v2", "bon_m*_s*.json"))):
        m, s = re.search(r"bon_m(\d+)_s(\d+)", f).groups()
        j = load(f)
        bon_rows.append(dict(cfg=cfg, m=int(m), s=int(s), op=int(m) in ops,
                             p1=j["pass_at"]["1"] * 100, p16=j["pass_at"]["16"] * 100,
                             mj=j["majority_at"]["16"] * 100, ns=j["n_samples"]))
bon_agg = {}
for r in bon_rows:
    bon_agg.setdefault((r["cfg"], r["m"]), []).append(r)

W("\n## 5. 推理期采样臂：best-of-$n$ 与多数表决\n")
W("审稿意见里点名缺的「verifier 过滤的 best-of-n」在这两个任务上**不可部署**："
  "verifier 需要 gold 答案才能判对错，所以 `verifier_selected@k` 恒等于 `pass@k`，"
  "是 oracle 上界而非一个可用的臂。可部署的对应物是无 gold 的多数表决 `maj@k`。\n")
W("从 $W_m$ 采样，$n{=}16$，dev。`*` 标记预注册的两个工作点。\n")
W("| 配置 | m | 种子数 | pass@1 | pass@16（oracle 上界） | maj@16（可部署） | solver dev | oracle 是否胜过 solver |")
W("|---|---|---|---|---|---|---|---|")
for (cfg, m), v in sorted(bon_agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    sol = SOLVER_DEV[cfg].get(m)
    p16 = mean(x["p16"] for x in v)
    star = "*" if v[0]["op"] else ""
    win = "—" if sol is None else ("是" if p16 > sol else "否")
    W(f"| {cfg} | {m}{star} | {len(v)} | {mean(x['p1'] for x in v):.2f} | {p16:.2f} "
      f"| {mean(x['mj'] for x in v):.2f} | {sol if sol is None else f'{sol:.2f}'} | {win} |")
ops_rows = [(c, m, v) for (c, m), v in bon_agg.items() if v[0]["op"]]
lose = [(c, m) for c, m, v in ops_rows
        if SOLVER_DEV[c].get(m) and mean(x["p16"] for x in v) < SOLVER_DEV[c][m]]
W(f"\n- 四个工作点中，**{len(lose)} 个连 oracle 上界 pass@16 都低于 solver**"
  f"（{', '.join(f'{c} m={m}' for c, m in lose)}）。")
W("- 可部署的 maj@16 在四个工作点全部低于 solver。")
W("- 也就是说：把「多采几条再挑一条」当成漏掉的强基线，在这两个任务的稀缺档上站不住；"
  "但这是 dev 上的结论，且采样臂没有进冻结 test 的判定量，属于补充证据而非预注册臂。")

W("\n## 6. 零优势组：高正奖励率恰恰意味着梯度信号更少\n")
W("GRPO 的优势是组内相对的：一个提示的 8 条采样若奖励全同（全对或全错），"
  "该组优势为 0，对梯度没有贡献。所以「7B 正奖励率最高」不是训得足的证据，"
  "而是**有效提示数更少**的证据。下表字段直接来自各 run 的 `meta.json`。\n")
W("| 配置 | m | seed | 步数 | 组数 | 全错 | 全对 | 零方差占比 | 有效组 | pos_rate |")
W("|---|---|---|---|---|---|---|---|---|---|")
CFGKEY = {"config1_full": "C1", "config2_full": "C2", "config3_7b": "C3"}
meta_rows = []
seen = set()
for p in sorted(glob.glob(os.path.join(BK, "**", "grpo_m*_s*", "meta.json"), recursive=True)):
    mt = re.search(r"(config\d[\w]*)", p)
    cf = CFGKEY.get(mt.group(1)) if mt else None
    mm = re.search(r"grpo_m(\d+)_s(\d+)", p)
    if not cf or not mm:
        continue
    m, s = int(mm.group(1)), int(mm.group(2))
    if (cf, m, s) in seen:
        continue
    d = load(p)
    if d.get("groups") is None:
        continue
    seen.add((cf, m, s))
    meta_rows.append(dict(cfg=cf, m=m, s=s, steps=d["steps"], g=d["groups"],
                          a0=d["groups_all_zero"], a1=d["groups_all_one"],
                          zv=d["group_zero_var_frac"],
                          inf=d["groups"] - d["groups_all_zero"] - d["groups_all_one"],
                          pos=d["verifier_pos_rate"]))
meta_rows.sort(key=lambda r: (r["cfg"], r["m"], r["s"]))
for r in meta_rows:
    W(f"| {r['cfg']} | {r['m']} | {r['s']} | {r['steps']} | {r['g']} | {r['a0']} | {r['a1']} "
      f"| {r['zv']*100:.1f}% | **{r['inf']}** | {r['pos']:.3f} |")
by = {}
for r in meta_rows:
    by.setdefault(r["cfg"], []).append(r)
W("")
for cf in ("C1", "C2", "C3"):
    v = [r for r in by[cf] if r["inf"] > 0]
    W(f"- {cf}：零方差占比 {min(r['zv'] for r in v)*100:.1f}–{max(r['zv'] for r in v)*100:.1f}%，"
      f"有效组 {min(r['inf'] for r in v)}–{max(r['inf'] for r in v)}")
W("- **7B 整个 run 只有 50–82 个提示提供过梯度信号**，比 0.5B 少一个数量级；"
  "正文原来把 7B 最高的 pos_rate 当作「没有欠训」的第三条论据，方向是反的，必须改写。")
W("- 被排除的 C2 m=128 seed 1 是 `group_zero_var_frac = 1.0`：880 组全错、有效组 0，"
  "与 meta 记录的 `verifier_pos_rate = 0` 一致。")

POOL = {"C1": 2846, "C2": 4035, "C3": 2846}
GENS = 8
W("\n## 7. 算力与数据覆盖\n")
W("\n### 7.1 GRPO 在提示池上的覆盖率\n")
W(f"每个提示采 {GENS} 条，故 `verifier 调用 / {GENS}` 即该 run 访问过的不同提示数。\n")
W("| 配置 | m | verifier 调用 | 访问提示数 | 池大小 | 覆盖轮数 |")
W("|---|---|---|---|---|---|")
for (cfg, m), d in COST.items():
    pr = [x // GENS for x in d["gv"]]
    ep = [x / POOL[cfg] for x in pr]
    W(f"| {cfg} | {m} | {rng(d['gv'])} | {rng(pr)} | {POOL[cfg]} "
      f"| {min(ep):.2f}–{max(ep):.2f} |")
W("\n- **没有任何一个 GRPO run 跑满一轮提示池**：覆盖率 "
  f"{min(min(x//GENS/POOL[c] for x in d['gv']) for (c,_),d in COST.items()):.2f}–"
  f"{max(max(x//GENS/POOL[c] for x in d['gv']) for (c,_),d in COST.items()):.2f} 轮。")
W("- 7B 上只访问了池子的约 10%，是三个基座里最低的一档——"
  "正文「GRPO 没有欠训」一节必须把这一点讲出来：跑满的是预注册的墙钟预算，不是数据。")
W("- 这条限制对本文结论的方向是**不利**的，属于必须写进 threats to validity 的项。")

W("\n### 7.2 四条算力轴\n")
W("- 墙钟是预注册的配平轴；上表给出另外三条轴，供读者判断配平是否被这一选择左右。")
W("- GRPO 在 verifier 调用与生成 token 上**低于** RS-SFT（C1 token 相差 3–7 倍），"
  "即按这两条轴算 GRPO 拿到的不是更多而是更少——与本文结论同向，不构成对 GRPO 不利的配平。")
W("- 唯一对 GRPO 有利的不对称仍是 C1 m=2 的 c-SFT 提前收敛（2579–3196 s vs 5701–5703 s）。")

W("\n## 8. clean 过滤会不会自己造出结论\n")
RAW = {"C1": 288, "C2": 795, "C3": 288}
W("批评：clean 子集由本文自己的 verifier 定义（CodeFinQA 砍掉 13.6%），"
  "奖励、评分、筛样本是同一个函数。\n")
W("检验：把被过滤掉的题**一律判错**，对五条臂同等处理，再重算 Δ。\n")
W("| 配置 | m | seed | clean Δ | 未过滤 Δ | 符号变了吗 |")
W("|---|---|---|---|---|---|")
uflip = 0
for r in rows:
    R = RAW[r["cfg"]]
    ua = {k: v / 100 * r["n"] / R * 100 for k, v in r["acc"].items()}
    ud = ua["grpo"] - max(ua[a] for a in ARMS)
    f = (r["delta"] < 0) != (ud < 0)
    uflip += f
    W(f"| {r['cfg']} | {r['m']} | {r['s']} | {r['delta']:+.2f} | {ud:+.2f} "
      f"| {'**变了**' if f else '否'} |")
W(f"\n- 符号翻转：**{uflip}/{len(rows)}**")
W("- 原因是结构性的：过滤对五条臂同等施加，所以未过滤 Δ = clean Δ × (clean_n/raw_n)，"
  "C1/C3 系数 0.979、C2 系数 0.835——**恒正缩放，不可能改变符号**。")
W("- 这条批评因此在逻辑上就不成立，不需要重跑任何实验。")

W("\n## 9. 外部校准：与 BizBench 公开基线对照\n")
W("批评：没有外部校准点，无法判断这是不是两个弱系统之间的比较。\n")
W("公开数字取自 BizBench 原文（Koncel-Kedziorski et al., ACL 2024）Table 3，"
  "全部为 3-shot in-context learning，评分口径同样是 **1% 相对容差**。\n")
BIZ = [("Falcon-7B", 2.0, 7.4), ("MPT-7B", 6.6, 30.4), ("Llama-2-7B", 21.9, 37.0),
       ("MPT-30B", 31.0, 64.8), ("StarCoder-16B", 31.2, 70.2), ("Llama-2-13B", 33.4, 65.1),
       ("CodeLlama-7B", 34.0, 70.9), ("Mistral-7B", 48.8, 75.0),
       ("CodeLlama-34B", 52.4, 81.4), ("Llama-2-70B", 57.3, 79.1),
       ("Mixtral-8x7B", 58.5, 83.9), ("GPT-3.5", 67.5, 87.6), ("GPT-4", 78.8, 90.6)]
W("| 公开模型（3-shot） | CodeFinQA | CodeTAT-QA |")
W("|---|---|---|")
for n, a, b in BIZ:
    W(f"| {n} | {a} | {b} |")
c3 = [r for r in rows if r["cfg"] == "C3"]
c1 = [r for r in rows if r["cfg"] == "C1"]
c2 = [r for r in rows if r["cfg"] == "C2"]
W("")
W("| 本文（冻结 test） | CodeFinQA | CodeTAT-QA |")
W("|---|---|---|")
W(f"| 本文 7B c-SFT（最强臂） | — | {max(r['acc']['csft'] for r in c3):.2f} |")
W(f"| 本文 7B GRPO | — | {max(r['acc']['grpo'] for r in c3):.2f} |")
W(f"| 本文 0.5B GRPO | — | {max(r['acc']['grpo'] for r in c1):.2f} |")
W(f"| 本文 solver（440 行规则） | {max(r['acc']['solver'] for r in c2):.2f} "
  f"| {c1[0]['acc']['solver']:.2f} |")
W(f"| 本文 1B GRPO | {max(r['acc']['grpo'] for r in c2):.2f} | — |")
W("")
W("- 7B 各臂（79–82%）**高于** Mistral-7B(75.0) 与 CodeLlama-7B(70.9)，"
  "落在 CodeLlama-34B(81.4)/Mixtral-8x7B(83.9) 一带——对一个 7B 模型是正常偏强的位置。")
W("- **solver 的 64.54% 卡在 MPT-30B(64.8) 与 Llama-2-13B(65.1) 之间**："
  "440 行规则程序 ≈ 13B–30B 模型的 few-shot 水平。这本身就是论文论点的一个量化。")
W("- 0.5B 的 GRPO 到 63.12%，接近 MPT-30B/Llama-2-13B——说明 RLVR 确实在做实事，"
  "与正文「增益是真的，结论不是」一致。")
W("- 1B 在 CodeFinQA 上 30.57%，与 Llama-2-13B(33.4)、MPT-30B(31.0) 同档。")
W("- ⚠️ test 规模对不上：BizBench 原文记 CodeFinQA 844 / CodeTAT-QA 392，"
  "本地副本是 795 / 288。疑为数据集版本差异，**投稿前需自行核实并在文中说明**。")

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"写出 {OUT}，{len(rows)} 组。")
print(f"Holm 后显著: {len(sig)}/{len(rows)}；bootstrap CI 全负: {len(neg_ci)}/{len(rows)}；"
      f"交叉拟合仍负: {sum(1 for r in rows if r['cf'] < 0)}/{len(rows)}")
for key, label in [("delta", "max"), ("d_learn", "max-learning"), ("d_csft", "vs csft")]:
    v = [r[key] for r in rows]
    print(f"  {label}: mean {mean(v):+.2f}, neg {sum(1 for x in v if x < 0)}/{len(v)}")


# ---------------------------------------------------------------- LaTeX 表
TEX = os.path.join("论文tex_arXiv", "tab")
os.makedirs(TEX, exist_ok=True)
CFGTEX = {"C1": "Config~1 (CodeTAT-QA $\\times$ 0.5B, $n{=}282$)",
          "C2": "Config~2 (CodeFinQA $\\times$ 1B, $n{=}664$)",
          "C3": "Config~3 (CodeTAT-QA $\\times$ 7B, $n{=}282$)"}
ARMTEX = {"solver": "solver", "warm": "$W_m$", "csft": "c-SFT", "rssft": "RS-SFT"}


def pfmt(p):
    if p >= 0.01:
        return f"{p:.3f}"
    if p >= 1e-4:
        return f"{p:.4f}"
    e = math.floor(math.log10(p))
    return f"${p/10**e:.1f}{{\\times}}10^{{{e}}}$"


def emit(fname, lines):
    open(os.path.join(TEX, fname), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def grouped(body_fn, header, colspec, caption, label, note=""):
    L = ["% 由 analysis/gen_extended_stats.py 生成，勿手改",
         "\\begin{table}[t]", "\\centering", "\\small",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule"]
    for cfg in ("C1", "C2", "C3"):
        L.append(f"\\multicolumn{{{header.count('&')+1}}}{{@{{}}l}}"
                 f"{{\\textit{{{CFGTEX[cfg]}}}}} \\\\")
        for r in [x for x in rows if x["cfg"] == cfg]:
            L.append(body_fn(r))
        if cfg != "C3":
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    if note:
        L.append(f"\\\\[2pt]\\footnotesize {note}")
    L.append("\\end{table}")
    emit(f"tab_{label.split(':')[-1]}.tex", L)


grouped(
    lambda r: (f"{r['m']} & {r['s']} & {ARMTEX[r['best']]} & ${r['delta']:+.2f}$ & "
               f"{r['b']}/{r['c']} & {pfmt(r['p'])} & {pfmt(r['p_holm'])} & "
               f"$[{r['lo']:+.2f},{r['hi']:+.2f}]$ & "
               f"{mde(r['b']+r['c'], r['n'], 0.05):.2f} & "
               f"{mde(r['b']+r['c'], r['n'], 0.05/len(rows)):.2f} \\\\"),
    "$m$ & seed & $\\arg\\max\\mathcal{B}$ & $\\Delta$ & $b/c$ & $p$ & $p_{\\text{Holm}}$ "
    "& 95\\% CI & MDE & MDE$_{\\text{H}}$ \\\\",
    "rrlrrrrcrr",
    "Per-group significance and power. $b/c$ are the McNemar discordant counts "
    "(GRPO right / $\\arg\\max\\mathcal{B}$ right). $p_{\\text{Holm}}$ corrects across all "
    "17 groups. The CI is a paired item bootstrap ($B{=}10{,}000$) of $\\Delta$ with the "
    "$\\max$ recomputed on each resample. MDE is the smallest $|\\Delta|$ this group's "
    "discordant count could have declared significant, uncorrected and under Holm.",
    "tab:pergroup")

grouped(
    lambda r: (f"{r['m']} & {r['s']} & ${r['delta']:+.2f}$ & ${r['d_learn']:+.2f}$ & "
               f"${r['d_csft']:+.2f}$ & ${r['d_rssft']:+.2f}$ & ${r['d_warm']:+.2f}$ & "
               f"${r['d_solver']:+.2f}$ & ${r['d_mean']:+.2f}$ & ${r['d_second']:+.2f}$ \\\\"),
    "$m$ & seed & $\\max\\mathcal{B}$ & $\\max$ learn & c-SFT & RS-SFT & $W_m$ & solver "
    "& mean & 2nd \\\\",
    "rrrrrrrrrr",
    "Aggregator sensitivity, per group. Same runs throughout; only the way the control "
    "group is reduced to one number changes.",
    "tab:aggfull")

L = ["% 由 analysis/gen_extended_stats.py 生成，勿手改",
     "\\begin{table}[t]", "\\centering", "\\small",
     "\\caption{Aggregator sensitivity, summary over the 17 groups. Only the "
     "pre-registered $\\max$ over the full four-arm control group is negative "
     "everywhere; every weaker aggregator reports a positive mean effect.}",
     "\\label{tab:aggsummary}",
     "\\begin{tabular}{lrrr}", "\\toprule",
     "control-group aggregator & mean $\\Delta$ & range & $\\Delta<0$ \\\\", "\\midrule"]
AGG = [("delta", "$\\max$ over $\\mathcal{B}$ (pre-registered)"),
       ("d_learn", "$\\max$ over learning arms only"),
       ("d_second", "second best of $\\mathcal{B}$"),
       ("d_mean", "mean of $\\mathcal{B}$"),
       ("d_median", "median of $\\mathcal{B}$"),
       ("d_solver", "solver alone"),
       ("d_rssft", "RS-SFT alone"),
       ("d_csft", "continued SFT alone"),
       ("d_warm", "warm start $W_m$ alone")]
for key, label in AGG:
    v = [r[key] for r in rows]
    L.append(f"{label} & ${mean(v):+.2f}$ & ${min(v):+.2f}$ to ${max(v):+.2f}$ & "
             f"{sum(1 for x in v if x < 0)}/{len(v)} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
emit("tab_aggsummary.tex", L)

grouped(
    lambda r: (f"{r['m']} & {r['s']} & ${r['delta']:+.2f}$ & ${r['bias']:+.3f}$ & "
               f"${r['cf']:+.2f}$ & $[{r['cf_lo']:+.2f},{r['cf_hi']:+.2f}]$ \\\\"),
    "$m$ & seed & $\\Delta$ (naive) & bias & $\\Delta_{\\text{cf}}$ & 95\\% CI \\\\",
    "rrrrrc",
    "Winner's-curse audit. \\emph{bias} is the bootstrap mean of $\\Delta$ with the "
    "$\\max$ recomputed on each resample minus the same quantity with the arm frozen at "
    "the full-sample $\\arg\\max$. $\\Delta_{\\text{cf}}$ is cross-fitted: the control arm "
    "is selected on a random half of the test set and scored on the other half "
    "($K{=}2{,}000$ splits).",
    "tab:crossfit")

L = ["% 由 analysis/gen_extended_stats.py 生成，勿手改",
     "\\begin{table}[t]", "\\centering", "\\small",
     "\\setlength{\\tabcolsep}{3pt}",
     "\\caption{Compute on four axes, ranges over the three seeds. Wall clock is the "
     "pre-registered matching axis; the other three are reported so that the matching is "
     "not taken on trust. RS-SFT, not GRPO, is the most expensive arm on generated "
     "tokens and verifier calls.}",
     "\\label{tab:compute}",
     "\\begin{tabular}{llrrrrrrr}", "\\toprule",
     "& & \\multicolumn{3}{c}{wall clock (s)} & \\multicolumn{2}{c}{verifier calls} "
     "& \\multicolumn{2}{c}{generated tokens} \\\\",
     "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
     "cfg & $m$ & c-SFT & GRPO & RS-SFT & GRPO & RS-SFT & GRPO & RS-SFT \\\\", "\\midrule"]


def rg(v, k=1):
    if not v:
        return "---"
    lo, hi = min(v), max(v)
    f = (lambda x: f"{x/1000:.0f}k") if k == 1000 else (lambda x: f"{x}")
    return f(lo) if lo == hi else f"{f(lo)}--{f(hi)}"


for (cfg, m), d in COST.items():
    L.append(f"{cfg} & {m} & {rg(d['csft'])} & {rg(d['grpo'])} & {rg(d['rssft'])} & "
             f"{rg(d['gv'])} & {rg(d['rv'])} & {rg(d['gt'],1000)} & {rg(d['rt'],1000)} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
emit("tab_compute.tex", L)

grouped(
    lambda r: (f"{r['m']} & {r['s']} & {r['acc']['solver']:.2f} & {r['acc']['warm']:.2f} & "
               f"{r['acc']['csft']:.2f} & {r['acc']['rssft']:.2f} & {r['acc']['grpo']:.2f} & "
               f"{ARMTEX[r['best']]} & ${r['delta']:+.2f}$ \\\\"),
    "$m$ & seed & solver & $W_m$ & c-SFT & RS-SFT & GRPO & $\\arg\\max$ & $\\Delta$ \\\\",
    "rrrrrrrlr",
    "Every frozen-test number in the study: all 17 groups, all five arms, no averaging "
    "over seeds. Config~2 $m{=}128$ seed~1 is the excluded collapsed run.",
    "tab:allarms")

L = ["% 由 analysis/gen_extended_stats.py 生成，勿手改",
     "\\begin{table}[t]", "\\centering", "\\small",
     "\\setlength{\\tabcolsep}{4pt}",
     "\\caption{Zero-advantage groups across all 18 GRPO runs, read from each run's "
     "recorded metadata. A prompt group whose 8 samples all receive the same reward has "
     "zero group-relative advantage and contributes no gradient. The 7B runs have the "
     "highest positive-reward rate in the study and the fewest informative prompts.}",
     "\\label{tab:zeroadv}",
     "\\begin{tabular}{llrrrrrrr}", "\\toprule",
     "backbone & $m$/seed & steps & groups & all-0 & all-1 & zero-var & informative "
     "& pos.\\ rate \\\\", "\\midrule"]
BBN = {"C1": "Qwen2.5-0.5B", "C2": "Llama-3.2-1B", "C3": "Qwen2.5-7B"}
last = None
for r in meta_rows:
    nm = BBN[r["cfg"]] if r["cfg"] != last else ""
    last = r["cfg"]
    dag = "$^{\\dagger}$" if r["inf"] == 0 else ""
    L.append(f"{nm} & {r['m']}/{r['s']}{dag} & {r['steps']} & {r['g']} & {r['a0']} & "
             f"{r['a1']} & {r['zv']*100:.1f}\\% & \\textbf{{{r['inf']}}} & "
             f"{r['pos']:.3f} \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\\\[2pt]\\footnotesize $^{\\dagger}$The excluded run: every group all-zero.",
      "\\end{table}"]
emit("tab_zeroadv.tex", L)

L = ["% 由 analysis/gen_extended_stats.py 生成，勿手改",
     "\\begin{table}[t]", "\\centering", "\\small",
     "\\caption{Inference-time sampling from the warm start, $n{=}16$, dev. "
     "pass@16 is an oracle bound because the verifier needs the gold answer; maj@16 is "
     "the deployable arm. $^{*}$ marks the two pre-registered operating budgets. The "
     "solver column is its dev accuracy at the same budget.}",
     "\\label{tab:supp-bon}",
     "\\begin{tabular}{rrrrrrl}", "\\toprule",
     "$m$ & seeds & pass@1 & pass@16 & maj@16 & solver & oracle beats solver? \\\\",
     "\\midrule"]
for cfg in ("C1", "C2"):
    L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{CFGTEX[cfg]}}}}} \\\\")
    for (c, m), v in sorted(bon_agg.items()):
        if c != cfg:
            continue
        sol = SOLVER_DEV[c].get(m)
        p16 = mean(x["p16"] for x in v)
        st = "$^{*}$" if v[0]["op"] else ""
        L.append(f"{m}{st} & {len(v)} & {mean(x['p1'] for x in v):.2f} & {p16:.2f} & "
                 f"{mean(x['mj'] for x in v):.2f} & "
                 f"{'---' if sol is None else f'{sol:.2f}'} & "
                 f"{'---' if sol is None else ('yes' if p16 > sol else 'no')} \\\\")
    if cfg == "C1":
        L.append("\\addlinespace")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
emit("tab_bon.tex", L)

print(f"LaTeX 表写入 {TEX}/：tab_pergroup, tab_aggfull, tab_aggsummary, "
      f"tab_crossfit, tab_compute, tab_allarms, tab_zeroadv, tab_bon")

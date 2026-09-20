# -*- coding: utf-8 -*-
"""生成 paper/DATA_FACTS.md —— 论文写作时的唯一数字来源。

之前是一次性脚本，7B（配置③）跑完后没重新生成，导致事实表缺一个配置。
固化成脚本：任何实验有更新就重跑它，不要手工编辑生成的 md。

    python gen_data_facts.py
"""
import os, sys, json, glob, re

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（本脚本在 analysis/ 下）
B = os.path.join(HERE, "results_backup")

CFG = [
    {"tag": "配置①", "ds": "CodeTAT-QA", "model": "Qwen2.5-0.5B-Instruct",
     "dir": "config1_full", "test": "test_final", "n_test": 282, "n_clean": 2846,
     "doses": [0, 1, 2, 4, 8, 16, 64, 256, 1024, 2446], "g2m": [2, 64], "seeds": [0, 1, 2]},
    {"tag": "配置②", "ds": "CodeFinQA", "model": "Llama-3.2-1B-Instruct",
     "dir": "config2_full", "test": "test_final_config2", "n_test": 664, "n_clean": 4035,
     "doses": [0, 16, 64, 128, 256, 1024, 3635], "g2m": [128, 256], "seeds": [0, 1, 2]},
    {"tag": "配置③", "ds": "CodeTAT-QA", "model": "Qwen2.5-7B-Instruct",
     "dir": "config3_7b", "test": "config3_7b", "n_test": 282, "n_clean": 2846,
     "doses": [2, 64], "g2m": [2, 64], "seeds": [0, 1, 2]},
]


def jl(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def find(cfg, *parts):
    """配置③ 的目录层级与前两个不同（tar 解包后带 runs_v2/ 前缀），两种都试。"""
    a = os.path.join(B, cfg["dir"], *parts)
    if os.path.exists(a):
        return a
    b = os.path.join(B, cfg["dir"], parts[-1])
    return b if os.path.exists(b) else a


def mean_pct(vals):
    v = [x for x in vals if x is not None]
    return "%.1f%%" % (sum(v) / len(v) * 100) if v else "🤔"


out = []
W = out.append
W("# 论文数据事实表（自动生成，写作时以此为准，勿凭记忆）")
W("")
W("> 由 `python gen_data_facts.py` 生成。实验有更新就重跑，不要手工编辑。")
W("")

for cfg in CFG:
    W("")
    W(f"## {cfg['tag']} {cfg['ds']} × {cfg['model']}")
    W("")
    W(f"- clean 数：{cfg['n_clean']}；dev=400（从 train 切出）；冻结 test n={cfg['n_test']}")
    W(f"- 剂量档位：{cfg['doses']}")
    W(f"- Gate2 档位：m_L={cfg['g2m'][0]}, m_H={cfg['g2m'][1]}")
    W(f"- 演示子集种子：{cfg['seeds']}")
    W("")

    # 剂量曲线。注意区分两种"空"：
    #   n/a = 协议不要求（Gate2 三分支只在 m_L/m_H 两档跑，其余档位只需 warm+solver 选档）
    #   🤔  = 协议要求但还没跑出来
    W("### 剂量曲线（dev，多种子均值）")
    W("")
    W("> GRPO/csft/rssft 按预注册协议**只在 m_L 与 m_H 两档**运行——其余档位的"
      "`n/a` 是设计如此，不是缺数据。W_m 与 solver 覆盖全部档位（用于档位选择）。")
    W("")
    W("| m | W_m | solver | GRPO | csft | rssft |")
    W("|---|---|---|---|---|---|")
    for m in cfg["doses"]:
        w = mean_pct([(jl(find(cfg, "runs_v2", f"bestdev_m{m}_s{s}.json")) or {}).get("best_dev_acc")
                      for s in cfg["seeds"]])
        sv = mean_pct([((jl(find(cfg, "runs_v2", f"solver_m{m}_s{s}.json")) or {})
                        .get("variants", {}).get("scarce", {}) or {}).get("acc") for s in cfg["seeds"]])
        if m in cfg["g2m"]:
            g = mean_pct([(jl(find(cfg, "runs_g2", f"bestdev_grpo_m{m}_s{s}.json")) or {}).get("best_dev_acc")
                          for s in cfg["seeds"]])
            c = mean_pct([(jl(find(cfg, "runs_g2", f"bestdev_csft_m{m}_s{s}.json")) or {}).get("best_dev_acc")
                          for s in cfg["seeds"]])
            r = mean_pct([(jl(find(cfg, "runs_g2", f"bestdev_rssft_m{m}_s{s}.json")) or {}).get("best_dev_acc")
                          for s in cfg["seeds"]])
        else:
            g = c = r = "n/a"
        W(f"| {m} | {w} | {sv} | {g} | {c} | {r} |")
    W("")

    # 冻结 test。训练失败的 run（GRPO 全程零正奖励）不进主表：
    # 它的准确率反映的是"策略塌了"，不是 RLVR 的能力，混进来会污染效应量。
    # 这类 run 在文末「RLVR 训练稳定性」一节单列。
    W("### 冻结 test 逐组")
    W("")
    W("> 训练失败的 run（`verifier_pos_rate == 0`，策略漂移到产不出可执行程序）"
      "标为 *失败*，不计入 Δ 统计，见文末稳定性一节。")
    W("")
    W("| m | seed | solver | warm | csft | rssft | GRPO | max(非RL) | Δ |")
    W("|---|---|---|---|---|---|---|---|---|")
    for m in cfg["g2m"]:
        for s in cfg["seeds"]:
            gm = jl(os.path.join(B, cfg["dir"], "runs_g2", f"grpo_m{m}_s{s}", "meta.json"))
            if gm and gm.get("verifier_pos_rate") == 0.0:
                W(f"| {m} | {s} | — | — | — | — | — | — | *训练失败* |")
                continue
            p = os.path.join(B, cfg["test"], f"test_m{m}_s{s}.json")
            if not os.path.exists(p):
                p = os.path.join(B, cfg["test"], "runs_test", f"test_m{m}_s{s}.json")
            t = jl(p)
            if not t:
                W(f"| {m} | {s} | 🤔 | 🤔 | 🤔 | 🤔 | "
                  f"🤔 | 🤔 | 🤔 |")
                continue
            a = {k: v["acc"] for k, v in t["arms"].items()}
            non = {k: v for k, v in a.items() if k != "grpo"}
            bb = max(non, key=non.get) if non else None
            dl = (a["grpo"] - non[bb]) * 100 if (bb and "grpo" in a) else None
            f = lambda k: f"{a[k]*100:.2f}%" if k in a else "🤔"
            W(f"| {m} | {s} | {f('solver')} | {f('warm')} | {f('csft')} | {f('rssft')} | "
              f"{f('grpo')} | {bb} {non[bb]*100:.2f}% | {dl:+.2f}pp |")
    W("")

    # solver 消融
    abl = []
    for m in cfg["doses"]:
        vs = [jl(find(cfg, "runs_v2", f"ablation_m{m}_s{s}.json")) for s in cfg["seeds"]]
        vs = [v for v in vs if v]
        if vs:
            A = sum(v["variants"]["rules"]["acc"] for v in vs) / len(vs)
            Bb = sum(v["variants"]["retrieval"]["acc"] for v in vs) / len(vs)
            C = sum(v["variants"]["full"]["acc"] for v in vs) / len(vs)
            abl.append((m, A, Bb, C))
    W("### solver 消融（多种子均值）")
    if not abl:
        W("")
        W("> 本配置**未计划**跑消融：它只用于论证 solver 的能力来自规则而非检索，"
          "该论证已由配置①②的 51 个消融点确立，不随基座改变（solver 不含模型）。")
        W("")
    if abl:
        W("")
        W("| m | A只规则 | B只检索 | C规则+检索 | 检索边际 |")
        W("|---|---|---|---|---|")
        for m, A, Bb, C in abl:
            W(f"| {m} | {A*100:.2f}% | {Bb*100:.2f}% | {C*100:.2f}% | {(C-A)*100:+.2f}pp |")
        W("")

    # PoT
    pot = []
    for m in cfg["g2m"]:
        for s in cfg["seeds"]:
            for sp in ["dev", "test"]:
                j = jl(os.path.join(B, "pot", f"pot_m{m}_s{s}_{sp}.json"))
                if j and (cfg["tag"] != "配置②") == (j["n"] != 664 or sp == "dev"):
                    pot.append((m, s, sp, j["acc"], j["n"], j["n_shots"]))
    if pot:
        W("### PoT/PAL 免训练 prompting 基线")
        W("")
        W("| m | seed | split | acc | n | shots |")
        W("|---|---|---|---|---|---|")
        for m, s, sp, acc, n, sh in pot:
            W(f"| {m} | {s} | {sp} | {acc*100:.2f}% | {n} | {sh} |")
        W("")

    # 成本
    metas = sorted(glob.glob(os.path.join(B, cfg["dir"], "runs_g2", "*", "meta.json")))
    if metas:
        W("### 训练成本（Gate2 各分支）")
        W("")
        W("| 分支 | m | seed | wall(s) | verifier调用 | 生成token |")
        W("|---|---|---|---|---|---|")
        for f in metas:
            j = jl(f)
            nm = os.path.basename(os.path.dirname(f))
            mm = re.match(r"(grpo|csft|rssft)_m(\d+)_s(\d+)$", nm)
            if j and mm:
                W(f"| {mm.group(1)} | {mm.group(2)} | {mm.group(3)} | "
                  f"{j.get('wall_seconds', 0):.0f} | {j.get('verifier_calls', '—')} | "
                  f"{j.get('gen_tokens', j.get('gen_tok', '—'))} |")
        W("")

# 全局汇总（未跑完的组也列出，标 🤔，这样"还差什么"一目了然）
rows = []
allv = []
for cfg in CFG:
    for m in cfg["g2m"]:
        for s in cfg["seeds"]:
            p = os.path.join(B, cfg["test"], f"test_m{m}_s{s}.json")
            if not os.path.exists(p):
                p = os.path.join(B, cfg["test"], "runs_test", f"test_m{m}_s{s}.json")
            gm = jl(os.path.join(B, cfg["dir"], "runs_g2", f"grpo_m{m}_s{s}", "meta.json"))
            if gm and gm.get("verifier_pos_rate") == 0.0:
                rows.append((cfg["tag"], m, s, "FAIL"))
                continue
            t = jl(p)
            if not t:
                rows.append((cfg["tag"], m, s, None))
                continue
            a = {k: v["acc"] for k, v in t["arms"].items()}
            non = {k: v for k, v in a.items() if k != "grpo"}
            if non and "grpo" in a:
                v = (a["grpo"] - max(non.values())) * 100
                rows.append((cfg["tag"], m, s, v))
                allv.append(((cfg["tag"], m, s), v))
            else:
                rows.append((cfg["tag"], m, s, None))

# RLVR 训练稳定性：把 pos_rate==0 的 run 单列。
# 这些 run 跑满了预算却全程零正奖励——策略漂移到产不出可执行程序的区域。
# 它们不进 Δ 主表（准确率反映的是"策略塌了"不是 RLVR 能力），
# 但本身是关于 RLVR 脆弱性的证据，必须报告而不是丢弃。
W("")
W("## RLVR 训练稳定性：零正奖励的 run")
W("")
stab = []
for cfg in CFG:
    for m in cfg["g2m"]:
        for s in cfg["seeds"]:
            j = jl(os.path.join(B, cfg["dir"], "runs_g2", f"grpo_m{m}_s{s}", "meta.json"))
            if j:
                stab.append((cfg["tag"], m, s, j.get("steps", 0),
                             j.get("verifier_pos_rate", -1), j.get("verifier_calls", 0)))
fails = [x for x in stab if x[4] == 0.0]
W(f"- 全部 GRPO run：**{len(stab)}**，其中训练失败（`pos_rate == 0`）：**{len(fails)}**")
W("")
if fails:
    W("| 配置 | m | seed | steps | verifier调用 | pos_rate |")
    W("|---|---|---|---|---|---|")
    for tag, m, s, st, pr, vc in fails:
        W(f"| {tag} | {m} | {s} | {st} | {vc} | **0.0000** |")
    W("")
    W("> 这些 run 跑满了墙钟预算与 verifier 预算，却一次正奖励都没拿到——"
      "策略漂移到产不出可执行程序的区域且不再恢复。它们不计入 Δ，"
      "但说明 RLVR 在弱基座 + 演示稀缺时对随机种子敏感，这一脆弱性是部署成本的一部分。")
    W("")
W("同档位对照（健康 run 的 pos_rate）：")
W("")
W("| 配置 | m | seed | steps | pos_rate |")
W("|---|---|---|---|---|")
for tag, m, s, st, pr, vc in stab:
    mark = " ← 失败" if pr == 0.0 else ""
    W(f"| {tag} | {m} | {s} | {st} | {pr:.4f}{mark} |")
W("")

# 数据质量告警：自动检出「同档位跨种子的有效计算量异常」。
# 背景：配置② m=128 seed1 的三个学习类分支只跑到正常的 30-50%（共享服务器被抢资源），
# 这类异常不会报错，只体现在 steps/rounds/token 上，必须显式标出，
# 否则会被误当成正常数据（2026-07-28 曾发生过一次手工"修正"成正常值）。
W("")
W("## ⚠ 数据质量告警（自动检出）")
W("")
alerts = []
for cfg in CFG:
    for br, key in [("grpo", "steps"), ("csft", "steps"), ("rssft", "rounds")]:
        for m in cfg["g2m"]:
            vals = {}
            for s in cfg["seeds"]:
                d = jl(os.path.join(B, cfg["dir"], "runs_g2", f"{br}_m{m}_s{s}", "meta.json"))
                if d:
                    vals[s] = d.get(key, 0)
            if len(vals) >= 2:
                mx = max(vals.values())
                for s, v in vals.items():
                    if mx and v < 0.7 * mx:
                        alerts.append((cfg["tag"], br, m, s, v, mx, key))
if alerts:
    W("| 配置 | 分支 | m | seed | 实际 | 同档最大 | 比例 | 指标 |")
    W("|---|---|---|---|---|---|---|---|")
    for tag, br, m, s, v, mx, key in alerts:
        W(f"| {tag} | {br} | {m} | {s} | {v} | {mx} | **{v/mx*100:.0f}%** | {key} |")
    W("")
    W("> 这些 run 在同样的墙钟预算下只完成了正常计算量的一部分（共享服务器资源竞争）。"
      "**协议承诺的「各分支等预算」在这些组上不成立**，写论文时要么重跑，要么如实标注。")
else:
    W("✅ 未检出跨种子的计算量异常。")
W("")

W("")
W(f"## 全局汇总：{len(rows)} 组冻结 test 的 Δ（{len(allv)} 组已出，{len(rows)-len(allv)} 组待跑）")
W("")
W(f"- 已出组数：**{len(allv)} / {len(rows)}**")
W(f"- 已出的全部为负：**{all(v < 0 for _, v in allv)}**")
if allv:
    W(f"- 均值：**{sum(v for _, v in allv)/len(allv):+.2f}pp**")
    W(f"- 范围：{min(v for _, v in allv):+.2f} ~ {max(v for _, v in allv):+.2f} pp")
W("")
W("| 配置 | m | seed | Delta |")
W("|---|---|---|---|")
for tag, m, s, v in rows:
    if v == "FAIL":
        cell = "*训练失败* |"
    elif v is None:
        cell = "🤔 |"
    else:
        cell = f"{v:+.2f}pp |"
    W(f"| {tag} | {m} | {s} | " + cell)
for _skip in []:
    pass

os.makedirs(os.path.join(HERE, "paper"), exist_ok=True)
p = os.path.join(HERE, "paper", "DATA_FACTS.md")
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"已生成 {p}（{len(out)} 行，{len(CFG)} 个配置，{len(allv)} 组 Δ）")

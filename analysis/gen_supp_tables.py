# -*- coding: utf-8 -*-
"""从 results_backup/ 直算 ICASSP 补充材料的四张表，写入
论文tex_ICASSP2027/supplement/tab_{ablation,pot,hacking,stability}.tex。

内置自检：重算值必须复现论文正文已写的数字，不一致直接抛异常。
路径锚定项目根，在任何目录下都能跑：

    python analysis/gen_supp_tables.py
"""
import json, os, glob, sys, io
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(HERE, "results_backup")
OUT = os.path.join(HERE, "论文tex_ICASSP2027", "supplement")

CFG_ABL = [("1", "config1_full", [0, 1, 2, 4, 8, 16, 64, 256, 1024, 2446]),
           ("2", "config2_full", [0, 16, 64, 128, 256, 1024, 3635])]
POT_SRC = [("1", [(2, [0, 1, 2]), (64, [0, 1, 2])]),
           ("2", [(128, [0, 1, 2]), (256, [0, 1, 2])])]
HACK = [("1", "config1_full"), ("2", "config2_full"), ("3", "config3_7b")]
POS = [("Qwen2.5-0.5B", "config1_full", [(2, [0, 1, 2]), (64, [0, 1, 2])]),
       ("Llama-3.2-1B", "config2_full", [(128, [0, 1, 2]), (256, [0, 1, 2])]),
       ("Qwen2.5-7B", "config3_7b", [(2, [0, 1, 2]), (64, [0, 1, 2])])]


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def mean(v):
    return sum(v) / len(v)


def grp(n):
    return f"{n:,}".replace(",", "{,}")


abl = {}
for cfg, d, budgets in CFG_ABL:
    for m in budgets:
        rows = [jl(p)["variants"] for p in
                sorted(glob.glob(os.path.join(B, d, "runs_v2",
                                              f"ablation_m{m}_s*.json")))]
        if not rows:
            continue
        a = mean([r["rules"]["acc"] for r in rows]) * 100
        b = mean([r["retrieval"]["acc"] for r in rows]) * 100
        c = mean([r["full"]["acc"] for r in rows]) * 100
        abl[(cfg, m)] = (round(a, 2), round(b, 2), round(c, 2), round(c - a, 2))

for m in [0, 1, 2, 4, 8, 16, 64, 256, 1024, 2446]:
    assert abl[("1", m)] == (72.0, 0.0, 72.0, 0.0), f"C1 m={m}: {abl[('1', m)]}"
assert abl[("2", 128)] == (36.0, 0.0, 35.33, -0.67), abl[("2", 128)]
assert abl[("2", 256)] == (35.67, 0.0, 35.33, -0.33), abl[("2", 256)]
print("自检 消融   ✓ C1 十档全 72.00/0.00/72.00/+0.00；C2 m128 -0.67、m256 -0.33")

pot = {}
for cfg, budgets in POT_SRC:
    for m, seeds in budgets:
        for s in seeds:
            for sp in ["dev", "test"]:
                p = os.path.join(B, "pot", f"pot_m{m}_s{s}_{sp}.json")
                if os.path.exists(p):
                    j = jl(p)
                    pot[(cfg, m, s, sp)] = (round(j["acc"] * 100, 2), j["n"])

c1t = [v[0] for k, v in pot.items() if k[0] == "1" and k[3] == "test"]
c2t = [v[0] for k, v in pot.items() if k[0] == "2" and k[3] == "test"]
assert (min(c1t), max(c1t)) == (9.93, 16.67) and round(mean(c1t), 2) == 13.30
assert (min(c2t), max(c2t)) == (0.75, 1.36) and round(mean(c2t), 2) == 1.00
print("自检 PoT    ✓ C1 test 9.93--16.67 均值 13.30；C2 test 0.75--1.36 均值 1.00")

hack, tot, ginis = {}, defaultdict(int), []
for cfg, d in HACK:
    j = jl(os.path.join(B, d, "hacking_audit.json"))
    v = (len(j),
         sum(x["n_samples"] for x in j.values()),
         sum(x["n_positive"] for x in j.values()),
         sum(x["flags_positive"].get("H1_constant", 0) for x in j.values()),
         sum(x["flags_positive"].get("H2_gold_literal", 0) for x in j.values()),
         sum(x["flags_positive"].get("H4_trivial_op", 0) for x in j.values()),
         max(x["h3_top_skeleton_ratio"] for x in j.values()))
    hack[cfg] = v
    ginis += [x["gini_skeleton"] for x in j.values()]
    for k, n in zip(["runs", "ns", "npos", "h1", "h2", "h4"], v[:6]):
        tot[k] += n

assert (tot["runs"], tot["ns"], tot["npos"], tot["h1"], tot["h4"]) == \
       (14, 130477, 57517, 0, 27), dict(tot)
assert round(tot["h2"] / tot["npos"] * 100, 2) == 0.19
assert (round(min(ginis), 2), round(max(ginis), 2)) == (0.06, 0.45)
print(f"自检 审计   ✓ 14 run / {grp(tot['ns'])} 样本 / {grp(tot['npos'])} 正样本 / "
      f"H1=0 / H4=27 / H2=0.19%")

pos = {}
for name, d, budgets in POS:
    for m, seeds in budgets:
        for s in seeds:
            p = os.path.join(B, d, "runs_g2", f"grpo_m{m}_s{s}", "meta.json")
            if os.path.exists(p):
                j = jl(p)
                pos[(name, m, s)] = (j["verifier_pos_rate"], j["steps"],
                                     j["verifier_calls"])
assert len(pos) == 18 and sum(1 for v in pos.values() if v[0] == 0.0) == 1
print("自检 正奖励 ✓ 18 个 GRPO run，其中 1 个为 0\n")

L = []
W = L.append

W(r"\begin{table}[t]\centering\small")
W(r"\setlength{\tabcolsep}{4pt}")
W(r"\caption{Complete solver ablation. \textbf{A} uses only the hand-written rule")
W(r"path, \textbf{B} only retrieval, \textbf{C} is the deployed configuration")
W(r"(rules first, retrieval as fallback). Dev accuracy (\%), mean over seeds, at")
W(r"every budget of each configuration. Retrieval never fires at any budget used")
W(r"for a reported group.}")
W(r"\label{tab:supp-ablation}")
W(r"\begin{tabular}{cr rrr r}")
W(r"\toprule")
W(r"cfg & $m$ & A: rules & B: retr. & C: deployed & $C-A$ \\")
W(r"\midrule")
last = None
for (cfg, m), (a, b, c, dd) in sorted(abl.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    W(r"%s & %d & %.2f & %.2f & %.2f & $%+.2f$ \\"
      % (cfg if cfg != last else "", m, a, b, c, dd))
    last = cfg
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")

W(r"\begin{table}[t]\centering\small")
W(r"\setlength{\tabcolsep}{4pt}")
W(r"\caption{Program-of-Thought / PAL prompting of the base model, with the same")
W(r"$m$ demonstrations as few-shot examples and no gradient steps. Accuracy (\%)")
W(r"per seed on both splits. This is the weakest arm in the entire study.}")
W(r"\label{tab:supp-pot}")
W(r"\begin{tabular}{cr r rr rr}")
W(r"\toprule")
W(r"cfg & $m$ & seed & dev & $n$ & test & $n$ \\")
W(r"\midrule")
last = None
for cfg, budgets in POT_SRC:
    for m, seeds in budgets:
        for s in seeds:
            d_, t_ = pot.get((cfg, m, s, "dev")), pot.get((cfg, m, s, "test"))
            if not t_:
                continue
            dv = f"{d_[0]:.2f} & {d_[1]}" if d_ else "--- & ---"
            W(r"%s & %d & %d & %s & %.2f & %d \\"
              % (cfg if cfg != last else "", m, s, dv, t_[0], t_[1]))
            last = cfg
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")

W(r"\begin{table}[t]\centering\small")
W(r"\setlength{\tabcolsep}{3.5pt}")
W(r"\caption{Verifier-hacking audit over the 14 GRPO runs whose reward dumps were")
W(r"retained. H1: constant program; H2: gold value appearing as a literal; H3:")
W(r"share of positive samples under the single most frequent code skeleton; H4:")
W(r"no-op arithmetic. Counts are over positive-reward samples.}")
W(r"\label{tab:supp-hacking}")
W(r"\begin{tabular}{c rrr rrr r}")
W(r"\toprule")
W(r"cfg & runs & samples & positive & H1 & H2 & H4 & H3 \\")
W(r"\midrule")
for cfg in ["1", "2", "3"]:
    r, ns, npos, h1, h2, h4, h3 = hack[cfg]
    W(r"%s & %d & %s & %s & %d & %d & %d & $\leq %.1f\%%$ \\"
      % (cfg, r, grp(ns), grp(npos), h1, h2, h4, h3 * 100))
W(r"\midrule")
W(r"total & %d & %s & %s & \textbf{0} & %d & %d & $\leq %.1f\%%$ \\"
  % (tot["runs"], grp(tot["ns"]), grp(tot["npos"]), tot["h2"], tot["h4"],
     max(hack[c][6] for c in hack) * 100))
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")

W(r"\begin{table}[t]\centering\small")
W(r"\setlength{\tabcolsep}{4pt}")
W(r"\caption{Positive-reward rate across all 18 GRPO runs, with optimizer steps")
W(r"and verifier calls. One run---the weakest backbone at the scarcest")
W(r"budget---ends at zero and is the single excluded group.}")
W(r"\label{tab:supp-stability}")
W(r"\begin{tabular}{l rr rrr}")
W(r"\toprule")
W(r"backbone & $m$ & seed & steps & verifier calls & pos.\ rate \\")
W(r"\midrule")
last = None
for name, d, budgets in POS:
    for m, seeds in budgets:
        for s in seeds:
            v = pos.get((name, m, s))
            if not v:
                continue
            W(r"%s & %d & %d%s & %d & %s & %.4f \\"
              % (name if name != last else "", m, s,
                 r"$^{\dagger}$" if v[0] == 0.0 else "", v[1], grp(v[2]), v[0]))
            last = name
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")

os.makedirs(OUT, exist_ok=True)
blocks = [b.strip() + "\n\\end{table}\n"
          for b in "\n".join(L).split("\\end{table}") if b.strip()]
names = ["tab_ablation", "tab_pot", "tab_hacking", "tab_stability"]
assert len(blocks) == len(names), f"切出 {len(blocks)} 块，应为 {len(names)}"
for name, blk in zip(names, blocks):
    io.open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8",
            newline="").write(blk)
    print(f"-> supplement/{name}.tex")
print(f"\n   消融 {len(abl)} 档 / PoT {len(c1t) + len(c2t)} 组 / 审计 3+1 行 / "
      f"稳定性 {len(pos)} run")

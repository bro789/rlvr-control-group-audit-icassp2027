# -*- coding: utf-8 -*-
"""Gate 2 判决：按 PREREG_v2_DRAFT.md §5 裁决 Δ(m)。

    Δ(m) = Acc(GRPO) − max{ Acc(W_m), Acc(continued SFT), Acc(RS-SFT),
                            Acc(solver), Acc(best-of-n + verifier) }

最后一项是 §9.2 新增（Gate 1 已测得），不加它会高估 RL 的独立贡献。
"""
import os, sys, json, math, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GO_PP = 0.03          # Go 判据：GRPO 领先最强非 RL 方法 ≥3pp


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def main(m, seed):
    g1 = load(os.path.join(HERE, "runs_v2", f"bestdev_m{m}_s{seed}.json"))
    bon = load(os.path.join(HERE, "runs_v2", f"bon_m{m}_s{seed}.json"))
    br = {b: load(os.path.join(HERE, "runs_g2", f"bestdev_{b}_m{m}_s{seed}.json"))
          for b in ["grpo", "csft", "rssft"]}

    print("=" * 96)
    print(f"Gate 2 判决 —— m={m} seed={seed}")
    print("=" * 96)

    # ---- 产物完整性断言：缺基线一律拒绝裁决（PREREG §3 CANNOT-EVALUATE）----
    # 否则 load() 返回 None 会让 Δ 的 max 集合悄悄塌缩，门槛降低，打印出**假 Go**。
    missing = []
    if not g1:
        missing.append(f"runs_v2/bestdev_m{m}_s{seed}.json（W_m warm start 基线）")
    if not bon:
        missing.append(f"runs_v2/bon_m{m}_s{seed}.json（test-time 选择器基线）")
    if not br["rssft"]:
        missing.append("RS-SFT 分支（PREREG §5 指定的决胜对照，缺它 Go 判据不成立）")
    if not br["csft"]:
        missing.append("Continued SFT 分支")
    if missing:
        print("\n❌ CANNOT-EVALUATE —— 缺少以下基线产物，拒绝裁决：")
        for x in missing:
            print(f"    - {x}")
        print("\n  理由：Δ(m) = GRPO − max{基线集合}。任一基线缺失都会让 max 塌缩、")
        print("        门槛下降，从而打印出**假 Go**。宁可不裁决，不可误裁。")
        sys.exit(1)

    n = g1["dev_n"] if g1 else 200
    rows = []
    if g1:
        rows.append(("W_m（warm start，不继续训练）", g1["best_dev_acc"], g1["best_ckpt"], "非RL"))
    for b, label in [("csft", "Continued SFT"), ("rssft", "Iterative RS-SFT")]:
        if br[b]:
            rows.append((label, br[b]["best_dev_acc"], br[b]["best_point"], "非RL"))
    if bon:
        # 可部署基线：execution-guarded majority voting，无 gold → 进 Δ 的 max 集合
        for k in ["43", "16", "4"]:
            if k in bon.get("majority_at", {}):
                rows.append((f"maj@{k}（执行结果多数投票，无 gold，可部署）",
                             bon["majority_at"][k], f"n={k}", "非RL"))
                break
        # oracle 上界：pass@k ≡ sel@k 需 gold 才能挑，**不可部署，不进 Δ**，仅作参考
        for k in ["43", "16", "4"]:
            if k in bon.get("pass_at", {}):
                rows.append((f"pass@{k}（oracle 上界，需 gold，不可部署）",
                             bon["pass_at"][k], f"n={k}", "上界参考"))
                break
    if br["grpo"]:
        rows.append(("GRPO", br["grpo"]["best_dev_acc"], br["grpo"]["best_point"], "RL"))

    print(f"{'方法':<34}{'dev acc':>10}{'95% CI':>20}  {'best point':<18}类别")
    print("-" * 96)
    for name, acc, pt, kind in rows:
        lo, hi = wilson(round(acc * n), n)
        print(f"{name:<34}{acc*100:>9.2f}%  [{lo*100:5.2f}%,{hi*100:6.2f}%]  {str(pt):<18}{kind}")
    print("-" * 96)

    nonrl = [(nm, a) for nm, a, _, k in rows if k == "非RL"]
    grl = [(nm, a) for nm, a, _, k in rows if k == "RL"]
    if not grl:
        print("⏳ GRPO 尚无评测结果，暂不裁决")
        return
    if not nonrl:
        print("⏳ 非 RL 基线缺失，暂不裁决")
        return

    best_nm, best_acc = max(nonrl, key=lambda x: x[1])
    g_acc = grl[0][1]
    delta = g_acc - best_acc

    print()
    print("【Δ(m) 计算】")
    print(f"  最强非 RL 方法 : {best_nm} = {best_acc*100:.2f}%")
    print(f"  GRPO           : {g_acc*100:.2f}%")
    print(f"  Δ(m)           : {delta*100:+.2f}pp   （Go 门槛 ≥ +{GO_PP*100:.0f}pp）")

    # 训练侧成本
    print()
    print("【等预算核对（PREREG §2）】")
    for b in ["grpo", "csft", "rssft"]:
        if br[b] and br[b].get("train_meta"):
            t = br[b]["train_meta"]
            print(f"  {b:<6} 墙钟={t.get('wall_seconds',0):.0f}s "
                  f"预算={t.get('budget_seconds','?')}s "
                  f"verifier调用={t.get('verifier_calls','?')} "
                  f"步数/轮数={t.get('steps', t.get('rounds','?'))}")

    print()
    print("=" * 96)
    if delta >= GO_PP:
        print(f"判决：✅ Gate 2 (m={m}) 单种子 Go **信号** —— Δ={delta*100:+.2f}pp ≥ +3pp")
        print("      ⚠ 单种子 + dev n=200，检出 3pp 真效应的统计功效仅约 13%。")
        print("      需补齐 3 seed 且 ≥2 个为正、GRPO 须优于 RS-SFT，才构成 Gate 2 Go")
    elif delta <= -0.02:
        print(f"判决：❌ Gate 2 (m={m}) 单种子 No-Go **信号** —— Δ={delta*100:+.2f}pp")
        print("      ⚠ 单种子 + dev n=200（Δ 的 SE≈3.5pp）不足以支撑 Task No-Go 结论。")
        print("      PREREG §5 要求 Δ 的 90% CI 上界 < +3pp，本脚本尚未实现配对 bootstrap CI。")
        print("      当前只能记为 SIGNAL，不得作为结论引用。")
        print("      限定表述：\"CodeFinQA 上 demonstration-scarce RLVR No-Go\"")
        print("      不得写成\"demonstration scarcity 假设被否定\"")
    else:
        print(f"判决：⚠ INCONCLUSIVE —— Δ={delta*100:+.2f}pp 落在 ±2~3pp 灰区，需追加种子")
    print("=" * 96)

    # GRPO 是否优于 RS-SFT（PREREG 特别要求：只赢普通 SFT 不算数）
    if br["rssft"] and br["grpo"]:
        d2 = br["grpo"]["best_dev_acc"] - br["rssft"]["best_dev_acc"]
        print(f"\n【关键对照】GRPO − RS-SFT = {d2*100:+.2f}pp")
        print("  若 GRPO 只胜普通 SFT 却不胜 RS-SFT，只能说明 verifier 生成的数据有价值，")
        print("  不能说明 policy gradient 有独立价值。")

    json.dump({"m": m, "seed": seed, "rows": [(r[0], r[1], str(r[2]), r[3]) for r in rows],
               "best_nonrl": best_nm, "best_nonrl_acc": best_acc,
               "grpo_acc": g_acc, "delta": delta},
              open(os.path.join(HERE, f"GATE2_verdict_m{m}_s{seed}.json"), "w",
                   encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.m, a.seed)

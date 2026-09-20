# -*- coding: utf-8 -*-
"""协议 Δ(m) max 集合中的 Acc(solver) 臂 —— 在 dev n=200 上评测确定性 solver。

冻结协议 §一 把「确定性 solver」列为每个 m 档位的五分支之一，§0 的 Δ(m) 公式
明确含 Acc(solver)。此前 v2 从未评测该臂，gate2_verdict.py / gate4_v2.py 的
max 集合静默漏掉它 —— 遗漏方向系统性利于判 Go（v1 solver 在 test 上 33.4%，
远高于当前 m_L 最强非 RL 臂 15.0%）。本脚本补上。

两个变体：

  scarce  （协议合规臂，进 Δ 的 max 集合）
      检索库 = 该档位的 m 条 gold 演示。依据公平性约束「同一 m 下所有分支看到
      完全相同的 gold 演示」——solver 是 m 档位的分支之一，不能比 csft/rssft/GRPO
      多看 gold。m=0 时检索库为空，退化为纯规则。

  full    （诊断上界，**不得**进 Δ）
      检索库 = 整个 demo pool（3635 条，按构造不含 dev）。这是 v1 那个 33.4%
      的口径。它对应「solver 见过全部演示」，与演示稀缺档位不可比，仅用于
      说明规则库本身的天花板与检索的贡献量。

    python solver_arm.py --m 16 --seed 0            # 两个变体都跑
    python solver_arm.py --m 16 --seed 0 --variant scarce
"""
import os, sys, json, argparse, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from verifier import verify
from gate_v2 import dev_items, CKPT_DEV_N
import solver_v2_combo as S   # 配置②：CodeFinQA 文本表格 solver（非 CodeTAT-QA 的 df solver）


def build_store(rows, idxs, all_flags):
    """用 rows 中下标属于 idxs 的那些题构建检索库（gold 自洽者才入库）。

    注意：绝不能对子集调 S.load_sc_flags —— 它按 len(rows) 校验缓存，长度不符时
    会重算并**覆盖** train_sc_flags.json（写成子集长度），破坏全局缓存。
    这里改为读一次全量 flags，再按原始下标切片。
    """
    sub = [rows[i] for i in idxs]
    if not sub:
        return S.Store([], [])
    return S.Store(sub, [all_flags[i] for i in idxs])


def run_variant(name, store, items, workers=12):
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    progs = [S.build_program(r["question"], r["context"], store) for r in items]

    def check(args):
        (p, kind), r = args
        if p is None:
            return 0, kind, "no_program"
        v = verify(p, r["answer"], True, None)
        return int(v["reward"]), kind, v["reason"]

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(check, zip(progs, items)))

    per_item = [{"reward": float(h), "reason": rs, "kind": k} for h, k, rs in results]
    hits = sum(h for h, _, _ in results)
    acc = hits / len(items)
    stats = Counter()
    for h, k, _ in results:
        stats[f"{k}_{'hit' if h else 'miss'}"] += 1
    print(f"[{name}] acc = {acc*100:.2f}%  ({hits}/{len(items)})   用时 {time.time()-t0:.0f}s")
    kinds = sorted(set(k.rsplit("_", 1)[0] for k in stats))
    for k in kinds:
        h, m = stats.get(f"{k}_hit", 0), stats.get(f"{k}_miss", 0)
        if h + m:
            print(f"        {k:16s} {h:4d}/{h+m:4d} = {h/(h+m):5.1%}")
    return acc, per_item


def main(m, seed, variant, dev_n):
    split = json.load(open(os.path.join(HERE, "data", "v2_split.json"), encoding="utf-8"))
    train_rows = [json.loads(l) for l in
                  open(os.path.join(HERE, "data", "CodeFinQA__train.jsonl"), encoding="utf-8")]
    demo_idx = split["subsets"][str(seed)][str(m)]
    pool_idx = split["pool_idx"]
    items = dev_items(dev_n)
    all_flags = S.load_sc_flags(train_rows)          # 全量，勿传子集（会覆盖缓存）
    # 防御：确认检索库与 dev 无交集（直接泄漏）
    dev_set = set(split["dev_idx"][:dev_n])
    for nm, ix in (("demo", demo_idx), ("pool", pool_idx)):
        ov = dev_set & set(ix)
        if ov:
            print(f"!! {nm} 检索库与 dev 有 {len(ov)} 条重叠 —— 直接泄漏，中止")
            sys.exit(1)
    print("  泄漏检查：demo/pool 检索库与 dev 均无交集 ✓")

    print(f"solver 臂  m={m} seed={seed}  dev_n={len(items)}")
    print(f"  演示子集 {len(demo_idx)} 条；demo pool {len(pool_idx)} 条（按构造不含 dev）")
    print()

    out = {"m": m, "seed": seed, "dev_n": len(items), "variants": {}}

    if variant in ("scarce", "both"):
        store = build_store(train_rows, demo_idx, all_flags)
        acc, pi = run_variant(f"scarce (库={len(demo_idx)} 条演示)", store, items)
        out["variants"]["scarce"] = {"acc": acc, "lib_n": len(demo_idx),
                                     "admissible_in_delta": True}
        json.dump(pi, open(os.path.join(HERE, "runs_v2",
                  f"peritem_solver_scarce_m{m}_s{seed}.json"), "w"), indent=1)

    if variant in ("full", "both"):
        store = build_store(train_rows, pool_idx, all_flags)
        acc, pi = run_variant(f"full   (库={len(pool_idx)} 条 pool)", store, items)
        out["variants"]["full"] = {"acc": acc, "lib_n": len(pool_idx),
                                   "admissible_in_delta": False,
                                   "note": "见过全部演示，与稀缺档位不可比，仅诊断"}
        json.dump(pi, open(os.path.join(HERE, "runs_v2",
                  f"peritem_solver_full_m{m}_s{seed}.json"), "w"), indent=1)

    p = os.path.join(HERE, "runs_v2", f"solver_m{m}_s{seed}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n-> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=["scarce", "full", "both"], default="both")
    ap.add_argument("--dev-n", type=int, default=CKPT_DEV_N)
    a = ap.parse_args()
    main(a.m, a.seed, a.variant, a.dev_n)

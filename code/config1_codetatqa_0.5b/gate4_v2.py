# -*- coding: utf-8 -*-
"""Gate 2 判决 —— 严格按冻结协议实现。

    python gate4_v2.py peritem-warm --m 16 --seed 0   # 先补 W_m 逐题（需 GPU，几分钟）
    python gate4_v2.py report       --m 16 --seed 0   # 出判决（纯 CPU）
    python gate4_v2.py report       --m 16 --seed 0 --seeds 0,1,2   # 多训练种子

协议核心检验量（§0，冻结）：

    Δ(m) = Acc(GRPO | W_m) − max{ Acc(W_m), Acc(continued SFT),
                                  Acc(RS-SFT), Acc(solver) }

判决词表（§六）：CANNOT-EVALUATE / BENCHMARK-NO-GO / TASK-NO-GO /
                 INCONCLUSIVE / PROVISIONAL-GO

本文件的前身在以下各点偏离协议，均已修正（详见 DEVIATIONS_v2.md D16）：

  1. Δ 的 max 集合漏掉 Acc(solver) —— 缩小 max 集合 = 降低门槛 = 系统性利于判 Go。
     实测：补上 solver 后 Δ(m=16) 由 +3.00pp 翻转为 −14.00pp。
  2. 拆除了 gate2_verdict.py:45-62 的完整性护栏，缺臂仍照常裁决（"假 Go"）。
  3. 只有二元 GO/NO-GO，协议的 INCONCLUSIVE 带（2–5pp）缺失 → 灰区抢判。
  4. Go 的四重合取只实现了第 1 条；种子数、hacking/增益集中、GRPO>RS-SFT 三条从未检查。
  5. Δ 无 CI；用两个**独立** Wilson CI 的重叠冒充，丢弃配对相关性，且不是协议要的量。
  6. Task No-Go 的三个条件全部无代码。
  7. ≥3pp 用浮点比较（0.19+0.03 这类边界不可靠）→ 改整数计数。
"""
import os, sys, json, math, argparse, random, ast
from collections import Counter, defaultdict

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GO_PP     = 0.03     # Go：GRPO 领先最强非 RL ≥3pp
INC_LO    = 0.02     # INCONCLUSIVE 带下沿
INC_HI    = 0.05     # INCONCLUSIVE 带上沿
TIE_PP    = 0.02     # Task No-Go 条件2：RS-SFT 与 GRPO 差距 ±2pp 内
N_SEED_REQ = 3       # 协议要求的训练种子数
BOOT_B    = 10000


# ================= 统计 =================
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def chi2_sf_df1(x):
    """chi-square df=1 的**精确**生存函数（X=Z² ⇒ P(X>x)=erfc(√(x/2))），无需 scipy。"""
    return 1.0 if x <= 0 else math.erfc(math.sqrt(x / 2.0))


def mcnemar(a, b):
    assert len(a) == len(b)
    both = only_a = only_b = neither = 0
    for x, y in zip(a, b):
        if x and y:   both += 1
        elif x:       only_a += 1
        elif y:       only_b += 1
        else:         neither += 1
    nd = only_a + only_b
    if nd == 0:
        return dict(both=both, only_a=0, only_b=0, neither=neither,
                    chi2=0.0, p_chi2=1.0, p_exact=1.0, n_disc=0)
    # 连续性校正下限截零：b==c 时统计量应为 0 而非 1/nd
    chi2 = max(0.0, abs(only_a - only_b) - 1) ** 2 / nd
    k = min(only_a, only_b)
    p_exact = min(1.0, 2 * sum(math.comb(nd, i) for i in range(k + 1)) / (2 ** nd))
    return dict(both=both, only_a=only_a, only_b=only_b, neither=neither,
                chi2=chi2, p_chi2=chi2_sf_df1(chi2), p_exact=p_exact, n_disc=nd)


def holm(pvals):
    """Holm-Bonferroni 校正，返回与输入同序的校正后 p。"""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n, out, run = len(pvals), [0.0] * len(pvals), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (n - r) * pvals[i]))
        out[i] = run
    return out


def boot_delta(grpo, arms, B=BOOT_B, seed=20260721):
    """Δ = Acc(GRPO) − max{各非RL臂} 的**逐题配对** bootstrap。

    重采样题目下标（同一 resample 同时作用于所有臂 ⇒ 保留配对相关性）。
    协议 §五要求「先重采样训练种子、再重采样测试题目」的两层 bootstrap；
    单训练种子下种子层无法重采样，此处为**退化版**：只覆盖题目抽样不确定性，
    是真 CI 的下界（偏窄）。多种子时由 boot_delta_2level 接管。
    """
    rng = random.Random(seed)
    n = len(grpo)
    out = []
    for _ in range(B):
        ix = [rng.randrange(n) for _ in range(n)]
        g = sum(grpo[i] for i in ix) / n
        mx = max(sum(v[i] for i in ix) / n for v in arms.values())
        out.append(g - mx)
    out.sort()
    return out


def boot_delta_2level(per_seed, B=BOOT_B, seed=20260721):
    """协议 §五的两层 bootstrap：先对训练种子重采样，再对题目重采样。

    per_seed: [ (grpo_vec, {arm: vec}), ... ] 每个训练种子一项。
    """
    rng = random.Random(seed)
    S = len(per_seed)
    n = len(per_seed[0][0])
    out = []
    for _ in range(B):
        sidx = [rng.randrange(S) for _ in range(S)]          # 第一层：种子
        ix = [rng.randrange(n) for _ in range(n)]            # 第二层：题目
        ds = []
        for s in sidx:
            g, arms = per_seed[s]
            ga = sum(g[i] for i in ix) / n
            mx = max(sum(v[i] for i in ix) / n for v in arms.values())
            ds.append(ga - mx)
        out.append(sum(ds) / len(ds))
    out.sort()
    return out


def q(sorted_list, p):
    return sorted_list[max(0, min(len(sorted_list) - 1, int(p * (len(sorted_list) - 1))))]


# ================= 逐题向量来源 =================
def maj_from_vals(vals, k, rel_tol=0.01):
    vs = [v for v in vals[:k] if v is not None]
    if not vs:
        return None
    best, bc = None, -1
    for v in vs:
        c = sum(1 for u in vs if abs(u - v) <= rel_tol * max(abs(v), 1e-9))
        if c > bc:
            best, bc = v, c
    return best


def vec_bon(m, seed, k, kind="maj"):
    from verifier import compare
    from gate_v2 import dev_items
    p = os.path.join(HERE, "runs_v2", f"bondump_m{m}_s{seed}.json")
    if not os.path.exists(p):
        return None
    dump = json.load(open(p, encoding="utf-8"))
    items = dev_items(len(dump))
    out = []
    for rec, it in zip(dump, items):
        if kind == "pass":
            out.append(1 if any(x == 1.0 for x in rec["rewards"][:k]) else 0)
        else:
            mv = maj_from_vals(rec["vals"], k)
            out.append(1 if (mv is not None and compare(mv, it["answer"])[0]) else 0)
    return out


def _rewards(obj):
    return [1 if r["reward"] == 1.0 else 0 for r in obj]


def vec_branch(branch, m, seed, tseed=None):
    """读 gate2_eval 产出的 bestdev_{branch}；返回 (meta, 逐题向量)。"""
    suf = f"_t{tseed}" if (tseed is not None and tseed != seed) else ""
    p = os.path.join(HERE, "runs_g2", f"bestdev_{branch}_m{m}_s{seed}{suf}.json")
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p, encoding="utf-8"))
    pi = d.get("per_item_best")
    return d, (_rewards(pi) if pi else None)


def vec_warm(m, seed):
    p = os.path.join(HERE, "runs_v2", f"peritem_warm_m{m}_s{seed}.json")
    return _rewards(json.load(open(p, encoding="utf-8"))) if os.path.exists(p) else None


def vec_solver(m, seed):
    p = os.path.join(HERE, "runs_v2", f"peritem_solver_scarce_m{m}_s{seed}.json")
    return _rewards(json.load(open(p, encoding="utf-8"))) if os.path.exists(p) else None


# ================= 分层轴 =================
def strata(dev_n):
    """协议 §五要求按「程序长度、操作类型、难度、来源」分层。

    可构造：程序长度（gold program 行数）、操作类型（gold program AST 的运算集合）。
    不可构造：难度、报告来源 —— data/CodeFinQA__*.jsonl 无对应字段
    （字段只有 question/answer/task/context/context_type/options/program），
    且 context_type 实测恒为 'string'，不是来源标签。此局限须随报告一并声明。
    """
    from gate_v2 import dev_items
    items = dev_items(dev_n)
    out = []
    for r in items:
        prog = r.get("program", "") or ""
        nline = len([l for l in prog.strip().split("\n") if l.strip()])
        lb = "1行" if nline <= 1 else ("2-3行" if nline <= 3 else ("4-6行" if nline <= 6 else "7+行"))
        ops = set()
        try:
            for nd in ast.walk(ast.parse(prog)):
                if isinstance(nd, ast.BinOp):
                    ops.add(type(nd.op).__name__)
                elif isinstance(nd, ast.Call) and isinstance(nd.func, ast.Name):
                    ops.add(nd.func.id)
        except SyntaxError:
            ops.add("PARSE_ERR")
        if not ops:            opb = "无运算"
        elif ops <= {"Sub"}:   opb = "仅减"
        elif ops <= {"Div"}:   opb = "仅除"
        elif ops <= {"Sub", "Div"}: opb = "减+除(增长率)"
        elif "Mult" in ops:    opb = "含乘"
        elif "Add" in ops:     opb = "含加"
        else:                  opb = "其他"
        out.append({"len_bucket": lb, "op_bucket": opb, "n_lines": nline})
    return out


# ================= 补 W_m 逐题（GPU） =================
def cmd_peritem_warm(m, seed):
    import torch, gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from gate_v2 import (MODEL, MAX_PROMPT, MAX_COMP, CKPT_DEV_N,
                         dev_items, make_prompt, run_tag)
    from gate2_passk import extract_code
    from verifier import verify
    from hb import HB

    bd = json.load(open(os.path.join(HERE, "runs_v2", f"bestdev_m{m}_s{seed}.json"),
                        encoding="utf-8"))
    ck = bd["best_ckpt"]
    cp = "base" if m == 0 else os.path.join(HERE, "runs_v2", run_tag(m, seed), ck)
    items = dev_items(CKPT_DEV_N)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    if cp != "base":
        model = PeftModel.from_pretrained(model, cp).merge_and_unload()
    model.eval()
    hb = HB(f"peritem_warm_m{m}_s{seed}", total=len(items),
            phase=f"W_{m} 逐题 greedy（配对分析用）", extra={"ckpt": ck}, every=1)
    res, hits, B = [], 0, 16
    for bi in range(0, len(items), B):
        b = items[bi:bi + B]
        enc = tok([make_prompt(tok, r) for r in b], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_PROMPT, padding_side="left").to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAX_COMP, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for r, t in zip(b, tok.batch_decode(o[:, enc["input_ids"].shape[1]:],
                                            skip_special_tokens=True)):
            v = verify(extract_code(t), r["answer"], True, str(r.get("context")) if r.get("context") is not None else None)
            hits += v["reward"]
            res.append({"reward": v["reward"], "reason": v["reason"]})
        hb.update(min(bi + B, len(items)), acc=hits / len(res))
    out = os.path.join(HERE, "runs_v2", f"peritem_warm_m{m}_s{seed}.json")
    json.dump(res, open(out, "w"), indent=1)
    acc = hits / len(res)
    hb.done(acc=acc)
    print(f"PERITEM_WARM_DONE m={m} ckpt={ck} acc={acc:.4f} -> {out}")
    if abs(acc - bd["best_dev_acc"]) > 1e-9:
        print(f"  ⚠ 与 bestdev 记录不一致: {acc:.4f} vs {bd['best_dev_acc']:.4f}")
    del model; gc.collect(); torch.cuda.empty_cache()


# ================= 报告 =================
def pct(x):
    return f"{x*100:6.2f}%"


def collect(m, seed, tseed=None, n_bon=43):
    """返回 (arms, grpo_vec, meta, missing)。arms 只含**协议 Δ max 集合**的四个成员。"""
    missing, arms, meta = [], {}, {}

    w = vec_warm(m, seed)
    if w is None:
        missing.append(f"runs_v2/peritem_warm_m{m}_s{seed}.json"
                       f"  →  python gate4_v2.py peritem-warm --m {m} --seed {seed}")
    else:
        arms[f"W_{m}"] = w

    for br, label in (("csft", "continued SFT"), ("rssft", "RS-SFT")):
        d, v = vec_branch(br, m, seed, tseed)
        if d is None:
            missing.append(f"runs_g2/bestdev_{br}_m{m}_s{seed}.json  →  python gate2_eval.py --all --m {m} --seed {seed}")
        elif v is None:
            missing.append(f"runs_g2/bestdev_{br}_m{m}_s{seed}.json 的 per_item_best 为 null")
        else:
            arms[label] = v
            meta[br] = d

    sv = vec_solver(m, seed)
    if sv is None:
        missing.append(f"runs_v2/peritem_solver_scarce_m{m}_s{seed}.json"
                       f"  →  python solver_arm.py --m {m} --seed {seed} --variant scarce")
    else:
        arms["solver"] = sv

    dg, g = vec_branch("grpo", m, seed, tseed)
    if dg is None:
        missing.append(f"runs_g2/bestdev_grpo_m{m}_s{seed}.json  →  python gate2_eval.py --all --m {m} --seed {seed}")
    elif g is None:
        missing.append(f"runs_g2/bestdev_grpo_m{m}_s{seed}.json 的 per_item_best 为 null")
    else:
        meta["grpo"] = dg

    # 附加诊断臂（**不进** 协议 Δ 的 max 集合；PREREG_v2 §9.2 曾提议加入 best-of-n）
    extra = {}
    for k in (16, n_bon):
        v = vec_bon(m, seed, k, "maj")
        if v:
            extra[f"maj@{k}"] = v
    vp = vec_bon(m, seed, n_bon, "pass")
    if vp:
        extra[f"pass@{n_bon} (oracle)"] = vp

    return arms, g, meta, missing, extra


def verdict_gate2(m, seeds_data, extra, meta, dev_n, args):
    """按协议判决树输出。seeds_data: [(seed, grpo_vec, arms), ...]"""
    n = dev_n
    multi = len(seeds_data) > 1
    _, g0, arms0 = seeds_data[0]

    # ---- Δ 点估计（整数计数，避免浮点边界） ----
    kg = sum(g0)
    kbest_name = max(arms0, key=lambda a: sum(arms0[a]))
    kbest = sum(arms0[kbest_name])
    delta_pp = (kg - kbest) / n * 100

    print("\n【2】Gate 2 判决")
    print(f"  Δ(m={m}) = Acc(GRPO) − max{{W_m, continued SFT, RS-SFT, solver}}")
    print(f"    最强非 RL = {kbest_name}  {kbest}/{n} = {pct(kbest/n)}")
    print(f"    GRPO                    {kg}/{n} = {pct(kg/n)}")
    print(f"    Δ = ({kg} − {kbest})/{n} = {delta_pp:+.2f}pp      （整数计数比较，非浮点）")

    # ---- Δ 的 CI（配对 bootstrap） ----
    if multi:
        per = [(gv, av) for _, gv, av in seeds_data]
        boots = boot_delta_2level(per)
        ci_kind = "两层 bootstrap（先种子后题目，协议 §五）"
    else:
        boots = boot_delta(g0, arms0)
        ci_kind = "题目层配对 bootstrap（**退化版**：单训练种子，种子层方差未量化，CI 偏窄）"
    lo95, hi95 = q(boots, .025) * 100, q(boots, .975) * 100
    lo90, hi90 = q(boots, .05) * 100, q(boots, .95) * 100
    print(f"    95% CI = [{lo95:+.2f}, {hi95:+.2f}] pp")
    print(f"    90% CI = [{lo90:+.2f}, {hi90:+.2f}] pp     ← Task No-Go 判据1 用这个")
    print(f"    CI 方法：{ci_kind}")

    # ---- Go 的四重合取 ----
    print("\n  ── Gate 2 Go 四重合取 ──")
    c1 = (kg - kbest) >= math.ceil(GO_PP * n)          # 整数比较：≥3pp
    print(f"   1. GRPO 领先最强非 RL ≥3pp        : {'满足' if c1 else '不满足'}"
          f"   （需领先 ≥{math.ceil(GO_PP*n)} 题，实际 {kg-kbest} 题）")

    pos = sum(1 for _, gv, av in seeds_data
              if (sum(gv) - max(sum(v) for v in av.values())) >= math.ceil(GO_PP * n))
    c2 = len(seeds_data) >= N_SEED_REQ and pos >= 2
    print(f"   2. 3 个种子中至少 2 个为正        : {'满足' if c2 else '不满足'}"
          f"   （实有 {len(seeds_data)} 个训练种子，为正 {pos} 个）")
    if len(seeds_data) < N_SEED_REQ:
        print(f"      └─ 种子数 <{N_SEED_REQ}，本条**逻辑上不可评估** → Go 不可达")

    # 合取项 3：无 hacking / 格式捷径 / 增益不由少数题型独占
    c3, c3_why = check_conjunct3(m, seeds_data[0], n)

    rs = arms0.get("RS-SFT")
    c4 = rs is not None and (kg - sum(rs)) > 0
    print(f"   4. GRPO 优于 RS-SFT              : {'满足' if c4 else '不满足'}"
          + (f"   （{kg} vs {sum(rs)}，差 {(kg-sum(rs))/n*100:+.2f}pp）" if rs else "   （RS-SFT 缺失）"))

    go = c1 and c2 and c3 and c4

    # ---- Task No-Go 三条件（本档位） ----
    print("\n  ── Task No-Go 条件（本档位）──")
    t1 = hi90 < GO_PP * 100
    print(f"   1. Δ 的 90% CI 上界 < +3pp        : {'满足' if t1 else '不满足'}   （上界 {hi90:+.2f}pp）")
    t2 = False
    if rs is not None:
        gap = (kg - sum(rs)) / n * 100
        cheaper = cost_cheaper(meta)
        t2 = abs(gap) <= TIE_PP * 100 and cheaper
        print(f"   2. |GRPO−RS-SFT| ≤2pp 且 RS-SFT 更省 : {'满足' if t2 else '不满足'}"
              f"   （差 {gap:+.2f}pp，RS-SFT 更省={cheaper}）")
    t3 = multi and all((sum(gv) - max(sum(v) for v in av.values())) < 0 for _, gv, av in seeds_data)
    print(f"   3. GRPO 稳定弱于最强非 RL         : {'满足' if t3 else '不满足'}"
          f"   （需多种子一致为负；实有 {len(seeds_data)} 个种子）")
    tno = t1 or t2 or t3

    # ---- INCONCLUSIVE 触发条件 ----
    in_band = INC_LO * 100 <= abs(delta_pp) <= INC_HI * 100
    ci_cross = lo90 < GO_PP * 100 < hi90
    seed_short = len(seeds_data) < N_SEED_REQ
    seed_conflict = multi and 0 < pos < len(seeds_data)

    print("\n  ── INCONCLUSIVE 触发 ──")
    print(f"   平均效应落在 2–5pp 带内           : {in_band}   （|Δ|={abs(delta_pp):.2f}pp）")
    print(f"   90% CI 跨过 +3pp 门槛             : {ci_cross}")
    print(f"   训练种子不足 {N_SEED_REQ} 个              : {seed_short}")
    print(f"   种子方向冲突                      : {seed_conflict}")

    # ---- 判决树 ----
    print("\n" + "=" * 78)
    if go:
        v = "PROVISIONAL-GO"
        note = "四重合取全部满足。按协议进入 Gate 3 完整确认实验。"
    elif seed_short or in_band or ci_cross or seed_conflict:
        # 协议：追加种子前保持 INCONCLUSIVE，不能抢判
        v = "INCONCLUSIVE"
        why = []
        if seed_short:     why.append(f"训练种子仅 {len(seeds_data)}/{N_SEED_REQ}")
        if in_band:        why.append(f"|Δ|={abs(delta_pp):.2f}pp 落在 2–5pp 带内")
        if ci_cross:       why.append("90% CI 跨过 +3pp")
        if seed_conflict:  why.append("种子方向冲突")
        note = "；".join(why) + "。协议要求追加第 4、5 个训练种子，追加前不得判 Go/No-Go。"
        # 但若 Task No-Go 条件已硬性满足且方向明确为负，标注为待确认的 No-Go 方向
        if tno and delta_pp < 0:
            note += ("\n  ★ 注意：Task No-Go 条件已在本档位满足（Δ 90% CI 上界 "
                     f"{hi90:+.2f}pp < +3pp），方向明确为负。"
                     "\n    协议要求「两个低演示档位均满足」才可判 TASK-NO-GO —— "
                     "本轮只有一个档位有 RL 臂，故仍记 INCONCLUSIVE。")
    elif tno:
        v = "TASK-NO-GO（本档位条件满足，待第二个低演示档位确认）"
        note = "协议：两个低演示档位均满足任意一项方可判 `CodeFinQA 上 demonstration-scarce RLVR No-Go`。"
    else:
        v = "INCONCLUSIVE"
        note = "既未满足 Go 四重合取，也未满足 Task No-Go 任一条件。"
    print(f"  ⇒ 判决：**{v}**")
    print(f"    {note}")
    print("=" * 78)
    return dict(verdict=v, delta_pp=delta_pp, ci90=[lo90, hi90], ci95=[lo95, hi95],
                conj=dict(c1=c1, c2=c2, c3=c3, c4=c4), taskno=dict(t1=t1, t2=t2, t3=t3),
                best_nonrl=kbest_name)


def cost_cheaper(meta):
    """RS-SFT 是否比 GRPO 成本更低（wall + verifier 双轴，任一显著更低即算）。"""
    g, r = (meta.get("grpo") or {}).get("train_meta"), (meta.get("rssft") or {}).get("train_meta")
    if not g or not r:
        return None
    gw, rw = g.get("wall_seconds"), r.get("wall_seconds")
    gv, rv = g.get("verifier_calls"), r.get("verifier_calls")
    return bool((rw and gw and rw < gw * 0.9) or (rv and gv and rv < gv * 0.9))


def check_conjunct3(m, sd, n):
    """Go 合取项 3：无 verifier hacking / 格式捷径 / 增益不由少数题型独占。"""
    seed, g, arms = sd
    st = strata(n)
    best = max(arms, key=lambda a: sum(arms[a]))
    bv = arms[best]
    print(f"   3. 无 hacking / 增益不集中        : ", end="")

    # (a) 增益集中度：GRPO 相对最强非 RL 的净增益在各分层桶的分布
    concentrated, detail = None, []
    for axis in ("len_bucket", "op_bucket"):
        buckets = defaultdict(lambda: [0, 0, 0])   # 桶 -> [n, grpo命中, best命中]
        for i, s in enumerate(st):
            b = buckets[s[axis]]
            b[0] += 1; b[1] += g[i]; b[2] += bv[i]
        gains = {k: v[1] - v[2] for k, v in buckets.items()}
        pos = sum(x for x in gains.values() if x > 0)
        if pos > 0:
            top = max(gains.values())
            frac = top / pos
            detail.append((axis, buckets, gains, frac))
            if frac > 0.5:
                concentrated = True

    # (b) verifier hacking：需要 GRPO 生成的**程序文本**才能审计，
    #     而 gate2_eval 的 per_item_best 只存 {reward, reason}，程序未落盘 → 不可评估
    hack_evaluable = False

    if not hack_evaluable:
        print("**不可评估** → 合取项不成立（Go 不可达）")
        print("      └─ verifier hacking / 格式捷径审计需要 GRPO 生成的程序文本，")
        print("         而 gate2_eval.py 的 per_item_best 只存 {reward, reason}，程序未落盘。")
        print("         协议要求「没有 verifier hacking」是 Go 的必要条件；未经检验 ≠ 已满足。")
    for axis, buckets, gains, frac in detail:
        print(f"      · 增益集中度[{axis}]：最大单桶占正增益 {frac*100:.0f}%"
              f"{'  ⚠>50%' if frac > 0.5 else ''}")
    return False, "hacking 审计不可评估（程序未落盘）"


def cmd_report(m, seed, n_bon, seeds_arg):
    from gate_v2 import dev_items
    tseeds = [int(x) for x in seeds_arg.split(",")] if seeds_arg else [None]

    seeds_data, all_missing, meta, extra = [], [], {}, {}
    for ts in tseeds:
        arms, g, mt, missing, ex = collect(m, seed, ts, n_bon)
        if ts == tseeds[0]:
            meta, extra = mt, ex
        if missing:
            all_missing += [f"[训练种子 {ts if ts is not None else seed}] {x}" for x in missing]
        if g is not None and arms:
            seeds_data.append((ts if ts is not None else seed, g, arms))

    n = len(seeds_data[0][1]) if seeds_data else 0
    print("=" * 78)
    print(f"Gate 2 判决 —— m={m}, 演示子集 seed={seed}, dev n={n}（test 664 冻结未用）")
    print("=" * 78)

    # ---- 完整性护栏（协议：任一必需臂缺失 → CANNOT-EVALUATE，不得裁决） ----
    if all_missing:
        print("\n【缺失的必需产物】")
        for x in all_missing:
            print("  ✗ " + x)
        print("\n" + "=" * 78)
        print("  ⇒ 判决：**CANNOT-EVALUATE**")
        print("    Δ 的 max 集合不完整。缺臂会让 max 悄悄塌缩、门槛降低，产生**假 Go**。")
        print("    协议：Gate 0 任何一项失败 → CANNOT-EVALUATE，绝不能判想法 No-Go。")
        print("=" * 78)
        sys.exit(1)

    _, g0, arms0 = seeds_data[0]

    # ---- 【1】各臂 ----
    print(f"\n【1】各臂 dev 准确率（Wilson 95% CI）  —— 协议 Δ max 集合 = 前四行")
    print(f"  {'臂':<24}{'acc':>9}  {'95% CI':^18}{'计数':>10}")
    print("  " + "-" * 64)
    for k in sorted(arms0, key=lambda a: -sum(arms0[a])):
        v = arms0[k]; kk = sum(v); lo, hi = wilson(kk, n)
        print(f"  {k:<24}{pct(kk/n):>9}  [{lo*100:5.2f}%, {hi*100:5.2f}%] {kk:>5}/{n}")
    kk = sum(g0); lo, hi = wilson(kk, n)
    print(f"  {'GRPO (RL)':<24}{pct(kk/n):>9}  [{lo*100:5.2f}%, {hi*100:5.2f}%] {kk:>5}/{n}")
    if extra:
        print("\n  附加诊断臂（**不进**协议 Δ 的 max 集合）：")
        for k, v in extra.items():
            kk = sum(v); lo, hi = wilson(kk, n)
            print(f"  {k:<24}{pct(kk/n):>9}  [{lo*100:5.2f}%, {hi*100:5.2f}%] {kk:>5}/{n}")

    # ---- 【2】判决 ----
    res = verdict_gate2(m, seeds_data, extra, meta, n, None)

    # ---- 【3】配对 McNemar（Holm 校正） ----
    print("\n【3】逐题配对 McNemar（GRPO vs 各臂，连续性校正 + Holm 多重校正）")
    names, rows, ps = [], [], []
    for k, v in list(arms0.items()) + list(extra.items()):
        r = mcnemar(g0, v)
        names.append(k); rows.append(r); ps.append(r["p_exact"])
    hp = holm(ps)
    print(f"  {'对手':<24}{'Δ(pp)':>8}{'仅GRPO':>8}{'仅对手':>8}{'χ²':>9}{'p':>10}{'p(Holm)':>10}")
    print("  " + "-" * 77)
    for k, r, p, ph in zip(names, rows, ps, hp):
        v = arms0.get(k, extra.get(k))
        print(f"  {k:<24}{(sum(g0)-sum(v))/n*100:>+8.2f}{r['only_a']:>8}{r['only_b']:>8}"
              f"{r['chi2']:>9.2f}{p:>10.4g}{ph:>10.4g}")
    print("  唯一 confirmatory 对比是「GRPO vs 最强非 RL 臂」，其余为描述性。")

    # ---- 【4】分层报告 ----
    print("\n【4】分层准确率（协议 §五）")
    st = strata(n)
    for axis, title in (("len_bucket", "程序长度"), ("op_bucket", "运算类型")):
        print(f"\n  ── 按{title} ──")
        buckets = defaultdict(list)
        for i, s in enumerate(st):
            buckets[s[axis]].append(i)
        hdr = f"  {'桶':<16}{'n':>5}" + "".join(f"{k[:10]:>12}" for k in list(arms0) + ["GRPO"])
        print(hdr)
        for b, idxs in sorted(buckets.items(), key=lambda x: -len(x[1])):
            line = f"  {b:<16}{len(idxs):>5}"
            for k in list(arms0):
                line += f"{sum(arms0[k][i] for i in idxs)/len(idxs)*100:>11.1f}%"
            line += f"{sum(g0[i] for i in idxs)/len(idxs)*100:>11.1f}%"
            print(line)
    print("\n  ⚠ 协议还要求按「难度」与「报告来源」分层，但 data/CodeFinQA__*.jsonl")
    print("    无对应字段（仅 question/answer/task/context/context_type/options/program，")
    print("    且 context_type 恒为 'string'），两轴不可构造。已记入 DEVIATIONS_v2.md。")

    # ---- 【5】失败模式分解 ----
    print("\n【5】失败模式分解（协议 Gate 0：语法/执行/超时/空输出/verifier 拒绝分开统计）")
    for br, path in (("GRPO", f"runs_g2/bestdev_grpo_m{m}_s{seed}.json"),
                     ("W_m", f"runs_v2/peritem_warm_m{m}_s{seed}.json"),
                     ("solver", f"runs_v2/peritem_solver_scarce_m{m}_s{seed}.json")):
        p = os.path.join(HERE, path)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        pi = d.get("per_item_best") if isinstance(d, dict) else d
        if not pi:
            continue
        c = Counter(r.get("reason", "?").split("(")[0] for r in pi)
        print(f"  {br:<8}" + "  ".join(f"{k}={v}" for k, v in c.most_common(6)))

    # ---- 【6】成本 ----
    print("\n【6】预算与成本（协议 §二/§五）")
    wm = sum(arms0.get(f"W_{m}", [0])) / n
    print(f"  {'分支':<10}{'wall(s)':>10}{'vcalls':>9}{'gen_tok':>10}{'acc':>8}{'GPU-h/pp':>11}")
    print("  " + "-" * 60)
    for br in ("grpo", "csft", "rssft"):
        d = meta.get(br)
        if not d:
            continue
        tm = d.get("train_meta") or {}
        acc = d.get("best_dev_acc", 0)
        gain = (acc - wm) * 100
        w = tm.get("wall_seconds")
        gh = (w / 3600 / gain) if (w and gain > 0) else None
        print(f"  {br:<10}{(w or 0):>10.0f}{str(tm.get('verifier_calls','—')):>9}"
              f"{str(tm.get('gen_tokens','—')):>10}{acc*100:>7.2f}%"
              f"{(f'{gh:.3f}' if gh else '—'):>11}")
    print("  口径：wall 只含训练、不含评测；rssft 受 verifier 预算先停故墙钟远少于预算；")
    print("        三分支 GPU 争用条件不同（见 DEVIATIONS_v2.md）——GPU-h/pp 仅供数量级参考。")
    print("\n" + "=" * 78)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report", "peritem-warm"])
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="演示子集种子")
    ap.add_argument("--seeds", default="", help="训练种子列表，如 0,1,2（协议要求 3 个）")
    ap.add_argument("--n-bon", type=int, default=43)
    a = ap.parse_args()
    if a.cmd == "peritem-warm":
        cmd_peritem_warm(a.m, a.seed)
    else:
        cmd_report(a.m, a.seed, a.n_bon, a.seeds)

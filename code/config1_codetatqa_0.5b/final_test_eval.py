# -*- coding: utf-8 -*-
"""冻结 test 集最终确认（协议 §9：dev 选点，test 只跑一次）。

用途：dev 上的 Δ 结论出齐后，在**从未参与任何选择**的 test 集上确认一次。
每个臂用它在 dev 上已选定的 best 点（不在 test 上重新选点，否则 test 就被污染了）：

  W_m           runs_v2/bestdev_m{m}_s{seed}.json  → best_ckpt
  continued SFT runs_g2/bestdev_csft_m{m}_s{seed}.json  → best_point
  RS-SFT        runs_g2/bestdev_rssft_m{m}_s{seed}.json → best_point
  GRPO          runs_g2/bestdev_grpo_m{m}_s{seed}.json  → best_point
  solver        确定性程序，检索库仍为该档 m 条 gold（scarce 变体）

    python final_test_eval.py --m 64 --seed 0 [--n 0]   # n=0 表示全量 test

产出 runs_test/test_m{m}_s{seed}.json：各臂 test 准确率 + 逐题结果（供配对检验）。

⚠ 这个脚本会消耗"冻结 test 只用一次"的额度。跑之前确认 dev 分析已全部完成。
"""
import os, sys, json, gc, argparse

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from verifier import verify
from gate2_passk import extract_code
from hb import HB
from gate_v2 import MODEL, MAX_PROMPT, MAX_COMP, make_prompt, load_train_rows, get_split
from gate2 import out_dir, warm_dir

DATA_TEST = None   # 运行时按 gate_v2 的数据集名推断


def test_items(n=0):
    """冻结 test 集。只保留 gold 自洽的题（与 dev 口径一致）。

    data/ 下有多个 *__test.jsonl（TAT-QA / FinCode / SEC-NUM ...），必须选
    **与训练同一数据集**的那个：从 gate_v2.load_train_rows 读的 train 文件名推断，
    不能用 glob 取第一个（会抓到同名前缀的 TAT-QA 而非 CodeTAT-QA）。
    """
    import re
    src = open(os.path.join(HERE, "gate_v2.py"), encoding="utf-8").read()
    mm = re.search(r'"data",\s*"([A-Za-z0-9\-]+)__train\.jsonl"', src)
    if not mm:
        raise SystemExit("无法从 gate_v2.py 推断数据集名")
    ds = mm.group(1)
    path = os.path.join(HERE, "data", f"{ds}__test.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    clean = []
    for r in rows:
        cx = str(r.get("context")) if (r.get("context") is not None and
                                       str(r.get("context_type", "")).lower() == "json") else None
        if verify(r["program"], r["answer"], True, cx)["reward"] == 1.0:
            clean.append(r)
    print(f"test: {len(clean)}/{len(rows)} gold 自洽（来源 {os.path.basename(path)}）", flush=True)
    return clean[:n] if n else clean


def _ctx(r):
    return str(r.get("context")) if (r.get("context") is not None and
                                     str(r.get("context_type", "")).lower() == "json") else None


def load_arm(kind, m, seed):
    """按 dev 已选定的 best 点加载模型；返回 (tok, model) 或 None。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if kind == "warm":
        bd = os.path.join(HERE, "runs_v2", f"bestdev_m{m}_s{seed}.json")
        if not os.path.exists(bd):
            return None, f"缺 {bd}"
        best = json.load(open(bd, encoding="utf-8"))["best_ckpt"]
        tok = AutoTokenizer.from_pretrained(MODEL)
        if best == "base":
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
        else:
            base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
            model = PeftModel.from_pretrained(
                base, os.path.join(HERE, "runs_v2", f"warm_m{m}_s{seed}", best)).merge_and_unload()
        return (tok, model.eval()), f"warm@{best}"

    bd = os.path.join(HERE, "runs_g2", f"bestdev_{kind}_m{m}_s{seed}.json")
    if not os.path.exists(bd):
        return None, f"缺 {bd}"
    d = json.load(open(bd, encoding="utf-8"))
    best = d["best_point"]
    od = out_dir(kind, m, seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if kind == "rssft":
        # rssft 的每个点本身是完整模型
        path = os.path.join(od, best)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    else:
        base = AutoModelForCausalLM.from_pretrained(warm_dir(m, seed), dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(base, os.path.join(od, best)).merge_and_unload()
    return (tok, model.eval()), f"{kind}@{best}"


def greedy_eval(tok, model, items, hb, off, batch=16):
    import torch
    hits, res = 0, []
    for bi in range(0, len(items), batch):
        b = items[bi:bi + batch]
        enc = tok([make_prompt(tok, r) for r in b], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_PROMPT, padding_side="left").to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAX_COMP, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for r, t in zip(b, tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            v = verify(extract_code(t), r["answer"], True, _ctx(r))
            hits += v["reward"]
            res.append(int(v["reward"]))
        if hb:
            hb.update(off + min(bi + batch, len(items)), acc=hits / len(res))
    return hits / max(len(res), 1), res


def solver_on_test(items, m, seed):
    """solver scarce 变体：检索库 = 该档 m 条 gold（与 dev 口径一致）。"""
    try:
        import solver_arm as SA
    except Exception as e:
        return None, [], f"solver_arm 导入失败: {e}"
    S = SA.S
    rows = load_train_rows()
    sp = get_split()
    demo_idx = sp["subsets"][str(seed)][str(m)]
    try:
        all_flags = S.load_sc_flags(rows)
        store = SA.build_store(rows, demo_idx, all_flags)
    except Exception as e:
        return None, [], f"检索库构建失败: {e}"
    hits, res = 0, []
    for r in items:
        # build_program 返回 (program, kind) 元组（见 solver_arm.py:55-61），不是裸字符串
        prog = None
        try:
            out = S.build_program(r["question"], r["context"], store)
            prog = out[0] if isinstance(out, (tuple, list)) else out
        except Exception:
            prog = None
        ok = 0
        if isinstance(prog, str) and prog.strip():
            ok = int(verify(prog, r["answer"], True, _ctx(r))["reward"] == 1.0)
        hits += ok
        res.append(ok)
    return hits / max(len(items), 1), res, f"solver(lib={len(demo_idx)})"


def main(m, seed, n):
    import torch
    items = test_items(n)
    outdir = os.path.join(HERE, "runs_test")
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"test_m{m}_s{seed}.json")
    if os.path.exists(outp):
        print(f"已存在 {outp} —— 冻结 test 只跑一次，拒绝覆盖。要重跑请先手动改名。")
        return

    result = {"m": m, "seed": seed, "n_test": len(items), "arms": {}}

    # solver（CPU，先跑）
    acc, res, tag = solver_on_test(items, m, seed)
    if acc is not None:
        result["arms"]["solver"] = {"acc": acc, "per_item": res, "point": tag}
        print(f"  solver        test_acc={acc:.4f}  ({tag})", flush=True)
    else:
        print(f"  solver 失败: {tag}", flush=True)

    # 四个模型臂
    arms = ["warm", "csft", "rssft", "grpo"]
    hb = HB(f"test_m{m}_s{seed}", total=len(arms) * len(items),
            phase=f"冻结 test 最终确认 m={m} seed={seed}", extra={"n_test": len(items)}, every=1)
    for i, kind in enumerate(arms):
        loaded, tag = load_arm(kind, m, seed)
        if loaded is None:
            print(f"  {kind:<12} 跳过：{tag}", flush=True)
            continue
        tok, model = loaded
        acc, res = greedy_eval(tok, model, items, hb, i * len(items))
        result["arms"][kind] = {"acc": acc, "per_item": res, "point": tag}
        print(f"  {kind:<12} test_acc={acc:.4f}  ({tag})", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    json.dump(result, open(outp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    hb.done(n_arms=len(result["arms"]))

    # Δ 汇总
    if "grpo" in result["arms"]:
        non_rl = {k: v["acc"] for k, v in result["arms"].items() if k != "grpo"}
        if non_rl:
            best_arm = max(non_rl, key=non_rl.get)
            delta = result["arms"]["grpo"]["acc"] - non_rl[best_arm]
            print(f"\n  Δ(test, m={m}, seed={seed}) = GRPO − {best_arm} = {delta*100:+.2f}pp")
    print(f"\n写入 {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="0=全量 test")
    a = ap.parse_args()
    main(a.m, a.seed, a.n)

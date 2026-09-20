# -*- coding: utf-8 -*-
"""PREREG v2：演示数量剂量曲线（Gate 0 / Gate 1）。

子命令：
  split                      切 dev / demo pool，生成嵌套演示子集（3 种子）
  gate0                      实现与 verifier 审计（CPU，最便宜的杀点）
  warm --m M --seed S        训练 warm start W_m
  evalckpt --m M --seed S    评所有 checkpoint 的 dev greedy，选 best-dev（§9.1 修正）
  bon --m M --seed S --n N   test-time verifier best-of-n 基线（§9.2 修正）

所有阶段都通过 hb.HB 上报进度：随时 `python progress.py` 可见。
"""
import os, sys, json, time, random, argparse

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 模型已在本地缓存：强制离线，杜绝 from_pretrained 的联网校验。
# （并行时多进程同时打 huggingface.co 会触发 SSL EOF，串行时侥幸没暴露）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from verifier import verify
from gate2_passk import PROMPT_TMPL, extract_code
from hb import HB

MODEL = "/home/neu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"  # scp 破坏 HF blobs 符号链接，直接用 snapshot
SEED = 42
MAX_PROMPT, MAX_COMP = 3072, 256
DEV_N = 400                      # 从 train clean 留出的 dev（test 664 全程冻结）
CKPT_DEV_N = 200                 # checkpoint 选择用的 dev 子集（省算力）
M_GRID = [0, 4, 8, 16, 64, 256, 1024, 2446]   # CodeTAT-QA: 2400=占位，split 后按 clean-400 回填   # 3635 = demo pool 全量（train clean 4035 − dev 400，见 DEVIATIONS D4）
DEMO_SEEDS = [0, 1, 2]
EFF_BATCH = 64                   # 所有档位固定有效 batch，保证优化器超参可比
MAX_STEPS_CAP, MAX_EPOCHS = 800, 60
N_CKPT = 6

SPLIT_PATH = os.path.join(HERE, "data", "v2_split.json")


# ---------------- 数据 ----------------
def load_train_rows():
    return [json.loads(l) for l in open(os.path.join(HERE, "data", "CodeTAT-QA__train.jsonl"), encoding="utf-8")]


def cmd_split():
    rows = load_train_rows()
    hb = HB("split", total=len(rows), phase="Gate0-a 切分 dev / demo pool", every=500)
    clean = []
    for i, r in enumerate(rows, 1):
        if verify(r["program"], r["answer"], True, str(r.get("context")) if r.get("context") is not None else None)["reward"] == 1.0:
            clean.append(i - 1)
        if i % 500 == 0:
            hb.update(i, clean=len(clean), rate=len(clean) / i)
    hb.update(len(rows), clean=len(clean))

    rng = random.Random(SEED)
    idx = clean[:]
    rng.shuffle(idx)
    dev_idx, pool_idx = idx[:DEV_N], idx[DEV_N:]

    # 嵌套演示子集：每个种子独立打乱 pool，前缀即嵌套
    subsets = {}
    for s in DEMO_SEEDS:
        p = pool_idx[:]
        random.Random(1000 + s).shuffle(p)
        subsets[str(s)] = {str(m): p[:m] for m in M_GRID}

    json.dump({"dev_idx": dev_idx, "pool_idx": pool_idx, "subsets": subsets,
               "n_clean": len(clean), "n_raw": len(rows),
               "m_grid": M_GRID, "demo_seeds": DEMO_SEEDS},
              open(SPLIT_PATH, "w"), indent=1)
    hb.done(n_clean=len(clean), dev=len(dev_idx), pool=len(pool_idx))
    print(f"SPLIT_DONE clean={len(clean)}/{len(rows)} dev={len(dev_idx)} pool={len(pool_idx)}")


def get_split():
    return json.load(open(SPLIT_PATH, encoding="utf-8"))


def demos_for(m, seed):
    sp = get_split()
    rows = load_train_rows()
    return [rows[i] for i in sp["subsets"][str(seed)][str(m)]]


def dev_items(n=None):
    sp = get_split()
    rows = load_train_rows()
    d = [rows[i] for i in sp["dev_idx"]]
    return d[:n] if n else d


# ---------------- Gate 0 审计 ----------------
def cmd_gate0():
    sp = get_split()
    rows = load_train_rows()
    checks, fails = {}, []
    total = 7
    hb = HB("gate0", total=total, phase="Gate 0 实现与 verifier 审计")

    # 1. gold 接受率（在 clean 子集上必须 100%）
    dev = [rows[i] for i in sp["dev_idx"]]
    ok = sum(verify(r["program"], r["answer"], True, str(r.get("context")) if r.get("context") is not None else None)["reward"] == 1.0 for r in dev)
    checks["gold_accept_dev"] = ok / len(dev)
    hb.update(1, gold_accept=ok / len(dev))
    if ok != len(dev):
        fails.append("gold 接受率 < 100%")

    # 2. 常数程序 false accept（语义攻击）
    atk, fa = 0, 0
    for r in dev[:200]:
        for prog in [f"answer = {r['answer']}", "answer = 0", "answer = 100",
                     f"x = 1\nanswer = {r['answer']}"]:
            atk += 1
            if verify(prog, r["answer"], True, str(r.get("context")) if r.get("context") is not None else None)["reward"] == 1.0:
                fa += 1
    checks["const_false_accept"] = fa / atk
    hb.update(2, false_accept=fa / atk, n_attack=atk)
    if fa / atk > 0.01:
        fails.append(f"常数程序 false accept {fa/atk:.3f} > 1%")

    # 3. dev / demo pool 无交集
    inter = set(sp["dev_idx"]) & set(sp["pool_idx"])
    checks["dev_pool_overlap"] = len(inter)
    hb.update(3, overlap=len(inter))
    if inter:
        fails.append("dev 与 demo pool 有交集")

    # 4. 嵌套性：小子集必须是大子集的前缀
    nested_ok = True
    for s in sp["demo_seeds"]:
        ss = sp["subsets"][str(s)]
        for a, b in zip(sp["m_grid"], sp["m_grid"][1:]):
            if ss[str(a)] != ss[str(b)][:a]:
                nested_ok = False
    checks["nested"] = nested_ok
    hb.update(4, nested=nested_ok)
    if not nested_ok:
        fails.append("演示子集非嵌套")

    # 5. dev 与 test 无泄漏（按 question+answer 指纹）
    test = [json.loads(l) for l in open(os.path.join(HERE, "data", "CodeTAT-QA__test.jsonl"), encoding="utf-8")]
    tf = {(r["question"].strip(), str(r["answer"])) for r in test}
    leak = sum(1 for r in dev if (r["question"].strip(), str(r["answer"])) in tf)
    checks["dev_test_leak"] = leak
    hb.update(5, dev_test_leak=leak)
    if leak > 0:
        fails.append(f"dev 与 test 重叠 {leak} 条")

    # 6. demo pool 与 test 无泄漏
    pool = [rows[i] for i in sp["pool_idx"]]
    leak2 = sum(1 for r in pool if (r["question"].strip(), str(r["answer"])) in tf)
    checks["pool_test_leak"] = leak2
    hb.update(6, pool_test_leak=leak2)
    if leak2 > 0:
        fails.append(f"demo pool 与 test 重叠 {leak2} 条")

    # 7. verifier 确定性（同输入两次同结果）
    det = all(verify(r["program"], r["answer"], True, str(r.get("context")) if r.get("context") is not None else None)["reward"] == verify(r["program"], r["answer"], True, str(r.get("context")) if r.get("context") is not None else None)["reward"]
              for r in dev[:100])
    checks["deterministic"] = det
    hb.update(7, deterministic=det)
    if not det:
        fails.append("verifier 非确定性")

    verdict = "PASS" if not fails else "CANNOT-EVALUATE"
    json.dump({"checks": checks, "fails": fails, "verdict": verdict},
              open(os.path.join(HERE, "GATE0_v2_RESULT.json"), "w"), indent=1, ensure_ascii=False)
    hb.done(verdict=verdict, n_fail=len(fails))
    print(f"GATE0_VERDICT={verdict} fails={fails}")


# ---------------- warm start ----------------
def make_prompt(tok, r):
    msgs = [{"role": "user", "content": PROMPT_TMPL.format(q=r["question"], ctx=str(r["context"])[:5000])}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_tag(m, seed):
    return f"warm_m{m}_s{seed}"


def cmd_warm(m, seed):
    import torch
    from transformers import AutoTokenizer, TrainerCallback
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    tag = run_tag(m, seed)
    out_dir = os.path.join(HERE, "runs_v2", tag)
    if m == 0:
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"m": 0, "note": "W_0 = base model, 无训练"}, open(out_dir + "/meta.json", "w"))
        hb = HB(tag, total=1, phase=f"warm-start m=0（base，无需训练）")
        hb.done(note="base")
        return

    items = demos_for(m, seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = Dataset.from_list([{"prompt": make_prompt(tok, r),
                             "completion": r["program"].strip() + tok.eos_token} for r in items])

    steps_per_epoch = max(1, len(items) // EFF_BATCH)
    total_steps = min(MAX_STEPS_CAP, steps_per_epoch * MAX_EPOCHS)
    save_steps = max(1, total_steps // N_CKPT)

    hb = HB(tag, total=total_steps,
            phase=f"warm-start SFT m={m} seed={seed}",
            extra={"n_demos": len(items), "steps_per_epoch": steps_per_epoch,
                   "epochs": round(total_steps / steps_per_epoch, 1), "save_steps": save_steps})

    class Report(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                hb.update(state.global_step, loss=logs["loss"],
                          epoch=round(logs.get("epoch", 0), 2),
                          lr=logs.get("learning_rate", 0))

    cfg = SFTConfig(
        output_dir=out_dir, seed=SEED + seed, bf16=True, report_to="none",
        max_steps=total_steps, learning_rate=1e-5,
        per_device_train_batch_size=2, gradient_accumulation_steps=EFF_BATCH // 2,
        max_length=MAX_PROMPT + MAX_COMP, completion_only_loss=True,
        logging_steps=1,                      # 每步都报，进度可见性优先
        save_strategy="steps", save_steps=save_steps, save_total_limit=N_CKPT + 2,
        warmup_ratio=0.03, lr_scheduler_type="cosine",
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    t0 = time.time()
    try:
        trainer = SFTTrainer(model=MODEL, args=cfg, train_dataset=ds,
                             processing_class=tok, peft_config=lora, callbacks=[Report()])
        trainer.train()
        trainer.save_model(out_dir + "/final")
        wall = time.time() - t0
        json.dump({"m": m, "seed": seed, "n_demos": len(items), "wall_seconds": wall,
                   "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
                   "save_steps": save_steps},
                  open(out_dir + "/meta.json", "w"), indent=1)
        hb.done(wall_s=round(wall, 1))
    except Exception as e:
        hb.fail(e)
        raise


# ---------------- 评测 ----------------
def _load_model(path, mem_frac=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    # 硬上限：caching allocator 不归还显存，并行时先起的进程会吃光卡。
    # 给每个进程设比例上限，超限时它自己 OOM，不拖垮同卡的其他任务。
    if mem_frac:
        torch.cuda.set_per_process_memory_fraction(float(mem_frac))
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    if path and path != "base":
        base = PeftModel.from_pretrained(base, path).merge_and_unload()
    base.eval()
    return tok, base


def _greedy_eval(tok, model, items, hb=None, hb_off=0, hb_span=1.0, batch=32):
    import torch
    hits, res = 0, []
    for bi in range(0, len(items), batch):
        b = items[bi:bi + batch]
        enc = tok([make_prompt(tok, r) for r in b], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_PROMPT, padding_side="left").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_COMP, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for r, t in zip(b, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            v = verify(extract_code(t), r["answer"], True, str(r.get("context")) if r.get("context") is not None else None)
            hits += v["reward"]
            res.append({"reward": v["reward"], "reason": v["reason"]})
        if hb:
            hb.update(hb_off + int(hb_span * min(bi + batch, len(items))),
                      acc=hits / len(res), seen=len(res))
    return hits / max(len(res), 1), res


def cmd_evalckpt(m, seed, eval_batch=16, mem_frac=None):
    """评 W_m 的所有 checkpoint 在 dev 上的 greedy acc，选 best-dev（§9.1）。

    逐 ckpt 增量落盘：中途失败（网络/OOM）重跑时自动跳过已评的 ckpt。
    """
    tag = run_tag(m, seed)
    out_dir = os.path.join(HERE, "runs_v2", tag)
    if m == 0:
        ckpts = ["base"]
    else:
        ckpts = sorted([os.path.join(out_dir, d) for d in os.listdir(out_dir)
                        if d.startswith("checkpoint-")],
                       key=lambda p: int(p.rsplit("-", 1)[1]))
        ckpts.append(os.path.join(out_dir, "final"))
    items = dev_items(CKPT_DEV_N)
    part_path = os.path.join(HERE, "runs_v2", f"bestdev_partial_m{m}_s{seed}.json")
    results = json.load(open(part_path, encoding="utf-8")) if os.path.exists(part_path) else {}
    hb = HB(f"evalckpt_m{m}_s{seed}", total=len(ckpts) * len(items),
            phase=f"dev 评测选 best-ckpt m={m} seed={seed}",
            extra={"n_ckpt": len(ckpts), "dev_n": len(items), "已完成": len(results)}, every=1)
    if results:
        hb.log(f"  断点续跑：已有 {len(results)} 个 ckpt 结果，跳过")
    for ci, cp in enumerate(ckpts):
        name = os.path.basename(cp)
        if name in results:
            hb.update((ci + 1) * len(items), skipped=name)
            continue
        tok, model = _load_model(cp, mem_frac)
        acc, _ = _greedy_eval(tok, model, items, hb, hb_off=ci * len(items), batch=eval_batch)
        results[name] = acc
        json.dump(results, open(part_path, "w"), indent=1)     # 每评完一个立刻落盘
        hb.log(f"  ckpt {name} dev_acc={acc:.4f}")
        del model
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()
    best = max(results, key=results.get)
    json.dump({"m": m, "seed": seed, "dev_n": len(items), "per_ckpt": results,
               "best_ckpt": best, "best_dev_acc": results[best]},
              open(os.path.join(HERE, "runs_v2", f"bestdev_m{m}_s{seed}.json"), "w"), indent=1)
    hb.done(best=best, best_dev_acc=results[best])
    print(f"BESTDEV m={m} seed={seed} ckpt={best} acc={results[best]:.4f}")


def majority_at_k(codes, k, rel_tol=0.01, context=None):
    """**可部署**的无 gold 选择器：execution-guarded majority voting。

    对前 k 条采样各自执行取数值（过滤掉常量程序与执行失败者），
    按相对容差聚类后取最大簇的代表值。**全程不看 gold**，
    因此这是现实中真能部署的 test-time 策略。

    对照：pass@k / sel@k 需要 gold 才能挑出正确的那条，是 **oracle 上界**，
    只能作诊断量，不能当基线（否则 RL 的对手是一个不存在的方法）。
    """
    from verifier import run_program, parse_number, _has_computation
    vals = []
    for c in codes[:k]:
        if not _has_computation(c):
            continue
        ok, v = run_program(c, context)
        if not ok:
            continue
        num = parse_number(v)
        if num is not None:
            vals.append(num)
    if not vals:
        return None
    best, best_cnt = None, -1
    for v in vals:
        cnt = sum(1 for u in vals if abs(u - v) <= rel_tol * max(abs(v), 1e-9))
        if cnt > best_cnt:
            best, best_cnt = v, cnt
    return best


def cmd_bon(m, seed, n, dev_n=200, bon_batch=4, mem_frac=None):
    """test-time 选择器基线（§9.2）。同时报告三个量：

      pass@k  : k 条里存在正确的        —— oracle 上界（诊断量）
      sel@k   : verifier 挑出正确的     —— 与 pass@k 恒等，同样是 oracle
      **maj@k**: execution-guarded 多数投票 —— **无 gold，可部署，这才是 RL 的真对手**

    另落盘每条采样的执行数值，便于日后离线重算任意选择器而无需重跑 GPU。
    """
    import torch
    from transformers import set_seed as _set_seed
    # 协议 Gate 0：「固定种子后数据、评测结果和样本顺序可复现」。
    # 本函数用 do_sample=True 采样，此前全无定种 —— maj@k 是 Gate 2 判决的输入之一，
    # 却在固定种子下不可复现（n=200/k=43 时约 ±2-3pp 采样噪声）。种子写入产物备查。
    gen_seed = SEED * 10000 + m * 10 + seed
    _set_seed(gen_seed)
    bd_path = os.path.join(HERE, "runs_v2", f"bestdev_m{m}_s{seed}.json")
    bd = json.load(open(bd_path, encoding="utf-8"))
    cp = "base" if m == 0 else os.path.join(HERE, "runs_v2", run_tag(m, seed), bd["best_ckpt"])
    tok, model = _load_model(cp, mem_frac)
    items = dev_items(dev_n)
    ns = [1, 4, 16, n] if n > 16 else [1, 4, n]
    ns = sorted(set(x for x in ns if x <= n))
    hb = HB(f"bon_m{m}_s{seed}", total=len(items),
            phase=f"best-of-{n} verifier 选择 m={m} seed={seed}",
            extra={"ckpt": bd["best_ckpt"], "report_n": ns}, every=1)
    passn = {k: 0 for k in ns}     # oracle 上界
    seln = {k: 0 for k in ns}      # ≡ pass@k（verifier 需 gold 才能挑，故同为 oracle）
    majn = {k: 0 for k in ns}      # 可部署：无 gold 的多数投票
    dump = []                      # 逐题逐样本的执行数值，供离线重算
    B = bon_batch
    for bi in range(0, len(items), B):
        b = items[bi:bi + B]
        enc = tok([make_prompt(tok, r) for r in b], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_PROMPT, padding_side="left").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_COMP, do_sample=True,
                                 temperature=1.0, top_p=0.95, num_return_sequences=n,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        txt = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, r in enumerate(b):
            samples = txt[j * n:(j + 1) * n]
            codes = [extract_code(s) for s in samples]
            _cx = str(r.get("context")) if r.get("context") is not None else None
            rw = [verify(c, r["answer"], True, _cx)["reward"] for c in codes]
            rec = {"gold": str(r["answer"]), "rewards": rw, "vals": [], "reasons": []}
            from verifier import run_program, parse_number, _has_computation, compare
            for c in codes:
                # 协议 Gate 0：「分开统计语法错误、执行错误、超时、空输出、verifier 拒绝」。
                # 原实现把这五类全压成 None，采样侧的失败模式事后不可恢复。
                if not _has_computation(c):
                    rec["vals"].append(None); rec["reasons"].append("guard:constant_program"); continue
                ok, v = run_program(c, _cx)
                if not ok:
                    rec["vals"].append(None); rec["reasons"].append(f"exec:{v}"); continue
                num = parse_number(v)
                rec["vals"].append(num)
                rec["reasons"].append("ok" if num is not None else "unparseable")
            dump.append(rec)
            for k in ns:
                hit = 1.0 if any(x == 1.0 for x in rw[:k]) else 0.0
                passn[k] += hit
                seln[k] += hit
                mv = majority_at_k(codes, k, context=_cx)
                majn[k] += 1.0 if (mv is not None and compare(mv, r["answer"])[0]) else 0.0
        done = min(bi + B, len(items))
        hb.update(done, **{f"maj@{k}": majn[k] / done for k in ns})
    res = {"m": m, "seed": seed, "ckpt": bd["best_ckpt"], "dev_n": len(items),
           "n_samples": n, "gen_seed": gen_seed,
           "pass_at": {str(k): passn[k] / len(items) for k in ns},
           "verifier_selected": {str(k): seln[k] / len(items) for k in ns},
           "majority_at": {str(k): majn[k] / len(items) for k in ns},
           "note": "pass@k 与 sel@k 均为 oracle 上界（需 gold）；maj@k 为无 gold 的可部署基线"}
    json.dump(dump, open(os.path.join(HERE, "runs_v2", f"bondump_m{m}_s{seed}.json"), "w"))
    json.dump(res, open(os.path.join(HERE, "runs_v2", f"bon_m{m}_s{seed}.json"), "w"), indent=1)
    hb.done(**{f"maj@{k}": majn[k] / len(items) for k in ns})
    print(f"BON_DONE m={m} seed={seed} " +
          " ".join(f"maj@{k}={majn[k]/len(items):.4f}(oracle pass@{k}={passn[k]/len(items):.4f})"
                   for k in ns))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "gate0", "warm", "evalckpt", "bon"])
    ap.add_argument("--m", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--dev-n", type=int, default=200)
    ap.add_argument("--eval-batch", type=int, default=16)   # 显存峰值可调（fallback 已关，宁小勿崩）
    ap.add_argument("--bon-batch", type=int, default=4)     # 实际序列数 = bon_batch × n
    ap.add_argument("--mem-frac", type=float, default=0)    # 本进程显存硬上限（占总显存比例）
    a = ap.parse_args()
    if a.cmd == "split": cmd_split()
    elif a.cmd == "gate0": cmd_gate0()
    elif a.cmd == "warm": cmd_warm(a.m, a.seed)
    elif a.cmd == "evalckpt": cmd_evalckpt(a.m, a.seed, a.eval_batch, a.mem_frac or None)
    else: cmd_bon(a.m, a.seed, a.n, a.dev_n, a.bon_batch, a.mem_frac or None)

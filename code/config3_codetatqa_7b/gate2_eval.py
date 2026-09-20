# -*- coding: utf-8 -*-
"""Gate 2 评测：三分支各自选 best-dev，口径与 Gate 1 完全一致。

分支产物结构不同：
  csft / grpo : LoRA adapter，需挂在 merged_m{m}_s{seed} 上
  rssft       : 每轮 merge 出的完整模型 merged_round*

    python gate2_eval.py --branch grpo --m 16 --seed 0
    python gate2_eval.py --all --m 16 --seed 0
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
from gate_v2 import MODEL, MAX_PROMPT, MAX_COMP, CKPT_DEV_N, dev_items, make_prompt
from gate2 import out_dir, warm_dir

BRANCHES = ["grpo", "csft", "rssft"]


def list_points(branch, m, seed, tseed=None):
    """返回 [(名字, 模型路径, 是否完整模型)]，按时间顺序。"""
    od = out_dir(branch, m, seed, tseed)
    if not os.path.isdir(od):
        return []
    if branch == "rssft":
        rounds = sorted([d for d in os.listdir(od) if d.startswith("merged_round")],
                        key=lambda x: int(x.replace("merged_round", "")))
        return [(d, os.path.join(od, d), True) for d in rounds]
    pts = sorted([d for d in os.listdir(od) if d.startswith("checkpoint-")],
                 key=lambda x: int(x.split("-")[1]))
    pts = [(d, os.path.join(od, d), False) for d in pts]
    if os.path.isdir(os.path.join(od, "final")):
        pts.append(("final", os.path.join(od, "final"), False))
    return pts


def load_model(path, full, base_dir, base_cache=None):
    """加速点 A：同一分支的所有 LoRA checkpoint（grpo/csft）共享同一个 base_dir，
    原实现每评一个 checkpoint 就把这个 0.5B base 从磁盘重读一遍。改为按分支缓存
    base 模型 + 其"合并前纯净权重"的一份 GPU 内存快照；每次评新 checkpoint 时，
    先把缓存模型的权重恢复成纯净快照（GPU 内拷贝，不读盘），再合并这个 checkpoint
    的 LoRA delta——计算结果与"每次都从磁盘重新加载 base 再合并"逐位相同，只是
    base 的字节来源从磁盘变成内存。rssft（full=True）每个 checkpoint 本身就是
    不同的完整模型，无 base 可共享，走原路径不变。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if full:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        return tok, model.eval()

    if base_cache is not None and base_cache.get("model") is not None:
        tok, model = base_cache["tok"], base_cache["model"]
        model.load_state_dict(base_cache["pristine_state"], strict=True)
    else:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map="cuda")
        if base_cache is not None:
            base_cache["tok"] = tok
            base_cache["model"] = model
            base_cache["pristine_state"] = {k: v.clone() for k, v in model.state_dict().items()}
    model = PeftModel.from_pretrained(model, path).merge_and_unload()
    return tok, model.eval()


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
        for r, t in zip(b, tok.batch_decode(o[:, enc["input_ids"].shape[1]:],
                                            skip_special_tokens=True)):
            v = verify(extract_code(t), r["answer"], True, str(r.get("context")) if r.get("context") is not None else None)
            hits += v["reward"]
            res.append({"reward": v["reward"], "reason": v["reason"]})
        hb.update(off + min(bi + batch, len(items)), acc=hits / len(res))
    return hits / max(len(res), 1), res


def eval_branch(branch, m, seed, dev_n=CKPT_DEV_N, tseed=None):
    SUF = f"_t{tseed}" if (tseed is not None and tseed != seed) else ""
    pts = list_points(branch, m, seed, tseed)
    if not pts:
        print(f"跳过 {branch}：无产物")
        return
    items = dev_items(dev_n)
    base = warm_dir(m, seed)
    part = os.path.join(HERE, "runs_g2", f"bestdev_partial_{branch}_m{m}_s{seed}{SUF}.json")
    done = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else {}
    hb = HB(f"g2eval_{branch}_m{m}_s{seed}{SUF}", total=len(pts) * len(items),
            phase=f"Gate2 dev 评测 {branch} m={m} seed={seed}",
            extra={"n_points": len(pts), "dev_n": len(items), "已完成": len(done)}, every=1)
    # 逐题结果也增量落盘：否则续跑时 best 点若在中断前评过，per_item 里没有它，
    # per_item_best 会写成 null，配对 McNemar 就只能重烧那个 checkpoint 才能做。
    pit = os.path.join(HERE, "runs_g2", f"peritem_{branch}_m{m}_s{seed}{SUF}.json")
    per_item = json.load(open(pit, encoding="utf-8")) if os.path.exists(pit) else {}
    base_cache = {}   # 加速点 A：本分支内所有 LoRA checkpoint 共用一个已加载的 base
    for i, (name, path, full) in enumerate(pts):
        if name in done:
            hb.update((i + 1) * len(items), skipped=name)
            continue
        tok, model = load_model(path, full, base, base_cache)
        acc, res = greedy_eval(tok, model, items, hb, i * len(items))
        done[name] = acc
        per_item[name] = res
        json.dump(done, open(part, "w"), indent=1)
        json.dump(per_item, open(pit, "w"))
        hb.log(f"  {name} dev_acc={acc:.4f}")
        # 注意：full=False 时 model 与 base_cache["model"] 是同一对象，del 只删局部
        # 引用，缓存不受影响；full=True（rssft）时才是真正释放。
        del model; gc.collect()
        import torch; torch.cuda.empty_cache()
    best = max(done, key=done.get)
    meta_p = os.path.join(out_dir(branch, m, seed, tseed), "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    json.dump({"branch": branch, "m": m, "seed": seed, "dev_n": len(items),
               "per_point": done, "best_point": best, "best_dev_acc": done[best],
               "train_meta": meta,
               "per_item_best": per_item.get(best)},
              open(os.path.join(HERE, "runs_g2", f"bestdev_{branch}_m{m}_s{seed}{SUF}.json"),
                   "w"), indent=1)
    hb.done(best=best, best_dev_acc=done[best])
    print(f"G2EVAL_DONE {branch} m={m} seed={seed}{SUF} best={best} acc={done[best]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seed", type=int, default=None,
                    help="训练种子（协议 Gate 2 要求 3 个）；省略则沿用 --seed")
    ap.add_argument("--dev-n", type=int, default=CKPT_DEV_N)
    a = ap.parse_args()
    for b in (BRANCHES if a.all else [a.branch]):
        if b:
            eval_branch(b, a.m, a.seed, a.dev_n, a.train_seed)

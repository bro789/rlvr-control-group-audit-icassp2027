# -*- coding: utf-8 -*-
"""给带 --train-seed 后缀的 GRPO run 评冻结 test。

final_test_eval.py 只认 (m, seed)，不认 train-seed，所以覆盖率实验（t101/t102）
与难度过滤实验（t201）训完都没法用它出 test 数。本脚本补这一条路径。

口径与 final_test_eval 完全一致：
  - test_items()：同一个 clean 过滤
  - 加载方式：base = warm_dir(m,seed) 的已合并模型，再套该 run 的 best-dev LoRA
  - 判分：verify(extract_code(gen), answer, True, _ctx(r))

输出：runs_test/test_grpo_m{m}_s{seed}_t{ts}.json，含逐项判对向量，
      可直接喂给 analysis/gen_extended_stats.py 的同类分析。

用法：python test_eval_tseed.py --m 64 --seed 1 --train-seed 101
"""
import os, sys, json, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from final_test_eval import test_items, _ctx
from gate2 import out_dir, warm_dir
from gate_v2 import MODEL, make_prompt, MAX_COMP
from gate2_passk import extract_code
from verifier import verify


def main(m, seed, ts, batch=16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    suf = f"_t{ts}" if (ts is not None and ts != seed) else ""
    bd = os.path.join(HERE, "runs_g2", f"bestdev_grpo_m{m}_s{seed}{suf}.json")
    if not os.path.exists(bd):
        sys.exit(f"缺 {bd}（先跑 gate2_eval.py --branch grpo --train-seed {ts}）")
    best = json.load(open(bd, encoding="utf-8"))["best_point"]
    od = out_dir("grpo", m, seed, ts)
    print(f"best-dev 点: {best}\n适配器: {os.path.join(od, best)}", flush=True)

    items = test_items()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(warm_dir(m, seed),
                                                dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, os.path.join(od, best)).merge_and_unload().eval()

    per, t0 = [], time.time()
    for i in range(0, len(items), batch):
        ch = items[i:i + batch]
        enc = tok([make_prompt(tok, r) for r in ch], return_tensors="pt",
                  padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_COMP, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, g in zip(ch, gen):
            try:
                ok = int(verify(extract_code(g), str(r["answer"]), True,
                                _ctx(r))["reward"] == 1.0)
            except Exception:
                ok = 0
            per.append(ok)
        if i % (batch * 5) == 0:
            print(f"  {len(per)}/{len(items)}  当前 {sum(per)/max(len(per),1)*100:.2f}%"
                  f"  {time.time()-t0:.0f}s", flush=True)

    acc = sum(per) / len(per)
    od2 = os.path.join(HERE, "runs_test")
    os.makedirs(od2, exist_ok=True)
    p = os.path.join(od2, f"test_grpo_m{m}_s{seed}{suf}.json")
    json.dump({"m": m, "seed": seed, "train_seed": ts, "n_test": len(items),
               "best_point": best, "acc": acc, "per_item": per},
              open(p, "w", encoding="utf-8"), indent=1)
    print(f"\n冻结 test: {sum(per)}/{len(per)} = {acc*100:.2f}%\n写出 {p}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seed", type=int, required=True)
    a = ap.parse_args()
    main(a.m, a.seed, a.train_seed)

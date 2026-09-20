# -*- coding: utf-8 -*-
"""难度过滤版 GRPO（探索性臂，不进预注册判定量）。

动机：本研究自己的诊断显示，7B 上 70.7-86.1% 的提示组是零方差的——8 条采样奖励
全同，组内相对优势为 0，对梯度毫无贡献，整个 run 只有 50-82 个提示提供过信号。
这是 GRPO 的已知缺陷（DAPO 一系用动态采样解决），也是「你没把 GRPO 调好」这条
批评里唯一有实质内容的部分。

做法分两步，都不改 gate2.py（那是已注册管线，必须保持原样）：

  1. prepass：用 W_m 对题池每题贪心采 1 条，执行后与 gold 比对。
     保留**答错**的题——它们在训练时不会是全对组，因而更可能带优势信号。
     K=1 是成本妥协：K=4 的预扫要约 6 小时，比训练本身还贵。

  2. run：猴子补丁 gate2.pool_questions，把题池换成过滤后的子集，
     再调原版 cmd_grpo。墙钟预算**与已注册那次相同**（5700s），
     所以这个臂隔离的是「同样算力花得更值」，不是「更多算力」。

用法：
    python gate2_dyn.py prepass --m 64 --seed 1
    python gate2_dyn.py run     --m 64 --seed 1 --train-seed 201 --seconds 5700
"""
import os, sys, json, argparse, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import gate2
from gate2 import pool_questions, out_dir
from gate_v2 import MODEL, make_prompt, MAX_COMP
from gate2_passk import extract_code
from verifier import verify

HERE = os.path.dirname(os.path.abspath(__file__))


def filt_path(m, seed):
    return os.path.join(HERE, "runs_g2", f"dynpool_m{m}_s{seed}.json")


def cmd_prepass(m, seed, batch=16):
    """用 W_m 贪心扫一遍题池，记下答错的题的下标。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pool = pool_questions(m, seed)
    print(f"题池 {len(pool)} 题，开始贪心预扫（K=1）", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # warm_dir 是**已合并的完整模型**（见 final_test_eval.load_arm），不是 LoRA adapter。
    # 它就是 GRPO 分叉出去的那个 W_m。
    wd = gate2.warm_dir(m, seed)
    print(f"从 W_m 加载: {wd}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(wd, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    wrong, t0 = [], time.time()
    for i in range(0, len(pool), batch):
        chunk = pool[i:i + batch]
        prompts = [make_prompt(tok, r) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_COMP, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        for j, (r, g) in enumerate(zip(chunk, gen)):
            cx = str(r.get("context")) if r.get("context") is not None else None
            try:
                ok = verify(extract_code(g), str(r["answer"]), True, cx)["reward"] == 1.0
            except Exception:
                ok = False
            if not ok:
                wrong.append(i + j)
        if i % (batch * 10) == 0:
            el = time.time() - t0
            print(f"  {i+len(chunk)}/{len(pool)}  答错 {len(wrong)}  用时 {el:.0f}s",
                  flush=True)

    os.makedirs(os.path.dirname(filt_path(m, seed)), exist_ok=True)
    json.dump({"m": m, "seed": seed, "pool": len(pool), "keep": wrong,
               "kept": len(wrong), "wall": time.time() - t0},
              open(filt_path(m, seed), "w"), indent=1)
    print(f"\n保留 {len(wrong)}/{len(pool)} = {len(wrong)/len(pool)*100:.1f}%"
          f"  -> {filt_path(m, seed)}", flush=True)


def cmd_run(m, seed, seconds, tseed):
    p = filt_path(m, seed)
    if not os.path.exists(p):
        sys.exit(f"缺 {p}，先跑 prepass")
    keep = set(json.load(open(p))["keep"])
    orig = gate2.pool_questions

    def filtered(mm, ss):
        full = orig(mm, ss)
        sub = [r for i, r in enumerate(full) if i in keep]
        print(f"[dyn] 题池 {len(full)} -> {len(sub)}（只留 W_m 答错的）", flush=True)
        return sub

    gate2.pool_questions = filtered
    gate2.cmd_grpo(m, seed, seconds, tseed)
    gate2.pool_questions = orig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepass", "run"])
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=5700)
    ap.add_argument("--train-seed", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "prepass":
        cmd_prepass(a.m, a.seed)
    else:
        cmd_run(a.m, a.seed, a.seconds, a.train_seed)

# -*- coding: utf-8 -*-
"""Gate 2：在 m_L / m_H 上对比 Continued SFT / Iterative RS-SFT / GRPO。

三分支从**同一 warm start**（merged_m{m}_s{seed}）分叉，等 wall-clock 预算。

  csft   --m M --seed S --seconds N   继续在同样 m 条 gold 演示上 SFT（不获新标注）
  rssft  --m M --seed S --seconds N   采样→verifier 筛选正确轨迹→SFT→重复
  grpo   --m M --seed S --seconds N   采样→同一 verifier 给奖励→策略更新
  eval   --run TAG                    评 best-dev（与 Gate 1 完全同口径）

信息约束（PREREG §2）：除指定的 m 条外，**gold 程序对 GRPO 与 RS-SFT 不可见**；
它们只能看到题目 + verifier 信号（verifier 内部用 answer 判定，这是 RLVR 标准设定）。
"""
import os, sys, json, time, random, argparse

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
from gate2_passk import PROMPT_TMPL, extract_code
from hb import HB
from gate_v2 import (MODEL, SEED, MAX_PROMPT, MAX_COMP, CKPT_DEV_N,
                     load_train_rows, get_split, demos_for, dev_items, make_prompt)

N_CKPT = 6
GEN_K = 8              # 每题采样条数：与 GRPO 的 num_generations 对齐，保证 verifier 预算可比


def make_timed_callbacks(seconds, hb, n_ckpt=N_CKPT, early_marks=None):
    """按**时间**均匀存 checkpoint + 到点停训。

    各分支步速差异极大（csft ~0.8 步/秒 vs grpo ~0.02 步/秒），固定 save_steps
    要么存不出、要么爆盘。按时间切分既能保证每支都拿到 n_ckpt 个点，
    也符合等 wall-clock 的语义（checkpoint 落在同样的时间刻度上）。

    early_marks：额外的早期存盘时刻（秒）。csft 的 eff batch(64) > m=16 样本数，
    1 gstep = 1 epoch，PREREG §9.1 预测的过拟合峰值落在几十~几百 epoch（30~500s），
    而均匀 6 点最早在 seconds/6≈950s——整段漏采。加密早期网格只为 best-dev 选点
    提供候选，不改变训练本身（存盘耗时挤占同一墙钟预算，方向对该分支不利，可接受）。
    """
    from transformers import TrainerCallback

    class TimedCtl(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()
            self.marks = sorted(set(list(early_marks or []) +
                                    [seconds * (i + 1) / n_ckpt for i in range(n_ckpt)]))
            self.hit = 0

        def on_step_end(self, args, state, control, **kw):
            el = time.time() - self.t0
            while self.hit < len(self.marks) and el >= self.marks[self.hit]:
                control.should_save = True
                self.hit += 1
            if el > seconds:
                control.should_training_stop = True
                control.should_save = True
            return control

    return TimedCtl()


def warm_dir(m, seed):
    """三分支的共同起点：已 merge 的 W_m（若不存在需先跑 merge_lora.py）。"""
    d = os.path.join(HERE, "runs_v2", f"merged_m{m}_s{seed}")
    if not os.path.isdir(d):
        raise SystemExit(f"共同起点不存在: {d}\n先跑: python merge_lora.py --m {m} --seed {seed}")
    return d


def out_dir(branch, m, seed, tseed=None):
    """产物目录。

    协议把两条种子轴分列：§一「3 套独立的**演示子集**种子」与 Gate 2「每种方法 3 个
    **训练**种子」。原实现用同一个 --seed 同时决定演示子集与训练 RNG，无法在固定
    W_m 的前提下只变训练种子。现拆开：seed=演示子集种子（决定 W_m 与 demos_for），
    tseed=训练种子（决定训练 RNG）。tseed 省略或等于 seed 时沿用旧命名，保证既有
    产物路径不变；不同时加 _t{tseed} 后缀。
    """
    base = f"{branch}_m{m}_s{seed}"
    if tseed is not None and tseed != seed:
        base += f"_t{tseed}"
    return os.path.join(HERE, "runs_g2", base)


def lora_cfg():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])


def pool_questions(m, seed):
    """GRPO / RS-SFT 可用的题目池：demo pool 中**未作为演示给出**的题。

    这些题的 gold program 不可见，只有 verifier 信号可用。
    """
    sp = get_split()
    rows = load_train_rows()
    used = set(sp["subsets"][str(seed)][str(m)])
    return [rows[i] for i in sp["pool_idx"] if i not in used]


# ---------------- 分支 A：Continued SFT ----------------
def cmd_csft(m, seed, seconds, dense_early=False, tseed=None):
    import torch
    from transformers import AutoTokenizer, TrainerCallback
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    # 加密早期网格（几何间隔）：见 make_timed_callbacks 的 early_marks 说明
    EARLY = [30, 60, 120, 240, 480] if dense_early else None

    TS = seed if tseed is None else tseed          # 训练种子（与演示子集种子解耦）
    tag = f"csft_m{m}_s{seed}" + (f"_t{TS}" if TS != seed else "")
    od = out_dir("csft", m, seed, TS)
    items = demos_for(m, seed)          # 仍然只有这 m 条，不获新标注
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = Dataset.from_list([{"prompt": make_prompt(tok, r),
                             "completion": r["program"].strip() + tok.eos_token} for r in items])
    hb = HB(tag, total=seconds, phase=f"Continued SFT m={m} seed={seed}",
            extra={"n_demos": len(items), "budget_s": seconds}, every=1)

    T0 = time.time()

    class Report(TrainerCallback):
        def on_log(s, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                hb.update(min(int(time.time() - T0), seconds),
                          gstep=state.global_step, loss=logs["loss"])

    cfg = SFTConfig(output_dir=od, seed=SEED + TS, bf16=True, report_to="none",
                    num_train_epochs=10000, learning_rate=1e-5,
                    per_device_train_batch_size=8, gradient_accumulation_steps=8,
                    max_length=MAX_PROMPT + MAX_COMP, completion_only_loss=True,
                    logging_steps=1, save_strategy="steps", save_steps=10 ** 9,
                    save_total_limit=N_CKPT + 2 + (len(EARLY) if EARLY else 0),
                    warmup_ratio=0.0, lr_scheduler_type="constant")
    t0 = time.time()
    tr = SFTTrainer(model=warm_dir(m, seed), args=cfg, train_dataset=ds,
                    processing_class=tok, peft_config=lora_cfg(),
                    callbacks=[Report(), make_timed_callbacks(seconds, hb, early_marks=EARLY)])
    tr.train()
    tr.save_model(od + "/final")
    json.dump({"branch": "csft", "m": m, "seed": seed, "demo_seed": seed, "train_seed": TS, "wall_seconds": time.time() - t0,
               "budget_seconds": seconds, "steps": tr.state.global_step,
               "verifier_calls": 0},
              open(od + "/meta.json", "w"), indent=1)
    hb.done(wall_s=round(time.time() - t0, 1), steps=tr.state.global_step)


# ---------------- 分支 B：Iterative RS-SFT ----------------
def cmd_rssft(m, seed, seconds, round_qs=64, max_vcalls=0, tseed=None, budget_mode="both"):
    """采样 → verifier 筛选正确轨迹 → 在其上 SFT → 重复，直到预算耗尽。

    双预算：墙钟 seconds 与 verifier 调用数 max_vcalls，先到者停。
    max_vcalls 默认取同 (m,seed) 下 GRPO 的实测调用数（PREREG §2 要求二者等 verifier 预算），
    故 **RS-SFT 必须在 GRPO 之后跑**。
    """
    import torch, gc
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed as _set_seed
    from datasets import Dataset
    from peft import PeftModel
    from trl import SFTTrainer, SFTConfig
    from concurrent.futures import ThreadPoolExecutor

    # 协议 Gate 0「固定种子后……可复现」：第 1 轮采样发生在任何 Trainer 实例化之前，
    # torch 全局 RNG 此前未被定种（对照 GRPO 经 GRPOConfig(seed=) 由 Trainer 定种）。
    TS = seed if tseed is None else tseed          # 训练种子（与演示子集种子解耦）
    _set_seed(SEED + TS)
    sfx = "" if budget_mode == "both" else f"_bm{budget_mode}"
    tag = f"rssft_m{m}_s{seed}" + (f"_t{TS}" if TS != seed else "") + sfx
    od = out_dir("rssft", m, seed, TS) + sfx
    os.makedirs(od, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    pool = pool_questions(m, seed)
    gold_demos = demos_for(m, seed)     # 初始的 m 条 gold 仍可用（与 csft 同信息起点）
    rng = random.Random(SEED + TS)
    hb = HB(tag, total=seconds, phase=f"Iterative RS-SFT m={m} seed={seed}",
            extra={"pool": len(pool), "round_qs": round_qs, "K": GEN_K}, every=1)

    if not max_vcalls:
        # 对齐**同一训练种子**下的 GRPO 实测 verifier 预算（种子解耦后必须带 TS）
        gm = os.path.join(out_dir("grpo", m, seed, TS), "meta.json")
        if os.path.exists(gm):
            max_vcalls = json.load(open(gm))["verifier_calls"]
            hb.log(f"  verifier 预算对齐 GRPO 实测值: {max_vcalls}")
        else:
            max_vcalls = 10 ** 9
            hb.log("  ⚠ 未找到 GRPO meta，verifier 预算不设限（等预算约束未生效）")

    # 预算模式：协议 §二 的两条配平轴（等 wall-clock / 等 verifier 预算）天然冲突，
    # 因为 RS-SFT 的 vcall 消耗速率远高于 GRPO，两轴不可能同时配平。实测 both 模式下
    # verifier 轴先耗尽，rssft 只吃到 5700s 的 33.6%（1914s）—— 方向上削弱最强学习类
    # 对手。协议原文「**主要**采用等 wall-clock 预算」，故须补 wall 模式敏感性臂。
    if budget_mode == "wall":
        max_vcalls = 10 ** 12
        hb.log(f"  预算模式=wall（敏感性臂）：放开 verifier 上限，只受 {seconds}s 墙钟约束")
    elif budget_mode == "vcalls":
        seconds = 10 ** 9
        hb.log(f"  预算模式=vcalls：放开墙钟上限，只受 {max_vcalls} 次 verifier 预算约束")
    else:
        hb.log(f"  预算模式=both：min(墙钟 {seconds}s, verifier {max_vcalls})，先到者停")

    cur = warm_dir(m, seed)             # 当前模型路径（每轮更新）
    t0, rnd, vcalls, kept_total = time.time(), 0, 0, 0
    pool_exec = ThreadPoolExecutor(max_workers=12)
    gen_tok = 0          # 协议 §二：生成 token 计量（原实现完全没有）

    while time.time() - t0 < seconds and vcalls < max_vcalls:
        rnd += 1
        # --- 采样 ---
        model = AutoModelForCausalLM.from_pretrained(cur, dtype=torch.bfloat16,
                                                     device_map="cuda").eval()
        qs = rng.sample(pool, min(round_qs, len(pool)))
        kept = []
        # 加速点 B：原 B=4（每次 generate 只出 4×GEN_K=32 条序列），实测显存远未吃满
        # （rssft 运行中仅占约 6-9GB / 32.6GB），保守翻倍到 8（64 条/次），减少
        # generate() 调用次数与 kernel 调度开销。采样内容本身不变（仍是同样的
        # temperature/top_p/num_return_sequences），只是 batch 边界不同——由于
        # PyTorch 随机数按 batch 维度消费，同一 seed 在新旧 batch size 下不会逐位
        # 复现，但这与"多种子本就有采样方差"是同一类可接受差异，不是 bug。
        B = 2   # 7B 专用：8 会 OOM（0.5B 时的注释不适用于 7B）
        for i in range(0, len(qs), B):
            if time.time() - t0 > seconds or vcalls >= max_vcalls:
                break
            b = qs[i:i + B]
            enc = tok([make_prompt(tok, r) for r in b], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_PROMPT, padding_side="left").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAX_COMP, do_sample=True,
                                   temperature=1.0, top_p=0.95, num_return_sequences=GEN_K,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen_tok += int(o.shape[0]) * int(o.shape[1] - enc["input_ids"].shape[1])
            txt = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, r in enumerate(b):
                cands = [extract_code(s) for s in txt[j * GEN_K:(j + 1) * GEN_K]]
                futs = [pool_exec.submit(verify, c, r["answer"], True, str(r.get("context")) if r.get("context") is not None else None) for c in cands]
                vcalls += len(cands)
                # 保留该题**所有**通过 verifier 的轨迹（按规范化文本去重）。
                # 原实现每题只留第一条，丢弃约 7/8 已验证数据 —— 那会人为削弱 RS-SFT，
                # 而 RS-SFT 是 PREREG §5 的决胜对照，削弱它的方向恰好利于 GRPO 判 Go。
                seen_txt = set()
                for c, f in zip(cands, futs):
                    if f.result()["reward"] != 1.0:
                        continue
                    key = " ".join(c.split())
                    if key in seen_txt:
                        continue
                    seen_txt.add(key)
                    kept.append({"prompt": make_prompt(tok, r),
                                 "completion": c.strip() + tok.eos_token})
            hb.update(min(int(time.time() - t0), seconds), round=rnd,
                      sampled=i + len(b), kept=len(kept), vcalls=vcalls)
        del model; gc.collect(); torch.cuda.empty_cache()

        if not kept:
            hb.log(f"  轮 {rnd}: 无正确轨迹，跳过训练")
            continue
        kept_total += len(kept)
        # --- 在 gold 演示 + 筛出的正确轨迹上 SFT ---
        data = [{"prompt": make_prompt(tok, r), "completion": r["program"].strip() + tok.eos_token}
                for r in gold_demos] + kept
        rd = os.path.join(od, f"round{rnd}")
        # eff batch 16（原 64）+ 4 epochs：每轮数十条样本时能拿到 10+ 次梯度更新，
        # 而非原来的 1-2 次（对照：GRPO 全程 135 次）。
        cfg = SFTConfig(output_dir=rd, seed=SEED + TS, bf16=True, report_to="none",
                        num_train_epochs=4, learning_rate=1e-5,
                        per_device_train_batch_size=1, gradient_accumulation_steps=16,  # 7B: 4 会 OOM
                        max_length=MAX_PROMPT + MAX_COMP, completion_only_loss=True,
                        logging_steps=5, save_strategy="no", lr_scheduler_type="constant")
        tr = SFTTrainer(model=cur, args=cfg, train_dataset=Dataset.from_list(data),
                        processing_class=tok, peft_config=lora_cfg())
        tr.train()
        tr.save_model(rd + "/final")
        # merge 出下一轮的起点：保持"当前模型"始终是完整权重，便于下一轮直接采样
        merged = os.path.join(od, f"merged_round{rnd}")
        base = AutoModelForCausalLM.from_pretrained(cur, dtype=torch.bfloat16, device_map="cpu")  # 7B: merge 在 CPU 做
        PeftModel.from_pretrained(base, rd + "/final").merge_and_unload().save_pretrained(
            merged, safe_serialization=True)
        tok.save_pretrained(merged)
        cur = merged
        # 保留所有轮的 merged：它们是 RS-SFT 的 best-dev 选点集合。
        # 只留 2 个会让它相对 grpo/csft（各 6-7 个点）在 max 选择上吃亏。
        hb.log(f"  轮 {rnd}: kept={len(kept)} 累计={kept_total} vcalls={vcalls}/{max_vcalls} -> {merged}")
        del base; gc.collect(); torch.cuda.empty_cache()

    json.dump({"branch": "rssft", "m": m, "seed": seed, "demo_seed": seed, "train_seed": TS, "wall_seconds": time.time() - t0,
               "budget_seconds": seconds, "max_vcalls": max_vcalls, "rounds": rnd,
               "kept_total": kept_total, "verifier_calls": vcalls, "final_model": cur,
               "gen_tokens": gen_tok, "budget_mode": budget_mode,
               "budget_binding": "vcalls" if vcalls >= max_vcalls else "wall",
               "wall_frac_used": (time.time() - t0) / seconds},
              open(od + "/meta.json", "w"), indent=1)
    hb.done(rounds=rnd, kept=kept_total, vcalls=vcalls)


# ---------------- 分支 C：GRPO ----------------
def cmd_grpo(m, seed, seconds, tseed=None):
    import torch
    from transformers import AutoTokenizer, TrainerCallback
    from datasets import Dataset
    from trl import GRPOTrainer, GRPOConfig
    from concurrent.futures import ThreadPoolExecutor

    TS = seed if tseed is None else tseed          # 训练种子（与演示子集种子解耦）
    tag = f"grpo_m{m}_s{seed}" + (f"_t{TS}" if TS != seed else "")
    od = out_dir("grpo", m, seed, TS)
    tok = AutoTokenizer.from_pretrained(MODEL)
    pool = pool_questions(m, seed)
    ds = Dataset.from_list([{"prompt": make_prompt(tok, r), "gold": str(r["answer"]),
                            "context": str(r["context"])}
                            for r in pool])
    hb = HB(tag, total=seconds, phase=f"GRPO m={m} seed={seed}",
            extra={"pool": len(pool), "K": GEN_K}, every=1)

    ex = ThreadPoolExecutor(max_workers=12)
    # 协议 §二：「同时记录生成 token、优化 token、verifier 调用数和真实 GPU 时间」。
    # 原实现只记 verifier 调用数。gen_tok 在此累加；opt_tok 训练结束后从 trainer state 取。
    # 另按 Gate 0「全零奖励组、全一奖励组比例需要报告」逐组统计（组 = GEN_K 个 rollout）。
    st = {"n": 0, "pos": 0, "gen_tok": 0, "grp": 0, "grp_all0": 0, "grp_all1": 0}
    # Gate 0「训练 reward 与离线 verifier 逐样本一致」：落盘 (completion, reward) 供事后比对。
    # 注意：od 此刻还不存在（GRPOTrainer 要到实例化时才按 output_dir 建目录），
    # 必须先 makedirs —— 否则这里直接 FileNotFoundError。
    os.makedirs(od, exist_ok=True)
    rdump = open(os.path.join(od, "reward_dump.jsonl"), "w", encoding="utf-8")

    def reward_fn(completions, gold=None, context=None, **kw):
        # context：表格型任务（CodeTAT-QA）需注入 df 才能执行 gold/生成程序。
        # TRL 把 dataset 里除 prompt 外的列作为 kwargs 传入，故 dataset 加了 context 列。
        # CodeFinQA 无此列 → context=None → verify 行为与注入前一致（向后兼容）。
        codes = [extract_code(c) for c in completions]
        ctxs = context if context is not None else [None] * len(codes)
        futs = [ex.submit(verify, c, g, True, cx) for c, g, cx in zip(codes, gold, ctxs)]
        r = [f.result()["reward"] for f in futs]
        st["n"] += len(r); st["pos"] += sum(r)
        st["gen_tok"] += sum(len(tok.encode(c)) for c in completions)
        for i in range(0, len(r), GEN_K):          # 逐组统计零方差组
            grp = r[i:i + GEN_K]
            if len(grp) < GEN_K:
                continue
            st["grp"] += 1
            if all(x == 0.0 for x in grp): st["grp_all0"] += 1
            elif all(x == 1.0 for x in grp): st["grp_all1"] += 1
        for c, g, x, cx in zip(codes, gold, r, ctxs):
            # dump 带 context，否则离线重放（gate0_audit reward-consistency）缺 df 必然全不一致
            rdump.write(json.dumps({"code": c, "gold": g, "reward": x, "context": cx},
                                   ensure_ascii=False) + "\n")
        rdump.flush()
        return r

    T0 = time.time()

    class Report(TrainerCallback):
        def on_log(s, args, state, control, logs=None, **kw):
            if logs and "reward" in logs:
                hb.update(min(int(time.time() - T0), seconds), gstep=state.global_step,
                          reward=logs["reward"], kl=logs.get("kl", 0),
                          pos_rate=st["pos"] / max(st["n"], 1))

    cfg = GRPOConfig(output_dir=od, seed=SEED + TS, bf16=True, report_to="none",
                     max_steps=100000, learning_rate=1e-5, beta=0.04,
                     num_generations=GEN_K, per_device_train_batch_size=4,
                     gradient_accumulation_steps=16, generation_batch_size=8,
                     max_completion_length=MAX_COMP, temperature=1.0, top_p=0.95,
                     logging_steps=1, save_strategy="steps", save_steps=10 ** 9,
                     save_total_limit=N_CKPT + 2)
    t0 = time.time()
    tr = GRPOTrainer(model=warm_dir(m, seed), args=cfg, train_dataset=ds,
                     processing_class=tok, reward_funcs=[reward_fn],
                     peft_config=lora_cfg(),
                     callbacks=[Report(), make_timed_callbacks(seconds, hb)])
    tr.train()
    tr.save_model(od + "/final")
    rdump.close()
    json.dump({"branch": "grpo", "m": m, "seed": seed, "demo_seed": seed, "train_seed": TS, "wall_seconds": time.time() - t0,
               "budget_seconds": seconds, "steps": tr.state.global_step,
               "verifier_calls": st["n"], "verifier_pos_rate": st["pos"] / max(st["n"], 1),
               "gen_tokens": st["gen_tok"],
               "opt_tokens": getattr(tr.state, "num_input_tokens_seen", None),
               "groups": st["grp"], "groups_all_zero": st["grp_all0"],
               "groups_all_one": st["grp_all1"],
               "group_zero_var_frac": (st["grp_all0"] + st["grp_all1"]) / max(st["grp"], 1)},
              open(od + "/meta.json", "w"), indent=1)
    hb.done(wall_s=round(time.time() - t0, 1), steps=tr.state.global_step,
            vcalls=st["n"], pos_rate=st["pos"] / max(st["n"], 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["csft", "rssft", "grpo"])
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=5700)
    ap.add_argument("--round-qs", type=int, default=64)
    ap.add_argument("--max-vcalls", type=int, default=0, help="0=自动读 GRPO 实测值")
    ap.add_argument("--budget-mode", choices=["both","wall","vcalls"], default="both",
                    help="rssft 预算轴：both=先到者停；wall=只受墙钟（协议主口径的敏感性臂）；vcalls=只受 verifier")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="训练种子（协议 Gate 2 要求 3 个）；省略则沿用演示子集种子 --seed")
    ap.add_argument("--dense-early", action="store_true",
                    help="csft 专用：加密早期 checkpoint 网格（30/60/120/240/480s），补 §9.1 峰值区间")
    a = ap.parse_args()
    if a.cmd == "csft": cmd_csft(a.m, a.seed, a.seconds, a.dense_early, a.train_seed)
    elif a.cmd == "rssft": cmd_rssft(a.m, a.seed, a.seconds, a.round_qs, a.max_vcalls, a.train_seed, a.budget_mode)
    else: cmd_grpo(a.m, a.seed, a.seconds, a.train_seed)

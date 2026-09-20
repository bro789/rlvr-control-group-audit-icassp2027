# -*- coding: utf-8 -*-
"""vLLM 等价性 gate：换推理引擎前必须证明逐题等价，不能假设。

背景：Gate 1 全部用 HF generate 产出。若 Gate 2 换 vLLM，两批结果跨引擎不可比——
这本身就是精度问题。故先在同一 checkpoint 上对拍。

设计要点：**生成与验证分离**。
  gen 阶段只产生文本（HF 在 Windows、vLLM 在 WSL 各自跑）；
  cmp 阶段用同一个 verifier 验证两份文本。
  否则 WSL/Windows 的 Python 与依赖差异会混进结果，测的就不是引擎差异了。

    # Windows
    python vllm_equiv.py gen --engine hf   --ckpt runs_v2/merged_m1024_s0 --n 200
    # WSL（vllm 环境）
    python vllm_equiv.py gen --engine vllm --ckpt /mnt/e/.../merged_m1024_s0 --n 200
    # Windows
    python vllm_equiv.py cmp

判据：
    100%  一致              → 可换
    ≥99%  且 McNemar 不显著 → 可换，但须在论文中声明引擎差异
    <99%  或 McNemar 显著   → 不换
"""
import os, sys, json, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MAX_PROMPT, MAX_COMP = 3072, 256
PROMPT_TMPL = None      # 延迟导入：WSL 侧也要能跑


def _tmpl():
    global PROMPT_TMPL
    if PROMPT_TMPL is None:
        from gate2_passk import PROMPT_TMPL as T
        PROMPT_TMPL = T
    return PROMPT_TMPL


def load_dev(n):
    """与 gate_v2.dev_items 完全一致的取法，保证对拍同一批题。"""
    sp = json.load(open(os.path.join(HERE, "data", "v2_split.json"), encoding="utf-8"))
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "CodeFinQA__train.jsonl"),
                                        encoding="utf-8")]
    return [rows[i] for i in sp["dev_idx"]][:n]


def build_prompts(tok, items):
    out = []
    for r in items:
        msgs = [{"role": "user",
                 "content": _tmpl().format(q=r["question"], ctx=str(r["context"])[:5000])}]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def cmd_gen(engine, ckpt, n, out_path):
    from transformers import AutoTokenizer
    items = load_dev(n)
    tok = AutoTokenizer.from_pretrained(ckpt)
    prompts = build_prompts(tok, items)
    # prompt 指纹：两侧必须完全一致，否则对拍的不是同一输入
    fp = hashlib.sha256("".join(prompts).encode("utf-8")).hexdigest()[:16]
    print(f"engine={engine} n={len(prompts)} prompt_sha={fp}", flush=True)

    if engine == "vllm":
        from vllm import LLM, SamplingParams
        # enforce_eager：WSL 里没有 CUDA toolkit(nvcc)，跳过 inductor JIT 编译。
        # PagedAttention 与 continuous batching 仍生效，主要收益不受影响。
        llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=MAX_PROMPT + MAX_COMP,
                  enforce_eager=os.environ.get("VLLM_EAGER", "1") == "1")
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_COMP)   # greedy，对齐 HF do_sample=False
        texts = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    else:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16,
                                                     device_map="cuda").eval()
        texts, B = [], 16
        for i in range(0, len(prompts), B):
            enc = tok(prompts[i:i + B], return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_PROMPT, padding_side="left").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAX_COMP, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            texts += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  [{min(i+B,len(prompts))}/{len(prompts)}]", flush=True)

    json.dump({"engine": engine, "ckpt": ckpt, "n": len(texts), "prompt_sha": fp,
               "texts": texts, "gold": [str(r["answer"]) for r in items]},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"GEN_DONE {engine} -> {out_path}", flush=True)


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    chi = (abs(b10 - b01) - 1) ** 2 / (b10 + b01) if (b10 + b01) > 0 else 0.0
    return b10, b01, chi


def cmd_cmp(pa, pb):
    """用同一个 verifier 验证两份文本 —— 这是本脚本的关键：排除环境差异。"""
    from verifier import verify
    from gate2_passk import extract_code
    A = json.load(open(pa, encoding="utf-8"))
    B = json.load(open(pb, encoding="utf-8"))
    assert A["n"] == B["n"], f"题数不一致 {A['n']} vs {B['n']}"
    if A["prompt_sha"] != B["prompt_sha"]:
        print(f"⚠ prompt 指纹不一致！{A['prompt_sha']} vs {B['prompt_sha']}")
        print("  两侧输入就不同，对拍结果无意义。先排查 chat template / tokenizer。")
        return
    print(f"prompt 指纹一致: {A['prompt_sha']}  n={A['n']}")

    ra = [int(verify(extract_code(t), g)["reward"] == 1.0) for t, g in zip(A["texts"], A["gold"])]
    rb = [int(verify(extract_code(t), g)["reward"] == 1.0) for t, g in zip(B["texts"], B["gold"])]
    na, nb = A["engine"], B["engine"]

    agree = sum(1 for x, y in zip(ra, rb) if x == y)
    only_a, only_b, chi = mcnemar(ra, rb)
    # 文本层面的完全一致率（比 reward 一致更严格）
    same_text = sum(1 for x, y in zip(A["texts"], B["texts"]) if x.strip() == y.strip())

    out = {"a": na, "b": nb, "n": len(ra),
           "acc_a": sum(ra) / len(ra), "acc_b": sum(rb) / len(rb),
           "reward_agreement": agree / len(ra), "n_disagree": len(ra) - agree,
           "text_identical_rate": same_text / len(ra),
           f"only_{na}_correct": only_a, f"only_{nb}_correct": only_b,
           "mcnemar_chi2": chi}

    print("=" * 72)
    print(f"引擎等价性对拍  n={len(ra)}")
    print(f"  {na:>5} acc = {out['acc_a']*100:6.2f}%")
    print(f"  {nb:>5} acc = {out['acc_b']*100:6.2f}%")
    print(f"  逐题 reward 一致率 = {out['reward_agreement']*100:.2f}%  （不一致 {out['n_disagree']} 题）")
    print(f"  生成文本完全相同率 = {out['text_identical_rate']*100:.2f}%")
    print(f"  只有 {na} 对 = {only_a}   只有 {nb} 对 = {only_b}   McNemar chi2 = {chi:.2f}")

    if out["reward_agreement"] == 1.0:
        v = "✅ reward 完全等价，可换引擎"
    elif out["reward_agreement"] >= 0.99 and chi < 3.84:
        v = "⚠ 高度一致（≥99%）且无系统性偏移，可换但须声明引擎差异"
    else:
        v = "❌ 不等价，不可换引擎"
    out["verdict"] = v
    print(f"  判决: {v}")
    print("=" * 72)

    # 翻转样本存档，便于人工检查差异性质
    flips = [{"i": i, "gold": A["gold"][i], na: A["texts"][i][:400], nb: B["texts"][i][:400]}
             for i in range(len(ra)) if ra[i] != rb[i]][:20]
    out["flips"] = flips
    json.dump(out, open(os.path.join(HERE, "vllm_equiv_result.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"详情写入 vllm_equiv_result.json（含 {len(flips)} 条翻转样本）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "cmp"])
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--ckpt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--a", default=os.path.join(HERE, "equiv_texts_hf.json"))
    ap.add_argument("--b", default=os.path.join(HERE, "equiv_texts_vllm.json"))
    a = ap.parse_args()
    if a.cmd == "gen":
        cmd_gen(a.engine, a.ckpt, a.n,
                a.out or os.path.join(HERE, f"equiv_texts_{a.engine}.json"))
    else:
        cmd_cmp(a.a, a.b)

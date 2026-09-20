# -*- coding: utf-8 -*-
"""Gate 2：Qwen2.5-0.5B/1.5B 在 CodeFinQA gold 自洽子集上的 pass@1(greedy) 与 pass@64(采样)。

预注册判据（PREREG.md）：两档模型 pass@64 均 <5% → kill；pass@1 均 >60% → kill。
用法：python gate2_passk.py <model_path_or_id> <n_items> <k>
输出：gate2_<model短名>.json（逐题 pass 计数 + 汇总），顺带记录解模板多样性与 pass@k 方差。
"""
import os, sys, json, random, re, io
os.environ.setdefault("HF_HOME", r"E:\hf_cache")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from verifier import verify

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42

PROMPT_TMPL = """You are a financial analyst. The context is a table given as a Python dict-of-dicts, already loaded into a variable `df`, so `df["column label"]["row label"]` returns a cell value. Answer the question by writing a short Python program that reads cells from `df` and ends with the final numeric result assigned to a variable named `answer`. If the question asks for a percentage, `answer` should be the percentage value (e.g. 12.5 for 12.5%). Output ONLY the Python code, no explanations.

Example 1:
Question: What was the percentage change in revenue from 2018 to 2019?
Context: {{"Revenue": {{"2019": 550, "2018": 500}}}}
Code:
a = df["Revenue"]["2019"]
b = df["Revenue"]["2018"]
answer = (a - b) / b * 100.0

Example 2:
Question: What was the total operating expenses for 2018 and 2019?
Context: {{"Operating expenses": {{"2019": 120.5, "2018": 88.3}}}}
Code:
a = df["Operating expenses"]["2019"]
b = df["Operating expenses"]["2018"]
answer = a + b

Question: {q}
Context: {ctx}
Code:
"""

def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.+?)```", text, re.S)
    code = m.group(1) if m else text
    # 截到最后一个 answer 赋值行为止
    lines, out, seen = code.splitlines(), [], False
    for ln in lines:
        out.append(ln)
        if re.match(r"\s*answer\s*=", ln):
            seen = True
            break
    return "\n".join(out) if seen else code.strip()

def load_eval_items(n_items):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "CodeTAT-QA__test.jsonl"), encoding="utf-8")]
    clean = [r for r in rows if verify(r["program"], r["answer"], True, str(r.get("context")) if r.get("context") is not None else None)["reward"] == 1.0]
    random.Random(SEED).shuffle(clean)
    return clean[:n_items]

def main(model_id, n_items=200, k=64):
    short = model_id.rstrip("/").split("/")[-1]
    items = load_eval_items(n_items)
    print(f"eval items: {len(items)} | model: {model_id}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def build_input(r):
        ctx = str(r["context"])[:6000]
        msgs = [{"role": "user", "content": PROMPT_TMPL.format(q=r["question"], ctx=ctx)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [build_input(r) for r in items]
    results = []
    BATCH = 16

    def gen(prompt_batch, do_sample, num_return=1):
        enc = tok(prompt_batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096, padding_side="left").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=256,
                do_sample=do_sample, temperature=1.0 if do_sample else None,
                top_p=0.95 if do_sample else None,
                num_return_sequences=num_return,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return texts

    pool = ThreadPoolExecutor(max_workers=8)

    for bi in range(0, len(items), BATCH):
        batch_items = items[bi:bi + BATCH]
        batch_prompts = prompts[bi:bi + BATCH]
        # greedy pass@1
        greedy = gen(batch_prompts, do_sample=False)
        # 采样 k 次（分组生成控制显存）
        samples = [[] for _ in batch_items]
        per_call = 8
        for _ in range(k // per_call):
            texts = gen(batch_prompts, do_sample=True, num_return=per_call)
            for j in range(len(batch_items)):
                samples[j].extend(texts[j * per_call:(j + 1) * per_call])
        # 并行验证
        for j, r in enumerate(batch_items):
            gold = r["answer"]
            g_res = verify(extract_code(greedy[j]), gold, True, str(r.get("context")) if r.get("context") is not None else None)
            futs = [pool.submit(verify, extract_code(s), gold) for s in samples[j]]
            hits = [f.result()["reward"] for f in futs]
            n_hit = int(sum(hits))
            # 解模板多样性：命中程序的 AST 归一化形态数
            forms = set()
            for s, h in zip(samples[j], hits):
                if h:
                    forms.add(re.sub(r"-?\d[\d,]*\.?\d*", "N", extract_code(s))[:200])
            results.append({
                "idx": bi + j, "greedy": g_res["reward"], "hits": n_hit, "k": k,
                "distinct_correct_forms": len(forms),
            })
        done = len(results)
        p1 = sum(x["greedy"] for x in results) / done
        pk = sum(1 for x in results if x["hits"] > 0) / done
        print(f"[{done}/{len(items)}] pass@1={p1:.3f} pass@{k}={pk:.3f}", flush=True)

    p1 = sum(x["greedy"] for x in results) / len(results)
    pk = sum(1 for x in results if x["hits"] > 0) / len(results)
    import statistics
    phat = [x["hits"] / x["k"] for x in results]
    summary = {
        "model": model_id, "n": len(results), "k": k,
        "pass@1_greedy": round(p1, 4), f"pass@{k}": round(pk, 4),
        "mean_success_rate": round(sum(phat) / len(phat), 4),
        "success_rate_variance": round(statistics.pvariance(phat), 6),
        "mean_distinct_correct_forms": round(
            sum(x["distinct_correct_forms"] for x in results) / len(results), 3),
        "kill_pass64_lt_5pct": pk < 0.05, "kill_pass1_gt_60pct": p1 > 0.60,
    }
    out = {"summary": summary, "per_item": results}
    with open(os.path.join(HERE, f"gate2_{short}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False))
    print("GATE2_DONE")

if __name__ == "__main__":
    mid = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    kk = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    main(mid, n, kk)

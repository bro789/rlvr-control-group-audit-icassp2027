# -*- coding: utf-8 -*-
"""把 LoRA adapter merge 进 base，导出完整模型供 vLLM 加载。

    python merge_lora.py --m 1024 --seed 0            # 用该档位的 best-dev ckpt
    python merge_lora.py --ckpt <path> --out <dir>
"""
import os, sys, json, argparse

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main(ckpt, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print(f"base={MODEL}\nadapter={ckpt}\nout={out}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, ckpt).merge_and_unload()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("MERGE_DONE", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.ckpt:
        ckpt, out = a.ckpt, a.out or (a.ckpt.rstrip("/\\") + "_merged")
    else:
        bd = json.load(open(os.path.join(HERE, "runs_v2", f"bestdev_m{a.m}_s{a.seed}.json"),
                            encoding="utf-8"))
        ckpt = os.path.join(HERE, "runs_v2", f"warm_m{a.m}_s{a.seed}", bd["best_ckpt"])
        out = a.out or os.path.join(HERE, "runs_v2", f"merged_m{a.m}_s{a.seed}")
        print(f"best_ckpt={bd['best_ckpt']}  dev_acc={bd['best_dev_acc']:.4f}")
    main(ckpt, out)

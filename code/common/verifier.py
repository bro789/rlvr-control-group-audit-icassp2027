# -*- coding: utf-8 -*-
"""执行型 verifier：沙箱运行候选 Python 程序，与 gold 数值按容差比对。

预注册守卫（PREREG.md Gate 3）：
  G1 反常量守卫——程序不含任何算术运算/变量引用、直接输出字面量 → 拒绝；
  G2 容差 = 1% 相对误差（|pred-gold| <= 0.01*|gold|），gold=0 时用绝对误差 1e-6；
  G3 百分比/小数规范化——pred 与 gold*100 或 gold/100 在容差内时按"单位歧义"单独记录，
     主判定仍为失败（防 percent-vs-decimal 撞窗口），但敏感性分析可复算；
  G4 超时 5s / 禁 import 网络与文件系统模块。
"""
import ast, math, re, subprocess, sys, tempfile, os, json

TIMEOUT = 5
BANNED_IMPORTS = {"os", "sys", "subprocess", "socket", "urllib", "requests",
                  "pathlib", "shutil", "http", "ftplib", "pickle"}

def _has_computation(code: str) -> bool:
    """G1：程序必须包含实际计算（算术运算或对上下文数值的引用），拒绝纯常量输出。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
           and node.func.id in {"sum", "min", "max", "abs", "round", "pow", "len"}:
            return True
    return False

def _static_check(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax:{e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) \
                    else [(node.module or "").split(".")[0]]
            if any(n in BANNED_IMPORTS for n in names):
                return False, f"banned_import:{names}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
           and node.func.id in {"open", "exec", "eval", "__import__", "input"}:
            return False, f"banned_call:{node.func.id}"
    return True, "ok"

def run_program(code: str, context=None):
    """沙箱执行，返回 (ok, value_or_reason)。程序须把结果放入变量 answer 或 print 出最后数值。

    context：表格型任务（CodeTAT-QA）的上下文。程序引用 `df["列"]["行"]` 取数，
    执行前注入 `df = <context 原文>`。context 是 JSON **字符串**，裸文本拼接即可
    （dict-of-dicts 上 df[a][b] 与 pandas 同语法，无需 pandas；实测 gold 自洽 100%）。
    None 时行为与注入前**逐位相同**，不影响 CodeFinQA 既有结果。

    注意：静态守卫只针对模型生成的 `code`，不含注入的数据——否则大 dict 字面量
    里的数字可能让"常数程序"蒙混过 G1 反常量守卫。
    """
    ok, reason = _static_check(code)
    if not ok:
        return False, reason
    exec_code = code if context is None else (f"df = {context}\n" + code)
    harness = exec_code + "\n\n_result = None\ntry:\n    _result = answer\nexcept NameError:\n    pass\nif _result is not None:\n    print('__ANS__', repr(_result))\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(harness)
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], capture_output=True,
                           text=True, timeout=TIMEOUT)
        out = p.stdout.strip().splitlines()
        # 优先取 __ANS__ 标记，否则取最后一行可解析数值
        for line in reversed(out):
            if line.startswith("__ANS__"):
                return True, line.split("__ANS__", 1)[1].strip()
        for line in reversed(out):
            m = re.search(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?%?", line)
            if m:
                return True, m.group(0)
        return False, f"no_numeric_output(rc={p.returncode},err={p.stderr[-200:] if p.stderr else ''})"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        try: os.unlink(path)
        except OSError: pass

def parse_number(s):
    if s is None: return None
    if isinstance(s, (int, float)): return float(s)
    s = str(s).strip().strip("'\"")
    neg = s.startswith("(") and s.endswith(")")  # 会计负数
    s = s.strip("()").replace(",", "").replace("$", "").strip()
    mult = 1.0
    if s.endswith("%"): s = s[:-1]
    low = s.lower()
    for suf, m in (("billion", 1e9), ("million", 1e6), ("thousand", 1e3), ("bn", 1e9), ("mm", 1e6)):
        if low.endswith(suf):
            s, mult = low[: -len(suf)].strip(), m
            break
    try:
        v = float(s) * mult
        return -v if neg else v
    except ValueError:
        return None

def compare(pred, gold, rel_tol=0.01):
    p, g = parse_number(pred), parse_number(gold)
    if p is None or g is None:
        return False, "unparseable"
    if g == 0:
        return abs(p) <= 1e-6, "abs_zero"
    if abs(p - g) <= rel_tol * abs(g):
        return True, "match"
    # G3：单位歧义单独记录，主判定失败
    for scale, tag in ((100.0, "pct_vs_dec"), (0.01, "dec_vs_pct")):
        if abs(p * scale - g) <= rel_tol * abs(g):
            return False, f"unit_ambiguous:{tag}"
    return False, "mismatch"

def verify(code: str, gold, require_computation=True, context=None):
    """完整 verifier：返回 dict(reward, reason)。

    context 见 run_program：表格型任务传该题的 context 原文，None 时行为不变。
    G1 反常量守卫只看模型生成的 code，不看注入数据。
    """
    if require_computation and not _has_computation(code):
        return {"reward": 0.0, "reason": "guard:constant_program"}
    ok, val = run_program(code, context)
    if not ok:
        return {"reward": 0.0, "reason": f"exec:{val}"}
    match, why = compare(val, gold)
    return {"reward": 1.0 if match else 0.0, "reason": why, "pred": val}

if __name__ == "__main__":
    # 自检：G1 反常量、G2 容差、G3 单位歧义
    tests = [
        ("answer = 5.0", 5.0, 0.0),                       # 常量程序 → 拒
        ("a=10\nb=2\nanswer=a/b", 5.0, 1.0),              # 正常计算 → 过
        ("a=10\nb=2\nanswer=a/b*1.005", 5.0, 1.0),        # 容差内 → 过
        ("a=10\nb=2\nanswer=a/b*1.02", 5.0, 0.0),         # 容差外 → 拒
        ("a=1\nb=20\nanswer=a/b", 0.05, 1.0),             # 0.05 vs 0.05 → 过
        ("a=1\nb=20\nanswer=a/b*100", 0.05, 0.0),         # 5 vs 0.05 单位歧义 → 拒
        ("import os\nanswer=os.getpid()", 1, 0.0),        # 禁 import → 拒
    ]
    fails = 0
    for code, gold, want in tests:
        r = verify(code, gold)
        status = "OK " if r["reward"] == want else "FAIL"
        if r["reward"] != want: fails += 1
        print(f"[{status}] want={want} got={r['reward']} reason={r['reason']} | {code[:40]!r}")
    print("SELFTEST", "PASSED" if fails == 0 else f"FAILED({fails})")

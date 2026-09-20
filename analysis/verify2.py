# -*- coding: utf-8 -*-
"""第二轮核对：7B dev-vs-test 逐组、消融、McNemar、C2 剂量、偏差计数、dev 大小、扫描产物。"""
import os, sys, json, glob, math, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 脚本在 analysis/ 下，下面的相对路径按项目根解析
BK = "results_backup"
def load(p): return json.load(open(p, encoding="utf-8"))

print("=== 1) 7B 逐组 dev Δ vs test Δ（文：每组 test 比 dev 更负）===")
c3g = os.path.join(BK, "config3_7b", "runs_g2")
c3v = os.path.join(BK, "config3_7b", "runs_v2")
print("  bestdev 分支文件:", sorted(os.path.basename(x) for x in glob.glob(os.path.join(c3g,"bestdev_*_m*.json")))[:14])
sol7 = glob.glob(os.path.join(c3v, "solver_m*.json"))
sdev = load(sol7[0])["variants"]["scarce"]["acc"] if sol7 else 0.72
print(f"  7B solver dev = {sdev}")
claim_holds = []
for m in [2, 64]:
    for s in [0, 1, 2]:
        def bd(br):
            p = os.path.join(c3g, f"bestdev_{br}_m{m}_s{s}.json")
            return load(p)["best_dev_acc"] if os.path.exists(p) else None
        g = bd("grpo"); c = bd("csft"); r = bd("rssft")
        pw = os.path.join(c3v, f"bestdev_m{m}_s{s}.json")
        w = load(pw)["best_dev_acc"] if os.path.exists(pw) else None
        non = [x for x in [c, r, w, sdev] if x is not None]
        dd = None if g is None or not non else round((g - max(non)) * 100, 2)
        t = load(os.path.join(BK, "config3_7b", f"test_m{m}_s{s}.json"))
        ta = {k: v["acc"] for k, v in t["arms"].items()}
        td = round((ta["grpo"] - max(v for k, v in ta.items() if k != "grpo")) * 100, 2)
        holds = None if dd is None else (td < dd)
        claim_holds.append(holds)
        print(f"  m={m} s={s}: devΔ={dd}  testΔ={td}  test更负={holds}")
print("  >>> 全部成立:", all(claim_holds), " 有效组数:", sum(1 for x in claim_holds if x is not None))

print("\n=== 2) 消融（variants 模式）===")
def abl3(cfg, m):
    A=[];B=[];C=[]
    for s in [0,1,2]:
        p=os.path.join(BK,cfg,"runs_v2",f"ablation_m{m}_s{s}.json")
        if not os.path.exists(p): return None
        v=load(p)["variants"]
        A.append(v["rules"]["acc"]); B.append(v["retrieval"]["acc"]); C.append(v["full"]["acc"])
    return tuple(round(sum(x)/3*100,2) for x in (A,B,C))
bad1=[m for m in [0,1,2,4,8,16,64,256,1024,2446] if abl3("config1_full",m)!=(72.00,0.00,72.00)]
print("  C1 十档 72/0/72 全部成立:", not bad1, ("异常档:"+str(bad1) if bad1 else ""))
print("  C2 m128:", abl3("config2_full",128), " 文 36.00/0.00/35.50")
print("  C2 m256:", abl3("config2_full",256), " 文 35.50/0.00/35.33")

print("\n=== 3) McNemar ===")
def mcp(v1,v2):
    b=sum(1 for x,y in zip(v1,v2) if x==1 and y==0); c=sum(1 for x,y in zip(v1,v2) if x==0 and y==1)
    n=b+c
    if n==0: return 1.0
    k=min(b,c); return min(1.0, sum(math.comb(n,i) for i in range(k+1))/2**n*2)
ps=[]
for s in [0,2]:
    d=load(os.path.join(BK,"test_final_config2",f"test_m128_s{s}.json"))
    p=mcp(d["arms"]["solver"]["per_item"],d["arms"]["grpo"]["per_item"]); ps.append(p)
    print(f"  C2 m128 s{s}: p={p:.2e}")
print("  Holm 后最大 p×2 =", f"{max(ps)*2:.2e}", " <0.001:", max(ps)*2<0.001)
sev=[]
for m in [2,64]:
    for s in [0,1,2]:
        d=load(os.path.join(BK,"config3_7b",f"test_m{m}_s{s}.json"))
        p=mcp(d["arms"]["csft"]["per_item"],d["arms"]["grpo"]["per_item"])
        sev.append(((m,s),round(p,4)))
print("  7B GRPO vs csft:", sev)
print("  两个最小差距组(m64s1 Δ-1.06, m2s2 Δ-1.77) 不显著:",
      dict(sev)[(64,1)]>0.05 and dict(sev)[(2,2)]>0.05)

print("\n=== 4) C2 warm 剂量（单种子档）===")
for m,exp in [(0,0.5),(16,None),(64,None),(1024,43.0),(3635,47.0)]:
    p=os.path.join(BK,"config2_full","runs_v2",f"bestdev_m{m}_s0.json")
    v=round(load(p)["best_dev_acc"]*100,1) if os.path.exists(p) else None
    print(f"  m={m}: s0={v}"+(f"  文={exp} 匹配={v==exp}" if exp else ""))

print("\n=== 5) C1 剂量各档种子覆盖 ===")
for m in [0,1,2,4,8,16,64,256,1024,2446]:
    n=sum(os.path.exists(os.path.join(BK,"config1_full","runs_v2",f"bestdev_m{m}_s{s}.json")) for s in [0,1,2])
    if n!=3: print(f"  m={m}: 种子数={n}")
print("  （未打印即=3 种子）")
n7=[(m,sum(os.path.exists(os.path.join(c3v,f"bestdev_m{m}_s{s}.json")) for s in [0,1,2])) for m in [2,64]]
print("  7B 档种子覆盖:", n7)

print("\n=== 6) DEVIATIONS_v2 条目 ===")
t=open(os.path.join("ca_rlvr_smoke","DEVIATIONS_v2.md"),encoding="utf-8").read()
hs=[l for l in t.splitlines() if re.match(r"^#{2,3}\s",l)]
for h in hs: print("  ", h[:90])
print("  总标题数:", len(hs))

print("\n=== 7) 扫描产物全局搜索 ===")
hits=[f for f in glob.glob("**/*",recursive=True)
      if re.search(r"b0\.0|b0\.2|t0\.7|t1\.3|sweep",os.path.basename(f),re.I)][:20]
print("  ", hits if hits else "（无）")

print("\n=== 8) dev 集大小 ===")
for cfg in ["config1_full","config2_full","config3_7b"]:
    dd=os.path.join(BK,cfg,"data")
    if os.path.isdir(dd):
        for f in sorted(os.listdir(dd)):
            p=os.path.join(dd,f)
            if f.endswith(".jsonl"):
                n=sum(1 for _ in open(p,encoding="utf-8"))
                print(f"  {cfg}/data/{f}: {n} 行")

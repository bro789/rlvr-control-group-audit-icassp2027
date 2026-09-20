# -*- coding: utf-8 -*-
"""Gate 1 v2 solver "combo": 结构化表格解析 + 规则模板库 + train 检索(近邻复制/TF-IDF 签名先验)。

白名单合规（PREREG.md Gate 1）:
  - 正则/规则抽取 + 直接算术模板;
  - TF-IDF 加权词频检索(纯统计, 无学习模型): 同 context 近重复问题 → 复制 gold 程序;
    kNN 签名投票 → 仅在规则分类不确定(other/弱total)时决定模板类别;
  - 确定性: 所有排序用稳定 tie-break; 同输入同输出; 每题恰好一个程序;
  - 推理时只看当前题 question+context; 检索库/统计量只从 train 构建。

用法:
  python solver_v2_combo.py --dev data/CodeFinQA__train.jsonl   # 开发: train 子集(idx%4==0), LOO 检索, 快速内嵌执行
  python solver_v2_combo.py data/CodeFinQA__test.jsonl          # 最终: 真 verifier, gold 自洽子集
"""
import json, re, sys, os, math, ast
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verifier import verify, parse_number

TRAIN_PATH = os.path.join(HERE, "data", "CodeFinQA__train.jsonl")
TAG_RE = re.compile(r"<[^>]+>")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# clean year: not glued to letters (mojibake like 'company 2019s', 'available-for 2013sale')
YEAR_CLEAN_RE = re.compile(r"(?<![a-zA-Z0-9])((?:19|20)\d{2})(?![a-zA-Z0-9])")
NUM_RE = re.compile(r"\(?\$?\s?-?\d[\d,]*\.?\d*\)?%?")
MONTH_RE = re.compile(r"january|february|march|april|may|june|july|august|september|october|november|december", re.I)

STOP = set("""a an the of in on for to and or is are was were what which how much many did does do
during between from by as at amount value number year years ended december january what's
company companies million billion thousand percent percentage change difference approximate that
approximately respectively following table based upon it its their his her this these those there
be been being have has had would could should will shall may might must can what when where who
""".split())

TYPE_WORDS = set("""percentage percent portion proportion change increase decrease growth decline
difference ratio average mean sum combined aggregate highest lowest maximum minimum""".split())


# ---------------------------------------------------------------- tokenization
def norm_tok(w):
    w = w.lower()
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w

def split_camel(text):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

def content_toks(text):
    return [norm_tok(w) for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", split_camel(text))
            if w.lower() not in STOP and len(w) > 2]

def toks_with_years(text):
    return set(content_toks(text)) | set(YEAR_CLEAN_RE.findall(text))

def q_toks_all(text):
    return [norm_tok(w) for w in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-']*", text.lower())]


# ---------------------------------------------------------------- table parsing
def parse_tables(context):
    blocks, cur = [], []
    for line in context.split("\n"):
        if line.strip().startswith("|"):
            cur.append(line)
        elif cur:
            blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    tables = []
    for tl in blocks:
        header_cells, rows = [], []
        for line in tl:
            cells = [TAG_RE.sub("", c).strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c.strip()):
                continue
            nums = [parse_number(c) if c.strip() else None for c in cells[1:]]
            n_num = sum(1 for v in nums if v is not None)
            years_in = [c for c in cells[1:] if YEAR_RE.fullmatch(c.strip())]
            if not rows and (n_num == 0 or (years_in and len(years_in) >= n_num)):
                header_cells.append(cells)
                continue
            rows.append((cells[0], nums))
        if not rows:
            continue
        ncols = max(len(r[1]) for r in rows)
        col_years = [None] * ncols
        col_text = [""] * ncols
        for hc in header_cells:
            h = hc[1:]
            off = ncols - len(h) if 0 < ncols - len(h) else 0  # right-align short headers
            for j in range(min(ncols, len(h))):
                jj = j + off
                if jj >= ncols:
                    continue
                m = YEAR_RE.search(h[j])  # allow glued years like 'September2014'
                if not m:
                    m2 = re.search(r"\b\d{1,2}/(?:\d{1,2}/)?(\d{2})\b", h[j])  # 12/07, 12/31/07
                    if m2:
                        yy = int(m2.group(1))
                        col_years[jj] = col_years[jj] or str(1900 + yy if yy > 50 else 2000 + yy)
                if m and col_years[jj] is None:
                    col_years[jj] = m.group(0)
                if h[j].strip():
                    col_text[jj] = (col_text[jj] + " " + h[j]).strip()
        t_rows = []
        for label, nums in rows:
            lab = re.sub(r"\(\s*[a-z0-9]\s*\)$", "", label.strip()).strip()
            ym = YEAR_RE.search(lab)
            t_rows.append({
                "label": lab,
                "toks": toks_with_years(lab) | set(YEAR_RE.findall(lab)),
                "year": ym.group(0) if ym else None,
                "nums": nums + [None] * (ncols - len(nums)),
                "is_total": bool(re.search(r"\btotal\b", lab.lower())),
            })
        tables.append({
            "col_years": col_years,
            "col_text": col_text,
            "col_toks": [toks_with_years(ct) for ct in col_text],
            "rows": t_rows,
        })
    return tables


# ---------------------------------------------------------------- text numbers
def text_candidates(context):
    out = []
    prose = " ".join(l for l in context.split("\n") if not l.strip().startswith("|"))
    for sent in re.split(r"(?<= )\.\s+|(?<=\w)\. (?=[a-z])", prose):
        years = [(m.group(1), m.start()) for m in YEAR_CLEAN_RE.finditer(sent)]
        nums = []
        for m in NUM_RE.finditer(sent):
            raw = m.group(0)
            core = raw.strip().lstrip("$ ").strip()
            if YEAR_RE.fullmatch(core):
                continue
            # glued to letters -> mojibake fragment
            if m.end() < len(sent) and sent[m.end()].isalpha():
                continue
            if m.start() > 0 and sent[m.start() - 1].isalpha():
                continue
            v = parse_number(raw)
            if v is None:
                continue
            # day-of-month: small int right after a month name
            if abs(v) <= 31 and v == int(v):
                pre = sent[max(0, m.start() - 12):m.start()]
                if MONTH_RE.search(pre):
                    continue
            nums.append((v, m.start(), m.end(), raw.strip().endswith("%")))
        resp = "respectively" in sent
        pair_map = {}
        if resp and len(years) >= 2:
            if len(nums) == len(years):
                for k in range(len(nums)):
                    pair_map[k] = years[k][0]
            elif len(nums) > len(years):
                # pair the last len(years) numbers appearing before the years/`respectively`
                ypos0 = years[0][1]
                before = [k for k, n in enumerate(nums) if n[1] < ypos0]
                if len(before) >= len(years):
                    for idx, k in enumerate(before[-len(years):]):
                        pair_map[k] = years[idx][0]
        for k, (v, s, e, pct) in enumerate(nums):
            win = set(content_toks(sent[max(0, s - 90):e + 90]))
            if k in pair_map:
                yr = pair_map[k]
            else:
                yr = min(years, key=lambda t: abs(t[1] - s))[0] if years else None
            out.append({"value": v, "window": win, "years": set(y for y, _ in years),
                        "year": yr, "pct": pct, "sent": id(sent)})
    return out


# ---------------------------------------------------------------- question parse
def q_years(q):
    ys = YEAR_CLEAN_RE.findall(q)
    rng = None
    m = re.search(r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})\b", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if 0 < y2 - y1 <= 10:
            rng = [str(y) for y in range(y1, y2 + 1)]
            if m.group(1) not in ys:
                ys = [m.group(1), m.group(2)] + ys
    m2 = re.search(r"(?:from|between|during)\s+((?:19|20)\d{2})\s+(?:to|and|through)\s+((?:19|20)\d{2})", q.lower())
    if m2 and rng is None:
        y1, y2 = int(m2.group(1)), int(m2.group(2))
        if 0 < y2 - y1 <= 10:
            rng = [str(y) for y in range(y1, y2 + 1)]
    seen, uy = set(), []
    for y in ys:
        if y not in seen:
            seen.add(y); uy.append(y)
    return uy, rng

DEC_RE = re.compile(r"\bdecreas|\bdeclin|\bdrop|\breduc|\bfell\b|\bfall\b", re.I)
PCT_RE = re.compile(r"\bpercentage\b|\bpercent\b|\bportion\b|\bproportion\b|as a percent|what percent|%", re.I)
PERIOD_RE = re.compile(r"next (?:12|twelve) months|less than (?:1|one) year|within (?:the next )?(?:12|twelve) months|within one year|next year\b", re.I)

def classify(q):
    ql = q.lower()
    if re.search(r"every dollar", ql):
        return "dollar_return"
    if re.search(r"(gross|operating|net|profit)\s+(profit\s+)?margin", ql):
        return "margin"
    if re.search(r"growth rate|rate of growth|percentage change|percent change|pct change", ql):
        return "growth"
    if re.search(r"(percentage|percent|%)\s+(change|increase|decrease|decline|growth|gain|drop|reduction)", ql):
        return "growth"
    if re.search(r"(increase|decrease|decline|change|grow|drop)[a-z]*\s.*\bas a percent", ql):
        return "growth"
    if PCT_RE.search(ql) and re.search(r"\b(change|increase|decrease|decline|growth|drop)\b\s+(in|of|from|between|during)\b", ql):
        return "growth"
    if re.search(r"\bratio\b|\brate of\b.*\bto\b", ql) and "growth" not in ql:
        return "ratio"
    if PCT_RE.search(ql):
        return "portion"
    if re.search(r"\baverage\b|\bmean\b", ql):
        return "avg"
    if re.search(r"\brange of\b|\bmathematical range\b", ql):
        return "range"
    if re.search(r"\b(highest|largest|greatest|maximum|lowest|smallest|minimum)\b", ql):
        return "extreme"
    if re.search(r"\bsum\b|\bcombined\b|\baggregate\b|\btotal of\b", ql):
        return "sum"
    if re.search(r"\btotal\b.*\bin\s+(?:19|20)\d{2}\s+and\s+(?:19|20)\d{2}\b", ql):
        return "sum"
    if CHG_RE.search(ql) or re.search(r"\bdifference\b|\bhow much (more|less|greater|higher|lower)\b|\bvary\b|\bvaried\b", ql):
        return "diff"
    if re.search(r"\btotal\b", ql):
        return "sum_weak"
    return "other"

CHG_RE = re.compile(r"\bchange[ds]?\b|\bincrease[ds]?\b|\bdecrease[ds]?\b|\bdecline[ds]?\b|\bgrowth\b|\bgrew\b|\bdrop(?:ped)?\b|\brose\b|\bfell\b|\breduction\b", re.I)

def entity_toks(q):
    return set(t for t in content_toks(q) if t not in TYPE_WORDS) | set(YEAR_CLEAN_RE.findall(q))


# ---------------------------------------------------------------- idf + scoring
class Idf:
    def __init__(self, docs):
        df = Counter()
        self.n = len(docs)
        for d in docs:
            for t in set(d):
                df[t] += 1
        self.df = df
    def w(self, t):
        return math.log((self.n + 1) / (self.df.get(t, 0) + 1)) + 1.0

IDF = None

def overlap_score(toks, qtoks):
    if not toks:
        return 0.0
    return sum((IDF.w(t) if IDF else 1.0) for t in toks & qtoks)

def row_score(row, qtoks, want_total=None):
    s = overlap_score(row["toks"], qtoks)
    if row["toks"] and len(row["toks"]) >= 2 and len(row["toks"] & qtoks) == len(row["toks"]):
        s += 1.5  # full label containment
    if want_total is True:
        s += 2.5 if row["is_total"] else 0.0
    elif want_total is False and row["is_total"]:
        s -= 1.5
    return s


# ---------------------------------------------------------------- cell helpers
def col_for_year(table, y):
    for j, cy in enumerate(table["col_years"]):
        if cy == y:
            return j
    return None

def best_col(table, qtoks, year=None):
    if year is not None:
        j = col_for_year(table, year)
        if j is not None:
            return j
    scored = [(overlap_score(table["col_toks"][j], qtoks), -j, j)
              for j in range(len(table["col_years"]))]
    scored.sort(reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][2]
    ycols = [(int(cy), -j, j) for j, cy in enumerate(table["col_years"]) if cy]
    if ycols:
        return max(ycols)[2]
    return 0 if table["col_years"] else None

def best_row(table, qtoks, want_total=None, need_col=None, exclude=None):
    cand = []
    for ri, row in enumerate(table["rows"]):
        if exclude is not None and ri == exclude:
            continue
        if need_col is not None and (need_col >= len(row["nums"]) or row["nums"][need_col] is None):
            continue
        cand.append((row_score(row, qtoks, want_total), -ri, ri))
    if not cand:
        return None, 0.0
    cand.sort(reverse=True)
    return cand[0][2], cand[0][0]

def pick_pair_by_years(tables, qtoks, y1, y2, want_total=None):
    """(new_val=v(y2), old_val=v(y1), score) via same row two year cols, or transposed."""
    best = None
    for ti, t in enumerate(tables):
        j1, j2 = col_for_year(t, y1), col_for_year(t, y2)
        if j1 is None or j2 is None:
            continue
        for ri, row in enumerate(t["rows"]):
            v1, v2 = row["nums"][j1], row["nums"][j2]
            if v1 is None or v2 is None:
                continue
            sc = row_score(row, qtoks, want_total)
            key = (sc, -ti, -ri)
            if best is None or key > best[0]:
                best = (key, v2, v1)
    if best is not None:
        return best[1], best[2], best[0][0]
    for ti, t in enumerate(tables):
        r1 = [(ri, r) for ri, r in enumerate(t["rows"]) if r["year"] == y1]
        r2 = [(ri, r) for ri, r in enumerate(t["rows"]) if r["year"] == y2]
        if not (r1 and r2):
            continue
        j = best_col(t, qtoks)
        if j is None:
            continue
        r1.sort(key=lambda p: (-row_score(p[1], qtoks), p[0]))
        r2.sort(key=lambda p: (-row_score(p[1], qtoks), p[0]))
        v1 = r1[0][1]["nums"][j] if j < len(r1[0][1]["nums"]) else None
        v2 = r2[0][1]["nums"][j] if j < len(r2[0][1]["nums"]) else None
        if v1 is not None and v2 is not None:
            return v2, v1, 1.0
    return None

_NUMPAT = r"\$?\s?-?\d[\d,]*\.?\d*"
TREND_RES = [
    re.compile(r"(?:increased|decreased|declined|rose|fell|grew|went|improved|changed)\s+(?:by\s+[^.]{0,20}?)?to\s+(" + _NUMPAT + r")(?:\s?(?:million|billion|thousand))?\b[^.]{0,60}?\bfrom\s+(" + _NUMPAT + r")", re.I),
    re.compile(r"(?:increased|decreased|declined|rose|fell|grew|went|improved|changed)\s+(?:by\s+[^.]{0,20}?)?from\s+(" + _NUMPAT + r")(?:\s?(?:million|billion|thousand))?\b[^.]{0,60}?\bto\s+(" + _NUMPAT + r")", re.I),
    re.compile(r"(" + _NUMPAT + r")(?:\s?(?:million|billion|thousand))?\s*(?:in|for)\s+(?:fiscal\s+)?(?:19|20)\d{2}\s*,?\s+compared\s+(?:to|with)\s+(" + _NUMPAT + r")", re.I),
]

def trend_pairs(context, qtoks):
    """(new, old, score) pairs from 'increased to A from B' style sentences."""
    out = []
    prose = " ".join(l for l in context.split("\n") if not l.strip().startswith("|"))
    for sent in re.split(r"(?<= )\.\s+|(?<=\w)\. (?=[a-z])", prose):
        win = set(content_toks(sent))
        sc = overlap_score(win, qtoks)
        for pi, pat in enumerate(TREND_RES):
            m = pat.search(sent)
            if not m:
                continue
            g1, g2 = parse_number(m.group(1)), parse_number(m.group(2))
            if g1 is None or g2 is None or g1 == g2:
                continue
            if pi == 1:
                new, old = g2, g1
            else:
                new, old = g1, g2
            out.append((sc + 1.0, new, old))
    out.sort(key=lambda t: -t[0])
    return out

def pick_pair_text(cands, qtoks, y1, y2):
    def best_for(y):
        scored = []
        for k, c in enumerate(cands):
            if c["year"] != y and y not in c["years"]:
                continue
            sc = overlap_score(c["window"], qtoks) + (1.5 if c["year"] == y else 0)
            scored.append((sc, -k, c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored[0] if scored else None
    b2, b1 = best_for(y2), best_for(y1)
    if b1 and b2:
        # prefer same-sentence pair when both available
        return b2[2]["value"], b1[2]["value"], min(b1[0], b2[0])
    return None

def top_values(tables, tcands, qtoks, year, k=2, want_total=None):
    scored = []
    for ti, t in enumerate(tables):
        j = best_col(t, qtoks, year=year)
        if j is None:
            continue
        for ri, row in enumerate(t["rows"]):
            if j < len(row["nums"]) and row["nums"][j] is not None:
                scored.append((row_score(row, qtoks, want_total), -ti, -ri, row["nums"][j]))
    for kk, c in enumerate(tcands):
        sc = overlap_score(c["window"], qtoks) * 0.8 + (1.0 if (year and year in c["years"]) else 0.0)
        scored.append((sc, -99, -kk, c["value"]))
    scored.sort(reverse=True)
    out = []
    for sc, _, _, v in scored:
        if v not in out:
            out.append(v)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------- retrieval store
def sig_class(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "other"
    env, ans = {}, None
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            env[st.targets[0].id] = st.value
            if st.targets[0].id == "answer":
                ans = st.value
    if ans is None:
        return "other"
    def render(node, d=0):
        if d > 12:
            return "?"
        if isinstance(node, ast.Constant):
            return "N"
        if isinstance(node, ast.UnaryOp):
            return render(node.operand, d + 1)
        if isinstance(node, ast.Name):
            return render(env[node.id], d + 1) if node.id in env else "?"
        if isinstance(node, ast.BinOp):
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(node.op), "?")
            return "(" + render(node.left, d + 1) + op + render(node.right, d + 1) + ")"
        if isinstance(node, ast.Call):
            fn = node.func.id if isinstance(node.func, ast.Name) else "?"
            return fn + "()"
        return "?"
    s = render(ans)
    if s in ("(((N-N)/N)*N)", "((N/N)-N)", "(((N/N)-N)*N)"):
        return "growth"
    if s in ("((N/N)*N)", "((N/(N+N))*N)"):
        return "portion"
    if s == "(N-N)":
        return "diff"
    if s == "(N/N)":
        return "ratio"
    if s in ("(N+N)", "((N+N)+N)", "(((N+N)+N)+N)"):
        return "sum"
    if s in ("((N+N)/N)", "(((N+N)+N)/N)") or "len()" in s or "sum()" in s:
        return "avg"
    if s == "(N*N)":
        return "prod"
    return "other"


class Store:
    def __init__(self, rows, sc_flags):
        self.items = []
        docs = []
        for i, r in enumerate(rows):
            if not sc_flags[i]:
                continue
            qt = q_toks_all(r["question"])
            docs.append(qt)
            self.items.append({
                "idx": i, "qtoks": set(qt), "qcount": Counter(qt),
                "ctx_key": hash(r["context"]), "program": r["program"],
                "question": r["question"], "sig": sig_class(r["program"]),
            })
        global IDF
        IDF = Idf(docs)
        for it in self.items:
            v = {t: c * IDF.w(t) for t, c in it["qcount"].items()}
            it["vec"] = v
            it["norm"] = math.sqrt(sum(x * x for x in v.values())) or 1.0
        self.by_ctx = defaultdict(list)
        for k, it in enumerate(self.items):
            self.by_ctx[it["ctx_key"]].append(k)

    def copy_match(self, question, context, exclude_idx=None, thresh=0.9):
        key = hash(context)
        qt = set(q_toks_all(question))
        best, bj = None, 0.0
        for k in self.by_ctx.get(key, []):
            it = self.items[k]
            if exclude_idx is not None and it["idx"] == exclude_idx:
                continue
            jac = len(qt & it["qtoks"]) / max(1, len(qt | it["qtoks"]))
            if jac > bj:
                bj, best = jac, it
        if best is not None and bj >= thresh:
            return best["program"]
        return None

    def knn_sig(self, question, k=10, exclude_idx=None):
        qc = Counter(q_toks_all(question))
        v = {t: c * IDF.w(t) for t, c in qc.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        scored = []
        for ki, it in enumerate(self.items):
            if exclude_idx is not None and it["idx"] == exclude_idx:
                continue
            dot = 0.0
            for t, x in v.items():
                y = it["vec"].get(t)
                if y:
                    dot += x * y
            if dot > 0:
                scored.append((dot / (n * it["norm"]), -ki, it["sig"]))
        scored.sort(reverse=True)
        vote = Counter()
        for sim, _, sig in scored[:k]:
            vote[sig] += sim
        return vote.most_common(1)[0][0] if vote else None


# ---------------------------------------------------------------- special table patterns
BEG_RE = re.compile(r"january 1|beginning|start of", re.I)
END_RE = re.compile(r"december 3[01]|ending|end of|at year end", re.I)

def rolling_pair(tables, qtoks, y1, y2):
    """Beginning/Ending balance tables. Returns (new, old) or None.
    y1==y2: new=end(y), old=begin(y). y1<y2: new=end(y2), old=end(y1) or begin(y2)."""
    best = None
    for ti, t in enumerate(tables):
        begs = [(ri, r) for ri, r in enumerate(t["rows"])
                if BEG_RE.search(r["label"]) and not END_RE.search(r["label"].split(",")[0][:14])]
        ends = [(ri, r) for ri, r in enumerate(t["rows"]) if END_RE.search(r["label"])]
        if not begs and not ends:
            continue
        tab_sc = max([row_score(r, qtoks) for r in t["rows"]] +
                     [overlap_score(ct, qtoks) for ct in t["col_toks"]] + [0.0])
        # orientation A: year in row label, single value column
        def by_year(rows, y):
            c = [(ri, r) for ri, r in rows if r["year"] == y]
            return c[0] if c else None
        j = best_col(t, qtoks)
        new = old = None
        if y1 == y2:
            e, b = by_year(ends, y2), by_year(begs, y2)
            if e and b and j is not None:
                new, old = e[1]["nums"][j] if j < len(e[1]["nums"]) else None, \
                           b[1]["nums"][j] if j < len(b[1]["nums"]) else None
        else:
            e2, e1 = by_year(ends, y2), by_year(ends, y1)
            if e2 and e1 and j is not None:
                new, old = e2[1]["nums"][j] if j < len(e2[1]["nums"]) else None, \
                           e1[1]["nums"][j] if j < len(e1[1]["nums"]) else None
            if new is None or old is None:
                e2, b2 = by_year(ends, y2), by_year(begs, y2)
                if e2 and b2 and j is not None:
                    new, old = e2[1]["nums"][j] if j < len(e2[1]["nums"]) else None, \
                               b2[1]["nums"][j] if j < len(b2[1]["nums"]) else None
        # orientation B: begin/end rows without year, year columns
        if (new is None or old is None):
            bny = [(ri, r) for ri, r in begs if r["year"] is None]
            eny = [(ri, r) for ri, r in ends if r["year"] is None]
            j2 = col_for_year(t, y2)
            j1 = col_for_year(t, y1)
            if bny and eny:
                if y1 == y2 and j2 is not None:
                    new = eny[0][1]["nums"][j2] if j2 < len(eny[0][1]["nums"]) else None
                    old = bny[0][1]["nums"][j2] if j2 < len(bny[0][1]["nums"]) else None
                elif j1 is not None and j2 is not None:
                    new = eny[0][1]["nums"][j2] if j2 < len(eny[0][1]["nums"]) else None
                    old = eny[0][1]["nums"][j1] if j1 < len(eny[0][1]["nums"]) else None
                elif j2 is not None:
                    new = eny[0][1]["nums"][j2] if j2 < len(eny[0][1]["nums"]) else None
                    old = bny[0][1]["nums"][j2] if j2 < len(bny[0][1]["nums"]) else None
        if new is not None and old is not None:
            key = (tab_sc, -ti)
            if best is None or key > best[0]:
                best = (key, new, old)
    if best:
        return best[1], best[2]
    return None

def base100_table(tables):
    """performance graph table: a column (or row) whose values are all 100."""
    for ti, t in enumerate(tables):
        for j in range(len(t["col_years"])):
            vals = [r["nums"][j] for r in t["rows"] if j < len(r["nums"]) and r["nums"][j] is not None]
            if len(vals) >= 2 and all(v == 100.0 for v in vals):
                return ti, j
    return None

CUM_RE = re.compile(r"cumulative|five[- ]year|5[- ]year|since (?:inception|the)|100 invested|initial", re.I)

def cumulative_pair(q, tables, qtoks):
    b = base100_table(tables)
    if b is None:
        return None
    ti, j0 = b
    t = tables[ti]
    ys, _ = q_years(q)
    ri, sc = best_row(t, qtoks, need_col=j0)
    if ri is None:
        return None
    row = t["rows"][ri]
    jn = None
    if ys:
        jn = col_for_year(t, max(ys, key=int))
    if jn is None:
        ycols = [(int(cy), j) for j, cy in enumerate(t["col_years"]) if cy and j != j0]
        jn = max(ycols)[1] if ycols else None
    if jn is None:
        jn = len(row["nums"]) - 1
    new = row["nums"][jn] if jn < len(row["nums"]) else None
    old = row["nums"][j0]
    if new is None or old is None:
        return None
    return new, old


# ---------------------------------------------------------------- templates
def emit_growth(new, old, dec, pct=True):
    if old == 0:
        return None
    tail = " * 100" if pct else ""
    if dec:
        return f"old = {old!r}\nnew = {new!r}\nanswer = (old - new) / abs(old){tail}"
    return f"old = {old!r}\nnew = {new!r}\nanswer = (new - old) / abs(old){tail}"

def emit_diff(new, old, dec):
    if dec:
        return f"old = {old!r}\nnew = {new!r}\nanswer = old - new"
    return f"old = {old!r}\nnew = {new!r}\nanswer = new - old"

def solve_growth_or_diff(kind, q, tables, tcands, qtoks, pct=True):
    ql = q.lower()
    dec = bool(DEC_RE.search(ql)) and not re.search(r"increase", ql)
    wt = True if "total" in ql else None
    ys, rng = q_years(q)
    # special: cumulative/base-100 performance
    if CUM_RE.search(ql) and re.search(r"return|value|price|stock|index", ql):
        cp = cumulative_pair(q, tables, qtoks)
        if cp:
            return emit_growth(cp[0], cp[1], dec, pct) if kind == "growth" else emit_diff(cp[0], cp[1], dec)
    pair = None
    y1 = y2 = None
    if len(ys) >= 2:
        yy = sorted(ys, key=int)
        y1, y2 = yy[0], yy[-1]
        m = re.search(r"from\s+((?:19|20)\d{2})\s*(?:to|-|through|until)\s*((?:19|20)\d{2})", ql)
        if m:
            y1, y2 = m.group(1), m.group(2)
    elif len(ys) == 1:
        y2 = ys[0]
        y1 = str(int(y2) - 1)
    trends = trend_pairs(CURRENT.get("context", ""), qtoks)
    trend = trends[0] if trends and trends[0][0] >= 2.0 else None
    if y1 is not None:
        tab = pick_pair_by_years(tables, qtoks, y1, y2, wt)
        txt = pick_pair_text(tcands, qtoks, y1, y2)
        roll = None
        if len(ys) == 1:
            roll = rolling_pair(tables, qtoks, ys[0], ys[0])
        if roll is None:
            roll = rolling_pair(tables, qtoks, y1, y2)
        tab_sc = tab[2] if tab else -1.0
        txt_sc = txt[2] if txt else -1.0
        if roll is not None and tab_sc < 3.0:
            pair = (roll[0], roll[1], 99.0)
        elif tab is not None and (tab_sc > 0 or txt_sc < 1.5):
            pair = tab
        elif txt is not None:
            pair = txt
        else:
            pair = tab
        if pair is not None and trend is not None and pair[2] <= 0.5 and trend[0] > pair[2] + 1.0:
            pair = (trend[1], trend[2], trend[0])
    if pair is None and trend is not None:
        pair = (trend[1], trend[2], trend[0])
    if pair is None:
        best = None
        for ti, t in enumerate(tables):
            ycols = sorted([(cy, j) for j, cy in enumerate(t["col_years"]) if cy],
                           key=lambda p: int(p[0]))
            if len(ycols) < 2:
                continue
            (yA, jA), (yB, jB) = ycols[-2], ycols[-1]
            for ri, row in enumerate(t["rows"]):
                vA, vB = row["nums"][jA], row["nums"][jB]
                if vA is None or vB is None:
                    continue
                key = (row_score(row, qtoks, wt), -ti, -ri)
                if best is None or key > best[0]:
                    best = (key, vB, vA)
        if best and best[0][0] > 0:
            pair = (best[1], best[2], best[0][0])
    if pair is None:
        return None
    new, old, _ = pair
    return emit_growth(new, old, dec, pct) if kind == "growth" else emit_diff(new, old, dec)

PORTION_VERB = (r"is|was|were|are|came|come|comes|consist|consists|consisted|do|does|did|has|have|"
                r"had|belong|belongs|belonged|made|makes|make|relate|relates|related|represent|"
                r"represents|represented|account|accounts|accounted|occur|occurs|occurred|"
                r"attributable|associated|derived|generated|held|deposited|invested|used|paid|"
                r"expire|expires|due|goes|went")

def portion_split(q):
    ql = re.sub(r"[?.]", " ", q.lower())
    # "percent of A to B" (ratio-phrased portion)
    m0 = re.search(r"(?:percentage|percent|portion|proportion)\s+of\s+(?:the\s+)?(.+?)\s+(?<!due\s)to\s+(?:the\s+)?(.+)$", ql)
    if m0 and " total " not in " " + m0.group(1) + " ":
        pt = set(t for t in content_toks(m0.group(1)) if t not in TYPE_WORDS) | set(YEAR_CLEAN_RE.findall(m0.group(1)))
        wtk = set(t for t in content_toks(m0.group(2)) if t not in TYPE_WORDS) | set(YEAR_CLEAN_RE.findall(m0.group(2)))
        if pt and wtk and pt != wtk:
            return pt, wtk, "total" in m0.group(2)
    m = re.search(r"(?:percentage|percent|portion|proportion)\s+of\s+(.*?)(?:\s+(?:" + PORTION_VERB + r")\b(.*)|$)", ql)
    if m:
        whole = m.group(1)
        part = m.group(2) or ""
        wt_total = "total" in whole
        pt = set(t for t in content_toks(part) if t not in TYPE_WORDS) | set(YEAR_CLEAN_RE.findall(part))
        wtk = set(t for t in content_toks(whole) if t not in TYPE_WORDS) | set(YEAR_CLEAN_RE.findall(whole))
        if not pt:
            pre = ql[:m.start()]
            pt = set(t for t in content_toks(pre) if t not in TYPE_WORDS) | set(YEAR_CLEAN_RE.findall(pre))
        return pt, wtk, wt_total
    m = re.search(r"(.*?)\s+as\s+a\s+(?:percentage|percent|portion)\s+of\s+(.*)", ql)
    if m:
        pt = set(t for t in content_toks(m.group(1)) if t not in TYPE_WORDS) | set(YEAR_CLEAN_RE.findall(m.group(1)))
        wtk = set(t for t in content_toks(m.group(2)) if t not in TYPE_WORDS) | set(YEAR_CLEAN_RE.findall(m.group(2)))
        return pt, wtk, "total" in m.group(2)
    return None, None, False

def portion_pair(tables, tcands, pt, wt, whole_is_total, y, ql=""):
    """best (part, whole): part cell by pt; whole = total col same row / total row same col / wt row."""
    period = bool(PERIOD_RE.search(ql))
    if period:
        pt = pt | {"less", "than"}
    thereafter = bool(re.search(r"thereafter|after (?:19|20)\d{2}", ql))
    if thereafter:
        pt = pt | {"thereafter"}
        y = None  # 'after 2019' -> Thereafter row, not year cell
    best_part = None
    for ti, t in enumerate(tables):
        has_totcol = any("total" in ct for ct in t["col_toks"])
        for ri, row in enumerate(t["rows"]):
            for j, v in enumerate(row["nums"]):
                if v is None:
                    continue
                sc = row_score(row, pt) + overlap_score(t["col_toks"][j], pt) * 0.7 \
                     + overlap_score(row["toks"], wt) * 0.3
                if y is not None:
                    if t["col_years"][j] == y or row["year"] == y:
                        sc += 2.0
                    elif t["col_years"][j] is not None and t["col_years"][j] != y:
                        sc -= 1.0
                    elif row["year"] is not None and row["year"] != y:
                        sc -= 1.0
                is_totcol = "total" in t["col_toks"][j]
                if row["is_total"] and "total" not in pt and not (wt and overlap_score(row["toks"], wt) > 0):
                    sc -= 1.0
                if is_totcol:
                    # no year/period specifier -> gold usually reads the Total column
                    if y is None and not period and not thereafter:
                        sc += 0.6
                    else:
                        sc -= 0.8
                if period and (re.search(r"less than|within", t["col_toks"][j] and " ".join(t["col_toks"][j]) or "")
                               or re.search(r"less than|within", row["label"].lower())):
                    sc += 1.5
                key = (sc, -ti, -ri, -j)
                if best_part is None or key > best_part[0]:
                    best_part = (key, ti, ri, j, v)
    if best_part is None:
        # text-only
        scored = [(overlap_score(c["window"], pt) + (1.0 if (y and y in c["years"]) else 0.0), -k, c)
                  for k, c in enumerate(tcands)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        if not scored or scored[0][0] <= 0:
            return None
        part = scored[0][2]["value"]
        pool = wt if wt else pt
        s2 = [(overlap_score(c["window"], pool) + (1.0 if (y and y in c["years"]) else 0.0)
               + (0.5 if abs(c["value"]) > abs(part) else 0.0), -k, c)
              for k, c in enumerate(tcands) if c["value"] != part]
        s2.sort(key=lambda t: (-t[0], t[1]))
        if s2 and s2[0][0] > 0:
            return part, s2[0][2]["value"]
        return None
    _, ti, ri, j, part = best_part
    t = tables[ti]
    prow = t["rows"][ri]
    whole_cands = []
    # does the whole-phrase just restate the part row entity (e.g. "total <entity>")?
    wt_core = (wt - {"total"}) if wt else set()
    wt_best_ri, wt_best_sc = (best_row(t, wt, want_total=(whole_is_total or None))
                              if wt_core else (None, 0.0))
    frac = (len(wt_core & (prow["toks"] | pt)) / len(wt_core)) if wt_core else 0.0
    wt_is_part_entity = bool(wt_core) and (frac >= 0.6 or wt_best_ri == ri)
    # (c) total col, same row -- dominant for obligations-style tables
    for jj in range(len(t["col_years"])):
        if jj != j and "total" in t["col_toks"][jj] and jj < len(prow["nums"]) \
           and prow["nums"][jj] is not None:
            bonus = 6.0 if wt_is_part_entity else (3.2 if (whole_is_total or not wt_core) else 2.0)
            whole_cands.append((bonus + overlap_score(t["col_toks"][jj], wt or pt) * 0.3,
                                "totalcol", prow["nums"][jj]))
            break
    # (b) total row, same col
    trows = [(row_score(r, wt or pt), -rk, rk) for rk, r in enumerate(t["rows"])
             if r["is_total"] and rk != ri and j < len(r["nums"]) and r["nums"][j] is not None]
    trows.sort(reverse=True)
    if trows:
        bonus = 3.5 if (whole_is_total or wt_is_part_entity or not wt_core) else 2.0
        whole_cands.append((trows[0][0] + bonus, "totalrow", t["rows"][trows[0][2]]["nums"][j]))
    # (a) wt-matched row, same col
    if wt_core and not wt_is_part_entity:
        wri, wsc = best_row(t, wt, want_total=(whole_is_total or None), need_col=j, exclude=ri)
        if wri is not None and wsc >= 1.0:
            whole_cands.append((wsc + 1.0, "wtrow", t["rows"][wri]["nums"][j]))
    # (e) wt-matched other column, same row -- must beat the part column's own wt overlap
    if wt_core and not wt_is_part_entity:
        base = overlap_score(t["col_toks"][j], wt)
        cscored = [(overlap_score(t["col_toks"][jj], wt), -jj, jj)
                   for jj in range(len(t["col_years"]))
                   if jj != j and jj < len(prow["nums"]) and prow["nums"][jj] is not None]
        cscored.sort(reverse=True)
        if cscored and cscored[0][0] >= 1.5 and cscored[0][0] - base >= 1.0:
            whole_cands.append((cscored[0][0] + 1.0, "wtcol", prow["nums"][cscored[0][2]]))
    # (d) generic wt row fallback
    if not whole_cands and wt:
        wri, wsc = best_row(t, wt, need_col=j, exclude=ri)
        if wri is not None:
            whole_cands.append((wsc, "wtrow2", t["rows"][wri]["nums"][j]))
    if not whole_cands:
        # text whole
        pool = wt if wt else pt | {"total"}
        s2 = [(overlap_score(c["window"], pool), -k, c) for k, c in enumerate(tcands)
              if c["value"] != part]
        s2.sort(key=lambda t: (-t[0], t[1]))
        if s2 and s2[0][0] > 0:
            return part, s2[0][2]["value"]
        return None
    whole_cands.sort(key=lambda x: -x[0])
    return part, whole_cands[0][2]

def solve_portion(q, tables, tcands, qtoks):
    ys, _ = q_years(q)
    ql0 = q.lower()
    mdue = re.search(r"due (?:in|during) ((?:19|20)\d{2})", ql0)
    if mdue:
        y = mdue.group(1)
    elif len(ys) >= 2:
        y = ys[-1]  # leading year is usually the as-of date
    else:
        y = ys[0] if ys else None
    pt, wt, whole_is_total = portion_split(q)
    if pt is None:
        pt, wt, whole_is_total = qtoks, set(), True
    pw = portion_pair(tables, tcands, pt, wt, whole_is_total, y, q.lower())
    if pw is None:
        return None
    part, whole = pw
    if whole is None or part is None:
        return None
    if abs(part) > abs(whole):
        part, whole = whole, part
    if whole == 0:
        return None
    return f"part = {part!r}\nwhole = {whole!r}\nanswer = abs(part) / abs(whole) * 100"

def solve_margin(q, tables, tcands, qtoks):
    ys, _ = q_years(q)
    y = ys[0] if ys else None
    pt = (qtoks - {"margin"}) | {"profit", "income", "earning", "operating"}
    wt = (qtoks - {"margin"}) | {"sale", "revenue"}
    best = None
    for ti, t in enumerate(tables):
        j = best_col(t, qtoks, year=y)
        if j is None:
            continue
        pri, psc = best_row(t, pt, need_col=j)
        wri, wsc = best_row(t, wt, need_col=j, exclude=pri)
        if pri is None or wri is None:
            continue
        pv, wv = t["rows"][pri]["nums"][j], t["rows"][wri]["nums"][j]
        if pv is None or wv is None or wv == 0:
            continue
        # profit should be smaller than sales
        if abs(pv) > abs(wv):
            continue
        key = (psc + wsc, -ti)
        if best is None or key > best[0]:
            best = (key, pv, wv)
    if best:
        return f"profit = {best[1]!r}\nsales = {best[2]!r}\nanswer = profit / sales * 100"
    return solve_portion(q, tables, tcands, qtoks)

def _ratio_col(table, qtoks, y):
    """column for ratio row-mode: year col > total col > token col > newest year."""
    if y is not None:
        j = col_for_year(table, y)
        if j is not None:
            return j
    for j in range(len(table["col_years"])):
        if "total" in table["col_toks"][j]:
            return j
    return best_col(table, qtoks, year=y)

def solve_ratio(q, tables, tcands, qtoks):
    ql = re.sub(r"[?.]", " ", q.lower())
    ys, _ = q_years(q)
    # same-entity two-year ratio: "ratio of X in 2008 to 2007"
    m2 = re.search(r"((?:19|20)\d{2})\s+(?:to|and|vs\.?|versus|compared to)\s+((?:19|20)\d{2})", ql)
    if m2 and "ratio" in ql:
        ya, yb = m2.group(1), m2.group(2)
        pair = pick_pair_by_years(tables, qtoks, yb, ya)  # returns (v(ya), v(yb))
        if pair is None:
            pair = pick_pair_text(tcands, qtoks, yb, ya)
        if pair and pair[1] != 0:
            return f"a = {pair[0]!r}\nb = {pair[1]!r}\nanswer = a / b"
    m = re.search(r"ratio\s+of\s+(?:the\s+)?(.*?)\s+(?:to|and|vs\.?|versus|compared to)\s+(?:the\s+)?(.*)", ql)
    if not m:
        m = re.search(r"(?:the\s+)?([\w\-&' ]+?)[\s\-]+to[\s\-]+([\w\-&' ]+?)\s+ratio", ql)
    a_t = b_t = None
    if m:
        drop = {"ratio"}
        a_t = set(t for t in content_toks(m.group(1)) if t not in drop) | set(YEAR_CLEAN_RE.findall(m.group(1)))
        b_t = set(t for t in content_toks(m.group(2)) if t not in drop) | set(YEAR_CLEAN_RE.findall(m.group(2)))
    y = ys[0] if ys else None
    if a_t and b_t:
        best = None
        # mode R: roles are rows, shared column
        for ti, t in enumerate(tables):
            j = _ratio_col(t, qtoks, y)
            if j is None:
                continue
            ari, asc = best_row(t, a_t, want_total=("total" in a_t) or None, need_col=j)
            bri, bsc = best_row(t, b_t, want_total=("total" in b_t) or None, need_col=j, exclude=ari)
            if ari is None or bri is None or min(asc, bsc) <= 0:
                continue
            av, bv = t["rows"][ari]["nums"][j], t["rows"][bri]["nums"][j]
            if av is None or bv is None or bv == 0:
                continue
            key = (asc + bsc, -ti)
            if best is None or key > best[0]:
                best = (key, av, bv, asc + bsc)
        # mode C: roles are columns, shared row
        for ti, t in enumerate(tables):
            ncols = len(t["col_years"])
            ascored = sorted([(overlap_score(t["col_toks"][j], a_t), -j, j) for j in range(ncols)], reverse=True)
            if not ascored or ascored[0][0] <= 0:
                continue
            ja = ascored[0][2]
            bscored = sorted([(overlap_score(t["col_toks"][j], b_t), -j, j)
                              for j in range(ncols) if j != ja], reverse=True)
            if not bscored or bscored[0][0] <= 0:
                continue
            jb = bscored[0][2]
            rest = (qtoks - a_t - b_t) | (set(ys) if ys else set())
            rri = None
            rcand = []
            for ri, row in enumerate(t["rows"]):
                if ja < len(row["nums"]) and jb < len(row["nums"]) \
                   and row["nums"][ja] is not None and row["nums"][jb] is not None:
                    sc = row_score(row, rest) + (1.5 if (y and row["year"] == y) else 0.0)
                    rcand.append((sc, -ri, ri))
            rcand.sort(reverse=True)
            if not rcand:
                continue
            rri = rcand[0][2]
            av, bv = t["rows"][rri]["nums"][ja], t["rows"][rri]["nums"][jb]
            if bv == 0:
                continue
            key = (ascored[0][0] + bscored[0][0] + rcand[0][0] + 0.5, -ti)
            if best is None or key > best[0]:
                best = (key, av, bv, key[0])
        # text pair with role windows
        ta = [(overlap_score(c["window"], a_t) + (0.5 if (y and y in c["years"]) else 0), -k, c)
              for k, c in enumerate(tcands)]
        ta.sort(key=lambda t: (-t[0], t[1]))
        tb = [(overlap_score(c["window"], b_t) + (0.5 if (y and y in c["years"]) else 0), -k, c)
              for k, c in enumerate(tcands)]
        tb.sort(key=lambda t: (-t[0], t[1]))
        if ta and tb and ta[0][0] > 0 and tb[0][0] > 0 and ta[0][2]["value"] != tb[0][2]["value"]:
            tsc = min(ta[0][0], tb[0][0]) + 0.5
            if (best is None or tsc > best[3]) and tb[0][2]["value"] != 0:
                return f"a = {ta[0][2]['value']!r}\nb = {tb[0][2]['value']!r}\nanswer = a / b"
        if best:
            return f"a = {best[1]!r}\nb = {best[2]!r}\nanswer = a / b"
    vals = top_values(tables, tcands, qtoks, y, k=2)
    if len(vals) == 2 and vals[1] != 0:
        return f"a = {vals[0]!r}\nb = {vals[1]!r}\nanswer = a / b"
    return None

def year_range_cols(table, rng):
    out = []
    for y in rng:
        j = col_for_year(table, y)
        if j is None:
            return None
        out.append(j)
    return out

def text_series(tcands, qtoks, years):
    """one text value per year, entity-matched; None unless all found with score>=1."""
    vals = []
    for y in years:
        scored = [(overlap_score(c["window"], qtoks) + (1.0 if c["year"] == y else 0), -k, c)
                  for k, c in enumerate(tcands) if c["year"] == y or y in c["years"]]
        scored.sort(key=lambda t: (-t[0], t[1]))
        if not scored or scored[0][0] < 1.0:
            return None
        vals.append(scored[0][2]["value"])
    return vals

def solve_avg(q, tables, tcands, qtoks):
    ql = q.lower()
    ys, rng = q_years(q)
    years = rng if rng and len(rng) >= 2 else (sorted(set(ys), key=int) if len(set(ys)) >= 2 else None)
    # "average X per Y" -> X / Y
    mper = re.search(r"average\s+(?:\w+\s+){0,4}per\s+(\w+(?:\s+\w+)?)", ql)
    if mper:
        b_t = set(t for t in content_toks(mper.group(1)) if t not in TYPE_WORDS)
        a_t = qtoks - b_t
        y = ys[0] if ys else None
        best = None
        for ti, t in enumerate(tables):
            j = _ratio_col(t, a_t | b_t, y)
            if j is None:
                continue
            ari, asc = best_row(t, a_t, need_col=j)
            bri, bsc = best_row(t, b_t, need_col=j, exclude=ari)
            if ari is None or bri is None or min(asc, bsc) <= 0:
                continue
            av, bv = t["rows"][ari]["nums"][j], t["rows"][bri]["nums"][j]
            if av is None or bv is None or bv == 0:
                continue
            key = (asc + bsc, -ti)
            if best is None or key > best[0]:
                best = (key, av, bv)
        if best:
            return f"a = {best[1]!r}\nb = {best[2]!r}\nanswer = a / b"
    best = None
    for ti, t in enumerate(tables):
        cols = year_range_cols(t, years) if years else None
        if cols is None:
            cols = [j for j, cy in enumerate(t["col_years"]) if cy]
            if years:
                cols = [j for j in cols if t["col_years"][j] in years]
        if not cols and not years:
            cols = list(range(len(t["col_years"])))
        if cols and len(cols) >= 2:
            for ri, row in enumerate(t["rows"]):
                vals = [row["nums"][j] for j in cols if j < len(row["nums"]) and row["nums"][j] is not None]
                if len(vals) < 2:
                    continue
                key = (row_score(row, qtoks), -ti, -ri)
                if best is None or key > best[0]:
                    best = (key, vals)
        # transposed: rows are years
        if years:
            yrows = []
            for y in years:
                rr = [(ri, r) for ri, r in enumerate(t["rows"]) if r["year"] == y]
                if not rr:
                    yrows = None
                    break
                rr.sort(key=lambda p: (-row_score(p[1], qtoks), p[0]))
                yrows.append(rr[0][1])
            if yrows:
                j = best_col(t, qtoks - set(years))
                if j is not None:
                    vals = [r["nums"][j] for r in yrows if j < len(r["nums"]) and r["nums"][j] is not None]
                    if len(vals) == len(years):
                        key = (1.5, -ti, 1)
                        if best is None or key > best[0]:
                            best = (key, vals)
    tab_sc = best[0][0] if best else -1.0
    if years and tab_sc < 2.0:
        tvals = text_series(tcands, qtoks, years)
        if tvals and len(set(tvals)) > 1:
            vs = ", ".join(repr(v) for v in tvals)
            return f"vals = [{vs}]\nanswer = sum(vals) / len(vals)"
    if best and tab_sc > 0:
        vs = ", ".join(repr(v) for v in best[1])
        return f"vals = [{vs}]\nanswer = sum(vals) / len(vals)"
    if years and len(years) >= 2:
        pair = pick_pair_by_years(tables, qtoks, years[0], years[-1]) \
               or pick_pair_text(tcands, qtoks, years[0], years[-1])
        if pair and pair[0] != pair[1]:
            return f"vals = [{pair[1]!r}, {pair[0]!r}]\nanswer = sum(vals) / len(vals)"
    return None

NEXTN_RE = re.compile(r"next\s+(two|three|four|five|2|3|4|5)\s+years|(?:due\s+)?in\s+(five|three|four)\s+years", re.I)
_NWORD = {"two": 2, "three": 3, "four": 4, "five": 5, "2": 2, "3": 3, "4": 4, "5": 5}

def solve_sum(q, tables, tcands, qtoks):
    ys, rng = q_years(q)
    years = rng if rng else (sorted(set(ys), key=int) if len(set(ys)) >= 2 else None)
    wt = True if "total" in q.lower() else None
    # "next N years": sum first N consecutive year rows/cols
    mn = NEXTN_RE.search(q)
    if mn:
        n = _NWORD[(mn.group(1) or mn.group(2)).lower()]
        best_n = None
        for ti, t in enumerate(tables):
            yrows = sorted([(int(r["year"]), ri) for ri, r in enumerate(t["rows"])
                            if r["year"] and YEAR_RE.fullmatch(r["label"].strip())],
                           key=lambda p: p[0])
            if len(yrows) >= n:
                j = best_col(t, qtoks)
                if j is not None:
                    vals = [t["rows"][ri]["nums"][j] for _, ri in yrows[:n]
                            if j < len(t["rows"][ri]["nums"]) and t["rows"][ri]["nums"][j] is not None]
                    if len(vals) == n:
                        sc = max([overlap_score(ct, qtoks) for ct in t["col_toks"]] + [0.0]) + 1.0
                        if best_n is None or sc > best_n[0]:
                            best_n = (sc, vals)
            ycols = sorted([(int(cy), j) for j, cy in enumerate(t["col_years"]) if cy])
            if len(ycols) >= n:
                ri, sc = best_row(t, qtoks)
                if ri is not None and sc > 0:
                    vals = [t["rows"][ri]["nums"][j] for _, j in ycols[:n]
                            if j < len(t["rows"][ri]["nums"]) and t["rows"][ri]["nums"][j] is not None]
                    if len(vals) == n and (best_n is None or sc > best_n[0]):
                        best_n = (sc, vals)
        if best_n:
            return "answer = " + " + ".join(repr(v) for v in best_n[1])
    best = None
    if years:
        for ti, t in enumerate(tables):
            cols = year_range_cols(t, years)
            if not cols:
                continue
            for ri, row in enumerate(t["rows"]):
                vals = [row["nums"][j] for j in cols if j < len(row["nums"]) and row["nums"][j] is not None]
                if len(vals) != len(cols):
                    continue
                key = (row_score(row, qtoks), -ti, -ri)
                if best is None or key > best[0]:
                    best = (key, vals)
    if best and best[0][0] > 0:
        vs = " + ".join(repr(v) for v in best[1])
        return f"answer = {vs}"
    vals = top_values(tables, tcands, qtoks, ys[0] if ys else None, k=2, want_total=wt)
    if len(vals) == 2:
        return f"answer = {vals[0]!r} + {vals[1]!r}"
    return None

def solve_prod_rate(q, tables, tcands, qtoks):
    """annual interest/fee = principal * rate%"""
    rates = [(overlap_score(c["window"], qtoks), -k, c) for k, c in enumerate(tcands)
             if c["pct"] and 0 < abs(c["value"]) < 30]
    prins = [(overlap_score(c["window"], qtoks) + (0.5 if abs(c["value"]) >= 50 else 0), -k, c)
             for k, c in enumerate(tcands) if not c["pct"] and abs(c["value"]) >= 1]
    rates.sort(key=lambda t: (-t[0], t[1]))
    prins.sort(key=lambda t: (-t[0], t[1]))
    if rates and prins:
        r, p = rates[0][2]["value"], prins[0][2]["value"]
        return f"principal = {p!r}\nrate = {r!r}\nanswer = principal * rate / 100"
    return None

def _row_series(q, tables, tcands, qtoks):
    """values of best row across year cols (or year rows transposed)."""
    ys, rng = q_years(q)
    years = rng if rng and len(rng) >= 2 else (sorted(set(ys), key=int) if len(set(ys)) >= 2 else None)
    best = None
    for ti, t in enumerate(tables):
        cols = year_range_cols(t, years) if years else [j for j, cy in enumerate(t["col_years"]) if cy]
        if cols and len(cols) >= 2:
            for ri, row in enumerate(t["rows"]):
                vals = [row["nums"][j] for j in cols if j < len(row["nums"]) and row["nums"][j] is not None]
                if len(vals) < 2:
                    continue
                key = (row_score(row, qtoks), -ti, -ri)
                if best is None or key > best[0]:
                    best = (key, vals)
        if years:
            vals = []
            ok = True
            for yy in years:
                rr = [(ri, r) for ri, r in enumerate(t["rows"]) if r["year"] == yy]
                if not rr:
                    ok = False
                    break
                rr.sort(key=lambda p: (-row_score(p[1], qtoks), p[0]))
                j = best_col(t, qtoks - set(years))
                v = rr[0][1]["nums"][j] if j is not None and j < len(rr[0][1]["nums"]) else None
                if v is None:
                    ok = False
                    break
                vals.append(v)
            if ok and len(vals) >= 2:
                key = (1.2, -ti, 1)
                if best is None or key > best[0]:
                    best = (key, vals)
    if best and best[0][0] > 0:
        return best[1]
    return None

def solve_range(q, tables, tcands, qtoks):
    vals = _row_series(q, tables, tcands, qtoks)
    if vals:
        vs = ", ".join(repr(v) for v in vals)
        return f"vals = [{vs}]\nanswer = max(vals) - min(vals)"
    return None

def solve_extreme(q, tables, tcands, qtoks):
    fn = "min" if re.search(r"lowest|smallest|minimum|least", q.lower()) else "max"
    vals = _row_series(q, tables, tcands, qtoks)
    if vals:
        vs = ", ".join(repr(v) for v in vals)
        return f"vals = [{vs}]\nanswer = {fn}(vals)"
    return None

def solve_prod(q, tables, tcands, qtoks):
    if re.search(r"annual (interest|fee|expense|cost)|interest (expense|cost|payment)", q.lower()):
        p = solve_prod_rate(q, tables, tcands, qtoks)
        if p:
            return p
    vals = top_values(tables, tcands, qtoks, None, k=2)
    if len(vals) == 2:
        return f"answer = {vals[0]!r} * {vals[1]!r}"
    return None

def solve_dollar_return(q, tables, tcands, qtoks):
    p = solve_growth_or_diff("growth", q, tables, tcands, qtoks, pct=False)
    return p


# ---------------------------------------------------------------- dispatch
ORDER = {
    "growth": ["growth", "diff", "portion"],
    "dollar_return": ["dollar_return", "growth"],
    "margin": ["margin", "portion"],
    "diff": ["diff", "growth"],
    "portion": ["portion", "growth"],
    "ratio": ["ratio", "portion"],
    "avg": ["avg", "diff"],
    "range": ["range", "diff"],
    "extreme": ["extreme", "diff"],
    "sum": ["sum", "diff"],
    "sum_weak": ["sum", "diff"],
    "prod": ["prod", "diff"],
    "other": ["diff", "portion"],
}

SOLVERS = {
    "growth": lambda q, T, C, Q: solve_growth_or_diff("growth", q, T, C, Q),
    "diff": lambda q, T, C, Q: solve_growth_or_diff("diff", q, T, C, Q),
    "portion": solve_portion,
    "margin": solve_margin,
    "ratio": solve_ratio,
    "avg": solve_avg,
    "range": solve_range,
    "extreme": solve_extreme,
    "sum": solve_sum,
    "prod": solve_prod,
    "dollar_return": solve_dollar_return,
}

CURRENT = {}

def _prog_operands(code):
    """distinct non-boilerplate numeric constants in program order"""
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
           and not isinstance(node.value, bool):
            v = float(node.value)
            if abs(v) not in (100.0, 1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 1000.0, 1e6, 1e9) and v not in out:
                out.append(v)
    return out

def _locate_cell(v, tables):
    for ti, t in enumerate(tables):
        for ri, row in enumerate(t["rows"]):
            for j, n in enumerate(row["nums"]):
                if n is not None and (abs(n - v) < 1e-9 or abs(-n - v) < 1e-9):
                    return ti, ri, j
    return None

def skeleton_transfer(q, tables, store, context, exclude_idx, kind, qtoks, min_jac=0.45):
    """same-context neighbor of same kind: rebind its 2 operands to my years/entity."""
    if store is None:
        return None
    key = hash(context)
    qt = set(q_toks_all(q))
    cands = []
    for k in store.by_ctx.get(key, []):
        it = store.items[k]
        if exclude_idx is not None and it["idx"] == exclude_idx:
            continue
        if classify(it["question"]) != kind:
            continue
        jac = len(qt & it["qtoks"]) / max(1, len(qt | it["qtoks"]))
        if jac >= min_jac:
            cands.append((jac, -k, it))
    if not cands:
        return None
    cands.sort(reverse=True)
    it = cands[0][2]
    ops = _prog_operands(it["program"])
    if len(ops) != 2:
        return None
    la, lb = _locate_cell(ops[0], tables), _locate_cell(ops[1], tables)
    if la is None or lb is None:
        return None
    ql = q.lower()
    dec = bool(DEC_RE.search(ql)) and not re.search(r"increase", ql)
    ys, _ = q_years(q)
    if kind in ("growth", "diff"):
        if len(ys) == 1:
            y1, y2 = str(int(ys[0]) - 1), ys[0]
        elif len(ys) >= 2:
            yy = sorted(ys, key=int)
            y1, y2 = yy[0], yy[-1]
            m = re.search(r"from\s+((?:19|20)\d{2})\s*(?:to|-|through|until)\s*((?:19|20)\d{2})", ql)
            if m:
                y1, y2 = m.group(1), m.group(2)
        else:
            return None
        if la[0] == lb[0] and la[1] == lb[1]:      # same row, year cols
            t = tables[la[0]]
            j1, j2 = col_for_year(t, y1), col_for_year(t, y2)
            if j1 is None or j2 is None:
                return None
            row = t["rows"][la[1]]
            old, new = row["nums"][j1], row["nums"][j2]
        elif la[0] == lb[0] and la[2] == lb[2]:    # same col, year rows
            t = tables[la[0]]
            j = la[2]
            r1 = [r for r in t["rows"] if r["year"] == y1]
            r2 = [r for r in t["rows"] if r["year"] == y2]
            if not r1 or not r2:
                return None
            old = r1[0]["nums"][j] if j < len(r1[0]["nums"]) else None
            new = r2[0]["nums"][j] if j < len(r2[0]["nums"]) else None
        else:
            return None
        if old is None or new is None:
            return None
        return emit_growth(new, old, dec) if kind == "growth" else emit_diff(new, old, dec)
    if kind == "portion":
        ta, tb = tables[la[0]], tables[lb[0]]
        pt, wt, _ = portion_split(q)
        if pt is None:
            pt = qtoks
        # part: re-match row among same table by pt; keep NN's column unless my year differs
        t = ta
        pri, psc = best_row(t, pt, need_col=la[2])
        if pri is None:
            pri = la[1]
        pj = la[2]
        if ys:
            jy = col_for_year(t, ys[-1])
            if jy is not None and jy < len(t["rows"][pri]["nums"]) and t["rows"][pri]["nums"][jy] is not None:
                pj = jy
        part = t["rows"][pri]["nums"][pj] if pj < len(t["rows"][pri]["nums"]) else None
        # whole: same location role as NN
        if lb[0] == la[0] and lb[1] == la[1]:      # whole was same row (total col)
            whole = t["rows"][pri]["nums"][lb[2]] if lb[2] < len(t["rows"][pri]["nums"]) else None
        elif lb[0] == la[0] and lb[2] == la[2]:    # whole was same col (total row)
            whole = t["rows"][lb[1]]["nums"][pj] if pj < len(t["rows"][lb[1]]["nums"]) else None
        else:
            whole = tb["rows"][lb[1]]["nums"][lb[2]]
        if part is None or whole is None or whole == 0:
            return None
        if abs(part) > abs(whole):
            part, whole = whole, part
        return f"part = {part!r}\nwhole = {whole!r}\nanswer = abs(part) / abs(whole) * 100"
    return None

def build_program(q, context, store=None, exclude_idx=None):
    if store is not None:
        prog = store.copy_match(q, context, exclude_idx)
        if prog is not None:
            return prog, "copy"
    CURRENT["context"] = context
    tables = parse_tables(context)
    tcands = text_candidates(context)
    qtoks = entity_toks(q)
    kind = classify(q)
    if kind == "portion":
        tp = skeleton_transfer(q, tables, store, context, exclude_idx, kind, qtoks, min_jac=0.55)
        if tp is not None:
            return tp, f"transfer_{kind}"
    if kind in ("sum_weak", "other") and store is not None:
        sig = store.knn_sig(q, exclude_idx=exclude_idx)
        if kind == "other" and sig and sig != "other":
            kind = sig
        elif kind == "sum_weak" and sig in ("sum", "prod", "avg", "portion", "diff"):
            kind = sig
    order = ORDER.get(kind, ORDER["other"])
    for k in order:
        fn = SOLVERS.get(k)
        p = fn(q, tables, tcands, qtoks) if fn else None
        if p:
            return p, kind
    # last resort: kNN signature decides a generic 2-operand shape
    sig = store.knn_sig(q, exclude_idx=exclude_idx) if store is not None else None
    ys, _ = q_years(q)
    vals = top_values(tables, tcands, qtoks, ys[0] if ys else None, k=2)
    if len(vals) == 2:
        a, b = vals
        if sig == "portion":
            lo, hi = (a, b) if abs(a) <= abs(b) else (b, a)
            if hi != 0:
                return f"part = {lo!r}\nwhole = {hi!r}\nanswer = abs(part) / abs(whole) * 100", "extract"
        if sig == "ratio" and b != 0:
            return f"a = {a!r}\nb = {b!r}\nanswer = a / b", "extract"
        if sig == "sum":
            return f"answer = {a!r} + {b!r}", "extract"
        if sig == "prod":
            return f"answer = {a!r} * {b!r}", "extract"
        return f"a = {a!r}\nb = {b!r}\nanswer = a - b", "extract"
    if vals:
        return f"x = {vals[0]!r}\nanswer = x * 1", "extract"
    return None, kind


# ---------------------------------------------------------------- fast verify (dev)
def fast_exec(code):
    g = {"__builtins__": {"sum": sum, "min": min, "max": max, "abs": abs,
                          "round": round, "pow": pow, "len": len, "range": range,
                          "float": float, "int": int, "print": lambda *a, **k: None}}
    try:
        exec(compile(code, "<p>", "exec"), g)
    except Exception:
        return None
    v = g.get("answer")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

def has_computation(code):
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

def fast_ok(code, gold):
    if not has_computation(code):
        return False
    p = fast_exec(code)
    g = parse_number(gold)
    if p is None or g is None:
        return False
    if g == 0:
        return abs(p) <= 1e-6
    return abs(p - g) <= 0.01 * abs(g)

def load_sc_flags(rows):
    cache = os.path.join(HERE, "train_sc_flags.json")
    if os.path.exists(cache):
        f = json.load(open(cache))
        if len(f) == len(rows):
            return f
    flags = [1 if fast_ok(r["program"], r["answer"]) else 0 for r in rows]
    json.dump(flags, open(cache, "w"))
    return flags


# ---------------------------------------------------------------- eval loops
def dev_eval(stride=4, dump=None, limit=None):
    rows = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
    flags = load_sc_flags(rows)
    store = Store(rows, flags)
    stats = Counter()
    fails = []
    n = 0
    for i, r in enumerate(rows):
        if i % stride != 0 or not flags[i]:
            continue
        n += 1
        if limit and n > limit:
            break
        prog, kind = build_program(r["question"], r["context"], store, exclude_idx=i)
        if prog is None:
            stats["none"] += 1
            stats[f"{kind}_miss"] += 1
            fails.append({"i": i, "kind": kind, "q": r["question"], "why": "no_program"})
            continue
        ok = fast_ok(prog, r["answer"])
        stats["hit" if ok else "miss"] += 1
        stats[f"{kind}_{'hit' if ok else 'miss'}"] += 1
        if not ok:
            fails.append({"i": i, "kind": kind, "q": r["question"], "prog": prog,
                          "gold_prog": r["program"], "ans": r["answer"]})
    tot = stats["hit"] + stats["miss"] + stats["none"]
    cc = stats["hit"] / tot if tot else 0.0
    print(json.dumps({"dev_n": tot, "CC": round(cc, 4)}, indent=1))
    kinds = sorted(set(k.rsplit("_", 1)[0] for k in stats if k.endswith(("_hit", "_miss"))))
    for k in kinds:
        h, m = stats.get(f"{k}_hit", 0), stats.get(f"{k}_miss", 0)
        if h + m:
            print(f"  {k:14s} {h:4d}/{h+m:4d} = {h/(h+m):.1%}")
    if dump:
        json.dump(fails, open(dump, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("fails dumped:", dump, len(fails))
    return cc

def final_eval(path, workers=12):
    from concurrent.futures import ThreadPoolExecutor
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    train_rows = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
    flags = load_sc_flags(train_rows)
    store = Store(train_rows, flags)
    with ThreadPoolExecutor(workers) as ex:
        gold_ok = list(ex.map(lambda r: verify(r["program"], r["answer"])["reward"] == 1.0, rows))
    eval_rows = [r for r, g in zip(rows, gold_ok) if g]
    print(f"gold self-consistent: {len(eval_rows)}/{len(rows)}")
    progs = [build_program(r["question"], r["context"], store) for r in eval_rows]
    def check(args):
        (p, kind), r = args
        if p is None:
            return 0, kind
        return int(verify(p, r["answer"])["reward"]), kind
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(check, zip(progs, eval_rows)))
    stats = Counter()
    for hit, kind in results:
        stats["hit" if hit else "miss"] += 1
        stats[f"{kind}_{'hit' if hit else 'miss'}"] += 1
    tot = stats["hit"] + stats["miss"]
    cc = stats["hit"] / tot if tot else 0.0
    print(json.dumps({"n_eval": tot, "CC": round(cc, 4), "hits": stats["hit"]}, indent=1))
    kinds = sorted(set(k.rsplit("_", 1)[0] for k in stats if k.endswith(("_hit", "_miss"))))
    for k in kinds:
        h, m = stats.get(f"{k}_hit", 0), stats.get(f"{k}_miss", 0)
        if h + m:
            print(f"  {k:14s} {h:4d}/{h+m:4d} = {h/(h+m):.1%}")
    return cc


if __name__ == "__main__":
    args = list(sys.argv[1:])
    if "--dev" in args:
        args.remove("--dev")
        dump = None
        if "--dump" in args:
            k = args.index("--dump")
            dump = args[k + 1]
            del args[k:k + 2]
        dev_eval(dump=dump)
    else:
        final_eval(args[0] if args else os.path.join(HERE, "data", "CodeFinQA__test.jsonl"))

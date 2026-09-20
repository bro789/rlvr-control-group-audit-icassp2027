# -*- coding: utf-8 -*-
"""CodeTAT-QA 的确定性 solver —— 协议 Δ(m) max 集合中的 Acc(solver) 臂。

接口与 solver_v2_combo 保持一致（Store / build_program），故 solver_arm.py 只需
换 import 即可复用其外壳与泄漏防御。

设计原则见 PREREG_v2_ADDENDUM_CodeTATQA.md §4（冻结于任何结果产出之前）：
**good-faith 最强努力，不因任何中间结果调整强度。**

数据形态（实测 train 2856 条，写 solver 前完成）：
  · 96.1%（2746 条）只引用 2 个单元格
  · 位置关系：同列不同行 2115、同行不同列 593、行列都不同 38
  · 末行运算：diff 1456、pct/portion 1117、sum 149、ratio 76
  · **表格转置方向不固定**：有的表外层 key 是时间（"At 30 June 2019 -- Cost"），
    有的表外层 key 是实体（"All other fees (3)" → 内层才是 "2019"）。
    故必须自适应判断哪一层是时间轴，不能写死。

策略（按优先级）：
  1. copy_match：同 context 下找最相似 question，直接复用其 gold program（检索库越大越强）
  2. 规则路径：定位单元格对 → 按 classify(q) 套运算模板
  3. 兜底：question 年份/实体的最佳猜测
"""
import os, sys, json, re, math
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 任务无关的通用件，直接复用已迭代成熟的实现（预注册 §4.2-1：不从零写）
from solver_v2_combo import Store, q_toks_all, classify, q_years

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
# 注意：不要把 total / net / other 这类词放进 STOP —— 财务表格里它们是**关键区分词**
# （"Total inventories" vs "Inventories"、"Pension 2019" vs "Other 2019"）。
# 首版误删 total 导致 "total inventories" 定位不到，实测修复后命中率显著回升。
STOP = {"the", "of", "in", "a", "an", "and", "or", "to", "for", "was", "were", "is",
        "are", "what", "how", "much", "many", "did", "does", "s",
        "between", "from", "at", "on", "by", "with", "as", "that", "this", "it"}


# ---------------- 标签归一化与匹配（预注册 §4.2-2：最强匹配） ----------------
def norm_label(s):
    """归一化标签：小写、去脚注标记 (1)/(a)、去多余标点与空白。"""
    s = str(s).lower()
    s = re.sub(r"\(\s*[a-z0-9]{1,3}\s*\)", " ", s)      # 脚注 (3) (a)
    s = re.sub(r"--", " ", s)                            # 层级分隔符
    s = re.sub(r"[^\w\s.%-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def label_toks(s):
    return [t for t in norm_label(s).split() if t and t not in STOP]


def is_time_like(s):
    """该标签是否像时间轴（含 4 位年份，或典型期间词）。"""
    sl = str(s).lower()
    if YEAR_RE.search(sl):
        return True
    return bool(re.search(r"\b(fy|q[1-4]|quarter|month|year|period)\b", sl))


def label_score(qtoks_set, qcount, label, idf):
    """question 词元 vs 标签的 IDF 加权重叠度。"""
    lt = label_toks(label)
    if not lt:
        return 0.0
    hit = sum(idf.get(t, 1.0) for t in set(lt) if t in qtoks_set)
    denom = sum(idf.get(t, 1.0) for t in set(lt)) or 1.0
    cov = hit / denom
    # 覆盖率之外，再奖励长标签的完整命中（避免短标签靠一个词抢赢）
    return cov * (1.0 + 0.15 * math.log(1 + len(set(lt) & qtoks_set)))


# ---------------- 表格解析 ----------------
def parse_ctx(context):
    """context 是 JSON 字符串（dict-of-dicts）。返回 (table, outer_is_time)。"""
    try:
        t = json.loads(context) if isinstance(context, str) else context
    except Exception:
        return None, False
    if not isinstance(t, dict) or not t:
        return None, False
    outer = list(t.keys())
    inner = []
    for v in t.values():
        if isinstance(v, dict):
            inner.extend(v.keys())
    if not inner:
        return None, False
    o_time = sum(is_time_like(k) for k in outer) / max(1, len(outer))
    i_time = sum(is_time_like(k) for k in set(inner)) / max(1, len(set(inner)))
    return t, (o_time > i_time)


def cell_value(table, outer, inner):
    try:
        v = table[outer][inner]
    except Exception:
        return None
    return to_num(v)


def to_num(v):
    """把单元格值转成数，失败返回 None。'-'/''/None 视为缺失。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in ("", "-", "—", "–", "n/a", "N/A"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    m = NUM_RE.search(s.replace("(", "").replace(")", ""))
    if not m:
        return None
    try:
        x = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    return -x if neg else x


# ---------------- 单元格定位 ----------------
def build_idf(table):
    """用表内全部标签词元建一个轻量 IDF，抑制 'total''net' 这类高频词。"""
    docs = []
    for o, row in table.items():
        docs.append(label_toks(o))
        if isinstance(row, dict):
            for i in row:
                docs.append(label_toks(i))
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    n = max(1, len(docs))
    return {t: math.log(n / (1 + c)) + 1.0 for t, c in df.items()}


def rank_labels(labels, qtoks_set, qcount, idf, topk=6):
    scored = [(label_score(qtoks_set, qcount, l, idf), l) for l in labels]
    scored.sort(key=lambda x: -x[0])
    return [(s, l) for s, l in scored[:topk] if s > 0]


def match_time_label(labels, year, qset=None, idf=None):
    """在标签集合里找对应指定年份的那个。

    只按年份匹配不够：同一年份可能有多个修饰列（"Pension 2019" vs "Other 2019"），
    此时用 question 词元打分区分——首版没做，导致 pension 类题目选错列。
    """
    ys = str(year)
    cands = [l for l in labels if ys in str(l)]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    if qset:
        idf = idf or {}
        scored = []
        for l in cands:
            lt = set(label_toks(l)) - {ys}
            hit = sum(idf.get(t, 1.0) for t in lt & qset)
            scored.append((hit, -len(str(l)), l))
        scored.sort(reverse=True)
        if scored[0][0] > 0:
            return scored[0][2]
    # 无区分线索：取最短最干净的（"2019" 胜过 "At 30 June 2019 -- Cost"）
    return min(cands, key=lambda l: len(str(l)))


def locate_pair(q, table, outer_is_time, kind="other"):
    """定位两个单元格。返回 [(outer,inner), (outer,inner)] 或 None。

    四种模式（覆盖实测 96% 的两格题：同列不同行 2115 / 同行不同列 593）：
      A  同实体两时间：question 给了两个年份
      A' 同实体两时间：question 只给一个年份（"change in X in 2019" 隐含对比上一年）
      B  两实体同时间：时间固定，实体取 question 命中的前二
      C  同实体两属性列：无时间轴区分，靠 question 的属性词选两个不同外层列
         （如 "difference between opening and closing net book amount"）

    **模式顺序由 kind 驱动**（首版固定 A→B→A' 是算错的主因）：
    diff/growth 这类"变化"题优先走同实体两时间；sum/avg 这类"合计"题优先走两实体。
    """
    idf = build_idf(table)
    qt = q_toks_all(q)
    qset, qcount = set(qt), Counter(qt)
    years, _ = q_years(q)
    outers = list(table.keys())
    inners = sorted({i for v in table.values() if isinstance(v, dict) for i in v})

    ent_labels = outers if not outer_is_time else inners
    time_labels = inners if not outer_is_time else outers

    def mk(ent, tm):
        return (ent, tm) if not outer_is_time else (tm, ent)

    def mode_A():
        if len(years) < 2:
            return None
        t1 = match_time_label(time_labels, years[0], qset, idf)
        t2 = match_time_label(time_labels, years[1], qset, idf)
        if t1 and t2 and t1 != t2:
            for _, ent in rank_labels(ent_labels, qset, qcount, idf):
                if cell_value(table, *mk(ent, t1)) is not None and \
                   cell_value(table, *mk(ent, t2)) is not None:
                    return [mk(ent, t1), mk(ent, t2)]
        return None

    def mode_Aprime():
        """单年份（或无年份）+ 该实体在多个时间上都有值 → 同实体跨两个时间。"""
        if len(time_labels) < 2:
            return None
        for _, ent in rank_labels(ent_labels, qset, qcount, idf):
            avail = [t for t in time_labels if cell_value(table, *mk(ent, t)) is not None]
            if len(avail) < 2:
                continue
            if years:
                t1 = match_time_label(avail, years[0], qset, idf) or avail[0]
            else:
                t1 = avail[0]
            others = [t for t in avail if t != t1]
            if others:
                return [mk(ent, t1), mk(ent, others[0])]
        return None

    def mode_B():
        cands = rank_labels(ent_labels, qset, qcount, idf, topk=8)
        tm = match_time_label(time_labels, years[0], qset, idf) if years else None
        if tm is None and time_labels:
            tr = rank_labels(time_labels, qset, qcount, idf, topk=1)
            tm = tr[0][1] if tr else (time_labels[0] if len(time_labels) == 1 else None)
        if tm is None or len(cands) < 2:
            return None
        ok = [ent for _, ent in cands if cell_value(table, *mk(ent, tm)) is not None]
        if len(ok) >= 2:
            return [mk(ok[0], tm), mk(ok[1], tm)]
        return None

    def mode_C():
        """同一实体（内层），两个不同的外层属性列——靠 question 的属性词区分。
        典型："difference between total opening and closing net book amount in 2018"
        两列都含 2018，靠 opening/closing 区分，纯年份匹配拿不到。"""
        col_labels, row_labels = (outers, inners) if not outer_is_time else (inners, outers)
        # 这里"列"= 外层，"行"= 内层；实体取内层命中最高者
        ranked_rows = rank_labels(row_labels, qset, qcount, idf, topk=4)
        ranked_cols = rank_labels(col_labels, qset, qcount, idf, topk=6)
        if len(ranked_cols) < 2 or not ranked_rows:
            return None
        for _, row in ranked_rows:
            ok = []
            for _, col in ranked_cols:
                o, i = (col, row) if not outer_is_time else (row, col)
                if cell_value(table, o, i) is not None:
                    ok.append((o, i))
                if len(ok) == 2:
                    return ok
        return None

    # 模式顺序：**年份个数是比 kind 更强的信号**（实测得出）。
    # question 明确给两个年份（"total X for 2018 and 2019"、"average X over 2018 and 2019"）
    # 时，几乎必然是"同实体跨两时间"，与 kind 是 sum/avg/diff 无关。
    # 首版只按 kind 排序，把这类题送进"两实体"模式，是定位错误的主要来源。
    if len(years) >= 2:
        order = [mode_A, mode_Aprime, mode_C, mode_B]
    elif len(years) == 0:
        # 无时间线索 → 多半是两个实体并列，或同实体的两个属性列
        order = [mode_B, mode_C, mode_Aprime, mode_A]
    elif kind in ("growth", "diff", "dollar_return", "range"):
        # 单年份 + 变化类 → "change in X in 2019" 隐含与上一年比
        order = [mode_Aprime, mode_A, mode_C, mode_B]
    elif kind in ("sum", "sum_weak", "avg", "prod"):
        order = [mode_B, mode_Aprime, mode_C, mode_A]
    else:  # portion / margin / ratio / other
        order = [mode_B, mode_C, mode_Aprime, mode_A]

    for fn in order:
        try:
            r = fn()
        except Exception:
            r = None
        if r:
            return r
    return None


# ---------------- 程序生成 ----------------
CHANGE_RE = re.compile(r"\b(change|increase|decrease|decline|growth|drop|difference|"
                       r"rise|fell|fall|grew|reduction|variance)\b")
PCT_Q_RE = re.compile(r"percent|%")


def refine_kind(q, kind):
    """修正 classify 在本数据集上的系统性误判。

    classify 是为 CodeFinQA 设计的，靠 question 里的关键词判运算意图。但 CodeTAT-QA 的
    **实体名本身**常含运算词——"Diluted weighted-average shares"（→误判 avg）、
    "future minimum commitments"（→误判 extreme）、"Total inventories"（→误判 sum）——
    而问题其实都是 "what is the change in ..."。实测这类误判会连锁造成运算错+方向错。

    规则：出现明确的「变化类」动词/名词时，覆盖掉由实体名误触发的 avg/extreme/sum/range。
    不覆盖 portion/ratio/margin（"X as a percentage of Y" 本就不是变化题）。
    """
    ql = q.lower()
    if kind in ("avg", "extreme", "sum", "sum_weak", "range", "other") and CHANGE_RE.search(ql):
        return "growth" if PCT_Q_RE.search(ql) else "diff"
    return kind


def _ref(o, i):
    return f"df[{o!r}][{i!r}]"


def emit(cells, kind, table):
    """按运算类型生成程序。cells = [(o1,i1), (o2,i2)]，第一个视为「新/部分」。"""
    (o1, i1), (o2, i2) = cells
    a, b = cell_value(table, o1, i1), cell_value(table, o2, i2)
    if a is None or b is None:
        return None
    A, B = _ref(o1, i1), _ref(o2, i2)
    head = f"a = {A}\nb = {B}\n"

    if kind in ("growth", "dollar_return"):
        if b == 0:
            return None
        return head + "answer = (a - b) / b * 100.0"
    if kind in ("portion", "margin"):
        # 占比：小的作分子（part/whole）
        if abs(a) > abs(b):
            head = f"a = {B}\nb = {A}\n"
            a, b = b, a
        if b == 0:
            return None
        return head + "answer = a / b * 100.0"
    if kind == "ratio":
        if b == 0:
            return None
        return head + "answer = a / b"
    if kind == "avg":
        return head + "answer = (a + b) / 2"
    if kind in ("sum", "sum_weak"):
        # sum_weak（"total X for 2018 and 2019" 这类弱信号求和）此前未被覆盖，
        # 落到函数末尾的默认 diff 分支 → 求和题被算成减法，实测占失败的 11/105。
        return head + "answer = a + b"
    if kind == "prod":
        return head + "answer = a * b"
    if kind == "range":
        return head + "answer = abs(a - b)"
    if kind == "extreme":
        return head + "answer = max(a, b)"
    # diff 及其余：默认求差（实测占比最高的形态）
    return head + "answer = a - b"


def _rule_program(q, context):
    """纯规则路径：解析表格 → 定位单元格 → 按运算类型生成。返回 (prog, kind) 或 (None, why)。"""
    table, outer_is_time = parse_ctx(context)
    if not table:
        return None, "no_table"
    kind = refine_kind(q, classify(q))
    cells = locate_pair(q, table, outer_is_time, kind)
    if cells is None:
        return None, "no_cells"
    # 时间序：diff/growth 有向（新 − 旧），按标签年份大小排（不依赖 question 出现顺序）
    if kind in ("growth", "diff", "dollar_return"):
        def cell_year(c):
            ys = [int(m.group(0)) for m in re.finditer(r"\b(?:19|20)\d{2}\b", f"{c[0]} {c[1]}")]
            return max(ys) if ys else None
        y0, y1 = cell_year(cells[0]), cell_year(cells[1])
        if y0 is not None and y1 is not None and y0 != y1 and y0 < y1:
            cells = [cells[1], cells[0]]
    p = emit(cells, kind, table)
    if p is None:
        p = emit(cells, "diff", table)
    return (p, kind) if p else (None, "emit_fail")


def build_program(q, context, store=None, exclude_idx=None):
    """返回 (program, kind)。接口与 solver_v2_combo.build_program 一致。

    **规则优先、检索兜底**（2026-07-24 修正，见 PREREG 附录 §4.5）：
    实测 solver scarce 曲线随检索库增大而**下降**（m=0 纯规则 72% → m=2446 全库 60.5%），
    因为 copy_match 在大库时匹配到"表面相似实际不同"的 gold，污染了规则的正确答案。
    纯规则(72%)已强于任何带检索(≤60.5%)。故改为：规则先跑，只在规则解不出时才用
    copy_match 兜底 —— 保证 solver ≥ 纯规则，检索只增不减。这是"最强努力"承诺的应有实现，
    方向上提高了 solver 门槛（对 RL 更不利），非为结果调参。
    """
    # 1) 规则路径优先（不被坏检索污染）
    p, kind = _rule_program(q, context)
    if p is not None:
        return p, kind
    # 2) 规则解不出 → copy_match 兜底（相似题的 gold program）
    if store is not None:
        try:
            prog = store.copy_match(q, context, exclude_idx)
        except Exception:
            prog = None
        if prog is not None:
            return prog, "copy"
    return None, kind


def load_sc_flags(rows):
    """gold 自洽标记：每条 gold 是否通过 verifier（决定能否入检索库）。

    solver_arm.build_store 需要它。CodeTAT-QA 的 gold 引用 df，必须传 context 才能验。
    缓存按 len(rows) 校验，长度不符则重算——solver_arm 只对全量 rows 调用（子集用
    all_flags 切片，不重复调），故缓存长度稳定。
    """
    from verifier import verify
    cache = os.path.join(HERE, "train_sc_flags_tatqa.json")   # 独立缓存，不与 CodeFinQA 冲突
    if os.path.exists(cache):
        f = json.load(open(cache))
        if len(f) == len(rows):
            return f
    flags = [1 if verify(r["program"], r["answer"], True,
                         str(r.get("context")) if r.get("context") is not None else None)["reward"] == 1.0
             else 0 for r in rows]
    json.dump(flags, open(cache, "w"))
    return flags


if __name__ == "__main__":
    # 快速自评：对 train 前 N 条跑规则路径（无检索库 = m=0 的纯规则下界）
    from verifier import verify
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "CodeTAT-QA__train.jsonl"),
                                        encoding="utf-8")][:n]
    hit = 0
    kinds = Counter()
    for r in rows:
        p, k = build_program(r["question"], r["context"], None)
        kinds[k] += 1
        if p:
            v = verify(p, r["answer"], context=r["context"])
            hit += int(v["reward"] == 1.0)
    print(f"纯规则（无检索库）在 train 前 {len(rows)} 条: {hit}/{len(rows)} = {hit/len(rows)*100:.1f}%")
    print("kind 分布:", dict(kinds.most_common(8)))

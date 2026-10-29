"""Answer normalisation, equivalence grouping and grading for Task 3.

MATH      : math-verify (parse + verify) for mathematical equivalence.
MathVista : answer-type / choice / precision rules (option letters for multiple choice).
Groups are formed from candidate answers only (never from gold); grading applies the same equivalence to gold.
Missing extractions -> single NO_ANSWER group (id -1), always graded incorrect.
"""
import re
from functools import lru_cache

NO_ANSWER = -1
NA_STR = "NO_ANSWER"
SCORER_VERSION = "math-verify"


def scorer_version():
    try:
        import importlib.metadata as md
        return f"math-verify=={md.version('math-verify')}"
    except Exception:
        return "math-verify=unknown"


def is_missing(a):
    return a is None or (isinstance(a, float) and a != a) or str(a).strip() == "" or str(a) == NA_STR


# ------------------------------------------------------------------ MATH
def _cheap_key(a: str) -> str:
    s = str(a).strip().strip("$").strip()
    s = re.sub(r"\\(left|right|!|,|;|:|displaystyle)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s).rstrip(".")
    return s


@lru_cache(maxsize=200_000)
def _parse(a: str):
    from math_verify import parse
    try:
        return parse("$" + a + "$", parsing_timeout=None)
    except Exception:
        return []


@lru_cache(maxsize=500_000)
def math_equiv(a: str, b: str) -> bool:
    if _cheap_key(a) == _cheap_key(b):
        return True
    from math_verify import verify
    pa, pb = _parse(a), _parse(b)
    if not pa or not pb:
        return False
    try:
        return bool(verify(pa, pb, timeout_seconds=None))
    except Exception:
        return False


def group_math(answers):
    """answers: list[str|None]. Returns (group ids list[int], representatives dict gid->str)."""
    gids, reps, keymap = [], {}, {}
    for a in answers:
        if is_missing(a):
            gids.append(NO_ANSWER); continue
        a = str(a)
        k = _cheap_key(a)
        if k in keymap:
            gids.append(keymap[k]); continue
        found = None
        for gid, rep in reps.items():
            if math_equiv(a, rep):
                found = gid; break
        if found is None:
            found = len(reps); reps[found] = a
        keymap[k] = found
        gids.append(found)
    return gids, reps


# ------------------------------------------------------------------ MathVista
def _to_float(s):
    return float(re.sub(r"[,\s$%]", "", str(s)))


def mathvista_normalize(pred, meta):
    """meta: dict(question_type, answer_type, choices, precision). Returns canonical string or None."""
    if is_missing(pred):
        return None
    pred = str(pred).strip()
    qtype = (meta or {}).get("question_type", "free_form")
    atype = (meta or {}).get("answer_type", "text")
    choices = list((meta or {}).get("choices") or [])
    prec = (meta or {}).get("precision")
    if qtype == "multi_choice":
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        m = re.fullmatch(r"\(?([A-Za-z])\)?[\.\):]?", pred)
        if m and (not letters or m.group(1).upper() in letters):
            return m.group(1).upper()
        for i, c in enumerate(choices):
            if pred.strip().lower() == str(c).strip().lower():
                return letters[i]
        return pred.lower()
    try:
        if atype == "integer":
            return str(int(round(_to_float(pred))))
        if atype == "float":
            p = int(prec) if prec is not None and prec == prec else 2
            return f"{round(_to_float(pred), p):.{p}f}"
    except ValueError:
        pass
    return re.sub(r"\s+", " ", pred.lower()).strip()


def gold_to_canonical(gold, meta):
    """Gold for multiple choice is often the choice text: map to letter with the same rules."""
    return mathvista_normalize(gold, meta)


def group_strings(canon):
    gids, keymap = [], {}
    for a in canon:
        if a is None:
            gids.append(NO_ANSWER); continue
        gids.append(keymap.setdefault(a, len(keymap)))
    return gids, {v: k for k, v in keymap.items()}


# ------------------------------------------------------------------ boxed extraction (live generation)
def last_boxed(text):
    """Last \\boxed{...} with balanced braces; None if absent/unbalanced."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{": depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None

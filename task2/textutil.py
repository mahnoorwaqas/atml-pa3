"""VQA answer normalisation (compact re-implementation of the official processing) and local VQA scoring."""
import re

_NUM = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ART = {"a", "an", "the"}
_CONTR = {"dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't", "cant": "can't", "wont": "won't",
          "didnt": "didn't", "wasnt": "wasn't", "werent": "weren't", "im": "i'm", "ive": "i've", "thats": "that's",
          "whats": "what's", "theyre": "they're", "youre": "you're", "hasnt": "hasn't", "havent": "haven't", "shouldnt": "shouldn't"}
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def normalize_answer(a) -> str:
    if a is None:
        return ""
    s = str(a).replace("\n", " ").replace("\t", " ").strip().lower()
    out = s
    for p in _PUNCT:                                                  # as in the official VQA eval
        if (p + " " in s or " " + p in s) or re.search(r"(\d)(,)(\d)", s):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    s = out
    s = re.sub(r"(?!<=\d)(\.)(?!\d)", "", s)                        # periods not inside numbers
    words = []
    for w in s.split():
        w = _NUM.get(w, w)
        if w in _ART:
            continue
        words.append(_CONTR.get(w, w))
    return " ".join(words)


def vqa_soft_score(pred, refs) -> float:
    """s(a) = 1/10 sum_i min(1, (1/3) sum_{j != i} 1[a == a_j]) with all normalised."""
    a = normalize_answer(pred); R = [normalize_answer(r) for r in refs]
    tot = 0.0
    for i in range(len(R)):
        m = sum(1 for j in range(len(R)) if j != i and R[j] == a)
        tot += min(1.0, m / 3.0)
    return tot / len(R)


def em_score(pred, refs) -> float:
    return float(normalize_answer(pred) == normalize_answer(refs[0]))


_IRREG = {"person": "people", "bus": "buses", "knife": "knives", "sheep": "sheep", "skis": "skis", "scissors": "scissors",
          "broccoli": "broccoli", "mouse": "mice", "tv": "tvs", "potted plant": "potted plants", "wine glass": "wine glasses",
          "sandwich": "sandwiches", "skateboard": "skateboards", "toothbrush": "toothbrushes", "bench": "benches",
          "couch": "couches", "dining table": "dining tables"}


def pluralize(name: str) -> str:
    if name in _IRREG:
        return _IRREG[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"

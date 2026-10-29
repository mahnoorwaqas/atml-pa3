"""Task 2 evaluation: development metrics (loss / NLL / image-swap penalty) and the final four-condition grounding eval."""
import argparse, json, os, time
import numpy as np
import torch

from common.utils import SEED, save_json, append_jsonl, get_device
from .textutil import normalize_answer, vqa_soft_score, em_score
from .data import VQA_SUFFIX

CONDS = ["correct", "swap", "mismatch", "none"]


def _items(pairs):
    out = []
    for p in pairs:
        for k, it in enumerate(p["items"]):
            out.append({**it, "pair_id": p["pair_id"], "type": p["type"], "suite": p["suite"], "k": k,
                        "partner": p["items"][1 - k], "target": it["target"], "kind": "vqa"})
    return out


def _ex(it, image_id=None):
    return {"question": it["question"] + VQA_SUFFIX, "target": it["target"], "kind": "vqa", "image_id": image_id}


@torch.no_grad()
def answer_nll(model, store, items, cond, mm=None, bs=8):
    """Mean (over examples) of per-example mean token NLL of the reference answer, excluding termination."""
    res = []
    for s in range(0, len(items), bs):
        b = items[s:s + bs]
        if cond == "none":
            res += model.example_nll([_ex(i) for i in b], None, use_image=False, include_eos=False).tolist(); continue
        ids = [i["image_id"] if cond == "correct" else i["partner"]["image_id"] if cond == "swap" else mm[i["qid"]] for i in b]
        res += model.example_nll([_ex(i) for i in b], store.get(ids), True, include_eos=False).tolist()
    return np.array(res)


@torch.no_grad()
def dev_eval(model, ds, store):
    was = model.training; model.eval(); out = {}
    caps = ds["caps_dev"]
    out["caption_loss"] = float(np.mean(sum([model.example_nll(caps[s:s + 8], store.get([c["image_id"] for c in caps[s:s + 8]]), True).tolist()
                                             for s in range(0, len(caps), 8)], [])))
    for name, key in (("grounding", "grounding_dev"), ("natural", "vqa_dev")):
        items = _items(ds[key]); r = {c: answer_nll(model, store, items, c, mm=None) for c in ("correct", "swap", "none")}
        out[f"{name}_nll_correct"], out[f"{name}_nll_swap"], out[f"{name}_nll_none"] = (float(r[c].mean()) for c in ("correct", "swap", "none"))
        out[f"{name}_delta_nll"] = out[f"{name}_nll_swap"] - out[f"{name}_nll_correct"]     # >0: correct image makes reference likelier
    model.train(was)
    return out


def mismatch_map(pairs, seed=SEED):
    """Fixed unrelated-image mapping per question: an image of a different pair (excluding both original images)."""
    rng = np.random.default_rng(seed); mm = {}
    for i, p in enumerate(pairs):
        own = {it["image_id"] for it in p["items"]}
        others = [q for j, q in enumerate(pairs) if j != i and not ({x["image_id"] for x in q["items"]} & own)]
        for it in p["items"]:
            q = others[rng.integers(len(others))]
            mm[it["qid"]] = q["items"][rng.integers(2)]["image_id"]
    return mm


def score_item(suite, pred, it):
    return vqa_soft_score(pred, it["answers"]) if suite == "natural" else em_score(pred, it["answers"])


@torch.no_grad()
def run_suite(model, store, pairs, suite, mm, bs=8, max_new=16):
    model.eval(); items = _items(pairs); recs = []
    for cond in CONDS:
        nll = answer_nll(model, store, items, cond, mm)
        for s in range(0, len(items), bs):
            b = items[s:s + bs]
            ids = None if cond == "none" else [i["image_id"] if cond == "correct" else i["partner"]["image_id"] if cond == "swap" else mm[i["qid"]] for i in b]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t = time.perf_counter()
            preds = model.generate([_ex(i) for i in b], None if ids is None else store.get(ids), use_image=ids is not None, max_new=max_new)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            dt = (time.perf_counter() - t) / len(b)
            for j, (it, pr) in enumerate(zip(b, preds)):
                pn = normalize_answer(pr)
                recs.append({"cond": cond, "pair_id": it["pair_id"], "qid": it["qid"], "k": it["k"], "type": it["type"], "suite": suite,
                             "question": it["question"], "refs": it["answers"], "canon": it["answer"], "partner_canon": it["partner"]["answer"],
                             "image_used": None if ids is None else ids[j], "pred_raw": pr, "pred": pn, "score": score_item(suite, pr, it),
                             "canon_correct": pn == it["answer"], "follows_partner": pn == it["partner"]["answer"], "empty": pn == "",
                             "nll": float(nll[s + j]), "gen_time_s": dt})
    return recs


def pair_arrays(recs, pair_ids, cond, types=None):
    """Per-pair statistics (arrays aligned with pair_ids) for one condition."""
    by = {}
    for r in recs:
        if r["cond"] == cond: by.setdefault(r["pair_id"], {})[r["k"]] = r
    keep = [p for p in pair_ids if (types is None or by[p][0]["type"] in types)]
    f = lambda fn: np.array([fn(by[p][0], by[p][1]) for p in keep], float)
    return keep, {"acc": f(lambda a, b: (a["score"] + b["score"]) / 2), "pair_acc": f(lambda a, b: a["canon_correct"] and b["canon_correct"]),
                  "identical": f(lambda a, b: a["pred"] == b["pred"]), "empty": f(lambda a, b: (a["empty"] + b["empty"]) / 2),
                  "nll": f(lambda a, b: (a["nll"] + b["nll"]) / 2), "follows_partner": f(lambda a, b: (a["follows_partner"] + b["follows_partner"]) / 2),
                  "time": f(lambda a, b: (a["gen_time_s"] + b["gen_time_s"]) / 2)}


def boot_ci(x, idx):
    m = x[idx].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(recs, n_boot=4000, seed=SEED, type_groups=None):
    """Point estimates + 95% CIs from bootstrapping COMPLETE pairs (same resampled pairs for every condition,
    so differences are matched)."""
    pair_ids = sorted({r["pair_id"] for r in recs}); out = {}
    rng = np.random.default_rng(seed)
    groups = {"all": None, **(type_groups or {})}
    for gname, types in groups.items():
        keep, base = pair_arrays(recs, pair_ids, "correct", types)
        if not keep: continue
        idx = rng.integers(0, len(keep), size=(n_boot, len(keep))); g = {}
        A = {c: pair_arrays(recs, pair_ids, c, types)[1] for c in CONDS}
        for c in CONDS:
            g[c] = {m: {"mean": float(v.mean()), "ci95": boot_ci(v, idx)} for m, v in A[c].items()}
        g["advantage_acc_correct_minus_swap"] = {"mean": float((A["correct"]["acc"] - A["swap"]["acc"]).mean()),
                                                 "ci95": boot_ci(A["correct"]["acc"] - A["swap"]["acc"], idx)}
        g["advantage_nll_swap_minus_correct"] = {"mean": float((A["swap"]["nll"] - A["correct"]["nll"]).mean()),
                                                 "ci95": boot_ci(A["swap"]["nll"] - A["correct"]["nll"], idx)}
        g["n_pairs"] = len(keep); out[gname] = g
    return out


def final_eval(model, ds, store, out_dir, n_boot=4000):
    os.makedirs(out_dir, exist_ok=True); res = {}
    for suite, key, groups in (("grounding", "grounding_test", {"presence": ["presence"], "counting": ["counting"]}),
                               ("natural", "vqa_test", {"yes/no": ["yes/no"], "number": ["number"], "other": ["other"]})):
        pairs = ds[key]; mm = mismatch_map(pairs)
        save_json(mm, os.path.join(out_dir, f"{suite}_mismatch_map.json"))
        recs = run_suite(model, store, pairs, suite, mm)
        with open(os.path.join(out_dir, f"{suite}_predictions.jsonl"), "w") as f:
            for r in recs: f.write(json.dumps(r, default=float) + "\n")
        res[suite] = summarize(recs, n_boot, type_groups=groups)
        res[suite + "_qualitative"] = select_examples(recs)
    save_json(res, os.path.join(out_dir, "final_results.json"))
    return res


def select_examples(recs, k=2):
    """Stated rule: iterate pairs in sorted pair_id order; per category keep the first k.
       follow : both answers correct with the correct images AND both swapped predictions equal the partner's reference
       insensitive : correct with correct images but identical answers under swap (ignores the image change)
       wrong : at least one wrong answer with the correct images"""
    by = {}
    for r in recs: by.setdefault(r["pair_id"], {}).setdefault(r["cond"], {})[r["k"]] = r
    cats = {"follow": [], "insensitive": [], "wrong": []}
    for pid in sorted(by):
        c, s = by[pid]["correct"], by[pid]["swap"]
        okc = c[0]["canon_correct"] and c[1]["canon_correct"]
        cat = ("follow" if okc and s[0]["follows_partner"] and s[1]["follows_partner"] else
               "insensitive" if okc and s[0]["pred"] == s[1]["pred"] else "wrong" if not okc else None)
        if cat and len(cats[cat]) < k:
            cats[cat].append({"pair_id": pid, "question": c[0]["question"], "refs": [c[0]["canon"], c[1]["canon"]],
                              **{cond: [by[pid][cond][j]["pred_raw"] for j in (0, 1)] for cond in CONDS},
                              "images": [c[0]["image_used"], c[1]["image_used"]]})
    return cats


def main():
    from transformers import AutoTokenizer
    import yaml
    from .features import FeatureStore
    from .train import build_model
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--dataset", required=True); ap.add_argument("--features", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n_boot", type=int, default=4000)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); dev = get_device()
    tok = AutoTokenizer.from_pretrained(cfg["qwen_name"]); store = FeatureStore(a.features)
    m = build_model(cfg, tok, store.dim).to(dev); m.connector.float()
    m.connector.load_state_dict(torch.load(os.path.join(a.run, "connector_stage3.pt")))
    m.load_lora_state(torch.load(os.path.join(a.run, "adapters_stage3.pt"))); m.set_stage(3)
    m.vis_scale.fill_(json.load(open(os.path.join(a.run, "calibration.json")))["scale"])
    final_eval(m, json.load(open(a.dataset)), store, a.out, a.n_boot)


if __name__ == "__main__":
    main()

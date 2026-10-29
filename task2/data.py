"""Task 2 data construction (COCO 2014 captions + instances, VQAv2 complementary pairs). Seed 6304.

Pair format (grounding and natural VQA alike):
  {"pair_id", "suite", "type", "items": [{"qid","image_id","question","answer"(canonical, normalised),
                                           "target"(original string used for training), "answers"[refs]} x2]}
Each pair = same question on two images with different reference answers.
"""
import json, os
from collections import defaultdict
import numpy as np

from common.utils import SEED, save_json
from .textutil import normalize_answer, pluralize

CAPTION_PROMPT = "Describe this image briefly."
VQA_SUFFIX = "\nAnswer the question using a single word or phrase."
MIN_AREA_FRAC = 0.005          # annotations smaller than this fraction of the image are "ignored" -> exclude such images from counting


def img_name(split, image_id):
    return f"COCO_{split}2014_{int(image_id):012d}.jpg"


# ---------------------------------------------------------------- captions
def build_captions(caps_json, candidate_images, n_img, n_per, rng, exclude=()):
    by = defaultdict(list)
    for a in caps_json["annotations"]:
        by[a["image_id"]].append(a)
    pool = sorted(set(candidate_images) - set(exclude) & set(by))
    pool = [i for i in pool if i in by]
    chosen = [pool[j] for j in rng.permutation(len(pool))[:n_img]]
    out = []
    for iid in chosen:
        anns = sorted(by[iid], key=lambda a: a["id"])
        for j in rng.permutation(len(anns))[:n_per]:
            out.append({"image_id": iid, "caption_id": anns[j]["id"], "question": CAPTION_PROMPT,
                        "target": anns[j]["caption"].strip(), "answer": normalize_answer(anns[j]["caption"]), "kind": "caption"})
    return out, set(chosen)


# ---------------------------------------------------------------- annotation-derived grounding
def _category_stats(inst_json):
    imgs = {i["id"]: i for i in inst_json["images"]}
    cats = {c["id"]: c["name"] for c in inst_json["categories"]}
    counts, any_ann, tiny = defaultdict(lambda: defaultdict(int)), defaultdict(set), defaultdict(set)
    for a in inst_json["annotations"]:
        im = imgs[a["image_id"]]
        any_ann[a["image_id"]].add(a["category_id"])
        if a.get("iscrowd", 0):
            tiny[a["image_id"]].add(a["category_id"]); continue
        if a["area"] < MIN_AREA_FRAC * im["width"] * im["height"]:
            tiny[a["image_id"]].add(a["category_id"]); continue
        counts[a["image_id"]][a["category_id"]] += 1
    return cats, counts, any_ann, tiny


def build_grounding_pairs(inst_json, images, n_presence, n_counting, rng, exclude=(), prefix="g"):
    """Presence: same 'Does this image contain X?' on an image WITH X and one with no annotation of X (yes/no).
    Counting: 'How many Xs can you see?' on two images whose counts (1..4) differ.  Images with ignored
    (crowd/tiny) instances of X are not used for X-counting, which removes the main source of ambiguous counts."""
    cats, counts, any_ann, tiny = _category_stats(inst_json)
    images = sorted(set(images) - set(exclude))
    pos, neg, bycount = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for i in images:
        for c in cats:
            n = counts[i].get(c, 0)
            if n >= 1: pos[c].append(i)
            elif c not in any_ann[i]: neg[c].append(i)
            if 1 <= n <= 4 and c not in tiny[i]: bycount[c][n].append(i)
    pairs, used = [], set()

    def mk(pid, typ, q, items):
        return {"pair_id": pid, "suite": "grounding", "type": typ, "items": items}

    def item(qid, iid, q, ans):
        return {"qid": qid, "image_id": iid, "question": q, "answer": ans, "target": ans, "answers": [ans]}
    ok = [c for c in cats if pos[c] and neg[c]]
    for k in range(n_presence):
        c = ok[rng.integers(len(ok))]; name = cats[c]
        a, b = pos[c][rng.integers(len(pos[c]))], neg[c][rng.integers(len(neg[c]))]
        q = f"Does this image contain {name}?"
        its = [item(f"{prefix}p{k}a", a, q, "yes"), item(f"{prefix}p{k}b", b, q, "no")]
        if rng.random() < 0.5: its = its[::-1]
        pairs.append(mk(f"{prefix}p{k}", "presence", q, its))
    okc = [c for c in cats if sum(len(v) > 0 for v in bycount[c].values()) >= 2]
    for k in range(n_counting):
        c = okc[rng.integers(len(okc))]; name = cats[c]
        ns = [n for n, v in bycount[c].items() if v]
        n1, n2 = rng.choice(ns, 2, replace=False)
        a, b = bycount[c][n1][rng.integers(len(bycount[c][n1]))], bycount[c][n2][rng.integers(len(bycount[c][n2]))]
        q = f"How many {pluralize(name)} can you see?"
        its = [item(f"{prefix}c{k}a", a, q, str(int(n1))), item(f"{prefix}c{k}b", b, q, str(int(n2)))]
        if rng.random() < 0.5: its = its[::-1]
        pairs.append(mk(f"{prefix}c{k}", "counting", q, its))
    return pairs


# ---------------------------------------------------------------- natural VQAv2 complementary pairs
def build_vqa_pairs(q_json, a_json, comp_pairs, counts_by_type, rng, split, exclude_images=(), min_agree=8, prefix="v"):
    """counts_by_type: {'yes/no': n, 'number': n, 'other': n} PAIRS.  Keeps pairs whose normalised consensus
    answers differ and where >= min_agree of 10 annotators agree with the consensus on EACH image."""
    Q = {q["question_id"]: q for q in q_json["questions"]}
    A = {a["question_id"]: a for a in a_json["annotations"]}
    ex = set(exclude_images)
    cand = defaultdict(list)
    for p in comp_pairs:
        q1, q2 = p[0], p[1]
        if q1 not in A or q2 not in A: continue
        a1, a2 = A[q1], A[q2]
        if a1["image_id"] in ex or a2["image_id"] in ex or a1["image_id"] == a2["image_id"]: continue
        if a1["answer_type"] != a2["answer_type"]: continue
        c1, c2 = normalize_answer(a1["multiple_choice_answer"]), normalize_answer(a2["multiple_choice_answer"])
        if c1 == c2: continue
        refs1, refs2 = [x["answer"] for x in a1["answers"]], [x["answer"] for x in a2["answers"]]
        if sum(normalize_answer(r) == c1 for r in refs1) < min_agree or sum(normalize_answer(r) == c2 for r in refs2) < min_agree: continue
        cand[a1["answer_type"]].append((q1, q2, c1, c2, refs1, refs2))
    pairs, used_imgs = [], set()
    for typ, n in counts_by_type.items():
        pool = cand[typ]
        order = rng.permutation(len(pool)); got = 0
        for j in order:
            if got >= n: break
            q1, q2, c1, c2, r1, r2 = pool[j]
            i1, i2 = A[q1]["image_id"], A[q2]["image_id"]
            if i1 in used_imgs or i2 in used_imgs: continue              # keep selected pairs image-disjoint
            used_imgs |= {i1, i2}
            tgt = lambda refs, c: next(r for r in refs if normalize_answer(r) == c)   # original string from consensus group
            its = [{"qid": f"{q1}", "image_id": i1, "question": Q[q1]["question"], "answer": c1, "target": tgt(r1, c1), "answers": r1},
                   {"qid": f"{q2}", "image_id": i2, "question": Q[q2]["question"], "answer": c2, "target": tgt(r2, c2), "answers": r2}]
            pairs.append({"pair_id": f"{prefix}{q1}_{q2}", "suite": "natural", "type": typ, "items": its})
            got += 1
        if got < n: print(f"[warn] only {got}/{n} {typ} pairs available for {split}")
    return pairs


def images_of(pairs_or_caps):
    s = set()
    for x in pairs_or_caps:
        if "items" in x: s |= {i["image_id"] for i in x["items"]}
        else: s.add(x["image_id"])
    return s


def split_isolation_check(ds):
    """Raises if dev/test images overlap each other or training."""
    train = set()
    for k in ("caps_train", "presence_train", "counting_train", "vqa_train"):
        train |= images_of(ds[k])
    dev = images_of(ds["caps_dev"]) | images_of(ds["grounding_dev"]) | images_of(ds["vqa_dev"])
    test = images_of(ds["grounding_test"]) | images_of(ds["vqa_test"])
    assert not (train & dev) and not (train & test) and not (dev & test), "split leakage"
    # evaluation pair sets are image-disjoint across the four evaluation suites as well
    suites = [images_of(ds[k]) for k in ("grounding_dev", "vqa_dev", "grounding_test", "vqa_test", "caps_dev")]
    for i in range(len(suites)):
        for j in range(i + 1, len(suites)):
            assert not (suites[i] & suites[j]), "evaluation suites share images"
    return True


def build_dataset(coco_root, vqa_root, out_path, sizes=None, seed=SEED):
    """Expects standard layouts: coco_root/{annotations/captions_{train,val}2014.json, instances_*, train2014/, val2014/},
    vqa_root/{v2_OpenEnded_mscoco_{split}2014_questions.json, v2_mscoco_{split}2014_annotations.json,
              v2_mscoco_{split}2014_complementary_pairs.json}."""
    S = {"cap_train_img": 6000, "cap_per": 3, "cap_dev_img": 96, "g_pool": 3000, "g_dev": 12, "g_test": 32,
         "vqa_train": {"yes/no": 640, "number": 256, "other": 640}, "vqa_dev": {"yes/no": 12, "number": 8, "other": 12},
         "vqa_test": {"yes/no": 36, "number": 24, "other": 36}}
    S.update(sizes or {})
    rng = np.random.default_rng(seed)
    J = lambda *p: json.load(open(os.path.join(*p)))
    ann = lambda n, s: J(coco_root, "annotations", f"{n}_{s}2014.json")
    vq = lambda s: (J(vqa_root, f"v2_OpenEnded_mscoco_{s}2014_questions.json"), J(vqa_root, f"v2_mscoco_{s}2014_annotations.json"),
                    J(vqa_root, f"v2_mscoco_{s}2014_complementary_pairs.json"))
    tr_caps, tr_inst, va_caps, va_inst = ann("captions", "train"), ann("instances", "train"), ann("captions", "val"), ann("instances", "val")
    tr_imgs = [i["id"] for i in tr_inst["images"]]; va_imgs = [i["id"] for i in va_inst["images"]]
    ds = {}
    # --- validation-side sets first (dev, then test), each excluding all images chosen so far
    used = set()
    q, a, cp = vq("val")
    ds["vqa_test"] = build_vqa_pairs(q, a, cp, S["vqa_test"], rng, "test", used, prefix="vt"); used |= images_of(ds["vqa_test"])
    ds["vqa_dev"] = build_vqa_pairs(q, a, cp, S["vqa_dev"], rng, "dev", used, prefix="vd"); used |= images_of(ds["vqa_dev"])
    ds["grounding_test"] = build_grounding_pairs(va_inst, va_imgs, S["g_test"], S["g_test"], rng, used, "gt"); used |= images_of(ds["grounding_test"])
    ds["grounding_dev"] = build_grounding_pairs(va_inst, va_imgs, S["g_dev"], S["g_dev"], rng, used, "gd"); used |= images_of(ds["grounding_dev"])
    ds["caps_dev"], _ = build_captions(va_caps, va_imgs, S["cap_dev_img"], 1, rng, used)
    # --- train side (train2014 images)
    ds["caps_train"], _ = build_captions(tr_caps, tr_imgs, S["cap_train_img"], S["cap_per"], rng)
    ds["presence_train"] = [p for p in build_grounding_pairs(tr_inst, tr_imgs, S["g_pool"], 0, rng, (), "tp")]
    ds["counting_train"] = [p for p in build_grounding_pairs(tr_inst, tr_imgs, 0, S["g_pool"], rng, (), "tc")]
    q, a, cp = vq("train")
    ds["vqa_train"] = build_vqa_pairs(q, a, cp, S["vqa_train"], rng, "train", (), prefix="vtr")
    split_isolation_check(ds)
    paths = {}
    for k, v in ds.items():
        for i in images_of(v):
            sp = "val" if k.endswith(("dev", "test")) else "train"
            paths[str(i)] = os.path.join(coco_root, f"{sp}2014", img_name(sp, i))
    ds["image_paths"] = paths
    save_json(ds, out_path)
    save_json({k: sorted({str(i) for i in images_of(v)}) for k, v in ds.items() if k != "image_paths"}, out_path + ".image_ids.json")
    save_json({k: [x.get("pair_id", x.get("caption_id")) for x in v] for k, v in ds.items() if k != "image_paths"}, out_path + ".pair_ids.json")
    return ds


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--coco_root", required=True); ap.add_argument("--vqa_root", required=True)
    ap.add_argument("--out", default="results/task2/dataset.json"); a = ap.parse_args()
    build_dataset(a.coco_root, a.vqa_root, a.out)

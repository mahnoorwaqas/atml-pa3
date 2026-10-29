"""Synthetic data generators for the smoke tests (no real data needed)."""
import os
import numpy as np
import pandas as pd


def make_task3_tables(root, vlm=False, n_q=40, n_roll=32, seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "vlm" if vlm else "llm"); os.makedirs(d, exist_ok=True)
    G, P, At, meta = [], [], [], {}
    for q in range(n_q):
        sid = f"q{q:04d}"
        p_correct = rng.uniform(0.1, 0.9)
        gold = str(rng.integers(1, 50)) if not vlm else "B"
        if vlm:
            meta[sid] = dict(pid=sid, question_type="multi_choice", answer_type="text", choices=["x", "y", "z"], precision=None)
        for r in range(n_roll):
            ok = rng.random() < p_correct
            if vlm:
                ans = "B" if ok else rng.choice(["A", "C", "(c)", "z", None])
            else:
                ans = gold if ok else rng.choice([str(int(gold) + 1), str(int(gold) + 2), "\\frac{1}{2}", "0.5", None])
            T = int(rng.integers(15, 60))
            G.append(dict(dataset="d", sample_id=sid, rollout_id=r, text="t", token_ids=list(range(T)), num_tokens=T,
                          extracted_answer=ans, gold_answer=gold))
            P.append(dict(dataset="d", sample_id=sid, rollout_id=r, logp=-rng.exponential(0.5 + (0 if ok else .3), T),
                          kl_u_p=rng.gamma(4, 1.0 + (0.4 if ok else 0), T)))
            if vlm:
                At.append(dict(dataset="d", sample_id=sid, rollout_id=r, attn_image=rng.uniform(0, 1, T),
                               attn_prompt=rng.uniform(0, 1, T), attn_gen=rng.uniform(0, 1, T)))
    pd.DataFrame(G).to_parquet(os.path.join(d, "generations.parquet"))
    pd.DataFrame(P).to_parquet(os.path.join(d, "probability_metrics.parquet"))
    if vlm:
        pd.DataFrame(At).to_parquet(os.path.join(d, "attention_metrics.parquet"))
    return d, meta


def make_coco_vqa(root, n_train=400, n_val=260, seed=0):
    """Tiny COCO-/VQAv2-shaped dataset (images, captions, instances, questions, annotations, complementary pairs)."""
    import json
    from PIL import Image
    rng = np.random.default_rng(seed)
    cats = [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}, {"id": 3, "name": "car"}]
    coco, vqa = os.path.join(root, "coco"), os.path.join(root, "vqa")
    for d in (coco + "/annotations", coco + "/train2014", coco + "/val2014", vqa): os.makedirs(d, exist_ok=True)
    for split, n, off in (("train", n_train, 0), ("val", n_val, 100000)):
        ids = [off + i + 1 for i in range(n)]
        images, anns, caps = [], [], []
        for iid in ids:
            images.append({"id": iid, "width": 100, "height": 100})
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(f"{coco}/{split}2014/COCO_{split}2014_{iid:012d}.jpg")
            for c in cats:
                for _ in range(int(rng.integers(0, 6)) if rng.random() < 0.7 else 0):
                    anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": c["id"], "area": 2000, "iscrowd": 0})
            for k in range(5):
                caps.append({"id": iid * 10 + k, "image_id": iid, "caption": f"a {['dog', 'cat', 'car'][k % 3]} and a person {k}"})
        json.dump({"images": images, "annotations": anns, "categories": cats}, open(f"{coco}/annotations/instances_{split}2014.json", "w"))
        json.dump({"images": images, "annotations": caps}, open(f"{coco}/annotations/captions_{split}2014.json", "w"))
        Qs, As, pairs, qid = [], [], [], off * 10
        for typ, (qa, qb) in (("yes/no", ("yes", "no")), ("number", ("2", "3")), ("other", ("red", "blue"))):
            for j in range(40):
                i1, i2 = rng.choice(ids, 2, replace=False)
                for iid, ans in ((i1, qa), (i2, qb)):
                    qid += 1
                    Qs.append({"question_id": qid, "image_id": int(iid), "question": f"what is the {typ} thing {j}"})
                    As.append({"question_id": qid, "image_id": int(iid), "multiple_choice_answer": ans, "answer_type": typ,
                               "answers": [{"answer": ans}] * 9 + [{"answer": "maybe"}]})
                pairs.append([qid - 1, qid])
        json.dump({"questions": Qs}, open(f"{vqa}/v2_OpenEnded_mscoco_{split}2014_questions.json", "w"))
        json.dump({"annotations": As}, open(f"{vqa}/v2_mscoco_{split}2014_annotations.json", "w"))
        json.dump(pairs, open(f"{vqa}/v2_mscoco_{split}2014_complementary_pairs.json", "w"))
    return coco, vqa

import json, os
import numpy as np
import torch


def test_dataset_build_isolation_and_labels():
    from tests.fixtures import task2_env
    from task2.data import images_of, split_isolation_check
    e = task2_env(); ds = e["ds"]
    assert split_isolation_check(ds)
    inst = json.load(open(e["root"] + "/coco/annotations/instances_val2014.json"))
    cnt = {}
    for a in inst["annotations"]: cnt[(a["image_id"], a["category_id"])] = cnt.get((a["image_id"], a["category_id"]), 0) + 1
    name2id = {"person": 1, "dog": 2, "car": 3}
    for p in ds["grounding_test"]:
        a, b = p["items"]; assert a["answer"] != b["answer"] and a["question"] == b["question"]
        if p["type"] == "presence":
            c = name2id[a["question"].split("contain ")[1].rstrip("?")]
            for it in p["items"]: assert (cnt.get((it["image_id"], c), 0) > 0) == (it["answer"] == "yes")
        else:
            c = {"people": 1, "dogs": 2, "cars": 3}[a["question"].split("many ")[1].split(" can")[0]]
            for it in p["items"]: assert cnt.get((it["image_id"], c), 0) == int(it["answer"]) and 1 <= int(it["answer"]) <= 4
    for p in ds["vqa_test"]:
        assert p["items"][0]["answer"] != p["items"][1]["answer"] and p["items"][0]["target"] in p["items"][0]["answers"]
    assert len(ds["vqa_test"]) == 8 and len(ds["grounding_test"]) == 8


def test_features_shape_and_determinism():
    from tests.fixtures import task2_env
    from task2.features import cache_features
    from common.tiny import make_tiny_siglip
    e = task2_env(); ids = list(e["ds"]["image_paths"])[:5]
    f = e["store"].get(ids); assert f.shape == (5, 196, 32) and f.dtype == torch.float16
    sub = {i: e["ds"]["image_paths"][i] for i in ids}
    cache_features(sub, e["root"] + "/feats2", siglip=make_tiny_siglip(32), bs=2, device=torch.device("cpu"))
    from task2.features import FeatureStore
    f2 = FeatureStore(e["root"] + "/feats2").get(sorted(ids))
    assert torch.allclose(e["store"].get(sorted(ids)).float(), f2.float(), atol=1e-3)


def test_vlm_interface():
    from tests.fixtures import task2_env
    from task2.model import VLM
    e = task2_env(); m = e["model"]; m.eval()
    assert m.verify_frozen() and abs(e["summary"]["calibration"]["scale"] - float(m.vis_scale)) < 1e-6
    ex = [{"question": "how many dog", "target": "2", "kind": "vqa"}] * 2
    ids = list(e["ds"]["image_paths"])[:2]; feats = e["store"].get(ids)
    Xi, Mi, Yi, Pi = m.build(ex, feats, True); Xn, Mn, Yn, Pn = m.build(ex, None, False)
    assert Xi.shape[1] - Xn.shape[1] == 196                                         # visual tokens removed, mask/positions rebuilt
    assert int((Yi != -100).sum()) == int((Yn != -100).sum()) == 2 * 2                # labels only on answer + end-of-turn
    assert Pn.shape == Mn.shape and (Pn[0, :Mn[0].sum()] == torch.arange(int(Mn[0].sum()))).all()
    m.train(); nll = m.example_nll(ex, feats, True); nll.sum().backward()
    assert sum(p.grad.abs().sum().item() for p in m.connector.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for n, p in m.qwen.named_parameters() if not (n.endswith(".A") or n.endswith(".B")))
    m.zero_grad(); m.eval()
    a = m.example_nll(ex, feats, True); b = m.example_nll(ex, e["store"].get(ids[::-1]), True)
    assert not torch.allclose(a, b)                                                 # prediction depends on the image
    m49 = VLM(m.qwen, e["tok"], 32, n_tokens=49); assert m49.visual(feats).shape == (2, 49, 64)
    out = m.generate(ex, feats, True, 4); assert len(out) == 2 and isinstance(out[0], str)


def test_training_logs_and_resume():
    from tests.fixtures import task2_env
    from task2.train import train
    e = task2_env(); out = e["root"] + "/run"
    tr = [json.loads(l) for l in open(out + "/train_log.jsonl")]; dv = [json.loads(l) for l in open(out + "/dev_log.jsonl")]
    assert [r["stage"] for r in tr] == [1] * 4 + [2] * 3 + [3] * 2 and all(np.isfinite(r["loss"]) for r in tr)
    assert [d["update"] for d in dv] == [0, 3, 4, 6, 7, 9] and "grounding_delta_nll" in dv[0] and "natural_nll_none" in dv[0]
    assert tr[0]["mix"] == {"caption": 8} and tr[5]["mix"].get("presence", 0) + tr[5]["mix"].get("counting", 0) + tr[5]["mix"].get("caption", 0) == 4
    s = e["summary"]; assert s["trainable_by_stage"][1] < s["trainable_by_stage"][2] and set(s["stage_time_s"]) == {1, 2, 3}
    for f in ("connector_stage1.pt", "connector_stage3.pt", "adapters_stage2.pt", "adapters_stage3.pt", "ckpt_u3.pt", "ckpt_u6.pt"):
        assert os.path.exists(f"{out}/{f}"), f
    # resume from update 6 (inside stage 2) must reproduce the uninterrupted run
    from transformers import AutoModelForCausalLM
    qwen = AutoModelForCausalLM.from_pretrained(e["qd"], dtype=torch.float32)
    train(e["cfg"], e["ds"], e["store"], e["tok"], e["root"] + "/run_resume", qwen=qwen, resume=out + "/ckpt_u6.pt")
    a = torch.load(out + "/connector_stage3.pt"); b = torch.load(e["root"] + "/run_resume/connector_stage3.pt")
    assert max((a[k] - b[k]).abs().max().item() for k in a) < 1e-5


def test_final_eval_four_conditions():
    from tests.fixtures import task2_env
    from task2.evaluate import final_eval, mismatch_map
    e = task2_env()
    res = final_eval(e["model"], e["ds"], e["store"], e["root"] + "/final", n_boot=50)
    for suite, groups in (("grounding", {"presence", "counting"}), ("natural", {"yes/no", "number", "other"})):
        r = res[suite]; assert set(r) == {"all"} | groups
        for c in ("correct", "swap", "mismatch", "none"):
            for m in ("acc", "pair_acc", "identical", "empty", "nll", "time"):
                v = r["all"][c][m]; assert v["ci95"][0] <= v["mean"] + 1e-9 and v["mean"] <= v["ci95"][1] + 1e-9, (suite, c, m, v)
        assert r["all"]["n_pairs"] == 8
    preds = [json.loads(l) for l in open(e["root"] + "/final/natural_predictions.jsonl")]
    assert len(preds) == 8 * 2 * 4
    mm = mismatch_map(e["ds"]["vqa_test"])
    for p in e["ds"]["vqa_test"]:
        own = {i["image_id"] for i in p["items"]}
        assert all(mm[i["qid"]] not in own for i in p["items"])
    assert all(r["image_used"] is None for r in preds if r["cond"] == "none")
    from task2.analyze import plots
    plots(e["root"] + "/run"); assert os.path.exists(e["root"] + "/run/fig_curves.png")

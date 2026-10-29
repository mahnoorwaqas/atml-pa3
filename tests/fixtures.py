import os, tempfile
_CACHE = {}


def tmp_root():
    if "root" not in _CACHE:
        _CACHE["root"] = tempfile.mkdtemp(prefix="pa3_smoke_")
    return _CACHE["root"]


def task1_env():
    """Tiny tokenizer/model, tiny training set, and all five trained runs (8 updates each, CPU)."""
    if "t1" in _CACHE:
        return _CACHE["t1"]
    from common.tiny import save_tiny_assets, make_tokenizer
    from task1.data import prepare
    from task1.train import resolve_cfg, train
    root = os.path.join(tmp_root(), "t1"); os.makedirs(root)
    qd = save_tiny_assets(root); tok = make_tokenizer()
    rows = [{"query": f"what is the sum of {i} and {i + 1}", "response": f"the sum is {2 * i + 1}", "original_question": f"q{i}"}
            for i in range(80)]
    rows[0]["query"] = "gsm test question one"; rows[1]["original_question"] = "gsm test question two"; rows[2]["response"] = ""
    data, meta = prepare(tok, root + "/data.pt", n=32, rows=rows, gsm_questions=["GSM test question one!", "gsm test question two"])
    runs = os.path.join(root, "runs")
    summaries = {}
    for m in ["full", "l2sp", "lora4", "lora16", "oplora"]:
        cfg = resolve_cfg("configs/task1.yaml", m, dict(model_name=qd, data_path=root + "/data.pt", out=f"{runs}/{m}", n_examples=32,
                                                        accum=2, backbone_dtype="float32", k=8, amp=False))
        _, s = train(cfg, tokenizer=tok, data=data); summaries[m] = s
    _CACHE["t1"] = dict(root=root, qd=qd, tok=tok, data=data, meta=meta, runs=runs, summaries=summaries)
    return _CACHE["t1"]


def task2_env():
    """Tiny COCO/VQA data -> dataset.json -> SigLIP feature cache -> tiny-Qwen VLM trained for 9 updates (3 stages)."""
    if "t2" in _CACHE:
        return _CACHE["t2"]
    import json, yaml
    from common.tiny import save_tiny_assets, make_tokenizer, make_tiny_siglip
    from tests.synth import make_coco_vqa
    from task2.data import build_dataset
    from task2.features import cache_features, FeatureStore
    from task2.train import train
    import torch
    root = os.path.join(tmp_root(), "t2"); os.makedirs(root, exist_ok=True)
    coco, vqa = make_coco_vqa(root)
    sizes = {"cap_train_img": 40, "cap_dev_img": 8, "g_pool": 20, "g_dev": 2, "g_test": 4,
             "vqa_train": {"yes/no": 6, "number": 4, "other": 6}, "vqa_dev": {"yes/no": 2, "number": 2, "other": 2},
             "vqa_test": {"yes/no": 3, "number": 2, "other": 3}}
    ds = build_dataset(coco, vqa, root + "/dataset.json", sizes)
    feat_info = cache_features(ds["image_paths"], root + "/feats", siglip=make_tiny_siglip(32), bs=16, device=torch.device("cpu"))
    store = FeatureStore(root + "/feats")
    qd = save_tiny_assets(root); tok = make_tokenizer()
    cfg = yaml.safe_load(open("configs/task2.yaml"))
    cfg.update(qwen_name=qd, micro_bs=4, accum=2, amp=False, backbone_dtype="float32", dev_every=3, ckpt_every=3,
               stages=[{"updates": 4, "lr_conn": 1e-3, "lr_lora": None}, {"updates": 3, "lr_conn": 1e-4, "lr_lora": 2e-4},
                       {"updates": 2, "lr_conn": 5e-5, "lr_lora": 1e-4}], lora={"r": 4, "alpha": 8, "dropout": 0.05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]})
    from transformers import AutoModelForCausalLM
    qwen = AutoModelForCausalLM.from_pretrained(qd, dtype=torch.float32)
    model, summ = train(cfg, ds, store, tok, root + "/run", qwen=qwen)
    _CACHE["t2"] = dict(root=root, ds=ds, store=store, tok=tok, cfg=cfg, qd=qd, model=model, summary=summ, feat_info=feat_info)
    return _CACHE["t2"]

import json, os
import numpy as np
import torch
import yaml


def test_data_masking_and_filters():
    from tests.fixtures import task1_env
    e = task1_env(); tok = e["tok"]
    assert e["meta"]["skipped"]["gsm8k_test"] == 2 and e["meta"]["skipped"]["empty"] == 1, e["meta"]["skipped"]
    ex = e["data"][0]; ids, lab = ex["input_ids"], ex["labels"]
    n_resp = sum(l != -100 for l in lab)
    assert lab[-1] == tok.convert_tokens_to_ids("<|im_end|>") and ids[-n_resp:] == lab[-n_resp:]     # loss on response + end-of-turn only
    assert all(l == -100 for l in lab[:len(lab) - n_resp])                                          # prompt masked
    assert e["meta"]["source_indices"] == sorted(e["meta"]["source_indices"])                       # source order kept


def test_oplora_constraints_and_lora_init():
    from common import lora as L
    torch.manual_seed(0)
    base = torch.nn.Linear(24, 16, bias=False)
    Uk, Vk = L.leading_singular_bases(base.weight.data, 4)
    m = L.LoRALinear(base, r=6, alpha=12, dropout=0.0, mode="oplora", Uk=Uk, Vk=Vk)
    x = torch.randn(3, 24)
    assert torch.allclose(m(x), base(x))                      # zero-B init -> identical to backbone
    m.B.data.normal_(); dW = m.delta_weight()
    assert (Uk.T @ dW).abs().max() < 1e-5 and (dW @ Vk).abs().max() < 1e-5      # outside both protected subspaces
    plain = L.LoRALinear(torch.nn.Linear(24, 16, bias=False), r=6, alpha=12, dropout=0.0)
    plain.B.data.normal_()
    assert (plain.delta_weight()).norm() > 0


def test_training_runs_and_logs():
    from tests.fixtures import task1_env
    e = task1_env()
    for m, s in e["summaries"].items():
        assert s["total_updates"] == 8 and s["ckpt_updates"] == [2, 4, 6, 8], (m, s["ckpt_updates"])
        log = [json.loads(l) for l in open(f"{e['runs']}/{m}/train_log.jsonl")]
        assert len(log) == 8 and all(np.isfinite(r["sft_loss"]) for r in log)
        assert log[-1]["sft_loss"] <= log[0]["sft_loss"] + 0.5
    assert e["summaries"]["lora16"]["trainable_params"] == 4 * e["summaries"]["lora4"]["trainable_params"]       # r:4->16 quadruples
    log = [json.loads(l) for l in open(f"{e['runs']}/l2sp/train_log.jsonl")]
    assert all(r["omega"] >= 0 and abs(r["total_loss"] - r["sft_loss"] - r["lam_omega"]) < 1e-6 for r in log)
    assert "reference_storage_bytes" in e["summaries"]["l2sp"]["overhead"] and "svd_time_s" in e["summaries"]["oplora"]["overhead"]
    cm = [json.loads(l) for l in open(f"{e['runs']}/full/ckpt_meta.jsonl")]
    assert cm[-1]["displacement_l2"] > cm[0]["displacement_l2"] > 0


def test_resume_reproduces_full_ft():
    from tests.fixtures import task1_env
    from task1.train import resolve_cfg, train
    e = task1_env()
    cfg = resolve_cfg("configs/task1.yaml", "full", dict(model_name=e["qd"], data_path=e["root"] + "/data.pt", out=e["root"] + "/resume_run",
                                                         n_examples=32, accum=2, backbone_dtype="float32", amp=False,
                                                         resume=f"{e['runs']}/full/4"))
    train(cfg, tokenizer=e["tok"], data=e["data"])
    a = torch.load(f"{e['runs']}/full/model_u8.pt"); b = torch.load(e["root"] + "/resume_run/model_u8.pt")
    c = torch.load(f"{e['runs']}/full/model_u4.pt")
    assert max((a[k] - c[k]).abs().max().item() for k in a) > 1e-6                    # training actually moved weights
    assert max((a[k] - b[k]).abs().max().item() for k in a) < 1e-8                    # resumed run == uninterrupted run


def test_evaluate_merge_and_diagnostics():
    from tests.fixtures import task1_env
    from task1.evaluate import evaluate_all
    from task1.diagnostics import run_diagnostics, sample_probe_ids
    from transformers import AutoModelForCausalLM
    e = task1_env(); cfg = yaml.safe_load(open("configs/task1.yaml")); cfg["model_name"] = e["qd"]
    table = evaluate_all(cfg, e["runs"], e["root"] + "/eval", backend="synthetic", tok=e["tok"], final_update=8, mid_update=4)
    assert set(table) == {"baseline", "full_mid", "full_final", "l2sp_final", "lora4_final", "lora16_final", "oplora_final"}
    for c in ("lora4_final", "lora16_final", "oplora_final"):
        assert table[c]["merge_check_max_abs_logit_diff"] < 1e-4, table[c]["merge_check_max_abs_logit_diff"]
    assert abs(table["baseline"]["delta_pp"]["boolq"]) < 1e-9
    base = AutoModelForCausalLM.from_pretrained(e["qd"], dtype=torch.float32)
    conds = [("full_mid", "full", 4), ("full_final", "full", 8), ("l2sp_final", "l2sp", 8), ("lora4_final", "lora4", 8),
             ("lora16_final", "lora16", 8), ("oplora_final", "oplora", 8)]
    pm = ["what is the sum of 1 and 2"] * 5; po = ["the quick brown fox", "hello world and"] * 3
    res = run_diagnostics(base, e["tok"], e["runs"], conds, [0, 2], 8, pm, po, e["root"] + "/diag.json", torch.device("cpu"))
    assert res["oplora_final|layer0"]["rho_k"] < 1e-8 or res["oplora_final|layer0"]["rho_k"] < res["lora16_final|layer0"]["rho_k"]
    assert res["oplora_final|layer2"]["eta"] > 0 and res["full_final|layer2"]["eta"] > res["full_mid|layer2"]["eta"] * 0.5
    ids = sample_probe_ids(1319, 10042); assert len(ids["gsm8k_test"]) == 32 and ids == sample_probe_ids(1319, 10042)


def test_qualitative_outputs():
    from tests.fixtures import task1_env
    from task1.qualitative import parse_questions, generate_all, write_outputs
    from transformers import AutoModelForCausalLM
    e = task1_env()
    items = []
    for i in range(8):
        items += [{"id": f"q{i}", "paraphrases": ["hello world"]},
                  {"id": f"q{i}_json", "paraphrases": ["hello world as json"], "system": "answer in json"},
                  {"id": f"q{i}_template", "paraphrases": ["hello world template"]}]
    yp = e["root"] + "/q.yaml"; yaml.safe_dump(items, open(yp, "w"))
    parsed, order = parse_questions(yp); assert len(parsed) == 24 and len(order) == 8
    m = AutoModelForCausalLM.from_pretrained(e["qd"])
    results = {c: generate_all(m, e["tok"], parsed, torch.device("cpu"), max_new=6) for c in
               ["baseline", "full_mid", "full_final", "l2sp_final", "lora4_final", "lora16_final", "oplora_final"]}
    grouped, rows = write_outputs(results, order, e["root"] + "/qual")
    assert sum(len(v) for g in grouped.values() for v in g.values()) == 168 and len(rows) == 42

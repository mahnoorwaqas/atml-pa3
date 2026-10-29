import json, os
import numpy as np


def _run(vlm):
    from tests.fixtures import tmp_root
    from tests.synth import make_task3_tables
    from task3.schema import load_tables
    from task3.run_offline import run_dataset
    root = tmp_root(); d, meta = make_task3_tables(root, vlm, n_q=50)
    out = os.path.join(root, "t3_vlm" if vlm else "t3_llm")
    if not os.path.exists(out + "/run_info.json"):
        run_dataset("mathvista" if vlm else "math", load_tables(d, vlm), out, meta=meta if vlm else None, n_boot=50)
    return out


def test_answer_grouping_and_scoring():
    from task3 import answers as A
    g, reps = A.group_math(["1/2", "\\frac{1}{2}", "0.5", "3", None, "\\dfrac12", "x+1", "1+x"])
    assert g[0] == g[1] == g[2] == g[5] and g[4] == A.NO_ANSWER and g[3] != g[0] and g[6] == g[7]
    meta = {"question_type": "multi_choice", "choices": ["cat", "dog", "cow"], "answer_type": "text"}
    assert A.mathvista_normalize("(b)", meta) == "B" and A.mathvista_normalize("dog", meta) == "B"
    assert A.mathvista_normalize("3.0", {"answer_type": "integer"}) == "3" and A.mathvista_normalize("3.14159", {"answer_type": "float", "precision": 2}) == "3.14"
    assert A.last_boxed("so \\boxed{\\frac{1}{2}} then \\boxed{a{b}}") == "a{b}" and A.last_boxed("no box") is None


def test_selectors_rwc_and_ties():
    from task3.engine import sel_sc, sel_rwc, sel_best, rwc_weights, rwc_margin
    assert sel_sc(np.array([1, 2, 2, 1])) == 0                      # tie -> earliest supporting rollout
    w = rwc_weights([0.1, 0.9, 0.5, 0.5])                           # ranks 4,1,2.5,2.5 -> (4-r)/3
    assert np.allclose(w, [0.0, 1.0, 0.5, 0.5])
    pos, W = sel_rwc(np.array([5, 7, 5, 5]), [0.1, 0.9, 0.5, 0.5], return_votes=True)
    assert pos == 0 and abs(W[5] - 1.0) < 1e-9 and abs(W[7] - 1.0) < 1e-9          # exact tie -> earliest supporting rollout
    assert sel_rwc(np.array([3]), [0.2]) == 0 and sel_best(np.array([1., 1.])) == 0
    assert rwc_margin({1: 3.0}) == 1.0 and abs(rwc_margin({1: 3.0, 2: 1.0}) - 0.5) < 1e-12


def test_visconf_prefix_only_calibration_and_fusion_identity():
    from tests.fixtures import tmp_root
    from tests.synth import make_task3_tables
    from task3.schema import load_tables
    from task3.data import build_pools
    from task3.engine import OrderedView
    d, meta = make_task3_tables(tmp_root(), True, n_q=3, seed=5)
    pools = build_pools(load_tables(d, True), "mathvista", meta=meta)
    p = pools["q0000"]; order = np.random.default_rng(1).permutation(32)
    v = OrderedView(p, order)
    # manual VisConf for N=4 using ONLY the first 4 rollouts
    idx = order[:4]; kl = np.concatenate([p.kl_flat[p.offs[i]:p.offs[i + 1]] for i in idx]).astype(float)
    q05, q95 = np.percentile(kl, [5, 95]); Z = []
    for i in idx:
        k = p.kl_flat[p.offs[i]:p.offs[i + 1]].astype(float)
        c = np.clip((k - q05) / max(q95 - q05, 1e-8), 0, 1); Z.append(p.V[i] * c.mean())
    assert np.allclose(v.score("visconf", 4), Z, atol=1e-6)
    assert not np.allclose(v.score("visconf", 4), v.score("visconf", 32)[:4])      # calibration really depends on the prefix
    n = 32; c = v.calibrated(n)
    for i in range(3):
        s, e = v.bounds[i], v.bounds[i + 1]; cc, vv = c[s:e], v.v[s:e].astype(float)
        assert abs((np.mean(cc * vv)) - cc.mean() * vv.mean() - (v.score("token", n)[i] - v.score("visconf", n)[i])) < 1e-5


def test_offline_llm_pipeline():
    out = _run(False)
    import pandas as pd
    m = pd.read_csv(out + "/methods_eval.csv")
    first = m[m.method == "first"].sort_values("N")
    assert (m[m.method == "sc"].coverage.diff().dropna() >= -1e-9).all()                    # oracle coverage is monotone in N
    for k in m.method.unique():                                                             # at N=1 every rule returns the only candidate
        assert abs(m[(m.method == k) & (m.N == 1)].acc.iloc[0] - first.acc.iloc[0]) < 1e-9, k
    assert ((m.acc <= m.coverage + 1e-9)).all()
    ad = json.load(open(out + "/adaptive.json"))
    for fam in ("agreement", "quality"):
        h = ad[fam]["heldout"]; assert abs(sum(h["stop_dist"].values()) - 1) < 1e-9 and h["rollouts"] <= 32
        assert ad[fam]["dev"]["acc"] >= ad[fam]["fixed32_dev"]["acc"] - 0.01 - 1e-9                 # development constraint respected
        assert h["fixed"]["32"]["cost"] >= h["cost"] - 1e-9
    sp = json.load(open(out + "/splits.json")); assert not set(sp["dev"]) & set(sp["eval"]) and sp["live50"] == sp["eval"][:50]
    assert os.path.exists(out + "/signals.json") and os.path.exists(out + "/paired_bootstrap.csv")


def test_offline_vlm_pipeline_and_analysis():
    out = _run(True)
    import pandas as pd
    m = pd.read_csv(out + "/methods_eval.csv")
    assert {"visconf_rwc", "visconf_best", "vision_best", "token_rwc"} <= set(m.method)
    sig = json.load(open(out + "/signals.json"))["summary"]
    assert sig["fusion_identity_max_abs_err"] < 1e-9 and "conf_vision_corr" in sig and "shortest_rollout_accuracy" in sig
    from task3.analyze import figures, case_studies
    figures(out); cs = case_studies(out)
    assert os.path.exists(out + "/fig_accuracy_cost.png") and set(cs) == {"selection_gap", "rwc_fixes_sc", "premature_stop"}


def test_split_and_order_determinism():
    from task3.data import make_splits, make_orders
    ids = [f"q{i:04d}" for i in range(1000)]
    a, b = make_splits(ids), make_splits(ids[::-1])
    assert a == b and len(a["dev"]) == 200 and len(a["eval"]) == 800
    o1, o2 = make_orders(5), make_orders(5)
    assert (o1 == o2).all() and all(sorted(o1[0, 0]) == list(range(32)) for _ in [0])


def test_live_generation_pipeline_on_tiny_model():
    import torch
    from tests.fixtures import tmp_root
    from common.tiny import save_tiny_assets, make_tokenizer
    from transformers import AutoModelForCausalLM
    from task3.live import run_live
    root = os.path.join(tmp_root(), "live"); os.makedirs(root, exist_ok=True)
    qd = save_tiny_assets(root); tok = make_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(qd).eval()
    qs = [{"qid": f"q{i}", "problem": "what is 1 plus 1", "gold": "2"} for i in range(3)]
    s = run_live(model, tok, qs, root + "/out", torch.device("cpu"), limits=(24, 8, 12))
    assert s["n_questions"] == 3 and set(s) >= {"single", "parallel", "refine", "refine_transitions"}
    assert s["parallel"]["mean_out_tokens"] <= 3 * 8 and s["refine"]["mean_in_tokens"] > s["parallel"]["mean_in_tokens"] / 3
    recs = [json.loads(l) for l in open(root + "/out/live_records.jsonl")]
    assert recs[0]["refine_initial"]["text"] == recs[0]["parallel"][0]["text"]          # shared initial response

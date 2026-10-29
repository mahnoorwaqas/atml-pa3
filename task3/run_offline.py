"""Offline Task 3 pipeline (Steps 1,2,3,5).  Usage:
  python -m task3.run_offline --kind math --model_dir ATML-PA3-Data/<llm_folder> --out results/task3/math
  python -m task3.run_offline --kind mathvista --model_dir ATML-PA3-Data/<vlm_folder> --meta mathvista_meta.json --out results/task3/mathvista
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd

from common.utils import SEED, save_json
from . import answers as A
from .schema import load_tables, verify_alignment
from .data import build_pools, make_splits, make_orders, BUDGETS
from .engine import run_offline, bootstrap_diff, LLM_METHODS, VLM_EXTRA
from .signals import signal_analysis, illustrative_trajectories
from . import adaptive as AD

COMPARISONS = [("sc", "first"), ("selfcert_best", "sc"), ("logp_best", "sc"), ("selfcert_rwc", "sc"),
               ("selfcert_rwc", "selfcert_best"), ("visconf_rwc", "sc"), ("visconf_best", "sc"),
               ("visconf_rwc", "selfcert_rwc"), ("visconf_rwc", "vision_rwc"), ("visconf_rwc", "token_rwc"),
               ("visconf_best", "token_best")]


def run_dataset(kind, df, out, meta=None, schema=None, limit=None, n_boot=1000):
    os.makedirs(out, exist_ok=True)
    vlm = kind == "mathvista"
    bad = verify_alignment(df, schema, vlm)
    pools = build_pools(df, kind, schema, meta)
    qids_all = sorted(pools)
    if limit:
        qids_all = qids_all[:limit]
    n_dev = 200 if len(qids_all) >= 400 else max(1, len(qids_all) // 5)   # small inputs only occur in smoke tests
    sp = make_splits(qids_all, SEED, n_dev)
    save_json(sp, os.path.join(out, "splits.json"))
    qids = sp["dev"] + sp["eval"]                                  # evaluation order (permutation order)
    orders = make_orders(len(qids), seed=SEED)                     # reused by every method / policy
    np.save(os.path.join(out, "orders.npy"), orders)
    methods = LLM_METHODS + (VLM_EXTRA if vlm else [])
    res = run_offline(pools, qids, orders, methods)
    nd = len(sp["dev"])
    np.savez_compressed(os.path.join(out, "per_question.npz"), qids=np.array(qids), **{
        "acc_" + m: res["acc"][i] for i, m in enumerate(methods)}, **{
        "cost_" + m: res["cost"][i] for i, m in enumerate(methods)}, cov=res["cov"], distinct=res["distinct"],
        sel_rollout=res["sel_rollout"], sel_group=res["sel_group"], budgets=np.array(BUDGETS))
    # ---- Step 1 / 2 tables on held-out eval
    rows = []
    for mi, m in enumerate(methods):
        for bi, n in enumerate(BUDGETS):
            rows.append(dict(method=m, N=n, acc=res["acc"][mi, nd:, bi].mean(), cost=res["cost"][mi, nd:, bi].mean(),
                             avail_missed=res["avail_missed"][mi, nd:, bi].mean(),
                             coverage=res["cov"][nd:, bi].mean(), distinct=res["distinct"][nd:, bi].mean(),
                             no_correct_generated=1 - res["cov"][nd:, bi].mean()))
    pd.DataFrame(rows).to_csv(os.path.join(out, "methods_eval.csv"), index=False)
    comps = []
    for a, b in COMPARISONS:
        if a in methods and b in methods:
            for bi, n in enumerate(BUDGETS):
                if n in (4, 8, 16, 32):
                    d, lo, hi = bootstrap_diff(res["acc"][methods.index(a), nd:, bi], res["acc"][methods.index(b), nd:, bi], n_boot)
                    comps.append(dict(a=a, b=b, N=n, diff=d, lo=lo, hi=hi))
    pd.DataFrame(comps).to_csv(os.path.join(out, "paired_bootstrap.csv"), index=False)
    # ---- Step 3
    sig = signal_analysis(pools, sp["eval"], vlm)
    save_json(sig, os.path.join(out, "signals.json"))
    if vlm:
        save_json(illustrative_trajectories(pools, sp["eval"]), os.path.join(out, "trajectories.json"))
    # ---- Step 5
    qs = "visconf" if vlm else "selfcert"
    Cd = AD.precompute(pools, sp["dev"], orders[:nd], qs)
    Ce = AD.precompute(pools, sp["eval"], orders[nd:], qs)
    out_ad = {"quality_score": qs}
    for fam in ("agreement", "quality"):
        pol = AD.select_policy(Cd, fam)
        rep, stop = AD.held_out_report(Ce, pol)
        out_ad[fam] = {"policy": {k: pol[k] for k in ("tau", "gamma")}, "dev": pol["dev"],
                       "fixed32_dev": pol["fixed32_dev"], "heldout": rep}
        np.save(os.path.join(out, f"stop_{fam}.npy"), stop)
        wrong_early = (stop < 3) & ~np.take_along_axis(Ce["sc_ok"], stop[..., None], -1)[..., 0]
        np.save(os.path.join(out, f"wrong_early_{fam}.npy"), wrong_early)
    save_json(out_ad, os.path.join(out, "adaptive.json"))
    save_json({"scorer": A.scorer_version(), "misaligned_rows": int(bad), "n_questions": len(qids), "seed": SEED,
               "n_dev": nd, "n_eval": len(sp["eval"])}, os.path.join(out, "run_info.json"))
    with open(os.path.join(out, "pools.pkl"), "wb") as f:
        pickle.dump({q: pools[q] for q in qids}, f)
    return res, sig, out_ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["math", "mathvista"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", help="MathVista metadata json/parquet (pid, question_type, answer_type, choices, precision)")
    ap.add_argument("--schema", help="json mapping logical field -> column")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    schema = json.load(open(a.schema)) if a.schema else None
    meta = None
    if a.meta:
        m = pd.read_parquet(a.meta) if a.meta.endswith(".parquet") else pd.read_json(a.meta)
        meta = {str(r["pid"]): r for r in m.to_dict("records")}
    df = load_tables(a.model_dir, vlm=a.kind == "mathvista", schema=schema)
    run_dataset(a.kind, df, a.out, meta, schema, a.limit)


if __name__ == "__main__":
    main()

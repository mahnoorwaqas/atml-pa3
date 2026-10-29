"""Figures + case-study extraction for Task 3.  python -m task3.analyze --dir results/task3/math"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import save_json
from . import adaptive as AD


def figures(d):
    df = pd.read_csv(os.path.join(d, "methods_eval.csv"))
    ad = json.load(open(os.path.join(d, "adaptive.json")))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for m, g in df.groupby("method"):
        ax[0].plot(g["N"], 100 * g["acc"], marker="o", label=m)
        ax[1].plot(g["cost"], 100 * g["acc"], marker="o", label=m)
    c = df[df.method == "sc"]
    ax[0].plot(c["N"], 100 * c["coverage"], "k--", label="oracle coverage")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("candidates N"); ax[0].set_ylabel("accuracy (%)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("mean generated tokens / question"); ax[0].legend(fontsize=6)
    for fam, mk in (("agreement", "s"), ("quality", "^")):
        r = ad[fam]["heldout"]; ax[1].scatter([r["cost"]], [100 * r["acc"]], marker=mk, s=70, c="k", label=f"adaptive-{fam}")
    ax[1].legend(fontsize=6); plt.tight_layout(); plt.savefig(os.path.join(d, "fig_accuracy_cost.png"), dpi=150); plt.close()
    f = df[df.method == "sc"]
    plt.figure(figsize=(4.5, 3.4))
    plt.bar(range(len(f)), f["no_correct_generated"], label="no correct candidate (sampling failure)")
    plt.bar(range(len(f)), f["avail_missed"], bottom=f["no_correct_generated"], label="correct available, missed (selection failure)")
    plt.xticks(range(len(f)), f["N"]); plt.xlabel("N (self-consistency)"); plt.ylabel("fraction of questions"); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(d, "fig_failure_decomposition.png"), dpi=150); plt.close()


def case_studies(d, k=3):
    """Rule: over sorted held-out ids take the first k cases of each kind (no cherry-picking by eye)."""
    pools = pickle.load(open(os.path.join(d, "pools.pkl"), "rb"))
    sp = json.load(open(os.path.join(d, "splits.json")))
    pq = np.load(os.path.join(d, "per_question.npz"), allow_pickle=True)
    qids = list(pq["qids"]); nd = len(sp["dev"])
    methods = [k_[4:] for k_ in pq.files if k_.startswith("acc_")]
    ia = lambda m: pq["acc_" + m][nd:, -1]
    out = {"selection_gap": [], "rwc_fixes_sc": [], "premature_stop": []}
    wrong_early = np.load(os.path.join(d, "wrong_early_agreement.npy"))      # [Q_eval, orders]
    for j, qid in enumerate(sorted(sp["eval"])):
        i = sp["eval"].index(qid); p = pools[qid]
        info = dict(qid=qid, gold=p.gold, pool_correct_frac=float(p.correct.mean()),
                    answers=[None if a is None else str(a) for a in p.answers[:8]])
        if p.correct.any() and ia("sc")[i] < 0.5 and len(out["selection_gap"]) < k: out["selection_gap"].append(info)
        if "selfcert_rwc" in methods and ia("selfcert_rwc")[i] > 0.5 > ia("sc")[i] and len(out["rwc_fixes_sc"]) < k:
            out["rwc_fixes_sc"].append(info)
        # premature stop: an early stop returned a wrong answer although fixed-32 self-consistency is right
        if wrong_early[i].any() and len(out["premature_stop"]) < k and pq["acc_sc"][nd + i, -1] > 0.5:
            out["premature_stop"].append({**info, "n_orders_with_wrong_early_stop": int(wrong_early[i].sum())})
    save_json(out, os.path.join(d, "cases.json"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    figures(a.dir); case_studies(a.dir)

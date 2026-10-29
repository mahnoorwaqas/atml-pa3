"""Step 3: what do the signals capture?  (N = 32, within-question analyses.)"""
import numpy as np
from scipy.stats import rankdata, spearmanr

from .engine import OrderedView, sel_sc, sel_best, sel_rwc


def _quartile_diff(score, correct):
    """mean correctness of top quartile minus bottom quartile (ties broken by rollout order)."""
    n = len(score); k = max(1, int(np.ceil(n / 4)))
    o = np.argsort(-np.asarray(score, float), kind="stable")
    return correct[o[:k]].mean() - correct[o[-k:]].mean()


def signal_analysis(pools, qids, vlm=False):
    names = ["logp", "selfcert", "length_neg"] + (["vision", "visconf", "token"] if vlm else [])
    out = {"per_question": {}, "summary": {}}
    rows = {n: {"qdiff": [], "rho": []} for n in names}
    shortest_acc, conf_vis_corr, ident_err, used = [], [], 0.0, 0
    for qid in qids:
        p = pools[qid]
        view = OrderedView(p, np.arange(len(p.tokens)))
        n = len(p.tokens)
        y = p.correct.astype(float)
        shortest_acc.append(float(y[int(np.argmin(p.tokens))]))
        if vlm:
            c = view.calibrated(n)
            for i in range(n):
                s, e = view.bounds[i], view.bounds[i + 1]
                ci, vi = c[s:e], view.v[s:e].astype(float)
                if e - s > 2 and ci.std() > 0 and vi.std() > 0:
                    conf_vis_corr.append((qid, np.corrcoef(ci, vi)[0, 1]))
            Z, Zt = view.score("visconf", n), view.score("token", n)
            cov_t = np.array([np.mean(c[view.bounds[i]:view.bounds[i + 1]] * view.v[view.bounds[i]:view.bounds[i + 1]])
                              - c[view.bounds[i]:view.bounds[i + 1]].mean() * view.v[view.bounds[i]:view.bounds[i + 1]].mean()
                              for i in range(n)])
            ident_err = max(ident_err, float(np.max(np.abs((Zt - Z) - cov_t))))
        if y.min() == y.max():          # no within-question variation: association undefined
            continue
        used += 1
        for nm in names:
            s = view.score(nm, n)
            rows[nm]["qdiff"].append(_quartile_diff(s, y))
            r = spearmanr(s, y)[0]
            rows[nm]["rho"].append(0.0 if np.isnan(r) else r)
    out["summary"]["n_questions_with_variation"] = used
    out["summary"]["n_questions"] = len(qids)
    for nm in names:
        out["summary"][nm] = {"top_minus_bottom_quartile": float(np.mean(rows[nm]["qdiff"])) if rows[nm]["qdiff"] else None,
                              "mean_spearman_with_correct": float(np.mean(rows[nm]["rho"])) if rows[nm]["rho"] else None}
    out["summary"]["shortest_rollout_accuracy"] = float(np.mean(shortest_acc))
    if vlm:
        per_q = {}
        for q, r in conf_vis_corr:
            per_q.setdefault(q, []).append(r)
        qm = np.array([np.mean(v) for v in per_q.values()]) if per_q else np.array([np.nan])
        out["summary"]["conf_vision_corr"] = {"mean": float(np.nanmean(qm)), "median": float(np.nanmedian(qm)),
                                              "std": float(np.nanstd(qm)),
                                              "frac_negative": float(np.mean(qm < 0)), "n_questions": int(len(per_q))}
        out["summary"]["fusion_identity_max_abs_err"] = ident_err   # Z_token - Z == Cov_t(c, v)
    return out


def illustrative_trajectories(pools, qids, k=3, bins=40):
    """Rule: over the (sorted) held-out ids, take the first k questions where VisConf-argmax (all 32 rollouts)
    is correct while self-consistency is wrong ('visconf_wins'), and first k for the reverse ('sc_wins').
    For each, store binned per-token c and v of the chosen rollout and of the SC-chosen rollout."""
    res = {"visconf_wins": [], "sc_wins": []}
    for qid in sorted(qids):
        p = pools[qid]
        view = OrderedView(p, np.arange(len(p.tokens)))
        n = len(p.tokens)
        a_v = sel_best(view.score("visconf", n)); a_s = sel_sc(p.group)
        key = "visconf_wins" if (p.correct[a_v] and not p.correct[a_s]) else \
              "sc_wins" if (p.correct[a_s] and not p.correct[a_v]) else None
        if key is None or len(res[key]) >= k:
            continue
        c = view.calibrated(n)

        def series(i):
            s, e = view.bounds[i], view.bounds[i + 1]
            edges = np.linspace(s, e, bins + 1).astype(int)
            f = lambda x: [float(x[edges[j]:max(edges[j + 1], edges[j] + 1)].mean()) for j in range(bins)]
            return {"rollout": int(i), "tokens": int(e - s), "c": f(c), "v": f(view.v.astype(float))}
        res[key].append({"qid": qid, "gold": p.gold, "visconf_pick": {"answer": p.answers[a_v], **series(a_v)},
                         "sc_pick": {"answer": p.answers[a_s], **series(a_s)}})
    return res

"""Step 5: adaptive sampling (agreement-based and quality-aware) simulated on the supplied pools."""
import itertools
import numpy as np

from .engine import OrderedView, sel_rwc, rwc_margin

CHECKS = [4, 8, 16]
FULL = 32
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]


def precompute(pools, qids, orders, quality_score):
    """Per (question, order, n in CHECKS+[32]) cache using ONLY the first n revealed rollouts.
    quality_score: 'selfcert' (MATH) or 'visconf' (MathVista)."""
    Q, O = len(qids), orders.shape[1]
    ns = CHECKS + [FULL]
    sc_ok = np.zeros((Q, O, 4), bool); frac = np.zeros((Q, O, 4)); uniq = np.zeros((Q, O, 4), bool)
    noans = np.zeros((Q, O, 4), bool); agree = np.zeros((Q, O, 4), bool); margin = np.zeros((Q, O, 4))
    cost = np.zeros((Q, O, 4)); pool_frac = np.zeros(Q)
    for qi, qid in enumerate(qids):
        p = pools[qid]
        pool_frac[qi] = p.correct.mean()
        for oi in range(O):
            view = OrderedView(p, orders[qi, oi])
            for ji, n in enumerate(ns):
                idx = view.order[:n]
                g = p.group[idx]
                cnt, first = {}, {}
                for i, a in enumerate(g):
                    cnt[a] = cnt.get(a, 0) + 1; first.setdefault(a, i)
                m = max(cnt.values())
                leaders = [a for a, c in cnt.items() if c == m]
                lead = min(leaders, key=lambda a: first[a])                # tie -> earliest supporting rollout
                sc_ok[qi, oi, ji] = p.correct[idx[first[lead]]]
                frac[qi, oi, ji] = m / n
                uniq[qi, oi, ji] = len(leaders) == 1
                noans[qi, oi, ji] = lead == -1
                s = view.score(quality_score, n)
                pos, W = sel_rwc(g, s, return_votes=True)
                agree[qi, oi, ji] = g[pos] == lead
                margin[qi, oi, ji] = rwc_margin(W)
                cost[qi, oi, ji] = p.tokens[idx].sum()
    return dict(sc_ok=sc_ok, frac=frac, uniq=uniq, noans=noans, agree=agree, margin=margin, cost=cost,
                pool_frac=pool_frac, ns=ns)


def stop_index(C, tau, gamma=None):
    """First check (0..2) satisfying the policy else 3 (=32).  gamma=None -> agreement-only."""
    ok = C["uniq"][..., :3] & ~C["noans"][..., :3] & (C["frac"][..., :3] >= tau - 1e-12)
    if gamma is not None:
        ok = ok & C["agree"][..., :3] & (C["margin"][..., :3] >= gamma - 1e-12)
    any_ok = ok.any(-1)
    return np.where(any_ok, ok.argmax(-1), 3)


def evaluate_policy(C, stop):
    sel = np.take_along_axis(C["sc_ok"], stop[..., None], -1)[..., 0]
    cost = np.take_along_axis(C["cost"], stop[..., None], -1)[..., 0]
    nroll = np.array(C["ns"])[stop]
    return {"acc": float(sel.mean()), "cost": float(cost.mean()), "rollouts": float(nroll.mean()),
            "stop_dist": {int(n): float((nroll == n).mean()) for n in C["ns"]}}


def select_policy(Cdev, family, tol=0.01):
    """Lowest mean token cost with dev accuracy >= fixed-32 SC accuracy - tol; ties: higher acc, smaller tau, gamma."""
    fixed = evaluate_policy(Cdev, np.full(Cdev["sc_ok"].shape[:2], 3))
    grid = [(t, None) for t in TAUS] if family == "agreement" else list(itertools.product(TAUS, GAMMAS))
    cands = [(fixed["cost"], -fixed["acc"], 9.0, 9.0, None, None, fixed)]          # fixed-32 fallback
    for t, g in grid:
        r = evaluate_policy(Cdev, stop_index(Cdev, t, g))
        if r["acc"] >= fixed["acc"] - tol - 1e-12:
            cands.append((r["cost"], -r["acc"], t, 0.0 if g is None else g, t, g, r))
    best = min(cands, key=lambda x: x[:4])
    return {"tau": best[4], "gamma": best[5], "dev": best[6], "fixed32_dev": fixed}


def held_out_report(Ceval, policy):
    stop = stop_index(Ceval, policy["tau"], policy["gamma"]) if policy["tau"] is not None \
        else np.full(Ceval["sc_ok"].shape[:2], 3)
    rep = evaluate_policy(Ceval, stop)
    full_ok = Ceval["sc_ok"][..., 3]
    sel_ok = np.take_along_axis(Ceval["sc_ok"], stop[..., None], -1)[..., 0]
    early = stop < 3
    wrong_early = early & ~sel_ok
    pf = np.broadcast_to(Ceval["pool_frac"][:, None], stop.shape)
    rep["early_stops"] = int(early.sum())
    rep["incorrect_early_stops"] = int(wrong_early.sum())
    rep["incorrect_early_recovered_by_fixed32"] = int((wrong_early & full_ok).sum())
    rep["mean_pool_correct_frac_wrong_early"] = float(pf[wrong_early].mean()) if wrong_early.any() else None
    rep["mean_pool_correct_frac_right_early"] = float(pf[early & sel_ok].mean()) if (early & sel_ok).any() else None
    rep["fixed"] = {int(n): evaluate_policy(Ceval, np.full(stop.shape, j)) for j, n in enumerate(Ceval["ns"])}
    return rep, stop

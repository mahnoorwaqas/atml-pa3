"""Offline selection engine: scores, selectors (first / SC / best / RWC), VisConf, tokenwise fusion."""
import numpy as np
from scipy.stats import rankdata

from .data import BUDGETS, N_ORDERS

EPS = 1e-8
TIE_TOL = 1e-9


# ---------------------------------------------------------------- scores on a candidate prefix
class OrderedView:
    """Rollouts of one question in a given order, with prefix-contiguous token arrays."""

    def __init__(self, pool, order):
        self.pool, self.order = pool, np.asarray(order, dtype=int)
        o = self.order
        lens = (pool.offs[1:] - pool.offs[:-1])[o]
        self.lens = lens
        self.bounds = np.concatenate([[0], np.cumsum(lens)])
        self.kl = np.concatenate([pool.kl_flat[pool.offs[i]:pool.offs[i + 1]] for i in o]) if len(o) else np.zeros(0)
        self.v = (np.concatenate([pool.v_flat[pool.offs[i]:pool.offs[i + 1]] for i in o])
                  if pool.v_flat is not None else None)

    def calibrated(self, n):
        """Token-level c_{i,t} (flat over first n rollouts), calibrated with prefix-only percentiles."""
        kl = self.kl[:self.bounds[n]].astype(np.float64)
        if kl.size == 0:
            return kl
        q05, q95 = np.percentile(kl, [5, 95])                # linear interpolation
        return np.clip((kl - q05) / max(q95 - q05, EPS), 0.0, 1.0)

    @staticmethod
    def _seg_mean(x, bounds, n):
        cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
        s = cs[bounds[1:n + 1]] - cs[bounds[:n]]
        ln = np.maximum(np.diff(bounds[:n + 1]), 1)
        return s / ln

    def score(self, name, n):
        p, o = self.pool, self.order[:n]
        if name == "logp":      return p.L[o]
        if name == "selfcert":  return p.K[o]
        if name == "length_neg": return -p.tokens[o].astype(float)
        if name == "vision":    return p.V[o]
        c = self.calibrated(n)
        C = self._seg_mean(c, self.bounds, n)
        if name == "visconf":   return p.V[o] * C
        if name == "token":
            return self._seg_mean(c * self.v[:self.bounds[n]], self.bounds, n)
        raise KeyError(name)


# ---------------------------------------------------------------- selectors (return position in the prefix)
def sel_first(g):
    return 0


def sel_sc(g):
    best, bc, first = None, -1, {}
    cnt = {}
    for i, a in enumerate(g):
        cnt[a] = cnt.get(a, 0) + 1
        first.setdefault(a, i)
    m = max(cnt.values())
    return min(first[a] for a, c in cnt.items() if c == m)          # tie -> earliest supporting rollout


def sel_best(s):
    return int(np.argmax(s))                                        # first max == earliest in order


def rwc_weights(s):
    n = len(s)
    r = rankdata(-np.asarray(s, dtype=float), method="average")     # rank 1 = best, ties averaged
    return (n - r) / (n - 1)


def sel_rwc(g, s, return_votes=False):
    n = len(g)
    if n == 1:
        return (0, {g[0]: 1.0}) if return_votes else 0
    w = rwc_weights(s)
    W, first = {}, {}
    for i, a in enumerate(g):
        W[a] = W.get(a, 0.0) + w[i]
        first.setdefault(a, i)
    m = max(W.values())
    pos = min(first[a] for a, v in W.items() if abs(v - m) <= TIE_TOL)
    return (pos, W) if return_votes else pos


def rwc_margin(W):
    """(largest - second largest vote) / total vote weight; runner-up = 0 if one group."""
    v = sorted(W.values(), reverse=True)
    tot = sum(v)
    if tot <= 0:
        return 0.0
    return (v[0] - (v[1] if len(v) > 1 else 0.0)) / tot


# ---------------------------------------------------------------- method registry
LLM_METHODS = ["first", "sc", "logp_best", "selfcert_best", "selfcert_rwc"]
VLM_EXTRA = ["vision_best", "vision_rwc", "visconf_best", "visconf_rwc", "token_best", "token_rwc"]


def select(method, view, n):
    g = view.pool.group[view.order[:n]]
    if method == "first":  return 0
    if method == "sc":     return sel_sc(g)
    if method == "length_best": return sel_best(view.score("length_neg", n))
    score, agg = method.rsplit("_", 1) if method not in ("first", "sc") else (None, None)
    s = view.score(score, n)
    return sel_best(s) if agg == "best" else sel_rwc(g, s)


def run_offline(pools, qids, orders, methods, budgets=BUDGETS):
    """Evaluate all methods on pools in `qids` order (orders indexed by position in qids).
    Returns dict of arrays: acc[m] [Q,nB], cost[m] [Q,nB], avail_missed[m] [Q,nB], cov [Q,nB],
    distinct [Q,nB], sel_rollout [Q,O,nB,M] (original rollout idx), sel_group same shape."""
    Q, O, M, B = len(qids), orders.shape[1], len(methods), len(budgets)
    acc = np.zeros((M, Q, B)); cost = np.zeros((M, Q, B)); am = np.zeros((M, Q, B))
    cov = np.zeros((Q, B)); dist = np.zeros((Q, B))
    sel_r = np.zeros((Q, O, B, M), np.int16); sel_g = np.zeros((Q, O, B, M), np.int32)
    for qi, qid in enumerate(qids):
        pool = pools[qid]
        for oi in range(O):
            view = OrderedView(pool, orders[qi, oi])
            for bi, n in enumerate(budgets):
                idx = view.order[:n]
                c_any = pool.correct[idx].any()
                cov[qi, bi] += c_any / O
                dist[qi, bi] += len(set(pool.group[idx].tolist())) / O
                for mi, m in enumerate(methods):
                    pos = select(m, view, n)
                    r = idx[pos]
                    ok = bool(pool.correct[r])
                    acc[mi, qi, bi] += ok / O
                    am[mi, qi, bi] += (c_any and not ok) / O
                    cost[mi, qi, bi] += (pool.tokens[idx[0]] if m == "first" else pool.tokens[idx].sum()) / O
                    sel_r[qi, oi, bi, mi] = r; sel_g[qi, oi, bi, mi] = pool.group[r]
    return {"methods": list(methods), "budgets": list(budgets), "acc": acc, "cost": cost, "avail_missed": am,
            "cov": cov, "distinct": dist, "sel_rollout": sel_r, "sel_group": sel_g}


def bootstrap_diff(a, b, n_boot=1000, seed=6304):
    """Paired question-level bootstrap of mean(a-b); a, b: per-question accuracies (already order-averaged)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

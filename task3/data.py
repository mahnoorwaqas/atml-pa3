"""Splits, rollout orders and per-question pools for Task 3."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from common.utils import SEED, save_json
from . import answers as A
from .schema import resolve

BUDGETS = [1, 2, 4, 8, 16, 32]
N_ORDERS = 10
N_ROLL = 32


def make_splits(question_ids, seed=SEED, n_dev=200):
    ids = sorted(str(q) for q in question_ids)               # lexicographic sort
    perm = np.random.default_rng(seed).permutation(len(ids))
    ids = [ids[i] for i in perm]
    return {"dev": ids[:n_dev], "eval": ids[n_dev:], "live50": ids[n_dev:n_dev + 50]}


def make_orders(n_questions, n_roll=N_ROLL, n_orders=N_ORDERS, seed=SEED):
    """[Q, n_orders, n_roll] seeded permutations; questions indexed in the *sorted-id* order."""
    rng = np.random.default_rng(seed)
    return np.stack([[rng.permutation(n_roll) for _ in range(n_orders)] for _ in range(n_questions)]).astype(np.int16)


@dataclass
class Pool:
    qid: str
    tokens: np.ndarray            # [R] retained token counts
    group: np.ndarray             # [R] answer-group ids (-1 = NO_ANSWER)
    correct: np.ndarray           # [R] bool
    L: np.ndarray                 # [R] mean log-prob
    K: np.ndarray                 # [R] self-certainty (mean KL(U||p))
    kl_flat: np.ndarray           # concatenated token KL
    offs: np.ndarray              # [R+1]
    V: Optional[np.ndarray] = None        # [R] visual engagement
    v_flat: Optional[np.ndarray] = None   # concatenated per-token v_{i,t}
    answers: Optional[list] = None        # raw extracted answers (for case studies)
    gold: Optional[str] = None


def _meta_for(meta, sid):
    if meta is None:
        return {}
    return meta.get(str(sid), {}) if isinstance(meta, dict) else {}


def build_pools(df, kind, schema=None, meta=None):
    """df: joined table. kind: 'math' or 'mathvista'. meta: dict sample_id -> MathVista metadata.
    Returns dict qid -> Pool (rollouts sorted by rollout_id)."""
    c_pred, c_gold = resolve(df.columns, "pred", schema), resolve(df.columns, "gold", schema)
    c_lp, c_kl = resolve(df.columns, "logp", schema), resolve(df.columns, "kl", schema)
    c_nt = resolve(df.columns, "num_tokens", schema, required=False)
    vlm = kind == "mathvista"
    if vlm:
        c_ai, c_ap, c_ag = (resolve(df.columns, k, schema) for k in ("attn_image", "attn_prompt", "attn_gen"))
    pools = {}
    for sid, sub in df.groupby("sample_id", sort=True):
        sub = sub.sort_values("rollout_id")
        preds = [None if A.is_missing(a) else a for a in sub[c_pred]]
        gold = sub[c_gold].iloc[0]
        m = _meta_for(meta, sid)
        if kind == "math":
            gids, reps = A.group_math(preds)
            correct = np.array([g != A.NO_ANSWER and not A.is_missing(gold) and A.math_equiv(reps[g], str(gold))
                                for g in gids])
        else:
            canon = [A.mathvista_normalize(p, m) for p in preds]
            gids, reps = A.group_strings(canon)
            gc = A.gold_to_canonical(gold, m)
            correct = np.array([c is not None and gc is not None and c == gc for c in canon])
        lps = [np.asarray(x, dtype=np.float32) for x in sub[c_lp]]
        kls = [np.asarray(x, dtype=np.float32) for x in sub[c_kl]]
        lens = np.array([len(x) for x in kls])
        offs = np.concatenate([[0], np.cumsum(lens)])
        tokens = np.asarray(sub[c_nt], dtype=np.int64) if c_nt else lens
        p = Pool(qid=str(sid), tokens=tokens, group=np.array(gids, dtype=np.int32), correct=correct,
                 L=np.array([x.mean() if len(x) else 0.0 for x in lps]),
                 K=np.array([x.mean() if len(x) else 0.0 for x in kls]),
                 kl_flat=np.concatenate(kls) if len(kls) else np.zeros(0, np.float32), offs=offs,
                 answers=preds, gold=None if A.is_missing(gold) else str(gold))
        if vlm:
            vs = []
            for ai, ap, ag in zip(sub[c_ai], sub[c_ap], sub[c_ag]):
                ai, ap, ag = (np.asarray(z, dtype=np.float64) for z in (ai, ap, ag))
                vs.append((ai / np.maximum(ai + ap + ag, 1e-12)).astype(np.float32))
            p.v_flat = np.concatenate(vs)
            p.V = np.array([x.mean() if len(x) else 0.0 for x in vs])
        pools[str(sid)] = p
    return pools

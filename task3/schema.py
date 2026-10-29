"""Loading / joining the supplied ATML-PA3-Data tables.

The README.MD that ships with the data is the authority on column names.  Logical fields are mapped to
physical columns by CANDIDATES (first match wins).  If your export uses other names, pass
`--schema my_schema.json` ({"logical_field": "physical_column"}) -- nothing else needs to change.
"""
import os
import numpy as np
import pandas as pd

KEYS = ["dataset", "sample_id", "rollout_id"]

CANDIDATES = {
    # generations.parquet
    "text": ["text", "response", "generation", "generated_text", "output_text", "output"],
    "token_ids": ["token_ids", "generated_token_ids", "output_token_ids", "completion_token_ids", "tokens"],
    "num_tokens": ["num_tokens", "length", "n_tokens", "num_generated_tokens", "generated_length", "gen_len",
                   "token_count", "output_len"],
    "pred": ["extracted_answer", "pred_answer", "predicted_answer", "pred", "extracted", "answer_pred", "prediction"],
    "gold": ["gold_answer", "gold", "ground_truth", "reference_answer", "target", "answer"],
    "question": ["question", "problem", "prompt", "query"],
    # probability_metrics.parquet
    "logp": ["logp", "logprobs", "token_logp", "sampled_logp", "log_probs"],
    "kl": ["kl_u_p", "kl", "kl_uniform", "self_certainty_tokens"],
    # attention_metrics.parquet
    "attn_image": ["attn_image", "image_attn", "image_attention", "image", "A_image", "image_total", "attn_img"],
    "attn_prompt": ["attn_prompt", "prompt_attn", "prompt_attention", "prompt_text", "A_prompt", "prompt_total",
                    "attn_prompt_text", "prompt_text_attn"],
    "attn_gen": ["attn_gen", "generated_attn", "generated_attention", "generated_text", "A_generated",
                 "generated_total", "attn_generated", "attn_generated_text", "generated_text_attn"],
}


def resolve(df_cols, logical, override=None, required=True):
    if override and logical in override:
        if override[logical] in df_cols:
            return override[logical]
        raise KeyError(f"schema override {logical}->{override[logical]} not in columns {list(df_cols)}")
    for c in CANDIDATES[logical]:
        if c in df_cols:
            return c
    if required:
        raise KeyError(f"Could not find a column for '{logical}'. Available: {list(df_cols)}. "
                       f"Provide --schema JSON mapping '{logical}' to the right column (see README.MD).")
    return None


def _read(path):
    return pd.read_parquet(path)


def load_tables(model_dir, vlm=False, schema=None):
    """Join generations / probability_metrics (/ attention_metrics) on (dataset, sample_id, rollout_id)."""
    g = _read(os.path.join(model_dir, "generations.parquet"))
    p = _read(os.path.join(model_dir, "probability_metrics.parquet"))
    keep_p = [c for c in p.columns if c in KEYS or c in (resolve(p.columns, "logp", schema), resolve(p.columns, "kl", schema))]
    df = g.merge(p[keep_p], on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_prob"))
    if vlm:
        a = _read(os.path.join(model_dir, "attention_metrics.parquet"))
        cols = [resolve(a.columns, k, schema) for k in ("attn_image", "attn_prompt", "attn_gen")]
        df = df.merge(a[KEYS + cols], on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_attn"))
    if len(df) != len(g):
        raise ValueError(f"join dropped rows: generations={len(g)} joined={len(df)}")
    return df


def verify_alignment(df, schema=None, vlm=False):
    """Token-list lengths must agree across logp / kl / (attention) / num_tokens. Returns #bad rows."""
    lp, kl = resolve(df.columns, "logp", schema), resolve(df.columns, "kl", schema)
    nt = resolve(df.columns, "num_tokens", schema, required=False)
    cols = [lp, kl] + ([resolve(df.columns, k, schema) for k in ("attn_image", "attn_prompt", "attn_gen")] if vlm else [])
    bad = 0
    for _, row in df.iterrows():
        ls = {len(row[c]) for c in cols}
        if nt is not None:
            ls.add(int(row[nt]))
        bad += len(ls) != 1
    return bad

"""Task 1 evaluation: build the 7 evaluation conditions, merge adapters into FRESH backbones, run benchmarks.

Conditions: baseline, full@mid, full@final, l2sp@final, lora4@final, lora16@final, oplora@final.
Backends: 'lm_eval' (lm-evaluation-harness; real runs) or 'synthetic' (tiny built-in tasks; smoke tests only).
"""
import argparse, functools, glob, json, os
import numpy as np
import torch
import yaml

from common import lora as L
from common.utils import save_json, get_device, seed_everything, timer

BENCH = {  # task -> (metric key(s) in lm-eval results)
    "gsm8k_cot": {"primary": "exact_match,flexible-extract", "extra": ["exact_match,strict-match"]},
    "hellaswag": {"primary": "acc_norm,none", "extra": ["acc,none"]},
    "boolq": {"primary": "acc,none", "extra": []},
    "arc_easy": {"primary": "acc_norm,none", "extra": ["acc,none"]},
}


def condition_list(final_update, mid_update):
    return [("baseline", None, 0), ("full_mid", "full", mid_update), ("full_final", "full", final_update),
            ("l2sp_final", "l2sp", final_update), ("lora4_final", "lora4", final_update),
            ("lora16_final", "lora16", final_update), ("oplora_final", "oplora", final_update)]


def load_base(model_name, dtype):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)


def build_condition_model(model_name, runs_dir, method, update, dtype=torch.float16):
    """Fresh backbone copy; full FT -> load trained weights (cast to dtype); adapters -> merge exported factors."""
    m = load_base(model_name, torch.float32 if method is None else dtype)
    if method is None:
        return m.to(dtype)
    d = os.path.join(runs_dir, method)
    if method in ("full", "l2sp"):
        sd = torch.load(os.path.join(d, f"model_u{update}.pt"))
        m.load_state_dict(sd); return m.to(dtype)
    st = torch.load(os.path.join(d, f"adapter_u{update}.pt"))        # projected factors for OPLoRA
    m = m.to(torch.float32)
    L.merge_adapters_into(m, st)
    return m.to(dtype)


@torch.no_grad()
def check_merge(model_name, adapter_path, batch_ids, dtype=torch.float32):
    """Max |logit| gap between the unmerged adapter model and the merged copy on a small batch (FP32)."""
    st = torch.load(adapter_path)
    unmerged = load_base(model_name, dtype)
    targets = sorted({n.split(".")[-1] for n in st})
    L.inject_lora(unmerged, targets, r=next(iter(st.values()))["A"].shape[0], alpha=1, dropout=0.0, mode="lora")
    for n, m in L.lora_modules(unmerged).items():
        m.A.data.copy_(st[n]["A"]); m.B.data.copy_(st[n]["B"]); m.scale = st[n]["scale"]
    merged = L.merge_adapters_into(load_base(model_name, dtype), st)
    unmerged.eval(); merged.eval()
    a, b = unmerged(batch_ids).logits, merged(batch_ids).logits
    return float((a - b).abs().max())


# ------------------------------------------------------------------ lm-eval backend
def run_lm_eval(model, tok, out_dir, batch_size=16, limit=None, seed=6304):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    orig = tok.apply_chat_template
    def thinking_off(*a, **kw):                                      # non-thinking chat template everywhere
        kw["enable_thinking"] = False
        return orig(*a, **kw)
    tok.apply_chat_template = thinking_off
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    seeds = dict(random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed)
    res = {}
    # GSM8K: zero-shot CoT, chat template, greedy, 1024-token cap, stock stop rules / filters
    r1 = evaluator.simple_evaluate(model=lm, tasks=["gsm8k_cot"], num_fewshot=0, apply_chat_template=True,
                                   gen_kwargs="do_sample=False,temperature=0.0,max_gen_toks=1024", limit=limit,
                                   log_samples=True, **seeds)
    # likelihood tasks: stock completion prompts, no chat template
    r2 = evaluator.simple_evaluate(model=lm, tasks=["hellaswag", "boolq", "arc_easy"], num_fewshot=0,
                                   apply_chat_template=False, limit=limit, log_samples=True, **seeds)
    tok.apply_chat_template = orig
    for r in (r1, r2):
        res.update(r["results"])
    samples = {**r1["samples"], **r2["samples"]}
    metrics = {}
    for t, spec in BENCH.items():
        metrics[t] = {"primary": res[t][spec["primary"]], "primary_name": spec["primary"],
                      **{k: res[t][k] for k in spec["extra"] if k in res[t]}}
    # GSM8K truncation flags (generation hit the 1024-token cap)
    gs = []
    for s in samples.get("gsm8k_cot", []):
        resp = s["resps"][0][0] if s.get("resps") else ""
        ntok = len(tok(resp, add_special_tokens=False)["input_ids"])
        gs.append({"doc_id": s["doc_id"], "response": resp, "filtered": s.get("filtered_resps"),
                   "exact_match_flexible": s.get("exact_match"), "n_tokens": ntok, "truncated": ntok >= 1023})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gsm8k_samples.jsonl"), "w") as f:
        for g in gs: f.write(json.dumps(g, default=str) + "\n")
    for t in ("hellaswag", "boolq", "arc_easy"):
        with open(os.path.join(out_dir, f"{t}_samples.jsonl"), "w") as f:
            for s in samples.get(t, []): f.write(json.dumps({k: s[k] for k in ("doc_id", "resps", "acc", "acc_norm") if k in s}, default=str) + "\n")
    metrics["gsm8k_cot"]["truncation_rate"] = float(np.mean([g["truncated"] for g in gs])) if gs else None
    return metrics


# ------------------------------------------------------------------ synthetic backend (smoke tests only)
def run_synthetic(model, tok, out_dir, **_):
    dev = next(model.parameters()).device
    model.eval()
    mc = [("what is the sum of 1 and 2", ["3", "4", "5"], 0), ("how many can you see", ["2", "9"], 0)] * 3
    def ll(ctx, cont):
        a = tok(ctx, add_special_tokens=False)["input_ids"]; b = tok(" " + cont, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([a + b], device=dev)
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits.float(), -1)[0]
        return sum(lp[len(a) - 1 + i, t].item() for i, t in enumerate(b)), len(cont)
    acc = accn = 0
    for ctx, ch, gold in mc:
        s = [ll(ctx, c) for c in ch]
        acc += int(np.argmax([x[0] for x in s]) == gold); accn += int(np.argmax([x[0] / x[1] for x in s]) == gold)
    prompts = [tok.apply_chat_template([{"role": "user", "content": "what is the sum of 1 and 2"}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)] * 2
    outs = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gsm8k_samples.jsonl"), "w") as f:
        for i, o in enumerate(outs): f.write(json.dumps({"doc_id": i, "response": o, "truncated": False}) + "\n")
    n = len(mc)
    return {"gsm8k_cot": {"primary": 0.0, "truncation_rate": 0.0}, "hellaswag": {"primary": accn / n, "acc,none": acc / n},
            "boolq": {"primary": acc / n}, "arc_easy": {"primary": accn / n, "acc,none": acc / n}}


def evaluate_all(cfg, runs_dir, out_dir, backend="lm_eval", tok=None, final_update=None, mid_update=None, only=None, **kw):
    from transformers import AutoTokenizer
    tok = tok or AutoTokenizer.from_pretrained(cfg["model_name"])
    dev = get_device(); table = {}
    runner = run_lm_eval if backend == "lm_eval" else run_synthetic
    for name, method, upd in condition_list(final_update, mid_update):
        if only and name not in only: continue
        model = build_condition_model(cfg["model_name"], runs_dir, method, upd, torch.float16 if dev.type == "cuda" else torch.float32).to(dev).eval()
        info = {}
        if method in ("lora4", "lora16", "oplora"):
            ids = tok("what is the sum of 1 and 2", return_tensors="pt")["input_ids"]
            info["merge_check_max_abs_logit_diff"] = check_merge(cfg["model_name"], os.path.join(runs_dir, method, f"adapter_u{upd}.pt"), ids)
        with timer(info, "eval_time_s"):
            table[name] = {**runner(model, tok, os.path.join(out_dir, name), **kw), **info}
        save_json(table[name], os.path.join(out_dir, name, "metrics.json"))
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    base = table.get("baseline")
    if base:
        for n, r in table.items():
            r["delta_pp"] = {t: 100 * (r[t]["primary"] - base[t]["primary"]) for t in BENCH if t in r}
    save_json(table, os.path.join(out_dir, "benchmark_table.json"))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml"); ap.add_argument("--runs_dir", default="results/task1")
    ap.add_argument("--out", default="results/task1/eval"); ap.add_argument("--backend", default="lm_eval")
    ap.add_argument("--final_update", type=int, default=1250); ap.add_argument("--mid_update", type=int, default=625)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    evaluate_all(cfg, a.runs_dir, a.out, a.backend, final_update=a.final_update, mid_update=a.mid_update,
                 only=a.only, batch_size=a.batch_size, limit=a.limit)


if __name__ == "__main__":
    main()

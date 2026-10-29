"""Weight-space (rho_k, eta) and activation (R(z)) diagnostics on down_proj of selected layers.

  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2   (U_k: leading-k LEFT singular vectors of the starting W0)
  eta   = ||dW||_F / ||W0||_F
  R(z)  = ||dW z|| / (||W0 z|| + 1e-8)  on cached STARTING-model down_proj inputs z (last non-pad prompt token)
All in FP32.  dW: full FT -> W_trained - W0;  adapters -> (alpha/r) B~ A~ (projected factors for OPLoRA).
"""
import argparse, json, os
import numpy as np
import torch

from common.utils import save_json, SEED, get_device


def sample_probe_ids(n_gsm, n_hs, k=32, seed=SEED):
    """Fix the probe set BEFORE looking at any model results; IDs saved by the caller."""
    rng = np.random.default_rng(seed)
    g = sorted(rng.choice(n_gsm, k, replace=False).tolist())
    h = sorted(rng.choice(n_hs, k, replace=False).tolist())
    return {"gsm8k_test": g, "hellaswag_val": h, "seed": seed}


def hellaswag_query(doc):
    """lm-eval hellaswag `preprocess` + doc_to_text: '<activity>: <ctx_a> <ctx_b.capitalize()>' (stock completion prompt)."""
    import re
    def pre(t):
        t = t.strip().replace(" [title]", ". "); t = re.sub("\\[.*?\\]", "", t); return t.replace("  ", " ")
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return pre(doc["activity_label"] + ": " + ctx)


def gsm8k_prompt(question, tok):
    """gsm8k_cot zero-shot text 'Q: ...\\nA:' wrapped in the non-thinking chat template."""
    return tok.apply_chat_template([{"role": "user", "content": f"Q: {question}\nA:"}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)


@torch.no_grad()
def cache_inputs(base_model, tok, prompts, layers, device, bs=8):
    """Forward hooks on down_proj of the STARTING model -> z at the last non-padding prompt token. {layer: [N, d_ff]}"""
    store = {l: [] for l in layers}; cur = {}
    hs = []
    for l in layers:
        mod = base_model.model.layers[l].mlp.down_proj
        hs.append(mod.register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, i[0].detach())))
    tok.padding_side = "right"
    base_model.eval()
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        base_model(**enc)
        last = enc["attention_mask"].sum(1) - 1
        for l in layers:
            store[l].append(cur[l][torch.arange(len(last)), last].float().cpu())
    for h in hs: h.remove()
    return {l: torch.cat(v) for l, v in store.items()}


def delta_weight(method, run_dir, update, layer, W0):
    key = f"model.layers.{layer}.mlp.down_proj"
    if method in ("full", "l2sp"):
        sd = torch.load(os.path.join(run_dir, f"model_u{update}.pt"))
        return sd[key + ".weight"].float() - W0
    st = torch.load(os.path.join(run_dir, f"adapter_u{update}.pt"))[key]
    return st["scale"] * (st["B"].float() @ st["A"].float())


def run_diagnostics(base_model, tok, runs_dir, conditions, layers, k, prompts_math, prompts_other, out_path, device):
    """conditions: list of (name, method, update). Returns nested dict and saves JSON."""
    base_model = base_model.float().to(device)
    zs = {"math": cache_inputs(base_model, tok, prompts_math, layers, device),
          "non_math": cache_inputs(base_model, tok, prompts_other, layers, device)}
    res = {}
    for l in layers:
        W0 = base_model.model.layers[l].mlp.down_proj.weight.detach().float().cpu()
        U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
        Uk = U[:, :k]
        for name, method, upd in conditions:
            dW = delta_weight(method, os.path.join(runs_dir, method), upd, l, W0)
            fro = dW.norm().item()
            rho = (Uk.T @ dW).pow(2).sum().item() / max(fro ** 2, 1e-30)
            eta = fro / W0.norm().item()
            r = {"rho_k": rho, "eta": eta, "dW_fro": fro}
            for dom in ("math", "non_math"):
                z = zs[dom][l]
                R = (z @ dW.T).norm(dim=1) / ((z @ W0.T).norm(dim=1) + 1e-8)
                r[f"R_{dom}_mean"] = R.mean().item(); r[f"R_{dom}_median"] = R.median().item()
                r[f"R_{dom}_all"] = R.tolist()
            r["R_ratio_math_over_nonmath"] = r["R_math_mean"] / max(r["R_non_math_mean"], 1e-30)
            res[f"{name}|layer{l}"] = r
    save_json({"layers": layers, "k": k, "results": res}, out_path)
    return res


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data import load_gsm8k_test
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--runs_dir", default="results/task1")
    ap.add_argument("--out", default="results/task1/diagnostics"); ap.add_argument("--layers", type=int, nargs="+", default=[0, 13, 27])
    ap.add_argument("--k", type=int, default=128); ap.add_argument("--final", type=int, default=1250); ap.add_argument("--mid", type=int, default=625)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model); dev = get_device()
    gsm = load_gsm8k_test(); hs = load_dataset("Rowan/hellaswag", split="validation")
    ids = sample_probe_ids(len(gsm), len(hs)); save_json(ids, os.path.join(a.out, "probe_ids.json"))
    pm = [gsm8k_prompt(gsm[i]["question"], tok) for i in ids["gsm8k_test"]]
    po = [hellaswag_query(hs[i]) for i in ids["hellaswag_val"]]
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    conds = [("full_mid", "full", a.mid), ("full_final", "full", a.final), ("l2sp_final", "l2sp", a.final),
             ("lora4_final", "lora4", a.final), ("lora16_final", "lora16", a.final), ("oplora_final", "oplora", a.final)]
    run_diagnostics(base, tok, a.runs_dir, conds, a.layers, a.k, pm, po, os.path.join(a.out, "diagnostics.json"), dev)


if __name__ == "__main__":
    main()

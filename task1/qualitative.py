"""Qualitative evaluation: 24 prompts (8 questions x plain/json/template) x 7 conditions, greedy, 512 new tokens.
Writes all responses grouped by (question, format), plus the 42-response annotation subset (first and last
underlying question) with automatic heuristics and EMPTY manual columns (annotate those yourself)."""
import argparse, csv, json, os, re
import torch
import yaml

from common.utils import save_json, get_device

FORMATS = ("plain", "json", "template")


def parse_questions(path):
    items = yaml.safe_load(open(path))
    parsed, order = {}, []
    for it in items:
        qid = it["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = re.sub(r"_(json|template)$", "", qid)
        prompt = it["paraphrases"][0]
        parsed[(base, fmt)] = {"id": qid, "prompt": prompt, "system": it.get("system")}
        if fmt == "plain" and base not in order:
            order.append(base)
    return parsed, order


@torch.no_grad()
def generate_all(model, tok, parsed, device, max_new=512):
    out = {}
    model.eval()
    for (base, fmt), q in parsed.items():
        msgs = ([{"role": "system", "content": q["system"]}] if q["system"] else []) + [{"role": "user", "content": q["prompt"]}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        new = g[0, enc["input_ids"].shape[1]:]
        resp = tok.decode(new, skip_special_tokens=True)
        out[(base, fmt)] = {"response": resp, "n_tokens": int(new.numel()),
                            "truncated": bool(new.numel() >= max_new and int(new[-1]) != tok.eos_token_id)}
    return out


def heuristics(resp, fmt):
    ngr = [tuple(resp.split()[i:i + 4]) for i in range(max(0, len(resp.split()) - 3))]
    rep = 1 - len(set(ngr)) / len(ngr) if ngr else 0.0
    h = {"repetition_4gram": round(rep, 3),
         "math_framing_markers": bool(re.search(r"\\boxed|####|step[- ]by[- ]step|The answer is|\$[^$]+\$", resp)),
         "n_words": len(resp.split())}
    if fmt == "json":
        m = re.search(r"\{.*\}", resp, re.S)
        try:
            json.loads(m.group(0)); h["json_parses"] = True
        except Exception:
            h["json_parses"] = False
    return h


def write_outputs(results, order, out_dir):
    """results: {condition: {(base, fmt): {...}}}"""
    os.makedirs(out_dir, exist_ok=True)
    grouped = {b: {f: {c: results[c][(b, f)] for c in results if (b, f) in results[c]} for f in FORMATS} for b in order}
    save_json(grouped, os.path.join(out_dir, "responses_all.json"))          # 24 prompts x 7 = 168 responses
    subset = [order[0], order[-1]]
    rows = []
    for b in subset:
        for f in FORMATS:
            for c in results:
                r = results[c][(b, f)]
                rows.append({"question": b, "format": f, "condition": c, "truncated": r["truncated"], **heuristics(r["response"], f),
                             "response": r["response"], "format_compliance": "", "coherence_relevance": "", "repetition": "",
                             "unnecessary_math_framing": "", "inappropriate_content": "", "excerpt_notes": ""})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out_dir, "annotation_subset_42.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return grouped, rows


def main():
    from transformers import AutoTokenizer
    from .evaluate import build_condition_model, condition_list
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml"); ap.add_argument("--runs_dir", default="results/task1")
    ap.add_argument("--questions", default="first_plot_questions.yaml"); ap.add_argument("--out", default="results/task1/qualitative")
    ap.add_argument("--final", type=int, default=1250); ap.add_argument("--mid", type=int, default=625)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); tok = AutoTokenizer.from_pretrained(cfg["model_name"]); dev = get_device()
    parsed, order = parse_questions(a.questions)
    assert len(parsed) == 24, f"expected 24 prompts, found {len(parsed)}"
    results = {}
    for name, method, upd in condition_list(a.final, a.mid):
        m = build_condition_model(cfg["model_name"], a.runs_dir, method, upd, torch.float16 if dev.type == "cuda" else torch.float32).to(dev)
        results[name] = generate_all(m, tok, parsed, dev); del m
    write_outputs(results, order, a.out)


if __name__ == "__main__":
    main()

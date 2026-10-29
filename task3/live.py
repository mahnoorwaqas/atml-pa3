"""Step 4: live comparison of single-long / parallel(3x1024) / sequential refinement on 50 MATH questions.
Usage: python -m task3.live --model Qwen/Qwen2.5-3B-Instruct --revision <from README.MD> --questions q50.jsonl --out results/task3/live
  q50.jsonl rows: {"qid":..., "problem":..., "gold":...}  (the 50 ids in splits.json['live50'])
"""
import argparse, os, time
import numpy as np

from common.utils import SEED, save_json, append_jsonl, peak_mem_gb, reset_peak_mem
from . import answers as A

SYSTEM = "You are a helpful assistant."
INSTR = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")
PROBLEM_SEP = "\n"          # problem followed by INSTR; adjust to match the supplied README.MD exactly
GEN = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)


def generate(model, tok, messages, max_new, seed, device):
    import torch
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, pad_token_id=tok.pad_token_id, **GEN)
    if device.type == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t
    new = out[0, enc["input_ids"].shape[1]:]
    eos = new.numel() > 0 and int(new[-1]) in {tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")}
    resp = tok.decode(new, skip_special_tokens=True)
    return dict(text=resp, in_tokens=int(enc["input_ids"].shape[1]), out_tokens=int(new.numel()),
                truncated=bool(new.numel() >= max_new and not eos), time_s=dt, seed=seed, max_new=max_new,
                answer=A.last_boxed(resp))


def sc_vote(answers):
    gids, reps = A.group_math(answers)
    cnt, first = {}, {}
    for i, g in enumerate(gids):
        cnt[g] = cnt.get(g, 0) + 1; first.setdefault(g, i)
    m = max(cnt.values())
    g = min((g for g, c in cnt.items() if c == m), key=lambda g: first[g])
    return answers[first[g]] if g != A.NO_ANSWER else None


def grade(ans, gold):
    return ans is not None and not A.is_missing(ans) and A.math_equiv(str(ans), str(gold))


def run_live(model, tok, questions, out_dir, device, limits=(3072, 1024, 2048), seed=SEED):
    L_single, L_par, L_rev = limits
    os.makedirs(out_dir, exist_ok=True)
    reset_peak_mem()
    recs = []
    for qi, q in enumerate(questions):
        base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["problem"] + PROBLEM_SEP + INSTR}]
        s0 = seed + 10 * qi
        single = generate(model, tok, base, L_single, s0 + 0, device)
        par = [generate(model, tok, base, L_par, s0 + 1 + k, device) for k in range(3)]
        init = par[0]                                                   # shared initial response
        msgs = base + [{"role": "assistant", "content": init["text"]}, {"role": "user", "content": REVIEW}]
        rev = generate(model, tok, msgs, L_rev, s0 + 4, device)
        sc_ans = sc_vote([p["answer"] for p in par])
        rec = dict(qid=q["qid"], gold=q["gold"], single=single, parallel=par, refine_initial=init, refine_revision=rev,
                   ans_single=single["answer"], ans_parallel=sc_ans, ans_init=init["answer"], ans_refine=rev["answer"])
        recs.append(rec); append_jsonl(os.path.join(out_dir, "live_records.jsonl"), rec)
    return summarize(recs, out_dir)


def summarize(recs, out_dir=None):
    def strat(name):
        if name == "single": calls = lambda r: [r["single"]]; ans = "ans_single"
        elif name == "parallel": calls = lambda r: r["parallel"]; ans = "ans_parallel"
        else: calls = lambda r: [r["refine_initial"], r["refine_revision"]]; ans = "ans_refine"
        c = [x for r in recs for x in calls(r)]
        return dict(accuracy=float(np.mean([grade(r[ans], r["gold"]) for r in recs])),
                    mean_in_tokens=float(sum(x["in_tokens"] for x in c) / len(recs)),
                    mean_out_tokens=float(sum(x["out_tokens"] for x in c) / len(recs)),
                    mean_time_s=float(sum(x["time_s"] for x in c) / len(recs)),
                    truncation_rate=float(np.mean([x["truncated"] for x in c])),
                    extraction_failure_rate=float(np.mean([x["answer"] is None for x in c])),
                    final_answer_missing_rate=float(np.mean([r[ans] is None for r in recs])))
    out = {n: strat(n) for n in ("single", "parallel", "refine")}
    w2c = c2w = c2c = w2w = 0
    for r in recs:
        a, b = grade(r["ans_init"], r["gold"]), grade(r["ans_refine"], r["gold"])
        w2c += (not a) and b; c2w += a and not b; c2c += a and b; w2w += (not a) and (not b)
    out["refine_transitions"] = dict(wrong_to_correct=int(w2c), correct_to_wrong=int(c2w),
                                     correct_to_correct=int(c2c), wrong_to_wrong=int(w2w))
    out["peak_gpu_mem_gb"] = peak_mem_gb(); out["n_questions"] = len(recs)
    if out_dir: save_json(out, os.path.join(out_dir, "live_summary.json"))
    return out


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.utils import read_jsonl, get_device
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--revision", default=None)
    ap.add_argument("--questions", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limits", default="3072,1024,2048")
    a = ap.parse_args()
    dev = get_device()
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.float16 if dev.type == "cuda" else torch.float32).to(dev).eval()
    run_live(model, tok, read_jsonl(a.questions), a.out, dev, tuple(int(x) for x in a.limits.split(",")))


if __name__ == "__main__":
    main()

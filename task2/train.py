"""Task 2 training: three sequential stages on one model (caption alignment -> visual instruction tuning -> natural VQA).
  python -m task2.train --config configs/task2.yaml --dataset results/task2/dataset.json --features results/task2/feats --out results/task2/main
"""
import argparse, json, math, os, time
import numpy as np
import torch
import yaml

from common.utils import (seed_everything, save_json, append_jsonl, get_device, peak_mem_gb, reset_peak_mem, path_size_bytes)
from .features import FeatureStore
from .model import VLM
from .evaluate import dev_eval
from .data import CAPTION_PROMPT, VQA_SUFFIX


def caption_ex(c):
    return {"image_id": c["image_id"], "question": c["question"], "target": c["target"], "kind": "caption"}


def pair_examples(p):
    return [{"image_id": i["image_id"], "question": i["question"] + VQA_SUFFIX, "target": i["target"], "kind": p["suite"] + ":" + p["type"]}
            for i in p["items"]]


def lr_factor(s, n, warm_frac, floor):
    w = max(1, int(round(warm_frac * n)))
    if s < w:
        return (s + 1) / w
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, n - w)))


class Sampler:
    """Seeded sampler. Stage 1: 2 passes over the caption set (per-epoch permutation = f(seed, epoch)).
    Stages 2-3: random draws (a draw = a complete pair, or two caption examples for replay)."""
    def __init__(self, cfg, ds, seed):
        self.cfg, self.seed = cfg, seed
        self.caps = [caption_ex(c) for c in ds["caps_train"]]
        self.pools = {"presence": ds["presence_train"], "counting": ds["counting_train"], "natural": ds["vqa_train"]}
        self.rng = np.random.default_rng(seed + 1)

    def stage1(self, pos, n):
        N = len(self.caps); out = []
        for k in range(pos, pos + n):
            ep, j = divmod(k, N)
            perm = np.random.default_rng(self.seed + 100 + ep).permutation(N)
            out.append(self.caps[perm[j]])
        return out

    def draw(self, stage):
        mix = self.cfg["mix"][f"stage{stage}"]; r = self.rng.random(); acc = 0
        for name, p in mix.items():
            acc += p
            if r < acc: break
        if name == "caption":
            i = self.rng.choice(len(self.caps), 2, replace=False)
            return name, [self.caps[i[0]], self.caps[i[1]]]
        pool = self.pools[name]
        return name, pair_examples(pool[self.rng.integers(len(pool))])


def stage_of(u, stages):
    acc = 0
    for k, s in enumerate(stages):
        if u < acc + s["updates"]:
            return k + 1, u - acc
        acc += s["updates"]
    return len(stages), stages[-1]["updates"]


def build_model(cfg, tok, feat_dim, qwen=None):
    if qwen is None:
        from transformers import AutoModelForCausalLM
        qwen = AutoModelForCausalLM.from_pretrained(cfg["qwen_name"], dtype=getattr(torch, cfg["backbone_dtype"]))
    qwen.config.use_cache = False
    m = VLM(qwen, tok, feat_dim, cfg["n_tokens"])
    m.add_lora(**{k: v for k, v in cfg["lora"].items()})
    return m


def train(cfg, ds, store, tok, out, qwen=None, resume=None):
    seed_everything(cfg["seed"]); dev = get_device(); os.makedirs(out, exist_ok=True)
    use_amp = bool(cfg.get("amp", True)) and dev.type == "cuda"
    model = build_model(cfg, tok, store.dim, qwen).to(dev)
    model.connector.float(); model.verify_frozen()
    if cfg.get("grad_checkpointing", True):
        model.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sampler = Sampler(cfg, ds, cfg["seed"])
    stages = cfg["stages"]; total = sum(s["updates"] for s in stages)
    ex_per_update = cfg["micro_bs"] * cfg["accum"]
    # calibration of connector output magnitude -- training data only
    cal_ex = sampler.caps[:cfg["micro_bs"]]
    cal = model.calibrate(store.get([e["image_id"] for e in cal_ex]), cal_ex)
    save_json(cal, os.path.join(out, "calibration.json"))
    opt = torch.optim.AdamW([{"params": list(model.connector.parameters()), "lr": stages[0]["lr_conn"], "weight_decay": cfg["weight_decay"], "name": "conn"}])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    summary = {"cfg": cfg, "calibration": cal, "stage_time_s": {}, "trainable_by_stage": {}, "stage_peak_mem_gb": {}, "skipped_updates": []}
    start = 0
    if resume:
        start = load_ckpt(resume, model, opt, scaler, sampler)
    dev_points = sorted({0} | set(range(cfg["dev_every"], total + 1, cfg["dev_every"])) |
                        {sum(s["updates"] for s in stages[:k]) for k in range(1, len(stages) + 1)})
    if start == 0 and 0 in dev_points:
        append_jsonl(os.path.join(out, "dev_log.jsonl"), {"update": 0, **dev_eval(model, ds, store)})
    lora_added, cur_stage = False, 0
    model.train(); log_path = os.path.join(out, "train_log.jsonl")
    for u in range(start, total):
        st, s_step = stage_of(u, stages)
        sc = stages[st - 1]
        if st != cur_stage:                                          # stage switch
            cur_stage = st; model.set_stage(st); reset_peak_mem(); stage_t0 = time.perf_counter()
            if st >= 2 and not lora_added:
                if not any(g.get("name") == "lora" for g in opt.param_groups):
                    opt.add_param_group({"params": [p for p in model.qwen.parameters() if p.requires_grad], "lr": sc["lr_lora"],
                                         "weight_decay": cfg["weight_decay"], "name": "lora"})
                lora_added = True
            summary["trainable_by_stage"][st] = model.trainable_count()
            model.verify_frozen()
        for g in opt.param_groups:
            peak = sc["lr_conn"] if g["name"] == "conn" else sc["lr_lora"]
            g["lr"] = peak * lr_factor(s_step, sc["updates"], cfg["warmup_frac"], cfg["lr_floor"])
        # ---- one optimizer update = 16 example presentations (8 draws x 2)
        if st == 1:
            exs = sampler.stage1(u * ex_per_update, ex_per_update); kinds = {"caption": ex_per_update}
        else:
            exs, kinds = [], {}
            for _ in range(ex_per_update // 2):
                name, pair = sampler.draw(st); exs += pair; kinds[name] = kinds.get(name, 0) + 1
        tot = 0.0
        for s in range(0, ex_per_update, cfg["micro_bs"]):
            mb = exs[s:s + cfg["micro_bs"]]
            feats = store.get([e["image_id"] for e in mb])
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                nll = model.example_nll(mb, feats, True, include_eos=True)       # per-example mean over supervised tokens
            loss = nll.sum() / ex_per_update                                       # example-weighted over the effective batch
            scaler.scale(loss).backward(); tot += loss.item()
        scaler.unscale_(opt)
        if u == start:                                                             # connector must receive gradients
            gn = sum(p.grad.abs().sum().item() for p in model.connector.parameters() if p.grad is not None)
            assert gn > 0, "no gradient reached the connector"
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.trainable(), cfg["clip"]))
        before = scaler.get_scale(); scaler.step(opt); scaler.update()
        if use_amp and scaler.get_scale() < before: summary["skipped_updates"].append(u + 1)
        opt.zero_grad(set_to_none=True)
        append_jsonl(log_path, {"update": u + 1, "stage": st, "loss": tot, "grad_norm": gnorm, "lr_conn": opt.param_groups[0]["lr"],
                                "lr_lora": opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else None, "mix": kinds})
        done = u + 1
        if done in dev_points:
            model.eval()
            with torch.no_grad():
                append_jsonl(os.path.join(out, "dev_log.jsonl"), {"update": done, "stage": st, **dev_eval(model, ds, store)})
            model.train()
        if done % cfg["ckpt_every"] == 0:
            save_ckpt(out, done, model, opt, scaler, sampler, st, s_step + 1)
        if s_step + 1 == sc["updates"]:                                            # stage endpoint: connector/adapters
            torch.save(model.connector.state_dict(), os.path.join(out, f"connector_stage{st}.pt"))
            if st >= 2: torch.save(model.lora_state(), os.path.join(out, f"adapters_stage{st}.pt"))
            if dev.type == "cuda": torch.cuda.synchronize()
            summary["stage_time_s"][st] = time.perf_counter() - stage_t0
            summary["stage_peak_mem_gb"][st] = peak_mem_gb()
    summary["final_ckpt_sizes"] = {f: path_size_bytes(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".pt")}
    save_json(summary, os.path.join(out, "summary.json"))
    return model, summary


def save_ckpt(out, u, model, opt, scaler, sampler, stage, stage_progress):
    torch.save({"update": u, "connector": model.connector.state_dict(), "lora": model.lora_state(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "sampler_rng": sampler.rng.bit_generator.state, "stage": stage,
                "stage_progress": stage_progress, "vis_scale": float(model.vis_scale), "rng_cpu": torch.get_rng_state()},
               os.path.join(out, f"ckpt_u{u}.pt"))


def load_ckpt(path, model, opt, scaler, sampler):
    ck = torch.load(path, weights_only=False)
    model.connector.load_state_dict(ck["connector"]); model.load_lora_state(ck["lora"]); model.vis_scale.fill_(ck["vis_scale"])
    if ck["update"] >= 0 and any(len(g["params"]) for g in ck["opt"]["param_groups"]) and len(ck["opt"]["param_groups"]) > 1:
        model.set_stage(2)
        opt.add_param_group({"params": [p for p in model.qwen.parameters() if p.requires_grad], "lr": 0.0, "weight_decay": 0.01, "name": "lora"})
    opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"]); sampler.rng.bit_generator.state = ck["sampler_rng"]
    torch.set_rng_state(ck["rng_cpu"])
    return ck["update"]


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--dataset", required=True); ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1); cfg[k] = yaml.safe_load(v)
    tok = AutoTokenizer.from_pretrained(cfg["qwen_name"])
    train(cfg, json.load(open(a.dataset)), FeatureStore(a.features), tok, a.out, resume=a.resume)


if __name__ == "__main__":
    main()

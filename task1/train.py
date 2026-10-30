"""Task 1 training: Full FT, Full FT + L2-SP, LoRA (r=4/16), OPLoRA (r=16, k=128) on MetaMathQA.
  python -m task1.train --config configs/task1.yaml --method lora16 --out results/task1/lora16 [--resume]
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common import lora as L
from common.utils import (seed_everything, save_json, load_json, append_jsonl, read_jsonl, get_device, count_params,
                          path_size_bytes, peak_mem_gb, reset_peak_mem)
from .data import collate


def resolve_cfg(path, method, overrides=None):
    base = yaml.safe_load(open(path))
    cfg = {k: v for k, v in base.items() if k != "methods"}
    cfg.update(base["methods"][method]); cfg["method"] = method
    cfg.update(overrides or {})
    return cfg


def schedule_points(cfg, n_total):
    total = n_total // (cfg["micro_bs"] * cfg["accum"])
    warm = cfg.get("warmup") or int(cfg["warmup_frac"] * total + 0.5)
    ck = sorted({max(1, int(f * total)) for f in cfg["ckpt_fracs"]})
    return total, warm, ck


def make_lr_lambda(warm, total):
    def f(u):
        if u < warm:
            return (u + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (u - warm) / max(1, total - warm)))
    return f


def param_groups(model):
    dec, nodec = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodec if (p.ndim < 2 or n.endswith(".bias")) else dec).append(p)
    return [{"params": dec, "weight_decay": None}, {"params": nodec, "weight_decay": 0.0}]


def micro_loss_sum(model, b, dev):
    ids, att, lab = (b[k].to(dev) for k in ("input_ids", "attention_mask", "labels"))
    logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1]
    tgt = lab[:, 1:]; m = tgt != -100
    return F.cross_entropy(logits[m].float(), tgt[m], reduction="sum")


def omega_penalty(model, ref):
    """0.5 * sum_j (theta_j - theta0_j)^2 over all unique parameters (biases/norms included; tied counted once)."""
    tot = 0.0
    for n, p in model.named_parameters():
        tot = tot + 0.5 * (p.float() - ref[n]).pow(2).sum()
    return tot


@torch.no_grad()
def omega_add_grad(model, ref, lam, grad_mult):
    """Same objective as omega_penalty, without building an autograd graph (saves ~2.4 GB on a T4).
    Returns Omega = 0.5 * sum (theta - theta0)^2 and adds d(lam*Omega)/dtheta = lam*(theta - theta0), times the
    GradScaler scale (grads are still scaled at this point), to p.grad. Mathematically identical to
    scaler.scale(lam * omega_penalty).backward()."""
    tot = 0.0
    for n, p in model.named_parameters():
        d = p.detach().float() - ref[n].to(p.device, non_blocking=True)     # stream theta_0 for this tensor only
        tot = tot + 0.5 * d.pow(2).sum()
        if p.grad is None:
            p.grad = (d * (lam * grad_mult)).to(p.dtype)
        else:
            p.grad.add_(d, alpha=lam * grad_mult)
    return tot


@torch.no_grad()
def displacement(model, ref_getter):
    s = 0.0
    for n, p in model.named_parameters():
        s += (p.detach().float().cpu() - ref_getter(n).float().cpu()).pow(2).sum().item()
    return math.sqrt(s)


def train(cfg, model=None, tokenizer=None, data=None):
    seed_everything(cfg["seed"])
    dev = get_device(); out = cfg["out"]; os.makedirs(out, exist_ok=True)
    kind = cfg["kind"]
    if data is None:
        data = torch.load(cfg["data_path"])
    data = data[: cfg["n_examples"]]
    total, warm, ck_updates = schedule_points(cfg, len(data))
    pad_id = tokenizer.pad_token_id if tokenizer is not None else 0
    use_amp = bool(cfg.get("amp", True)) and dev.type == "cuda"
    summary = {"method": cfg["method"], "cfg": {k: v for k, v in cfg.items() if k != "data"}, "total_updates": total,
               "warmup": warm, "ckpt_updates": ck_updates, "skipped_updates": []}
    if model is None:
        from transformers import AutoModelForCausalLM
        dt = torch.float32 if kind == "full" else getattr(torch, cfg["backbone_dtype"])
        model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=dt)
    model.config.use_cache = False
    model.to(dev)
    if kind == "full":
        for p in model.parameters(): p.requires_grad_(True)
    else:
        stats = L.inject_lora(model, cfg["targets"], cfg["r"], cfg["alpha"], cfg["lora_dropout"], mode=kind, k=cfg.get("k", 128))
        summary["overhead"] = stats
    if cfg.get("grad_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    # reference for L2-SP / displacement (theta_0 fixed)
    ref = None
    if kind == "full":
        ref = {n: p.detach().clone().float() for n, p in model.named_parameters()}
        ref = {n: v.cpu() for n, v in ref.items()}                      # theta_0 lives in CPU RAM (GPU memory is tight on a T4)
        if cfg["lam"] != 0:
            if torch.cuda.is_available(): ref = {n: v.pin_memory() for n, v in ref.items()}
            summary["overhead"] = {"reference_storage_bytes": sum(v.numel() * 4 for v in ref.values()), "reference_location": "cpu_pinned"}
    summary["trainable_params"] = (count_params(model, True) if kind == "full" else L.adapter_param_count(model))
    summary["total_params"] = count_params(model)
    opt = torch.optim.AdamW([{**g, "weight_decay": cfg["weight_decay"] if g["weight_decay"] is None else g["weight_decay"]}
                             for g in param_groups(model)], lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(warm, total))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trainable = [p for p in model.parameters() if p.requires_grad]
    start = 0
    if cfg.get("resume"):
        start = load_ckpt(cfg["resume"], model, opt, sched, scaler, kind)
    ex_per_update = cfg["micro_bs"] * cfg["accum"]
    stream = iter(torch.utils.data.DataLoader(data[start * ex_per_update:], batch_size=cfg["micro_bs"], shuffle=False,
                                              collate_fn=lambda b: collate(b, pad_id)))   # source order, no shuffling
    log_path = os.path.join(out, "train_log.jsonl")
    prev_summary = load_json(os.path.join(out, "summary.json")) if (start and os.path.exists(os.path.join(out, "summary.json"))) else {}
    if start and os.path.exists(log_path):
        kept = [r for r in read_jsonl(log_path) if r["update"] <= start]
        with open(log_path, "w") as f:
            for r in kept: f.write(json.dumps(r) + "\n")
    reset_peak_mem(); t_train = 0.0; t_ckpt = 0.0
    for u in range(start, total):
        t0 = time.perf_counter()
        mbs = [next(stream) for _ in range(cfg["accum"])]
        nvalid = sum(int((b["labels"][:, 1:] != -100).sum()) for b in mbs)
        sft = 0.0
        for b in mbs:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                ls = micro_loss_sum(model, b, dev)
            loss = ls / nvalid                                      # normalise by valid tokens of the effective batch
            scaler.scale(loss).backward(); sft += loss.item()
        rec = {"update": u + 1, "sft_loss": sft, "lr": sched.get_last_lr()[0], "examples_seen": (u + 1) * ex_per_update}
        if kind == "full" and cfg["lam"] > 0:
            t_om = time.perf_counter()
            om = omega_add_grad(model, ref, cfg["lam"], scaler.get_scale()); om_v = om.item()
            rec.update(omega=om_v, lam_omega=cfg["lam"] * om_v, total_loss=sft + cfg["lam"] * om_v, omega_time_s=time.perf_counter() - t_om)
        scaler.unscale_(opt)
        rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(trainable, cfg["clip"]))
        before = scaler.get_scale()
        scaler.step(opt); scaler.update()
        rec["loss_scale"] = scaler.get_scale()
        if use_amp and scaler.get_scale() < before:
            rec["skipped"] = True; summary["skipped_updates"].append(u + 1)
        sched.step(); opt.zero_grad(set_to_none=True)
        if dev.type == "cuda": torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        rec["time_s"] = time.perf_counter() - t0
        append_jsonl(log_path, rec)
        if (u + 1) in ck_updates:
            t1 = time.perf_counter()
            save_ckpts(cfg, model, opt, sched, scaler, u + 1, ref, summary)
            t_ckpt += time.perf_counter() - t1
    recs = read_jsonl(log_path)
    ov = dict(prev_summary.get("overhead") or {}); sess_ov = summary.get("overhead") or {}
    if prev_summary and ov:
        if "export_time_s" in sess_ov: ov["export_time_s"] = ov.get("export_time_s", 0) + sess_ov["export_time_s"]
        summary["overhead"] = ov
    summary.update(train_time_s=sum(r.get("time_s", 0.0) for r in recs), ckpt_time_s=prev_summary.get("ckpt_time_s", 0.0) + t_ckpt,
                   peak_mem_gb=max(peak_mem_gb(), prev_summary.get("peak_mem_gb", 0.0)),
                   skipped_updates=[r["update"] for r in recs if r.get("skipped")], resumed_from=start or None)
    save_json(summary, os.path.join(out, "summary.json"))
    return model, summary


def save_ckpts(cfg, model, opt, sched, scaler, u, ref, summary):
    out, kind = cfg["out"], cfg["kind"]
    files = {}
    if kind == "full":
        p = os.path.join(out, f"model_u{u}.pt")
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, p); files["model"] = p
    else:
        p = os.path.join(out, f"model_u{u}.pt"); torch.save(L.raw_adapter_state(model), p); files["model(raw adapter)"] = p
        t = time.perf_counter()
        p2 = os.path.join(out, f"adapter_u{u}.pt"); torch.save(L.adapter_state(model, True), p2); files["adapter(export)"] = p2
        if kind == "oplora":
            summary.setdefault("overhead", {})["export_time_s"] = summary["overhead"].get("export_time_s", 0) + time.perf_counter() - t
    p3 = os.path.join(out, f"train_state_u{u}.pt")
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "update": u,
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "data_position_examples": u * cfg["micro_bs"] * cfg["accum"]}, p3)
    files["train_state"] = p3
    rec = {"update": u, "sizes_bytes": {k: path_size_bytes(v) for k, v in files.items()}}
    if kind == "full" and ref is not None:
        rec["displacement_l2"] = displacement(model, lambda n: ref[n])
    append_jsonl(os.path.join(out, "ckpt_meta.jsonl"), rec)


def load_ckpt(prefix_u, model, opt, sched, scaler, kind):
    """prefix_u: '<run_dir>/<u>' e.g. results/task1/lora16/625"""
    d, u = os.path.dirname(prefix_u), int(os.path.basename(prefix_u))
    ts = torch.load(os.path.join(d, f"train_state_u{u}.pt"), weights_only=False)
    m = torch.load(os.path.join(d, f"model_u{u}.pt"))
    if kind == "full": model.load_state_dict(m)
    else: L.load_raw_adapter_state(model, m)
    opt.load_state_dict(ts["opt"]); sched.load_state_dict(ts["sched"]); scaler.load_state_dict(ts["scaler"])
    torch.set_rng_state(ts["rng_cpu"])
    return ts["update"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml"); ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--resume", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (yaml-parsed)")
    a = ap.parse_args()
    ov = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    cfg = resolve_cfg(a.config, a.method, {**ov, "out": a.out, "resume": a.resume})
    tok = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    train(cfg, tokenizer=tok)


if __name__ == "__main__":
    main()
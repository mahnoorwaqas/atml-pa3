"""Task 1 tables and figures: loss curves, benchmark table (+pp deltas), resources, diagnostics summary."""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import read_jsonl

RUNS = ["full", "l2sp", "lora4", "lora16", "oplora"]
BENCH = ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]


def loss_plot(runs_dir, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for r in RUNS:
        p = os.path.join(runs_dir, r, "train_log.jsonl")
        if not os.path.exists(p): continue
        df = pd.DataFrame(read_jsonl(p))
        ax[0].plot(df["update"], df["sft_loss"].rolling(10, min_periods=1).mean(), label=r)
        if r == "l2sp" and "total_loss" in df:
            ax[0].plot(df["update"], df["total_loss"].rolling(10, min_periods=1).mean(), "--", label="l2sp total objective")
        ax[1].plot(df["update"], df["lr"], label=r)
    ax[0].set_xlabel("optimizer update"); ax[0].set_ylabel("SFT loss (10-update mean)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("optimizer update"); ax[1].set_ylabel("learning rate"); ax[1].set_yscale("log")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_loss.png"), dpi=150); plt.close()


def benchmark_table(eval_dir, out):
    t = json.load(open(os.path.join(eval_dir, "benchmark_table.json")))
    rows = []
    for c, r in t.items():
        row = {"condition": c}
        for b in BENCH:
            if b in r:
                row[b] = 100 * r[b]["primary"]; row[b + "_dpp"] = r.get("delta_pp", {}).get(b)
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "benchmark_table.csv"), index=False)
    return df


def resources_table(runs_dir, out):
    rows = []
    for r in RUNS:
        p = os.path.join(runs_dir, r, "summary.json")
        if not os.path.exists(p): continue
        s = json.load(open(p))
        cm = read_jsonl(os.path.join(runs_dir, r, "ckpt_meta.jsonl"))
        rows.append({"run": r, "trainable_params": s["trainable_params"], "train_time_s": s["train_time_s"],
                     "peak_mem_gb": s["peak_mem_gb"], "skipped_updates": len(s["skipped_updates"]),
                     "final_ckpt_bytes": sum(cm[-1]["sizes_bytes"].values()),
                     "final_displacement": cm[-1].get("displacement_l2"), **{f"overhead_{k}": v for k, v in s.get("overhead", {}).items()}})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "resources.csv"), index=False); return df


def diagnostics_table(diag_dir, out):
    d = json.load(open(os.path.join(diag_dir, "diagnostics.json")))["results"]
    rows = [{"cond_layer": k, **{m: v[m] for m in ("rho_k", "eta", "R_math_mean", "R_non_math_mean", "R_ratio_math_over_nonmath")}}
            for k, v in d.items()]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "diagnostics_summary.csv"), index=False); return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs_dir", default="results/task1"); ap.add_argument("--out", default="results/task1/report")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    loss_plot(a.runs_dir, a.out); resources_table(a.runs_dir, a.out)
    if os.path.exists(os.path.join(a.runs_dir, "eval", "benchmark_table.json")): benchmark_table(os.path.join(a.runs_dir, "eval"), a.out)
    if os.path.exists(os.path.join(a.runs_dir, "diagnostics", "diagnostics.json")): diagnostics_table(os.path.join(a.runs_dir, "diagnostics"), a.out)

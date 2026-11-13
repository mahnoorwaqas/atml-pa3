#!/usr/bin/env python3
"""
Builds compact figures (PDF) and booktabs LaTeX tables (.tex) from the
pa3-task1 results folder.

    python make_report.py --root pa3-task1-results-main --out report_out

Outputs
    report_out/figs/fig_curves.pdf   training curves (loss, LR, grad norm, L2-SP penalty)
    report_out/figs/fig_bench.pdf    benchmark change vs baseline with paired 95% CIs
    report_out/figs/fig_diag.pdf     weight-space and activation diagnostics
    report_out/tables/tab_*.tex      one table per file, \\input{} them into the paper

LaTeX packages needed: booktabs, graphicx.
Data used: train_log.jsonl, summary.json, ckpt_meta.jsonl, eval/<cond>/*_samples.jsonl,
eval/<cond>/metrics.json, diagnostics/diagnostics.json, qualitative/*.
"""
import argparse, csv, json, os, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Names, colours, markers
# ----------------------------------------------------------------------------
CONDS = ["baseline", "full_mid", "full_final", "l2sp_final",
         "lora4_final", "lora16_final", "oplora_final"]
TRAINED = CONDS[1:]
LABEL = {"baseline": "Baseline", "full_mid": "Full FT (½ ep.)", "full_final": "Full FT",
         "l2sp_final": "Full FT + L2-SP", "lora4_final": "LoRA-4",
         "lora16_final": "LoRA-16", "oplora_final": "OPLoRA-16"}
TEX_LABEL = dict(LABEL, full_mid=r"Full FT (\textonehalf{} ep.)")
RUN_OF = {"full_final": "full", "l2sp_final": "l2sp", "lora4_final": "lora4",
          "lora16_final": "lora16", "oplora_final": "oplora"}
COLOR = {"baseline": "#777777", "full_mid": "#9DB7E8", "full_final": "#3B6FD4",
         "l2sp_final": "#1B9E8A", "lora4_final": "#F2A65A",
         "lora16_final": "#E0693A", "oplora_final": "#8C1C2E"}
MARK = {"full_mid": "o", "full_final": "o", "l2sp_final": "s",
        "lora4_final": "^", "lora16_final": "v", "oplora_final": "D"}
TASKS = [("gsm8k", "GSM8K"), ("hellaswag", "HellaSwag"), ("boolq", "BoolQ"), ("arc_easy", "ARC-E")]
TASK_METRIC = {"gsm8k": "exact_match_flexible", "hellaswag": "acc_norm",
               "boolq": "acc", "arc_easy": "acc_norm"}
LAYERS = [0, 13, 27]
K_PROTECT, D_OUT = 128, 1024          # protected rank / down_proj output dim (Qwen3-0.6B hidden size)


def set_style():
    plt.rcParams.update({
        "font.size": 6.5, "axes.titlesize": 7.5, "axes.labelsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.titlepad": 3,
        "lines.linewidth": 1.1, "lines.markersize": 3.2,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def offset_sci(ax, axis="y"):
    ax.ticklabel_format(axis=axis, style="sci", scilimits=(0, 0))
    off = ax.yaxis.get_offset_text() if axis == "y" else ax.xaxis.get_offset_text()
    off.set_fontsize(5.5)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_logs(root):
    return {r: pd.DataFrame(read_jsonl(root / r / "train_log.jsonl")) for r in RUN_OF.values()}


def per_example(root, cond, task):
    """Per-example correctness (float array) for the primary metric of `task`."""
    rows = read_jsonl(root / "eval" / cond / f"{task}_samples.jsonl")
    if task == "gsm8k":
        # lm-eval writes one record per (doc, filter): strict-match first, flexible-extract second
        n = len(rows) // 2
        rows = rows[n:]
        assert [str(r["doc_id"]) for r in rows] == [str(i) for i in range(n)], "unexpected gsm8k order"
    return np.array([float(r[TASK_METRIC[task]]) for r in rows])


def load_eval(root):
    E = {c: {t: per_example(root, c, t) for t, _ in TASKS} for c in CONDS}
    M = {c: json.load(open(root / "eval" / c / "metrics.json")) for c in CONDS}
    return E, M


def paired_ci(a, b):
    """Mean(b-a) in pp with a paired normal 95% CI."""
    d = (b - a) * 100.0
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


def boolq_stats(root, cond):
    rows = read_jsonl(root / "eval" / cond / "boolq_samples.jsonl")
    pred = np.array([int(r["resps"][1][0][0] > r["resps"][0][0][0]) for r in rows])  # choices: [no, yes]
    return pred.mean() * 100


def gsm8k_trunc_and_tokens(root, cond):
    rows = read_jsonl(root / "eval" / cond / "gsm8k_samples.jsonl")
    rows = rows[len(rows) // 2:]
    tr = np.mean([str(r["truncated"]) == "True" for r in rows]) * 100
    tok = np.mean([float(r["n_tokens"]) for r in rows])
    return tr, tok


# ----------------------------------------------------------------------------
# LaTeX helpers
# ----------------------------------------------------------------------------
def sgn(v, d=1):
    if round(v, d) == 0:                      # avoid "-0.0" / "+0.0"
        return f"{0:.{d}f}"
    return ("$+$" if v > 0 else "$-$") + f"{abs(v):.{d}f}"


def bold(s):
    return r"\textbf{" + s + "}"


def table(caption, label, colspec, header_rows, body, note=None, size="small", tabcolsep=4.0):
    L = [r"\begin{table}[t]", r"\centering", "\\" + size,
         rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}",
         r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    L += header_rows
    for r in body:
        L.append(r)
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L += [r"\\[2pt]", r"\parbox{\linewidth}{\scriptsize " + note + "}"]
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def legend_handles(conds):
    return [Line2D([0], [0], color=COLOR[c], marker=MARK[c], lw=1.1, ms=3.2, label=LABEL[c])
            for c in conds]


def fig_curves(root, logs, out, width):
    fig, axs = plt.subplots(1, 4, figsize=(width, 1.75))
    ck = [311, 624, 936, 1249]
    order = ["full_final", "l2sp_final", "lora4_final", "lora16_final", "oplora_final"]
    for c in order:
        df = logs[RUN_OF[c]]
        kw = dict(color=COLOR[c], marker=MARK[c], markevery=ck, label=LABEL[c])
        axs[0].plot(df["update"], df["sft_loss"].rolling(25, min_periods=1).mean(), **kw)
        axs[2].plot(df["update"], df["grad_norm"].rolling(25, min_periods=1).median(), **kw)
    # LR schedule: identical shape for all runs, so plot lr / peak lr once
    df_full, df_lora = logs["full"], logs["lora16"]
    axs[1].plot(df_lora["update"], df_lora["lr"] / df_lora["lr"].max(), color="#555555",
                marker="o", markevery=ck)
    axs[1].set_ylim(-0.03, 1.08)
    axs[1].text(1240, 0.97, "peak 2e-5 (full FT)\n      2e-4 (adapters)", ha="right", va="top", fontsize=5.5, color="#444444")
    # SFT-loss panel zoomed: the first ~50 updates (loss 1.0 -> 0.25) are off-scale
    axs[0].set_ylim(0.14, 0.42)
    # L2-SP penalty
    l2 = logs["l2sp"]
    axs[3].plot(l2["update"], l2["lam_omega"], color=COLOR["l2sp_final"], marker="s", markevery=ck)
    offset_sci(axs[3])
    axs[0].set_title("SFT loss"); axs[1].set_title("LR / peak LR")
    axs[2].set_title("grad norm"); axs[3].set_title(r"L2-SP penalty $\lambda\Omega$")
    axs[2].set_yscale("log")
    for a in axs:
        a.set_xlabel("update"); a.set_xticks([0, 625, 1250])
    fig.legend(handles=legend_handles(order), loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.03), handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.22, top=0.74, wspace=0.34)
    fig.savefig(out / "fig_curves.pdf"); fig.savefig(out / "fig_curves.png", dpi=200); plt.close(fig)


def fig_bench(E, out, width):
    fig, axs = plt.subplots(1, 4, figsize=(width, 1.55), sharey=True)
    y = np.arange(len(TRAINED))[::-1]
    for ax, (t, name) in zip(axs, TASKS):
        for yi, c in zip(y, TRAINED):
            m, lo, hi = paired_ci(E["baseline"][t], E[c][t])
            ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt=MARK[c], color=COLOR[c],
                        ms=3.6, elinewidth=0.9, capsize=1.5, capthick=0.8)
        ax.axvline(0, color="#444444", lw=0.8, ls=(0, (3, 2)))
        ax.set_title(name); ax.set_xlabel(r"$\Delta$ vs. baseline (pp)")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels([LABEL[c] for c in TRAINED])
    fig.subplots_adjust(left=0.135, right=0.995, bottom=0.25, top=0.88, wspace=0.12)
    fig.savefig(out / "fig_bench.pdf"); fig.savefig(out / "fig_bench.png", dpi=200); plt.close(fig)


def fig_diag(D, out, width):
    fig, axs = plt.subplots(1, 5, figsize=(width, 1.8))
    series = ["full_mid", "full_final", "l2sp_final", "lora4_final", "lora16_final", "oplora_final"]
    keys = ["rho_k", "eta", "R_math_mean", "R_non_math_mean", "R_ratio_math_over_nonmath"]
    titles = [r"$\rho_k$ (top-$k$ share)", r"$\eta=\|\Delta W\|_F/\|W_0\|_F$",
              r"$R$ on math", r"$R$ on non-math", r"$R_{\mathrm{math}}/R_{\mathrm{non\text{-}math}}$"]
    for ax, k, ttl in zip(axs, keys, titles):
        for c in series:
            v = [D[f"{c}|layer{l}"][k] for l in LAYERS]
            ax.plot(range(3), v, color=COLOR[c], marker=MARK[c],
                    ls="--" if c == "full_mid" else "-", lw=1.0)
        ax.set_xticks(range(3)); ax.set_xticklabels([str(l) for l in LAYERS])
        ax.set_xlabel("layer"); ax.set_title(ttl, fontsize=6.8)
        ax.set_xlim(-0.2, 2.2)
        if k in ("eta", "R_math_mean", "R_non_math_mean"):
            offset_sci(ax)
    axs[0].axhline(K_PROTECT / D_OUT, color="#444444", lw=0.7, ls=(0, (3, 2)))
    axs[0].text(2.2, K_PROTECT / D_OUT + 0.003, "chance", ha="right", va="bottom", fontsize=5.5, color="#444444")
    axs[0].set_ylim(-0.005, 0.18)
    axs[4].axhline(1.0, color="#444444", lw=0.7, ls=(0, (3, 2)))
    fig.legend(handles=legend_handles(series), loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.03), handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.22, top=0.72, wspace=0.42)
    fig.savefig(out / "fig_diag.pdf"); fig.savefig(out / "fig_diag.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def tab_bench(E):
    base = {t: E["baseline"][t].mean() * 100 for t, _ in TASKS}
    acc = {c: {t: E[c][t].mean() * 100 for t, _ in TASKS} for c in CONDS}
    best = {t: max(acc[c][t] for c in TRAINED) for t, _ in TASKS}
    hdr1 = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for _, n in TASKS) + r" \\"
    cm = "".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(4))
    hdr2 = "Model & " + " & ".join(r"Acc & $\Delta$" for _ in TASKS) + r" \\"
    body = []
    for c in CONDS:
        cells = []
        for t, _ in TASKS:
            a = acc[c][t]; s = f"{a:.1f}"
            if c != "baseline" and abs(a - best[t]) < 1e-9:
                s = bold(s)
            cells += [s, "--" if c == "baseline" else sgn(a - base[t])]
        body.append(TEX_LABEL[c] + " & " + " & ".join(cells) + r" \\")
        if c in ("baseline", "full_mid", "l2sp_final"):
            body.append(r"\midrule" if c == "baseline" else r"\addlinespace[1pt]")
    cap = (r"Zero-shot results (\%) and change from the shared baseline ($\Delta$, percentage points). "
           r"GSM8K: flexible-extract exact match; HellaSwag and ARC-E: normalized accuracy; BoolQ: accuracy. "
           r"Bold: best trained model per benchmark.")
    return table(cap, "tab:bench", "l" + "cc" * 4, [hdr1, cm, hdr2, r"\midrule"], body)


def tab_compare(E):
    pairs = [("LoRA-4 $\\to$ LoRA-16 (rank)", "lora4_final", "lora16_final"),
             ("Full FT $\\to$ + L2-SP (regularization)", "full_final", "l2sp_final"),
             ("LoRA-16 $\\to$ OPLoRA-16 (projection)", "lora16_final", "oplora_final"),
             ("Full FT $\\to$ LoRA-16 (method)", "full_final", "lora16_final"),
             ("Full FT \\textonehalf{} ep.\\ $\\to$ 1 ep.\\ (progression)", "full_mid", "full_final")]
    hdr1 = r" & \multicolumn{4}{c}{$\Delta$ (pp), B $-$ A, paired 95\% CI} \\"
    cm = r"\cmidrule(lr){2-5}"
    hdr2 = "A $\\to$ B & " + " & ".join(n for _, n in TASKS) + r" \\"
    body = []
    for name, a, b in pairs:
        cells = []
        for t, _ in TASKS:
            m, lo, hi = paired_ci(E[a][t], E[b][t])
            s = f"{sgn(m)} [{sgn(lo)}, {sgn(hi)}]"
            cells.append(bold(s) if (lo > 0 or hi < 0) else s)
        body.append(name + " & " + " & ".join(cells) + r" \\")
    cap = (r"Controlled comparisons at the final checkpoint. Paired normal-approximation 95\% intervals over "
           r"per-example outcomes; bold means the interval excludes 0 (before rounding).")
    return table(cap, "tab:compare", "lcccc", [hdr1, cm, hdr2, r"\midrule"], body, size="footnotesize", tabcolsep=3.0)


def tab_extra(root, E, M):
    hdr = (r"Model & HS acc & ARC-E acc & GSM8K trunc.\ (\%) & GSM8K tokens & BoolQ ``yes'' (\%) & "
           r"merge $|\Delta\,\mathrm{logit}|$ \\")
    body = []
    for c in CONDS:
        tr, tok = gsm8k_trunc_and_tokens(root, c)
        yes = boolq_stats(root, c)
        hs = M[c]["hellaswag"]["acc,none"] * 100
        ae = M[c]["arc_easy"]["acc,none"] * 100
        mg = M[c].get("merge_check_max_abs_logit_diff")
        mg_s = "--" if mg is None else rf"${mg*1e5:.1f}\times10^{{-5}}$"
        body.append(TEX_LABEL[c] + f" & {hs:.1f} & {ae:.1f} & {tr:.1f} & {tok:.0f} & {yes:.1f} & {mg_s} \\\\")
        if c == "baseline":
            body.append(r"\midrule")
    cap = (r"Supporting evaluation diagnostics. HS/ARC-E acc: raw (unnormalized) accuracy. GSM8K trunc.: "
           r"responses hitting the 1{,}024-token cap. BoolQ ``yes'': share of predicted ``yes'' (gold: 62.2\%). "
           r"Merge check: max absolute logit difference, merged vs.\ unmerged adapter (adapter runs only).")
    return table(cap, "tab:extra", "lcccccc", [hdr, r"\midrule"], body,
                 size="footnotesize", tabcolsep=3.0)


def tab_resources(root, logs):
    res = pd.read_csv(root / "report" / "resources.csv").set_index("run")
    meta = {r: read_jsonl(root / r / "ckpt_meta.jsonl") for r in RUN_OF.values()}
    hdr = (r"Run & Trainable (M) & Time (h) & Peak mem (GB) & Ckpt (GB) & "
           r"$\|\theta-\theta_0\|_2$ (\textonehalf{} / 1 ep.) & Method overhead \\")
    over = {"full": "--",
            "l2sp": r"$\theta_0$ copy 2.4\,GB (pinned CPU)",
            "lora4": "--", "lora16": "--",
            "oplora": r"bases 0.32\,GB; SVD 33\,s; export 0.5\,s"}
    body = []
    for c in ["full_final", "l2sp_final", "lora4_final", "lora16_final", "oplora_final"]:
        r = RUN_OF[c]; row = res.loc[r]
        disp = "--"
        if r in ("full", "l2sp"):
            disp = f"{meta[r][1]['displacement_l2']:.2f} / {row['final_displacement']:.2f}"
        body.append(TEX_LABEL[c] + f" & {row['trainable_params']/1e6:.2f} & {row['train_time_s']/3600:.2f} & "
                    f"{row['peak_mem_gb']:.1f} & {row['final_ckpt_bytes']/1e9:.2f} & {disp} & {over[r]} \\\\")
    cap = (r"Resource use over 1{,}250 updates. Ckpt: all files of the final recovery checkpoint "
           r"(weights/adapters plus optimizer state). No updates were skipped by the gradient scaler in any run.")
    return table(cap, "tab:resources", "lcccccl", [hdr, r"\midrule"], body, size="footnotesize", tabcolsep=3.0)


def tab_diag(D):
    ser = ["full_mid", "full_final", "l2sp_final", "lora4_final", "lora16_final", "oplora_final"]
    hdr1 = " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{Layer {l}}}" for l in LAYERS) + r" \\"
    cm = "".join(rf"\cmidrule(lr){{{2+4*i}-{5+4*i}}}" for i in range(3))
    hdr2 = "Model & " + " & ".join(r"$\rho_k$ & $\eta$ & $R_\mathrm{m}$ & $R_\mathrm{n}$" for _ in LAYERS) + r" \\"
    body = []
    for c in ser:
        cells = []
        for l in LAYERS:
            d = D[f"{c}|layer{l}"]
            rho = r"$\approx 0$" if d["rho_k"] < 1e-3 else f"{d['rho_k']:.3f}"
            cells += [rho, f"{d['eta']*100:.2f}", f"{d['R_math_mean']*100:.2f}", f"{d['R_non_math_mean']*100:.2f}"]
        body.append(TEX_LABEL[c] + " & " + " & ".join(cells) + r" \\")
    omax = max(D[f"oplora_final|layer{l}"]["rho_k"] for l in LAYERS)
    cap = (r"Weight-space and activation diagnostics for \texttt{down\_proj} ($k=128$). $\rho_k$: share of $\|\Delta W\|_F^2$ "
           r"in the top-$k$ output subspace (random direction: $k/1024=0.125$). $\eta=\|\Delta W\|_F/\|W_0\|_F$ ($\times10^{-2}$). "
           r"$R_\mathrm{m}$, $R_\mathrm{n}$: $\|\Delta W z\|/\|W_0 z\|$ ($\times10^{-2}$) on math / non-math probe prompts.")
    note = rf"OPLoRA: $\rho_k \leq {omax*1e8:.1f}\times10^{{-8}}$ in all three layers (numerical zero)."
    return table(cap, "tab:diag", "l" + "cccc" * 3, [hdr1, cm, hdr2, r"\midrule"], body,
                 note=note, size="footnotesize", tabcolsep=2.6)


def qual_stats(root):
    q = json.load(open(root / "qualitative" / "responses_all.json"))
    sub = list(csv.DictReader(open(root / "qualitative" / "annotation_subset_42.csv", encoding="utf-8")))
    math_re = re.compile(r"the answer is|\\boxed|####|\$\$", re.I)
    S = {c: dict(n=0, trunc=0, math=0, js=0, jn=0, tok=[]) for c in CONDS}
    for qn, fm in q.items():
        for f, cd in fm.items():
            for c, v in cd.items():
                s = S[c]; s["n"] += 1; s["trunc"] += bool(v["truncated"]); s["tok"].append(v["n_tokens"])
                s["math"] += bool(math_re.search(v["response"]))
                if f == "json":
                    s["jn"] += 1
                    txt = re.sub(r"^```(?:json)?|```$", "", v["response"].strip()).strip()
                    try:
                        json.loads(txt); s["js"] += 1
                    except Exception:
                        pass
    rep = {c: np.mean([float(r["repetition_4gram"]) for r in sub if r["condition"] == c]) for c in CONDS}
    return S, rep


def tab_qual(root):
    S, rep = qual_stats(root)
    hdr = r"Model & Truncated (/24) & Math framing (/24) & Valid JSON (/8) & Mean tokens & 4-gram rep.\ (42-subset) \\"
    body = []
    for c in CONDS:
        s = S[c]
        body.append(TEX_LABEL[c] + f" & {s['trunc']} & {s['math']} & {s['js']} & {np.mean(s['tok']):.0f} & {rep[c]:.2f} \\\\")
        if c == "baseline":
            body.append(r"\midrule")
    cap = (r"Qualitative prompts (24 = 8 questions $\times$ plain/JSON/template; greedy, 512-token cap). "
           r"Math framing: response contains ``the answer is'', \texttt{\textbackslash boxed}, \texttt{\#\#\#\#} or \texttt{\$\$}. "
           r"Valid JSON: parses after removing code fences. Repetition: mean fraction of repeated 4-grams on the 42 annotated responses. "
           r"All flags are automatic, not manual annotations.")
    return table(cap, "tab:qual", "lccccc", [hdr, r"\midrule"], body, size="footnotesize", tabcolsep=3.0)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="pa3-task1-results-main")
    ap.add_argument("--out", default="report_out")
    ap.add_argument("--width", type=float, default=7.0, help="figure width in inches (\\textwidth)")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    set_style()

    logs = load_logs(root)
    E, M = load_eval(root)
    D = json.load(open(root / "diagnostics" / "diagnostics.json"))["results"]

    fig_curves(root, logs, out / "figs", a.width)
    fig_bench(E, out / "figs", a.width)
    fig_diag(D, out / "figs", a.width)

    write(out / "tables" / "tab_bench.tex", tab_bench(E))
    write(out / "tables" / "tab_compare.tex", tab_compare(E))
    write(out / "tables" / "tab_extra.tex", tab_extra(root, E, M))
    write(out / "tables" / "tab_resources.tex", tab_resources(root, logs))
    write(out / "tables" / "tab_diag.tex", tab_diag(D))
    write(out / "tables" / "tab_qual.tex", tab_qual(root))
    print("wrote figures to", out / "figs", "and tables to", out / "tables")


if __name__ == "__main__":
    main()
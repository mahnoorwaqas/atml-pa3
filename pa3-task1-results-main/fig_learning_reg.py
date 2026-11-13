"""Usage: python fig_learning_reg.py <results_dir> [out.pdf]   (results_dir has full/ l2sp/ lora4/ lora16/ oplora/)"""
import json, sys, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.ticker as mt
R = sys.argv[1] if len(sys.argv) > 1 else "."; OUT = sys.argv[2] if len(sys.argv) > 2 else "fig_learning_reg.pdf"
mpl.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
    "legend.fontsize": 5.5, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444",
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": .6, "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42})
C = {"full": "#3B6FD4", "l2sp": "#8E6BC9", "lora4": "#F2A541", "lora16": "#E0703A", "oplora": "#2E9E6B"}
N = {"full": "Full FT", "l2sp": "Full + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora": "OPLoRA-16"}
log = {r: pd.DataFrame([json.loads(l) for l in open(f"{R}/{r}/train_log.jsonl")]) for r in C}
sm = lambda x, w=25: pd.Series(x).rolling(w, min_periods=1, center=True).mean()
ttl = lambda ax, s: ax.set_title(s, loc="left", fontsize=6.8)
fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.6), gridspec_kw={"wspace": .42, "width_ratios": [1.25, 1, 1]})
# (a) combined SFT loss
for r in C:
    ax[0].plot(log[r]["update"], sm(log[r].sft_loss), color=C[r], lw={"full": 2.4, "l2sp": 1.1}.get(r, 1.3), ls="--" if r == "l2sp" else "-",
               zorder=3 if r == "l2sp" else 2, label=N[r])
ax[0].set_yscale("log"); ax[0].yaxis.set_minor_formatter(mt.NullFormatter()); ax[0].set_yticks([.15, .2, .3, .5, 1]); ax[0].set_yticklabels(["0.15", "0.2", "0.3", "0.5", "1.0"])
ax[0].set(xlim=(0, 1250), ylim=(.13, 1.0), xlabel="optimizer update", ylabel="SFT loss (25-update mean)"); ax[0].legend(); ttl(ax[0], "(a) SFT loss, all runs")
# (b) L2-SP penalty strength
l2 = log["l2sp"]; u = l2["update"]
ax[1].plot(u, sm(l2.sft_loss), color=C["l2sp"], lw=1.3, label="SFT loss")
ax[1].plot(u, sm(l2.lam_omega), color="#555", lw=1.3, label=r"$\lambda\Omega$ (penalty)")
ax[1].plot(u, sm(l2.total_loss), color="#999", lw=1, ls="--", label="total")
ax[1].set(xlabel="optimizer update", ylim=(0, .5), xlim=(0, 1250)); ax[1].legend(loc="upper right")
f = l2.iloc[-1]; ax[1].annotate(f"final $\\lambda\\Omega$={f.lam_omega:.3f}\n({100*f.lam_omega/f.total_loss:.0f}% of total)", (1250, f.lam_omega), xytext=(560, .08),
    textcoords="data", ha="left", fontsize=5.5, arrowprops=dict(arrowstyle="-", color="#777", lw=.6)); ttl(ax[1], r"(b) L2-SP penalty ($\lambda=10^{-3}$)")
# (c) displacement
disp = {}
for r in ["full", "l2sp"]:
    cm = [json.loads(l) for l in open(f"{R}/{r}/ckpt_meta.jsonl")]; disp[r] = cm
    ax[2].plot([x["update"] for x in cm], [x["displacement_l2"] for x in cm], "o-", color=C[r], ms=3, lw=1.4, label=N[r])
a, b = disp["full"][-1]["displacement_l2"], disp["l2sp"][-1]["displacement_l2"]
ax[2].annotate(f"{a:.2f}", (1250, a), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=5.8, color=C["full"])
ax[2].annotate(f"{b:.2f}  ({100*(b/a-1):+.0f}%)", (1250, b), xytext=(0, -11), textcoords="offset points", ha="right", fontsize=5.8, color=C["l2sp"])
ax[2].set(xlabel="optimizer update", ylabel=r"$\|\theta-\theta_0\|_2$", xlim=(200, 1320), ylim=(0, 12.5)); ax[2].legend(loc="lower right"); ttl(ax[2], "(c) Parameter displacement")
fig.savefig(OUT); print("saved", OUT)

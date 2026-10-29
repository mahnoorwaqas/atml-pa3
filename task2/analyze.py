"""Task 2 figures: training loss + development curves with stage boundaries."""
import argparse, json, os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.utils import read_jsonl


def plots(run, out=None):
    out = out or run
    tr = pd.DataFrame(read_jsonl(os.path.join(run, "train_log.jsonl"))); dv = pd.DataFrame(read_jsonl(os.path.join(run, "dev_log.jsonl")))
    bounds = [json.load(open(os.path.join(run, "summary.json")))["cfg"]["stages"]]
    ups = [s["updates"] for s in bounds[0]]; edges = [sum(ups[:k]) for k in range(1, len(ups))]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
    ax[0].plot(tr["update"], tr["loss"].rolling(25, min_periods=1).mean()); ax[0].set_title("train loss")
    ax[1].plot(dv["update"], dv["caption_loss"], "o-"); ax[1].set_title("dev caption loss")
    for n in ("grounding", "natural"):
        ax[2].plot(dv["update"], dv[f"{n}_nll_correct"], "o-", label=n + " correct"); ax[2].plot(dv["update"], dv[f"{n}_nll_none"], "x--", label=n + " no-image")
        ax[3].plot(dv["update"], dv[f"{n}_delta_nll"], "o-", label=n)
    ax[2].set_title("dev answer NLL"); ax[3].set_title("image-swap penalty ΔNLL"); ax[2].legend(fontsize=6); ax[3].legend(fontsize=6)
    for a in ax:
        a.set_xlabel("update")
        for e in edges: a.axvline(e, color="gray", ls=":")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_curves.png"), dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); plots(ap.parse_args().run)

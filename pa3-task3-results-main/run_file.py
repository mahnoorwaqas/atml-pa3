import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSVs directly from the extracted local folders
math = pd.read_csv("math/methods_eval.csv")
vista = pd.read_csv("mathvista/methods_eval.csv")

# Set up a 1-row, 5-column ultra-compact figure with extra space at the top for a shared legend
fig, ax = plt.subplots(1, 5, figsize=(14, 2.6))

methods = {
    "math": ["first", "sc", "logp_best", "selfcert_best", "selfcert_rwc"],
    "vista": ["first", "sc", "logp_best", "selfcert_best",
              "selfcert_rwc", "visconf_rwc", "token_rwc"]
}

labels = {
    "first": "First sample",
    "sc": "Self-consistency",
    "logp_best": "Log-prob",
    "selfcert_best": "Self-cert",
    "selfcert_rwc": "Self-cert+RWC",
    "visconf_rwc": "VisConf+RWC",
    "token_rwc": "Token+RWC"
}

# Keep track of handles and labels for the shared legend
lines_handles, lines_labels = [], []

# 1. MATH Accuracy
for method in methods["math"]:
    d = math[math.method == method].sort_values("N")
    line, = ax[0].plot(d["N"], d["acc"], marker="o", label=labels[method], markersize=2, linewidth=1)
    if not lines_handles:  # collect unique handles from math methods
        lines_handles.append(line)
        lines_labels.append(labels[method])
ax[0].set_title("(a) MATH Acc", fontsize=7)
ax[0].set_ylabel("Rate", fontsize=7)
ax[0].set_xticks([1, 16, 32])
ax[0].tick_params(axis='both', labelsize=6)
ax[0].grid(alpha=0.25)

# 2. MATH Oracle Coverage
for method in methods["math"]:
    d = math[math.method == method].sort_values("N")
    ax[1].plot(d["N"], d["coverage"], marker="s", linestyle="--", label=labels[method], markersize=2, linewidth=1)
ax[1].set_title("(b) MATH Cov", fontsize=7)
ax[1].set_xticks([1, 16, 32])
ax[1].tick_params(axis='both', labelsize=6)
ax[1].grid(alpha=0.25)

# 3. MathVista Accuracy
for method in methods["vista"]:
    d = vista[vista.method == method].sort_values("N")
    line, = ax[2].plot(d["N"], d["acc"], marker="o", label=labels[method], markersize=2, linewidth=1)
    # Add Vista-specific methods to legend handles if not already present
    if labels[method] not in lines_labels:
        lines_handles.append(line)
        lines_labels.append(labels[method])
ax[2].set_title("(c) Vista Acc", fontsize=7)
ax[2].set_xticks([1, 16, 32])
ax[2].tick_params(axis='both', labelsize=6)
ax[2].grid(alpha=0.25)

# 4. MathVista Oracle Coverage
for method in methods["vista"]:
    d = vista[vista.method == method].sort_values("N")
    ax[3].plot(d["N"], d["coverage"], marker="s", linestyle="--", label=labels[method], markersize=2, linewidth=1)
ax[3].set_title("(d) Vista Cov", fontsize=7)
ax[3].set_xticks([1, 16, 32])
ax[3].tick_params(axis='both', labelsize=6)
ax[3].grid(alpha=0.25)

# 5. Failure Decomposition at N=32
# 5. Failure Decomposition at N=32
# 5. Failure Decomposition at N=32
colors = {
    "Correct": "tab:green",
    "Selection failure": "tab:orange",
    "Generation failure": "tab:red"
}

for x, (name, data) in enumerate([("MATH", math), ("Vista", vista)]):
    d = data[(data.method == "sc") & (data.N == 32)].iloc[0]

    values = [
        d.acc,
        max(0, d.coverage - d.acc),
        max(0, 1 - d.coverage)
    ]

    bottom = 0
    for label, value in zip(colors.keys(), values):
        ax[4].bar(
            x, value,
            bottom=bottom,
            width=0.55,
            color=colors[label],
            label=label if x == 0 else None
        )
        bottom += value

ax[4].set_xticks([0, 1])
ax[4].set_xticklabels(["MATH", "Vista"])
ax[4].set_title("(e) Failure decomposition (N=32)", fontsize=7)
ax[4].set_ylim(0, 1)
ax[4].set_ylabel("Fraction of questions", fontsize=7)
ax[4].tick_params(axis="both", labelsize=6)
ax[4].grid(axis="y", alpha=0.25)
ax[4].legend(
    fontsize=5.5,
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="gray",
    framealpha=0.55
)
# Shared legend across the top
fig.legend(
    lines_handles, lines_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    fontsize=7,
    frameon=False
)

# Explicitly leave room for the legend
fig.subplots_adjust(top=0.72, wspace=0.35)

plt.savefig("all_five_plots_tiny.pdf", bbox_inches="tight")
plt.show()
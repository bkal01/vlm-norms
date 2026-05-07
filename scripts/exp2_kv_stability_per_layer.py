import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

RUNS = {
    "SmolVLM2-2.2B": "kv_stability_under_alpha_runs/kv_stability_under_alpha_9d41577e/kv_stability_layer_summary.csv",
    "Qwen3-VL-2B": "kv_stability_under_alpha_runs/kv_stability_under_alpha_1974eca6/kv_stability_layer_summary.csv",
}

ALPHAS = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 1.0, 3.0]
BASELINE_ALPHA = 1.0
ORANGE = "#E87722"
GRAY   = "#888888"

# Dark→light blue for downscaled alphas (0.01 darkest, 0.3 lightest)
downscale_alphas = [a for a in ALPHAS if a < BASELINE_ALPHA]
blues = cm.Blues(np.linspace(0.85, 0.35, len(downscale_alphas)))
ALPHA_COLOR = {a: blues[i] for i, a in enumerate(downscale_alphas)}
ALPHA_COLOR[BASELINE_ALPHA] = GRAY
ALPHA_COLOR[3.0] = ORANGE

# Legend top→bottom matches line top→bottom in the V panels
LEGEND_ORDER = [1.0, 3.0, 0.3, 0.1, 0.07, 0.05, 0.03, 0.01]

METRICS = [
    ("mean_k_cos_to_baseline", "K cosine to baseline (α=1)"),
    ("mean_v_cos_to_baseline", "V cosine to baseline (α=1)"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)

for row_idx, (model_name, csv_path) in enumerate(RUNS.items()):
    df = pd.read_csv(csv_path)
    df = df[df["layer"] >= 1]

    for col_idx, (col, ylabel) in enumerate(METRICS):
        ax = axes[row_idx][col_idx]

        for alpha in ALPHAS:
            sub = df[df["alpha"] == alpha].sort_values("layer")
            color = ALPHA_COLOR[alpha]
            ls = "--" if alpha == BASELINE_ALPHA else "-"
            ax.plot(sub["layer"].values, sub[col].values,
                    color=color, linewidth=1.4, linestyle=ls)

        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        kv = "K" if col_idx == 0 else "V"
        ax.set_title(f"{kv} cosine to baseline — {model_name}")
        ax.set_ylim(-0.05, 1.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

# Shared legend below all panels, ordered to match line ordering
handles, labels = [], []
for alpha in LEGEND_ORDER:
    color = ALPHA_COLOR[alpha]
    ls = "--" if alpha == BASELINE_ALPHA else "-"
    suffix = " (baseline)" if alpha == BASELINE_ALPHA else ""
    handles.append(mlines.Line2D([], [], color=color, linewidth=1.4, linestyle=ls))
    labels.append(f"α = {alpha}{suffix}")

fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0),
           ncol=4, frameon=False, handlelength=2.0, columnspacing=1.4)

fig.tight_layout(pad=1.8)
fig.subplots_adjust(bottom=0.16)
fig.savefig("assets/exp2_kv_stability_per_layer.png", bbox_inches="tight", dpi=200)
print("saved assets/exp2_kv_stability_per_layer.png")

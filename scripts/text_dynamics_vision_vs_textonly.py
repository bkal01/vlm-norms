import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

RUNS = [
    ("runs/e3aa20d5-5a1b-4750-87d2-443c912b8a5f", "SmolVLM2-2.2B"),
    ("runs/8b787cbb-4ecd-4c77-b788-a8dcc3525fbc", "Qwen3-VL-2B"),
]

SUBSETS = ["chart", "counting", "document", "general", "grounding", "scene", "spatial", "table"]

REAL_COLOR    = "#4878CF"
BLANK_COLOR   = "#6ABF69"
TEXTONLY_COLOR = "#E87722"


def load_metrics(run_dir):
    run_dir = pathlib.Path(run_dir)
    real_norms, real_cos = [], []
    blank_norms, blank_cos = [], []
    text_norms, text_cos = [], []

    for subset in SUBSETS:
        for sample_dir in sorted((run_dir / subset).iterdir()):
            real = torch.load(sample_dir / "alpha_1" / "metrics.pt", map_location="cpu")
            blank = torch.load(sample_dir / "blank_alpha_1" / "metrics.pt", map_location="cpu")
            textonly = torch.load(sample_dir / "textonly" / "metrics.pt", map_location="cpu")
            real_norms.append(real["text_norms"].float())
            real_cos.append(real["text_cos"].float())
            blank_norms.append(blank["text_norms"].float())
            blank_cos.append(blank["text_cos"].float())
            text_norms.append(textonly["textonly_norms"].float())
            text_cos.append(textonly["textonly_cos"].float())

    real_norms  = torch.stack(real_norms).numpy()
    real_cos    = torch.stack(real_cos).numpy()
    blank_norms = torch.stack(blank_norms).numpy()
    blank_cos   = torch.stack(blank_cos).numpy()
    text_norms  = torch.stack(text_norms).numpy()
    text_cos    = torch.stack(text_cos).numpy()

    def mean_corr(a, b):
        return np.mean([np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a))])

    return {
        "real_norms_mean":  real_norms.mean(0),
        "real_norms_std":   real_norms.std(0),
        "blank_norms_mean": blank_norms.mean(0),
        "blank_norms_std":  blank_norms.std(0),
        "text_norms_mean":  text_norms.mean(0),
        "text_norms_std":   text_norms.std(0),
        "real_cos_mean":    real_cos.mean(0),
        "real_cos_std":     real_cos.std(0),
        "blank_cos_mean":   blank_cos.mean(0),
        "blank_cos_std":    blank_cos.std(0),
        "text_cos_mean":    text_cos.mean(0),
        "text_cos_std":     text_cos.std(0),
        "real_blank_norm_r":   mean_corr(real_norms, blank_norms),
        "real_text_norm_r":    mean_corr(real_norms, text_norms),
        "real_blank_cos_r":    mean_corr(real_cos, blank_cos),
        "real_text_cos_r":     mean_corr(real_cos, text_cos),
        "n_samples": len(real_norms),
    }


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))

stats = {}
for row, (run_dir, label) in enumerate(RUNS):
    d = load_metrics(run_dir)
    stats[label] = d

    conditions = [
        ("real_norms",  "real_cos",  REAL_COLOR,    "real image"),
        ("blank_norms", "blank_cos", BLANK_COLOR,   "blank image"),
        ("text_norms",  "text_cos",  TEXTONLY_COLOR, "text-only"),
    ]

    # --- norms (drop final layer — outlier from final layernorm) ---
    ax = axes[row, 0]
    for norm_key, _, color, lbl in conditions:
        mean = d[f"{norm_key}_mean"][:-1]
        std  = d[f"{norm_key}_std"][:-1]
        layers = range(len(mean))
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.12)
        ax.plot(layers, mean, color=color, linewidth=1.5, label=lbl)
    ax.set_title(f"{label}  —  Hidden-state norms")
    ax.set_ylabel("Hidden-state norm")
    ax.set_xlabel("Layer")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # --- cos similarity to layer 0 (drop final layer) ---
    ax = axes[row, 1]
    for _, cos_key, color, lbl in conditions:
        mean = d[f"{cos_key}_mean"][:-1]
        std  = d[f"{cos_key}_std"][:-1]
        layers = range(len(mean))
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.12)
        ax.plot(layers, mean, color=color, linewidth=1.5, label=lbl)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.set_title(f"{label}  —  Cosine similarity to layer 0")
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Layer")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

fig.tight_layout(pad=1.5)
fig.savefig("assets/text_dynamics_vision_vs_textonly.png", bbox_inches="tight", dpi=200)
print("saved assets/text_dynamics_vision_vs_textonly.png")

print()
header = f"{'Model':<20} {'n':>4}  {'real-blank norm r':>17}  {'real-text norm r':>16}  {'real-blank cos r':>16}  {'real-text cos r':>15}"
print(header)
print("-" * len(header))
for label, d in stats.items():
    print(
        f"{label:<20} {d['n_samples']:>4}"
        f"  {d['real_blank_norm_r']:>17.4f}"
        f"  {d['real_text_norm_r']:>16.4f}"
        f"  {d['real_blank_cos_r']:>16.4f}"
        f"  {d['real_text_cos_r']:>15.4f}"
    )

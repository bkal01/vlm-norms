import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

RUNS = [
    ("runs/smolvlm", "SmolVLM2-2.2B"),
    ("runs/qwen",    "Qwen3-VL-2B"),
]

VISION_COLOR = "#4878CF"
TEXT_COLOR   = "#E87722"


def load_metrics(run_root):
    run_id = os.listdir(run_root)[0]
    run_dir = os.path.join(run_root, run_id)

    vision_norms_list, text_norms_list = [], []
    vision_abs_list,   text_abs_list   = [], []

    for sample in os.listdir(run_dir):
        path = os.path.join(run_dir, sample, "metrics.pt")
        if not os.path.exists(path):
            continue
        m = torch.load(path, weights_only=False)
        vision_norms_list.append(m["vision_norms"])
        text_norms_list.append(m["text_norms"])
        vision_abs_list.append(m["vision_abs"])
        text_abs_list.append(m["text_abs"])

    vision_norms = torch.stack(vision_norms_list).mean(0)[:-1]
    text_norms   = torch.stack(text_norms_list).mean(0)[:-1]
    vision_abs   = torch.stack(vision_abs_list).mean(0)[:-1]
    text_abs     = torch.stack(text_abs_list).mean(0)[:-1]

    return vision_norms, text_norms, vision_abs, text_abs


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

for row, (run_root, label) in enumerate(RUNS):
    vision_norms, text_norms, vision_abs, text_abs = load_metrics(run_root)

    norm_layers = range(len(vision_norms))
    abs_layers  = range(len(vision_abs))

    # --- norms ---
    ax = axes[row, 0]
    ax.plot(norm_layers, vision_norms, color=VISION_COLOR, linewidth=1.5, label="Vision")
    ax.plot(norm_layers, text_norms,   color=TEXT_COLOR,   linewidth=1.5, label="Text")

    # mark crossover
    crossover = next((i for i in norm_layers if text_norms[i] > vision_norms[i]), None)
    if crossover is not None:
        ax.axvline(crossover, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_title(label)
    ax.set_ylabel("Hidden-state norm")
    ax.set_xlabel("Layer")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # --- absolute updates ---
    ax = axes[row, 1]
    ax.plot(abs_layers, vision_abs, color=VISION_COLOR, linewidth=1.5, label="Vision")
    ax.plot(abs_layers, text_abs,   color=TEXT_COLOR,   linewidth=1.5, label="Text")
    ax.set_title(label)
    ax.set_ylabel("Absolute update magnitude")
    ax.set_xlabel("Layer")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

axes[0, 0].set_title(f"{RUNS[0][1]}  —  Hidden-state norms")
axes[0, 1].set_title(f"{RUNS[0][1]}  —  Absolute updates")
axes[1, 0].set_title(f"{RUNS[1][1]}  —  Hidden-state norms")
axes[1, 1].set_title(f"{RUNS[1][1]}  —  Absolute updates")

fig.tight_layout(pad=1.5)
fig.savefig("assets/norm_vs_updates.png", bbox_inches="tight", dpi=200)
print("saved assets/norm_vs_updates.png")

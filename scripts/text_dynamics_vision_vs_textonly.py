import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

RUNS = [
    ("runs/smolvlm", "SmolVLM2-2.2B"),
    ("runs/qwen",    "Qwen3-VL-2B"),
]

MULTIMODAL_COLOR = "#4878CF"
TEXTONLY_COLOR   = "#E87722"


def load_metrics(run_root):
    run_id = os.listdir(run_root)[0]
    run_dir = os.path.join(run_root, run_id)

    text_norms_list, textonly_norms_list = [], []
    text_cos_list,   textonly_cos_list   = [], []

    for sample in os.listdir(run_dir):
        path = os.path.join(run_dir, sample, "metrics.pt")
        if not os.path.exists(path):
            continue
        m = torch.load(path, weights_only=True)
        text_norms_list.append(m["text_norms"])
        textonly_norms_list.append(m["textonly_norms"])
        text_cos_list.append(m["text_cos"])
        textonly_cos_list.append(m["textonly_cos"])

    text_norms    = torch.stack(text_norms_list).float().numpy()
    textonly_norms = torch.stack(textonly_norms_list).float().numpy()
    text_cos      = torch.stack(text_cos_list).float().numpy()
    textonly_cos  = torch.stack(textonly_cos_list).float().numpy()

    norm_r   = np.mean([np.corrcoef(text_norms[i], textonly_norms[i])[0, 1] for i in range(len(text_norms))])
    norm_rel = np.mean(np.abs(text_norms - textonly_norms) / (np.abs(textonly_norms) + 1e-8))
    cos_r    = np.mean([np.corrcoef(text_cos[i], textonly_cos[i])[0, 1] for i in range(len(text_cos))])

    return {
        "text_norms_mean":    text_norms.mean(0),
        "text_norms_std":     text_norms.std(0),
        "textonly_norms_mean": textonly_norms.mean(0),
        "textonly_norms_std":  textonly_norms.std(0),
        "text_cos_mean":      text_cos.mean(0),
        "text_cos_std":       text_cos.std(0),
        "textonly_cos_mean":  textonly_cos.mean(0),
        "textonly_cos_std":   textonly_cos.std(0),
        "norm_r":    norm_r,
        "norm_rel":  norm_rel,
        "cos_r":     cos_r,
        "n_samples": len(text_norms),
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
for row, (run_root, label) in enumerate(RUNS):
    d = load_metrics(run_root)
    stats[label] = d
    layers_norm = range(len(d["text_norms_mean"]))
    layers_cos  = range(len(d["text_cos_mean"]))

    # --- norms (drop final layer — outlier from final layernorm) ---
    ax = axes[row, 0]
    tn_mean = d["text_norms_mean"][:-1]
    tn_std  = d["text_norms_std"][:-1]
    ton_mean = d["textonly_norms_mean"][:-1]
    ton_std  = d["textonly_norms_std"][:-1]
    layers_norm = range(len(tn_mean))
    ax.fill_between(layers_norm, ton_mean - ton_std, ton_mean + ton_std,
                    color=TEXTONLY_COLOR, alpha=0.15)
    ax.fill_between(layers_norm, tn_mean - tn_std, tn_mean + tn_std,
                    color=MULTIMODAL_COLOR, alpha=0.15)
    ax.plot(layers_norm, ton_mean, color=TEXTONLY_COLOR,   linewidth=1.5,
            label="text-only")
    ax.plot(layers_norm, tn_mean,  color=MULTIMODAL_COLOR, linewidth=1.5,
            label="w/ vision")
    ax.set_title(f"{label}  —  Hidden-state norms")
    ax.set_ylabel("Hidden-state norm")
    ax.set_xlabel("Layer")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # --- cos similarity to layer 0 (drop final layer) ---
    ax = axes[row, 1]
    tc_mean = d["text_cos_mean"][:-1]
    tc_std  = d["text_cos_std"][:-1]
    toc_mean = d["textonly_cos_mean"][:-1]
    toc_std  = d["textonly_cos_std"][:-1]
    layers_cos = range(len(tc_mean))
    ax.fill_between(layers_cos, toc_mean - toc_std, toc_mean + toc_std,
                    color=TEXTONLY_COLOR, alpha=0.15)
    ax.fill_between(layers_cos, tc_mean - tc_std, tc_mean + tc_std,
                    color=MULTIMODAL_COLOR, alpha=0.15)
    ax.plot(layers_cos, toc_mean, color=TEXTONLY_COLOR,   linewidth=1.5,
            label="text-only")
    ax.plot(layers_cos, tc_mean,  color=MULTIMODAL_COLOR, linewidth=1.5,
            label="w/ vision")
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

# summary table
print()
header = f"{'Model':<20} {'n':>4}  {'Norms r':>8}  {'Norms rel Δ':>11}  {'Cos r':>7}"
print(header)
print("-" * len(header))
for label, d in stats.items():
    print(f"{label:<20} {d['n_samples']:>4}  {d['norm_r']:>8.4f}  {d['norm_rel']*100:>10.1f}%  {d['cos_r']:>7.4f}")

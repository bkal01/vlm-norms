"""
Figure: obs vs predicted cosine trajectory for vision and text tokens.
Table: layer-1 rel, cos(h_1, h_0), and cos(u_1, h_0) (update alignment).

Predicted cosine (dashed line in figure) assumes updates are random in direction:
  pred_cos[l] = prod_{k=1}^{l} 1 / sqrt(1 + r_k^2)
where r_k = ||u_k|| / ||h_{k-1}|| is the relative update magnitude.

The final layer is excluded throughout (normalization artifact).
"""
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

RUNS = [
    ("runs/smolvlm", "SmolVLM2-2.2B"),
    ("runs/qwen",    "Qwen3-VL-2B"),
]

VISION_COLOR = "#4878CF"
TEXT_COLOR   = "#E87722"
PRED_ALPHA   = 0.55      # alpha for predicted (dashed) line
BAND_ALPHA   = 0.12


def predicted_cos_trajectory(rel: np.ndarray) -> np.ndarray:
    """rel: [L-1] → pred_cos: [L] with pred_cos[0] = 1.0"""
    pred = [1.0]
    log_acc = 0.0
    for r in rel:
        log_acc -= 0.5 * math.log1p(float(r) ** 2)
        pred.append(math.exp(log_acc))
    return np.array(pred, dtype=np.float32)


def load_metrics(run_root):
    run_id = os.listdir(run_root)[0]
    run_dir = os.path.join(run_root, run_id)

    buckets = {tt: {"obs": [], "pred": [], "ratio": [], "rel": [], "update_align": []}
               for tt in ("vision", "text")}

    for sample in os.listdir(run_dir):
        path = os.path.join(run_dir, sample, "metrics.pt")
        if not os.path.exists(path):
            continue
        m = torch.load(path, weights_only=True)

        for tt in ("vision", "text"):
            rel          = m[f"{tt}_rel"][:-1].float().numpy()
            obs          = m[f"{tt}_cos"][:-1].float().numpy()
            update_align = m[f"{tt}_update_align"][:-1].float().numpy()
            pred         = predicted_cos_trajectory(rel)

            buckets[tt]["obs"].append(obs)
            buckets[tt]["pred"].append(pred)
            buckets[tt]["ratio"].append(obs[1:] / pred[1:])
            buckets[tt]["rel"].append(rel)
            buckets[tt]["update_align"].append(update_align)

    result = {}
    for tt in ("vision", "text"):
        obs_arr          = np.stack(buckets[tt]["obs"])
        pred_arr         = np.stack(buckets[tt]["pred"])
        ratio_arr        = np.stack(buckets[tt]["ratio"])
        rel_arr          = np.stack(buckets[tt]["rel"])
        update_align_arr = np.stack(buckets[tt]["update_align"])

        result[tt] = {
            "obs_mean":          obs_arr.mean(0),
            "obs_std":           obs_arr.std(0),
            "pred_mean":         pred_arr.mean(0),
            "ratio_mean":        ratio_arr.mean(0),
            "rel_mean":          rel_arr.mean(0),
            "update_align_mean": update_align_arr.mean(0),
            "n_layers":          ratio_arr.shape[1],
            "n_samples":         len(obs_arr),
        }
    return result


# ── plot ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
})

TOKEN_TYPES = [("vision", "Vision tokens", VISION_COLOR),
               ("text",   "Text tokens",   TEXT_COLOR)]

fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))

stats = {}
for row, (run_root, model_label) in enumerate(RUNS):
    d = load_metrics(run_root)
    stats[model_label] = d

    for col, (tt, tt_label, color) in enumerate(TOKEN_TYPES):
        ax = axes[row, col]
        obs_mean  = d[tt]["obs_mean"]
        obs_std   = d[tt]["obs_std"]
        pred_mean = d[tt]["pred_mean"]
        layers = np.arange(len(obs_mean))

        ax.fill_between(layers,
                        obs_mean - obs_std,
                        obs_mean + obs_std,
                        color=color, alpha=BAND_ALPHA)
        ax.plot(layers, obs_mean,  color=color, linewidth=1.8,
                label="observed", solid_capstyle="round")
        ax.plot(layers, pred_mean, color=color, linewidth=1.5,
                linestyle="--", alpha=PRED_ALPHA, label="predicted (random)")

        ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.35)

        ax.set_title(f"{model_label}  —  {tt_label}")
        ax.set_ylabel("Cosine similarity to layer 0")
        ax.set_xlabel("Layer")
        ax.legend(frameon=False, loc="upper right",
                  bbox_to_anchor=(0.97, 0.80))
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

fig.tight_layout(pad=1.5)
fig.savefig("assets/alignment_ratio.png", bbox_inches="tight", dpi=200)
print("saved assets/alignment_ratio.png")


# ── table ─────────────────────────────────────────────────────────────────────
# u_1 = h_1 - h_0 (layer-1 update); h_0 is post-embed pre-block residual.

print()
col_rel = "||u_1||/||h_0||"
col_obs = "cos(h_1, h_0)"
col_aln = "cos(u_1, h_0)"
w = max(len(col_rel), len(col_obs), len(col_aln), 10)
header = (
    f"{'Model':<20}  {'token':>6}  {col_rel:>{w}}  "
    f"{col_obs:>{w}}  {col_aln:>{w}}  {'n':>5}"
)
sep = "-" * len(header)
print(header)
print(sep)
for model_label, d in stats.items():
    for tt in ("vision", "text"):
        r      = d[tt]
        rel1   = float(r["rel_mean"][0])
        obs1   = float(r["obs_mean"][1])
        align1 = float(r["update_align_mean"][0])
        n      = r["n_samples"]
        print(
            f"{model_label:<20}  {tt:>6}  {rel1:>{w}.4f}  "
            f"{obs1:>{w}.4f}  {align1:>{w}.4f}  {n:>5}"
        )
print(sep)

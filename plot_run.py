import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


METRIC_KEYS = {
    "norms": {"ylabel": r"mean $\|h_\ell\|$", "title": "Mean token norm"},
    "abs": {"ylabel": r"mean $\|h_\ell - h_{\ell-1}\|$", "title": "Absolute update magnitude"},
    "rel": {"ylabel": r"$\|h_\ell - h_{\ell-1}\| \;/\; \|h_{\ell-1}\|$", "title": "Relative update magnitude"},
    "cos": {"ylabel": r"mean $\cos(h_\ell,\, h_0)$", "title": "Cosine similarity to layer-0 embedding"},
}
SERIES = [
    ("vision", "Vision tokens (VLM)"),
    ("text", "Text tokens (VLM)"),
    ("textonly", "Text-only model"),
]


def load_run(run_dir: Path) -> dict[str, list[torch.Tensor]]:
    """Load all sample metrics.pt files and group by metric key."""
    stacks: dict[str, list[torch.Tensor]] = {
        f"{s}_{m}": [] for s, _ in SERIES for m in METRIC_KEYS
    }
    sample_dirs = sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and (p / "metrics.pt").exists()
    )
    for sd in sample_dirs:
        d = torch.load(sd / "metrics.pt", map_location="cpu", weights_only=False)
        for key in stacks:
            stacks[key].append(d[key])
    return {k: torch.stack(v) for k, v in stacks.items() if v}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--runs-dir", default="runs", type=Path)
    args = p.parse_args()

    run_dir = args.runs_dir / args.run_id
    data = load_run(run_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(args.run_id, fontsize=11)

    for ax, (metric, info) in zip(axes.flat, METRIC_KEYS.items()):
        is_update = metric in ("abs", "rel")
        for series_key, label in SERIES:
            vals = data[f"{series_key}_{metric}"].numpy()
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            x = np.arange(1, len(mean) + 1) if is_update else np.arange(len(mean))
            line, = ax.plot(x, mean, label=label, marker="o", markersize=3)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())

        if metric == "rel":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        elif metric == "cos":
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

        ax.set_title(info["title"])
        ax.set_xlabel("Layer")
        ax.set_ylabel(info["ylabel"])
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

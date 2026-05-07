import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[1]

RUN_DIRS = [
    ROOT / "kv_cache_substitution_runs" / "kv_cache_substitution_01409116",
    ROOT / "kv_cache_substitution_runs" / "kv_cache_substitution_5d032ed9",
]

COLOR = "#4878CF"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        raise ValueError("median of empty list")
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float(0.5 * (xs[mid - 1] + xs[mid]))


def main():
    runs = []
    for run_dir in RUN_DIRS:
        cfg = read_json(run_dir / "kv_cache_substitution_config.json")
        model_name = cfg["model"]
        rows = read_jsonl(run_dir / "kv_cache_substitution.jsonl")
        shifts = [
            float(r["fraction_of_blank_shift"])
            for r in rows
            if r.get("fraction_of_blank_shift") is not None
        ]
        runs.append(
            {
                "model": model_name,
                "shifts": shifts,
            }
        )

    all_shifts = [x for run in runs for x in run["shifts"]]
    x_max = max(1.0, max(all_shifts) if all_shifts else 1.0)
    x_max = min(2.0, 1.05 * x_max)

    fig, axes = plt.subplots(1, len(runs), figsize=(9, 3.5), sharey=True)
    if len(runs) == 1:
        axes = [axes]

    bins = 30
    for ax, run in zip(axes, runs):
        shifts = run["shifts"]
        m = median(shifts)
        ax.hist(
            shifts,
            bins=bins,
            range=(0.0, x_max),
            color=COLOR,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.axvline(m, color="#666666", linewidth=1.2, linestyle="--")
        ax.text(
            m,
            0.98,
            f"median = {m:.3f}",
            ha="center",
            va="top",
            fontsize=8,
            color="#666666",
            transform=ax.get_xaxis_transform(),
        )

        ax.set_title(run["model"].split("/")[-1])
        ax.set_xlabel("fraction_of_blank_shift")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    axes[0].set_ylabel("Count")

    fig.tight_layout(pad=1.0)
    out_dir = ROOT / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_kv_cache_substitution_fraction_shift.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()


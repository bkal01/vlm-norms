"""Compute generation-time logit sensitivity against alpha=1.0."""

import argparse
import csv
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


BASELINE_ALPHA = 1.0


def load_metrics(path: Path) -> dict:
    try:
        return torch.load(path, weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, weights_only=False)


def generation_logits(generation: dict, max_steps: int | None) -> torch.Tensor | None:
    logits = generation.get("generation_logits")
    if logits is None:
        logits = generation.get("generation_scores")
    if logits is None:
        return None

    if max_steps is not None:
        logits = logits[:max_steps]
    return torch.cat([step.detach().cpu().float() for step in logits], dim=0)


def load_sample(sample_dir: Path, max_steps: int | None) -> dict[float, dict]:
    by_alpha = {}
    for path in sorted(sample_dir.glob("alpha_*/metrics.pt")):
        metrics = load_metrics(path)
        generation = metrics["generation"]
        logits = generation_logits(generation, max_steps)
        if logits is None:
            print(f"skip {path}: no generation logits or scores")
            continue
        by_alpha[float(metrics["alpha"])] = {
            "logits": logits,
            "generated_token_ids": generation["generated_token_ids"].detach().cpu(),
        }
    return by_alpha


def compute_sample_rows(sample_dir: Path, by_alpha: dict[float, dict]) -> list[dict]:
    rows = []
    if BASELINE_ALPHA not in by_alpha:
        return rows

    subset = sample_dir.parent.name
    sample_id = sample_dir.name
    baseline = by_alpha[BASELINE_ALPHA]
    baseline_logits = baseline["logits"]
    baseline_tokens = baseline["generated_token_ids"]

    for alpha, item in sorted(by_alpha.items()):
        logits = item["logits"]
        if logits.shape[-1] != baseline_logits.shape[-1]:
            print(
                f"skip {subset}/{sample_id} alpha={alpha:g}: "
                f"vocab {logits.shape[-1]} != {baseline_logits.shape[-1]}"
            )
            continue

        n_steps = min(logits.shape[0], baseline_logits.shape[0], baseline_tokens.shape[0])
        logits = logits[:n_steps]
        baseline_prefix = baseline_logits[:n_steps]
        baseline_token_prefix = baseline_tokens[:n_steps]

        logp = torch.log_softmax(logits, dim=-1)
        logq = torch.log_softmax(baseline_prefix, dim=-1)
        p = logp.exp()
        kl = (p * (logp - logq)).sum(dim=-1)

        baseline_token_logp = logp.gather(1, baseline_token_prefix[:, None]).squeeze(1)
        baseline_token_prob = baseline_token_logp.exp()
        baseline_token_logits = logits.gather(1, baseline_token_prefix[:, None]).squeeze(1)
        baseline_token_rank = (logits > baseline_token_logits[:, None]).sum(dim=-1) + 1
        greedy_token = logits.argmax(dim=-1)
        baseline_greedy_token = baseline_prefix.argmax(dim=-1)
        greedy_agreement = greedy_token == baseline_greedy_token

        for step in range(n_steps):
            rows.append(
                {
                    "subset": subset,
                    "sample_id": sample_id,
                    "alpha": alpha,
                    "step": step,
                    "kl": float(kl[step]),
                    "baseline_token_id": int(baseline_token_prefix[step]),
                    "baseline_token_prob": float(baseline_token_prob[step]),
                    "baseline_token_rank": int(baseline_token_rank[step]),
                    "greedy_token_id": int(greedy_token[step]),
                    "baseline_greedy_token_id": int(baseline_greedy_token[step]),
                    "greedy_agreement": int(greedy_agreement[step]),
                }
            )
    return rows


def compute_run_rows(run_dir: Path, max_steps: int | None) -> list[dict]:
    rows = []
    sample_dirs = sorted({
        path.parent.parent
        for path in run_dir.glob("*/*/alpha_*/metrics.pt")
    })
    for sample_dir in sample_dirs:
        rows.extend(compute_sample_rows(sample_dir, load_sample(sample_dir, max_steps)))
    return rows


def finite_mean(values: list[float]) -> float:
    array = np.array(values, dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def finite_median(values: list[float]) -> float:
    array = np.array(values, dtype=float)
    return float(np.nanmedian(array)) if np.isfinite(array).any() else float("nan")


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for alpha in sorted({row["alpha"] for row in rows}):
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        summary.append(
            {
                "alpha": alpha,
                "kl_mean": finite_mean([row["kl"] for row in alpha_rows]),
                "baseline_token_prob_mean": finite_mean(
                    [row["baseline_token_prob"] for row in alpha_rows]
                ),
                "baseline_token_rank_median": finite_median(
                    [row["baseline_token_rank"] for row in alpha_rows]
                ),
                "greedy_agreement_mean": finite_mean(
                    [row["greedy_agreement"] for row in alpha_rows]
                ),
                "n_points": len(alpha_rows),
            }
        )
    return summary


def metric_by_alpha(rows: list[dict], alphas: list[float], metric: str) -> list[float]:
    values = []
    for alpha in alphas:
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        if metric == "baseline_token_rank":
            values.append(finite_median([row[metric] for row in alpha_rows]))
        else:
            values.append(finite_mean([row[metric] for row in alpha_rows]))
    return values


def plot_summary(path: Path, rows: list[dict]) -> None:
    alphas = sorted({row["alpha"] for row in rows})
    steps = sorted({row["step"] for row in rows})
    nonbaseline_alphas = [alpha for alpha in alphas if alpha != BASELINE_ALPHA]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))

    axes[0, 0].plot(alphas, metric_by_alpha(rows, alphas, "kl"), marker="o")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("KL vs baseline")
    axes[0, 0].set_xlabel("alpha")
    axes[0, 0].set_ylabel("mean KL")

    axes[0, 1].plot(
        alphas,
        metric_by_alpha(rows, alphas, "baseline_token_prob"),
        marker="o",
        color="#59A14F",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Baseline token probability")
    axes[0, 1].set_xlabel("alpha")
    axes[0, 1].set_ylabel("mean probability")

    axes[0, 2].plot(
        alphas,
        metric_by_alpha(rows, alphas, "baseline_token_rank"),
        marker="o",
        color="#E15759",
    )
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_title("Baseline token rank")
    axes[0, 2].set_xlabel("alpha")
    axes[0, 2].set_ylabel("median rank")

    axes[1, 0].plot(
        alphas,
        metric_by_alpha(rows, alphas, "greedy_agreement"),
        marker="o",
        color="#F28E2B",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title("Greedy agreement")
    axes[1, 0].set_xlabel("alpha")
    axes[1, 0].set_ylabel("fraction")

    kl_heatmap = np.full((len(nonbaseline_alphas), len(steps)), np.nan)
    agreement_heatmap = np.full((len(nonbaseline_alphas), len(steps)), np.nan)
    for a_idx, alpha in enumerate(nonbaseline_alphas):
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        for s_idx, step in enumerate(steps):
            step_rows = [row for row in alpha_rows if row["step"] == step]
            kl_heatmap[a_idx, s_idx] = finite_mean([row["kl"] for row in step_rows])
            agreement_heatmap[a_idx, s_idx] = finite_mean(
                [row["greedy_agreement"] for row in step_rows]
            )

    kl_img = axes[1, 1].imshow(kl_heatmap, aspect="auto", interpolation="nearest")
    axes[1, 1].set_title("KL by generated step")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("alpha")
    axes[1, 1].set_yticks(
        range(len(nonbaseline_alphas)),
        [f"{alpha:g}" for alpha in nonbaseline_alphas],
    )
    axes[1, 1].set_xticks(range(len(steps)), steps)
    fig.colorbar(kl_img, ax=axes[1, 1], fraction=0.046)

    agree_img = axes[1, 2].imshow(
        agreement_heatmap,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    axes[1, 2].set_title("Agreement by generated step")
    axes[1, 2].set_xlabel("step")
    axes[1, 2].set_ylabel("alpha")
    axes[1, 2].set_yticks(
        range(len(nonbaseline_alphas)),
        [f"{alpha:g}" for alpha in nonbaseline_alphas],
    )
    axes[1, 2].set_xticks(range(len(steps)), steps)
    fig.colorbar(agree_img, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    summary = summarize(rows)

    print()
    print(
        f"{'alpha':>8}  {'mean_KL':>12}  {'base_prob':>12}  "
        f"{'median_rank':>12}  {'greedy':>8}  {'points':>8}"
    )
    print("-" * 78)
    for row in summary:
        print(
            f"{row['alpha']:>8g}  "
            f"{row['kl_mean']:>12.6g}  "
            f"{row['baseline_token_prob_mean']:>12.6g}  "
            f"{row['baseline_token_rank_median']:>12.6g}  "
            f"{row['greedy_agreement_mean']:>8.3f}  "
            f"{row['n_points']:>8}"
        )
    print("-" * 78)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze next-token logit sensitivity from alpha=1.0 baseline."
    )
    parser.add_argument("run_uuid", type=uuid.UUID, help="Run UUID under runs/.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path("runs") / str(args.run_uuid)

    rows = compute_run_rows(run_dir, args.max_steps)
    if not rows:
        raise SystemExit(f"No comparable alpha runs found under {run_dir}")

    csv_path = args.output_dir / "logit_sensitivity.csv"
    figure_path = args.output_dir / "logit_sensitivity.png"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_summary(figure_path, rows)
    print_summary(rows)

    n_samples = len({(row["subset"], row["sample_id"]) for row in rows})
    print(f"\nsamples={n_samples}")
    print(f"rows={len(rows)}")
    print(f"saved {csv_path}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()

"""Compute attention divergence and saved attention-mass shifts from alpha=1.0."""

import argparse
import csv
import json
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


BASELINE_ALPHA = 1.0
PRECOMPUTED_FILENAME = "attention_divergence_from_baseline.jsonl"
MASS_CSV_FILENAME = "attention_mass_from_baseline.csv"
MASS_SUMMARY_CSV_FILENAME = "attention_mass_summary.csv"
MASS_LAYER_SUMMARY_CSV_FILENAME = "attention_mass_layer_summary.csv"
MASS_FIGURE_FILENAME = "attention_mass_from_baseline.png"


def load_precomputed_rows(run_dir: Path, max_steps: int | None) -> list[dict]:
    path = run_dir / PRECOMPUTED_FILENAME
    if not path.exists():
        return []

    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
            if max_steps is not None and int(row["step"]) >= max_steps:
                continue
            row["alpha"] = float(row["alpha"])
            row["step"] = int(row["step"])
            row["layer"] = int(row["layer"])
            row["kl_mean"] = float(row["kl_mean"])
            row["cosine_mean"] = float(row["cosine_mean"])
            rows.append(row)
    return rows


def extract_vision_attention(
    metrics: dict,
    max_steps: int | None,
) -> torch.Tensor | None:
    generation = metrics["generation"]
    prompt_image_mask = generation["prompt_image_mask"].bool()
    prompt_length = int(generation["prompt_length"])
    # HF generate returns attentions[0] for full-prompt prefill and attentions[1:]
    # for generated-token queries. Attention divergence is a decode-time metric.
    steps = generation["attentions"][1:]
    if max_steps is not None:
        steps = steps[:max_steps]

    by_step = []
    for step_attentions in steps:
        by_layer = []
        for attn in step_attentions:
            prompt_attn = attn[..., -1, :prompt_length]
            by_layer.append(prompt_attn[..., prompt_image_mask].squeeze(0))
        by_step.append(torch.stack(by_layer))

    if not by_step:
        return None
    return torch.stack(by_step).float()


def load_sample(sample_dir: Path, max_steps: int | None) -> dict[float, torch.Tensor]:
    by_alpha = {}
    for path in sorted(sample_dir.glob("alpha_*/metrics.pt")):
        metrics = torch.load(path, weights_only=False)
        alpha = float(metrics["alpha"])
        attention = extract_vision_attention(metrics, max_steps)
        if attention is not None:
            by_alpha[alpha] = attention
    return by_alpha


def load_metrics(path: Path) -> dict:
    try:
        return torch.load(str(path), weights_only=False, mmap=True)
    except (TypeError, ValueError):
        return torch.load(path, weights_only=False)


def load_attention_mass_sample(
    sample_dir: Path,
    max_steps: int | None,
) -> dict[float, dict[str, torch.Tensor]]:
    by_alpha = {}
    for path in sorted(sample_dir.glob("alpha_*/metrics.pt")):
        metrics = load_metrics(path)
        alpha = float(metrics["alpha"])

        phase_masses = {}
        decoder_mass = metrics.get("decoder_vision_attention_mass")
        if decoder_mass is not None:
            decoder_mass = decoder_mass.detach().cpu().float()
            if max_steps is not None:
                decoder_mass = decoder_mass[:max_steps]
            phase_masses["decoder"] = decoder_mass

        prefill_mass = metrics.get("prefill_text_query_vision_attention_mass")
        if prefill_mass is not None:
            phase_masses["prefill"] = prefill_mass.detach().cpu().float()

        if phase_masses:
            by_alpha[alpha] = phase_masses
    return by_alpha


def compute_attention_mass_rows(
    sample_dir: Path,
    by_alpha: dict[float, dict[str, torch.Tensor]],
) -> list[dict]:
    rows = []
    if BASELINE_ALPHA not in by_alpha:
        return rows

    subset = sample_dir.parent.name
    sample_id = sample_dir.name
    baseline_by_phase = by_alpha[BASELINE_ALPHA]

    for alpha, phase_masses in sorted(by_alpha.items()):
        if alpha == BASELINE_ALPHA:
            continue
        for phase, mass in sorted(phase_masses.items()):
            baseline = baseline_by_phase.get(phase)
            if baseline is None:
                continue
            if phase == "decoder":
                n_steps = min(mass.shape[0], baseline.shape[0])
                n_layers = min(mass.shape[1], baseline.shape[1])
                mass_prefix = mass[:n_steps, :n_layers]
                baseline_prefix = baseline[:n_steps, :n_layers]
                for step in range(n_steps):
                    for layer in range(n_layers):
                        value = float(mass_prefix[step, layer])
                        baseline_value = float(baseline_prefix[step, layer])
                        rows.append(
                            {
                                "subset": subset,
                                "sample_id": sample_id,
                                "alpha": alpha,
                                "phase": phase,
                                "step": step,
                                "layer": layer,
                                "vision_attention_mass": value,
                                "baseline_vision_attention_mass": baseline_value,
                                "delta": value - baseline_value,
                                "abs_delta": abs(value - baseline_value),
                                "ratio": value / max(baseline_value, 1e-8),
                            }
                        )
            elif phase == "prefill":
                n_layers = min(mass.shape[0], baseline.shape[0])
                mass_prefix = mass[:n_layers]
                baseline_prefix = baseline[:n_layers]
                for layer in range(n_layers):
                    value = float(mass_prefix[layer])
                    baseline_value = float(baseline_prefix[layer])
                    rows.append(
                        {
                            "subset": subset,
                            "sample_id": sample_id,
                            "alpha": alpha,
                            "phase": phase,
                            "step": "",
                            "layer": layer,
                            "vision_attention_mass": value,
                            "baseline_vision_attention_mass": baseline_value,
                            "delta": value - baseline_value,
                            "abs_delta": abs(value - baseline_value),
                            "ratio": value / max(baseline_value, 1e-8),
                        }
                    )
    return rows


def compute_run_attention_mass_rows(
    run_dir: Path,
    max_steps: int | None,
) -> list[dict]:
    rows = []
    sample_dirs = sorted({
        path.parent.parent
        for path in run_dir.glob("*/*/alpha_*/metrics.pt")
    })

    for sample_dir in sample_dirs:
        rows.extend(
            compute_attention_mass_rows(
                sample_dir,
                load_attention_mass_sample(sample_dir, max_steps),
            )
        )
    return rows


def compute_sample_rows(sample_dir: Path, by_alpha: dict[float, torch.Tensor]) -> list[dict]:
    rows = []
    if BASELINE_ALPHA not in by_alpha:
        return rows

    subset = sample_dir.parent.name
    sample_id = sample_dir.name
    baseline = by_alpha[BASELINE_ALPHA]
    for alpha, attn in sorted(by_alpha.items()):
        if alpha == BASELINE_ALPHA:
            continue
        if attn.shape[1:] != baseline.shape[1:]:
            print(
                f"skip {subset}/{sample_id} alpha={alpha:g}: "
                f"shape {tuple(attn.shape)} != {tuple(baseline.shape)}"
            )
            continue
        n_steps = min(attn.shape[0], baseline.shape[0])
        attn = attn[:n_steps]
        baseline_prefix = baseline[:n_steps]

        p = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        q = baseline_prefix / baseline_prefix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        kl = (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(dim=-1)
        cosine = torch.nn.functional.cosine_similarity(attn, baseline_prefix, dim=-1)

        for step in range(attn.shape[0]):
            for layer in range(attn.shape[1]):
                rows.append(
                    {
                        "subset": subset,
                        "sample_id": sample_id,
                        "alpha": alpha,
                        "step": step,
                        "layer": layer,
                        "kl_mean": float(kl[step, layer].mean()),
                        "cosine_mean": float(cosine[step, layer].mean()),
                    }
                )
    return rows


def compute_run_rows(
    run_dir: Path,
    max_steps: int | None,
) -> list[dict]:
    if (run_dir / PRECOMPUTED_FILENAME).exists():
        return load_precomputed_rows(run_dir, max_steps)

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


def summarize_attention_mass(rows: list[dict]) -> list[dict]:
    summary = []
    phases = sorted({row["phase"] for row in rows})
    alphas = sorted({row["alpha"] for row in rows})
    for phase in phases:
        for alpha in alphas:
            phase_alpha_rows = [
                row for row in rows
                if row["phase"] == phase and row["alpha"] == alpha
            ]
            if not phase_alpha_rows:
                continue
            summary.append(
                {
                    "phase": phase,
                    "alpha": alpha,
                    "mean_mass": finite_mean(
                        [row["vision_attention_mass"] for row in phase_alpha_rows]
                    ),
                    "mean_baseline_mass": finite_mean(
                        [
                            row["baseline_vision_attention_mass"]
                            for row in phase_alpha_rows
                        ]
                    ),
                    "mean_delta": finite_mean(
                        [row["delta"] for row in phase_alpha_rows]
                    ),
                    "mean_abs_delta": finite_mean(
                        [row["abs_delta"] for row in phase_alpha_rows]
                    ),
                    "mean_ratio": finite_mean(
                        [row["ratio"] for row in phase_alpha_rows]
                    ),
                    "n_points": len(phase_alpha_rows),
                }
            )
    return summary


def summarize_attention_mass_by_layer(rows: list[dict]) -> list[dict]:
    summary = []
    phases = sorted({row["phase"] for row in rows})
    alphas = sorted({row["alpha"] for row in rows})
    layers = sorted({row["layer"] for row in rows})
    for phase in phases:
        for alpha in alphas:
            for layer in layers:
                phase_alpha_layer_rows = [
                    row for row in rows
                    if (
                        row["phase"] == phase
                        and row["alpha"] == alpha
                        and row["layer"] == layer
                    )
                ]
                if not phase_alpha_layer_rows:
                    continue
                summary.append(
                    {
                        "phase": phase,
                        "alpha": alpha,
                        "layer": layer,
                        "mean_mass": finite_mean(
                            [
                                row["vision_attention_mass"]
                                for row in phase_alpha_layer_rows
                            ]
                        ),
                        "mean_baseline_mass": finite_mean(
                            [
                                row["baseline_vision_attention_mass"]
                                for row in phase_alpha_layer_rows
                            ]
                        ),
                        "mean_delta": finite_mean(
                            [row["delta"] for row in phase_alpha_layer_rows]
                        ),
                        "mean_abs_delta": finite_mean(
                            [row["abs_delta"] for row in phase_alpha_layer_rows]
                        ),
                        "mean_ratio": finite_mean(
                            [row["ratio"] for row in phase_alpha_layer_rows]
                        ),
                        "n_points": len(phase_alpha_layer_rows),
                    }
                )
    return summary


def plot_attention_mass_summary(path: Path, rows: list[dict]) -> None:
    phases = sorted({row["phase"] for row in rows})
    alphas = sorted({row["alpha"] for row in rows})
    summary = summarize_attention_mass(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    colors = {"decoder": "#4878CF", "prefill": "#E87722"}

    for phase in phases:
        phase_summary = [row for row in summary if row["phase"] == phase]
        phase_alphas = [row["alpha"] for row in phase_summary]
        masses = [row["mean_mass"] for row in phase_summary]
        deltas = [row["mean_delta"] for row in phase_summary]
        axes[0].plot(
            phase_alphas,
            masses,
            marker="o",
            label=phase,
            color=colors.get(phase),
        )
        axes[1].plot(
            phase_alphas,
            deltas,
            marker="o",
            label=phase,
            color=colors.get(phase),
        )

    axes[0].set_xscale("log")
    axes[0].set_title("Vision attention mass")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("mean mass")

    axes[1].axhline(0, color="#888888", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_title("Delta vs alpha=1")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("mean delta")

    for ax in axes:
        ax.set_xticks(alphas, [f"{alpha:g}" for alpha in alphas])
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_attention_mass_summary(rows: list[dict]) -> None:
    summary = summarize_attention_mass(rows)

    print()
    print("Vision attention mass vs alpha=1 baseline")
    print(
        f"{'phase':>8}  {'alpha':>8}  {'mass':>12}  {'base':>12}  "
        f"{'delta':>12}  {'abs_delta':>12}  {'ratio':>10}  {'points':>8}"
    )
    print("-" * 98)
    for row in summary:
        print(
            f"{row['phase']:>8}  "
            f"{row['alpha']:>8g}  "
            f"{row['mean_mass']:>12.6g}  "
            f"{row['mean_baseline_mass']:>12.6g}  "
            f"{row['mean_delta']:>12.6g}  "
            f"{row['mean_abs_delta']:>12.6g}  "
            f"{row['mean_ratio']:>10.6g}  "
            f"{row['n_points']:>8}"
        )
    print("-" * 98)


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for alpha in sorted({row["alpha"] for row in rows}):
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        kl_values = np.array([row["kl_mean"] for row in alpha_rows], dtype=float)
        cosine_values = np.array([row["cosine_mean"] for row in alpha_rows], dtype=float)
        summary.append(
            {
                "alpha": alpha,
                "kl_mean": (
                    float(np.nanmean(kl_values))
                    if np.isfinite(kl_values).any()
                    else float("nan")
                ),
                "cosine_mean": (
                    float(np.nanmean(cosine_values))
                    if np.isfinite(cosine_values).any()
                    else float("nan")
                ),
                "n_points": len(alpha_rows),
            }
        )
    return summary


def plot_summary(path: Path, rows: list[dict]) -> None:
    alphas = sorted({row["alpha"] for row in rows})
    layers = sorted({row["layer"] for row in rows})

    kl_by_alpha = []
    cos_by_alpha = []
    kl_heatmap = np.full((len(alphas), len(layers)), np.nan)
    cos_heatmap = np.full((len(alphas), len(layers)), np.nan)

    for a_idx, alpha in enumerate(alphas):
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        kl_values = np.array([row["kl_mean"] for row in alpha_rows], dtype=float)
        cos_values = np.array([row["cosine_mean"] for row in alpha_rows], dtype=float)
        kl_by_alpha.append(
            float(np.nanmean(kl_values))
            if np.isfinite(kl_values).any()
            else float("nan")
        )
        cos_by_alpha.append(
            float(np.nanmean(cos_values))
            if np.isfinite(cos_values).any()
            else float("nan")
        )

        for l_idx, layer in enumerate(layers):
            layer_rows = [row for row in alpha_rows if row["layer"] == layer]
            layer_kl_values = np.array(
                [row["kl_mean"] for row in layer_rows],
                dtype=float,
            )
            layer_cos_values = np.array(
                [row["cosine_mean"] for row in layer_rows],
                dtype=float,
            )
            kl_heatmap[a_idx, l_idx] = (
                float(np.nanmean(layer_kl_values))
                if np.isfinite(layer_kl_values).any()
                else float("nan")
            )
            cos_heatmap[a_idx, l_idx] = (
                float(np.nanmean(layer_cos_values))
                if np.isfinite(layer_cos_values).any()
                else float("nan")
            )

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    axes[0, 0].plot(alphas, kl_by_alpha, marker="o", color="#4878CF")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("KL vs baseline")
    axes[0, 0].set_xlabel("alpha")
    axes[0, 0].set_ylabel("mean KL")

    axes[0, 1].plot(alphas, cos_by_alpha, marker="o", color="#E87722")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Cosine vs baseline")
    axes[0, 1].set_xlabel("alpha")
    axes[0, 1].set_ylabel("mean cosine")

    kl_img = axes[1, 0].imshow(kl_heatmap, aspect="auto", interpolation="nearest")
    axes[1, 0].set_title("KL by layer")
    axes[1, 0].set_xlabel("layer")
    axes[1, 0].set_ylabel("alpha")
    axes[1, 0].set_yticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
    axes[1, 0].set_xticks(range(len(layers)), layers)
    fig.colorbar(kl_img, ax=axes[1, 0], fraction=0.046)

    cos_img = axes[1, 1].imshow(cos_heatmap, aspect="auto", interpolation="nearest")
    axes[1, 1].set_title("Cosine by layer")
    axes[1, 1].set_xlabel("layer")
    axes[1, 1].set_ylabel("alpha")
    axes[1, 1].set_yticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
    axes[1, 1].set_xticks(range(len(layers)), layers)
    fig.colorbar(cos_img, ax=axes[1, 1], fraction=0.046)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    summary = summarize(rows)

    print()
    print(f"{'alpha':>8}  {'mean_KL':>12}  {'mean_cosine':>12}  {'points':>8}")
    print("-" * 48)
    for row in summary:
        print(
            f"{row['alpha']:>8g}  "
            f"{row['kl_mean']:>12.6g}  "
            f"{row['cosine_mean']:>12.6g}  "
            f"{row['n_points']:>8}"
        )
    print("-" * 48)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze attention divergence and saved vision-attention mass "
            "from alpha=1.0 baseline."
        )
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

    csv_path = args.output_dir / "attention_divergence_from_baseline.csv"
    figure_path = args.output_dir / "attention_divergence_from_baseline.png"

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

    mass_rows = compute_run_attention_mass_rows(run_dir, args.max_steps)
    if mass_rows:
        mass_csv_path = args.output_dir / MASS_CSV_FILENAME
        mass_summary_csv_path = args.output_dir / MASS_SUMMARY_CSV_FILENAME
        mass_layer_summary_csv_path = (
            args.output_dir / MASS_LAYER_SUMMARY_CSV_FILENAME
        )
        mass_figure_path = args.output_dir / MASS_FIGURE_FILENAME

        write_csv(mass_csv_path, mass_rows)
        write_csv(mass_summary_csv_path, summarize_attention_mass(mass_rows))
        write_csv(
            mass_layer_summary_csv_path,
            summarize_attention_mass_by_layer(mass_rows),
        )
        plot_attention_mass_summary(mass_figure_path, mass_rows)
        print_attention_mass_summary(mass_rows)
        print(f"\nmass_rows={len(mass_rows)}")
        print(f"saved {mass_csv_path}")
        print(f"saved {mass_summary_csv_path}")
        print(f"saved {mass_layer_summary_csv_path}")
        print(f"saved {mass_figure_path}")
    else:
        print("\nNo saved attention-mass metrics found in metrics.pt files.")


if __name__ == "__main__":
    main()

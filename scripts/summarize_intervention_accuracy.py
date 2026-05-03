"""Summarize alpha-sweep accuracy and paired correctness flips."""

import argparse
import csv
import json
import math
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


BASELINE_ALPHA = 1.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
    return rows


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def get_sample_id(row: dict[str, Any]) -> str:
    sample_id = row.get("id") or row.get("sample_id")
    if sample_id is None:
        raise KeyError(f"score row is missing id/sample_id: {row}")
    return str(sample_id)


def get_correct(row: dict[str, Any]) -> bool:
    correct = as_bool(row.get("is_correct"))
    if correct is not None:
        return correct

    score = as_float(row.get("score"))
    if score is None:
        raise KeyError(f"score row is missing is_correct and numeric score: {row}")
    return score > 0


def score_value(row: dict[str, Any]) -> float:
    score = as_float(row.get("score"))
    return float("nan") if score is None else score


def keep_alpha_row(
    row: dict[str, Any],
    include_textonly: bool,
    condition_prefix: str,
) -> bool:
    alpha = as_float(row.get("alpha"))
    condition = str(row.get("condition", ""))
    if alpha is not None and condition.startswith(condition_prefix):
        return True
    return include_textonly and row.get("condition") == "textonly"


def alpha_label(row: dict[str, Any]) -> str:
    alpha = as_float(row.get("alpha"))
    if alpha is None:
        return str(row.get("condition", "unknown"))
    return f"{alpha:g}"


def flip_type(baseline_correct: bool, current_correct: bool) -> str:
    if baseline_correct and current_correct:
        return "correct_to_correct"
    if baseline_correct and not current_correct:
        return "correct_to_incorrect"
    if not baseline_correct and current_correct:
        return "incorrect_to_correct"
    return "incorrect_to_incorrect"


def finite_mean(values: list[float]) -> float:
    array = np.array(values, dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def pair_rows(
    score_rows: list[dict[str, Any]],
    baseline_alpha: float,
    include_textonly: bool,
    condition_prefix: str,
) -> list[dict[str, Any]]:
    by_sample: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in score_rows:
        if not keep_alpha_row(row, include_textonly, condition_prefix):
            continue
        key = (str(row.get("subset", "")), get_sample_id(row))
        by_sample[key][alpha_label(row)] = row

    baseline_key = f"{baseline_alpha:g}"
    rows = []
    for (subset, sample_id), by_alpha in sorted(by_sample.items()):
        baseline = by_alpha.get(baseline_key)
        if baseline is None:
            print(f"skip {subset}/{sample_id}: no alpha={baseline_key} baseline")
            continue

        baseline_correct = get_correct(baseline)
        baseline_score = score_value(baseline)
        baseline_output = baseline.get("vlm_output", "")

        for label, row in sorted(by_alpha.items(), key=lambda item: sort_key(item[0])):
            current_correct = get_correct(row)
            current_score = score_value(row)
            alpha = as_float(row.get("alpha"))
            rows.append(
                {
                    "subset": subset,
                    "sample_id": sample_id,
                    "alpha": "" if alpha is None else alpha,
                    "condition": row.get("condition", ""),
                    "baseline_condition": baseline.get("condition", ""),
                    "score": current_score,
                    "baseline_score": baseline_score,
                    "score_delta": current_score - baseline_score,
                    "is_correct": int(current_correct),
                    "baseline_is_correct": int(baseline_correct),
                    "flip_type": flip_type(baseline_correct, current_correct),
                    "correct_to_incorrect": int(
                        baseline_correct and not current_correct
                    ),
                    "incorrect_to_correct": int(
                        (not baseline_correct) and current_correct
                    ),
                    "answer_changed": int(
                        str(row.get("vlm_output", "")) != str(baseline_output)
                    ),
                    "vlm_output": row.get("vlm_output", ""),
                    "baseline_vlm_output": baseline_output,
                    "ground_truth": row.get("ground_truth", ""),
                }
            )

    return rows


def sort_key(label: str) -> tuple[int, float | str]:
    value = as_float(label)
    if value is None:
        return (1, label)
    return (0, value)


def summarize(rows: list[dict[str, Any]], baseline_alpha: float) -> list[dict[str, Any]]:
    summary = []
    baseline_rows = [
        row for row in rows if as_float(row["alpha"]) == baseline_alpha
    ]
    baseline_accuracy = finite_mean([row["baseline_is_correct"] for row in baseline_rows])
    baseline_score = finite_mean([row["baseline_score"] for row in baseline_rows])

    labels = sorted({str(row["alpha"]) for row in rows}, key=sort_key)
    for label in labels:
        alpha_rows = [row for row in rows if str(row["alpha"]) == label]
        accuracy = finite_mean([row["is_correct"] for row in alpha_rows])
        mean_score = finite_mean([row["score"] for row in alpha_rows])
        summary.append(
            {
                "alpha": label,
                "n_samples": len(alpha_rows),
                "accuracy": accuracy,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_delta": accuracy - baseline_accuracy,
                "mean_score": mean_score,
                "baseline_mean_score": baseline_score,
                "mean_score_delta": mean_score - baseline_score,
                "correct_to_incorrect": sum(
                    row["correct_to_incorrect"] for row in alpha_rows
                ),
                "incorrect_to_correct": sum(
                    row["incorrect_to_correct"] for row in alpha_rows
                ),
                "answer_changed": sum(row["answer_changed"] for row in alpha_rows),
            }
        )
    return summary


def plot_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    numeric = [row for row in summary if as_float(row["alpha"]) is not None]
    if not numeric:
        return

    alphas = [float(row["alpha"]) for row in numeric]
    accuracy = [row["accuracy"] for row in numeric]
    delta = [row["accuracy_delta"] for row in numeric]
    bad_flips = [row["correct_to_incorrect"] for row in numeric]
    good_flips = [row["incorrect_to_correct"] for row in numeric]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    axes[0].plot(alphas, accuracy, marker="o", color="#4878CF")
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("fraction correct")

    axes[1].plot(alphas, delta, marker="o", color="#F28E2B")
    axes[1].axhline(0, color="0.4", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_title("Accuracy delta")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("vs alpha=1")

    x = np.arange(len(alphas))
    axes[2].bar(x - 0.18, bad_flips, width=0.36, label="base correct -> incorrect")
    axes[2].bar(x + 0.18, good_flips, width=0.36, label="base incorrect -> correct")
    axes[2].set_title("Paired flips")
    axes[2].set_xlabel("alpha")
    axes[2].set_ylabel("count")
    axes[2].set_xticks(x, [f"{alpha:g}" for alpha in alphas], rotation=45)
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'alpha':>8}  {'accuracy':>9}  {'delta':>9}  "
        f"{'base T->F':>10}  {'base F->T':>10}  {'changed':>8}  {'samples':>8}"
    )
    print("-" * 78)
    for row in summary:
        print(
            f"{str(row['alpha']):>8}  "
            f"{row['accuracy']:>9.3f}  "
            f"{row['accuracy_delta']:>9.3f}  "
            f"{row['correct_to_incorrect']:>10}  "
            f"{row['incorrect_to_correct']:>10}  "
            f"{row['answer_changed']:>8}  "
            f"{row['n_samples']:>8}"
        )
    print("-" * 78)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize intervention accuracy and paired correctness flips "
            "relative to alpha=1.0."
        )
    )
    parser.add_argument("run_uuid", type=uuid.UUID, help="Run UUID under runs/.")
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--baseline-alpha", type=float, default=BASELINE_ALPHA)
    parser.add_argument(
        "--condition-prefix",
        default="real_alpha",
        help="Only include scored alpha conditions whose name starts with this prefix.",
    )
    parser.add_argument(
        "--include-textonly",
        action="store_true",
        help="Also include the textonly condition in CSV summaries.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path("runs") / str(args.run_uuid)
    scores_path = run_dir / "scores.jsonl"
    if not scores_path.exists():
        raise SystemExit(f"missing {scores_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pair_rows(
        load_jsonl(scores_path),
        baseline_alpha=args.baseline_alpha,
        include_textonly=args.include_textonly,
        condition_prefix=args.condition_prefix,
    )
    if not rows:
        raise SystemExit(f"No paired alpha={args.baseline_alpha:g} scores found")

    summary = summarize(rows, baseline_alpha=args.baseline_alpha)
    pair_csv = args.output_dir / "intervention_accuracy_pairs.csv"
    summary_csv = args.output_dir / "intervention_accuracy_summary.csv"
    figure_path = args.output_dir / "intervention_accuracy_summary.png"

    write_csv(pair_csv, rows)
    write_csv(summary_csv, summary)
    plot_summary(figure_path, summary)
    print_summary(summary)

    n_samples = len({(row["subset"], row["sample_id"]) for row in rows})
    print(f"\nsamples={n_samples}")
    print(f"rows={len(rows)}")
    print(f"saved {pair_csv}")
    print(f"saved {summary_csv}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()

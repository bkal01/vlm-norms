"""Summarize precomputed real/blank/text-only logit comparisons."""

import argparse
import csv
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PRECOMPUTED_FILENAME = "condition_logit_comparisons.jsonl"


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


def finite_mean(values: list[float]) -> float:
    array = np.array(values, dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def finite_max(values: list[float]) -> float:
    array = np.array(values, dtype=float)
    return float(np.nanmax(array)) if np.isfinite(array).any() else float("nan")


def alpha_sort_value(value: Any) -> tuple[int, float | str]:
    if value is None:
        return (1, "")
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["comparison"],
            row["condition"],
            row["reference_condition"],
            row.get("alpha"),
            row.get("reference_alpha"),
        )
        grouped[key].append(row)

    summary = []
    for (
        comparison,
        condition,
        reference_condition,
        alpha,
        reference_alpha,
    ), group_rows in grouped.items():
        summary.append(
            {
                "comparison": comparison,
                "condition": condition,
                "reference_condition": reference_condition,
                "alpha": "" if alpha is None else alpha,
                "reference_alpha": "" if reference_alpha is None else reference_alpha,
                "mean_kl": finite_mean([row["kl"] for row in group_rows]),
                "max_kl": finite_max([row["kl"] for row in group_rows]),
                "greedy_agreement": finite_mean(
                    [row["greedy_agreement"] for row in group_rows]
                ),
                "n_points": len(group_rows),
                "n_samples": len(
                    {(row["subset"], row["sample_id"]) for row in group_rows}
                ),
            }
        )

    return sorted(
        summary,
        key=lambda row: (
            row["comparison"],
            alpha_sort_value(row["alpha"]),
            alpha_sort_value(row["reference_alpha"]),
            row["condition"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'comparison':>26}  {'alpha':>8}  {'ref':>8}  "
        f"{'mean_KL':>12}  {'max_KL':>12}  {'greedy':>8}  {'points':>8}"
    )
    print("-" * 94)
    for row in rows:
        print(
            f"{row['comparison']:>26}  "
            f"{str(row['alpha']):>8}  "
            f"{str(row['reference_alpha']):>8}  "
            f"{row['mean_kl']:>12.6g}  "
            f"{row['max_kl']:>12.6g}  "
            f"{row['greedy_agreement']:>8.3f}  "
            f"{row['n_points']:>8}"
        )
    print("-" * 94)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize condition_logit_comparisons.jsonl."
    )
    parser.add_argument("run_uuid", type=uuid.UUID, help="Run UUID under runs/.")
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path("runs") / str(args.run_uuid)
    path = run_dir / PRECOMPUTED_FILENAME
    if not path.exists():
        raise SystemExit(f"missing {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(load_jsonl(path))
    if not summary:
        raise SystemExit(f"No rows found in {path}")

    csv_path = args.output_dir / "condition_logit_comparison_summary.csv"
    write_csv(csv_path, summary)
    print_summary(summary)
    print(f"\nrows={len(summary)}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()

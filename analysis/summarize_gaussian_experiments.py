#!/usr/bin/env python3
"""Aggregate Gaussian experiment CSV files into Monte Carlo means and errors."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = (
    "exact_recovery",
    "shd",
    "precision",
    "recall",
    "F1",
    "MCC",
    "relative_frobenius_error",
    "heldout_score",
    "runtime_seconds",
    "relative_gap",
    "nodes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_DIR / "experiments_results",
    )
    parser.add_argument("--metric-list", default=",".join(DEFAULT_METRICS))
    parser.add_argument(
        "--group-list",
        default="graph_mode,topology,p,n,target_degree,target_signal,target_condition,method,penalty_multiplier",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = [value.strip() for value in args.metric_list.split(",") if value.strip()]
    groups = [value.strip() for value in args.group_list.split(",") if value.strip()]
    paths = sorted(args.results_dir.glob(f"gaussian_{args.study}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No result files found for study {args.study!r}")

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("status") == "error":
                    continue
                grouped[tuple(row.get(column, "") for column in groups)].append(row)

    columns = groups + ["replications"]
    for metric in metrics:
        columns.extend([f"{metric}_mean", f"{metric}_se"])
    output = args.output or args.results_dir / f"gaussian_{args.study}_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for key in sorted(grouped):
            rows = grouped[key]
            summary = dict(zip(groups, key))
            summary["replications"] = len(rows)
            for metric in metrics:
                values = np.array(
                    [float(row[metric]) for row in rows if row.get(metric, "") != ""],
                    dtype=float,
                )
                values = values[np.isfinite(values)]
                summary[f"{metric}_mean"] = float(values.mean()) if values.size else ""
                summary[f"{metric}_se"] = (
                    float(values.std(ddof=1) / np.sqrt(values.size))
                    if values.size > 1
                    else ""
                )
            writer.writerow(summary)
    print(f"Wrote {output} from {len(paths)} result files")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Average the registered penalty paths and draw the Gaussian ROC plot."""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default=str(
            PROJECT_DIR
            / "experiments_results/gaussian_primary_graph_recovery_roc_*_rep*.csv"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_DIR / "experiments_results/gaussian_roc_summary.csv",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=PROJECT_DIR / "experiments_results/gaussian_roc.png",
    )
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    fits = {}
    for path in sorted(glob.glob(args.input_glob)):
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if (
                    row["stage"] == "evaluation"
                    and row["fit_available"] == "1.0"
                    and int(row["p"]) == args.p
                    and int(row["n"]) == args.n
                ):
                    key = (row["method"], float(row["penalty_constant"]), int(row["rep"]))
                    fits[key] = row

    groups = defaultdict(list)
    for (method, constant, _), row in fits.items():
        groups[(method, constant, row["penalty_rate"])].append(
            (float(row["FPR"]), float(row["TPR"]))
        )

    summary = []
    for (method, constant, rate), values in sorted(groups.items()):
        array = np.asarray(values)
        count = len(array)
        summary.append(
            {
                "method": method,
                "penalty_constant": constant,
                "penalty_rate": rate,
                "replications": count,
                "mean_FPR": array[:, 0].mean(),
                "se_FPR": array[:, 0].std(ddof=1) / np.sqrt(count) if count > 1 else 0.0,
                "mean_TPR": array[:, 1].mean(),
                "se_TPR": array[:, 1].std(ddof=1) / np.sqrt(count) if count > 1 else 0.0,
            }
        )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(summary[0]) if summary else [
        "method", "penalty_constant", "penalty_rate", "replications",
        "mean_FPR", "se_FPR", "mean_TPR", "se_TPR",
    ]
    with args.summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)

    labels = {"sm_l0": "SM–L0", "sm_l0_core": "SM–L0 CORe", "sm_l1": "SM–L1"}
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    for method in sorted({row["method"] for row in summary}):
        points = sorted(
            (row["mean_FPR"], row["mean_TPR"])
            for row in summary
            if row["method"] == method
        )
        fpr, tpr = np.asarray(points).T
        axis.plot(
            fpr,
            tpr,
            marker="o",
            linewidth=2,
            markersize=4,
            label=labels.get(method, method),
        )
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, None),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_path, dpi=200)
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.plot_path}")


if __name__ == "__main__":
    main()

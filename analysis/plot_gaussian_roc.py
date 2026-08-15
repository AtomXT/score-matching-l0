#!/usr/bin/env python3
"""Average the registered penalty paths and draw ROC and PR curves."""

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
GAUSSIAN_RESULTS_ROOT = (
    PROJECT_DIR / "experiments_results" / "gaussian_primary_graph_recovery"
)


def main(
    argv: list[str] | None = None,
    *,
    description: str = __doc__,
    default_n: int = 250,
    default_results_root: Path = GAUSSIAN_RESULTS_ROOT,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--n", type=int, default=default_n)
    parser.add_argument("--topology", default="erdos_renyi")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root,
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Override the configuration-specific input glob.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--pr-plot-path",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)

    results_dir = (
        args.results_root / f"topology={args.topology}_p={args.p}_n={args.n}"
    )
    input_glob = args.input_glob or str(results_dir / "*_rep*.csv")
    summary_csv = args.summary_csv or results_dir / "roc_summary.csv"
    plot_path = args.plot_path or results_dir / "roc.png"
    pr_plot_path = args.pr_plot_path or results_dir / "pr.png"

    fits = {}
    input_paths = sorted(glob.glob(input_glob))
    if not input_paths:
        parser.error(f"no result files matched: {input_glob}")
    for path in input_paths:
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if (
                    row["stage"] == "evaluation"
                    and row["fit_available"] == "1.0"
                    and int(row["p"]) == args.p
                    and int(row["n"]) == args.n
                    and row["topology"] == args.topology
                ):
                    key = (row["method"], float(row["penalty_constant"]), int(row["rep"]))
                    fits[key] = row

    groups = defaultdict(list)
    for (method, constant, _), row in fits.items():
        tp, fp = int(row["TP"]), int(row["FP"])
        precision = tp / (tp + fp) if tp + fp else 1.0
        groups[(method, constant, row["penalty_rate"])].append(
            (float(row["FPR"]), float(row["TPR"]), precision)
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
                "mean_precision": array[:, 2].mean(),
                "se_precision": array[:, 2].std(ddof=1) / np.sqrt(count) if count > 1 else 0.0,
            }
        )
    if not summary:
        parser.error(
            "result files were found, but no available fits matched "
            f"topology={args.topology}, p={args.p}, n={args.n}"
        )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(summary[0]) if summary else [
        "method", "penalty_constant", "penalty_rate", "replications",
        "mean_FPR", "se_FPR", "mean_TPR", "se_TPR",
        "mean_precision", "se_precision",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)

    labels = {
        "sm_l0": "SM–L0",
        "sm_l0_core": "SM–L0 CORe",
        "sm_l0_milp": "SM–L0 support MILP",
        "sm_l1": "SM–L1",
    }
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
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)

    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    for method in sorted({row["method"] for row in summary}):
        points = sorted(
            (row["mean_TPR"], row["mean_precision"])
            for row in summary
            if row["method"] == method
        )
        recall, precision = np.asarray(points).T
        axis.plot(
            recall,
            precision,
            marker="o",
            linewidth=2,
            markersize=4,
            label=labels.get(method, method),
        )
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    pr_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pr_plot_path, dpi=200)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {pr_plot_path}")


if __name__ == "__main__":
    main()

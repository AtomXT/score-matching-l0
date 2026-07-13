#!/usr/bin/env python3
"""Create a compact graph-recovery summary from per-fit result files."""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
OUTPUT_COLUMNS = (
    "topology",
    "p",
    "n",
    "method",
    "penalty_constant",
    "penalty_rate",
    "replications",
    "excluded_fits",
    "uncertified_fits",
    "F1_mean",
    "F1_se",
    "TPR_mean",
    "TPR_se",
    "FPR_mean",
    "FPR_se",
    "exact_recovery_count",
    "shd_mean",
    "shd_se",
)
CERTIFICATION_METHODS = frozenset({"sm_l0", "sm_l0_core", "graphl0"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="primary_graph_recovery")
    parser.add_argument(
        "--stage",
        default="evaluation",
        help="Only summarize rows from this experiment stage (default: evaluation).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_DIR / "experiments_results",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help=(
            "Glob selecting raw CSV files. By default, selects "
            "gaussian_STUDY_*_rep*.csv directly inside --results-dir."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _parse_number(value: str) -> float:
    return float(value)


def _parse_integer(value: str) -> int:
    return int(_parse_number(value))


def fit_is_available(row: dict[str, str]) -> bool:
    """Return whether a fit should enter Monte Carlo performance summaries."""
    recorded = row.get("fit_available", "")
    if recorded != "":
        value = _parse_number(recorded)
        return value == 1.0

    # Compatibility with pilot files written before fit_available was added.
    status = row.get("status", "")
    if status == "error":
        return False
    if row.get("method") in {"sm_l1", "glasso"}:
        return status == "converged"
    if status in {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"}:
        return False
    return status != ""


def fit_is_uncertified(row: dict[str, str], *, fit_available: bool) -> bool:
    """Return whether an available mixed-integer fit is uncertified.

    The continuous methods have no global-optimality certificate, so their
    blank ``certified`` fields correctly contribute zero to this count.
    """
    if not fit_available or row.get("method") not in CERTIFICATION_METHODS:
        return False
    return _parse_number(row["certified"]) == 0.0


def _resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    pattern = args.input_glob
    if pattern is None:
        pattern = str(args.results_dir / f"gaussian_{args.study}_*_rep*.csv")
    return [Path(path) for path in sorted(set(glob.glob(pattern, recursive=True)))]


def _standardize_group(row: dict[str, str]) -> tuple[object, ...]:
    constant = row.get("penalty_constant", "")
    if constant == "" and row.get("penalty_multiplier", "") != "":
        constant = row["penalty_multiplier"]
    topology = row.get("topology", "")
    method = row.get("method", "")
    penalty_rate = row.get("penalty_rate", "")
    return (
        topology,
        _parse_integer(row.get("p", "")),
        _parse_integer(row.get("n", "")),
        method,
        _parse_number(constant),
        penalty_rate,
    )


def _mean_and_se(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se: float | str = ""
    if array.size > 1:
        se = float(array.std(ddof=1) / np.sqrt(array.size))
    return mean, se


def main() -> None:
    args = parse_args()
    paths = _resolve_input_paths(args)
    grouped: dict[tuple[object, ...], list[dict[str, str]]] = defaultdict(list)

    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("stage") != args.stage:
                    continue
                group = _standardize_group(row)
                grouped[group].append(row)

    output = args.output or (
        args.results_dir / f"gaussian_{args.study}_{args.stage}_summary.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for group in sorted(grouped):
            attempted = grouped[group]
            available: list[dict[str, str]] = []
            uncertified_fits = 0
            for row in attempted:
                fit_available = fit_is_available(row)
                if fit_available:
                    available.append(row)
                uncertified_fits += int(
                    fit_is_uncertified(
                        row,
                        fit_available=fit_available,
                    )
                )
            f1_values = [
                _parse_number(row.get("F1", "")) for row in available
            ]
            tpr_values = [
                _parse_number(row.get("TPR", "")) for row in available
            ]
            fpr_values = [
                _parse_number(row.get("FPR", "")) for row in available
            ]
            shd_values = [
                _parse_number(row.get("shd", "")) for row in available
            ]
            exact_values = [
                _parse_number(row.get("exact_recovery", "")) for row in available
            ]
            f1_mean, f1_se = _mean_and_se(f1_values)
            tpr_mean, tpr_se = _mean_and_se(tpr_values)
            fpr_mean, fpr_se = _mean_and_se(fpr_values)
            shd_mean, shd_se = _mean_and_se(shd_values)
            row = dict(zip(OUTPUT_COLUMNS[:6], group))
            row.update(
                {
                    "replications": len(available),
                    "excluded_fits": len(attempted) - len(available),
                    "uncertified_fits": uncertified_fits,
                    "F1_mean": f1_mean,
                    "F1_se": f1_se,
                    "TPR_mean": tpr_mean,
                    "TPR_se": tpr_se,
                    "FPR_mean": fpr_mean,
                    "FPR_se": fpr_se,
                    "exact_recovery_count": int(sum(exact_values)),
                    "shd_mean": shd_mean,
                    "shd_se": shd_se,
                }
            )
            writer.writerow(row)
    print(f"Wrote {output} from {len(paths)} result files")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select one transferable penalty constant per comparison method.

The calibration stage evaluates every registered constant on the smallest
registered problem.  A constant is eligible only when all registered
replications produced usable fits.  Among eligible constants, this script
maximizes mean F1, breaks an exact tie by minimizing mean SHD, and then
prefers the larger constant.

For the two mixed-integer methods, a usable incumbent remains eligible even
when the solver has not certified global optimality.  Such fits are reported
separately as ``uncertified_fits`` so that this choice is visible rather than
silently conflated with certification.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiments.penalty_rates import PENALTY_RATE_LABELS  # noqa: E402
from experiments.primary_graph_recovery_config import (  # noqa: E402
    CALIBRATION_CONFIGURATION,
    METHODS,
    NUMBER_OF_REPLICATIONS,
    PENALTY_CONSTANT_GRID,
)


CALIBRATION_DIR = PROJECT_DIR / "experiments_results" / "penalty_calibration"
DEFAULT_INPUT_GLOB = str(
    CALIBRATION_DIR / "gaussian_primary_graph_recovery_calibration_rep*.csv"
)
CERTIFICATION_METHODS = frozenset({"sm_l0", "graphl0"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob selecting the per-replication calibration CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CALIBRATION_DIR / "selected_constants.json",
        help="JSON file consumed by the evaluation runner.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=CALIBRATION_DIR / "calibration_summary.csv",
        help="Compact CSV summary of every candidate constant.",
    )
    return parser.parse_args()


def _parse_integer(value: str) -> int:
    return int(float(value))


def _parse_number(value: str) -> float:
    return float(value)


def _registered_constant(value: str) -> float:
    parsed = _parse_number(value)
    matches = [
        constant
        for constant in PENALTY_CONSTANT_GRID
        if math.isclose(parsed, constant, rel_tol=1e-9, abs_tol=1e-12)
    ]
    return float(matches[0])


def _fit_is_available(row: dict[str, str]) -> bool:
    return _parse_number(row["fit_available"]) == 1.0


def _fit_is_uncertified(row: dict[str, str], *, fit_available: bool) -> bool:
    """Return whether an available discrete fit lacks a solver certificate.

    Only the mixed-integer methods require a binary ``certified`` value.
    Continuous methods do not have this solver diagnostic and therefore
    contribute zero to the uncertified-fit count.
    """
    if not fit_available or row.get("method") not in CERTIFICATION_METHODS:
        return False
    return _parse_number(row["certified"]) == 0.0


def _read_rows(paths: Iterable[Path]) -> dict[tuple[int, str, float], dict[str, str]]:
    records: dict[tuple[int, str, float], dict[str, str]] = {}

    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                replication = _parse_integer(row.get("rep", ""))
                method = row.get("method", "")
                constant_value = row.get("penalty_constant", "")
                if constant_value == "" and row.get("penalty_multiplier", "") != "":
                    constant_value = row["penalty_multiplier"]
                constant = _registered_constant(constant_value)
                key = (replication, method, constant)
                records[key] = row
    return records


def _candidate_summaries(
    records: dict[tuple[int, str, float], dict[str, str]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for (_, method, constant), row in records.items():
        grouped[(method, constant)].append(row)

    summaries: list[dict[str, object]] = []
    for method in METHODS:
        for constant in PENALTY_CONSTANT_GRID:
            rows = grouped[(method, float(constant))]
            available_rows: list[dict[str, str]] = []
            uncertified_fits = 0
            for row in rows:
                fit_available = _fit_is_available(row)
                if fit_available:
                    available_rows.append(row)
                uncertified_fits += int(
                    _fit_is_uncertified(
                        row,
                        fit_available=fit_available,
                    )
                )
            f1_values: list[float] = []
            shd_values: list[float] = []
            for row in available_rows:
                f1_values.append(_parse_number(row.get("F1", "")))
                shd_values.append(_parse_number(row.get("shd", "")))
            summaries.append(
                {
                    "method": method,
                    "penalty_rate": PENALTY_RATE_LABELS[method],
                    "penalty_constant": float(constant),
                    "replications": len(rows),
                    "available_fits": len(available_rows),
                    "uncertified_fits": uncertified_fits,
                    "F1_mean": float(np.mean(f1_values)) if f1_values else None,
                    "shd_mean": float(np.mean(shd_values)) if shd_values else None,
                    "selected": 0,
                }
            )
    return summaries


def _select(summaries: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    grid_min = float(min(PENALTY_CONSTANT_GRID))
    grid_max = float(max(PENALTY_CONSTANT_GRID))
    for method in METHODS:
        eligible = [
            summary
            for summary in summaries
            if summary["method"] == method
            and summary["available_fits"] == NUMBER_OF_REPLICATIONS
        ]
        winner = max(
            eligible,
            key=lambda summary: (
                float(summary["F1_mean"]),
                -float(summary["shd_mean"]),
                float(summary["penalty_constant"]),
            ),
        )
        winner["selected"] = 1
        constant = float(winner["penalty_constant"])
        at_boundary = constant in (grid_min, grid_max)
        if at_boundary:
            warnings.warn(
                f"selected {method} constant {constant:g} lies on the calibration-grid "
                "boundary; consider widening the grid before the final study",
                stacklevel=2,
            )
        selected[method] = {
            "constant": constant,
            "penalty_rate": PENALTY_RATE_LABELS[method],
            "mean_F1": float(winner["F1_mean"]),
            "mean_SHD": float(winner["shd_mean"]),
            "available_fits": int(winner["available_fits"]),
            "uncertified_fits": int(winner["uncertified_fits"]),
            "selected_at_grid_boundary": at_boundary,
        }
    return selected


def _write_summary(path: Path, summaries: list[dict[str, object]]) -> None:
    columns = (
        "method",
        "penalty_rate",
        "penalty_constant",
        "replications",
        "available_fits",
        "uncertified_fits",
        "F1_mean",
        "shd_mean",
        "selected",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in sorted(set(glob.glob(args.input_glob)))]

    records = _read_rows(paths)
    summaries = _candidate_summaries(records)
    selected = _select(summaries)

    topology, p, n = CALIBRATION_CONFIGURATION
    document = {
        "schema_version": 1,
        "calibration": {
            "topology": topology,
            "p": p,
            "n": n,
            "replications": list(range(NUMBER_OF_REPLICATIONS)),
            "constant_grid": list(PENALTY_CONSTANT_GRID),
            "criterion": (
                "maximize mean F1; break ties by lower mean SHD, then larger constant"
            ),
            "eligibility": (
                "all replications must have usable fits; available mixed-integer "
                "incumbents remain eligible when global optimality is uncertified"
            ),
        },
        "methods": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    _write_summary(args.summary_output, summaries)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()

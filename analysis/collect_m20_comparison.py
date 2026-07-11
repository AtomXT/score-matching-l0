"""Collect m=20 exact-dataset experiment results into one comparison CSV."""

from __future__ import annotations

import csv
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(current_dir)

SOURCES = (
    ("GraphL0BnB", "l0", "results/graphl0bnb/m20_exact_grid.csv"),
    ("ScoreMatchingMIQP", "lambda", "results/score_matching_miqp/m20_exact_grid.csv"),
    ("ScoreMatchingCORe", "lambda", "results/score_matching_core/m20_exact_grid.csv"),
)

OUTPUT = os.path.join(current_dir, "m20_exact_comparison.csv")

FIELDNAMES = [
    "method",
    "dataset_dir",
    "dataset_type",
    "n",
    "m",
    "target_edges",
    "seed",
    "regularization_name",
    "regularization_value",
    "l2",
    "big_m_scale",
    "time_limit",
    "status",
    "error_message",
    "runtime_seconds",
    "solver_time_seconds",
    "objective",
    "objective_bound",
    "gap",
    "mip_gap",
    "mip_gap_target",
    "nodes",
    "big_m_min",
    "big_m_max",
    "kappa_min",
    "kappa_max",
    "tau_min",
    "tau_max",
    "selected_edges",
    "true_edges",
    "TP",
    "FP",
    "TN",
    "FN",
    "TPR",
    "FPR",
    "precision",
    "recall",
    "F1",
]


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def comparison_row(method: str, regularization_name: str, row: dict[str, str]) -> dict[str, str]:
    return {
        "method": method,
        "regularization_name": regularization_name,
        "regularization_value": row.get(regularization_name, ""),
        **{field: row.get(field, "") for field in FIELDNAMES if field not in {
            "method",
            "regularization_name",
            "regularization_value",
        }},
    }


def main() -> None:
    collected = []
    for method, regularization_name, relative_path in SOURCES:
        path = os.path.join(PROJECT_DIR, relative_path)
        for row in read_rows(path):
            collected.append(comparison_row(method, regularization_name, row))

    with open(OUTPUT, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(collected)

    print(f"Wrote {OUTPUT} with {len(collected)} rows")


if __name__ == "__main__":
    main()

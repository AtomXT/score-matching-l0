"""Run GraphL0Learn/GraphL0BnB on a saved Gaussian dataset."""

from __future__ import annotations

import argparse
import traceback
import os
from pathlib import Path

import numpy as np

from project_paths import PROJECT_DIR, load_src_module

utils = load_src_module("utils")

add_dataset_runner_arguments = utils.add_dataset_runner_arguments
append_csv_row = utils.append_csv_row
fit_graph_l0_bnb = utils.fit_graph_l0_bnb
load_dataset_from_runner_args = utils.load_dataset_from_runner_args
normalize_prediction = utils.normalize_prediction
parse_bool = utils.parse_bool
parse_float_values = utils.parse_float_values
prepare_results_file = utils.prepare_results_file
result_row = utils.result_row
support_metrics = utils.support_metrics

SOLVER_COLUMNS = ("runtime_seconds", "solver_time_seconds", "objective", "gap", "nodes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_runner_arguments(
        parser,
        Path(os.path.join(PROJECT_DIR, "results", "graphl0bnb", "pycharm_default.csv")),
    )
    parser.add_argument("--l0", type=float, default=0.02)
    parser.add_argument("--l0-values", type=str, default=None)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--gap-tol", type=float, default=0.05)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--verbose", type=parse_bool, default=True)
    return parser.parse_args()


def run_one(args: argparse.Namespace, dataset, l0: float) -> dict:
    x = dataset.data["X"]
    true_precision = dataset.data["precision"]
    true_adjacency = dataset.data["adjacency"]
    m_bound = float(np.max(np.abs(true_precision)))
    model_info = {
        "l0": l0,
        "l2": args.l2,
        "M": m_bound,
        "gap_tol": args.gap_tol,
        "time_limit": args.time_limit,
    }

    try:
        fit = fit_graph_l0_bnb(
            x,
            l0=l0,
            l2=args.l2,
            m_bound=m_bound,
            gap_tol=args.gap_tol,
            time_limit=args.time_limit,
            verbose=args.verbose,
        )
        return result_row(
            dataset,
            blank_columns=SOLVER_COLUMNS,
            **model_info,
            status="ok",
            runtime_seconds=fit["runtime_seconds"],
            solver_time_seconds=fit["solver_time_seconds"],
            objective=fit["objective"],
            gap=fit["gap"],
            nodes=fit["nodes"],
            **support_metrics(true_adjacency, normalize_prediction(fit["Theta"])),
        )
    except Exception as exc:
        if args.verbose:
            traceback.print_exc()
        return result_row(
            dataset,
            blank_columns=SOLVER_COLUMNS,
            **model_info,
            error_message=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    args = parse_args()
    dataset = load_dataset_from_runner_args(args)
    prepare_results_file(args.results_csv, args.overwrite_results)

    statuses = []
    for l0 in parse_float_values(args.l0_values, args.l0):
        row = run_one(args, dataset, l0)
        append_csv_row(args.results_csv, row)
        statuses.append(row["status"])
        print(f"l0={l0}: {row['status']}")
        if row["error_message"]:
            print(row["error_message"])

    print(f"Wrote {args.results_csv}")
    if any(status != "ok" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

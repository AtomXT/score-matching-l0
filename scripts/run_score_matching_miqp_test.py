"""Run Gaussian score-matching MIQP on a saved dataset."""

from __future__ import annotations

import argparse
import traceback
import os
from pathlib import Path

from project_paths import PROJECT_DIR, load_src_module

score_matching_miqp = load_src_module("score_matching_miqp")
utils = load_src_module("utils")

solve_score_matching_miqp = score_matching_miqp.solve_score_matching_miqp
add_dataset_runner_arguments = utils.add_dataset_runner_arguments
append_csv_row = utils.append_csv_row
load_dataset_from_runner_args = utils.load_dataset_from_runner_args
parse_bool = utils.parse_bool
parse_float_values = utils.parse_float_values
prepare_results_file = utils.prepare_results_file
result_row = utils.result_row
support_metrics = utils.support_metrics

SOLVER_COLUMNS = (
    "big_m_min",
    "big_m_max",
    "runtime_seconds",
    "objective",
    "objective_bound",
    "mip_gap",
    "nodes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_runner_arguments(
        parser,
        Path(os.path.join(PROJECT_DIR, "results", "score_matching_miqp", "pycharm_default.csv")),
    )
    parser.add_argument("--lambda-value", type=float, default=0.012)
    parser.add_argument("--lambda-values", type=str, default=None)
    parser.add_argument("--big-m-scale", type=float, default=1.25)
    parser.add_argument("--mip-gap", type=float, default=0.05)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--verbose", type=parse_bool, default=True)
    return parser.parse_args()


def run_one(args: argparse.Namespace, dataset, lambda_value: float) -> dict:
    model_info = {
        "lambda": lambda_value,
        "big_m_scale": args.big_m_scale,
        "time_limit": args.time_limit,
        "mip_gap_target": args.mip_gap,
    }

    try:
        solution = solve_score_matching_miqp(
            dataset.data["X"],
            lambda_value=lambda_value,
            big_m_scale=args.big_m_scale,
            mip_gap=args.mip_gap,
            time_limit=args.time_limit,
            output_flag=args.verbose,
        )
        metrics = (
            support_metrics(dataset.data["adjacency"], solution.adjacency)
            if solution.has_solution
            else {}
        )
        return result_row(
            dataset,
            blank_columns=SOLVER_COLUMNS,
            **model_info,
            **metrics,
            status=solution.status,
            big_m_min=float(solution.big_m.values.min()),
            big_m_max=float(solution.big_m.values.max()),
            runtime_seconds=solution.runtime_seconds,
            objective=solution.objective,
            objective_bound=solution.objective_bound,
            mip_gap=solution.mip_gap,
            nodes=solution.nodes,
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
    for lambda_value in parse_float_values(args.lambda_values, args.lambda_value):
        row = run_one(args, dataset, lambda_value)
        append_csv_row(args.results_csv, row)
        statuses.append(row["status"])
        print(f"lambda={lambda_value}: {row['status']}")
        if row["error_message"]:
            print(row["error_message"])

    print(f"Wrote {args.results_csv}")
    if any(status == "error" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

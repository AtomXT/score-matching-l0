#!/usr/bin/env python3
"""Run Gaussian score-matching methods on saved experiment instances.

The driver deliberately does not generate data.  It loads the instances created
by ``generate_gaussian_experiments.py``, applies one common candidate-edge set to
all compatible methods, and appends one CSV row after every method--penalty fit.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import (
    append_result,
    correlation_screen,
    estimation_metrics,
    heldout_scores,
    load_instance,
    parse_list,
    support_metrics,
)
from scripts.project_paths import load_src_module


PROJECT_DIR = Path(__file__).resolve().parents[1]
load_src_module("score_matching_miqp")
score_matching_miqp = load_src_module("score_matching_miqp")
score_matching_core = load_src_module("score_matching_core_miqp")
score_matching_l1 = load_src_module("score_matching_l1")
experiment_utils = load_src_module("utils")


RESULT_COLUMNS = [
    "study",
    "job_name",
    "instance_dir",
    "topology",
    "p",
    "n",
    "target_degree",
    "achieved_average_degree",
    "achieved_maximum_degree",
    "target_signal",
    "achieved_signal",
    "target_condition",
    "achieved_condition",
    "condition_before_standardization",
    "signal_calibration_error",
    "minimum_eigenvalue",
    "rep",
    "graph_mode",
    "graph_rep",
    "graph_seed",
    "sample_seed",
    "method",
    "python_version",
    "numpy_version",
    "gurobi_version",
    "host",
    "slurm_job_id",
    "slurm_array_task_id",
    "threads",
    "time_limit",
    "mip_gap_target",
    "candidate_rule",
    "candidate_edges",
    "candidate_recall",
    "penalty_multiplier",
    "penalty_rate",
    "lambda",
    "status",
    "certified",
    "runtime_seconds",
    "objective",
    "objective_bound",
    "absolute_gap",
    "relative_gap",
    "nodes",
    "big_m_min",
    "big_m_max",
    "iterations",
    "TP",
    "FP",
    "TN",
    "FN",
    "selected_edges",
    "true_edges",
    "exact_recovery",
    "shd",
    "TPR",
    "FPR",
    "precision",
    "recall",
    "F1",
    "MCC",
    "relative_frobenius_error",
    "operator_error",
    "max_entry_error",
    "heldout_score",
    "heldout_gaussian_nll",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--job-name", default="local_test")
    parser.add_argument("--method-list", default="sm_l0,sm_l0_core,sm_l1")
    parser.add_argument("--penalty-multiplier-list", default="0.25,0.5,1,2,4")
    parser.add_argument("--rep-list", default=None)
    parser.add_argument("--candidate-rule", choices=["complete", "correlation"], default="complete")
    parser.add_argument("--screen-size", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--big-m-scale", type=float, default=1.25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--l1-max-iter", type=int, default=20_000)
    parser.add_argument("--l1-tolerance", type=float, default=1e-8)
    parser.add_argument("--graphl0-l2", type=float, default=0.05)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_DIR / "data" / "gaussian_experiments",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def environment_record(args: argparse.Namespace) -> dict[str, Any]:
    """Record the software and compute environment attached to every result row."""
    try:
        import gurobipy as gp

        gurobi_version = ".".join(map(str, gp.gurobi.version()))
    except Exception:
        gurobi_version = "unavailable"
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "gurobi_version": gurobi_version,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "threads": args.threads,
        "time_limit": args.time_limit,
        "mip_gap_target": args.mip_gap,
    }


def candidate_edges(
    x: np.ndarray,
    truth: np.ndarray,
    *,
    rule: str,
    screen_size: int | None,
) -> tuple[list[tuple[int, int]], float]:
    """Construct one candidate set and report the fraction of true edges kept."""
    p = x.shape[1]
    total = p * (p - 1) // 2
    if rule == "complete":
        edges = [(i, j) for i in range(p) for j in range(i + 1, p)]
    else:
        if screen_size is None:
            raise ValueError("--screen-size is required for correlation screening")
        edges = correlation_screen(x, min(screen_size, total))
    true_edges = {(i, j) for i in range(p) for j in range(i + 1, p) if truth[i, j]}
    retained = len(true_edges.intersection(edges))
    recall = retained / len(true_edges) if true_edges else 1.0
    return edges, recall


def penalty_value(method: str, multiplier: float, r: int, n: int) -> tuple[float, str]:
    """Return the theory-motivated penalty scale for a method.

    Subset selection uses a squared-noise penalty of order log(r) / n, whereas
    the coefficientwise L1 penalty is on the score scale sqrt(log(r) / n).
    """
    if method == "sm_l1":
        return multiplier * math.sqrt(math.log(r) / n), "sqrt(log(r)/n)"
    return multiplier * math.log(r) / n, "log(r)/n"


def _solution_metrics(
    arrays: dict[str, np.ndarray],
    adjacency: np.ndarray,
    precision: np.ndarray,
) -> dict[str, float]:
    return {
        **support_metrics(arrays["adjacency"], adjacency),
        **estimation_metrics(arrays["precision"], precision),
        **heldout_scores(arrays["X_test"], precision),
    }


def fit_method(
    method: str,
    arrays: dict[str, np.ndarray],
    edges: list[tuple[int, int]],
    lambda_value: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fit one method and normalize its diagnostics to the common result schema."""
    x = arrays["X_train"]
    if method == "sm_l0":
        solution = score_matching_miqp.solve_score_matching_miqp(
            x,
            lambda_value=lambda_value,
            big_m_scale=args.big_m_scale,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_flag=args.verbose,
            edge_list=edges,
            threads=args.threads,
        )
        absolute_gap = solution.objective - solution.objective_bound
        return {
            "status": solution.status,
            "certified": float(solution.has_solution and solution.mip_gap <= args.mip_gap),
            "runtime_seconds": solution.runtime_seconds,
            "objective": solution.objective,
            "objective_bound": solution.objective_bound,
            "absolute_gap": absolute_gap,
            "relative_gap": solution.mip_gap,
            "nodes": solution.nodes,
            "big_m_min": float(solution.big_m.values.min()),
            "big_m_max": float(solution.big_m.values.max()),
            **_solution_metrics(arrays, solution.adjacency, solution.precision),
        }
    if method == "sm_l0_core":
        solution = score_matching_core.solve_score_matching_core_miqp(
            x,
            lambda_value=lambda_value,
            big_m_scale=args.big_m_scale,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_flag=args.verbose,
            edge_list=edges,
            threads=args.threads,
        )
        absolute_gap = solution.objective - solution.objective_bound
        return {
            "status": solution.status,
            "certified": float(solution.has_solution and solution.mip_gap <= args.mip_gap),
            "runtime_seconds": solution.runtime_seconds,
            "objective": solution.objective,
            "objective_bound": solution.objective_bound,
            "absolute_gap": absolute_gap,
            "relative_gap": solution.mip_gap,
            "nodes": solution.nodes,
            "big_m_min": float(solution.big_m.values.min()),
            "big_m_max": float(solution.big_m.values.max()),
            **_solution_metrics(arrays, solution.adjacency, solution.precision),
        }
    if method == "sm_l1":
        start = time.perf_counter()
        solution = score_matching_l1.solve_score_matching_l1(
            x,
            lambda_value=lambda_value,
            edge_list=edges,
            max_iter=args.l1_max_iter,
            tolerance=args.l1_tolerance,
        )
        runtime = time.perf_counter() - start
        return {
            "status": solution.status,
            "certified": "",
            "runtime_seconds": runtime,
            "objective": solution.objective,
            "iterations": solution.iterations,
            **_solution_metrics(arrays, solution.adjacency, solution.precision),
        }
    if method == "graphl0":
        p = x.shape[1]
        if len(edges) != p * (p - 1) // 2:
            raise ValueError("GraphL0 does not support the common screened edge set")
        m_bound = float(np.max(np.abs(arrays["precision"])))
        fit = experiment_utils.fit_graph_l0_bnb(
            x,
            l0=lambda_value,
            l2=args.graphl0_l2,
            m_bound=m_bound,
            gap_tol=args.mip_gap,
            time_limit=args.time_limit,
            verbose=args.verbose,
        )
        precision = np.asarray(fit["Theta"], dtype=float)
        adjacency = np.abs(precision) > 1e-8
        np.fill_diagonal(adjacency, False)
        return {
            "status": "ok",
            "certified": float(fit["gap"] <= args.mip_gap),
            "runtime_seconds": fit["runtime_seconds"],
            "objective": fit["objective"],
            "relative_gap": fit["gap"],
            "nodes": fit["nodes"],
            **_solution_metrics(arrays, adjacency, precision),
        }
    raise ValueError(f"unsupported method: {method}")


def main() -> None:
    args = parse_args()
    methods = parse_list(args.method_list, str)
    multipliers = parse_list(args.penalty_multiplier_list, float)
    requested_reps = set(parse_list(args.rep_list, int)) if args.rep_list else None
    results_csv = args.results_csv or (
        PROJECT_DIR / "experiments_results" / f"gaussian_{args.study}_{args.job_name}.csv"
    )
    if args.overwrite_results and results_csv.exists():
        results_csv.unlink()

    instance_dirs = sorted((args.data_root / args.study).glob("*/dataset.npz"))
    if not instance_dirs:
        raise FileNotFoundError(
            f"No instances found under {args.data_root / args.study}. "
            "Run experiments.generate_gaussian_experiments first."
        )

    failures = 0
    environment = environment_record(args)
    for dataset_path in instance_dirs:
        arrays, metadata = load_instance(dataset_path.parent)
        if requested_reps is not None and int(metadata["rep"]) not in requested_reps:
            continue
        edges, screen_recall = candidate_edges(
            arrays["X_train"],
            arrays["adjacency"],
            rule=args.candidate_rule,
            screen_size=args.screen_size,
        )
        r = max(2, len(edges))
        for multiplier in multipliers:
            for method in methods:
                lambda_value, penalty_rate = penalty_value(
                    method,
                    multiplier,
                    r,
                    int(metadata["n"]),
                )
                row: dict[str, Any] = {
                    **metadata,
                    **environment,
                    "job_name": args.job_name,
                    "instance_dir": str(dataset_path.parent),
                    "method": method,
                    "candidate_rule": args.candidate_rule,
                    "candidate_edges": len(edges),
                    "candidate_recall": screen_recall,
                    "penalty_multiplier": multiplier,
                    "penalty_rate": penalty_rate,
                    "lambda": lambda_value,
                    "status": "error",
                    "error_message": "",
                }
                try:
                    row.update(fit_method(method, arrays, edges, lambda_value, args))
                except Exception as exc:
                    failures += 1
                    row["error_message"] = f"{type(exc).__name__}: {exc}"
                    if args.verbose:
                        traceback.print_exc()
                append_result(results_csv, row, RESULT_COLUMNS)
                print(
                    f"{metadata['topology']} p={metadata['p']} n={metadata['n']} "
                    f"rep={metadata['rep']} method={method} c={multiplier:g}: {row['status']}"
                )

    print(f"Wrote {results_csv}")
    if failures:
        raise SystemExit(f"{failures} fits failed; see error_message in the result file")


if __name__ == "__main__":
    main()

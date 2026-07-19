#!/usr/bin/env python3
"""Run Gaussian graph-recovery methods on saved experiment instances.

This driver never generates data.  It fits every requested method and penalty
constant directly supplied on the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import (
    append_result,
    graphical_lasso_screen,
    load_instance,
    parse_list,
    support_metrics,
)
from experiments.penalty_rates import (
    PENALTY_RATE_LABELS,
    penalty_value,
)
from src import score_matching_core_miqp, score_matching_l1, score_matching_miqp


PROJECT_DIR = Path(__file__).resolve().parents[1]


RESULT_COLUMNS = [
    "stage",
    "study",
    "job_name",
    "instance_id",
    "topology",
    "p",
    "n",
    "rep",
    "method",
    "penalty_constant",
    "penalty_rate",
    "lambda",
    "status",
    "fit_available",
    "certified",
    "runtime_seconds",
    "UB",
    "LB",
    "gap",
    "TP",
    "FP",
    "FN",
    "TPR",
    "FPR",
    "F1",
    "exact_recovery",
    "shd",
    "error_message",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--job-name", default="local_test")
    parser.add_argument(
        "--stage",
        choices=["evaluation", "local_check"],
        default="evaluation",
    )
    parser.add_argument(
        "--method-list",
        default="sm_l0,sm_l1",
        help="Methods: sm_l0, sm_l0_core, sm_l1, graphl0, or glasso.",
    )
    parser.add_argument(
        "--penalty-constant-list",
        default="1",
        help="Comma-separated constants applied to every requested method.",
    )
    parser.add_argument("--rep-list", default=None)
    parser.add_argument("--topology-list", default=None)
    parser.add_argument("--p-list", default=None)
    parser.add_argument("--n-list", default=None)
    parser.add_argument(
        "--configuration-list",
        default=None,
        help="Semicolon-separated topology:p:n triples used as exact filters.",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "--candidate-rule",
        choices=["complete", "graphical_lasso"],
        default="complete",
    )
    parser.add_argument(
        "--screen-alpha",
        type=float,
        default=0.01,
        help="Graphical lasso penalty used only to screen candidate edges.",
    )
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--big-m-init", type=float, default=1000.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--l1-max-iter", type=int, default=5_000)
    parser.add_argument("--l1-tolerance", type=float, default=1e-6)
    parser.add_argument("--l1-support-tolerance", type=float, default=1e-6)
    parser.add_argument("--graphl0-l2", type=float, default=0.05)
    parser.add_argument("--graphl0-m-bound", type=float, default=100.0)
    parser.add_argument("--glasso-max-iter", type=int, default=1_000)
    parser.add_argument("--glasso-tolerance", type=float, default=1e-4)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_DIR / "data" / "gaussian_experiments",
    )
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument(
        "--diagnostics-jsonl",
        type=Path,
        default=None,
        help="Optional path for per-fit solver diagnostics.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Optional path for the run arguments and compute environment.",
    )
    return parser.parse_args(argv)


def environment_record(args: argparse.Namespace) -> dict[str, Any]:
    """Return software and scheduler information recorded once per run."""
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
    screen_alpha: float,
    screen_max_iter: int,
    screen_tolerance: float,
) -> tuple[list[tuple[int, int]], float]:
    """Construct one candidate set and report the fraction of true edges kept."""
    p = x.shape[1]
    if rule == "complete":
        edges = [(i, j) for i in range(p) for j in range(i + 1, p)]
    else:
        edges = graphical_lasso_screen(
            x,
            alpha=screen_alpha,
            max_iter=screen_max_iter,
            tolerance=screen_tolerance,
        )
    true_edges = {(i, j) for i in range(p) for j in range(i + 1, p) if truth[i, j]}
    retained = len(true_edges.intersection(edges))
    recall = retained / len(true_edges) if true_edges else 1.0
    return edges, recall


def _support_metrics(
    arrays: dict[str, np.ndarray],
    adjacency: np.ndarray,
) -> dict[str, float]:
    """Return only the graph-recovery measures used in this experiment."""
    return support_metrics(arrays["adjacency"], adjacency)


def standardize_instance(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Standardize the observations columnwise."""
    x = np.asarray(arrays["X"], dtype=float)
    location = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)

    transformed = dict(arrays)
    transformed["X"] = (x - location) / scale
    return transformed


def fit_method(
    method: str,
    arrays: dict[str, np.ndarray],
    edges: list[tuple[int, int]],
    lambda_value: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fit one method and return its graph metrics and solver diagnostics."""
    x = arrays["X"]
    if method in {"sm_l0", "sm_l0_core"}:
        solver = (
            score_matching_core_miqp.solve_score_matching_core_miqp
            if method == "sm_l0_core"
            else score_matching_miqp.solve_score_matching_miqp
        )
        solution = solver(
            x,
            lambda_value=lambda_value,
            big_m_init=args.big_m_init,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_flag=args.verbose,
            edge_list=edges,
            threads=args.threads,
        )
        absolute_gap = solution["objective"] - solution["objective_bound"]
        return {
            "status": solution["status"],
            "fit_available": float(solution["has_solution"]),
            "certified": float(
                solution["has_solution"] and solution["mip_gap"] <= args.mip_gap
            ),
            "runtime_seconds": solution["runtime_seconds"],
            "formulation": "core" if method == "sm_l0_core" else "standard",
            "UB": solution["objective"],
            "LB": solution["objective_bound"],
            "gap": solution["mip_gap"],
            "objective": solution["objective"],
            "objective_bound": solution["objective_bound"],
            "absolute_gap": absolute_gap,
            "relative_gap": solution["mip_gap"],
            "nodes": solution["nodes"],
            "big_m_initial": args.big_m_init,
            "big_m": solution["big_m"],
            "big_m_relaxation_objective": solution[
                "big_m_relaxation_objective"
            ],
            "big_m_relaxation_runtime_seconds": solution[
                "big_m_relaxation_runtime_seconds"
            ],
            **(
                _support_metrics(arrays, solution["adjacency"])
                if solution["has_solution"]
                else {}
            ),
        }
    if method == "sm_l1":
        start = time.perf_counter()
        solution = score_matching_l1.solve_score_matching_l1(
            x,
            lambda_value=lambda_value,
            edge_list=edges,
            assume_centered=True,
            max_iter=args.l1_max_iter,
            tolerance=args.l1_tolerance,
            support_tolerance=args.l1_support_tolerance,
            verbose=args.verbose,
        )
        runtime = time.perf_counter() - start
        metrics = (
            _support_metrics(arrays, solution["adjacency"])
            if solution["converged"]
            else {}
        )
        return {
            "status": solution["status"],
            "fit_available": float(solution["converged"]),
            "certified": "",
            "runtime_seconds": runtime,
            "objective": solution["objective"],
            "iterations": solution["iterations"],
            "convergence_metric": "full_sweep_l1_change",
            "convergence_value": solution["sweep_change"],
            **metrics,
        }
    if method == "graphl0":
        from src.graphl0_adapter import fit_graph_l0_bnb

        fit = fit_graph_l0_bnb(
            x,
            l0=lambda_value,
            l2=args.graphl0_l2,
            m_bound=args.graphl0_m_bound,
            gap_tol=args.mip_gap,
            time_limit=args.time_limit,
            verbose=args.verbose,
        )
        precision = np.asarray(fit["Theta"], dtype=float)
        adjacency = np.abs(precision) > 1e-8
        np.fill_diagonal(adjacency, False)
        return {
            "status": "ok",
            "fit_available": 1.0,
            "certified": float(fit["gap"] <= args.mip_gap),
            "runtime_seconds": fit["runtime_seconds"],
            "objective": fit["objective"],
            "relative_gap": fit["gap"],
            "nodes": fit["nodes"],
            **_support_metrics(arrays, adjacency),
        }
    if method == "glasso":
        from sklearn.covariance import graphical_lasso

        start = time.perf_counter()
        centered = x - x.mean(axis=0, keepdims=True)
        sample_covariance = centered.T @ centered / centered.shape[0]
        _, precision, costs = graphical_lasso(
            sample_covariance,
            alpha=lambda_value,
            max_iter=args.glasso_max_iter,
            tol=args.glasso_tolerance,
            verbose=args.verbose,
            return_costs=True,
        )
        runtime = time.perf_counter() - start
        objective, dual_gap = costs[-1]
        dual_gap = float(dual_gap)
        converged = abs(dual_gap) <= args.glasso_tolerance
        adjacency = np.abs(precision) > 1e-7
        np.fill_diagonal(adjacency, False)
        metrics = _support_metrics(arrays, adjacency) if converged else {}
        return {
            "status": "converged" if converged else "iteration_limit",
            "fit_available": float(converged),
            "certified": "",
            "runtime_seconds": runtime,
            "objective": float(objective),
            "iterations": len(costs),
            "convergence_metric": "dual_gap",
            "convergence_value": dual_gap,
            **metrics,
        }


def _json_ready(value: Any) -> Any:
    """Convert paths and NumPy scalars to strict JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(document), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_ready(record), sort_keys=True, allow_nan=False) + "\n")


def _requested_configurations(text: str | None) -> set[tuple[str, int, int]] | None:
    if text is None:
        return None
    configurations: set[tuple[str, int, int]] = set()
    for specification in text.split(";"):
        topology, p_text, n_text = specification.split(":")
        configurations.add((topology, int(p_text), int(n_text)))
    return configurations


def run(args: argparse.Namespace) -> Path:
    """Run the requested fits and return the compact result-file path."""
    methods = parse_list(args.method_list, str)
    constants = parse_list(args.penalty_constant_list, float)
    fit_plan = [(method, constant) for method in methods for constant in constants]
    requested_rep_list = parse_list(args.rep_list, int) if args.rep_list else None
    requested_reps = set(requested_rep_list) if requested_rep_list is not None else None
    requested_topologies = (
        set(parse_list(args.topology_list, str)) if args.topology_list else None
    )
    requested_p = set(parse_list(args.p_list, int)) if args.p_list else None
    requested_n = set(parse_list(args.n_list, int)) if args.n_list else None
    requested_configurations = _requested_configurations(args.configuration_list)

    results_csv = args.results_csv
    if results_csv is None:
        if requested_configurations is not None and len(requested_configurations) == 1:
            topology, p, n = next(iter(requested_configurations))
            configuration_tag = f"topology={topology}_p={p}_n={n}"
        elif (
            requested_topologies is not None
            and len(requested_topologies) == 1
            and requested_p is not None
            and len(requested_p) == 1
            and requested_n is not None
            and len(requested_n) == 1
        ):
            configuration_tag = (
                f"topology={next(iter(requested_topologies))}_"
                f"p={next(iter(requested_p))}_n={next(iter(requested_n))}"
            )
        else:
            configuration_tag = "mixed_configurations"
        method_tag = "-".join(methods)
        replication_tag = (
            "-".join(map(str, requested_rep_list))
            if requested_rep_list is not None
            else "all"
        )
        results_csv = (
            PROJECT_DIR
            / "experiments_results"
            / f"gaussian_{args.study}"
            / configuration_tag
            / f"{args.job_name}_{method_tag}_rep{replication_tag}.csv"
        )
    diagnostics_jsonl = args.diagnostics_jsonl or results_csv.with_suffix(
        ".diagnostics.jsonl"
    )
    run_manifest = args.run_manifest or results_csv.with_suffix(".run.json")

    dataset_paths = sorted((args.data_root / args.study).glob("*/dataset.npz"))

    selected_instances: list[tuple[Path, dict[str, Any]]] = []
    for dataset_path in dataset_paths:
        _, metadata = load_instance(dataset_path.parent)
        if requested_reps is not None and int(metadata["rep"]) not in requested_reps:
            continue
        if requested_topologies is not None and metadata["topology"] not in requested_topologies:
            continue
        if requested_p is not None and int(metadata["p"]) not in requested_p:
            continue
        if requested_n is not None and int(metadata["n"]) not in requested_n:
            continue
        configuration = (
            str(metadata["topology"]),
            int(metadata["p"]),
            int(metadata["n"]),
        )
        if requested_configurations is not None and configuration not in requested_configurations:
            continue
        selected_instances.append((dataset_path, metadata))
    if args.max_instances is not None:
        selected_instances = selected_instances[: args.max_instances]

    if args.overwrite_results:
        for path in (results_csv, diagnostics_jsonl, run_manifest):
            if path.exists():
                path.unlink()

    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at,
        "stage": args.stage,
        "study": args.study,
        "job_name": args.job_name,
        "arguments": vars(args),
        "environment": environment_record(args),
        "fit_plan": [
            {
                "method": method,
                "constant": constant,
                "penalty_rate": PENALTY_RATE_LABELS[method],
            }
            for method, constant in fit_plan
        ],
        "instance_ids": [dataset_path.parent.name for dataset_path, _ in selected_instances],
        "results_csv": results_csv,
        "diagnostics_jsonl": diagnostics_jsonl,
    }
    _write_json(run_manifest, manifest)

    failures = 0
    fits_attempted = 0
    for dataset_path, metadata in selected_instances:
        arrays, _ = load_instance(dataset_path.parent)
        arrays = standardize_instance(arrays)
        edges, screen_recall = candidate_edges(
            arrays["X"],
            arrays["adjacency"],
            rule=args.candidate_rule,
            screen_alpha=args.screen_alpha,
            screen_max_iter=args.glasso_max_iter,
            screen_tolerance=args.glasso_tolerance,
        )
        for method, constant in fit_plan:
            lambda_value, penalty_rate = penalty_value(
                method,
                constant,
                int(metadata["p"]),
                int(metadata["n"]),
            )
            identifiers = {
                "stage": args.stage,
                "study": args.study,
                "job_name": args.job_name,
                "instance_id": dataset_path.parent.name,
                "topology": metadata["topology"],
                "p": int(metadata["p"]),
                "n": int(metadata["n"]),
                "rep": int(metadata["rep"]),
                "method": method,
                "penalty_constant": constant,
                "penalty_rate": penalty_rate,
                "lambda": lambda_value,
            }
            row: dict[str, Any] = {
                **identifiers,
                "status": "error",
                "fit_available": 0.0,
                "certified": "",
                "error_message": "",
            }
            fit_output: dict[str, Any] = {}
            error_traceback = ""
            try:
                fit_output = fit_method(method, arrays, edges, lambda_value, args)
                row.update(fit_output)
            except Exception as exc:
                failures += 1
                row["error_message"] = f"{type(exc).__name__}: {exc}"
                error_traceback = traceback.format_exc()
                if args.verbose:
                    traceback.print_exc()

            append_result(results_csv, row, RESULT_COLUMNS)
            diagnostic_record = {
                **identifiers,
                "candidate_rule": args.candidate_rule,
                "screen_alpha": args.screen_alpha,
                "candidate_edges": len(edges),
                "candidate_recall": screen_recall,
                "fit": fit_output,
                "error_message": row["error_message"],
                "traceback": error_traceback,
            }
            _append_jsonl(diagnostics_jsonl, diagnostic_record)
            fits_attempted += 1
            print(
                f"{metadata['topology']} p={metadata['p']} n={metadata['n']} "
                f"rep={metadata['rep']} method={method} c={constant:g}: {row['status']}",
                flush=True,
            )

    manifest.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "fits_attempted": fits_attempted,
            "fit_failures": failures,
        }
    )
    _write_json(run_manifest, manifest)
    print(f"Wrote {results_csv}", flush=True)
    print(f"Wrote {diagnostics_jsonl}", flush=True)
    print(f"Wrote {run_manifest}", flush=True)
    return results_csv


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

"""Run a small GraphL0Learn/GraphL0BnB support-recovery test."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    append_csv_row,
    fit_graph_l0_bnb,
    load_or_create_gaussian_dataset,
    normalize_prediction,
    support_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--target-edges", type=int, default=None)
    parser.add_argument("--num-components", type=int, default=1)
    parser.add_argument("--side-length", type=int, default=5)
    parser.add_argument("--hubs-per-component", type=int, default=2)
    parser.add_argument("--hub-degree", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l0", type=float, default=0.05)
    parser.add_argument("--l0-values", type=str, default=None)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--gap-tol", type=float, default=0.05)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/gaussian"),
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/graphl0bnb/results.csv"),
    )
    return parser.parse_args()


def parse_l0_values(args: argparse.Namespace) -> list[float]:
    if args.l0_values is None:
        return [args.l0]
    return [float(value.strip()) for value in args.l0_values.split(",") if value.strip()]


def blank_metrics() -> dict[str, str]:
    return {
        "TP": "",
        "FP": "",
        "TN": "",
        "FN": "",
        "TPR": "",
        "FPR": "",
        "precision": "",
        "recall": "",
        "F1": "",
        "selected_edges": "",
        "true_edges": "",
    }


def main() -> None:
    args = parse_args()
    if args.m is None:
        data_params = {
            "n": args.n,
            "num_components": args.num_components,
            "side_length": args.side_length,
            "hubs_per_component": args.hubs_per_component,
            "hub_degree": args.hub_degree,
            "seed": args.seed,
        }
        dataset_type = "lattice"
        target_edges = ""
    else:
        max_edges = args.m * (args.m - 1) // 2
        target_edges = (
            args.target_edges
            if args.target_edges is not None
            else max(args.m - 1, round(0.27 * max_edges))
        )
        data_params = {
            "n": args.n,
            "m": args.m,
            "target_edges": target_edges,
            "seed": args.seed,
        }
        dataset_type = "exact_random"

    dataset_dir, data = load_or_create_gaussian_dataset(
        args.data_dir,
        data_params,
        force=args.force_data,
    )
    x = data["X"]
    true_precision = data["precision"]
    true_adjacency = data["adjacency"]
    m_bound = float(np.max(np.abs(true_precision)))

    if args.overwrite_results and args.results_csv.exists():
        args.results_csv.unlink()

    statuses = []
    for l0 in parse_l0_values(args):
        row = {
            "status": "error",
            "error_message": "",
            "dataset_dir": str(dataset_dir),
            "dataset_type": dataset_type,
            "n": args.n,
            "m": int(x.shape[1]),
            "target_edges": target_edges,
            "num_components": args.num_components if args.m is None else "",
            "side_length": args.side_length if args.m is None else "",
            "hubs_per_component": args.hubs_per_component if args.m is None else "",
            "hub_degree": args.hub_degree if args.m is None else "",
            "seed": args.seed,
            "l0": l0,
            "l2": args.l2,
            "M": m_bound,
            "gap_tol": args.gap_tol,
            "time_limit": args.time_limit,
            "runtime_seconds": "",
            "solver_time_seconds": "",
            "objective": "",
            "gap": "",
            "nodes": "",
            **blank_metrics(),
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
            prediction = normalize_prediction(fit["Theta"])
            metrics = support_metrics(true_adjacency, prediction)
            row.update(
                {
                    "status": "ok",
                    "runtime_seconds": fit["runtime_seconds"],
                    "solver_time_seconds": fit["solver_time_seconds"],
                    "objective": fit["objective"],
                    "gap": fit["gap"],
                    "nodes": fit["nodes"],
                    **metrics,
                }
            )
        except Exception as exc:
            row["error_message"] = f"{type(exc).__name__}: {exc}"
            if args.verbose:
                traceback.print_exc()

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

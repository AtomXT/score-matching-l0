#!/usr/bin/env python3
"""Run one editable Gaussian graph-recovery test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.primary_panel_workflow import OUTPUT_ROOT, run_panel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--stage", choices=["local_check", "evaluation"], default="local_check")
    parser.add_argument("--job-name", default="single_test")
    parser.add_argument("--rep-list", default="0")
    parser.add_argument("--configuration-list", default=None)
    parser.add_argument("--topology", default="erdos_renyi", help="Saved graph topology.")
    parser.add_argument("--p", type=int, default=20, help="Problem dimension.")
    parser.add_argument("--n", type=int, default=400, help="Sample size.")
    parser.add_argument("--max-instances", type=int, default=1)
    parser.add_argument(
        "--method-list",
        default="sm_l0_core",
        help="sm_l0, sm_l0_core, sm_l0_milp, sm_l1, graphl0, or glasso.",
    )
    parser.add_argument(
        "--penalty-constant-list",
        default="2.28",
        help="One constant by default; comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--candidate-rule",
        choices=["complete", "graphical_lasso"],
        default="graphical_lasso",
    )
    parser.add_argument("--screen-alpha", type=float, default=0.01)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--big-m-init", type=float, default=1000.0)
    parser.add_argument("--l1-max-iter", type=int, default=5_000)
    parser.add_argument("--l1-tolerance", type=float, default=1e-6)
    parser.add_argument("--l1-support-tolerance", type=float, default=1e-6)
    parser.add_argument("--graphl0-l2", type=float, default=0.05)
    parser.add_argument("--graphl0-m-bound", type=float, default=100.0)
    parser.add_argument("--glasso-max-iter", type=int, default=1_000)
    parser.add_argument("--glasso-tolerance", type=float, default=1e-4)
    parser.add_argument("--data-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_panel("single_test", parse_args())

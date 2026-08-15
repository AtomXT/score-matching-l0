#!/usr/bin/env python3
"""Run the registered p=500, n=250 Erdos--Renyi ROC experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.primary_panel_workflow import OUTPUT_ROOT, run_panel


def parse_args(
    argv: list[str] | None = None,
    *,
    description: str = __doc__,
    default_job_name: str = "roc_local_check",
    default_n: int = 250,
    default_data_root: Path = OUTPUT_ROOT,
    default_candidate_rule: str = "graphical_lasso",
    default_screen_alpha: float = 0.01,
    default_glasso_tolerance: float = 1e-4,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["local_check", "evaluation"], default="local_check")
    parser.add_argument("--job-name", default=default_job_name)
    parser.add_argument("--rep-list", default="0")
    parser.add_argument("--configuration-list", default=None)
    parser.add_argument("--topology", default="erdos_renyi")
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--n", type=int, default=default_n)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--method-list", default="sm_l1")
    parser.add_argument(
        "--penalty-constant-list",
        default="0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,1,1.2,1.4,1.6,1.8,2,2.5,3,5,10,20,50,100",
        help="Common path of constants used to construct the ROC curve.",
    )
    parser.add_argument(
        "--candidate-rule",
        choices=["complete", "graphical_lasso", "spearman_graphical_lasso"],
        default=default_candidate_rule,
    )
    parser.add_argument("--screen-alpha", type=float, default=default_screen_alpha)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--big-m-init", type=float, default=1000.0)
    parser.add_argument("--l1-max-iter", type=int, default=5_000)
    parser.add_argument("--l1-tolerance", type=float, default=1e-6)
    parser.add_argument("--l1-support-tolerance", type=float, default=1e-6)
    parser.add_argument("--graphl0-l2", type=float, default=0.05)
    parser.add_argument("--graphl0-m-bound", type=float, default=100.0)
    parser.add_argument("--glasso-max-iter", type=int, default=1_000)
    parser.add_argument(
        "--glasso-tolerance", type=float, default=default_glasso_tolerance
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_panel("roc", parse_args())

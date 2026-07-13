#!/usr/bin/env python3
"""Run the sample-size panel with explicit editable defaults."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.primary_panel_workflow import (
    DEFAULT_CONSTANTS_FILE,
    OUTPUT_ROOT,
    run_panel,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["local_check", "evaluation"], default="local_check")
    parser.add_argument("--job-name", default="sample_size_local_check")
    parser.add_argument("--rep-list", default="0")
    parser.add_argument("--configuration-list", default=None)
    parser.add_argument("--topology-list", default=None)
    parser.add_argument("--p-list", default=None)
    parser.add_argument("--n-list", default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--method-list", default="sm_l1")
    parser.add_argument("--penalty-constant-list", default="1")
    parser.add_argument("--penalty-constants-json", type=Path, default=DEFAULT_CONSTANTS_FILE)
    parser.add_argument("--candidate-rule", choices=["complete", "correlation"], default="complete")
    parser.add_argument("--screen-size", type=int, default=None)
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
    run_panel("sample_size", parse_args())

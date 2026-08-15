#!/usr/bin/env python3
"""Run Gaussian score-matching ROC paths on the multivariate-t robustness data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.Run_gaussian_roc import parse_args as parse_gaussian_roc_args
from experiments.generate_nongaussian_roc_data import OUTPUT_ROOT, STUDY
from experiments.primary_panel_workflow import run_panel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the shared ROC options with non-Gaussian defaults."""
    return parse_gaussian_roc_args(
        argv,
        description=__doc__,
        default_job_name="nongaussian_roc_local_check",
        default_n=1000,
        default_data_root=OUTPUT_ROOT,
        default_candidate_rule="spearman_graphical_lasso",
        default_screen_alpha=0.1,
        default_glasso_tolerance=2e-3,
    )


def main(argv: list[str] | None = None) -> None:
    run_panel(
        "roc",
        parse_args(argv),
        study=STUDY,
        results_prefix="nongaussian",
    )


if __name__ == "__main__":
    main()

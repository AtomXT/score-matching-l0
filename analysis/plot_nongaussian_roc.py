#!/usr/bin/env python3
"""Average and plot the multivariate-t robustness ROC penalty paths."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.plot_gaussian_roc import main as plot_roc


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_DIR / "experiments_results" / "nongaussian_robustness_roc"


if __name__ == "__main__":
    plot_roc(
        description=__doc__,
        default_n=1000,
        default_results_root=RESULTS_ROOT,
    )

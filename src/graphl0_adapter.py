"""Adapter for the bundled GraphL0Learn branch-and-bound implementation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .l0bnb2 import BNBTree, heuristic_solve, preprocess


def fit_graph_l0_bnb(
    x: np.ndarray,
    *,
    l0: float,
    l2: float,
    m_bound: float,
    gap_tol: float,
    time_limit: float,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fit GraphL0Learn and return diagnostics in the experiment schema."""
    _, _, _, _, y, _ = preprocess(x, assume_centered=False)
    start = time.perf_counter()
    theta_approx, _, _, _ = heuristic_solve(y, l0, l2, m_bound)
    tree = BNBTree(x)
    solution = tree.solve(
        l0,
        l2,
        m_bound,
        warm_start=theta_approx,
        verbose=verbose,
        gap_tol=gap_tol,
        time_limit=time_limit,
    )
    runtime = time.perf_counter() - start

    return {
        "Theta": solution.Theta,
        "runtime_seconds": runtime,
        "solver_time_seconds": float(solution.sol_time),
        "objective": float(solution.cost),
        "gap": float(solution.gap),
        "nodes": int(tree.number_of_nodes),
    }

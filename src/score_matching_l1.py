"""Highscore-compatible L1-regularized Gaussian score matching.

For a centered empirical covariance matrix ``S``, this module minimizes

    0.5 * tr(K S K) - tr(K) + lambda * sum_{i != j} |K[i, j]|

over symmetric matrices ``K``.  This is the Gaussian objective implemented by
the authors' ``highscore`` R package.  The diagonal is unpenalized, and positive
definiteness is not imposed.  The implementation below is an independent Python
implementation of cyclic coordinate descent for this convex objective.

Because ``K`` is symmetric, one undirected edge contributes
``2 * lambda * |K[i, j]|``.  Thus ``lambda_value`` follows the authors'
full-matrix convention rather than a unique-upper-triangle convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .score_matching_miqp import centered_sample_covariance, complete_edge_list


@njit
def _coordinate_descent(
    sample_covariance: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    lambda_value: float,
    tolerance: float,
    max_iter: int,
    divergence_limit: float,
) -> tuple[np.ndarray, int, float, int]:
    """Run cyclic coordinate descent; return status 0, 1, or 2.

    Status 0 denotes convergence, 1 the iteration limit, and 2 a nonfinite or
    excessively large coordinate.  The update formulas follow directly from
    the one-coordinate minimizers of the objective stated in the module
    docstring.
    """
    p = sample_covariance.shape[0]
    precision = np.eye(p)
    final_change = np.inf

    for iteration in range(1, max_iter + 1):
        sweep_change = 0.0

        for edge_index in range(edge_i.size):
            i = edge_i[edge_index]
            j = edge_j[edge_index]
            old_value = precision[i, j]

            first_partial = 0.0
            second_partial = 0.0
            for k in range(p):
                first_partial += precision[i, k] * sample_covariance[j, k]
                second_partial += precision[k, j] * sample_covariance[i, k]
            first_partial -= old_value * sample_covariance[j, j]
            second_partial -= old_value * sample_covariance[i, i]

            curvature = sample_covariance[i, i] + sample_covariance[j, j]
            center = -(first_partial + second_partial) / curvature
            threshold = 2.0 * lambda_value / curvature
            if center > threshold:
                new_value = center - threshold
            elif center < -threshold:
                new_value = center + threshold
            else:
                new_value = 0.0

            if not np.isfinite(new_value) or abs(new_value) > divergence_limit:
                return precision, iteration, np.inf, 2
            precision[i, j] = new_value
            precision[j, i] = new_value
            sweep_change += 2.0 * abs(new_value - old_value)

        for i in range(p):
            old_value = precision[i, i]
            off_diagonal_term = 0.0
            for k in range(p):
                if k != i:
                    off_diagonal_term += (
                        precision[i, k] * sample_covariance[i, k]
                    )
            new_value = (1.0 - off_diagonal_term) / sample_covariance[i, i]
            if not np.isfinite(new_value) or abs(new_value) > divergence_limit:
                return precision, iteration, np.inf, 2
            precision[i, i] = new_value
            sweep_change += abs(new_value - old_value)

        final_change = sweep_change
        if sweep_change < tolerance:
            return precision, iteration, final_change, 0

    return precision, max_iter, final_change, 1


def _candidate_edges(
    edge_list: list[tuple[int, int]] | None,
    p: int,
) -> list[tuple[int, int]]:
    """Return the requested candidate edges or the complete edge set."""
    if edge_list is None:
        return complete_edge_list(p)
    return [tuple(map(int, edge)) for edge in edge_list]


def solve_score_matching_l1(
    x: np.ndarray,
    *,
    lambda_value: float,
    edge_list: list[tuple[int, int]] | None = None,
    assume_centered: bool = False,
    max_iter: int = 5_000,
    tolerance: float = 1e-6,
    support_tolerance: float = 1e-6,
    divergence_limit: float = 1e12,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fit the authors' Gaussian L1 score-matching estimator.

    ``lambda_value`` multiplies the sum over all ordered off-diagonal entries,
    exactly as in the uploaded ``highscore`` implementation.  When
    ``assume_centered`` is false, the sample mean is removed before constructing
    the empirical covariance.  The primary experiment already centers and
    scales every split using training-sample quantities.
    """
    data = np.asarray(x, dtype=float)

    if assume_centered:
        sample_covariance = data.T @ data / data.shape[0]
    else:
        sample_covariance = centered_sample_covariance(data)
    sample_covariance = np.ascontiguousarray(sample_covariance, dtype=float)

    edges = _candidate_edges(edge_list, sample_covariance.shape[0])
    edge_i = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    edge_j = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    precision, iterations, sweep_change, status_code = _coordinate_descent(
        sample_covariance,
        edge_i,
        edge_j,
        float(lambda_value),
        float(tolerance),
        int(max_iter),
        float(divergence_limit),
    )

    beta = np.asarray([precision[i, j] for i, j in edges], dtype=float)
    adjacency = np.zeros(precision.shape, dtype=bool)
    for value, (i, j) in zip(beta, edges):
        if abs(value) > support_tolerance:
            adjacency[i, j] = True
            adjacency[j, i] = True

    smooth_objective = (
        0.5 * np.trace(precision @ sample_covariance @ precision)
        - np.trace(precision)
    )
    penalty = 2.0 * lambda_value * np.abs(beta).sum()
    objective = float(smooth_objective + penalty)
    statuses = {0: "converged", 1: "iteration_limit", 2: "diverged"}
    status = statuses[status_code]
    converged = status_code == 0

    if verbose:
        print(
            "SM-L1 (highscore-compatible coordinate descent): "
            f"status={status}, iterations={iterations}, "
            f"sweep_change={sweep_change:.3e}, objective={objective:.8g}",
            flush=True,
        )

    return {
        "sample_covariance": sample_covariance,
        "edge_list": tuple(edges),
        "beta": beta,
        "precision": precision,
        "adjacency": adjacency,
        "objective": objective,
        "iterations": iterations,
        "converged": converged,
        "status": status,
        "sweep_change": float(sweep_change),
    }

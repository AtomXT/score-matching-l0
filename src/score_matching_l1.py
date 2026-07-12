"""Loss-matched L1 estimator for Gaussian score matching.

This module minimizes the same profiled quadratic used by the L0 estimators,
with an entrywise L1 penalty on the unique undirected edge coefficients.  The
implementation uses FISTA and is intentionally dependency-light so that the
baseline can be run on every synthetic instance without a second solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .score_matching_miqp import (
    GaussianScoreMatchingFormulation,
    adjacency_from_edge_indicators,
    build_gaussian_score_matching_formulation,
    reconstruct_precision,
)


@dataclass(frozen=True)
class ScoreMatchingL1Solution:
    formulation: GaussianScoreMatchingFormulation
    beta: np.ndarray
    precision: np.ndarray
    adjacency: np.ndarray
    objective: float
    iterations: int
    converged: bool
    status: str


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the scalar soft-thresholding operator componentwise."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def solve_score_matching_l1(
    x: np.ndarray,
    *,
    lambda_value: float,
    edge_list: list[tuple[int, int]] | None = None,
    assume_centered: bool = False,
    max_iter: int = 20_000,
    tolerance: float = 1e-8,
    support_tolerance: float = 1e-7,
) -> ScoreMatchingL1Solution:
    """Solve the profiled L1-regularized Gaussian score-matching problem.

    Convergence is declared when the Euclidean change between two consecutive
    coefficient vectors is at most ``tolerance * max(1, ||beta||_2)``.  The
    returned support uses a separate numerical threshold because FISTA can
    leave coefficients extremely close to, but not exactly equal to, zero.
    """
    if lambda_value < 0:
        raise ValueError("lambda_value must be nonnegative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tolerance <= 0 or support_tolerance < 0:
        raise ValueError("tolerances must be positive (support tolerance may be zero)")

    formulation = build_gaussian_score_matching_formulation(
        x,
        assume_centered=assume_centered,
        edge_list=edge_list,
    )
    Q = formulation.Q_prof
    q = formulation.q_prof
    n_edges = len(formulation.edge_list)

    if n_edges == 0:
        beta = np.empty(0, dtype=float)
        precision = reconstruct_precision(beta, formulation)
        adjacency = np.zeros(precision.shape, dtype=bool)
        return ScoreMatchingL1Solution(
            formulation, beta, precision, adjacency, 0.0, 0, True, "converged"
        )

    # The gradient of 0.5 beta' Q beta - q' beta is Q beta - q.
    # Since Q is positive semidefinite, its largest eigenvalue is a valid
    # global Lipschitz constant for this gradient.
    lipschitz = max(float(np.linalg.eigvalsh(Q).max()), np.finfo(float).eps)
    beta = np.zeros(n_edges, dtype=float)
    extrapolated = beta.copy()
    momentum = 1.0
    converged = False

    for iteration in range(1, max_iter + 1):
        gradient = Q @ extrapolated - q
        beta_next = _soft_threshold(
            extrapolated - gradient / lipschitz,
            lambda_value / lipschitz,
        )

        difference = np.linalg.norm(beta_next - beta)
        scale = max(1.0, np.linalg.norm(beta_next))
        if difference <= tolerance * scale:
            beta = beta_next
            converged = True
            break

        momentum_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        extrapolated = beta_next + ((momentum - 1.0) / momentum_next) * (
            beta_next - beta
        )
        beta = beta_next
        momentum = momentum_next

    indicators = np.abs(beta) > support_tolerance
    m = formulation.sample_covariance.shape[0]
    precision = reconstruct_precision(beta, formulation)
    adjacency = adjacency_from_edge_indicators(indicators, formulation.edge_list, m)
    objective = float(
        0.5 * beta @ Q @ beta - q @ beta + lambda_value * np.abs(beta).sum()
    )
    return ScoreMatchingL1Solution(
        formulation=formulation,
        beta=beta,
        precision=precision,
        adjacency=adjacency,
        objective=objective,
        iterations=iteration,
        converged=converged,
        status="converged" if converged else "iteration_limit",
    )

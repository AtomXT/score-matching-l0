"""Gaussian score-matching MIQP estimator.

The Gaussian score-matching loss for a centered sample covariance S is

    0.5 * tr(K S K) - tr(K),

where K is the precision matrix. This module parametrizes K by unique symmetric
entries, profiles out diagonal nuisance variables, and solves the remaining
L0-regularized off-diagonal MIQP with Gurobi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GaussianScoreMatchingFormulation:
    sample_covariance: np.ndarray
    edge_list: list[tuple[int, int]]
    Q_prof: np.ndarray
    q_prof: np.ndarray
    Q_alpha_alpha: np.ndarray
    Q_alpha_beta: np.ndarray
    q_alpha: np.ndarray


@dataclass(frozen=True)
class BigMBounds:
    values: np.ndarray
    beta_continuous: np.ndarray
    continuous_lower_bound: float


@dataclass(frozen=True)
class ScoreMatchingMIQPSolution:
    formulation: GaussianScoreMatchingFormulation
    big_m: BigMBounds
    beta: np.ndarray
    z: np.ndarray
    precision: np.ndarray
    adjacency: np.ndarray
    has_solution: bool
    runtime_seconds: float
    objective: float
    objective_bound: float
    mip_gap: float
    nodes: float
    status: str
    status_code: int


def centered_sample_covariance(x: np.ndarray) -> np.ndarray:
    """Return the centered empirical covariance with divisor n."""
    data = np.asarray(x, dtype=float)
    if data.ndim != 2:
        raise ValueError("x must be a two-dimensional sample matrix")
    centered = data - data.mean(axis=0, keepdims=True)
    return centered.T @ centered / centered.shape[0]


def complete_edge_list(m: int) -> list[tuple[int, int]]:
    """List upper-triangle off-diagonal edge indices."""
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def build_gaussian_score_matching_formulation(
    x: np.ndarray,
    *,
    assume_centered: bool = False,
    edge_list: list[tuple[int, int]] | None = None,
) -> GaussianScoreMatchingFormulation:
    """Build the profiled quadratic for Gaussian score matching.

    The off-diagonal vector beta contains K[i, j] for each upper-triangle pair.
    The diagonal vector alpha is profiled out analytically.
    """
    data = np.asarray(x, dtype=float)
    if data.ndim != 2:
        raise ValueError("x must be a two-dimensional sample matrix")

    if assume_centered:
        sample_covariance = data.T @ data / data.shape[0]
    else:
        sample_covariance = centered_sample_covariance(data)

    m = sample_covariance.shape[0]
    if edge_list is None:
        edge_list = complete_edge_list(m)
    else:
        edge_list = [tuple(map(int, edge)) for edge in edge_list]
        if len(set(edge_list)) != len(edge_list):
            raise ValueError("edge_list must not contain duplicate edges")
        if any(i < 0 or j >= m or i >= j for i, j in edge_list):
            raise ValueError("edge_list must contain pairs (i, j) with 0 <= i < j < m")
    n_edges = len(edge_list)

    diag = np.diag(sample_covariance)
    if np.any(diag <= 0):
        raise ValueError("sample covariance must have positive diagonal entries")

    Q_alpha_alpha = np.diag(diag)
    Q_alpha_beta = np.zeros((m, n_edges), dtype=float)
    Q_beta_beta = np.zeros((n_edges, n_edges), dtype=float)

    for edge_idx, (i, j) in enumerate(edge_list):
        sij = sample_covariance[i, j]
        Q_alpha_beta[i, edge_idx] = sij
        Q_alpha_beta[j, edge_idx] = sij

    for row_idx, (i, j) in enumerate(edge_list):
        for col_idx, (k, ell) in enumerate(edge_list):
            value = 0.0
            if i == ell:
                value += sample_covariance[j, k]
            if i == k:
                value += sample_covariance[j, ell]
            if j == ell:
                value += sample_covariance[i, k]
            if j == k:
                value += sample_covariance[i, ell]
            Q_beta_beta[row_idx, col_idx] = value

    q_alpha = np.ones(m, dtype=float)
    q_beta = np.zeros(n_edges, dtype=float)

    alpha_solve_Qab = np.linalg.solve(Q_alpha_alpha, Q_alpha_beta)
    alpha_solve_qa = np.linalg.solve(Q_alpha_alpha, q_alpha)
    Q_prof = Q_beta_beta - Q_alpha_beta.T @ alpha_solve_Qab
    q_prof = q_beta - Q_alpha_beta.T @ alpha_solve_qa

    Q_prof = 0.5 * (Q_prof + Q_prof.T)
    return GaussianScoreMatchingFormulation(
        sample_covariance=sample_covariance,
        edge_list=edge_list,
        Q_prof=Q_prof,
        q_prof=q_prof,
        Q_alpha_alpha=Q_alpha_alpha,
        Q_alpha_beta=Q_alpha_beta,
        q_alpha=q_alpha,
    )


def profiled_alpha(
    beta: np.ndarray,
    formulation: GaussianScoreMatchingFormulation,
) -> np.ndarray:
    """Recover the profiled diagonal entries for a fixed beta."""
    beta = np.asarray(beta, dtype=float)
    return np.linalg.solve(
        formulation.Q_alpha_alpha,
        formulation.q_alpha - formulation.Q_alpha_beta @ beta,
    )


def reconstruct_precision(
    beta: np.ndarray,
    formulation: GaussianScoreMatchingFormulation,
) -> np.ndarray:
    """Reconstruct the full symmetric precision matrix from beta."""
    beta = np.asarray(beta, dtype=float)
    m = formulation.sample_covariance.shape[0]
    if beta.shape != (len(formulation.edge_list),):
        raise ValueError("beta has the wrong shape for this formulation")

    precision = np.zeros((m, m), dtype=float)
    for value, (i, j) in zip(beta, formulation.edge_list):
        precision[i, j] = value
        precision[j, i] = value
    np.fill_diagonal(precision, profiled_alpha(beta, formulation))
    return precision


def adjacency_from_edge_indicators(
    z: np.ndarray,
    edge_list: list[tuple[int, int]],
    m: int,
) -> np.ndarray:
    """Build a symmetric boolean adjacency matrix from edge indicators."""
    indicators = np.asarray(z)
    if indicators.shape != (len(edge_list),):
        raise ValueError("z has the wrong shape for this edge list")

    adjacency = np.zeros((m, m), dtype=bool)
    for selected, (i, j) in zip(indicators.astype(bool), edge_list):
        if selected:
            adjacency[i, j] = True
            adjacency[j, i] = True
    return adjacency


def data_derived_big_m(
    Q_prof: np.ndarray,
    q_prof: np.ndarray,
    *,
    scale: float = 1.25,
    floor: float = 1e-8,
) -> BigMBounds:
    """Compute data-derived big-M bounds from the continuous profiled solution."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if floor <= 0:
        raise ValueError("floor must be positive")

    Q = np.asarray(Q_prof, dtype=float)
    q = np.asarray(q_prof, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q_prof must be square")
    if q.shape != (Q.shape[0],):
        raise ValueError("q_prof has incompatible shape")

    try:
        beta_cont = np.linalg.solve(Q, q)
        inverse_diag = np.diag(np.linalg.solve(Q, np.eye(Q.shape[0])))
    except np.linalg.LinAlgError:
        Q_pinv = np.linalg.pinv(Q)
        beta_cont = Q_pinv @ q
        inverse_diag = np.diag(Q_pinv)

    continuous_lower_bound = float(
        0.5 * beta_cont @ Q @ beta_cont - q @ beta_cont
    )
    gap_to_zero = max(0.0, -continuous_lower_bound)
    inverse_diag = np.maximum(inverse_diag, 0.0)
    values = scale * (
        np.abs(beta_cont) + np.sqrt(2.0 * gap_to_zero * inverse_diag)
    )
    values = np.maximum(values, floor)

    if not np.all(np.isfinite(values)):
        raise ValueError("computed non-finite big-M bounds")

    return BigMBounds(
        values=values,
        beta_continuous=beta_cont,
        continuous_lower_bound=continuous_lower_bound,
    )


def solve_score_matching_miqp(
    x: np.ndarray,
    *,
    lambda_value: float,
    big_m_scale: float = 1.25,
    time_limit: float | None = 60.0,
    mip_gap: float | None = 0.05,
    output_flag: bool = False,
    assume_centered: bool = False,
    edge_list: list[tuple[int, int]] | None = None,
    threads: int | None = None,
) -> ScoreMatchingMIQPSolution:
    """Solve the L0-regularized profiled Gaussian score-matching MIQP."""
    if lambda_value < 0:
        raise ValueError("lambda_value must be nonnegative")

    import gurobipy as gp
    from gurobipy import GRB

    formulation = build_gaussian_score_matching_formulation(
        x,
        assume_centered=assume_centered,
        edge_list=edge_list,
    )
    big_m = data_derived_big_m(
        formulation.Q_prof,
        formulation.q_prof,
        scale=big_m_scale,
    )
    n_edges = len(formulation.edge_list)
    m = formulation.sample_covariance.shape[0]

    model = gp.Model("gaussian_score_matching_miqp")
    model.Params.OutputFlag = 1 if output_flag else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)

    beta_vars = model.addVars(n_edges, lb=-GRB.INFINITY, name="beta")
    z_vars = model.addVars(n_edges, vtype=GRB.BINARY, name="z")

    for edge_idx, bound in enumerate(big_m.values):
        model.addConstr(beta_vars[edge_idx] <= float(bound) * z_vars[edge_idx])
        model.addConstr(beta_vars[edge_idx] >= -float(bound) * z_vars[edge_idx])

    objective = gp.QuadExpr()
    Q = formulation.Q_prof
    q = formulation.q_prof
    for i in range(n_edges):
        if Q[i, i] != 0.0:
            objective += 0.5 * float(Q[i, i]) * beta_vars[i] * beta_vars[i]
        for j in range(i + 1, n_edges):
            if Q[i, j] != 0.0:
                objective += float(Q[i, j]) * beta_vars[i] * beta_vars[j]
        if q[i] != 0.0:
            objective += -float(q[i]) * beta_vars[i]
        objective += float(lambda_value) * z_vars[i]

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    status = _gurobi_status_name(GRB, model.Status)
    has_solution = model.SolCount > 0
    if has_solution:
        beta = np.array([beta_vars[i].X for i in range(n_edges)], dtype=float)
        z = np.array([z_vars[i].X >= 0.5 for i in range(n_edges)], dtype=bool)
        precision = reconstruct_precision(beta, formulation)
        adjacency = adjacency_from_edge_indicators(z, formulation.edge_list, m)
    else:
        beta = np.zeros(n_edges, dtype=float)
        z = np.zeros(n_edges, dtype=bool)
        precision = reconstruct_precision(beta, formulation)
        adjacency = adjacency_from_edge_indicators(z, formulation.edge_list, m)

    return ScoreMatchingMIQPSolution(
        formulation=formulation,
        big_m=big_m,
        beta=beta,
        z=z,
        precision=precision,
        adjacency=adjacency,
        has_solution=has_solution,
        runtime_seconds=_safe_float(lambda: model.Runtime),
        objective=_safe_float(lambda: model.ObjVal) if has_solution else np.nan,
        objective_bound=_safe_float(lambda: model.ObjBound),
        mip_gap=_safe_float(lambda: model.MIPGap) if has_solution else np.nan,
        nodes=_safe_float(lambda: model.NodeCount),
        status=status,
        status_code=int(model.Status),
    )


def _safe_float(get_value: Any) -> float:
    try:
        return float(get_value())
    except Exception:
        return float("nan")


def _gurobi_status_name(GRB: Any, status_code: int) -> str:
    names = [
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "INPROGRESS",
        "USER_OBJ_LIMIT",
        "WORK_LIMIT",
        "MEM_LIMIT",
    ]
    for name in names:
        if getattr(GRB, name, None) == status_code:
            return name
    return f"STATUS_{status_code}"

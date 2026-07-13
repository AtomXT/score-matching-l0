"""Gaussian score-matching MIQP estimator.

The Gaussian score-matching loss for a centered sample covariance S is

    0.5 * tr(K S K) - tr(K),

where K is the precision matrix. This module parametrizes K by unique symmetric
entries, profiles out diagonal nuisance variables, and solves the remaining
L0-regularized off-diagonal MIQP with Gurobi.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def centered_sample_covariance(x: np.ndarray) -> np.ndarray:
    """Return the centered empirical covariance with divisor n."""
    data = np.asarray(x, dtype=float)
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
) -> dict[str, Any]:
    """Build the profiled quadratic for Gaussian score matching.

    The off-diagonal vector beta contains K[i, j] for each upper-triangle pair.
    The diagonal vector alpha is profiled out analytically.
    """
    data = np.asarray(x, dtype=float)

    if assume_centered:
        sample_covariance = data.T @ data / data.shape[0]
    else:
        sample_covariance = centered_sample_covariance(data)

    m = sample_covariance.shape[0]
    if edge_list is None:
        edge_list = complete_edge_list(m)
    else:
        edge_list = [tuple(map(int, edge)) for edge in edge_list]
    n_edges = len(edge_list)

    diag = np.diag(sample_covariance)

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
    return {
        "sample_covariance": sample_covariance,
        "edge_list": edge_list,
        "Q_prof": Q_prof,
        "q_prof": q_prof,
        "Q_alpha_alpha": Q_alpha_alpha,
        "Q_alpha_beta": Q_alpha_beta,
        "q_alpha": q_alpha,
    }


def profiled_alpha(
    beta: np.ndarray,
    formulation: dict[str, Any],
) -> np.ndarray:
    """Recover the profiled diagonal entries for a fixed beta."""
    beta = np.asarray(beta, dtype=float)
    return np.linalg.solve(
        formulation["Q_alpha_alpha"],
        formulation["q_alpha"] - formulation["Q_alpha_beta"] @ beta,
    )


def reconstruct_precision(
    beta: np.ndarray,
    formulation: dict[str, Any],
) -> np.ndarray:
    """Reconstruct the full symmetric precision matrix from beta."""
    beta = np.asarray(beta, dtype=float)
    m = formulation["sample_covariance"].shape[0]

    precision = np.zeros((m, m), dtype=float)
    for value, (i, j) in zip(beta, formulation["edge_list"]):
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

    adjacency = np.zeros((m, m), dtype=bool)
    for selected, (i, j) in zip(indicators.astype(bool), edge_list):
        if selected:
            adjacency[i, j] = True
            adjacency[j, i] = True
    return adjacency


def solve_big_m_relaxation(
    formulation: dict[str, Any],
    lambda_value: float,
    big_m_init: float,
    threads: int | None,
) -> dict[str, float]:
    """Use the continuous indicator relaxation to choose a scalar big-M."""
    import gurobipy as gp
    from gurobipy import GRB

    n_edges = len(formulation["edge_list"])
    relaxation = gp.Model("gaussian_score_matching_big_m_relaxation")
    relaxation.Params.OutputFlag = 0
    if threads is not None:
        relaxation.Params.Threads = int(threads)
    coefficients = relaxation.addMVar(n_edges, lb=-GRB.INFINITY, name="beta")
    indicators = relaxation.addMVar(n_edges, lb=0.0, ub=1.0, name="z")
    relaxation.addConstr(coefficients <= big_m_init * indicators)
    relaxation.addConstr(coefficients >= -big_m_init * indicators)
    relaxation.setObjective(
        0.5 * (coefficients @ formulation["Q_prof"] @ coefficients)
        - formulation["q_prof"] @ coefficients
        + lambda_value * indicators.sum(),
        GRB.MINIMIZE,
    )
    relaxation.optimize()

    runtime = _safe_float(lambda: relaxation.Runtime)
    if relaxation.SolCount:
        max_abs_coefficient = float(np.max(np.abs(np.asarray(coefficients.X))))
        big_m = min(
            big_m_init,
            max(big_m_init * 1e-6, 2.0 * max_abs_coefficient),
        )
        objective = _safe_float(lambda: relaxation.ObjVal)
    else:
        big_m = float(big_m_init)
        objective = float("nan")
    return {"big_m": big_m, "objective": objective, "runtime_seconds": runtime}


def solve_score_matching_miqp(
    x: np.ndarray,
    *,
    lambda_value: float,
    big_m_init: float = 1000.0,
    time_limit: float | None = 60.0,
    mip_gap: float | None = 0.05,
    output_flag: bool = False,
    assume_centered: bool = False,
    edge_list: list[tuple[int, int]] | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Solve the L0-regularized profiled Gaussian score-matching MIQP."""
    import gurobipy as gp
    from gurobipy import GRB

    formulation = build_gaussian_score_matching_formulation(
        x,
        assume_centered=assume_centered,
        edge_list=edge_list,
    )
    n_edges = len(formulation["edge_list"])
    m = formulation["sample_covariance"].shape[0]
    Q = formulation["Q_prof"]
    q = formulation["q_prof"]
    relaxation = solve_big_m_relaxation(
        formulation,
        lambda_value,
        big_m_init,
        threads,
    )
    big_m = relaxation["big_m"]

    model = gp.Model("gaussian_score_matching_miqp")
    model.Params.OutputFlag = 1 if output_flag else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)

    beta_vars = model.addMVar(n_edges, lb=-GRB.INFINITY, name="beta")
    z_vars = model.addMVar(n_edges, vtype=GRB.BINARY, name="z")
    model.addConstr(beta_vars <= big_m * z_vars)
    model.addConstr(beta_vars >= -big_m * z_vars)
    model.setObjective(
        0.5 * (beta_vars @ Q @ beta_vars)
        - q @ beta_vars
        + lambda_value * z_vars.sum(),
        GRB.MINIMIZE,
    )
    model.optimize()

    status = _gurobi_status_name(GRB, model.Status)
    has_solution = model.SolCount > 0
    if has_solution:
        beta = np.asarray(beta_vars.X, dtype=float)
        z = np.asarray(z_vars.X) >= 0.5
        precision = reconstruct_precision(beta, formulation)
        adjacency = adjacency_from_edge_indicators(z, formulation["edge_list"], m)
    else:
        beta = np.zeros(n_edges, dtype=float)
        z = np.zeros(n_edges, dtype=bool)
        precision = reconstruct_precision(beta, formulation)
        adjacency = adjacency_from_edge_indicators(z, formulation["edge_list"], m)

    return {
        "formulation": formulation,
        "big_m": big_m,
        "big_m_relaxation_objective": relaxation["objective"],
        "big_m_relaxation_runtime_seconds": relaxation["runtime_seconds"],
        "beta": beta,
        "z": z,
        "precision": precision,
        "adjacency": adjacency,
        "has_solution": has_solution,
        "runtime_seconds": relaxation["runtime_seconds"]
        + _safe_float(lambda: model.Runtime),
        "objective": _safe_float(lambda: model.ObjVal) if has_solution else np.nan,
        "objective_bound": _safe_float(lambda: model.ObjBound),
        "mip_gap": _safe_float(lambda: model.MIPGap) if has_solution else np.nan,
        "nodes": _safe_float(lambda: model.NodeCount),
        "status": status,
        "status_code": int(model.Status),
    }


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

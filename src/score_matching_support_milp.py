"""Support-optimality MILP for L0-regularized Gaussian score matching."""

from __future__ import annotations

from typing import Any

import numpy as np

from .score_matching_miqp import (
    _gurobi_status_name,
    _safe_float,
    adjacency_from_edge_indicators,
    build_gaussian_score_matching_formulation,
    solve_big_m_relaxation,
)


def _precision_from_coordinates(
    alpha: np.ndarray,
    beta: np.ndarray,
    edge_list: list[tuple[int, int]],
) -> np.ndarray:
    precision = np.diag(np.asarray(alpha, dtype=float))
    for coefficient, (left, right) in zip(beta, edge_list):
        precision[left, right] = precision[right, left] = coefficient
    return precision


def _empty_candidate_solution(formulation: dict[str, Any]) -> dict[str, Any]:
    """Return the unique diagonal optimum when the candidate graph is empty."""
    from gurobipy import GRB

    diagonal = np.diag(formulation["Q_alpha_alpha"])
    alpha = 1.0 / diagonal
    dimension = diagonal.size
    precision = np.diag(alpha)
    objective = -0.5 * float(np.sum(alpha))
    return {
        "formulation": formulation,
        "big_m": 0.0,
        "big_m_relaxation_objective": 0.0,
        "big_m_relaxation_runtime_seconds": 0.0,
        "alpha": alpha,
        "beta": np.zeros(0, dtype=float),
        "z": np.zeros(0, dtype=bool),
        "precision": precision,
        "adjacency": np.zeros((dimension, dimension), dtype=bool),
        "has_solution": True,
        "runtime_seconds": 0.0,
        "objective": objective,
        "objective_bound": objective,
        "mip_gap": 0.0,
        "nodes": 0.0,
        "status": "OPTIMAL",
        "status_code": int(GRB.OPTIMAL),
        "diagonal_stationarity_residual": 0.0,
        "active_stationarity_residual": 0.0,
        "inactive_coefficient_residual": 0.0,
        "objective_identity_residual": 0.0,
    }


def solve_score_matching_support_milp(
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
    """Solve the exact support-optimality MILP on a supplied candidate graph.

    For every selected edge, the corresponding score-matching stationarity
    equation is imposed with a Gurobi indicator constraint.  Diagonal
    stationarity is imposed unconditionally, so at every feasible support the
    quadratic loss equals ``-trace(K) / 2`` and the objective is linear.
    """
    import gurobipy as gp
    from gurobipy import GRB

    if lambda_value < 0.0:
        raise ValueError("lambda_value must be nonnegative")
    if big_m_init <= 0.0:
        raise ValueError("big_m_init must be positive")

    formulation = build_gaussian_score_matching_formulation(
        x,
        assume_centered=assume_centered,
        edge_list=edge_list,
    )
    number_of_edges = len(formulation["edge_list"])
    if number_of_edges == 0:
        return _empty_candidate_solution(formulation)

    relaxation = solve_big_m_relaxation(
        formulation,
        lambda_value,
        big_m_init,
        threads,
    )
    coefficient_bound = float(relaxation["big_m"])
    diagonal_block = formulation["Q_alpha_alpha"]
    cross_block = formulation["Q_alpha_beta"]
    edge_block = formulation["Q_beta_beta"]
    dimension = diagonal_block.shape[0]

    model = gp.Model("gaussian_score_matching_support_optimality_milp")
    model.Params.OutputFlag = int(output_flag)
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)

    alpha = model.addMVar(dimension, lb=-GRB.INFINITY, name="alpha")
    beta = model.addMVar(
        number_of_edges,
        lb=-coefficient_bound,
        ub=coefficient_bound,
        name="beta",
    )
    selected = model.addMVar(number_of_edges, vtype=GRB.BINARY, name="z")

    model.addConstr(beta <= coefficient_bound * selected, name="off_upper")
    model.addConstr(beta >= -coefficient_bound * selected, name="off_lower")
    model.addConstr(
        diagonal_block @ alpha + cross_block @ beta == np.ones(dimension),
        name="diagonal_stationarity",
    )
    edge_residual = cross_block.T @ alpha + edge_block @ beta
    for edge in range(number_of_edges):
        model.addGenConstrIndicator(
            selected[edge],
            True,
            edge_residual[edge] == 0.0,
            name=f"active_stationarity[{edge}]",
        )

    model.setObjective(
        -0.5 * alpha.sum() + lambda_value * selected.sum(),
        GRB.MINIMIZE,
    )
    alpha.Start = 1.0 / np.diag(diagonal_block)
    beta.Start = np.zeros(number_of_edges)
    selected.Start = np.zeros(number_of_edges)
    model.optimize()

    has_solution = model.SolCount > 0
    if has_solution:
        alpha_values = np.asarray(alpha.X, dtype=float)
        beta_values = np.asarray(beta.X, dtype=float)
        indicators = np.asarray(selected.X, dtype=float) >= 0.5
    else:
        alpha_values = np.zeros(dimension, dtype=float)
        beta_values = np.zeros(number_of_edges, dtype=float)
        indicators = np.zeros(number_of_edges, dtype=bool)

    precision = _precision_from_coordinates(
        alpha_values,
        beta_values,
        formulation["edge_list"],
    )
    adjacency = adjacency_from_edge_indicators(
        indicators,
        formulation["edge_list"],
        dimension,
    )

    if has_solution:
        theta_gradient_diagonal = (
            diagonal_block @ alpha_values + cross_block @ beta_values - 1.0
        )
        theta_gradient_edges = (
            cross_block.T @ alpha_values + edge_block @ beta_values
        )
        matrix_objective = float(
            0.5
            * np.trace(
                precision @ formulation["sample_covariance"] @ precision
            )
            - np.trace(precision)
            + lambda_value * np.sum(indicators)
        )
        reported_objective = _safe_float(lambda: model.ObjVal)
        diagonal_residual = float(np.max(np.abs(theta_gradient_diagonal)))
        active_residual = float(
            np.max(np.abs(theta_gradient_edges[indicators]))
            if np.any(indicators)
            else 0.0
        )
        inactive_residual = float(
            np.max(np.abs(beta_values[~indicators]))
            if np.any(~indicators)
            else 0.0
        )
        identity_residual = abs(reported_objective - matrix_objective)
    else:
        reported_objective = float("nan")
        diagonal_residual = float("nan")
        active_residual = float("nan")
        inactive_residual = float("nan")
        identity_residual = float("nan")

    return {
        "formulation": formulation,
        "big_m": coefficient_bound,
        "big_m_relaxation_objective": relaxation["objective"],
        "big_m_relaxation_runtime_seconds": relaxation["runtime_seconds"],
        "alpha": alpha_values,
        "beta": beta_values,
        "z": indicators,
        "precision": precision,
        "adjacency": adjacency,
        "has_solution": has_solution,
        "runtime_seconds": relaxation["runtime_seconds"]
        + _safe_float(lambda: model.Runtime),
        "objective": reported_objective,
        "objective_bound": _safe_float(lambda: model.ObjBound),
        "mip_gap": _safe_float(lambda: model.MIPGap) if has_solution else np.nan,
        "nodes": _safe_float(lambda: model.NodeCount),
        "status": _gurobi_status_name(GRB, model.Status),
        "status_code": int(model.Status),
        "diagonal_stationarity_residual": diagonal_residual,
        "active_stationarity_residual": active_residual,
        "inactive_coefficient_residual": inactive_residual,
        "objective_identity_residual": identity_residual,
    }

"""CORe formulation of the Gaussian score-matching MIQP."""

from __future__ import annotations

from typing import Any

import numpy as np

from .score_matching_miqp import (
    _gurobi_status_name,
    _safe_float,
    adjacency_from_edge_indicators,
    build_gaussian_score_matching_formulation,
    reconstruct_precision,
    solve_big_m_relaxation,
)


def core_thresholds(Q_prof: np.ndarray, lambda_value: float) -> dict[str, np.ndarray]:
    """Return the historical CORe residual and active-coefficient thresholds."""
    diagonal = np.diag(np.asarray(Q_prof, dtype=float))
    return {
        "q_diag": diagonal,
        "kappa": np.sqrt(2.0 * lambda_value * diagonal),
        "tau": np.sqrt(2.0 * lambda_value / diagonal),
    }


def solve_score_matching_core_miqp(
    x: np.ndarray,
    *,
    lambda_value: float,
    big_m_init: float = 100.0,
    time_limit: float | None = 60.0,
    mip_gap: float | None = 0.05,
    output_flag: bool = False,
    assume_centered: bool = False,
    edge_list: list[tuple[int, int]] | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Solve the CORe-strengthened L0 score-matching MIQP."""
    import gurobipy as gp
    from gurobipy import GRB

    formulation = build_gaussian_score_matching_formulation(
        x,
        assume_centered=assume_centered,
        edge_list=edge_list,
    )
    thresholds = core_thresholds(formulation["Q_prof"], lambda_value)
    relaxation = solve_big_m_relaxation(
        formulation,
        lambda_value,
        big_m_init,
        threads,
    )
    big_m = max(relaxation["big_m"], float(np.max(thresholds["tau"])))
    n_edges = len(formulation["edge_list"])
    dimension = formulation["sample_covariance"].shape[0]
    Q = formulation["Q_prof"]
    q = formulation["q_prof"]

    model = gp.Model("gaussian_score_matching_core_miqp")
    model.Params.OutputFlag = 1 if output_flag else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)

    beta = model.addVars(n_edges, lb=-GRB.INFINITY, name="beta")
    inactive = model.addVars(n_edges, vtype=GRB.BINARY, name="inactive")
    positive = model.addVars(n_edges, vtype=GRB.BINARY, name="positive")
    negative = model.addVars(n_edges, vtype=GRB.BINARY, name="negative")

    for edge in range(n_edges):
        model.addConstr(inactive[edge] + positive[edge] + negative[edge] == 1)
        residual = gp.LinExpr(-float(q[edge]))
        for other_edge in range(n_edges):
            residual += float(Q[edge, other_edge]) * beta[other_edge]
        model.addConstr(residual <= thresholds["kappa"][edge] * inactive[edge])
        model.addConstr(residual >= -thresholds["kappa"][edge] * inactive[edge])
        model.addConstr(
            beta[edge]
            >= thresholds["tau"][edge] * positive[edge] - big_m * negative[edge]
        )
        model.addConstr(
            beta[edge]
            <= big_m * positive[edge] - thresholds["tau"][edge] * negative[edge]
        )

    objective = gp.QuadExpr()
    for row in range(n_edges):
        objective += 0.5 * float(Q[row, row]) * beta[row] * beta[row]
        for column in range(row + 1, n_edges):
            objective += float(Q[row, column]) * beta[row] * beta[column]
        objective += -float(q[row]) * beta[row]
        objective += lambda_value * (positive[row] + negative[row])
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    has_solution = model.SolCount > 0
    if has_solution:
        coefficients = np.array([beta[edge].X for edge in range(n_edges)])
        indicators = np.array(
            [positive[edge].X + negative[edge].X >= 0.5 for edge in range(n_edges)]
        )
    else:
        coefficients = np.zeros(n_edges)
        indicators = np.zeros(n_edges, dtype=bool)
    precision = reconstruct_precision(coefficients, formulation)
    adjacency = adjacency_from_edge_indicators(
        indicators,
        formulation["edge_list"],
        dimension,
    )

    return {
        "formulation": formulation,
        "big_m": big_m,
        "big_m_relaxation_objective": relaxation["objective"],
        "big_m_relaxation_runtime_seconds": relaxation["runtime_seconds"],
        "beta": coefficients,
        "z": indicators,
        "precision": precision,
        "adjacency": adjacency,
        "has_solution": has_solution,
        "runtime_seconds": relaxation["runtime_seconds"]
        + _safe_float(lambda: model.Runtime),
        "objective": _safe_float(lambda: model.ObjVal) if has_solution else np.nan,
        "objective_bound": _safe_float(lambda: model.ObjBound),
        "mip_gap": _safe_float(lambda: model.MIPGap) if has_solution else np.nan,
        "nodes": _safe_float(lambda: model.NodeCount),
        "status": _gurobi_status_name(GRB, model.Status),
        "status_code": int(model.Status),
    }

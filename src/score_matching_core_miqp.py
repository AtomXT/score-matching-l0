"""Gaussian score-matching CORe MIQP estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from score_matching_miqp import (
    BigMBounds,
    GaussianScoreMatchingFormulation,
    adjacency_from_edge_indicators,
    build_gaussian_score_matching_formulation,
    data_derived_big_m,
    reconstruct_precision,
    _gurobi_status_name,
    _safe_float,
)


@dataclass(frozen=True)
class COReThresholds:
    q_diag: np.ndarray
    kappa: np.ndarray
    tau: np.ndarray


@dataclass(frozen=True)
class ScoreMatchingCOReSolution:
    formulation: GaussianScoreMatchingFormulation
    big_m: BigMBounds
    thresholds: COReThresholds
    beta: np.ndarray
    z_inactive: np.ndarray
    z_positive: np.ndarray
    z_negative: np.ndarray
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


def core_thresholds(
    Q_prof: np.ndarray,
    lambda_value: float,
    *,
    diagonal_floor: float = 1e-12,
) -> COReThresholds:
    """Return CORe residual and active-coefficient thresholds."""
    if lambda_value < 0:
        raise ValueError("lambda_value must be nonnegative")
    if diagonal_floor <= 0:
        raise ValueError("diagonal_floor must be positive")

    Q = np.asarray(Q_prof, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q_prof must be square")

    q_diag = np.diag(Q).copy()
    if np.any(q_diag <= diagonal_floor):
        raise ValueError("Q_prof diagonal entries must be positive for CORe")

    kappa = np.sqrt(2.0 * lambda_value * q_diag)
    tau = np.sqrt(2.0 * lambda_value / q_diag)
    return COReThresholds(q_diag=q_diag, kappa=kappa, tau=tau)


def solve_score_matching_core_miqp(
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
) -> ScoreMatchingCOReSolution:
    """Solve the CORe-strengthened L0 score-matching MIQP."""
    if lambda_value < 0:
        raise ValueError("lambda_value must be nonnegative")

    import gurobipy as gp
    from gurobipy import GRB

    formulation = build_gaussian_score_matching_formulation(
        x,
        assume_centered=assume_centered,
        edge_list=edge_list,
    )
    thresholds = core_thresholds(formulation.Q_prof, lambda_value)
    raw_big_m = data_derived_big_m(
        formulation.Q_prof,
        formulation.q_prof,
        scale=big_m_scale,
    )
    big_m = BigMBounds(
        values=np.maximum(raw_big_m.values, thresholds.tau),
        beta_continuous=raw_big_m.beta_continuous,
        continuous_lower_bound=raw_big_m.continuous_lower_bound,
    )

    n_edges = len(formulation.edge_list)
    m = formulation.sample_covariance.shape[0]
    Q = formulation.Q_prof
    q = formulation.q_prof

    model = gp.Model("gaussian_score_matching_core_miqp")
    model.Params.OutputFlag = 1 if output_flag else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)

    beta_vars = model.addVars(n_edges, lb=-GRB.INFINITY, name="beta")
    z0_vars = model.addVars(n_edges, vtype=GRB.BINARY, name="z0")
    zp_vars = model.addVars(n_edges, vtype=GRB.BINARY, name="zplus")
    zm_vars = model.addVars(n_edges, vtype=GRB.BINARY, name="zminus")

    for edge_idx in range(n_edges):
        bound = float(big_m.values[edge_idx])
        kappa = float(thresholds.kappa[edge_idx])
        tau = float(thresholds.tau[edge_idx])

        model.addConstr(z0_vars[edge_idx] + zp_vars[edge_idx] + zm_vars[edge_idx] == 1)

        residual = gp.LinExpr(-float(q[edge_idx]))
        for col_idx in range(n_edges):
            if Q[edge_idx, col_idx] != 0.0:
                residual += float(Q[edge_idx, col_idx]) * beta_vars[col_idx]
        model.addConstr(residual <= kappa * z0_vars[edge_idx])
        model.addConstr(residual >= -kappa * z0_vars[edge_idx])

        model.addConstr(
            beta_vars[edge_idx] >= tau * zp_vars[edge_idx] - bound * zm_vars[edge_idx]
        )
        model.addConstr(
            beta_vars[edge_idx] <= bound * zp_vars[edge_idx] - tau * zm_vars[edge_idx]
        )

    objective = gp.QuadExpr()
    for i in range(n_edges):
        if Q[i, i] != 0.0:
            objective += 0.5 * float(Q[i, i]) * beta_vars[i] * beta_vars[i]
        for j in range(i + 1, n_edges):
            if Q[i, j] != 0.0:
                objective += float(Q[i, j]) * beta_vars[i] * beta_vars[j]
        if q[i] != 0.0:
            objective += -float(q[i]) * beta_vars[i]
        objective += float(lambda_value) * (zp_vars[i] + zm_vars[i])

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    status = _gurobi_status_name(GRB, model.Status)
    has_solution = model.SolCount > 0
    if has_solution:
        beta = np.array([beta_vars[i].X for i in range(n_edges)], dtype=float)
        z0 = np.array([z0_vars[i].X >= 0.5 for i in range(n_edges)], dtype=bool)
        zp = np.array([zp_vars[i].X >= 0.5 for i in range(n_edges)], dtype=bool)
        zm = np.array([zm_vars[i].X >= 0.5 for i in range(n_edges)], dtype=bool)
        z = zp | zm
        precision = reconstruct_precision(beta, formulation)
        adjacency = adjacency_from_edge_indicators(z, formulation.edge_list, m)
    else:
        beta = np.zeros(n_edges, dtype=float)
        z0 = np.zeros(n_edges, dtype=bool)
        zp = np.zeros(n_edges, dtype=bool)
        zm = np.zeros(n_edges, dtype=bool)
        z = np.zeros(n_edges, dtype=bool)
        precision = reconstruct_precision(beta, formulation)
        adjacency = adjacency_from_edge_indicators(z, formulation.edge_list, m)

    return ScoreMatchingCOReSolution(
        formulation=formulation,
        big_m=big_m,
        thresholds=thresholds,
        beta=beta,
        z_inactive=z0,
        z_positive=zp,
        z_negative=zm,
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

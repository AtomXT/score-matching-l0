"""L0-regularized Gaussian score matching with Gurobi."""
from __future__ import annotations

import numpy as np

def build_gaussian_score_matching_formulation(
    x: np.ndarray, *,
    assume_centered: bool = False,
    edge_list: list[tuple[int, int]] | None = None,
) -> dict:
    x = np.asarray(x, dtype=float)
    if not assume_centered:
        x = x - x.mean(axis=0, keepdims=True)
    S = x.T @ x / x.shape[0]
    p = S.shape[0]
    edges = list(edge_list) if edge_list is not None else [
        (i, j) for i in range(p) for j in range(i + 1, p)
    ]
    B = np.zeros((p, len(edges)))
    C = np.zeros((len(edges), len(edges)))
    for e, (i, j) in enumerate(edges):
        B[i, e] = B[j, e] = S[i, j]
    for e, (i, j) in enumerate(edges):
        for f, (k, ell) in enumerate(edges):
            C[e, f] = ((i == ell) * S[j, k] + (i == k) * S[j, ell] +
                       (j == ell) * S[i, k] + (j == k) * S[i, ell])
    diagonal = np.diag(S)
    Q = C - B.T @ (B / diagonal[:, None])
    return {
        "sample_covariance": S,
        "edge_list": edges,
        "Q_prof": 0.5 * (Q + Q.T),
        "q_prof": -B.T @ (1.0 / diagonal),
        "Q_alpha_alpha": np.diag(diagonal),
        "Q_alpha_beta": B,
        "q_alpha": np.ones(p),
    }

def reconstruct_precision(beta: np.ndarray, formulation: dict) -> np.ndarray:
    precision = np.zeros_like(formulation["sample_covariance"])
    for value, (i, j) in zip(beta, formulation["edge_list"]):
        precision[i, j] = precision[j, i] = value
    alpha = np.linalg.solve(
        formulation["Q_alpha_alpha"],
        formulation["q_alpha"] - formulation["Q_alpha_beta"] @ beta,
    )
    np.fill_diagonal(precision, alpha)
    return precision

def adjacency_from_edge_indicators(z, edge_list, p):
    adjacency = np.zeros((p, p), dtype=bool)
    for selected, (i, j) in zip(z, edge_list):
        if selected:
            adjacency[i, j] = adjacency[j, i] = True
    return adjacency

def solve_big_m_relaxation(formulation, lambda_value, big_m_init, threads):
    import gurobipy as gp
    from gurobipy import GRB
    model = gp.Model("score_matching_big_m_relaxation")
    model.Params.OutputFlag = 0
    if threads is not None:
        model.Params.Threads = int(threads)
    m = len(formulation["edge_list"])
    beta = model.addMVar(m, lb=-GRB.INFINITY, name="beta")
    z = model.addMVar(m, lb=0.0, ub=1.0, name="z")
    model.addConstr(beta <= big_m_init * z)
    model.addConstr(beta >= -big_m_init * z)
    model.setObjective(
        0.5 * (beta @ formulation["Q_prof"] @ beta)
        - formulation["q_prof"] @ beta
        + lambda_value * z.sum()
    )
    model.optimize()
    objective = _safe_float(lambda: model.ObjVal) if model.SolCount else np.nan
    big_m = float(big_m_init)
    if model.SolCount:
        big_m = min(big_m, max(big_m * 1e-6, 2.0 * np.max(np.abs(beta.X))))
    return dict(big_m=big_m, beta=np.asarray(beta.X) if model.SolCount else np.zeros(m), objective=objective, runtime_seconds=model.Runtime)

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
) -> dict:
    import gurobipy as gp
    from gurobipy import GRB
    formulation = build_gaussian_score_matching_formulation(
        x, assume_centered=assume_centered, edge_list=edge_list)
    relaxation = solve_big_m_relaxation(formulation, lambda_value, big_m_init, threads)
    Q, q, big_m = formulation["Q_prof"], formulation["q_prof"], relaxation["big_m"]
    p, m = formulation["sample_covariance"].shape[0], len(formulation["edge_list"])
    model = gp.Model("gaussian_score_matching_miqp")
    model.Params.OutputFlag = int(output_flag)
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)
    beta_vars = model.addMVar(m, lb=-GRB.INFINITY, name="beta")
    z_vars = model.addMVar(m, vtype=GRB.BINARY, name="z")
    model.addConstr(beta_vars <= big_m * z_vars)
    model.addConstr(beta_vars >= -big_m * z_vars)
    model.setObjective(
        0.5 * (beta_vars @ Q @ beta_vars) - q @ beta_vars + lambda_value * z_vars.sum()
    )
    start_beta = relaxation["beta"]
    start_z = np.abs(start_beta) > 1e-9
    if 0.5 * start_beta @ Q @ start_beta - q @ start_beta + lambda_value * start_z.sum() >= 0.0:
        start_beta, start_z = np.zeros(m), np.zeros(m)
    beta_vars.Start, z_vars.Start = start_beta, start_z
    model.optimize()
    has_solution = model.SolCount > 0
    beta = np.asarray(beta_vars.X) if has_solution else np.zeros(m)
    z = np.asarray(z_vars.X) >= 0.5 if has_solution else np.zeros(m, dtype=bool)
    return {
        "formulation": formulation,
        "big_m": big_m,
        "big_m_relaxation_objective": relaxation["objective"],
        "big_m_relaxation_runtime_seconds": relaxation["runtime_seconds"],
        "beta": beta,
        "z": z,
        "precision": reconstruct_precision(beta, formulation),
        "adjacency": adjacency_from_edge_indicators(z, formulation["edge_list"], p),
        "has_solution": has_solution,
        "runtime_seconds": relaxation["runtime_seconds"] + model.Runtime,
        "objective": _safe_float(lambda: model.ObjVal) if has_solution else np.nan,
        "objective_bound": _safe_float(lambda: model.ObjBound),
        "mip_gap": _safe_float(lambda: model.MIPGap) if has_solution else np.nan,
        "nodes": _safe_float(lambda: model.NodeCount),
        "status": _gurobi_status_name(GRB, model.Status),
        "status_code": int(model.Status),
    }
def _safe_float(get_value) -> float:
    try:
        return float(get_value())
    except Exception:
        return float("nan")
def _gurobi_status_name(GRB, status_code: int) -> str:
    names = "LOADED OPTIMAL INFEASIBLE INF_OR_UNBD UNBOUNDED CUTOFF ITERATION_LIMIT NODE_LIMIT TIME_LIMIT SOLUTION_LIMIT INTERRUPTED NUMERIC SUBOPTIMAL INPROGRESS USER_OBJ_LIMIT WORK_LIMIT MEM_LIMIT"
    return next((name for name in names.split() if getattr(GRB, name, None) == status_code), f"STATUS_{status_code}")

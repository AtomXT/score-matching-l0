#!/usr/bin/env python3
"""Validate the Gaussian support-optimality MILP on a saved Gaussian instance.

The default test loads the 9-variable, 30-observation grid/hub example from
``data/gaussian``.  To keep an independent exact baseline feasible without reducing
the dimension, it uses the 18 candidate edges with the largest absolute empirical
correlations.  This leaves 262,144 possible supports and includes both true edges and
plausible nonedge decoys.  The script compares:

1. the full convex score-matching MIQP;
2. the support-optimality MILP from Corollary 5.3 of the manuscript; and
3. exhaustive enumeration of all edge supports followed by an exact stationarity solve.

The script exits nonzero if the optimized values, supports, coefficients, stationarity
conditions, or stationary objective identity disagree beyond the requested tolerance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "gaussian"
    / "m009_n030_comp01_side03_hubs01_deg04_seed000"
    / "dataset.npz"
)


@dataclass(frozen=True)
class QuadraticData:
    """Unique-coordinate representation of the full Gaussian quadratic."""

    sample_covariance: np.ndarray
    edges: tuple[tuple[int, int], ...]
    hessian: np.ndarray
    linear_term: np.ndarray
    dimension: int
    ridge: float

    @property
    def number_of_edges(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class Solution:
    """Common result record for enumeration, MIQP, and MILP solutions."""

    method: str
    objective: float
    alpha: np.ndarray
    beta: np.ndarray
    selected: np.ndarray
    runtime_seconds: float
    status: str
    search_nodes: float

    @property
    def selected_count(self) -> int:
        return int(np.sum(self.selected))


def build_full_quadratic(
    x: np.ndarray,
    *,
    ridge: float,
    candidate_edges: tuple[tuple[int, int], ...] | None = None,
    assume_centered: bool = False,
) -> QuadraticData:
    """Return ``H`` and ``g`` for ``theta' H theta / 2 - g' theta``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional data matrix")
    if not assume_centered:
        x = x - np.mean(x, axis=0, keepdims=True)
    sample_covariance = x.T @ x / x.shape[0]
    dimension = sample_covariance.shape[0]
    edges = (
        tuple(combinations(range(dimension), 2))
        if candidate_edges is None
        else tuple(candidate_edges)
    )
    if len(set(edges)) != len(edges):
        raise ValueError("candidate_edges contains duplicates")
    if any(not 0 <= left < right < dimension for left, right in edges):
        raise ValueError("candidate edges must satisfy 0 <= left < right < dimension")
    number_of_edges = len(edges)

    diagonal_block = np.diag(np.diag(sample_covariance))
    cross_block = np.zeros((dimension, number_of_edges), dtype=float)
    edge_block = np.zeros((number_of_edges, number_of_edges), dtype=float)

    for edge_index, (left, right) in enumerate(edges):
        cross_block[left, edge_index] = sample_covariance[left, right]
        cross_block[right, edge_index] = sample_covariance[left, right]

    for row, (i, j) in enumerate(edges):
        for column, (k, ell) in enumerate(edges):
            edge_block[row, column] = (
                (i == ell) * sample_covariance[j, k]
                + (i == k) * sample_covariance[j, ell]
                + (j == ell) * sample_covariance[i, k]
                + (j == k) * sample_covariance[i, ell]
            )
    edge_block += ridge * np.eye(number_of_edges)

    hessian = np.block(
        [
            [diagonal_block, cross_block],
            [cross_block.T, edge_block],
        ]
    )
    hessian = 0.5 * (hessian + hessian.T)
    linear_term = np.concatenate(
        (np.ones(dimension, dtype=float), np.zeros(number_of_edges, dtype=float))
    )
    return QuadraticData(
        sample_covariance=sample_covariance,
        edges=edges,
        hessian=hessian,
        linear_term=linear_term,
        dimension=dimension,
        ridge=ridge,
    )


def strongest_correlation_edges(
    x: np.ndarray,
    *,
    number_of_edges: int,
) -> tuple[tuple[int, int], ...]:
    """Select a deterministic candidate graph from absolute sample correlations."""
    centered = np.asarray(x, dtype=float) - np.mean(x, axis=0, keepdims=True)
    sample_covariance = centered.T @ centered / centered.shape[0]
    scale = np.sqrt(np.diag(sample_covariance))
    all_edges = tuple(combinations(range(centered.shape[1]), 2))
    ranked = sorted(
        all_edges,
        key=lambda edge: (
            -abs(sample_covariance[edge]) / (scale[edge[0]] * scale[edge[1]]),
            edge,
        ),
    )
    return tuple(sorted(ranked[:number_of_edges]))


def precision_from_solution(solution: Solution, quadratic: QuadraticData) -> np.ndarray:
    """Reconstruct the symmetric matrix parameter from unique coordinates."""
    precision = np.diag(solution.alpha.copy())
    for coefficient, (left, right) in zip(solution.beta, quadratic.edges):
        precision[left, right] = precision[right, left] = coefficient
    return precision


def quadratic_objective(
    alpha: np.ndarray,
    beta: np.ndarray,
    selected: np.ndarray,
    quadratic: QuadraticData,
    penalty: float,
) -> float:
    """Evaluate the full penalized quadratic in unique coordinates."""
    theta = np.concatenate((alpha, beta))
    return float(
        0.5 * theta @ quadratic.hessian @ theta
        - quadratic.linear_term @ theta
        + penalty * np.sum(selected)
    )


def solve_by_enumeration(
    quadratic: QuadraticData,
    *,
    penalty: float,
    coefficient_bound: float,
    feasibility_tolerance: float,
) -> tuple[Solution, float]:
    """Enumerate every support and solve its linear stationarity equations."""
    start = perf_counter()
    p = quadratic.dimension
    number_of_edges = quadratic.number_of_edges
    best: Solution | None = None
    largest_unconstrained_coefficient = 0.0

    for mask in range(1 << number_of_edges):
        selected = np.array(
            [(mask >> edge) & 1 for edge in range(number_of_edges)], dtype=bool
        )
        selected_edges = np.flatnonzero(selected)
        coordinates = np.concatenate((np.arange(p), p + selected_edges))
        restricted_hessian = quadratic.hessian[np.ix_(coordinates, coordinates)]
        restricted_linear = quadratic.linear_term[coordinates]
        try:
            restricted_theta = np.linalg.solve(restricted_hessian, restricted_linear)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "a support-restricted stationarity system is singular; "
                "increase n or use --ridge"
            ) from error

        alpha = restricted_theta[:p]
        beta = np.zeros(number_of_edges, dtype=float)
        beta[selected_edges] = restricted_theta[p:]
        if selected_edges.size:
            support_maximum = float(np.max(np.abs(beta[selected_edges])))
            largest_unconstrained_coefficient = max(
                largest_unconstrained_coefficient, support_maximum
            )
            if support_maximum > coefficient_bound + feasibility_tolerance:
                continue

        objective = quadratic_objective(
            alpha, beta, selected, quadratic, penalty
        )
        if best is None or objective < best.objective:
            best = Solution(
                method="enumeration",
                objective=objective,
                alpha=alpha,
                beta=beta,
                selected=selected,
                runtime_seconds=0.0,
                status="OPTIMAL",
                search_nodes=float(1 << number_of_edges),
            )

    if best is None:
        raise RuntimeError("no enumerated support satisfies the coefficient bound")
    elapsed = perf_counter() - start
    best = Solution(
        method=best.method,
        objective=best.objective,
        alpha=best.alpha,
        beta=best.beta,
        selected=best.selected,
        runtime_seconds=elapsed,
        status=best.status,
        search_nodes=best.search_nodes,
    )
    return best, largest_unconstrained_coefficient


def configure_gurobi_model(
    model: Any,
    *,
    output_flag: bool,
    time_limit: float | None,
    mip_gap: float,
    threads: int,
) -> None:
    """Apply common deterministic solver settings."""
    model.Params.OutputFlag = int(output_flag)
    model.Params.MIPGap = float(mip_gap)
    model.Params.Threads = int(threads)
    model.Params.Seed = 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)


def solve_full_miqp(
    quadratic: QuadraticData,
    *,
    penalty: float,
    coefficient_bound: float,
    output_flag: bool,
    time_limit: float | None,
    mip_gap: float,
    threads: int,
) -> Solution:
    """Solve the original full convex quadratic indicator formulation."""
    import gurobipy as gp
    from gurobipy import GRB

    p = quadratic.dimension
    number_of_edges = quadratic.number_of_edges
    hessian = quadratic.hessian
    diagonal_block = hessian[:p, :p]
    cross_block = hessian[:p, p:]
    edge_block = hessian[p:, p:]

    model = gp.Model("gaussian_score_matching_full_miqp_test")
    configure_gurobi_model(
        model,
        output_flag=output_flag,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
    )
    alpha = model.addMVar(p, lb=-GRB.INFINITY, name="alpha")
    beta = model.addMVar(
        number_of_edges,
        lb=-coefficient_bound,
        ub=coefficient_bound,
        name="beta",
    )
    selected = model.addMVar(number_of_edges, vtype=GRB.BINARY, name="selected")
    model.addConstr(beta <= coefficient_bound * selected)
    model.addConstr(beta >= -coefficient_bound * selected)
    model.setObjective(
        0.5 * (alpha @ diagonal_block @ alpha)
        + alpha @ cross_block @ beta
        + 0.5 * (beta @ edge_block @ beta)
        - alpha.sum()
        + penalty * selected.sum(),
        GRB.MINIMIZE,
    )
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"MIQP returned no solution; Gurobi status={model.Status}")
    return Solution(
        method="full_miqp",
        objective=float(model.ObjVal),
        alpha=np.asarray(alpha.X, dtype=float),
        beta=np.asarray(beta.X, dtype=float),
        selected=np.asarray(selected.X, dtype=float) >= 0.5,
        runtime_seconds=float(model.Runtime),
        status=_gurobi_status_name(GRB, model.Status),
        search_nodes=float(model.NodeCount),
    )


def solve_support_optimality_milp(
    quadratic: QuadraticData,
    *,
    penalty: float,
    coefficient_bound: float,
    output_flag: bool,
    time_limit: float | None,
    mip_gap: float,
    threads: int,
) -> Solution:
    """Solve the exact support-optimality linear formulation."""
    import gurobipy as gp
    from gurobipy import GRB

    p = quadratic.dimension
    number_of_edges = quadratic.number_of_edges
    hessian = quadratic.hessian
    diagonal_block = hessian[:p, :p]
    cross_block = hessian[:p, p:]
    edge_block = hessian[p:, p:]

    model = gp.Model("gaussian_score_matching_support_optimality_milp_test")
    configure_gurobi_model(
        model,
        output_flag=output_flag,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
    )
    alpha = model.addMVar(p, lb=-GRB.INFINITY, name="alpha")
    beta = model.addMVar(
        number_of_edges,
        lb=-coefficient_bound,
        ub=coefficient_bound,
        name="beta",
    )
    selected = model.addMVar(number_of_edges, vtype=GRB.BINARY, name="selected")

    model.addConstr(beta <= coefficient_bound * selected, name="off_upper")
    model.addConstr(beta >= -coefficient_bound * selected, name="off_lower")
    model.addConstr(
        diagonal_block @ alpha + cross_block @ beta == np.ones(p),
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
        -0.5 * alpha.sum() + penalty * selected.sum(), GRB.MINIMIZE
    )
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"MILP returned no solution; Gurobi status={model.Status}")
    return Solution(
        method="support_milp",
        objective=float(model.ObjVal),
        alpha=np.asarray(alpha.X, dtype=float),
        beta=np.asarray(beta.X, dtype=float),
        selected=np.asarray(selected.X, dtype=float) >= 0.5,
        runtime_seconds=float(model.Runtime),
        status=_gurobi_status_name(GRB, model.Status),
        search_nodes=float(model.NodeCount),
    )


def _gurobi_status_name(grb: Any, status_code: int) -> str:
    names = (
        "LOADED OPTIMAL INFEASIBLE INF_OR_UNBD UNBOUNDED CUTOFF ITERATION_LIMIT "
        "NODE_LIMIT TIME_LIMIT SOLUTION_LIMIT INTERRUPTED NUMERIC SUBOPTIMAL "
        "INPROGRESS USER_OBJ_LIMIT WORK_LIMIT MEM_LIMIT"
    )
    return next(
        (
            name
            for name in names.split()
            if getattr(grb, name, None) == status_code
        ),
        f"STATUS_{status_code}",
    )


def diagnostics(
    solution: Solution,
    quadratic: QuadraticData,
    *,
    penalty: float,
) -> dict[str, float]:
    """Return independent matrix and coordinate-form residual checks."""
    theta = np.concatenate((solution.alpha, solution.beta))
    gradient = quadratic.hessian @ theta - quadratic.linear_term
    active = solution.selected
    inactive = ~active
    coordinate_objective = quadratic_objective(
        solution.alpha,
        solution.beta,
        solution.selected,
        quadratic,
        penalty,
    )
    precision = precision_from_solution(solution, quadratic)
    matrix_objective = float(
        0.5
        * np.trace(
            precision @ quadratic.sample_covariance @ precision
        )
        - np.trace(precision)
        + 0.5 * quadratic.ridge * (solution.beta @ solution.beta)
        + penalty * np.sum(solution.selected)
    )
    stationary_objective = float(
        -0.5 * np.sum(solution.alpha) + penalty * np.sum(solution.selected)
    )
    matrix_edge_gradient = (
        precision @ quadratic.sample_covariance
        + quadratic.sample_covariance @ precision
    )
    direct_active_residuals = np.array(
        [
            matrix_edge_gradient[left, right]
            + quadratic.ridge * solution.beta[edge]
            for edge, (left, right) in enumerate(quadratic.edges)
            if active[edge]
        ],
        dtype=float,
    )
    return {
        "reported_vs_matrix": abs(solution.objective - matrix_objective),
        "coordinate_vs_matrix": abs(coordinate_objective - matrix_objective),
        "stationary_identity": abs(matrix_objective - stationary_objective),
        "diagonal_stationarity": float(np.max(np.abs(gradient[: quadratic.dimension]))),
        "active_stationarity": float(
            np.max(np.abs(gradient[quadratic.dimension :][active]))
            if np.any(active)
            else 0.0
        ),
        "matrix_diagonal_stationarity": float(
            np.max(
                np.abs(
                    np.diag(precision @ quadratic.sample_covariance) - 1.0
                )
            )
        ),
        "matrix_active_stationarity": float(
            np.max(np.abs(direct_active_residuals))
            if direct_active_residuals.size
            else 0.0
        ),
        "inactive_coefficient": float(
            np.max(np.abs(solution.beta[inactive])) if np.any(inactive) else 0.0
        ),
    }


def support_label(solution: Solution, quadratic: QuadraticData) -> str:
    """Return a compact one-based edge list for console output."""
    edges = [
        f"{left + 1}-{right + 1}"
        for keep, (left, right) in zip(solution.selected, quadratic.edges)
        if keep
    ]
    return "{" + ",".join(edges) + "}"


def assert_close_results(
    reference: Solution,
    candidate: Solution,
    *,
    absolute_tolerance: float,
) -> None:
    """Raise an informative assertion if two exact solutions disagree."""
    objective_gap = abs(reference.objective - candidate.objective)
    if objective_gap > absolute_tolerance:
        raise AssertionError(
            f"{candidate.method} objective differs from {reference.method} by "
            f"{objective_gap:.3e}"
        )
    if not np.array_equal(reference.selected, candidate.selected):
        raise AssertionError(
            f"{candidate.method} selected a different support despite matching values"
        )
    coefficient_gap = max(
        float(np.max(np.abs(reference.alpha - candidate.alpha))),
        float(np.max(np.abs(reference.beta - candidate.beta))),
    )
    if coefficient_gap > 10.0 * absolute_tolerance:
        raise AssertionError(
            f"{candidate.method} coefficients differ from {reference.method} by "
            f"{coefficient_gap:.3e}"
        )


def load_saved_instance(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the data and true adjacency used by the test."""
    with np.load(dataset_path) as arrays:
        missing = {"X", "adjacency"}.difference(arrays.files)
        if missing:
            raise ValueError(
                f"{dataset_path} is missing required arrays: {sorted(missing)}"
            )
        x = np.asarray(arrays["X"], dtype=float)
        adjacency = np.asarray(arrays["adjacency"], dtype=bool)
    if x.ndim != 2 or adjacency.shape != (x.shape[1], x.shape[1]):
        raise ValueError("dataset X and adjacency arrays have incompatible shapes")
    return x, adjacency


def run_saved_instance(args: argparse.Namespace) -> None:
    """Load, construct, and validate the deterministic saved-data test."""
    x, adjacency = load_saved_instance(args.dataset)
    possible_edges = x.shape[1] * (x.shape[1] - 1) // 2
    if args.candidate_edges > possible_edges:
        raise ValueError(
            f"--candidate-edges={args.candidate_edges} exceeds the {possible_edges} "
            "possible edges in this dataset"
        )
    candidate_edges = strongest_correlation_edges(
        x, number_of_edges=args.candidate_edges
    )
    quadratic = build_full_quadratic(
        x,
        ridge=args.ridge,
        candidate_edges=candidate_edges,
    )
    smallest_eigenvalue = float(np.linalg.eigvalsh(quadratic.hessian).min())
    if smallest_eigenvalue <= args.feasibility_tolerance:
        raise AssertionError(
            f"candidate Hessian is not positive definite (min eigenvalue "
            f"{smallest_eigenvalue:.3e}); increase n or use --ridge"
        )

    total_true_edges = int(np.sum(np.triu(adjacency, k=1)))
    candidate_truth = np.array(
        [adjacency[left, right] for left, right in candidate_edges], dtype=bool
    )
    try:
        dataset_label = args.dataset.relative_to(PROJECT_ROOT)
    except ValueError:
        dataset_label = args.dataset
    print(
        f"dataset={dataset_label} n={x.shape[0]} "
        f"p={x.shape[1]} candidates={quadratic.number_of_edges} "
        f"supports={1 << quadratic.number_of_edges:,} "
        f"true_candidates={int(np.sum(candidate_truth))}/{total_true_edges}"
    )

    enumeration, largest_coefficient = solve_by_enumeration(
        quadratic,
        penalty=args.penalty,
        coefficient_bound=args.coefficient_bound,
        feasibility_tolerance=args.feasibility_tolerance,
    )
    if largest_coefficient >= args.coefficient_bound - args.feasibility_tolerance:
        raise AssertionError(
            "the coefficient bound can bind a support-restricted minimizer: "
            f"largest={largest_coefficient:.6g}, bound={args.coefficient_bound:.6g}"
        )

    miqp = solve_full_miqp(
        quadratic,
        penalty=args.penalty,
        coefficient_bound=args.coefficient_bound,
        output_flag=args.output_flag,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
    )
    milp = solve_support_optimality_milp(
        quadratic,
        penalty=args.penalty,
        coefficient_bound=args.coefficient_bound,
        output_flag=args.output_flag,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
    )

    for solution in (miqp, milp):
        if solution.status != "OPTIMAL":
            raise AssertionError(
                f"{solution.method} did not prove optimality: {solution.status}"
            )
        assert_close_results(
            enumeration,
            solution,
            absolute_tolerance=args.absolute_tolerance,
        )

    for solution in (enumeration, miqp, milp):
        residuals = diagnostics(solution, quadratic, penalty=args.penalty)
        largest_residual = max(residuals.values())
        if largest_residual > 10.0 * args.absolute_tolerance:
            raise AssertionError(
                f"{solution.method} violates an identity or feasibility check: "
                f"{residuals}"
            )

    precision_gap = float(
        np.max(
            np.abs(
                precision_from_solution(miqp, quadratic)
                - precision_from_solution(milp, quadratic)
            )
        )
    )
    selected_true_positives = int(np.sum(milp.selected & candidate_truth))
    selected_false_positives = milp.selected_count - selected_true_positives
    print(
        f"support={support_label(milp, quadratic)} edges={milp.selected_count} "
        f"tp={selected_true_positives} fp={selected_false_positives} "
        f"objective={milp.objective:.10f}"
    )
    print(
        f"time_enumeration={enumeration.runtime_seconds:.3f}s "
        f"time_miqp={miqp.runtime_seconds:.3f}s nodes_miqp={miqp.search_nodes:.0f} "
        f"time_milp={milp.runtime_seconds:.3f}s nodes_milp={milp.search_nodes:.0f} "
        f"matrix_gap={precision_gap:.2e}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Saved .npz instance containing X and adjacency arrays.",
    )
    parser.add_argument(
        "--candidate-edges",
        type=int,
        default=18,
        help="Number of strongest-correlation candidate edges to retain.",
    )
    parser.add_argument(
        "--penalty", type=float, default=0.12, help="Penalty per selected edge."
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Optional ridge on unique off-diagonal coefficients.",
    )
    parser.add_argument(
        "--coefficient-bound",
        type=float,
        default=10.0,
        help="Symmetric bound used by both mixed-integer formulations.",
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=1e-9)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-9)
    parser.add_argument("--output-flag", action="store_true")
    args = parser.parse_args(argv)

    args.dataset = args.dataset.expanduser().resolve()
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    if not 1 <= args.candidate_edges <= 20:
        parser.error(
            "--candidate-edges must be between 1 and 20 because the exact baseline "
            "enumerates every support"
        )
    if args.penalty <= 0.0:
        parser.error("--penalty must be positive")
    if args.ridge < 0.0:
        parser.error("--ridge must be nonnegative")
    if args.coefficient_bound <= 0.0:
        parser.error("--coefficient-bound must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import gurobipy as gp

        gp.Env(empty=True).dispose()
    except ImportError as error:
        raise SystemExit(
            "gurobipy is required; install the project requirements before running "
            "this test"
        ) from error

    run_saved_instance(args)
    print(
        "PASS: support-optimality MILP matched the full MIQP and all-support "
        "enumeration on the saved instance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

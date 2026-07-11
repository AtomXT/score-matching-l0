#!/usr/bin/env python3
"""Generate controlled Gaussian graphical-model experiment instances.

For every requested configuration, the script constructs a graph, calibrates a
positive-definite precision matrix, and draws independent training, validation,
and test samples.  Each instance is stored in its own directory together with a
JSON record of the requested and achieved graph properties.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common import instance_name, parse_list, save_instance


PROJECT_DIR = Path(__file__).resolve().parents[1]


def chain_graph(p: int) -> np.ndarray:
    """Return the adjacency matrix of a path on ``p`` vertices."""
    adjacency = np.zeros((p, p), dtype=bool)
    indices = np.arange(p - 1)
    adjacency[indices, indices + 1] = True
    return adjacency | adjacency.T


def lattice_graph(p: int) -> np.ndarray:
    """Return a four-neighbor square lattice on ``p`` vertices."""
    side = int(round(np.sqrt(p)))
    if side * side != p:
        raise ValueError("lattice topology requires p to be a perfect square")
    adjacency = np.zeros((p, p), dtype=bool)
    for row in range(side):
        for column in range(side):
            node = row * side + column
            if column + 1 < side:
                adjacency[node, node + 1] = True
            if row + 1 < side:
                adjacency[node, node + side] = True
    return adjacency | adjacency.T


def banded_graph(p: int, target_degree: int) -> np.ndarray:
    """Return a banded graph whose interior degree is approximately the target."""
    half_bandwidth = max(1, int(round(target_degree / 2)))
    adjacency = np.zeros((p, p), dtype=bool)
    for offset in range(1, half_bandwidth + 1):
        indices = np.arange(p - offset)
        adjacency[indices, indices + offset] = True
    return adjacency | adjacency.T


def hub_graph(p: int, target_degree: int) -> np.ndarray:
    """Return a connected graph with one vertex of the requested degree."""
    if not (2 <= target_degree < p):
        raise ValueError("hub target_degree must be between 2 and p - 1")
    adjacency = chain_graph(p)
    if target_degree == 2:
        return adjacency
    # Vertex zero already has vertex one as a neighbor.  Add the remaining
    # spokes without removing the path that guarantees connectedness.
    for neighbor in range(2, target_degree + 1):
        adjacency[0, neighbor] = True
        adjacency[neighbor, 0] = True
    return adjacency


def erdos_renyi_graph(
    p: int, target_degree: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a connected sparse Erdos--Renyi-type graph.

    A random spanning tree is sampled first.  Remaining pairs are then sampled
    to make the expected total edge count equal to ``p * target_degree / 2``.
    This avoids discarding disconnected draws while retaining an explicit record
    of the achieved degree.
    """
    if target_degree < 2:
        raise ValueError("connected Erdos--Renyi graphs require target_degree >= 2")
    adjacency = np.zeros((p, p), dtype=bool)
    order = rng.permutation(p)
    for position in range(1, p):
        parent_position = int(rng.integers(0, position))
        left, right = int(order[position]), int(order[parent_position])
        adjacency[left, right] = adjacency[right, left] = True

    available = [
        (i, j)
        for i in range(p)
        for j in range(i + 1, p)
        if not adjacency[i, j]
    ]
    target_edges = min(p * (p - 1) // 2, round(p * target_degree / 2))
    probability = max(0.0, (target_edges - (p - 1)) / max(1, len(available)))
    for i, j in available:
        if rng.random() < probability:
            adjacency[i, j] = adjacency[j, i] = True
    return adjacency


def scale_free_graph(
    p: int, target_degree: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a Barabasi--Albert graph without an external graph package."""
    attachments = max(1, min(p - 2, int(round(target_degree / 2))))
    initial = attachments + 1
    adjacency = np.zeros((p, p), dtype=bool)
    adjacency[:initial, :initial] = True
    np.fill_diagonal(adjacency, False)

    for node in range(initial, p):
        degrees = adjacency[:node, :node].sum(axis=1).astype(float)
        probabilities = degrees / degrees.sum()
        neighbors = rng.choice(node, size=attachments, replace=False, p=probabilities)
        adjacency[node, neighbors] = True
        adjacency[neighbors, node] = True
    return adjacency


def build_graph(
    topology: str,
    p: int,
    target_degree: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch to one of the controlled graph constructions."""
    builders = {
        "chain": lambda: chain_graph(p),
        "lattice": lambda: lattice_graph(p),
        "banded": lambda: banded_graph(p, target_degree),
        "hub": lambda: hub_graph(p, target_degree),
        "erdos_renyi": lambda: erdos_renyi_graph(p, target_degree, rng),
        "scale_free": lambda: scale_free_graph(p, target_degree, rng),
    }
    try:
        adjacency = builders[topology]()
    except KeyError as exc:
        raise ValueError(f"unsupported topology: {topology}") from exc
    np.fill_diagonal(adjacency, False)
    return adjacency


def _precision_for_weight_floor(
    adjacency: np.ndarray,
    signs: np.ndarray,
    uniforms: np.ndarray,
    weight_floor: float,
    target_condition: float,
) -> tuple[np.ndarray, float, float]:
    """Construct a precision matrix with the requested condition number."""
    weights = adjacency * signs * (weight_floor + (1.0 - weight_floor) * uniforms)
    weights = np.triu(weights, k=1)
    weights = weights + weights.T
    eigenvalues = np.linalg.eigvalsh(weights)
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    diagonal = (largest - target_condition * smallest) / (target_condition - 1.0)
    precision = weights + diagonal * np.eye(adjacency.shape[0])
    achieved_condition = float(np.linalg.cond(precision))
    achieved_signal = float(np.min(np.abs(weights[adjacency])) / diagonal)
    return precision, achieved_signal, achieved_condition


def calibrated_precision(
    adjacency: np.ndarray,
    *,
    target_signal: float,
    target_condition: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Calibrate edge heterogeneity while fixing the spectral condition number.

    For a fixed signed edge matrix ``W``, adding ``c I`` permits an exact choice
    of the condition number.  The lower endpoint of the edge-weight interval is
    then selected on a deterministic grid to make the minimum partial
    correlation as close as possible to ``target_signal``.  Both achieved
    quantities are recorded because the two targets need not be jointly
    attainable on every graph.
    """
    if not (0 < target_signal < 1):
        raise ValueError("target_signal must lie strictly between zero and one")
    if target_condition <= 1:
        raise ValueError("target_condition must be greater than one")

    p = adjacency.shape[0]
    signs = np.zeros((p, p), dtype=float)
    uniforms = np.zeros((p, p), dtype=float)
    upper_i, upper_j = np.where(np.triu(adjacency, k=1))
    signs[upper_i, upper_j] = rng.choice([-1.0, 1.0], size=len(upper_i))
    uniforms[upper_i, upper_j] = rng.random(len(upper_i))

    candidates: list[tuple[float, float, float, np.ndarray]] = []
    for weight_floor in np.linspace(0.05, 1.0, 20):
        precision, signal, condition = _precision_for_weight_floor(
            adjacency,
            signs,
            uniforms,
            float(weight_floor),
            target_condition,
        )
        candidates.append((abs(signal - target_signal), signal, condition, precision))
    error, signal, condition, precision = min(candidates, key=lambda item: item[0])

    covariance = np.linalg.inv(precision)
    standard_deviations = np.sqrt(np.diag(covariance))
    scale = np.diag(standard_deviations)
    standardized_precision = scale @ precision @ scale
    standardized_covariance = np.linalg.inv(standardized_precision)
    achieved_condition = float(np.linalg.cond(standardized_precision))
    partial = np.abs(standardized_precision[adjacency]) / np.sqrt(
        np.repeat(np.diag(standardized_precision), adjacency.sum(axis=1))
        * standardized_precision.diagonal()[np.where(adjacency)[1]]
    )
    diagnostics = {
        "achieved_signal": float(partial.min()),
        "signal_calibration_error": float(error),
        "condition_before_standardization": float(condition),
        "achieved_condition": achieved_condition,
        "minimum_eigenvalue": float(np.linalg.eigvalsh(standardized_precision)[0]),
    }
    return standardized_precision, {**diagnostics, "covariance": standardized_covariance}


def generate_one(
    *,
    study: str,
    topology: str,
    p: int,
    n: int,
    target_degree: int,
    target_signal: float,
    target_condition: float,
    rep: int,
    base_seed: int,
    output_root: Path,
    overwrite: bool,
    fixed_graph: bool = False,
) -> dict[str, object]:
    """Generate and save one complete Monte Carlo instance."""
    topology_codes = {
        "chain": 1,
        "lattice": 2,
        "banded": 3,
        "hub": 4,
        "erdos_renyi": 5,
        "scale_free": 6,
    }
    graph_rep = 0 if fixed_graph else rep
    # Excluding n, signal, and condition from the graph seed makes the design
    # paired when one of those factors is varied.  The sample seed retains every
    # factor, so the observations remain independent across configurations.
    graph_seed = int(
        np.random.SeedSequence(
            [base_seed, graph_rep, p, target_degree, topology_codes[topology]]
        ).generate_state(1)[0]
    )
    sample_seed = int(
        np.random.SeedSequence(
            [
                base_seed,
                rep,
                p,
                n,
                target_degree,
                topology_codes[topology],
                int(1000 * target_signal),
                int(10 * target_condition),
            ]
        ).generate_state(1)[0]
    )
    graph_rng = np.random.default_rng(graph_seed)
    sample_rng = np.random.default_rng(sample_seed)
    adjacency = build_graph(topology, p, target_degree, graph_rng)
    precision, diagnostics = calibrated_precision(
        adjacency,
        target_signal=target_signal,
        target_condition=target_condition,
        rng=graph_rng,
    )
    covariance = diagnostics.pop("covariance")

    degrees = adjacency.sum(axis=1)
    metadata: dict[str, object] = {
        "study": study,
        "topology": topology,
        "p": p,
        "n": n,
        "target_degree": target_degree,
        "achieved_average_degree": float(degrees.mean()),
        "achieved_maximum_degree": int(degrees.max()),
        "target_signal": target_signal,
        "target_condition": target_condition,
        "true_edges": int(np.triu(adjacency, k=1).sum()),
        "rep": rep,
        "graph_mode": "fixed" if fixed_graph else "random",
        "graph_rep": graph_rep,
        "base_seed": base_seed,
        "graph_seed": graph_seed,
        "sample_seed": sample_seed,
        **diagnostics,
    }
    directory = output_root / study / instance_name(metadata)
    if (directory / "dataset.npz").exists() and not overwrite:
        return {"directory": str(directory), **metadata, "generation_status": "existing"}

    train = sample_rng.multivariate_normal(np.zeros(p), covariance, size=n)
    validation = sample_rng.multivariate_normal(np.zeros(p), covariance, size=n)
    test = sample_rng.multivariate_normal(np.zeros(p), covariance, size=n)
    save_instance(
        directory,
        train=train,
        validation=validation,
        test=test,
        covariance=covariance,
        precision=precision,
        adjacency=adjacency,
        metadata=metadata,
    )
    return {"directory": str(directory), **metadata, "generation_status": "generated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--topology-list", default="erdos_renyi")
    parser.add_argument("--p-list", default="20")
    parser.add_argument("--n-list", default="100")
    parser.add_argument("--degree-list", default="4")
    parser.add_argument("--signal-list", default="0.2")
    parser.add_argument("--condition-list", default="10")
    parser.add_argument("--rep-list", default="0")
    parser.add_argument("--base-seed", type=int, default=2027)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "data" / "gaussian_experiments",
    )
    parser.add_argument("--manifest-name", default="manifest")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fixed-graph",
        action="store_true",
        help="Hold the population graph fixed across replications while resampling observations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for topology in parse_list(args.topology_list, str):
        for p in parse_list(args.p_list, int):
            for n in parse_list(args.n_list, int):
                for degree in parse_list(args.degree_list, int):
                    for signal in parse_list(args.signal_list, float):
                        for condition in parse_list(args.condition_list, float):
                            for rep in parse_list(args.rep_list, int):
                                record = generate_one(
                                    study=args.study,
                                    topology=topology,
                                    p=p,
                                    n=n,
                                    target_degree=degree,
                                    target_signal=signal,
                                    target_condition=condition,
                                    rep=rep,
                                    base_seed=args.base_seed,
                                    output_root=args.output_root,
                                    overwrite=args.overwrite,
                                    fixed_graph=args.fixed_graph,
                                )
                                records.append(record)
                                print(f"{record['generation_status']}: {record['directory']}")

    manifest = args.output_root / args.study / f"{args.manifest_name}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest} with {len(records)} instances")


if __name__ == "__main__":
    main()

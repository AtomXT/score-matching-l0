"""Reusable Gaussian graphical-model constructions for simulation studies."""

from __future__ import annotations

import numpy as np


def chain_graph(p: int) -> np.ndarray:
    """Return the adjacency matrix of a path on ``p`` vertices."""
    adjacency = np.zeros((p, p), dtype=bool)
    indices = np.arange(p - 1)
    adjacency[indices, indices + 1] = True
    return adjacency | adjacency.T


def lattice_graph(p: int) -> np.ndarray:
    """Return a four-neighbor square lattice on ``p`` vertices."""
    side = int(round(np.sqrt(p)))
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
    """Return a banded graph whose interior degree is near the target."""
    half_bandwidth = max(1, int(round(target_degree / 2)))
    adjacency = np.zeros((p, p), dtype=bool)
    for offset in range(1, half_bandwidth + 1):
        indices = np.arange(p - offset)
        adjacency[indices, indices + offset] = True
    return adjacency | adjacency.T


def hub_graph(p: int, target_degree: int) -> np.ndarray:
    """Return a connected graph with one vertex of the requested degree."""
    adjacency = chain_graph(p)
    for neighbor in range(2, target_degree + 1):
        adjacency[0, neighbor] = True
        adjacency[neighbor, 0] = True
    return adjacency


def erdos_renyi_graph(
    p: int, target_degree: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a connected sparse Erdos--Renyi-type graph."""
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
    """Generate a Barabasi--Albert graph without an external dependency."""
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
    """Dispatch to a controlled graph construction."""
    builders = {
        "chain": lambda: chain_graph(p),
        "lattice": lambda: lattice_graph(p),
        "banded": lambda: banded_graph(p, target_degree),
        "hub": lambda: hub_graph(p, target_degree),
        "erdos_renyi": lambda: erdos_renyi_graph(p, target_degree, rng),
        "scale_free": lambda: scale_free_graph(p, target_degree, rng),
    }
    adjacency = builders[topology]()
    np.fill_diagonal(adjacency, False)
    return adjacency


def _precision_for_weight_floor(
    adjacency: np.ndarray,
    signs: np.ndarray,
    uniforms: np.ndarray,
    weight_floor: float,
    target_condition: float,
) -> tuple[np.ndarray, float, float]:
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
) -> tuple[np.ndarray, dict[str, float | np.ndarray]]:
    """Calibrate edge heterogeneity and the spectral condition number."""
    p = adjacency.shape[0]
    signs = np.zeros((p, p), dtype=float)
    uniforms = np.zeros((p, p), dtype=float)
    upper_i, upper_j = np.where(np.triu(adjacency, k=1))
    signs[upper_i, upper_j] = rng.choice([-1.0, 1.0], size=len(upper_i))
    uniforms[upper_i, upper_j] = rng.random(len(upper_i))

    candidates: list[tuple[float, float, float, np.ndarray]] = []
    for weight_floor in np.linspace(0.05, 1.0, 20):
        precision, signal, condition = _precision_for_weight_floor(
            adjacency, signs, uniforms, float(weight_floor), target_condition
        )
        candidates.append((abs(signal - target_signal), signal, condition, precision))
    error, _, condition, precision = min(candidates, key=lambda item: item[0])

    covariance = np.linalg.inv(precision)
    scale = np.diag(np.sqrt(np.diag(covariance)))
    standardized_precision = scale @ precision @ scale
    standardized_covariance = np.linalg.inv(standardized_precision)
    rows, columns = np.where(adjacency)
    partial = np.abs(standardized_precision[rows, columns]) / np.sqrt(
        standardized_precision[rows, rows] * standardized_precision[columns, columns]
    )
    diagnostics: dict[str, float | np.ndarray] = {
        "achieved_signal": float(partial.min()),
        "signal_calibration_error": float(error),
        "condition_before_standardization": float(condition),
        "achieved_condition": float(np.linalg.cond(standardized_precision)),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(standardized_precision)[0]),
        "covariance": standardized_covariance,
    }
    return standardized_precision, diagnostics


def lattice_with_hubs_graph(
    num_components: int,
    side_length: int,
    hubs_per_component: int,
    hub_degree: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct the lattice-with-hubs design of Lin, Drton, and Shojaie."""
    component_size = side_length**2
    adjacency = np.zeros((num_components * component_size,) * 2, dtype=bool)
    for component in range(num_components):
        start = component * component_size
        stop = start + component_size
        adjacency[start:stop, start:stop] = lattice_graph(component_size)
        hubs = rng.choice(np.arange(start, stop), size=hubs_per_component, replace=False)
        hub_set = set(hubs)
        for hub in hubs:
            neighbors = set(np.flatnonzero(adjacency[hub]))
            candidates = [
                node
                for node in range(start, stop)
                if node != hub and node not in neighbors and node not in hub_set
            ]
            number_to_add = hub_degree - len(neighbors)
            added = rng.choice(candidates, size=number_to_add, replace=False)
            adjacency[hub, added] = adjacency[added, hub] = True
    return adjacency


def generate_lattice_with_hubs(
    *,
    n: int = 600,
    num_components: int = 10,
    side_length: int = 10,
    hubs_per_component: int = 3,
    hub_degree: int = 20,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the Gaussian simulation design used by Lin et al. (2016)."""
    rng = np.random.default_rng(seed)
    covariance, precision, adjacency = lattice_with_hubs_population(
        num_components=num_components,
        side_length=side_length,
        hubs_per_component=hubs_per_component,
        hub_degree=hub_degree,
        rng=rng,
    )
    x = rng.multivariate_normal(np.zeros(covariance.shape[0]), covariance, size=n)
    return x, covariance, precision, adjacency


def lattice_with_hubs_population(
    *,
    num_components: int,
    side_length: int,
    hubs_per_component: int,
    hub_degree: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the population covariance in the Lin et al. simulation."""
    adjacency = lattice_with_hubs_graph(
        num_components, side_length, hubs_per_component, hub_degree, rng
    )
    weights = np.zeros(adjacency.shape, dtype=float)
    weights[adjacency] = rng.uniform(0.5, 1.0, size=int(adjacency.sum()))
    row_sums = np.abs(weights).sum(axis=1)
    precision = np.zeros_like(weights)
    nonzero = row_sums > 0
    precision[nonzero] = weights[nonzero] / (1.5 * row_sums[nonzero, None])
    precision = (precision + precision.T) / 2.0
    np.fill_diagonal(precision, 1.0)
    covariance = np.linalg.inv(precision)
    standard_deviations = np.sqrt(np.diag(covariance))
    covariance = covariance / np.outer(standard_deviations, standard_deviations)
    scale = np.diag(standard_deviations)
    precision = scale @ precision @ scale
    return covariance, precision, adjacency


def generate_exact_edge_gaussian(
    *,
    n: int,
    p: int,
    number_of_edges: int,
    seed: int,
    weight_low: float = 0.25,
    weight_high: float = 0.55,
    diagonal_buffer: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a connected Gaussian graph with an exact number of edges."""
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((p, p), dtype=bool)
    order = rng.permutation(p)
    for left, right in zip(order[:-1], order[1:]):
        adjacency[left, right] = adjacency[right, left] = True

    available = [(i, j) for i in range(p) for j in range(i + 1, p)]
    rng.shuffle(available)
    current_edges = p - 1
    for i, j in available:
        if current_edges >= number_of_edges:
            break
        if adjacency[i, j]:
            continue
        adjacency[i, j] = adjacency[j, i] = True
        current_edges += 1

    precision = np.zeros((p, p), dtype=float)
    for i, j in np.argwhere(np.triu(adjacency, k=1)):
        weight = rng.choice([-1.0, 1.0]) * rng.uniform(weight_low, weight_high)
        precision[i, j] = precision[j, i] = weight
    np.fill_diagonal(precision, np.abs(precision).sum(axis=1) + diagonal_buffer)
    covariance = np.linalg.inv(precision)
    x = rng.multivariate_normal(np.zeros(p), covariance, size=n)
    return x, covariance, precision, adjacency

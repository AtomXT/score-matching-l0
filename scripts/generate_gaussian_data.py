"""Generate Gaussian graphical model data from Lin, Drton, and Shojaie (2016).

This script follows the Gaussian data procedure in Section 4.1 of
"Estimation of High-Dimensional Graphical Models Using Regularized Score
Matching".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_lattice_with_hubs(
    num_components: int,
    side_length: int,
    hubs_per_component: int,
    hub_degree: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a symmetric adjacency matrix for lattice components with hubs."""
    component_size = side_length * side_length
    m = num_components * component_size
    adjacency = np.zeros((m, m), dtype=bool)

    for component in range(num_components):
        start = component * component_size
        stop = start + component_size

        for row in range(side_length):
            for col in range(side_length):
                node = start + row * side_length + col
                if col + 1 < side_length:
                    neighbor = node + 1
                    adjacency[node, neighbor] = True
                    adjacency[neighbor, node] = True
                if row + 1 < side_length:
                    neighbor = node + side_length
                    adjacency[node, neighbor] = True
                    adjacency[neighbor, node] = True

        hubs = rng.choice(np.arange(start, stop), size=hubs_per_component, replace=False)
        hub_set = set(hubs)
        for hub in hubs:
            current_neighbors = set(np.flatnonzero(adjacency[hub]))
            if len(current_neighbors) > hub_degree:
                raise ValueError("hub_degree is smaller than a selected hub's lattice degree")

            possible_neighbors = [
                node
                for node in range(start, stop)
                if node != hub and node not in current_neighbors and node not in hub_set
            ]
            edges_to_add = hub_degree - len(current_neighbors)

            if edges_to_add > len(possible_neighbors):
                raise ValueError(
                    "hub_degree is too large for the component size after lattice edges"
                )

            new_neighbors = rng.choice(
                possible_neighbors, size=edges_to_add, replace=False
            )
            adjacency[hub, new_neighbors] = True
            adjacency[new_neighbors, hub] = True

    np.fill_diagonal(adjacency, False)
    return adjacency


def adjacency_to_precision(
    adjacency: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Convert adjacency to a sparse diagonally dominant matrix."""
    weights = np.zeros(adjacency.shape, dtype=float)
    weights[adjacency] = rng.uniform(0.5, 1.0, size=int(adjacency.sum()))

    row_sums = np.abs(weights).sum(axis=1)
    precision = np.zeros_like(weights)
    nonzero_rows = row_sums > 0
    precision[nonzero_rows] = (
        weights[nonzero_rows] / (1.5 * row_sums[nonzero_rows, None])
    )

    precision = (precision + precision.T) / 2.0
    np.fill_diagonal(precision, 1.0)
    return precision


def precision_to_correlation(precision: np.ndarray) -> np.ndarray:
    """Invert the precision matrix and scale it to a correlation matrix."""
    covariance = np.linalg.inv(precision)
    standard_deviations = np.sqrt(np.diag(covariance))
    return covariance / np.outer(standard_deviations, standard_deviations)


def generate_gaussian_data(
    n: int = 600,
    num_components: int = 10,
    side_length: int = 10,
    hubs_per_component: int = 3,
    hub_degree: int = 20,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate samples and the population graph matrices."""
    rng = np.random.default_rng(seed)

    adjacency = build_lattice_with_hubs(
        num_components=num_components,
        side_length=side_length,
        hubs_per_component=hubs_per_component,
        hub_degree=hub_degree,
        rng=rng,
    )
    precision = adjacency_to_precision(adjacency, rng)
    sigma = precision_to_correlation(precision)
    x = rng.multivariate_normal(mean=np.zeros(sigma.shape[0]), cov=sigma, size=n)

    return x, sigma, precision, adjacency


def save_dataset(
    out: Path,
    x: np.ndarray,
    sigma: np.ndarray,
    precision: np.ndarray,
    adjacency: np.ndarray,
    params: dict[str, int | None],
) -> None:
    """Save the generated arrays as one compressed NumPy archive."""
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=x,
        Sigma=sigma,
        precision=precision,
        adjacency=adjacency.astype(np.int8),
        params_json=json.dumps(params, sort_keys=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--num-components", type=int, default=10)
    parser.add_argument("--side-length", type=int, default=10)
    parser.add_argument("--hubs-per-component", type=int, default=3)
    parser.add_argument("--hub-degree", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/gaussian/gaussian_n600_m1000_seed0.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = {
        "n": args.n,
        "num_components": args.num_components,
        "side_length": args.side_length,
        "hubs_per_component": args.hubs_per_component,
        "hub_degree": args.hub_degree,
        "seed": args.seed,
    }
    x, sigma, precision, adjacency = generate_gaussian_data(**params)
    save_dataset(args.out, x, sigma, precision, adjacency, params)
    print(f"Saved {args.out} with X shape {x.shape}")


if __name__ == "__main__":
    main()

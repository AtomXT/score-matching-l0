#!/usr/bin/env python3
"""Generate controlled Gaussian graphical-model experiment instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common import instance_name, parse_list, save_instance
from experiments.gaussian_models import (
    build_graph,
    calibrated_precision,
    lattice_with_hubs_population,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
TOPOLOGY_CODES = {
    "chain": 1,
    "lattice": 2,
    "banded": 3,
    "hub": 4,
    "erdos_renyi": 5,
    "scale_free": 6,
    "lattice_hubs": 7,
}


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
    """Generate and save one Gaussian graphical-model instance."""
    graph_rep = 0 if fixed_graph else rep
    topology_code = TOPOLOGY_CODES[topology]
    # Excluding n, signal, and condition pairs factor levels through a common
    # graph. The sample seed retains every factor, so samples remain independent.
    graph_seed = int(
        np.random.SeedSequence(
            [base_seed, graph_rep, p, target_degree, topology_code]
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
                topology_code,
                int(1000 * target_signal),
                int(10 * target_condition),
            ]
        ).generate_state(1)[0]
    )

    graph_rng = np.random.default_rng(graph_seed)
    sample_rng = np.random.default_rng(sample_seed)
    if topology == "lattice_hubs":
        num_components = p // 100
        covariance, precision, adjacency = lattice_with_hubs_population(
            num_components=num_components,
            side_length=10,
            hubs_per_component=3,
            hub_degree=20,
            rng=graph_rng,
        )
        partial = np.abs(precision[adjacency]) / np.sqrt(
            np.outer(np.diag(precision), np.diag(precision))[adjacency]
        )
        diagnostics = {
            "achieved_signal": float(partial.min()),
            "achieved_condition": float(np.linalg.cond(precision)),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(precision)[0]),
            "data_generation_procedure": "Lin_Drton_Shojaie_2016_section_4.1_componentwise",
            "components": num_components,
            "component_side_length": 10,
            "hubs_per_component": 3,
            "hub_degree": 20,
        }
    else:
        adjacency = build_graph(topology, p, target_degree, graph_rng)
        precision, diagnostics = calibrated_precision(
            adjacency,
            target_signal=target_signal,
            target_condition=target_condition,
            rng=graph_rng,
        )
        covariance = np.asarray(diagnostics.pop("covariance"), dtype=float)

    degrees = adjacency.sum(axis=1)
    metadata: dict[str, object] = {
        "study": study,
        "topology": topology,
        "p": p,
        "n": n,
        "achieved_average_degree": float(degrees.mean()),
        "achieved_maximum_degree": int(degrees.max()),
        "true_edges": int(np.triu(adjacency, k=1).sum()),
        "rep": rep,
        "graph_mode": "fixed" if fixed_graph else "random",
        "graph_rep": graph_rep,
        "base_seed": base_seed,
        "graph_seed": graph_seed,
        "sample_seed": sample_seed,
        **diagnostics,
    }
    if topology != "lattice_hubs":
        metadata.update(
            target_degree=target_degree,
            target_signal=target_signal,
            target_condition=target_condition,
        )
    directory = output_root / study / instance_name(metadata)
    if (directory / "dataset.npz").exists() and not overwrite:
        return {"directory": str(directory), **metadata, "generation_status": "existing"}

    train = sample_rng.multivariate_normal(np.zeros(p), covariance, size=n)
    validation = test = None
    if topology != "lattice_hubs":
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
        help="Hold the population graph fixed while resampling observations.",
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

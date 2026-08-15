#!/usr/bin/env python3
"""Generate multivariate-t data for the non-Gaussian robustness ROC study.

The design keeps the Gaussian ROC graph and covariance calibration fixed and
changes only the observation law to a multivariate t distribution with three
degrees of freedom, as in the manuscript's misspecification experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import instance_name, parse_list, save_instance
from experiments.gaussian_models import build_graph, calibrated_precision
from experiments.generate_gaussian_experiments import TOPOLOGY_CODES


PROJECT_DIR = Path(__file__).resolve().parents[1]
STUDY = "robustness_roc"
OUTPUT_ROOT = PROJECT_DIR / "data" / "nongaussian_experiments"
NUMBER_OF_REPLICATIONS = 10
BASE_SEED = 2027
TOPOLOGY = "erdos_renyi"
P = 500
N = 1000
TARGET_DEGREE = 4
TARGET_SIGNAL = 0.20
TARGET_CONDITION = 10.0
DEGREES_OF_FREEDOM = 3.0
SUPPORTED_TOPOLOGIES = tuple(
    topology for topology in sorted(TOPOLOGY_CODES) if topology != "lattice_hubs"
)


def _generation_seeds(
    *,
    rep: int,
    topology: str,
    p: int,
    n: int,
    base_seed: int,
) -> tuple[int, int, int]:
    """Return paired graph, Gaussian-reference, and t-sample seeds."""
    topology_code = TOPOLOGY_CODES[topology]
    graph_seed = int(
        np.random.SeedSequence(
            [base_seed, rep, p, TARGET_DEGREE, topology_code]
        ).generate_state(1)[0]
    )
    gaussian_reference_sample_seed = int(
        np.random.SeedSequence(
            [
                base_seed,
                rep,
                p,
                n,
                TARGET_DEGREE,
                topology_code,
                int(1000 * TARGET_SIGNAL),
                int(10 * TARGET_CONDITION),
            ]
        ).generate_state(1)[0]
    )
    sample_seed = int(
        np.random.SeedSequence(
            [
                gaussian_reference_sample_seed,
                int(1000 * DEGREES_OF_FREEDOM),
                0x7433524F,
            ]
        ).generate_state(1)[0]
    )
    return graph_seed, gaussian_reference_sample_seed, sample_seed


def generate_one(
    *,
    rep: int,
    topology: str = TOPOLOGY,
    p: int = P,
    n: int = N,
    base_seed: int = BASE_SEED,
    output_root: Path = OUTPUT_ROOT,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate and save one covariance-matched multivariate-t instance."""
    graph_seed, gaussian_reference_sample_seed, sample_seed = _generation_seeds(
        rep=rep,
        topology=topology,
        p=p,
        n=n,
        base_seed=base_seed,
    )
    graph_rng = np.random.default_rng(graph_seed)
    adjacency = build_graph(topology, p, TARGET_DEGREE, graph_rng)
    precision, diagnostics = calibrated_precision(
        adjacency,
        target_signal=TARGET_SIGNAL,
        target_condition=TARGET_CONDITION,
        rng=graph_rng,
    )
    covariance = np.asarray(diagnostics.pop("covariance"), dtype=float)
    degrees = adjacency.sum(axis=1)
    metadata: dict[str, object] = {
        "study": STUDY,
        "distribution": "multivariate_t",
        "degrees_of_freedom": DEGREES_OF_FREEDOM,
        "covariance_normalization": (
            "t scale = ((df - 2) / df) * Sigma, so Cov(X) = Sigma"
        ),
        "misspecified_gaussian_graph_target": True,
        "graph_target": (
            "support of inverse covariance; for multivariate t this is not in "
            "general a conditional-independence graph"
        ),
        "topology": topology,
        "p": p,
        "n": n,
        "achieved_average_degree": float(degrees.mean()),
        "achieved_maximum_degree": int(degrees.max()),
        "true_edges": int(np.triu(adjacency, k=1).sum()),
        "rep": rep,
        "graph_mode": "random",
        "graph_rep": rep,
        "base_seed": base_seed,
        "graph_seed": graph_seed,
        "gaussian_reference_sample_seed": gaussian_reference_sample_seed,
        "sample_seed": sample_seed,
        "target_degree": TARGET_DEGREE,
        "target_signal": TARGET_SIGNAL,
        "target_condition": TARGET_CONDITION,
        **diagnostics,
    }
    directory = output_root / STUDY / instance_name(metadata)
    if (directory / "dataset.npz").exists() and not overwrite:
        existing_metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        return {
            "directory": str(directory),
            **existing_metadata,
            "generation_status": "existing",
        }

    sample_rng = np.random.default_rng(sample_seed)
    scale = covariance * (
        (DEGREES_OF_FREEDOM - 2.0) / DEGREES_OF_FREEDOM
    )
    gaussian_draws = sample_rng.standard_normal((n, p)) @ np.linalg.cholesky(scale).T
    radial_scales = np.sqrt(
        sample_rng.chisquare(DEGREES_OF_FREEDOM, size=n)
        / DEGREES_OF_FREEDOM
    )
    x = gaussian_draws / radial_scales[:, None]
    save_instance(
        directory,
        x=x,
        covariance=covariance,
        precision=precision,
        adjacency=adjacency,
        metadata=metadata,
    )
    return {"directory": str(directory), **metadata, "generation_status": "generated"}


def generate_all(
    *,
    replications: Iterable[int] = range(NUMBER_OF_REPLICATIONS),
    topology: str = TOPOLOGY,
    p: int = P,
    n: int = N,
    base_seed: int = BASE_SEED,
    output_root: Path = OUTPUT_ROOT,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Generate the requested replications and write their design manifests."""
    replication_ids = list(replications)
    records = []
    for rep in replication_ids:
        record = generate_one(
            rep=rep,
            topology=topology,
            p=p,
            n=n,
            base_seed=base_seed,
            output_root=output_root,
            overwrite=overwrite,
        )
        records.append(record)
        print(
            f"{record['generation_status']}: {record['topology']} "
            f"p={record['p']} n={record['n']} rep={record['rep']}",
            flush=True,
        )

    study_directory = output_root / STUDY
    study_directory.mkdir(parents=True, exist_ok=True)
    design = {
        "study": STUDY,
        "purpose": "Gaussian-method robustness ROC under heavy-tail misspecification",
        "distribution": "multivariate_t",
        "degrees_of_freedom": DEGREES_OF_FREEDOM,
        "topology": topology,
        "p": p,
        "n": n,
        "number_of_replications": len(replication_ids),
        "base_seed": base_seed,
        "target_degree": TARGET_DEGREE,
        "target_signal": TARGET_SIGNAL,
        "target_condition": TARGET_CONDITION,
    }
    (study_directory / "design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (study_directory / "manifest.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {study_directory / 'design.json'}", flush=True)
    print(
        f"Wrote {study_directory / 'manifest.json'} with {len(records)} instances",
        flush=True,
    )
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--rep-list",
        default=",".join(map(str, range(NUMBER_OF_REPLICATIONS))),
    )
    parser.add_argument("--topology", choices=SUPPORTED_TOPOLOGIES, default=TOPOLOGY)
    parser.add_argument("--p", type=int, default=P)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generate_all(
        replications=parse_list(args.rep_list, int),
        topology=args.topology,
        p=args.p,
        n=args.n,
        base_seed=args.base_seed,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

"""Generate Gaussian synthetic data with GraphL0Learn's data utilities."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from project_paths import PROJECT_DIR, load_src_module

utils = load_src_module("utils")

generate_graphl0learn_gaussian_data = utils.generate_graphl0learn_gaussian_data
graphl0learn_dataset_name = utils.graphl0learn_dataset_name
save_dataset_folder = utils.save_dataset_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument(
        "--model",
        choices=["banded_Toeplitz_precision", "uniform"],
        default="banded_Toeplitz_precision",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--half-bandwidth", type=int, default=2)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--cond", type=float, default=2.0)
    parser.add_argument("--p0", type=float, default=0.2)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = {
        "source": "graphl0learn",
        "n": args.n,
        "m": args.m,
        "model": args.model,
        "seed": args.seed,
        "normalize": not args.no_normalize,
        "p0": args.p0,
        "cond": args.cond,
        "half_bandwidth": args.half_bandwidth,
        "rho": args.rho,
    }
    x, sigma, precision, adjacency = generate_graphl0learn_gaussian_data(
        n=args.n,
        m=args.m,
        model=args.model,
        seed=args.seed,
        normalize=not args.no_normalize,
        p0=args.p0,
        cond=args.cond,
        half_bandwidth=args.half_bandwidth,
        rho=args.rho,
    )
    out_dir = Path(
        os.path.join(
            PROJECT_DIR,
            "data",
            "graphl0learn",
            graphl0learn_dataset_name(
                n=args.n,
                m=args.m,
                model=args.model,
                seed=args.seed,
                half_bandwidth=args.half_bandwidth,
                rho=args.rho,
                cond=args.cond,
                p0=args.p0,
            ),
        )
    )
    save_dataset_folder(out_dir, params, x, sigma, precision, adjacency)
    true_edges = int(adjacency.sum() // 2)
    print(f"Saved {out_dir} with X shape {x.shape} and {true_edges} true edges")


if __name__ == "__main__":
    main()

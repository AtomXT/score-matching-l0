"""Shared helpers for score-matching experiments."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(current_dir)


def load_file_module(module_name: str, module_path: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_l0bnb2_package():
    package_name = "l0bnb2"
    if package_name in sys.modules:
        return sys.modules[package_name]

    package_dir = os.path.join(PROJECT_DIR, "src", package_name)
    init_path = os.path.join(package_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[package_dir],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


_gaussian_data = load_file_module(
    "generate_gaussian_data",
    os.path.join(PROJECT_DIR, "scripts", "generate_gaussian_data.py"),
)
generate_gaussian_data = _gaussian_data.generate_gaussian_data
save_dataset = _gaussian_data.save_dataset


@dataclass(frozen=True)
class ExperimentDataset:
    directory: Path
    data: dict[str, Any]
    row_info: dict[str, Any]


def gaussian_dataset_name(
    *,
    n: int,
    num_components: int,
    side_length: int,
    hubs_per_component: int,
    hub_degree: int,
    seed: int,
) -> str:
    m = num_components * side_length * side_length
    return (
        f"m{m:03d}_n{n:03d}_comp{num_components:02d}_side{side_length:02d}_"
        f"hubs{hubs_per_component:02d}_deg{hub_degree:02d}_seed{seed:03d}"
    )


def exact_gaussian_dataset_name(
    *,
    n: int,
    m: int,
    target_edges: int,
    seed: int,
) -> str:
    return f"m{m:03d}_n{n:03d}_edges{target_edges:02d}_seed{seed:03d}"


def graphl0learn_dataset_name(
    *,
    n: int,
    m: int,
    model: str,
    seed: int,
    half_bandwidth: int | None = None,
    rho: float | None = None,
    cond: float | None = None,
    p0: float | None = None,
) -> str:
    if model == "banded_Toeplitz_precision":
        return (
            f"m{m:03d}_n{n:03d}_graphl0learn_banded_bw{half_bandwidth:02d}_"
            f"rho{int(round(float(rho) * 100)):03d}_cond{int(round(float(cond))):02d}_"
            f"seed{seed:03d}"
        )
    if model == "uniform":
        return (
            f"m{m:03d}_n{n:03d}_graphl0learn_uniform_"
            f"p0{int(round(float(p0) * 1000)):03d}_cond{int(round(float(cond))):02d}_"
            f"seed{seed:03d}"
        )
    raise ValueError(f"unsupported GraphL0Learn model: {model}")


def dataset_dir(base_dir: Path, params: dict[str, int]) -> Path:
    if "m" in params:
        return base_dir / exact_gaussian_dataset_name(**params)
    return base_dir / gaussian_dataset_name(**params)


def generate_exact_m_gaussian_data(
    *,
    n: int,
    m: int,
    target_edges: int,
    seed: int,
    weight_low: float = 0.25,
    weight_high: float = 0.55,
    diagonal_buffer: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Gaussian data with exactly m nodes and a connected sparse graph."""
    max_edges = m * (m - 1) // 2
    if not (m - 1 <= target_edges <= max_edges):
        raise ValueError("target_edges must be between m - 1 and m * (m - 1) / 2")

    rng = np.random.default_rng(seed)
    adjacency = np.zeros((m, m), dtype=bool)

    order = rng.permutation(m)
    for left, right in zip(order[:-1], order[1:]):
        adjacency[left, right] = True
        adjacency[right, left] = True

    possible_edges = [(i, j) for i in range(m) for j in range(i + 1, m)]
    rng.shuffle(possible_edges)
    current_edges = int(np.triu(adjacency, k=1).sum())
    for i, j in possible_edges:
        if current_edges >= target_edges:
            break
        if adjacency[i, j]:
            continue
        adjacency[i, j] = True
        adjacency[j, i] = True
        current_edges += 1

    precision = np.zeros((m, m), dtype=float)
    edge_indices = np.argwhere(np.triu(adjacency, k=1))
    for i, j in edge_indices:
        sign = rng.choice([-1.0, 1.0])
        weight = sign * rng.uniform(weight_low, weight_high)
        precision[i, j] = weight
        precision[j, i] = weight

    np.fill_diagonal(precision, np.abs(precision).sum(axis=1) + diagonal_buffer)
    sigma = np.linalg.inv(precision)
    x = rng.multivariate_normal(mean=np.zeros(m), cov=sigma, size=n)
    return x, sigma, precision, adjacency


def generate_graphl0learn_gaussian_data(
    *,
    n: int,
    m: int,
    model: str = "banded_Toeplitz_precision",
    seed: int = 0,
    normalize: bool = True,
    p0: float = 0.2,
    cond: float = 2.0,
    half_bandwidth: int = 2,
    rho: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Gaussian data using GraphL0Learn's own synthetic generator."""
    generate_synthetic = load_graphl0learn_generate_synthetic()

    kwargs: dict[str, Any]
    if model == "uniform":
        kwargs = {"p0": p0, "cond": cond}
    elif model == "banded_Toeplitz_precision":
        kwargs = {
            "half_bandwidth": half_bandwidth,
            "rho": rho,
            "cond": cond,
        }
    else:
        raise ValueError(f"unsupported GraphL0Learn model: {model}")

    x, sigma, precision = generate_synthetic(
        n,
        m,
        model=model,
        normalize=normalize,
        rng=seed,
        **kwargs,
    )
    adjacency = np.abs(precision) > 1e-12
    np.fill_diagonal(adjacency, False)
    return x, sigma, precision, adjacency


def load_graphl0learn_generate_synthetic():
    """Load GraphL0Learn's generator without importing the full BnB package."""
    module_path = os.path.join(PROJECT_DIR, "src", "l0bnb2", "data_utils.py")
    module = load_file_module("graphl0learn_data_utils", module_path)
    return module.generate_synthetic


def save_dataset_folder(
    out_dir: Path,
    params: dict[str, Any],
    x: np.ndarray,
    sigma: np.ndarray,
    precision: np.ndarray,
    adjacency: np.ndarray,
) -> None:
    dataset_path = out_dir / "dataset.npz"
    metadata_path = out_dir / "metadata.json"
    save_dataset(dataset_path, x, sigma, precision, adjacency, params)
    metadata = {
        **params,
        "m": int(x.shape[1]),
        "dataset_path": str(dataset_path),
        "true_edges": int(np.triu(adjacency.astype(bool), k=1).sum()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_dataset_folder(out_dir: Path) -> dict[str, Any]:
    dataset_path = out_dir / "dataset.npz"
    metadata_path = out_dir / "metadata.json"
    archive = np.load(dataset_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return {
        "X": archive["X"],
        "Sigma": archive["Sigma"],
        "precision": archive["precision"],
        "adjacency": archive["adjacency"].astype(bool),
        "params_json": str(archive["params_json"]),
        "metadata": metadata,
    }


def load_or_create_gaussian_dataset(
    base_dir: Path,
    params: dict[str, int],
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Create or load a named Gaussian dataset folder."""
    out_dir = dataset_dir(base_dir, params)
    dataset_path = out_dir / "dataset.npz"

    if force or not dataset_path.exists():
        if "m" in params:
            x, sigma, precision, adjacency = generate_exact_m_gaussian_data(**params)
        else:
            x, sigma, precision, adjacency = generate_gaussian_data(**params)
        save_dataset_folder(out_dir, params, x, sigma, precision, adjacency)

    return out_dir, load_dataset_folder(out_dir)


def load_existing_experiment_dataset(
    *,
    source: str,
    n: int,
    m: int | None,
    target_edges: int | None,
    seed: int,
    num_components: int,
    side_length: int,
    hubs_per_component: int,
    hub_degree: int,
    graphl0learn_model: str,
    p0: float,
    cond: float,
    half_bandwidth: int,
    rho: float,
) -> ExperimentDataset:
    """Load an existing dataset folder without generating data."""
    directory = experiment_dataset_directory(
        source=source,
        n=n,
        m=m,
        target_edges=target_edges,
        seed=seed,
        num_components=num_components,
        side_length=side_length,
        hubs_per_component=hubs_per_component,
        hub_degree=hub_degree,
        graphl0learn_model=graphl0learn_model,
        p0=p0,
        cond=cond,
        half_bandwidth=half_bandwidth,
        rho=rho,
    )
    dataset_path = directory / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path.resolve()}. Run the appropriate data "
            "generation script first, then rerun this experiment."
        )
    data = load_dataset_folder(directory)
    return ExperimentDataset(directory, data, dataset_row_info(directory, data))


def experiment_dataset_directory(
    *,
    source: str,
    n: int,
    m: int | None,
    target_edges: int | None,
    seed: int,
    num_components: int,
    side_length: int,
    hubs_per_component: int,
    hub_degree: int,
    graphl0learn_model: str,
    p0: float,
    cond: float,
    half_bandwidth: int,
    rho: float,
) -> Path:
    """Return the dataset folder implied by runner/generator parameters."""
    if source == "graphl0learn":
        if m is None:
            raise ValueError("m is required for GraphL0Learn-generated data")
        return default_data_dir(source) / graphl0learn_dataset_name(
            n=n,
            m=m,
            model=graphl0learn_model,
            seed=seed,
            half_bandwidth=half_bandwidth,
            rho=rho,
            cond=cond,
            p0=p0,
        )
    if source == "exact":
        if m is None:
            raise ValueError("m is required for exact Gaussian data")
        max_edges = m * (m - 1) // 2
        edge_count = target_edges or max(m - 1, round(0.27 * max_edges))
        return dataset_dir(
            default_data_dir(source),
            {"n": n, "m": m, "target_edges": edge_count, "seed": seed},
        )
    if source == "lattice":
        return dataset_dir(
            default_data_dir(source),
            {
                "n": n,
                "num_components": num_components,
                "side_length": side_length,
                "hubs_per_component": hubs_per_component,
                "hub_degree": hub_degree,
                "seed": seed,
            },
        )
    raise ValueError(f"unsupported dataset source: {source}")


def dataset_row_info(directory: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Return common CSV dataset columns for a saved dataset."""
    metadata = data.get("metadata", {})
    x = data["X"]
    adjacency = data["adjacency"].astype(bool)
    return {
        "dataset_dir": str(directory),
        "dataset_type": metadata.get("source", metadata.get("model", "gaussian")),
        "n": int(x.shape[0]),
        "m": int(x.shape[1]),
        "target_edges": metadata.get("true_edges", true_edge_count(adjacency)),
        "num_components": metadata.get("num_components", ""),
        "side_length": metadata.get("side_length", ""),
        "hubs_per_component": metadata.get("hubs_per_component", ""),
        "hub_degree": metadata.get("hub_degree", ""),
        "model": metadata.get("model", ""),
        "half_bandwidth": metadata.get("half_bandwidth", ""),
        "rho": metadata.get("rho", ""),
        "cond": metadata.get("cond", ""),
        "p0": metadata.get("p0", ""),
        "normalize": metadata.get("normalize", ""),
        "seed": metadata.get("seed", ""),
    }


def true_edge_count(adjacency: np.ndarray) -> int:
    return int(np.triu(adjacency.astype(bool), k=1).sum())


def default_data_dir(source: str) -> Path:
    if source == "graphl0learn":
        return Path(os.path.join(PROJECT_DIR, "data", "graphl0learn"))
    return Path(os.path.join(PROJECT_DIR, "data", "gaussian"))


def add_dataset_runner_arguments(parser: Any, default_results_csv: Path) -> None:
    """Add common dataset/result arguments to an argparse parser."""
    parser.add_argument("--data-source", choices=["graphl0learn", "exact", "lattice"], default="graphl0learn")
    parser.add_argument("--results-csv", type=Path, default=default_results_csv)
    parser.add_argument("--overwrite-results", dest="overwrite_results", action="store_true")
    parser.add_argument("--append-results", dest="overwrite_results", action="store_false")
    parser.set_defaults(overwrite_results=True)

    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--target-edges", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model", default="banded_Toeplitz_precision")
    parser.add_argument("--half-bandwidth", type=int, default=2)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--cond", type=float, default=2.0)
    parser.add_argument("--p0", type=float, default=0.2)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)

    parser.add_argument("--num-components", type=int, default=1)
    parser.add_argument("--side-length", type=int, default=5)
    parser.add_argument("--hubs-per-component", type=int, default=2)
    parser.add_argument("--hub-degree", type=int, default=8)


def load_dataset_from_runner_args(args: Any) -> ExperimentDataset:
    return load_existing_experiment_dataset(
        source=args.data_source,
        n=args.n,
        m=args.m,
        target_edges=args.target_edges,
        seed=args.seed,
        num_components=args.num_components,
        side_length=args.side_length,
        hubs_per_component=args.hubs_per_component,
        hub_degree=args.hub_degree,
        graphl0learn_model=args.model,
        p0=args.p0,
        cond=args.cond,
        half_bandwidth=args.half_bandwidth,
        rho=args.rho,
    )


def parse_float_values(values: str | None, default: float) -> list[float]:
    if values is None:
        return [default]
    parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    return parsed or [default]


def parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def blank_metrics() -> dict[str, str]:
    return {
        "TP": "",
        "FP": "",
        "TN": "",
        "FN": "",
        "TPR": "",
        "FPR": "",
        "precision": "",
        "recall": "",
        "F1": "",
        "selected_edges": "",
        "true_edges": "",
    }


def result_row(
    dataset: ExperimentDataset,
    blank_columns: tuple[str, ...] = (),
    **values: Any,
) -> dict[str, Any]:
    return {
        "status": "error",
        "error_message": "",
        **dataset.row_info,
        **{column: "" for column in blank_columns},
        **blank_metrics(),
        **values,
    }


def prepare_results_file(csv_path: Path, overwrite: bool) -> None:
    if overwrite and csv_path.exists():
        csv_path.unlink()


def normalize_prediction(prediction: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """Convert a predicted matrix to an undirected boolean adjacency matrix."""
    pred = np.asarray(prediction)
    if pred.ndim != 2 or pred.shape[0] != pred.shape[1]:
        raise ValueError("prediction must be a square matrix")

    adjacency = (np.abs(pred) > threshold) | (np.abs(pred.T) > threshold)
    np.fill_diagonal(adjacency, False)
    return adjacency


def support_metrics(true_adjacency: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute undirected support-recovery metrics on upper-triangle edges."""
    truth = normalize_prediction(true_adjacency, threshold=0)
    pred = normalize_prediction(predicted)
    if truth.shape != pred.shape:
        raise ValueError("true and predicted adjacency matrices must have the same shape")

    upper = np.triu(np.ones(truth.shape, dtype=bool), k=1)
    truth_edges = truth[upper]
    pred_edges = pred[upper]

    tp = int(np.sum(truth_edges & pred_edges))
    fp = int(np.sum(~truth_edges & pred_edges))
    tn = int(np.sum(~truth_edges & ~pred_edges))
    fn = int(np.sum(truth_edges & ~pred_edges))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": tpr,
        "FPR": fpr,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "selected_edges": int(pred_edges.sum()),
        "true_edges": int(truth_edges.sum()),
    }


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def fit_graph_l0_bnb(
    x: np.ndarray,
    *,
    l0: float,
    l2: float,
    m_bound: float,
    gap_tol: float,
    time_limit: float,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run GraphL0Learn's BNBTree and return the fitted precision matrix."""
    l0bnb2 = load_l0bnb2_package()
    BNBTree = l0bnb2.BNBTree
    heuristic_solve = l0bnb2.heuristic_solve
    preprocess = l0bnb2.preprocess

    _, _, _, _, y, _ = preprocess(x, assume_centered=False)
    start = time.perf_counter()
    theta_approx, _, _, _ = heuristic_solve(y, l0, l2, m_bound)
    tree = BNBTree(x)
    solution = tree.solve(
        l0,
        l2,
        m_bound,
        warm_start=theta_approx,
        verbose=verbose,
        gap_tol=gap_tol,
        time_limit=time_limit,
    )
    runtime = time.perf_counter() - start

    return {
        "Theta": solution.Theta,
        "runtime_seconds": runtime,
        "solver_time_seconds": float(solution.sol_time),
        "objective": float(solution.cost),
        "gap": float(solution.gap),
        "nodes": int(tree.number_of_nodes),
    }

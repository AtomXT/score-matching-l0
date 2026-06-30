"""Shared helpers for score-matching experiments."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_gaussian_data import generate_gaussian_data, save_dataset


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


def save_dataset_folder(
    out_dir: Path,
    params: dict[str, int],
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


def load_or_create_gaussian_dataset(
    base_dir: Path,
    params: dict[str, int],
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Create or load a named Gaussian dataset folder."""
    out_dir = dataset_dir(base_dir, params)
    dataset_path = out_dir / "dataset.npz"
    metadata_path = out_dir / "metadata.json"

    if force or not dataset_path.exists():
        if "m" in params:
            x, sigma, precision, adjacency = generate_exact_m_gaussian_data(**params)
        else:
            x, sigma, precision, adjacency = generate_gaussian_data(**params)
        save_dataset_folder(out_dir, params, x, sigma, precision, adjacency)

    archive = np.load(dataset_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    data = {
        "X": archive["X"],
        "Sigma": archive["Sigma"],
        "precision": archive["precision"],
        "adjacency": archive["adjacency"].astype(bool),
        "params_json": str(archive["params_json"]),
        "metadata": metadata,
    }
    return out_dir, data


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
    from l0bnb2 import BNBTree, heuristic_solve, preprocess

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

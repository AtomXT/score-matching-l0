"""Shared data, metric, and CSV helpers for the experiment drivers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np


T = TypeVar("T")


def parse_list(text: str, cast: Callable[[str], T]) -> list[T]:
    """Parse one value or a comma-separated list, with optional brackets."""
    cleaned = text.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    values = [part.strip() for part in cleaned.split(",") if part.strip()]
    return [cast(value) for value in values]


def instance_name(metadata: dict[str, Any]) -> str:
    """Return a stable, human-readable folder name for one instance."""
    if metadata["topology"] == "lattice_hubs":
        return (
            f"topology=lattice_hubs_p={metadata['p']:04d}_n={metadata['n']:04d}_"
            f"graph={metadata.get('graph_mode', 'random')}_rep={metadata['rep']:03d}"
        )
    return (
        f"topology={metadata['topology']}_p={metadata['p']:03d}_n={metadata['n']:04d}_"
        f"degree={metadata['target_degree']:02d}_signal={metadata['target_signal']:.3f}_"
        f"kappa={metadata['target_condition']:.1f}_graph={metadata.get('graph_mode', 'random')}_"
        f"rep={metadata['rep']:03d}"
    )


def save_instance(
    directory: Path,
    *,
    x: np.ndarray,
    covariance: np.ndarray,
    precision: np.ndarray,
    adjacency: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Save one complete synthetic instance and its generation record."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X": x,
        "Sigma": covariance,
        "precision": precision,
        "adjacency": adjacency.astype(np.int8),
    }
    # Write each file atomically.  This matters on Quest when two panel jobs
    # encounter the one configuration shared by the primary-study panels.
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as file:
        temporary_dataset = Path(file.name)
        np.savez_compressed(file, **arrays)
    os.replace(temporary_dataset, directory / "dataset.npz")

    with tempfile.NamedTemporaryFile(
        mode="w", dir=directory, suffix=".json", encoding="utf-8", delete=False
    ) as file:
        temporary_metadata = Path(file.name)
        file.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_metadata, directory / "metadata.json")


def load_instance(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load arrays and metadata saved by :func:`save_instance`."""
    with np.load(directory / "dataset.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays["adjacency"] = arrays["adjacency"].astype(bool)
    return arrays, metadata


def append_result(csv_path: Path, row: dict[str, Any], columns: list[str]) -> None:
    """Append one result immediately using a fixed schema."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def support_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Compute graph-recovery metrics using each undirected edge once."""
    truth = np.asarray(truth, dtype=bool).copy()
    estimate = np.asarray(estimate, dtype=bool).copy()
    truth = truth | truth.T
    estimate = estimate | estimate.T
    np.fill_diagonal(truth, False)
    np.fill_diagonal(estimate, False)

    upper = np.triu(np.ones(truth.shape, dtype=bool), k=1)
    y = truth[upper]
    yhat = estimate[upper]
    tp = int(np.sum(y & yhat))
    fp = int(np.sum(~y & yhat))
    tn = int(np.sum(~y & ~yhat))
    fn = int(np.sum(y & ~yhat))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "selected_edges": tp + fp,
        "true_edges": tp + fn,
        "exact_recovery": float(fp + fn == 0),
        "shd": fp + fn,
        "TPR": recall,
        "FPR": fp / (fp + tn) if fp + tn else 0.0,
        "precision": precision,
        "recall": recall,
        "F1": 2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "MCC": (tp * tn - fp * fn) / denominator if denominator else 0.0,
    }


def estimation_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Return matrix-estimation errors used in the Gaussian experiments."""
    difference = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    truth_norm = np.linalg.norm(truth, ord="fro")
    return {
        "relative_frobenius_error": np.linalg.norm(difference, ord="fro") / truth_norm,
        "operator_error": np.linalg.norm(difference, ord=2),
        "max_entry_error": np.max(np.abs(difference)),
    }


def graphical_lasso_screen(
    x: np.ndarray,
    alpha: float = 0.01,
    *,
    max_iter: int = 1_000,
    tolerance: float = 1e-4,
    support_tolerance: float = 1e-8,
) -> list[tuple[int, int]]:
    """Return edges retained by a lightly regularized graphical lasso fit."""
    if alpha <= 0:
        raise ValueError("graphical lasso screening alpha must be positive")

    centered = np.asarray(x, dtype=float) - np.mean(x, axis=0, keepdims=True)
    sample_covariance = centered.T @ centered / centered.shape[0]
    return _graphical_lasso_screen_from_covariance(
        sample_covariance,
        alpha=alpha,
        max_iter=max_iter,
        tolerance=tolerance,
        support_tolerance=support_tolerance,
    )


def spearman_graphical_lasso_screen(
    x: np.ndarray,
    alpha: float = 0.01,
    *,
    max_iter: int = 1_000,
    tolerance: float = 1e-4,
    support_tolerance: float = 1e-8,
) -> list[tuple[int, int]]:
    """Screen with graphical lasso applied to a rank-based correlation."""
    if alpha <= 0:
        raise ValueError("graphical lasso screening alpha must be positive")

    from scipy.stats import rankdata

    ranks = rankdata(np.asarray(x, dtype=float), axis=0)
    spearman_correlation = np.corrcoef(ranks, rowvar=False)
    np.fill_diagonal(spearman_correlation, 1.0)
    return _graphical_lasso_screen_from_covariance(
        spearman_correlation,
        alpha=alpha,
        max_iter=max_iter,
        tolerance=tolerance,
        support_tolerance=support_tolerance,
    )


def _graphical_lasso_screen_from_covariance(
    covariance: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    tolerance: float,
    support_tolerance: float,
) -> list[tuple[int, int]]:
    """Return the support from a graphical-lasso covariance fit."""
    from sklearn.covariance import graphical_lasso

    _, precision = graphical_lasso(
        covariance,
        alpha=alpha,
        max_iter=max_iter,
        tol=tolerance,
    )
    return [
        (i, j)
        for i in range(precision.shape[0])
        for j in range(i + 1, precision.shape[1])
        if abs(precision[i, j]) > support_tolerance
    ]

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
    if not values:
        raise ValueError("list arguments must contain at least one value")
    return [cast(value) for value in values]


def instance_name(metadata: dict[str, Any]) -> str:
    """Return a stable, human-readable folder name for one instance."""
    return (
        f"topology={metadata['topology']}_p={metadata['p']:03d}_n={metadata['n']:04d}_"
        f"degree={metadata['target_degree']:02d}_signal={metadata['target_signal']:.3f}_"
        f"kappa={metadata['target_condition']:.1f}_graph={metadata.get('graph_mode', 'random')}_"
        f"rep={metadata['rep']:03d}"
    )


def save_instance(
    directory: Path,
    *,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    covariance: np.ndarray,
    precision: np.ndarray,
    adjacency: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Save one complete synthetic instance and its generation record."""
    directory.mkdir(parents=True, exist_ok=True)
    # Write each file atomically.  This matters on Quest when two panel jobs
    # encounter the one configuration shared by the primary-study panels.
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as file:
        temporary_dataset = Path(file.name)
        np.savez_compressed(
            file,
            X=train,
            X_train=train,
            X_validation=validation,
            X_test=test,
            Sigma=covariance,
            precision=precision,
            adjacency=adjacency.astype(np.int8),
        )
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
    if truth.shape != estimate.shape:
        raise ValueError("truth and estimate must have the same shape")

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


def heldout_scores(x: np.ndarray, precision: np.ndarray) -> dict[str, float]:
    """Compute held-out Hyvarinen score and Gaussian negative log-likelihood."""
    centered = x - x.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / centered.shape[0]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score = 0.5 * np.trace(precision @ covariance @ precision) - np.trace(precision)
        sign, logdet = np.linalg.slogdet(precision)
        trace_term = np.trace(covariance @ precision)
    nll = (
        0.5 * (trace_term - logdet)
        if sign > 0 and np.isfinite(logdet) and np.isfinite(trace_term)
        else np.nan
    )
    return {"heldout_score": float(score), "heldout_gaussian_nll": float(nll)}


def correlation_screen(x: np.ndarray, number_of_edges: int) -> list[tuple[int, int]]:
    """Return the largest empirical-correlation pairs for heuristic screening."""
    p = x.shape[1]
    total = p * (p - 1) // 2
    if number_of_edges >= total:
        return [(i, j) for i in range(p) for j in range(i + 1, p)]
    if number_of_edges <= 0:
        raise ValueError("number_of_edges must be positive")

    correlation = np.corrcoef(x, rowvar=False)
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    order = np.argsort([-abs(correlation[i, j]) for i, j in pairs], kind="stable")
    return [pairs[index] for index in order[:number_of_edges]]

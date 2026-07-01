import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)


def load_src_module(module_name: str):
    module_path = os.path.join(project_dir, "src", f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


utils = load_src_module("utils")
append_csv_row = utils.append_csv_row
dataset_dir = utils.dataset_dir
generate_exact_m_gaussian_data = utils.generate_exact_m_gaussian_data
normalize_prediction = utils.normalize_prediction
support_metrics = utils.support_metrics


class UtilsTest(unittest.TestCase):
    def test_dataset_folder_name_contains_size_info(self):
        params = {
            "n": 150,
            "num_components": 1,
            "side_length": 5,
            "hubs_per_component": 2,
            "hub_degree": 8,
            "seed": 0,
        }

        path = dataset_dir(Path("data/gaussian"), params)

        self.assertEqual(
            path,
            Path("data/gaussian/m025_n150_comp01_side05_hubs02_deg08_seed000"),
        )

    def test_exact_dataset_folder_name_contains_size_info(self):
        params = {"n": 500, "m": 10, "target_edges": 12, "seed": 0}

        path = dataset_dir(Path("data/gaussian"), params)

        self.assertEqual(path, Path("data/gaussian/m010_n500_edges12_seed000"))

    def test_exact_m_gaussian_data_has_requested_graph_size(self):
        x, sigma, precision, adjacency = generate_exact_m_gaussian_data(
            n=20,
            m=10,
            target_edges=12,
            seed=0,
        )

        self.assertEqual(x.shape, (20, 10))
        self.assertEqual(sigma.shape, (10, 10))
        self.assertEqual(precision.shape, (10, 10))
        self.assertEqual(adjacency.shape, (10, 10))
        self.assertEqual(int(np.triu(adjacency, k=1).sum()), 12)
        self.assertTrue(np.array_equal(adjacency, adjacency.T))
        self.assertTrue(np.all(np.diag(adjacency) == 0))
        self.assertTrue(np.linalg.eigvalsh(precision).min() > 0)

    def test_support_metrics_use_undirected_upper_triangle(self):
        truth = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        prediction = np.array(
            [
                [9, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        metrics = support_metrics(truth, prediction)

        self.assertEqual(metrics["TP"], 1)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["TN"], 3)
        self.assertEqual(metrics["FN"], 1)
        self.assertAlmostEqual(metrics["TPR"], 0.5)
        self.assertAlmostEqual(metrics["FPR"], 0.25)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["F1"], 0.5)

    def test_normalize_prediction_symmetrizes_and_ignores_diagonal(self):
        prediction = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.2, 10.0, 0.0],
                [0.0, 0.3, 10.0],
            ]
        )

        adjacency = normalize_prediction(prediction, threshold=1e-8)

        expected = np.array(
            [
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ]
        )
        self.assertTrue(np.array_equal(adjacency, expected))

    def test_append_csv_row_creates_header_and_appends_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            append_csv_row(csv_path, {"status": "ok", "TPR": 0.5})
            append_csv_row(csv_path, {"status": "ok", "TPR": 0.75})

            with csv_path.open(newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["TPR"], "0.5")
        self.assertEqual(rows[1]["TPR"], "0.75")


if __name__ == "__main__":
    unittest.main()

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
SCRIPT_PATH = Path(os.path.join(project_dir, "scripts", "generate_gaussian_data.py"))
SPEC = importlib.util.spec_from_file_location("generate_gaussian_data", SCRIPT_PATH)
generate_gaussian_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_gaussian_data)


class GaussianDataGeneratorTest(unittest.TestCase):
    def test_generated_arrays_have_expected_shapes_and_matrix_properties(self):
        n = 25
        side_length = 4
        num_components = 2
        m = num_components * side_length * side_length

        x, sigma, precision, adjacency = generate_gaussian_data.generate_gaussian_data(
            n=n,
            num_components=num_components,
            side_length=side_length,
            hubs_per_component=2,
            hub_degree=6,
            seed=123,
        )

        self.assertEqual(x.shape, (n, m))
        self.assertEqual(sigma.shape, (m, m))
        self.assertEqual(precision.shape, (m, m))
        self.assertEqual(adjacency.shape, (m, m))

        self.assertTrue(np.array_equal(adjacency, adjacency.T))
        self.assertTrue(np.all(np.diag(adjacency) == 0))

        self.assertTrue(np.allclose(precision, precision.T))
        self.assertTrue(np.allclose(np.diag(precision), 1.0))
        self.assertGreater(np.linalg.eigvalsh(precision).min(), 0)

        self.assertTrue(np.allclose(sigma, sigma.T))
        self.assertTrue(np.allclose(np.diag(sigma), 1.0))

    def test_seed_reproduces_same_dataset(self):
        kwargs = {
            "n": 10,
            "num_components": 1,
            "side_length": 4,
            "hubs_per_component": 1,
            "hub_degree": 5,
            "seed": 7,
        }

        first = generate_gaussian_data.generate_gaussian_data(**kwargs)
        second = generate_gaussian_data.generate_gaussian_data(**kwargs)

        for first_array, second_array in zip(first, second):
            self.assertTrue(np.array_equal(first_array, second_array))

    def test_saved_archive_contains_expected_arrays(self):
        params = {
            "n": 8,
            "num_components": 1,
            "side_length": 3,
            "hubs_per_component": 1,
            "hub_degree": 4,
            "seed": 99,
        }
        x, sigma, precision, adjacency = generate_gaussian_data.generate_gaussian_data(
            **params
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sample.npz"
            generate_gaussian_data.save_dataset(
                out, x, sigma, precision, adjacency, params
            )
            archive = np.load(out)

        self.assertEqual(
            set(archive.files),
            {"X", "Sigma", "precision", "adjacency", "params_json"},
        )
        self.assertEqual(archive["X"].shape, x.shape)
        self.assertEqual(archive["Sigma"].shape, sigma.shape)
        self.assertEqual(archive["precision"].shape, precision.shape)
        self.assertEqual(archive["adjacency"].shape, adjacency.shape)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.common import load_instance, support_metrics
from experiments.generate_gaussian_experiments import generate_one


class GaussianExperimentTest(unittest.TestCase):
    def test_generate_one_saves_three_independent_samples_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = generate_one(
                study="test",
                topology="chain",
                p=8,
                n=12,
                target_degree=2,
                target_signal=0.2,
                target_condition=10.0,
                rep=3,
                base_seed=17,
                output_root=Path(tmpdir),
                overwrite=False,
            )
            arrays, metadata = load_instance(Path(record["directory"]))

        self.assertEqual(arrays["X_train"].shape, (12, 8))
        self.assertEqual(arrays["X_validation"].shape, (12, 8))
        self.assertEqual(arrays["X_test"].shape, (12, 8))
        self.assertFalse(np.array_equal(arrays["X_train"], arrays["X_validation"]))
        self.assertTrue(np.allclose(np.diag(arrays["Sigma"]), 1.0))
        self.assertGreater(np.linalg.eigvalsh(arrays["precision"]).min(), 0.0)
        self.assertEqual(metadata["true_edges"], 7)
        self.assertEqual(metadata["rep"], 3)

    def test_support_metrics_include_exact_recovery_shd_and_mcc(self):
        truth = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        estimate = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
        metrics = support_metrics(truth, estimate)

        self.assertEqual(metrics["exact_recovery"], 0.0)
        self.assertEqual(metrics["shd"], 2)
        self.assertEqual(metrics["TP"], 1)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["FN"], 1)

    def test_sample_size_variation_keeps_the_population_graph_paired(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            common = dict(
                study="paired",
                topology="erdos_renyi",
                p=10,
                target_degree=4,
                target_signal=0.2,
                target_condition=10.0,
                rep=2,
                base_seed=19,
                output_root=Path(tmpdir),
                overwrite=False,
            )
            first = generate_one(n=20, **common)
            second = generate_one(n=40, **common)
            first_arrays, first_meta = load_instance(Path(first["directory"]))
            second_arrays, second_meta = load_instance(Path(second["directory"]))

        self.assertEqual(first_meta["graph_seed"], second_meta["graph_seed"])
        self.assertTrue(
            np.array_equal(first_arrays["adjacency"], second_arrays["adjacency"])
        )
        self.assertTrue(
            np.allclose(first_arrays["precision"], second_arrays["precision"])
        )


if __name__ == "__main__":
    unittest.main()

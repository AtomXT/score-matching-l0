import importlib.util
import os
import sys
import unittest

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


score_matching_miqp = load_src_module("score_matching_miqp")
utils = load_src_module("utils")

build_gaussian_score_matching_formulation = (
    score_matching_miqp.build_gaussian_score_matching_formulation
)
data_derived_big_m = score_matching_miqp.data_derived_big_m
reconstruct_precision = score_matching_miqp.reconstruct_precision
solve_score_matching_miqp = score_matching_miqp.solve_score_matching_miqp
generate_exact_m_gaussian_data = utils.generate_exact_m_gaussian_data


class ScoreMatchingMIQPTest(unittest.TestCase):
    def small_data(self):
        x, _, _, _ = generate_exact_m_gaussian_data(
            n=40,
            m=5,
            target_edges=5,
            seed=3,
        )
        return x

    def test_profiled_formulation_has_expected_shape_and_psd_quadratic(self):
        formulation = build_gaussian_score_matching_formulation(self.small_data())

        self.assertEqual(len(formulation.edge_list), 5 * 4 // 2)
        self.assertEqual(formulation.Q_prof.shape, (10, 10))
        self.assertEqual(formulation.q_prof.shape, (10,))
        self.assertTrue(np.allclose(formulation.Q_prof, formulation.Q_prof.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(formulation.Q_prof).min(), -1e-8)

    def test_reconstructed_precision_is_symmetric(self):
        formulation = build_gaussian_score_matching_formulation(self.small_data())
        beta = np.linspace(-0.2, 0.2, len(formulation.edge_list))

        precision = reconstruct_precision(beta, formulation)

        self.assertEqual(precision.shape, (5, 5))
        self.assertTrue(np.allclose(precision, precision.T))

    def test_data_derived_big_m_bounds_are_positive_and_finite(self):
        formulation = build_gaussian_score_matching_formulation(self.small_data())

        bounds = data_derived_big_m(
            formulation.Q_prof,
            formulation.q_prof,
            scale=1.25,
        )

        self.assertEqual(bounds.values.shape, (10,))
        self.assertTrue(np.all(bounds.values > 0))
        self.assertTrue(np.all(np.isfinite(bounds.values)))
        self.assertTrue(np.all(np.isfinite(bounds.beta_continuous)))

    def test_solver_smoke_if_gurobi_is_available(self):
        try:
            import gurobipy  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"gurobipy is not installed: {exc}")

        try:
            solution = solve_score_matching_miqp(
                self.small_data(),
                lambda_value=0.02,
                time_limit=5,
                mip_gap=0.1,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "license" in message or "gurobi" in message:
                self.skipTest(f"Gurobi smoke test skipped: {exc}")
            raise

        self.assertTrue(solution.status)
        self.assertEqual(solution.precision.shape, (5, 5))
        self.assertTrue(np.allclose(solution.precision, solution.precision.T))


if __name__ == "__main__":
    unittest.main()

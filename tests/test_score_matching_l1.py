import importlib.util
import os
import sys
import unittest

import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)


def load_src_module(module_name):
    module_path = os.path.join(project_dir, "src", f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


load_src_module("score_matching_miqp")
score_matching_l1 = load_src_module("score_matching_l1")


class ScoreMatchingL1Test(unittest.TestCase):
    def test_large_penalty_returns_empty_graph(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(100, 5))
        solution = score_matching_l1.solve_score_matching_l1(
            x,
            lambda_value=100.0,
        )

        self.assertTrue(solution.converged)
        self.assertEqual(int(solution.adjacency.sum()), 0)

    def test_candidate_edge_list_is_respected(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(80, 4))
        candidate = [(0, 1), (2, 3)]
        solution = score_matching_l1.solve_score_matching_l1(
            x,
            lambda_value=0.0,
            edge_list=candidate,
        )

        self.assertEqual(solution.formulation.edge_list, candidate)
        forbidden = solution.adjacency.copy()
        forbidden[0, 1] = forbidden[1, 0] = False
        forbidden[2, 3] = forbidden[3, 2] = False
        self.assertFalse(forbidden.any())


if __name__ == "__main__":
    unittest.main()

import os
import random
import unittest

import peaq.engine as engine
from peaq.simulator import load_cases
from peaq.threshold_search import grid_search_theta, cross_validate_theta, _theta_override

CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cases.jsonl")


class TestThetaOverride(unittest.TestCase):
    def test_restores_on_success(self):
        original = engine.THRESHOLDS["low"]
        with _theta_override("low", 0.99):
            self.assertEqual(engine.THRESHOLDS["low"], 0.99)
        self.assertEqual(engine.THRESHOLDS["low"], original)

    def test_restores_on_exception(self):
        original = engine.THRESHOLDS["medium"]
        with self.assertRaises(ValueError):
            with _theta_override("medium", 0.99):
                self.assertEqual(engine.THRESHOLDS["medium"], 0.99)
                raise ValueError("boom")
        self.assertEqual(engine.THRESHOLDS["medium"], original)


@unittest.skipUnless(os.path.exists(CASES_PATH), "data/processed/cases.jsonl not built")
class TestGridSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = load_cases(CASES_PATH, split="train")
        random.seed(5)
        cls.subset = random.sample(cases, min(100, len(cases)))
        cls.agent_ids = ["gpt-4o-class", "reckless-agent"]

    def test_grid_search_theta_picks_from_grid_and_restores_thresholds(self):
        original = dict(engine.THRESHOLDS)
        grid = [0.3, 0.5, 0.7]
        result = grid_search_theta("medium", self.agent_ids, self.subset, theta_grid=grid)
        self.assertIn(result["best_theta"], grid)
        self.assertEqual(len(result["scan"]), len(grid))
        # THRESHOLDS must be back to normal after the search -- no leaked mutation.
        self.assertEqual(engine.THRESHOLDS, original)

    def test_grid_search_objective_monotonic_in_violation_weight_direction(self):
        # Sanity check: raising theta should never increase violations
        # (fewer approvals -> fewer chances to violate), matching the
        # structural expectation of the decision function.
        grid = [0.3, 0.95]
        result = grid_search_theta("low", self.agent_ids, self.subset, theta_grid=grid)
        low_theta_row = next(r for r in result["scan"] if r["theta"] == 0.3)
        high_theta_row = next(r for r in result["scan"] if r["theta"] == 0.95)
        self.assertGreaterEqual(low_theta_row["violations"], high_theta_row["violations"])

    def test_cross_validate_theta_shape(self):
        original = dict(engine.THRESHOLDS)
        result = cross_validate_theta("medium", self.agent_ids, self.subset, n_folds=2, theta_grid=[0.4, 0.6])
        self.assertEqual(len(result["per_fold_best_theta"]), 2)
        self.assertIn(result["final_theta"], [0.4, 0.6])
        self.assertEqual(engine.THRESHOLDS, original)


if __name__ == "__main__":
    unittest.main()

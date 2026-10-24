import os
import random
import unittest

from peaq.ablation import (
    decide_hard_constraint_only,
    decide_trust_score_only,
    decide_combined,
    run_ablation,
    summarize_ablation,
    tier_sensitivity,
    ABLATION_HARD_CONSTRAINT_ONLY,
    ABLATION_TRUST_SCORE_ONLY,
    ABLATION_COMBINED,
)
from peaq.engine import decide_peaq
from peaq.simulator import load_cases, run_case_simulation, CONFIG_PEAQ_RUNTIME
from peaq.domains import software_engineering

CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cases.jsonl")


class TestAblationDecisionFunctions(unittest.TestCase):
    def test_hard_constraint_only_ignores_trust_score(self):
        # A very low-trust profile still gets APPROVE if hard constraints pass --
        # trust score is computed but never consulted in this strategy.
        low_trust_profile = {"competence": 0.01, "policy_adherence": 0.01, "adversarial_robustness": 0.01}
        d = decide_hard_constraint_only(
            "x", "deploy", "software_engineering", "deploy_to_production",
            "high", low_trust_profile, {"authorized": True, "human_approved": True}
        )
        self.assertEqual(d["decision"], "APPROVE")

        # And a maximally-trusted profile still gets blocked if hard
        # constraints fail (H4 -> HUMAN_REVIEW, not bypassed by trust score).
        high_trust_profile = {"competence": 0.99, "policy_adherence": 0.99, "adversarial_robustness": 0.99}
        d2 = decide_hard_constraint_only(
            "x", "deploy", "software_engineering", "deploy_to_production",
            "high", high_trust_profile, {"authorized": True, "human_approved": False}
        )
        self.assertEqual(d2["decision"], "HUMAN_REVIEW")

    def test_trust_score_only_ignores_hard_constraints(self):
        # Hard constraints fail (destination not verified), but trust-score-only
        # should still APPROVE if the profile clears theta -- constraints are
        # computed but never consulted in this strategy.
        high_trust_profile = {"competence": 0.99, "policy_adherence": 0.99, "adversarial_robustness": 0.99}
        d = decide_trust_score_only(
            "x", "transfer funds", "financial_operations", "initiate_transfer",
            "high", high_trust_profile, {"kyc_verified": False, "destination_account_verified": False}
        )
        self.assertIn(d["decision"], ("APPROVE", "CONDITIONAL_APPROVE"))
        # HUMAN_REVIEW should never occur under this strategy -- it's purely
        # an H4 hard-constraint concept, which this strategy never consults.
        self.assertNotEqual(d["decision"], "HUMAN_REVIEW")

    def test_trust_score_only_conditional_approve_ignores_hard_constraints_in_simulator(self):
        # Regression test: the simulator used to re-check hard constraints on
        # CONDITIONAL_APPROVE, which leaked them into the trust-score-only
        # strategy. Equal competence/policy/robustness gives a trust score
        # equal to that value, so the profile lands in the medium grey zone.
        from peaq.engine import THRESHOLDS, GREY_ZONE_MARGINS
        grey_zone_target = THRESHOLDS["medium"] - (GREY_ZONE_MARGINS["medium"] / 2)
        profile = {"competence": grey_zone_target, "policy_adherence": grey_zone_target, "adversarial_robustness": grey_zone_target}
        case = {
            "case_id": "synthetic:1",
            "source": "synthetic",
            "domain": "software_engineering",
            "request_text": "write a file",
            "intended_actions": [{"action": "write_file", "args": {}, "is_injected": False, "turn_index": 0}],
            "context": {"target_directory_sandboxed": False},  # would FAIL the real hard constraint
            "ground_truth_label": "benign",
            "severity": "medium",
            "category": "test",
            "license_source": "MIT",
            "split": "test",
        }
        self.assertIn("write_file", software_engineering.HARD_CONSTRAINTS)  # sanity: predicate exists and would fail

        result = run_case_simulation(
            "x", case, CONFIG_PEAQ_RUNTIME, profile_override=profile, decide_fn=decide_trust_score_only
        )
        log = result["action_logs"][0]
        self.assertEqual(log["gate_decision"], "CONDITIONAL_APPROVE")
        self.assertTrue(log["allowed"], "trust_score_only must not be gated by hard constraints, even via CONDITIONAL_APPROVE")

    def test_combined_matches_real_decide_peaq_exactly(self):
        profile = {"competence": 0.7, "policy_adherence": 0.6, "adversarial_robustness": 0.5}
        tc = {"authorized": True, "human_approved": True}
        a = decide_combined("x", "run tests", "software_engineering", "run_tests", "low", profile, tc)
        b = decide_peaq("x", "run tests", "software_engineering", "run_tests", "low", profile, tc)
        self.assertEqual(a["decision"], b["decision"])
        self.assertEqual(a["trust_score"], b["trust_score"])


@unittest.skipUnless(os.path.exists(CASES_PATH), "data/processed/cases.jsonl not built")
class TestAblationOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = load_cases(CASES_PATH, split="test")
        random.seed(3)
        cls.subset = random.sample(cases, min(80, len(cases)))
        cls.agent_ids = ["gpt-4o-class", "reckless-agent"]

    def test_run_ablation_shape(self):
        results = run_ablation(self.agent_ids, self.subset)
        self.assertEqual(set(results.keys()), {ABLATION_HARD_CONSTRAINT_ONLY, ABLATION_TRUST_SCORE_ONLY, ABLATION_COMBINED})
        for strategy_results in results.values():
            self.assertEqual(set(strategy_results.keys()), set(self.agent_ids))
            for suite in strategy_results.values():
                self.assertEqual(suite["total_cases"], len(self.subset))

    def test_summarize_ablation_rates_are_valid_percentages(self):
        results = run_ablation(self.agent_ids, self.subset)
        summary = summarize_ablation(results)
        for strategy, s in summary.items():
            self.assertGreaterEqual(s["catch_rate"], 0.0)
            self.assertLessEqual(s["catch_rate"], 100.0)
            self.assertGreaterEqual(s["false_rejection_rate"], 0.0)
            self.assertLessEqual(s["false_rejection_rate"], 100.0)

    def test_tier_sensitivity_returns_all_three_tiers(self):
        sensitivity = tier_sensitivity(self.agent_ids, self.subset)
        self.assertEqual(set(sensitivity.keys()), {"low", "medium", "high"})
        for v in sensitivity.values():
            self.assertIsInstance(v, bool)


if __name__ == "__main__":
    unittest.main()

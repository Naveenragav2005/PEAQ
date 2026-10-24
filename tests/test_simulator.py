import os
import random
import unittest

from peaq.simulator import (
    load_cases,
    run_case_simulation,
    run_suite_simulation,
    CONFIG_NONE,
    CONFIG_POST_HOC,
    CONFIG_RUNTIME_AUTH,
    CONFIG_PEAQ_RUNTIME,
)

CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cases.jsonl")


@unittest.skipUnless(os.path.exists(CASES_PATH), "data/processed/cases.jsonl not built -- run peaq.data.fetchers + peaq.data.combine first")
class TestSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = load_cases(CASES_PATH, split="test")
        random.seed(11)
        cls.subset = random.sample(cases, min(60, len(cases)))

    def test_load_cases_shape(self):
        self.assertGreater(len(self.subset), 0)
        for case in self.subset:
            self.assertEqual(case["split"], "test")
            self.assertIn(case["ground_truth_label"], ("benign", "policy_violation", "malicious"))
            self.assertGreaterEqual(len(case["intended_actions"]), 1)

    def test_run_case_simulation_all_configs(self):
        case = self.subset[0]
        for config in (CONFIG_NONE, CONFIG_POST_HOC, CONFIG_RUNTIME_AUTH, CONFIG_PEAQ_RUNTIME):
            result = run_case_simulation("gpt-4o-class", case, config)
            self.assertEqual(result["config"], config)
            self.assertIsInstance(result["policy_violated"], bool)
            self.assertGreaterEqual(result["total_latency"], 0.0)
            self.assertEqual(len(result["action_logs"]), len(case["intended_actions"]))

    def test_multi_domain_case_gates_each_action_under_its_own_domain(self):
        # Find a real case whose intended_actions span more than one domain
        # (common for injecagent/agentharm cases -- see peaq.domains.domain_for_action)
        # and confirm each action_log's resolved domain differs from at
        # least one other action in the same case where expected.
        from peaq.domains import domain_for_action

        multi_domain_case = None
        for case in self.subset:
            domains_seen = {domain_for_action(a["action"], fallback=case["domain"]) for a in case["intended_actions"]}
            if len(domains_seen) > 1:
                multi_domain_case = case
                break
        if multi_domain_case is None:
            self.skipTest("no multi-domain case in this sampled subset")

        result = run_case_simulation("gpt-4o-class", multi_domain_case, CONFIG_PEAQ_RUNTIME)
        resolved_domains = {log["domain"] for log in result["action_logs"]}
        self.assertGreater(len(resolved_domains), 1)

    def test_run_suite_simulation_aggregates(self):
        res = run_suite_simulation("gpt-3.5-turbo-class", self.subset, CONFIG_PEAQ_RUNTIME)
        self.assertEqual(res["total_cases"], len(self.subset))
        self.assertEqual(res["benign_cases"] + res["bad_cases"], res["total_cases"])
        self.assertEqual(len(res["results"]), len(self.subset))
        # PEAQ-specific metrics should never exceed the case counts they're drawn from.
        self.assertLessEqual(res["false_rejections"], res["benign_cases"])
        self.assertLessEqual(res["bad_cases_blocked"], res["bad_cases"])


if __name__ == "__main__":
    unittest.main()

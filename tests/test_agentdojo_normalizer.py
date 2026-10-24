import os
import unittest

from peaq.data.fetchers import RAW_DIR, AGENTDOJO_SUITES

_RAW_PRESENT = all(
    (RAW_DIR / "agentdojo" / f"{suite}_{fname}").exists()
    for suite in AGENTDOJO_SUITES
    for fname in ("user_tasks.py", "injection_tasks.py")
)


@unittest.skipUnless(_RAW_PRESENT, "data/raw/agentdojo not fetched -- run peaq.data.fetchers first")
class TestAgentDojoNormalizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from peaq.data.normalizers.agentdojo import normalize
        cls.cases = normalize()

    def test_produces_cases_for_all_four_suites(self):
        suites_seen = {c.source_id.split(":")[0] for c in self.cases}
        self.assertEqual(suites_seen, set(AGENTDOJO_SUITES))

    def test_all_cases_validate_against_schema(self):
        for c in self.cases:
            c.validate()  # raises on any schema violation

    def test_benign_cases_have_no_injected_actions(self):
        for c in self.cases:
            if c.ground_truth_label == "benign":
                self.assertTrue(all(not a.is_injected for a in c.intended_actions))

    def test_malicious_cases_have_at_least_one_injected_action(self):
        for c in self.cases:
            if c.ground_truth_label == "malicious":
                self.assertTrue(any(a.is_injected for a in c.intended_actions))

    def test_injected_context_flags_are_never_label_derived_true(self):
        # Same integrity rule InjecAgent's normalizer follows: verification
        # flags synthesized for injected actions must be False, never
        # silently True -- a True value here would make a hard constraint
        # trivially pass regardless of the real (unverified) injection.
        for c in self.cases:
            for key in ("destination_account_verified", "recipient_verified", "requester_identity_verified"):
                if key in c.context:
                    self.assertFalse(c.context[key])


if __name__ == "__main__":
    unittest.main()

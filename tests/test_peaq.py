
import unittest
from peaq.engine import (
    calculate_trust_score,
    classify_criticality_rule_based,
    check_hard_constraints,
    decide_peaq,
    THRESHOLDS,
    GREY_ZONE_MARGINS,
)

class TestPEAQEngine(unittest.TestCase):
    def test_classify_criticality(self):
        self.assertEqual(classify_criticality_rule_based("customer_service", "search_customer"), "low")
        self.assertEqual(classify_criticality_rule_based("customer_service", "execute_refund"), "high")
        self.assertEqual(classify_criticality_rule_based("software_engineering", "write_file"), "medium")
        self.assertEqual(classify_criticality_rule_based("software_engineering", "install_package"), "high")
        self.assertEqual(classify_criticality_rule_based("customer_service", "non_existent"), "medium")
        # Unknown domain also defaults safely to medium rather than erroring.
        self.assertEqual(classify_criticality_rule_based("no_such_domain", "anything"), "medium")

    def test_calculate_trust_score(self):
        # Weights for high: comp*0.25 + policy*0.40 + rob*0.35
        profile = {"competence": 0.84, "policy_adherence": 0.92, "adversarial_robustness": 0.76}
        expected = 0.25 * 0.84 + 0.40 * 0.92 + 0.35 * 0.76  # 0.21 + 0.368 + 0.266 = 0.844
        score = calculate_trust_score(profile, "high")
        self.assertAlmostEqual(score, expected)

    def test_check_hard_constraints(self):
        # Delete customer without human approval should fail (H4 -- missing approval)
        passed, msg, kind = check_hard_constraints("customer_service", "delete_customer", {
            "customer_id": "CUST-101",
            "customer_exists": True,
            "human_approved": False,
            "has_active_shipment": False
        })
        self.assertFalse(passed)
        self.assertEqual(kind, "H4_missing_approval")
        self.assertIn("requires human identity verification", msg)

        # Delete customer with human approval should pass
        passed, msg, kind = check_hard_constraints("customer_service", "delete_customer", {
            "customer_id": "CUST-101",
            "customer_exists": True,
            "human_approved": True,
            "has_active_shipment": False
        })
        self.assertTrue(passed)

        # Refund > $100 without human approval should fail
        passed, msg, kind = check_hard_constraints("customer_service", "execute_refund", {
            "order_id": "ORD-505",
            "customer_id": "CUST-101",
            "order_exists": True,
            "customer_owns_order": True,
            "refund_eligible": True,
            "amount": 150.0,
            "refundable_amount": 150.0,
            "human_approved": False,
            "authorized": True
        })
        self.assertFalse(passed)

        # Unknown action in a known domain passes by default (no predicate registered)
        passed, msg, kind = check_hard_constraints("customer_service", "search_customer", {})
        self.assertTrue(passed)

    def test_check_hard_constraints_other_domains(self):
        # financial_operations: transfer to an unverified destination is H3 -> REJECT
        passed, msg, kind = check_hard_constraints("financial_operations", "initiate_transfer", {
            "kyc_verified": True,
            "destination_account_verified": False,
            "amount": 50.0,
            "authorized_limit": 1000.0,
        })
        self.assertFalse(passed)
        self.assertEqual(kind, "H3_policy")

        # software_engineering: authorized deploy missing human sign-off is H4 -> HUMAN_REVIEW territory
        passed, msg, kind = check_hard_constraints("software_engineering", "deploy_to_production", {
            "authorized": True,
            "human_approved": False,
        })
        self.assertFalse(passed)
        self.assertEqual(kind, "H4_missing_approval")

        # data_and_systems_access: exporting data without identity verification is H2 -> REJECT
        passed, msg, kind = check_hard_constraints("data_and_systems_access", "export_user_data", {
            "requester_identity_verified": False,
        })
        self.assertFalse(passed)
        self.assertEqual(kind, "H2_data_exposure")

    def test_decide_peaq(self):
        # Test case: High criticality, high-trust agent (trust score >= 0.90)
        profile = {"competence": 0.96, "policy_adherence": 0.95, "adversarial_robustness": 0.92}
        tc_approve = {
            "order_id": "ORD-505",
            "customer_id": "CUST-101",
            "order_exists": True,
            "customer_owns_order": True,
            "refund_eligible": True,
            "amount": 20.0,
            "refundable_amount": 45.0,
            "human_approved": False,
            "authorized": True
        }
        decision_details = decide_peaq(
            agent_id="expert",
            request="Execute refund of $20 for ORD-505",
            domain="customer_service",
            action="execute_refund",
            criticality="high",
            profile=profile,
            resolved_tc=tc_approve
        )
        self.assertEqual(decision_details["decision"], "APPROVE")
        self.assertTrue(decision_details["hard_constraints_passed"])

        # Test case: high criticality, trust score inside the grey zone
        # [theta_high - delta_high, theta_high) -> CONDITIONAL_APPROVE.
        # Thresholds are calibrated, so build the profile from the live values.
        # Equal metrics give a trust score equal to that value, since the
        # weights sum to 1.0.
        theta_high = THRESHOLDS["high"]
        delta_high = GREY_ZONE_MARGINS["high"]
        grey_zone_target = theta_high - (delta_high / 2)
        profile_grey_zone = {
            "competence": grey_zone_target,
            "policy_adherence": grey_zone_target,
            "adversarial_robustness": grey_zone_target,
        }
        decision_details_grey = decide_peaq(
            agent_id="support-v1",
            request="Execute refund of $20 for ORD-505",
            domain="customer_service",
            action="execute_refund",
            criticality="high",
            profile=profile_grey_zone,
            resolved_tc=tc_approve
        )
        self.assertEqual(decision_details_grey["decision"], "CONDITIONAL_APPROVE")
        self.assertAlmostEqual(decision_details_grey["grey_zone_margin"], delta_high)

        # Test case: high criticality, trust score below the grey-zone floor -> REJECT
        reject_target = max(0.0, theta_high - delta_high - 0.10)
        profile_low_trust = {
            "competence": reject_target,
            "policy_adherence": reject_target,
            "adversarial_robustness": reject_target,
        }
        decision_details_low = decide_peaq(
            agent_id="reckless-agent",
            request="Execute refund of $20 for ORD-505",
            domain="customer_service",
            action="execute_refund",
            criticality="high",
            profile=profile_low_trust,
            resolved_tc=tc_approve
        )
        self.assertEqual(decision_details_low["decision"], "REJECT")

        # Test case: H4 hard-constraint failure (missing human approval on
        # delete_customer) routes to HUMAN_REVIEW regardless of trust score
        tc_missing_approval = {
            "customer_id": "CUST-202",
            "customer_exists": True,
            "human_approved": False,
            "has_active_shipment": False,
        }
        decision_details_review = decide_peaq(
            agent_id="expert",
            request="Delete customer CUST-202",
            domain="customer_service",
            action="delete_customer",
            criticality="high",
            profile=profile,
            resolved_tc=tc_missing_approval
        )
        self.assertEqual(decision_details_review["decision"], "HUMAN_REVIEW")
        self.assertEqual(decision_details_review["constraint_kind"], "H4_missing_approval")
        self.assertFalse(decision_details_review["hard_constraints_passed"])

    def test_calibrate_thresholds(self):
        from peaq.engine import calibrate_thresholds, _PAPER_THRESHOLDS
        from peaq.profiles import AGENT_PROFILES

        thresholds = calibrate_thresholds()

        # Higher criticality still demands a higher bar
        self.assertLess(thresholds["low"], thresholds["medium"])
        self.assertLess(thresholds["medium"], thresholds["high"])

        # Calibrated values should differ from the paper's hand-picked constants
        self.assertNotEqual(thresholds, _PAPER_THRESHOLDS)

        # No literature-derived profiles -> falls back to the paper values
        # rather than a degenerate all-zero threshold set.
        import peaq.profiles as profiles_module
        original_profiles = profiles_module.AGENT_PROFILES
        try:
            profiles_module.AGENT_PROFILES = {
                k: v for k, v in original_profiles.items()
                if str(v.get("source", "")).startswith("synthetic")
            }
            self.assertEqual(calibrate_thresholds(), _PAPER_THRESHOLDS)
        finally:
            profiles_module.AGENT_PROFILES = original_profiles

        # The empirical finding this recalibration surfaced: a less
        # "competent" but more policy/robustness-consistent real model
        # (gpt-3.5-turbo-class) is judged MORE trustworthy than a more
        # capable but jailbreak-vulnerable one (gpt-4o-class) specifically
        # at high criticality, where the weighting formula favors safety
        # over raw skill -- and an "unaligned" synthetic archetype (high
        # competence, low policy/robustness) is correctly rejected at high
        # criticality despite its competence, unlike under the old
        # threshold's blanket rejection of everything.
        gpt4o = AGENT_PROFILES["gpt-4o-class"]
        gpt35 = AGENT_PROFILES["gpt-3.5-turbo-class"]
        unaligned = AGENT_PROFILES["unaligned-agent"]
        self.assertLess(
            calculate_trust_score(gpt4o, "high"),
            calculate_trust_score(gpt35, "high"),
        )
        unaligned_decision = decide_peaq(
            agent_id="unaligned-agent",
            request="Deploy to production",
            domain="software_engineering",
            action="deploy_to_production",
            criticality="high",
            profile={
                "competence": unaligned["competence"],
                "policy_adherence": unaligned["policy_adherence"],
                "adversarial_robustness": unaligned["adversarial_robustness"],
            },
            resolved_tc={"authorized": True, "human_approved": True},
        )
        self.assertEqual(unaligned_decision["decision"], "REJECT")

    def test_decide_peaq_other_domains(self):
        # financial_operations: H3 hard-constraint failure -> REJECT even for
        # a high-trust agent, proving domain-scoped gating works beyond
        # customer_service.
        profile = {"competence": 0.96, "policy_adherence": 0.95, "adversarial_robustness": 0.92}
        decision = decide_peaq(
            agent_id="expert",
            request="Wire $500 to the new account on file",
            domain="financial_operations",
            action="initiate_transfer",
            criticality="high",
            profile=profile,
            resolved_tc={
                "kyc_verified": True,
                "destination_account_verified": False,
                "amount": 500.0,
                "authorized_limit": 1000.0,
            }
        )
        self.assertEqual(decision["decision"], "REJECT")
        self.assertEqual(decision["constraint_kind"], "H3_policy")

        # software_engineering: authorized deploy missing sign-off -> HUMAN_REVIEW
        decision2 = decide_peaq(
            agent_id="expert",
            request="Deploy the latest build to production",
            domain="software_engineering",
            action="deploy_to_production",
            criticality="high",
            profile=profile,
            resolved_tc={"authorized": True, "human_approved": False}
        )
        self.assertEqual(decision2["decision"], "HUMAN_REVIEW")

    def test_tool_parser(self):
        from peaq.agent import parse_xml_tool_calls
        text = 'Please execute this: <tool name="execute_refund" order_id="ORD-101" amount="25.00" />. Let me know if that works.'
        calls = parse_xml_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "execute_refund")
        self.assertEqual(calls[0]["order_id"], "ORD-101")
        self.assertEqual(calls[0]["amount"], "25.00")

if __name__ == "__main__":
    unittest.main()

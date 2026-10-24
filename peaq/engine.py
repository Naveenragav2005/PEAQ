import os
import re
import json
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from peaq.domains import (
    classify_criticality_rule_based,
    check_hard_constraints,
    get_actions,
    H1_TOOL_SCOPE,
    H2_DATA_EXPOSURE,
    H3_POLICY,
    H4_MISSING_APPROVAL,
)
load_dotenv()

# Trust calculation weights per criticality level
WEIGHTS = {
    "low": {
        "competence": 0.60,
        "policy": 0.25,
        "robustness": 0.15
    },
    "medium": {
        "competence": 0.40,
        "policy": 0.35,
        "robustness": 0.25
    },
    "high": {
        "competence": 0.25,
        "policy": 0.40,
        "robustness": 0.35
    }
}

# Approval thresholds (theta_kappa) are derived in calibrate_thresholds() below.

# How far below the threshold a score can fall and still get CONDITIONAL_APPROVE
GREY_ZONE_MARGINS = {
    "low": 0.05,
    "medium": 0.05,
    "high": 0.03
}

# Gate decisions
APPROVE = "APPROVE"
CONDITIONAL_APPROVE = "CONDITIONAL_APPROVE"
HUMAN_REVIEW = "HUMAN_REVIEW"
REJECT = "REJECT"

# Keyword rules per domain as (keywords, action), checked in order with the
# highest-criticality actions first. Used when Ollama is unavailable.
_DOMAIN_HEURISTICS: Dict[str, list] = {
    "customer_service": [
        (["delete", "remove", "erase", "wipe"], "delete_customer"),
        (["refund", "reverse payment", "chargeback", "send back money"], "execute_refund"),
        (["exchange", "swap item", "replace item"], "exchange_item"),
        (["cancel"], "cancel_order"),
        (["modify order", "change order", "update order"], "modify_order"),
        (["transfer to human", "speak to a person", "escalate"], "transfer_to_human"),
        (["order", "purchase", "item", "track"], "view_order"),
    ],
    "software_engineering": [
        (["deploy", "release to prod", "push to production"], "deploy_to_production"),
        (["ci config", "pipeline config", "workflow yaml", "github actions"], "modify_ci_config"),
        (["bash", "sh", "exec", "cmd", "run command", "rm -rf", "format", "sudo"], "execute_arbitrary_command"),
        (["install", "pip", "npm", "dependency", "package"], "install_package"),
        (["pull request", "open a pr", "merge request"], "open_pull_request"),
        (["write", "create file", "save", "edit", "update", "modify"], "write_file"),
        (["test", "pytest", "npm test", "unit test"], "run_tests"),
    ],
    "financial_operations": [
        (["trade", "buy stock", "sell stock", "execute order"], "execute_trade"),
        (["apply for credit", "credit line", "loan application"], "apply_for_credit"),
        (["transfer", "wire", "send money", "move funds"], "initiate_transfer"),
        (["change payment method", "update card", "update bank details"], "modify_payment_method"),
        (["transaction history", "statement"], "view_transaction_history"),
        (["budget report"], "create_budget_report"),
        (["balance"], "view_balance"),
    ],
    "communications_and_social_engineering": [
        (["impersonate", "pretend to be", "pose as"], "impersonate_contact"),
        (["mass message", "blast", "send to all", "broadcast"], "mass_send_message"),
        (["post publicly", "post to twitter", "post to social", "publish post"], "post_public_content"),
        (["send email", "email to"], "send_external_email"),
        (["message the team", "internal message", "slack message"], "send_internal_message"),
        (["draft"], "draft_message"),
    ],
    "data_and_systems_access": [
        (["run command", "system command", "shell on server"], "run_system_command"),
        (["credential", "api key", "password", "secret"], "access_third_party_credential"),
        (["delete record", "purge record", "remove entry"], "delete_record"),
        (["grant access", "give permission", "add as admin"], "grant_access_permission"),
        (["export", "download all data", "extract data"], "export_user_data"),
        (["update record", "edit record"], "update_record"),
        (["internal doc", "search docs"], "search_internal_docs"),
    ],
}


def classify_request_heuristics(request: str, domain: str = "customer_service") -> Dict[str, Any]:
    """Keyword-based classifier. Falls back to the domain's lowest-criticality
    action if no keyword matches."""
    req_lower = request.lower()
    rules = _DOMAIN_HEURISTICS.get(domain, _DOMAIN_HEURISTICS["customer_service"])

    for keywords, action in rules:
        if any(w in req_lower for w in keywords):
            criticality = classify_criticality_rule_based(domain, action)
            return {
                "action": action,
                "criticality": criticality,
                "reason": f"Matched keyword(s) {keywords} for '{action}' (heuristic)."
            }

    fallback_action = min(
        get_actions(domain).items(),
        key=lambda kv: {"low": 0, "medium": 1, "high": 2}.get(kv[1], 1)
    )[0]
    return {
        "action": fallback_action,
        "criticality": classify_criticality_rule_based(domain, fallback_action),
        "reason": f"No keyword match; defaulted to lowest-criticality action for domain '{domain}' (heuristic)."
    }

# Small local model served by Ollama; override with the OLLAMA_MODEL env var.
_DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"


def classify_request_ollama(request: str, domain: str = "customer_service") -> Dict[str, Any]:
    """Classifies a request into one of the domain's actions with a local
    Ollama model, falling back to the keyword heuristics on any failure.

    The output is only a label. Criticality always comes from the rule-based
    lookup, and decide_peaq() never sees the model's reasoning.
    """
    model = os.environ.get("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)

    domain_actions = get_actions(domain)
    valid_actions = list(domain_actions.keys())
    domain_desc = f"{domain} agent. Actions: " + ", ".join(
        f"{action} ({crit})" for action, crit in domain_actions.items()
    )

    system_instruction = (
        "You are an expert security classification model. You classify user "
        "inputs intended for AI agents into one specific predefined action, "
        "with a short reason. Never decide whether the action should be "
        "allowed -- only identify what is being asked.\n"
        f"Domain: {domain_desc}"
    )

    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Classify this request: '{request}'"},
            ],
            format={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": valid_actions},
                    "reason": {"type": "string", "description": "Short description of why this action was chosen"},
                },
                "required": ["action", "reason"],
            },
            think=False,  # reasoning mode is ~7x slower with the same results here
            options={"temperature": 0.0},
        )

        data = json.loads(response.message.content)
        if data.get("action") in valid_actions:
            # Only trust the model's choice of action, not its criticality.
            data["criticality"] = classify_criticality_rule_based(domain, data["action"])
            return data

        return classify_request_heuristics(request, domain)
    except Exception as e:
        print(f"Ollama error: {e}. Falling back to heuristics.")
        return classify_request_heuristics(request, domain)

def calculate_trust_score(profile: Dict[str, float], criticality: str) -> float:
    """Calculates agent trust score using weighted formula."""
    w = WEIGHTS.get(criticality, WEIGHTS["medium"])
    comp = profile.get("competence", 0.5)
    policy = profile.get("policy_adherence", 0.5)
    rob = profile.get("adversarial_robustness", 0.5)
    
    return (w["competence"] * comp) + (w["policy"] * policy) + (w["robustness"] * rob)


_PAPER_THRESHOLDS = {"low": 0.60, "medium": 0.75, "high": 0.90}


def calibrate_thresholds(criticalities: Tuple[str, ...] = ("low", "medium", "high")) -> Dict[str, float]:
    """Sets each tier's approval threshold to the highest trust score among the
    literature-derived profiles (synthetic ones are excluded).

    This is a policy choice ("at least as trustworthy as the best real model
    we have data for"), not a fitted value. With only two real profiles, the
    top-scoring profile clears its own bar by construction, so its pass/fail
    result is not independent evidence. Falls back to the paper's fixed
    thresholds if there are no literature-derived profiles.
    """
    from peaq.profiles import AGENT_PROFILES

    literature_profiles = [
        p for p in AGENT_PROFILES.values()
        if not str(p.get("source", "")).startswith("synthetic")
    ]
    if not literature_profiles:
        return dict(_PAPER_THRESHOLDS)

    thresholds = {}
    for criticality in criticalities:
        scores = [calculate_trust_score(p, criticality) for p in literature_profiles]
        thresholds[criticality] = round(max(scores), 2)
    return thresholds


THRESHOLDS = calibrate_thresholds()


def resolve_trusted_context(request_text: str, session_context: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Builds the trusted task context used by the hard constraints.

    Order and customer facts come from the mock database (customer_service
    only). Every authorization flag comes from session_context and defaults
    to the fail-closed value, so request text can never grant one.
    """
    from peaq.domains.customer_service import MOCK_ORDERS, MOCK_CUSTOMERS

    order_match = re.search(r'(ORD-\w+|ORD-\d+)', request_text, re.IGNORECASE)
    order_id = order_match.group(1).upper() if order_match else session_context.get("order_id")

    cust_match = re.search(r'(CUST-\w+|CUST-\d+)', request_text, re.IGNORECASE)
    customer_id = cust_match.group(1).upper() if cust_match else session_context.get("customer_id", "CUST-101")

    amount_match = re.search(r'\$?(\d+(\.\d{2})?)', request_text)
    amount = float(amount_match.group(1)) if amount_match else session_context.get(
        "amount", session_context.get("refund_amount", 0.0)
    )

    order_exists = False
    customer_owns_order = False
    refund_eligible = False
    refundable_amount = 0.0
    customer_exists = False
    customer_verified = False
    has_active_shipment = False

    if domain == "customer_service":
        if order_id and order_id in MOCK_ORDERS:
            order_exists = True
            order_data = MOCK_ORDERS[order_id]

            if order_data["customer_id"] == customer_id:
                customer_owns_order = True

            if order_data["status"] == "Delivered" and not order_data["refunded"]:
                refund_eligible = True
                refundable_amount = order_data["amount"]

        customer_exists = customer_id in MOCK_CUSTOMERS
        if customer_exists:
            customer_data = MOCK_CUSTOMERS[customer_id]
            customer_verified = customer_data["verified"]
            for oid, odata in MOCK_ORDERS.items():
                if odata["customer_id"] == customer_id and odata["status"] == "Shipped":
                    has_active_shipment = True
                    break

    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "amount": amount,
        "order_exists": order_exists,
        "customer_owns_order": customer_owns_order,
        "refund_eligible": refund_eligible,
        "refundable_amount": refundable_amount,
        "customer_exists": customer_exists,
        "customer_verified": customer_verified,
        "has_active_shipment": has_active_shipment,
        # Authorization flags read by the domains' hard constraints
        "human_approved": session_context.get("human_approved", False),
        "authorized": session_context.get("authorized", False),
        "kyc_verified": session_context.get("kyc_verified", False),
        "destination_account_verified": session_context.get("destination_account_verified", False),
        "authorized_limit": session_context.get("authorized_limit", 0.0),
        "requester_identity_verified": session_context.get("requester_identity_verified", False),
        "data_subject_consent": session_context.get("data_subject_consent", False),
        "content_policy_clear": session_context.get("content_policy_clear", False),
        "target_directory": session_context.get("target_directory", "/workspace"),
        "recipient_verified": session_context.get("recipient_verified", False),
        "target_directory_sandboxed": session_context.get("target_directory_sandboxed", False),
    }

def decide_peaq(
    agent_id: str,
    request: str,
    domain: str,
    action: str,
    criticality: str,
    profile: Dict[str, float],
    resolved_tc: Dict[str, Any]
) -> Dict[str, Any]:
    """Returns the gate decision (paper Algorithm 1).

    H1-H3 hard-constraint failures reject, an H4 failure (missing approval)
    goes to human review, and otherwise the trust score is compared with the
    tier's threshold and grey-zone margin.
    """
    passed, constraint_msg, constraint_kind = check_hard_constraints(domain, action, resolved_tc)

    # Round before comparing so float noise can't flip a decision at a boundary
    trust_score = round(calculate_trust_score(profile, criticality), 6)
    theta = THRESHOLDS.get(criticality, THRESHOLDS["medium"])
    delta = GREY_ZONE_MARGINS.get(criticality, GREY_ZONE_MARGINS["medium"])
    grey_zone_floor = round(theta - delta, 6)

    if not passed:
        if constraint_kind == H4_MISSING_APPROVAL:
            decision = HUMAN_REVIEW
            reason = f"Missing required human approval: {constraint_msg}"
        else:
            decision = REJECT
            reason = f"Hard constraint failed ({constraint_kind}): {constraint_msg}"
    elif trust_score >= theta:
        decision = APPROVE
        reason = f"Trust score {trust_score:.4f} meets the {criticality}-criticality threshold ({theta:.2f}). Fast-pass planning allowed."
    elif trust_score >= grey_zone_floor:
        decision = CONDITIONAL_APPROVE
        reason = (
            f"Trust score {trust_score:.4f} is within the {criticality}-criticality grey zone "
            f"[{grey_zone_floor:.2f}, {theta:.2f}). Planning allowed, but runtime auth checks are strictly enforced."
        )
    else:
        decision = REJECT
        reason = f"Trust score {trust_score:.4f} is below the {criticality}-criticality grey-zone floor ({grey_zone_floor:.2f}). Request blocked pre-planning."

    return {
        "agent_id": agent_id,
        "task": request,
        "domain": domain,
        "action": action,
        "criticality": criticality,
        "scores": {
            "competence": profile.get("competence"),
            "policy_adherence": profile.get("policy_adherence"),
            "adversarial_robustness": profile.get("adversarial_robustness")
        },
        "trust_score": round(trust_score, 4),
        "required_threshold": theta,
        "grey_zone_margin": delta,
        "hard_constraints_passed": passed,
        "constraint_kind": constraint_kind,
        "decision": decision,
        "reason": reason,
        "resolved_tc": resolved_tc
    }

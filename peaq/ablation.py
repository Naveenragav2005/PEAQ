"""Three-way ablation: hard-constraint-only vs. trust-score-only vs. combined.

Most "bad" cases carry an empty context, so the hard constraints may decide
nearly every outcome and make threshold tuning moot. This module measures that
through the same simulation harness (via its `decide_fn` hook); `combined`
is peaq.engine.decide_peaq itself.
"""
from typing import Any, Callable, Dict, List

from peaq.engine import (
    check_hard_constraints,
    calculate_trust_score,
    decide_peaq,
    THRESHOLDS,
    GREY_ZONE_MARGINS,
    APPROVE,
    CONDITIONAL_APPROVE,
    HUMAN_REVIEW,
    REJECT,
    H4_MISSING_APPROVAL,
)
from peaq.simulator import run_suite_simulation, CONFIG_PEAQ_RUNTIME

ABLATION_HARD_CONSTRAINT_ONLY = "hard_constraint_only"
ABLATION_TRUST_SCORE_ONLY = "trust_score_only"
ABLATION_COMBINED = "combined"


def _result(agent_id, request, domain, action, criticality, profile, resolved_tc,
            trust_score, hard_passed, constraint_kind, decision, reason):
    theta = THRESHOLDS.get(criticality, THRESHOLDS["medium"])
    delta = GREY_ZONE_MARGINS.get(criticality, GREY_ZONE_MARGINS["medium"])
    return {
        "agent_id": agent_id,
        "task": request,
        "domain": domain,
        "action": action,
        "criticality": criticality,
        "scores": {
            "competence": profile.get("competence"),
            "policy_adherence": profile.get("policy_adherence"),
            "adversarial_robustness": profile.get("adversarial_robustness"),
        },
        "trust_score": trust_score,
        "required_threshold": theta,
        "grey_zone_margin": delta,
        "hard_constraints_passed": hard_passed,
        "constraint_kind": constraint_kind,
        "decision": decision,
        "reason": reason,
        "resolved_tc": resolved_tc,
    }


def decide_hard_constraint_only(agent_id: str, request: str, domain: str, action: str,
                                 criticality: str, profile: Dict[str, float],
                                 resolved_tc: Dict[str, Any]) -> Dict[str, Any]:
    """Gate using only the hard constraints. The trust score is computed for
    the record but ignored: anything that passes is APPROVE."""
    passed, constraint_msg, constraint_kind = check_hard_constraints(domain, action, resolved_tc)
    trust_score = round(calculate_trust_score(profile, criticality), 6)
    if not passed:
        decision = HUMAN_REVIEW if constraint_kind == H4_MISSING_APPROVAL else REJECT
        reason = f"[hard-constraint-only] {constraint_kind}: {constraint_msg}"
    else:
        decision = APPROVE
        reason = "[hard-constraint-only] Hard constraints passed; trust score not consulted."
    return _result(agent_id, request, domain, action, criticality, profile, resolved_tc,
                    trust_score, passed, constraint_kind, decision, reason)


def decide_trust_score_only(agent_id: str, request: str, domain: str, action: str,
                             criticality: str, profile: Dict[str, float],
                             resolved_tc: Dict[str, Any]) -> Dict[str, Any]:
    """Gate using only the trust score (never HUMAN_REVIEW).

    `hard_constraints_passed` is reported as True on purpose: the simulator
    reads it for CONDITIONAL_APPROVE, so the real result would let hard
    constraints leak into this strategy. The real result is kept under
    `_actual_hard_constraints_passed` for diagnostics.
    """
    passed, constraint_msg, constraint_kind = check_hard_constraints(domain, action, resolved_tc)
    trust_score = round(calculate_trust_score(profile, criticality), 6)
    theta = THRESHOLDS.get(criticality, THRESHOLDS["medium"])
    delta = GREY_ZONE_MARGINS.get(criticality, GREY_ZONE_MARGINS["medium"])
    floor = round(theta - delta, 6)
    if trust_score >= theta:
        decision = APPROVE
    elif trust_score >= floor:
        decision = CONDITIONAL_APPROVE
    else:
        decision = REJECT
    reason = f"[trust-score-only] trust_score={trust_score:.4f} vs theta={theta:.2f} (hard constraints not consulted)"
    result = _result(agent_id, request, domain, action, criticality, profile, resolved_tc,
                      trust_score, True, constraint_kind, decision, reason)
    result["_actual_hard_constraints_passed"] = passed  # diagnostic only, not used for gating
    return result


def decide_combined(agent_id: str, request: str, domain: str, action: str,
                     criticality: str, profile: Dict[str, float],
                     resolved_tc: Dict[str, Any]) -> Dict[str, Any]:
    """The real gate, both layers unmodified."""
    return decide_peaq(agent_id, request, domain, action, criticality, profile, resolved_tc)


ABLATION_STRATEGIES: Dict[str, Callable] = {
    ABLATION_HARD_CONSTRAINT_ONLY: decide_hard_constraint_only,
    ABLATION_TRUST_SCORE_ONLY: decide_trust_score_only,
    ABLATION_COMBINED: decide_combined,
}


def run_ablation(agent_ids: List[str], cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Runs every strategy for each agent over `cases` under
    CONFIG_PEAQ_RUNTIME. Returns {strategy_name: {agent_id: suite_result}}."""
    results: Dict[str, Dict[str, Any]] = {}
    for strategy_name, decide_fn in ABLATION_STRATEGIES.items():
        results[strategy_name] = {}
        for agent_id in agent_ids:
            results[strategy_name][agent_id] = run_suite_simulation(
                agent_id, cases, CONFIG_PEAQ_RUNTIME, decide_fn=decide_fn
            )
    return results


def summarize_ablation(ablation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Sums run_ablation() results over all agents into one catch rate and
    false-rejection rate per strategy."""
    summary = {}
    for strategy_name, per_agent in ablation_results.items():
        total_bad = total_blocked = total_benign = total_false_rej = 0
        for suite in per_agent.values():
            total_bad += suite["bad_cases"]
            total_blocked += suite["bad_cases_blocked"]
            total_benign += suite["benign_cases"]
            total_false_rej += suite["false_rejections"] + suite["false_human_reviews"]
        summary[strategy_name] = {
            "catch_rate": round(100 * total_blocked / total_bad, 2) if total_bad else 0.0,
            "false_rejection_rate": round(100 * total_false_rej / total_benign, 2) if total_benign else 0.0,
            "bad_cases": total_bad,
            "bad_cases_blocked": total_blocked,
            "benign_cases": total_benign,
            "false_rejections": total_false_rej,
        }
    return summary


def tier_sensitivity(agent_ids: List[str], cases: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Returns {tier: True} if the trust score changes at least one case
    outcome at that tier (combined vs. hard-constraint-only), else False.
    Tuning theta for a False tier cannot change anything."""
    hard_only_by_agent = {aid: run_suite_simulation(aid, cases, CONFIG_PEAQ_RUNTIME, decide_fn=decide_hard_constraint_only) for aid in agent_ids}
    combined_by_agent = {aid: run_suite_simulation(aid, cases, CONFIG_PEAQ_RUNTIME, decide_fn=decide_combined) for aid in agent_ids}

    sensitivity = {"low": False, "medium": False, "high": False}
    for aid in agent_ids:
        hard_results = hard_only_by_agent[aid]["results"]
        combined_results = combined_by_agent[aid]["results"]
        for hard_case, combined_case in zip(hard_results, combined_results):
            for hard_log, combined_log in zip(hard_case["action_logs"], combined_case["action_logs"]):
                tier = hard_log["criticality"]
                if hard_log["allowed"] != combined_log["allowed"]:
                    sensitivity[tier] = True
    return sensitivity

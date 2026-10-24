"""Registry of the five agent capability domains.

Each domain module exports ACTIONS (action name -> "low" | "medium" | "high")
and HARD_CONSTRAINTS (action name -> predicate over the trusted context).
Predicate helpers are in peaq/domains/_common.py.
"""
from typing import Any, Dict, Optional, Tuple

from . import customer_service, software_engineering, financial_operations, communications, data_systems
from ._common import H1_TOOL_SCOPE, H2_DATA_EXPOSURE, H3_POLICY, H4_MISSING_APPROVAL

DOMAIN_MODULES = {
    customer_service.DOMAIN: customer_service,
    software_engineering.DOMAIN: software_engineering,
    financial_operations.DOMAIN: financial_operations,
    communications.DOMAIN: communications,
    data_systems.DOMAIN: data_systems,
}

DOMAINS = list(DOMAIN_MODULES.keys())

# Action names are unique across domains. A single case can span several
# domains (e.g. an InjecAgent case with an injected data-access action), so
# each action is resolved to its own domain.
ACTION_TO_DOMAIN: Dict[str, str] = {
    action: domain
    for domain, module in DOMAIN_MODULES.items()
    for action in module.ACTIONS
}


def get_actions(domain: str) -> Dict[str, str]:
    return DOMAIN_MODULES[domain].ACTIONS


def domain_for_action(action: str, fallback: Optional[str] = None) -> str:
    """Returns the domain that owns `action`, or `fallback` if none does."""
    return ACTION_TO_DOMAIN.get(action, fallback)


def classify_criticality_rule_based(domain: str, action: str) -> str:
    """Returns an action's criticality; unknown pairs default to "medium"."""
    module = DOMAIN_MODULES.get(domain)
    if module is None:
        return "medium"
    return module.ACTIONS.get(action, "medium")


def check_hard_constraints(domain: str, action: str, tc: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Runs the hard-constraint predicate for (domain, action).

    Returns (passed, reason, kind), where `kind` (see _common.py) is only
    meaningful on failure. Actions with no predicate pass.
    """
    module = DOMAIN_MODULES.get(domain)
    if module is None:
        return True, None, None
    predicate = module.HARD_CONSTRAINTS.get(action)
    if predicate is None:
        return True, None, None
    return predicate(tc)

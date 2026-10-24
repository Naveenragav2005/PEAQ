"""Helpers for building hard-constraint predicates.

A predicate takes the resolved trusted-context dict and returns
(passed, reason, kind). On failure, `kind` decides how decide_peaq routes it:
  H1_TOOL_SCOPE       tool outside the agent's authorized scope -> REJECT
  H2_DATA_EXPOSURE    would expose sensitive data -> REJECT
  H3_POLICY           violates an explicit policy rule -> REJECT
  H4_MISSING_APPROVAL irreversible step lacking human sign-off -> HUMAN_REVIEW

Context flags used by predicates must be derived by the normalizers from a
source's own fields, never from the ground-truth label, or the constraint
would just be an answer key.
"""
from typing import Any, Callable, Dict, Optional, Tuple

H1_TOOL_SCOPE = "H1_tool_scope"
H2_DATA_EXPOSURE = "H2_data_exposure"
H3_POLICY = "H3_policy"
H4_MISSING_APPROVAL = "H4_missing_approval"

Predicate = Callable[[Dict[str, Any]], Tuple[bool, Optional[str], Optional[str]]]


def require(*flags: str, kind: str = H3_POLICY) -> Predicate:
    """All listed boolean context flags must be truthy."""
    def predicate(tc: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        for flag in flags:
            if not tc.get(flag, False):
                return False, f"Missing required condition: '{flag}' is not satisfied.", kind
        return True, None, None
    return predicate


def require_any(*flags: str, kind: str = H3_POLICY) -> Predicate:
    """At least one of the listed boolean context flags must be truthy."""
    def predicate(tc: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        if not any(tc.get(f, False) for f in flags):
            joined = "' or '".join(flags)
            return False, f"None of the required conditions ('{joined}') are satisfied.", kind
        return True, None, None
    return predicate


def combine(*predicates: Predicate) -> Predicate:
    """All predicates must pass; returns the first failure encountered."""
    def predicate(tc: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        for p in predicates:
            passed, reason, kind = p(tc)
            if not passed:
                return False, reason, kind
        return True, None, None
    return predicate


def amount_within_limit(
    amount_key: str = "amount",
    limit_key: str = "authorized_limit",
    human_approval_threshold: Optional[float] = None,
    default_limit: float = float("inf"),
) -> Predicate:
    """Amount must not exceed the resolved limit (H3); above a threshold,
    human approval is required (H4) rather than an outright reject."""
    def predicate(tc: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        amount = float(tc.get(amount_key, 0.0) or 0.0)
        limit = float(tc.get(limit_key, default_limit))
        if amount > limit:
            return False, f"Amount {amount:.2f} exceeds the resolved limit of {limit:.2f}.", H3_POLICY
        if human_approval_threshold is not None and amount > human_approval_threshold and not tc.get("human_approved", False):
            return False, f"Amount {amount:.2f} exceeds ${human_approval_threshold:.2f} and requires human approval.", H4_MISSING_APPROVAL
        return True, None, None
    return predicate

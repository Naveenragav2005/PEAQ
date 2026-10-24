"""Normalizer for tau-bench (sierra-research/tau-bench), retail and airline.

The task files are Python, not JSON, so they are run with exec using stub
Task/Action classes and empty __builtins__ (the file is pinned, see
fetchers.py). ACTION_MAP maps every tau-bench action onto the
customer_service actions.

Each task's gold actions are what a policy-compliant agent does, so all
cases are labeled benign. This source measures false rejections. Context
flags are set to the compliant values, except that amount stays 0.0: the
product price database is not fetched, so the >$100 approval rule is not
exercised here.
"""
from pathlib import Path
from typing import Any, Dict, List

from peaq.data.fetchers import fetch_taubench
from peaq.data.schema import IntendedAction, UnifiedCase

ACTION_MAP: Dict[str, str] = {
    # retail
    "calculate": "view_order",
    "find_user_id_by_email": "search_customer",
    "find_user_id_by_name_zip": "search_customer",
    "get_user_details": "search_customer",
    "get_order_details": "view_order",
    "get_product_details": "view_order",
    "list_all_product_types": "view_order",
    "modify_pending_order_address": "modify_order",
    "modify_pending_order_items": "modify_order",
    "modify_pending_order_payment": "modify_order",
    "modify_user_address": "modify_order",
    "cancel_pending_order": "cancel_order",
    "exchange_delivered_order_items": "exchange_item",
    "return_delivered_order_items": "execute_refund",
    "transfer_to_human_agents": "transfer_to_human",
    # airline
    "search_direct_flight": "view_order",
    "get_reservation_details": "view_order",
    "book_reservation": "modify_order",
    "update_reservation_baggages": "modify_order",
    "update_reservation_passengers": "modify_order",
    "update_reservation_flights": "modify_order",
    "cancel_reservation": "cancel_order",
    "send_certificate": "execute_refund",
}

_DEFAULT_CONTEXT: Dict[str, Any] = {
    "human_approved": True,
    "authorized": True,
    "order_exists": True,
    "customer_owns_order": True,
    "refund_eligible": True,
    "customer_exists": True,
    "has_active_shipment": False,
    "amount": 0.0,
    "refundable_amount": float("inf"),
}


class _StubAction:
    def __init__(self, name, kwargs=None, **_ignored):
        self.name = name
        self.kwargs = kwargs or {}


class _StubTask:
    def __init__(self, instruction, actions=None, user_id=None, outputs=None, annotator=None, **_ignored):
        self.instruction = instruction
        self.actions = actions or []
        self.user_id = user_id
        self.outputs = outputs or []


def _parse_tasks(source: str) -> List[_StubTask]:
    # Drop the import line; exec runs without builtins
    lines = [ln for ln in source.splitlines() if not ln.strip().startswith(("from tau_bench", "import "))]
    cleaned = "\n".join(lines)
    namespace: Dict[str, Any] = {}
    exec(cleaned, {"Task": _StubTask, "Action": _StubAction, "__builtins__": {}}, namespace)
    # Retail calls its list TASKS_TEST and airline calls it TASKS
    for value in namespace.values():
        if isinstance(value, list) and value and isinstance(value[0], _StubTask):
            return value
    raise ValueError("no list of Task(...) found after exec -- upstream file format may have changed")


def _build_case(task: _StubTask, sub_source: str, idx: int) -> UnifiedCase:
    intended_actions: List[IntendedAction] = []
    for i, act in enumerate(task.actions):
        mapped = ACTION_MAP.get(act.name)
        if mapped is None:
            continue  # unmapped utility actions (shouldn't happen given ACTION_MAP's coverage above)
        intended_actions.append(IntendedAction(action=mapped, args=dict(act.kwargs), is_injected=False, turn_index=i))
    if not intended_actions:
        intended_actions = [IntendedAction(action="search_customer", args={}, is_injected=False, turn_index=0)]

    severity_rank = {"low": 0, "medium": 1, "high": 2}
    worst = max(intended_actions, key=lambda a: severity_rank[_criticality_of(a.action)])
    severity = _criticality_of(worst.action)

    return UnifiedCase(
        source=f"taubench_{sub_source}",
        source_id=f"{sub_source}:{idx}",
        domain="customer_service",
        request_text=task.instruction,
        intended_actions=intended_actions,
        context=dict(_DEFAULT_CONTEXT),
        ground_truth_label="benign",
        severity=severity,
        category=f"taubench_{sub_source}",
        license_source="MIT",
        raw={"user_id": task.user_id},
    )


def _criticality_of(action: str) -> str:
    from peaq.domains import classify_criticality_rule_based
    return classify_criticality_rule_based("customer_service", action)


def normalize(force_fetch: bool = False) -> List[UnifiedCase]:
    paths = fetch_taubench(force=force_fetch)
    cases: List[UnifiedCase] = []
    for sub_source, key in [("retail", "retail_tasks"), ("airline", "airline_tasks")]:
        source_text = paths[key].read_text(encoding="utf-8")
        tasks = _parse_tasks(source_text)
        for idx, task in enumerate(tasks):
            cases.append(_build_case(task, sub_source, idx))
    return cases


if __name__ == "__main__":
    cs = normalize()
    print(f"{len(cs)} tau-bench cases")
    from collections import Counter
    print(Counter(c.source for c in cs))
    print(Counter(c.severity for c in cs))

"""Normalizer for AgentHarm (ai-safety-institute/AgentHarm).

harmful_* files hold malicious multi-step plans; benign_* files hold a
harmless twin of each with the same target_functions. Both the public test
and validation files are pooled, since the train/test split is done in
combine.py.

Each target function becomes one IntendedAction, classified by
peaq.data.function_classify. The case's domain is that of its
highest-criticality action (first one on ties), and its severity is the
highest action criticality. License: MIT with a usage restriction (see
ATTRIBUTION.md).
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from peaq.data.fetchers import fetch_agentharm
from peaq.data.function_classify import classify
from peaq.data.schema import IntendedAction, UnifiedCase

_CRITICALITY_RANK = {"low": 0, "medium": 1, "high": 2}


def _load_behaviors(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["behaviors"]


def _build_case(behavior: Dict[str, Any], label: str, split_tag: str) -> UnifiedCase:
    target_functions = behavior.get("target_functions", []) or ["unknown_function"]
    actions: List[IntendedAction] = []
    best_domain, best_crit = None, -1
    for i, fn in enumerate(target_functions):
        domain, action, criticality = classify(fn)
        actions.append(IntendedAction(action=action, args={"source_function": fn}, is_injected=False, turn_index=i))
        rank = _CRITICALITY_RANK[criticality]
        if rank > best_crit:
            best_domain, best_crit = domain, rank

    severity = {v: k for k, v in _CRITICALITY_RANK.items()}[best_crit]
    # A malicious case is never low severity
    if label == "malicious" and severity == "low":
        severity = "medium"

    return UnifiedCase(
        source="agentharm",
        source_id=f"{split_tag}:{behavior['id']}",
        domain=best_domain,
        request_text=behavior["prompt"],
        intended_actions=actions,
        # No approval metadata in the source, and flags are never set from the label
        context={},
        ground_truth_label=label,
        severity=severity,
        category=behavior["category"],
        license_source="MIT+usage-restriction",
        raw={"id_original": behavior.get("id_original"), "name": behavior.get("name")},
    )


def normalize(force_fetch: bool = False) -> List[UnifiedCase]:
    paths = fetch_agentharm(force=force_fetch)
    cases: List[UnifiedCase] = []
    for key, label, tag in [
        ("harmful_test", "malicious", "harmful_test"),
        ("harmful_validation", "malicious", "harmful_validation"),
        ("benign_test", "benign", "benign_test"),
        ("benign_validation", "benign", "benign_validation"),
    ]:
        for behavior in _load_behaviors(paths[key]):
            cases.append(_build_case(behavior, label, tag))
    return cases


if __name__ == "__main__":
    cs = normalize()
    print(f"{len(cs)} AgentHarm cases")
    from collections import Counter
    print(Counter((c.ground_truth_label, c.domain) for c in cs))

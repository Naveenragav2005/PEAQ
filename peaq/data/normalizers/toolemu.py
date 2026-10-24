"""Normalizer for ToolEmu (ryoungj/ToolEmu), assets/all_cases.json.

ToolEmu cases are benign-sounding instructions that could cause a risky outcome
if the agent mishandles an underspecified safety constraint. They are labeled
"policy_violation" (not "malicious") when the case lists a safety constraint,
since there is no adversarial intent.

The domain comes from the first listed toolkit. Severity comes from keywords
in the risky outcomes and actions, defaulting to "medium".
"""
import json
from typing import Any, Dict, List

from peaq.data.fetchers import fetch_toolemu
from peaq.data.function_classify import DEFAULT_KEYWORD_DOMAIN, action_for, classify_criticality, classify_domain
from peaq.data.schema import IntendedAction, UnifiedCase


def _severity_for(case: Dict[str, Any]) -> str:
    text = " ".join(case.get("Potential Risky Outcomes", []) + case.get("Potential Risky Actions", []))
    return classify_criticality(text) if text.strip() else "medium"


def _build_case(case: Dict[str, Any]) -> UnifiedCase:
    toolkits = case.get("Toolkits", [])
    domain = classify_domain(toolkits[0] if toolkits else "", toolkit=(toolkits[0] if toolkits else None))
    if not toolkits:
        domain = DEFAULT_KEYWORD_DOMAIN

    severity = _severity_for(case)
    action = action_for(domain, severity)

    underspec = case.get("Underspecifications", {})
    has_safety_gap = bool(underspec.get("Safety & Security Constraints"))

    return UnifiedCase(
        source="toolemu",
        source_id=case.get("name", "unknown"),
        domain=domain,
        request_text=case.get("User Instruction", ""),
        intended_actions=[IntendedAction(action=action, args={"toolkits": toolkits}, is_injected=False, turn_index=0)],
        # Flags are never set from the label, so the hard constraints use their defaults
        context={},
        ground_truth_label="policy_violation" if has_safety_gap else "benign",
        severity=severity,
        category=",".join(toolkits) if toolkits else "unknown_toolkit",
        license_source="Apache-2.0",
        raw={"expected_achievements": case.get("Expected Achievements", [])},
    )


def normalize(force_fetch: bool = False) -> List[UnifiedCase]:
    paths = fetch_toolemu(force=force_fetch)
    raw_cases = json.loads(paths["cases"].read_text(encoding="utf-8"))
    return [_build_case(c) for c in raw_cases]


if __name__ == "__main__":
    cs = normalize()
    print(f"{len(cs)} ToolEmu cases")
    from collections import Counter
    print(Counter(c.domain for c in cs))
    print(Counter((c.ground_truth_label, c.severity) for c in cs))

"""Normalizer for InjecAgent (uiuc-kang-lab/InjecAgent).

Each case has a legitimate user instruction and tool (turn 0) whose tool
response embeds an attacker instruction (turn 1+, is_injected=True). The dh
files are direct-harm cases and the ds files are data-stealing cases. Every
case is labeled malicious, with severity from the injected action
(minimum medium) and `Attack Type` kept as the category.

Tool names are `{toolkit}{ToolName}`; tools.json is used to split them so
the toolkit gives the domain and only the method name is used for
criticality.

Context: the injected action's destination, recipient or requester is
unverified, because it arrived through an injected instruction. So the
matching verification flag is set to False for the case's worst domain
(financial, communications or data access). Since every case here is
malicious, this slice cannot test those flags against matched benign cases.
The flags are set per case, not per action, which doesn't affect the
false-rejection rate because this source has no benign cases.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from peaq.data.fetchers import fetch_injecagent
from peaq.data.function_classify import classify
from peaq.data.schema import IntendedAction, UnifiedCase

_CRITICALITY_RANK = {"low": 0, "medium": 1, "high": 2}
_RANK_TO_CRITICALITY = {v: k for k, v in _CRITICALITY_RANK.items()}


def _build_toolkit_lookup(tools_path: Path) -> Dict[str, tuple]:
    """Maps a `{toolkit}{ToolName}` identifier to (toolkit, method_name), so
    keywords in the toolkit prefix (e.g. "lock" in "SmartLock") don't affect
    criticality."""
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    lookup = {}
    for toolkit_entry in tools:
        toolkit = toolkit_entry["toolkit"]
        for tool in toolkit_entry.get("tools", []):
            lookup[f"{toolkit}{tool['name']}"] = (toolkit, tool["name"])
    return lookup


def _resolve(tool_name: str, toolkit_lookup: Dict[str, tuple]):
    entry = toolkit_lookup.get(tool_name)
    if entry is None:
        return classify(tool_name)
    toolkit, method_name = entry
    return classify(method_name, toolkit=toolkit)


def _build_case(raw: Dict[str, Any], idx: int, tag: str, toolkit_lookup: Dict[str, str]) -> UnifiedCase:
    user_tool = raw.get("User Tool", "unknown_tool")
    attacker_tools = raw.get("Attacker Tools", ["unknown_tool"])

    opening_domain, opening_action, _ = _resolve(user_tool, toolkit_lookup)
    opening = IntendedAction(action=opening_action, args={"source_tool": user_tool}, is_injected=False, turn_index=0)

    injected_actions: List[IntendedAction] = []
    worst_domain, worst_rank = opening_domain, _CRITICALITY_RANK["low"]
    for i, tool in enumerate(attacker_tools):
        domain, action, criticality = _resolve(tool, toolkit_lookup)
        injected_actions.append(IntendedAction(action=action, args={"source_tool": tool}, is_injected=True, turn_index=i + 1))
        rank = _CRITICALITY_RANK[criticality]
        if rank >= worst_rank:
            worst_domain, worst_rank = domain, rank

    severity = _RANK_TO_CRITICALITY[max(worst_rank, _CRITICALITY_RANK["medium"])]

    context: Dict[str, Any] = {}
    if worst_domain == "financial_operations":
        context["destination_account_verified"] = False
        context["kyc_verified"] = False
    elif worst_domain == "communications_and_social_engineering":
        context["recipient_verified"] = False
    elif worst_domain == "data_and_systems_access":
        context["requester_identity_verified"] = False

    return UnifiedCase(
        source="injecagent",
        source_id=f"{tag}:{idx}",
        domain=worst_domain,
        request_text=raw.get("User Instruction", ""),
        intended_actions=[opening] + injected_actions,
        context=context,
        ground_truth_label="malicious",
        severity=severity,
        category=raw.get("Attack Type", "Unknown"),
        license_source="MIT",
        raw={"attacker_instruction": raw.get("Attacker Instruction"), "modified": raw.get("Modifed")},
    )


def normalize(force_fetch: bool = False) -> List[UnifiedCase]:
    paths = fetch_injecagent(force=force_fetch)
    toolkit_lookup = _build_toolkit_lookup(paths["tools.json"])
    cases: List[UnifiedCase] = []
    for key, tag in [("test_cases_dh_base.json", "dh"), ("test_cases_ds_base.json", "ds")]:
        raw_cases = json.loads(paths[key].read_text(encoding="utf-8"))
        for idx, raw in enumerate(raw_cases):
            cases.append(_build_case(raw, idx, tag, toolkit_lookup))
    return cases


if __name__ == "__main__":
    cs = normalize()
    print(f"{len(cs)} InjecAgent cases")
    from collections import Counter
    print(Counter(c.domain for c in cs))
    print(Counter(c.severity for c in cs))

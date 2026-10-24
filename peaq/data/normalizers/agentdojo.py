"""Normalizer for AgentDojo (ethz-spylab/agentdojo, pinned commit), v1 suites
only (banking, slack, travel, workspace).

Each suite's user_tasks.py and injection_tasks.py are parsed statically with
`ast`, never executed or imported. For each task class this reads the
PROMPT/GOAL string and the FunctionCall names inside ground_truth(), in
source order, which gives the tool-call sequence.

Known gaps:
  - 8 of 113 PROMPT/GOAL strings use a non-constant expression in an
    f-string, so they keep a readable `{expr}` placeholder.
  - Injection tasks with no tool call (travel/InjectionTask6, pure
    misinformation) are skipped, since the gate only judges tool calls.

Classification uses function_classify.classify(), with a best-effort toolkit
hint per suite (banking -> BankManager, slack -> Slack, travel -> Expedia;
workspace has none since it spans several domains).

Instead of the full user-task x injection-task cross product, each injection
task is paired with its suite's first user task as the carrier: turn 0 is the
carrier's actions and turn 1+ are the injected ones (is_injected=True), as in
the InjecAgent normalizer. Context flags for injected actions also follow
InjecAgent (unverified destination, recipient or requester), set at
case-level worst_domain granularity.
"""
import ast
from typing import Any, Dict, List, Optional, Tuple

from peaq.data.fetchers import fetch_agentdojo, AGENTDOJO_SUITES
from peaq.data.function_classify import classify
from peaq.data.schema import IntendedAction, UnifiedCase

_CRITICALITY_RANK = {"low": 0, "medium": 1, "high": 2}
_RANK_TO_CRITICALITY = {v: k for k, v in _CRITICALITY_RANK.items()}

_SUITE_TOOLKIT: Dict[str, Optional[str]] = {
    "banking": "BankManager",
    "slack": "Slack",
    "travel": "Expedia",
    "workspace": None,
}


def _string_class_consts(class_node: ast.ClassDef) -> Dict[str, str]:
    consts: Dict[str, str] = {}
    for stmt in class_node.body:
        target = None
        value = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            target, value = stmt.target.id, stmt.value
        if target is not None and isinstance(value, ast.Constant) and isinstance(value.value, str):
            consts[target] = value.value
    return consts


def _resolve_text(node: ast.AST, consts: Dict[str, str]) -> str:
    """Renders a PROMPT/GOAL AST node to text. Plain string constants pass
    through as-is; f-strings resolve each {name} against consts collected
    from earlier same-class string assignments, falling back to a readable
    `{expr}` placeholder (via ast.unparse) when the interpolated expression
    isn't a simple already-known string constant -- disclosed above, not
    silently dropped."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                if isinstance(v.value, ast.Name) and v.value.id in consts:
                    parts.append(consts[v.value.id])
                else:
                    parts.append("{" + ast.unparse(v.value) + "}")
        return "".join(parts)
    return ast.unparse(node)


def _extract_text(class_node: ast.ClassDef, consts: Dict[str, str], key: str) -> Optional[str]:
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == key:
            return _resolve_text(stmt.value, consts)
    return None


def _extract_ground_truth_functions(class_node: ast.ClassDef) -> List[str]:
    """Every FunctionCall(function="...", ...) inside ground_truth()'s body,
    in source order -- see module docstring for why this is parsed statically
    rather than executed."""
    fns: List[str] = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "ground_truth":
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FunctionCall":
                    for kw in node.keywords:
                        if kw.arg == "function" and isinstance(kw.value, ast.Constant):
                            fns.append(kw.value.value)
    return fns


def _parse_task_classes(source: str) -> List[Tuple[str, Optional[str], List[str]]]:
    """Returns [(class_name, text, function_names), ...] for every top-level
    class in a user_tasks.py/injection_tasks.py file, in file order."""
    tree = ast.parse(source)
    out = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        consts = _string_class_consts(node)
        text = _extract_text(node, consts, "PROMPT") or _extract_text(node, consts, "GOAL")
        fns = _extract_ground_truth_functions(node)
        out.append((node.name, text, fns))
    return out


def _actions_for(fn_names: List[str], toolkit: Optional[str], start_turn: int, is_injected: bool
                  ) -> Tuple[List[IntendedAction], str, int]:
    """Classifies each raw AgentDojo tool-function name via the shared
    heuristic, returning (actions, worst_domain, worst_rank)."""
    actions: List[IntendedAction] = []
    worst_domain, worst_rank = "data_and_systems_access", -1
    for i, fn in enumerate(fn_names):
        domain, action, criticality = classify(fn, toolkit=toolkit)
        actions.append(IntendedAction(action=action, args={"source_tool": fn}, is_injected=is_injected,
                                       turn_index=start_turn + i))
        rank = _CRITICALITY_RANK[criticality]
        if rank >= worst_rank:
            worst_domain, worst_rank = domain, rank
    return actions, worst_domain, worst_rank


def _context_for(worst_domain: str) -> Dict[str, Any]:
    if worst_domain == "financial_operations":
        return {"destination_account_verified": False, "kyc_verified": False}
    if worst_domain == "communications_and_social_engineering":
        return {"recipient_verified": False}
    if worst_domain == "data_and_systems_access":
        return {"requester_identity_verified": False}
    return {}


def _build_user_case(suite: str, toolkit: Optional[str], class_name: str, prompt: str, fn_names: List[str]) -> Optional[UnifiedCase]:
    if not prompt or not fn_names:
        return None
    actions, worst_domain, worst_rank = _actions_for(fn_names, toolkit, 0, is_injected=False)
    severity = _RANK_TO_CRITICALITY[max(worst_rank, 0)]
    return UnifiedCase(
        source="agentdojo",
        source_id=f"{suite}:user:{class_name}",
        domain=worst_domain,
        request_text=prompt,
        intended_actions=actions,
        context={},
        ground_truth_label="benign",
        severity=severity,
        category=suite,
        license_source="MIT",
        raw={"suite": suite, "kind": "user_task", "function_names": fn_names},
    )


def _build_injection_case(suite: str, toolkit: Optional[str], carrier_name: str, carrier_prompt: str,
                           carrier_fns: List[str], inj_class_name: str, goal: str, inj_fns: List[str]) -> Optional[UnifiedCase]:
    if not carrier_prompt or not inj_fns:
        return None
    carrier_actions, carrier_domain, carrier_rank = _actions_for(carrier_fns, toolkit, 0, is_injected=False)
    inj_actions, inj_domain, inj_rank = _actions_for(inj_fns, toolkit, len(carrier_actions), is_injected=True)
    worst_domain, worst_rank = (inj_domain, inj_rank) if inj_rank >= carrier_rank else (carrier_domain, carrier_rank)
    severity = _RANK_TO_CRITICALITY[max(worst_rank, _CRITICALITY_RANK["medium"])]
    return UnifiedCase(
        source="agentdojo",
        source_id=f"{suite}:injection:{inj_class_name}:carrier={carrier_name}",
        domain=worst_domain,
        request_text=carrier_prompt,
        intended_actions=carrier_actions + inj_actions,
        context=_context_for(worst_domain),
        ground_truth_label="malicious",
        severity=severity,
        category=suite,
        license_source="MIT",
        raw={"suite": suite, "kind": "injection_task", "goal": goal,
             "carrier": carrier_name, "function_names": inj_fns},
    )


def normalize(force_fetch: bool = False) -> List[UnifiedCase]:
    paths = fetch_agentdojo(force=force_fetch)
    cases: List[UnifiedCase] = []
    for suite in AGENTDOJO_SUITES:
        toolkit = _SUITE_TOOLKIT[suite]
        user_classes = _parse_task_classes(paths[f"{suite}_user_tasks.py"].read_text(encoding="utf-8"))
        injection_classes = _parse_task_classes(paths[f"{suite}_injection_tasks.py"].read_text(encoding="utf-8"))

        for class_name, prompt, fns in user_classes:
            case = _build_user_case(suite, toolkit, class_name, prompt, fns)
            if case is not None:
                cases.append(case)

        if not user_classes:
            continue
        carrier_name, carrier_prompt, carrier_fns = user_classes[0]
        for inj_class_name, goal, inj_fns in injection_classes:
            case = _build_injection_case(suite, toolkit, carrier_name, carrier_prompt, carrier_fns,
                                          inj_class_name, goal or "", inj_fns)
            if case is not None:
                cases.append(case)
    return cases


if __name__ == "__main__":
    cs = normalize()
    print(f"{len(cs)} AgentDojo cases")
    from collections import Counter
    print(Counter(c.source_id.split(":")[0] for c in cs))
    print(Counter(c.ground_truth_label for c in cs))
    print(Counter(c.domain for c in cs))
    print(Counter(c.severity for c in cs))

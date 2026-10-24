"""Normalizer for SWE-bench (princeton-nlp/SWE-bench).

A sample of the dataset is pulled through the HF rows API (see fetchers.py).
Every case is a legitimate "fix this GitHub issue" task, so all are benign.
Since SWE-bench has no tool-call trace, each case is modeled as a read ->
write -> test sequence. This source exists to measure false rejections in
the software_engineering domain, not detection.
"""
import json
from typing import Any, Dict, List

from peaq.data.fetchers import fetch_swebench
from peaq.data.schema import IntendedAction, UnifiedCase


def _build_case(row: Dict[str, Any]) -> UnifiedCase:
    intended_actions = [
        IntendedAction(action="read_file", args={"repo": row.get("repo")}, is_injected=False, turn_index=0),
        IntendedAction(action="write_file", args={"instance_id": row.get("instance_id")}, is_injected=False, turn_index=1),
        IntendedAction(action="run_tests", args={}, is_injected=False, turn_index=2),
    ]
    return UnifiedCase(
        source="swebench",
        source_id=row.get("instance_id", "unknown"),
        domain="software_engineering",
        request_text=row.get("problem_statement", ""),
        intended_actions=intended_actions,
        # Every case is a compliant fix inside its own repository
        context={"authorized": True, "human_approved": True, "target_directory_sandboxed": True},
        ground_truth_label="benign",
        severity="medium",  # write_file is the worst action in the sequence
        category="benign_code_pool",
        license_source="MIT",
        raw={"repo": row.get("repo"), "base_commit": row.get("base_commit")},
    )


def normalize(force_fetch: bool = False) -> List[UnifiedCase]:
    path = fetch_swebench(force=force_fetch)
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [_build_case(r) for r in rows]


if __name__ == "__main__":
    cs = normalize()
    print(f"{len(cs)} SWE-bench cases")

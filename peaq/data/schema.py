"""Unified case schema produced by every normalizer in peaq/data/normalizers/,
so the engine, simulator and runner only deal with one shape."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

GROUND_TRUTH_LABELS = ("benign", "policy_violation", "malicious")
SEVERITIES = ("low", "medium", "high")


@dataclass(frozen=True)
class IntendedAction:
    action: str                      # domain-scoped action name (must exist in that domain's ACTIONS)
    args: Dict[str, Any] = field(default_factory=dict)
    is_injected: bool = False        # True for InjecAgent's post-tool-result malicious action
    turn_index: int = 0              # 0 = opening/legitimate action, 1+ = later actions in the same case


@dataclass(frozen=True)
class UnifiedCase:
    source: str                      # "agentharm" | "injecagent" | "toolemu" | "taubench_retail" | "taubench_airline" | "swebench"
    source_id: str                   # id/name from the original source file
    domain: str                      # one of peaq.domains.DOMAINS
    request_text: str                # opening user-facing instruction
    intended_actions: List[IntendedAction]
    context: Dict[str, Any]          # synthesized trusted-context flags for hard constraints
    ground_truth_label: str          # "benign" | "policy_violation" | "malicious"
    severity: str                    # "low" | "medium" | "high"
    category: str                    # original source-specific taxonomy label, kept for stratified reporting
    license_source: str              # e.g. "MIT", "MIT+usage-restriction", "Apache-2.0"
    raw: Dict[str, Any] = field(default_factory=dict)   # audit-only passthrough, never read by engine/simulator
    split: str = ""                  # "train" | "test", assigned once by combine.py

    @property
    def case_id(self) -> str:
        return f"{self.source}:{self.source_id}"

    def validate(self) -> None:
        if self.ground_truth_label not in GROUND_TRUTH_LABELS:
            raise ValueError(f"{self.case_id}: invalid ground_truth_label {self.ground_truth_label!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"{self.case_id}: invalid severity {self.severity!r}")
        if not self.intended_actions:
            raise ValueError(f"{self.case_id}: must have at least one intended action")
        if not self.request_text:
            raise ValueError(f"{self.case_id}: missing request_text")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["case_id"] = self.case_id
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "UnifiedCase":
        d = dict(d)
        d.pop("case_id", None)
        actions = [IntendedAction(**a) for a in d.pop("intended_actions")]
        return UnifiedCase(intended_actions=actions, **d)

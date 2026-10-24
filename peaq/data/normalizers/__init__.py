from typing import List

from peaq.data.schema import UnifiedCase

from . import agentharm, injecagent, toolemu, taubench, swebench, agentdojo

NORMALIZERS = {
    "agentharm": agentharm.normalize,
    "injecagent": injecagent.normalize,
    "toolemu": toolemu.normalize,
    "taubench": taubench.normalize,
    "swebench": swebench.normalize,
    "agentdojo": agentdojo.normalize,
}


def normalize_all(force_fetch: bool = False) -> List[UnifiedCase]:
    cases: List[UnifiedCase] = []
    for name, fn in NORMALIZERS.items():
        source_cases = fn(force_fetch=force_fetch)
        for c in source_cases:
            c.validate()
        cases.extend(source_cases)
    return cases

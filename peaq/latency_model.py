"""Simulated per-stage latency constants and an illustrative cost model.

The simulator is a deterministic offline simulation, not a timing of a real
deployment. The constants live here so every report traces back to one place.
"""
from typing import Dict

# --- Stage latencies (seconds) ---------------------------------------------

# Pre-planning gate: request classification plus decide_peaq()
GATE_LATENCY_HEURISTIC = 0.15
# Mean over 100 real qwen3:1.7b calls via Ollama; varies with hardware and load
GATE_LATENCY_LLM_CLASSIFIER = 10.73

# Execution time by criticality: riskier actions are modeled as slower
EXECUTION_LATENCY_BY_CRITICALITY: Dict[str, float] = {
    "low": 1.0,
    "medium": 1.5,
    "high": 2.2,
}

# Post-hoc verification baseline: runs after execution and never prevents it
POST_HOC_LATENCY = 0.3

# One runtime-authorization check, used by the runtime-auth baseline and by
# PEAQ's CONDITIONAL_APPROVE path
RUNTIME_AUTH_LATENCY = 4.5

# Pre-planning human review (HUMAN_REVIEW decision)
HUMAN_REVIEW_LATENCY = 3.5


def gate_latency_for(use_llm_classifier: bool) -> float:
    return GATE_LATENCY_LLM_CLASSIFIER if use_llm_classifier else GATE_LATENCY_HEURISTIC


def execution_latency_for(criticality: str) -> float:
    return EXECUTION_LATENCY_BY_CRITICALITY.get(criticality, EXECUTION_LATENCY_BY_CRITICALITY["medium"])


# --- Illustrative cost model ------------------------------------------------
#
# Placeholder rates, not real pricing data. They only show the shape of the
# tradeoff (fast-pass is nearly free, human review is expensive), so don't
# cite them.
DO_NOT_CITE_COST_PER_SECOND_USD = {
    "gate": 0.0005,          # heuristic classification: negligible compute cost
    "runtime_auth": 0.01,    # an automated policy/DB check
    "human_review": 0.15,    # roughly a $30k/month reviewer
    "execution": 0.0,        # the underlying system's cost, not PEAQ's
    "post_hoc": 0.005,
}
# Alias for older callers
ILLUSTRATIVE_COST_PER_SECOND_USD = DO_NOT_CITE_COST_PER_SECOND_USD


def illustrative_cost(stage: str, latency_seconds: float) -> float:
    """Illustrative cost in USD of `latency_seconds` spent in `stage`."""
    rate = DO_NOT_CITE_COST_PER_SECOND_USD.get(stage, 0.0)
    return round(rate * latency_seconds, 6)

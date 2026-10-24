from typing import Any, Dict

# Literature-derived profiles, built from tau-bench (arXiv:2406.12045),
# AgentHarm (arXiv:2410.09024) and InjecAgent (arXiv:2403.02691).
#
# Only numbers measured on the same model checkpoint are combined, so only
# GPT-4o and GPT-3.5-turbo have profiles. For other model families the papers
# tested different versions, which would give a profile of a model that never existed.
#
#   competence             = tau-bench pass^1, averaged over retail and airline
#   policy_adherence       = 1 - AgentHarm harm score without a jailbreak
#   adversarial_robustness = 1 - AgentHarm harm score with a jailbreak
AGENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "gpt-4o-class": {
        "name": "GPT-4o-class agent (literature-derived)",
        "competence": 0.482,              # tau-bench Table 2: (61.2% retail + 35.2% airline) / 2
        "policy_adherence": 0.516,        # AgentHarm Table 9: 1 - 48.4% no-jailbreak harm score
        "adversarial_robustness": 0.273,  # AgentHarm Table 9: 1 - 72.7% with-jailbreak-template harm score
        "source": "tau-bench arXiv:2406.12045 Table 2; AgentHarm arXiv:2410.09024 Table 9 (gpt-4o row, both papers)",
        "last_evaluated": "2026-08-18",
    },
    "gpt-3.5-turbo-class": {
        "name": "GPT-3.5-turbo-class agent (literature-derived)",
        "competence": 0.154,              # tau-bench Table 2: (20.0% retail + 10.8% airline) / 2
        # No AgentHarm figure exists for gpt-3.5-turbo, so policy_adherence
        # reuses the InjecAgent number below as an approximation.
        "policy_adherence": 0.763,        # approximation, see above
        "adversarial_robustness": 0.763,  # InjecAgent Table 3: 1 - 23.7% overall ASR (base setting)
        "source": "tau-bench arXiv:2406.12045 Table 2; InjecAgent arXiv:2403.02691 Table 3 (gpt-3.5-turbo row)",
        "last_evaluated": "2026-08-18",
    },

    # Synthetic archetypes: hand-picked values for stress-testing the gate,
    # not real model measurements.
    "expert-agent": {
        "name": "Expert Secure Agent (synthetic, high trust)",
        "competence": 0.96,
        "policy_adherence": 0.95,
        "adversarial_robustness": 0.92,
        "source": "synthetic -- not literature-derived",
        "last_evaluated": "2026-07-19",
    },
    "unaligned-agent": {
        "name": "Competent but Unaligned Agent (synthetic)",
        "competence": 0.90,
        "policy_adherence": 0.55,
        "adversarial_robustness": 0.35,
        "source": "synthetic -- not literature-derived",
        "last_evaluated": "2026-07-15",
    },
    "careful-but-weak-agent": {
        "name": "Policy-Strict but Low Competence Agent (synthetic)",
        "competence": 0.52,
        "policy_adherence": 0.98,
        "adversarial_robustness": 0.85,
        "source": "synthetic -- not literature-derived",
        "last_evaluated": "2026-07-17",
    },
    "reckless-agent": {
        "name": "Reckless & Vulnerable Agent (synthetic)",
        "competence": 0.70,
        "policy_adherence": 0.40,
        "adversarial_robustness": 0.20,
        "source": "synthetic -- not literature-derived",
        "last_evaluated": "2026-07-12",
    },
}

DEFAULT_PROFILE_ID = "gpt-4o-class"


def get_agent_profile(agent_id: str) -> Dict[str, float]:
    """Returns the trust metrics for an agent, or for DEFAULT_PROFILE_ID if
    the id is unknown."""
    profile = AGENT_PROFILES.get(agent_id, AGENT_PROFILES[DEFAULT_PROFILE_ID])
    return {
        "competence": profile["competence"],
        "policy_adherence": profile["policy_adherence"],
        "adversarial_robustness": profile["adversarial_robustness"],
    }

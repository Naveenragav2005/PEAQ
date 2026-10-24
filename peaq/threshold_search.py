"""Cross-validated grid search for the per-criticality threshold theta.

Only theta is searched; the grey-zone margin stays fixed. Each tier is
searched independently, since a case's actions at one tier are unaffected by
the theta of another.
"""
import random
import statistics
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import peaq.engine as engine
from peaq.simulator import run_suite_simulation, CONFIG_PEAQ_RUNTIME

DEFAULT_THETA_GRID = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 .. 0.95


@contextmanager
def _theta_override(tier: str, theta: float):
    """Temporarily sets engine.THRESHOLDS[tier] (decide_peaq reads it at call
    time), restoring the original afterward."""
    original = engine.THRESHOLDS[tier]
    engine.THRESHOLDS[tier] = theta
    try:
        yield
    finally:
        engine.THRESHOLDS[tier] = original


def _score_theta(tier: str, theta: float, agent_ids: List[str], cases: List[Dict[str, Any]],
                  violation_weight: float) -> Tuple[int, int, float]:
    """Returns (violations, false_rejections, objective) for one candidate
    theta, summed over all agents and counting only actions in `tier`."""
    with _theta_override(tier, theta):
        violations = 0
        false_rej = 0
        for agent_id in agent_ids:
            res = run_suite_simulation(agent_id, cases, CONFIG_PEAQ_RUNTIME)
            for r in res["results"]:
                tier_actions = [a for a in r["action_logs"] if a["criticality"] == tier]
                if not tier_actions:
                    continue
                if r["ground_truth_label"] != "benign":
                    if any(a["is_bad"] and a["allowed"] for a in tier_actions):
                        violations += 1
                else:
                    if r["false_rejection"] or r["false_human_review"]:
                        false_rej += 1
    objective = -(violations * violation_weight) - false_rej
    return violations, false_rej, objective


def grid_search_theta(tier: str, agent_ids: List[str], cases: List[Dict[str, Any]],
                       theta_grid: List[float] = None, violation_weight: float = 100) -> Dict[str, Any]:
    """Grid search without CV. Returns the best theta and the full scan.
    `violation_weight` is a policy choice: the cost of a missed violation
    relative to one wrongful rejection."""
    theta_grid = theta_grid or DEFAULT_THETA_GRID
    scan = []
    best_theta, best_objective = theta_grid[0], float("-inf")
    for theta in theta_grid:
        violations, false_rej, objective = _score_theta(tier, theta, agent_ids, cases, violation_weight)
        scan.append({"theta": theta, "violations": violations, "false_rejections": false_rej, "objective": objective})
        if objective > best_objective:
            best_objective = objective
            best_theta = theta
    return {"tier": tier, "best_theta": best_theta, "scan": scan}


def cross_validate_theta(tier: str, agent_ids: List[str], train_cases: List[Dict[str, Any]],
                          n_folds: int = 5, theta_grid: List[float] = None,
                          violation_weight: float = 100, seed: int = 20260817) -> Dict[str, Any]:
    """K-fold CV: grid-search theta on the other folds for each fold and
    report the mean and stdev of the chosen values, since a theta that swings
    between folds is unstable."""
    theta_grid = theta_grid or DEFAULT_THETA_GRID
    rng = random.Random(seed)
    shuffled = list(train_cases)
    rng.shuffle(shuffled)
    folds = [shuffled[i::n_folds] for i in range(n_folds)]

    per_fold_best = []
    for fold_idx in range(n_folds):
        fold_train = [c for i, f in enumerate(folds) if i != fold_idx for c in f]
        result = grid_search_theta(tier, agent_ids, fold_train, theta_grid, violation_weight)
        per_fold_best.append(result["best_theta"])

    mean_theta = round(statistics.mean(per_fold_best), 4)
    stdev_theta = round(statistics.stdev(per_fold_best), 4) if len(per_fold_best) > 1 else 0.0
    return {
        "tier": tier,
        "per_fold_best_theta": per_fold_best,
        "mean_theta": mean_theta,
        "stdev_theta": stdev_theta,
        "final_theta": round(mean_theta, 2),
    }

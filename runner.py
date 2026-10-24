"""Runs the full evaluation on the benchmark dataset (data/processed/cases.jsonl).

Steps:
  1. Load the train and test splits.
  2. Ablation on train, plus tier_sensitivity() to find which tiers are
     worth tuning. The ablation runs first because tuning theta for a tier
     where hard constraints decide every outcome cannot change anything.
  3. 5-fold CV grid search on train for each tunable tier.
  4. Evaluate on the held-out test split with the resulting thresholds.
  5. Test ablation summary, per-domain/per-source breakdown and an
     illustrative cost table.

The simulator is deterministic, so repeated runs give identical output.
"""
import argparse
import os
import random

import peaq.engine as engine
from peaq.ablation import run_ablation, summarize_ablation, tier_sensitivity, ABLATION_STRATEGIES
from peaq.simulator import (
    load_cases,
    run_suite_simulation,
    CONFIG_NONE,
    CONFIG_POST_HOC,
    CONFIG_RUNTIME_AUTH,
    CONFIG_PEAQ_RUNTIME,
)
from peaq.threshold_search import cross_validate_theta
from peaq.profiles import AGENT_PROFILES
from peaq.latency_model import illustrative_cost

CASES_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "cases.jsonl")

CRITICALITIES = ("low", "medium", "high")
BASELINE_CONFIGS = (CONFIG_NONE, CONFIG_POST_HOC, CONFIG_RUNTIME_AUTH, CONFIG_PEAQ_RUNTIME)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_ablation_table(summary: dict) -> None:
    print(f"{'strategy':22s} {'catch_rate':>11s} {'false_rej_rate':>15s} {'blocked/bad':>13s}")
    for strategy in ABLATION_STRATEGIES:
        s = summary[strategy]
        print(f"{strategy:22s} {s['catch_rate']:>10.2f}% {s['false_rejection_rate']:>14.2f}% "
              f"{s['bad_cases_blocked']:>6d}/{s['bad_cases']:<6d}")


def print_domain_source_breakdown(agent_ids, test_cases) -> None:
    domains = sorted({c["domain"] for c in test_cases})
    sources = sorted({c["source"] for c in test_cases})
    for agent_id in agent_ids:
        print(f"\n--- {agent_id} ---")
        print(f"{'domain':40s} {'source':16s} {'n':>5s} {'catch_rate':>11s} {'false_rej_rate':>15s}")
        for domain in domains:
            for source in sources:
                subset = [c for c in test_cases if c["domain"] == domain and c["source"] == source]
                if not subset:
                    continue
                res = run_suite_simulation(agent_id, subset, CONFIG_PEAQ_RUNTIME)
                catch = 100 * res["bad_cases_blocked"] / res["bad_cases"] if res["bad_cases"] else float("nan")
                fr = 100 * res["false_rejections"] / res["benign_cases"] if res["benign_cases"] else float("nan")
                print(f"{domain:40s} {source:16s} {len(subset):>5d} "
                      f"{catch:>10.1f}% {fr:>14.1f}%")


def print_cost_effectiveness_table(agent_ids, test_cases) -> None:
    print("(illustrative dollar figures only, see peaq/latency_model.py)")
    print(f"{'agent':22s} {'config':14s} {'avg_latency_s':>13s} {'illustrative_$/case':>20s}")
    for agent_id in agent_ids:
        for config in BASELINE_CONFIGS:
            res = run_suite_simulation(agent_id, test_cases, config)
            cost = (
                illustrative_cost("gate", res["avg_gate_latency"])
                + illustrative_cost("runtime_auth", res["avg_runtime_auth_latency"])
                + illustrative_cost("human_review", res["avg_human_review_latency"])
                + illustrative_cost("execution", res["avg_execution_latency"])
            )
            print(f"{agent_id:22s} {config:14s} {res['avg_latency']:>13.3f} {cost:>20.6f}")


def main():
    parser = argparse.ArgumentParser(description="PEAQ: full evaluation pipeline on the benchmark dataset.")
    parser.add_argument("--agents", nargs="*", default=list(AGENT_PROFILES.keys()),
                         help="Agent profile IDs to evaluate (default: all of peaq.profiles.AGENT_PROFILES).")
    parser.add_argument("--subset", type=int, default=None,
                         help="Limit to N cases per split (train/test), for faster iteration.")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds for threshold search (default: 5).")
    parser.add_argument("--violation-weight", type=float, default=100,
                         help="Grid-search objective: cost of one missed violation relative to one false rejection (default: 100; a policy choice).")
    parser.add_argument("--theta-grid-step", type=float, default=0.05, help="Grid search step size (default: 0.05).")
    parser.add_argument("--exclude-sources", nargs="*", default=[],
                         help="Case sources to drop before running (e.g. agentdojo).")
    args = parser.parse_args()

    theta_grid = [round(0.30 + args.theta_grid_step * i, 2)
                  for i in range(int((0.95 - 0.30) / args.theta_grid_step) + 1)]

    train_cases = load_cases(CASES_PATH, split="train")
    test_cases = load_cases(CASES_PATH, split="test")
    if args.exclude_sources:
        excluded = set(args.exclude_sources)
        train_cases = [c for c in train_cases if c["source"] not in excluded]
        test_cases = [c for c in test_cases if c["source"] not in excluded]
    if args.subset:
        rng = random.Random(20260817)
        train_cases = rng.sample(train_cases, min(args.subset, len(train_cases)))
        test_cases = rng.sample(test_cases, min(args.subset, len(test_cases)))

    section("STEP 1: DATASET")
    print(f"train: {len(train_cases)} cases   test: {len(test_cases)} cases   agents: {args.agents}")

    section("STEP 2: ABLATION (on TRAIN, decides which tiers are worth tuning)")
    train_ablation = run_ablation(args.agents, train_cases)
    train_summary = summarize_ablation(train_ablation)
    print_ablation_table(train_summary)
    sensitivity = tier_sensitivity(args.agents, train_cases)
    print("\ntier_sensitivity (can trust-score tuning change any outcome at this tier?):")
    for tier in CRITICALITIES:
        print(f"  {tier:8s} {sensitivity[tier]}")
    tunable_tiers = [t for t in CRITICALITIES if sensitivity[t]]
    skipped_tiers = [t for t in CRITICALITIES if not sensitivity[t]]
    print(f"\n-> will grid-search: {tunable_tiers}")
    print(f"-> skipping (provably inert -- hard constraints already decide every outcome): {skipped_tiers}")

    section("STEP 3: 5-FOLD CV THRESHOLD SEARCH (TRAIN, tunable tiers only)")
    original_thresholds = dict(engine.THRESHOLDS)
    final_thresholds = dict(engine.THRESHOLDS)
    for tier in tunable_tiers:
        cv_result = cross_validate_theta(tier, args.agents, train_cases, n_folds=args.n_folds,
                                          theta_grid=theta_grid, violation_weight=args.violation_weight)
        print(f"\n{tier}: per-fold best theta = {cv_result['per_fold_best_theta']}")
        print(f"  mean = {cv_result['mean_theta']}  stdev = {cv_result['stdev_theta']}  -> final theta = {cv_result['final_theta']}")
        final_thresholds[tier] = cv_result["final_theta"]
    for tier in skipped_tiers:
        print(f"\n{tier}: not tuned -- keeping policy-calibrated theta = {original_thresholds[tier]}")

    section("STEP 4: HELD-OUT TEST EVALUATION (headline numbers)")
    print(f"policy-calibrated thresholds (unchanged where not tunable)    : {original_thresholds}")
    print(f"grid-searched thresholds (this run, tunable tiers only):            {final_thresholds}")

    for label, thresholds in [("policy-calibrated", original_thresholds), ("grid-searched", final_thresholds)]:
        engine.THRESHOLDS = dict(thresholds)
        print(f"\n--- {label} ---")
        print(f"{'agent':22s} {'config':14s} {'catch_rate':>11s} {'false_rej_rate':>15s} {'avg_latency_s':>13s}")
        for agent_id in args.agents:
            for config in BASELINE_CONFIGS:
                res = run_suite_simulation(agent_id, test_cases, config)
                catch = 100 * res["bad_cases_blocked"] / res["bad_cases"] if res["bad_cases"] else float("nan")
                fr = 100 * res["false_rejections"] / res["benign_cases"] if (config == CONFIG_PEAQ_RUNTIME and res["benign_cases"]) else float("nan")
                print(f"{agent_id:22s} {config:14s} {catch:>10.1f}% {fr:>14.1f}% {res['avg_latency']:>13.3f}")
                if config == CONFIG_PEAQ_RUNTIME and label == "grid-searched" and res["benign_cases"] and fr >= 90.0:
                    print(f"    ^^ CAUTION: {fr:.1f}% false-rejection rate is not a usable operating point -- "
                          f"this threshold set (violation_weight={args.violation_weight}) is blocking almost every "
                          f"benign request for this profile, not discriminating.")

    section("STEP 5: TEST ABLATION SUMMARY (headline, under POLICY-CALIBRATED thresholds)")
    print("Uses the policy-calibrated thresholds, since the grid-searched set can block almost every benign request.")
    engine.THRESHOLDS = dict(original_thresholds)
    test_ablation = run_ablation(args.agents, test_cases)
    test_summary = summarize_ablation(test_ablation)
    print_ablation_table(test_summary)

    section("STEP 6: PER-DOMAIN x PER-SOURCE BREAKDOWN (TEST, PEAQ config, policy-calibrated thresholds)")
    print_domain_source_breakdown(args.agents, test_cases)

    section("STEP 7: ILLUSTRATIVE COST-EFFECTIVENESS (TEST, policy-calibrated thresholds)")
    print_cost_effectiveness_table(args.agents, test_cases)

    engine.THRESHOLDS = original_thresholds


if __name__ == "__main__":
    main()

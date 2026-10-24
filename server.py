"""PEAQ dashboard backend (Starlette).

Endpoints:
  GET  /api/config     thresholds, weights, margins, agent profiles, domain actions
  POST /api/config     update thresholds/weights in memory (reset on restart)
  GET  /api/manifest   dataset manifest
  GET  /api/cases      sample of benchmark cases, filterable by domain and split
  POST /api/classify   classify a request into (action, criticality)
  POST /api/simulate   decision trace for a single request
  POST /api/benchmark  run the four configurations over a slice of the dataset
  POST /api/ablation   hard-constraint-only vs trust-score-only vs combined

This is an interactive view over the same engine; the evaluation numbers
come from runner.py.
"""
import json
import os
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse, FileResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn

import peaq.engine as engine
from peaq.profiles import AGENT_PROFILES, get_agent_profile
from peaq.domains import DOMAINS, get_actions, classify_criticality_rule_based
from peaq.simulator import (
    load_cases,
    run_suite_simulation,
    CONFIG_NONE,
    CONFIG_POST_HOC,
    CONFIG_RUNTIME_AUTH,
    CONFIG_PEAQ_RUNTIME,
)
from peaq.ablation import run_ablation, summarize_ablation

BASE_DIR = os.path.dirname(__file__)
CASES_PATH = os.path.join(BASE_DIR, "data", "processed", "cases.jsonl")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "processed", "manifest.json")
SAMPLE_SEED = 20260817

BASELINE_CONFIGS = (CONFIG_NONE, CONFIG_POST_HOC, CONFIG_RUNTIME_AUTH, CONFIG_PEAQ_RUNTIME)

_cases_cache: Optional[List[Dict[str, Any]]] = None


def _all_cases() -> List[Dict[str, Any]]:
    global _cases_cache
    if _cases_cache is None:
        _cases_cache = load_cases(CASES_PATH) if os.path.exists(CASES_PATH) else []
    return _cases_cache


def _dataset_missing_response() -> JSONResponse:
    return JSONResponse(
        {"error": "Dataset not built. Run `python -m peaq.data.fetchers` then `python -m peaq.data.combine`."},
        status_code=404,
    )


def _json_safe(value: Any) -> Any:
    """Replaces non-finite floats with None, since JSONResponse can't
    serialize them (tau-bench uses inf for "no refund cap")."""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _filter_cases(cases: List[Dict[str, Any]], domain: Optional[str], split: Optional[str]) -> List[Dict[str, Any]]:
    out = cases
    if split:
        out = [c for c in out if c["split"] == split]
    if domain and domain != "all":
        out = [c for c in out if c["domain"] == domain]
    return out


def _build_logs(decision: Dict[str, Any]) -> List[str]:
    logs = [
        "SYSTEM: PEAQ pre-execution gate invoked.",
        f"Classified action '{decision['action']}' (criticality: {decision['criticality'].upper()}) "
        f"in domain '{decision['domain']}'.",
        f"Trust score computed: {decision['trust_score']:.4f}  "
        f"(required threshold theta={decision['required_threshold']:.2f}, "
        f"grey-zone margin delta={decision['grey_zone_margin']:.2f}).",
    ]
    if decision["hard_constraints_passed"]:
        logs.append("Hard constraint check PASSED -- no domain policy violation detected.")
    else:
        logs.append(f"Hard constraint check FAILED -- {decision['reason']}")

    d = decision["decision"]
    if d == "APPROVE":
        logs.append("DECISION: APPROVE -- fast pass, planning capability granted immediately.")
    elif d == "CONDITIONAL_APPROVE":
        logs.append("DECISION: CONDITIONAL_APPROVE -- trust score in the grey zone; routed to runtime authorization.")
    elif d == "HUMAN_REVIEW":
        logs.append("DECISION: HUMAN_REVIEW -- action blocked pending human sign-off (H4, missing approval).")
    else:
        logs.append("DECISION: REJECT -- action blocked, agent not qualified for this action.")
    return logs


# --- Routes -----------------------------------------------------------------

async def get_config(request):
    return JSONResponse({
        "thresholds": engine.THRESHOLDS,
        "weights": engine.WEIGHTS,
        "grey_zone_margins": engine.GREY_ZONE_MARGINS,
        "profiles": AGENT_PROFILES,
        "domains": {d: get_actions(d) for d in DOMAINS},
    })


async def update_config(request):
    try:
        data = await request.json()
        if "thresholds" in data:
            engine.THRESHOLDS.update(data["thresholds"])
        if "weights" in data:
            for criticality, w_dict in data["weights"].items():
                if criticality in engine.WEIGHTS:
                    engine.WEIGHTS[criticality].update(w_dict)
        return JSONResponse({"status": "success", "thresholds": engine.THRESHOLDS, "weights": engine.WEIGHTS})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


async def get_manifest(request):
    if not os.path.exists(MANIFEST_PATH):
        return _dataset_missing_response()
    return JSONResponse(json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8")))


async def get_cases(request):
    cases = _all_cases()
    if not cases:
        return _dataset_missing_response()
    domain = request.query_params.get("domain")
    split = request.query_params.get("split", "test")
    limit = int(request.query_params.get("limit", 30))

    subset = _filter_cases(cases, domain, split)
    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(subset, min(limit, len(subset)))
    out = [{
        "case_id": c["case_id"],
        "source": c["source"],
        "domain": c["domain"],
        "request_text": c["request_text"],
        "ground_truth_label": c["ground_truth_label"],
        "severity": c["severity"],
        "context": c["context"],
        "intended_actions": c["intended_actions"],
    } for c in sample]
    return JSONResponse(_json_safe(out))


async def classify_text(request):
    try:
        data = await request.json()
        req_text = data.get("request", "")
        domain = data.get("domain", DOMAINS[0])
        use_llm = data.get("use_llm", False)
        res = engine.classify_request_ollama(req_text, domain) if use_llm else engine.classify_request_heuristics(req_text, domain)
        return JSONResponse(res)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


async def simulate(request):
    """Single-action decision trace, powering the Gate Simulator tab."""
    try:
        data = await request.json()
        agent_id = data.get("agent_id", "gpt-4o-class")
        req_text = data.get("request", "")
        domain = data.get("domain", DOMAINS[0])
        use_llm = data.get("use_llm", False)
        context = data.get("context", {}) or {}
        action_override = data.get("action")

        if action_override:
            action = action_override
            criticality = classify_criticality_rule_based(domain, action)
            classification_reason = "Action taken directly from the selected predefined test case."
        else:
            classification = (
                engine.classify_request_ollama(req_text, domain) if use_llm
                else engine.classify_request_heuristics(req_text, domain)
            )
            action = classification["action"]
            criticality = classification.get("criticality") or classify_criticality_rule_based(domain, action)
            classification_reason = classification.get("reason", "")

        profile = get_agent_profile(agent_id)
        resolved_tc = engine.resolve_trusted_context(req_text, context, domain)
        decision = engine.decide_peaq(agent_id, req_text, domain, action, criticality, profile, resolved_tc)
        decision["classification_reason"] = classification_reason
        decision["logs"] = _build_logs(decision)
        return JSONResponse(_json_safe(decision))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


async def run_benchmark(request):
    try:
        data = await request.json()
        agent_id = data.get("agent_id", "gpt-4o-class")
        domain = data.get("domain", "all")
        split = data.get("split", "test")
        limit = data.get("limit")

        cases = _all_cases()
        if not cases:
            return _dataset_missing_response()
        subset = _filter_cases(cases, domain, split)
        if limit:
            rng = random.Random(SAMPLE_SEED)
            subset = rng.sample(subset, min(int(limit), len(subset)))
        if not subset:
            return JSONResponse({"error": f"No cases match domain={domain!r} split={split!r}."}, status_code=404)

        bench_results = {}
        for cfg in BASELINE_CONFIGS:
            res = run_suite_simulation(agent_id, subset, cfg)
            catch_rate = round(100 * res["bad_cases_blocked"] / res["bad_cases"], 2) if res["bad_cases"] else None
            false_rej_rate = (
                round(100 * res["false_rejections"] / res["benign_cases"], 2)
                if (cfg == CONFIG_PEAQ_RUNTIME and res["benign_cases"]) else None
            )
            entry = {k: v for k, v in res.items() if k != "results"}
            entry["catch_rate"] = catch_rate
            entry["false_rejection_rate"] = false_rej_rate
            if cfg == CONFIG_PEAQ_RUNTIME:
                entry["case_results"] = [{
                    "case_id": r["case_id"],
                    "source": r["source"],
                    "domain": r["domain"],
                    "ground_truth_label": r["ground_truth_label"],
                    "policy_violated": r["policy_violated"],
                    "adversarial_bypass": r["adversarial_bypass"],
                    "false_rejection": r["false_rejection"],
                    "false_human_review": r["false_human_review"],
                    "blocked_bad_action": r["blocked_bad_action"],
                    "total_latency": r["total_latency"],
                    "action_logs": r["action_logs"],
                } for r in res["results"]]
            bench_results[cfg] = entry

        return JSONResponse({
            "agent_id": agent_id, "domain": domain, "split": split,
            "n_cases": len(subset), "benchmarks": bench_results,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


async def run_ablation_endpoint(request):
    try:
        data = await request.json()
        agent_id = data.get("agent_id", "gpt-4o-class")
        domain = data.get("domain", "all")
        split = data.get("split", "test")
        limit = data.get("limit")

        cases = _all_cases()
        if not cases:
            return _dataset_missing_response()
        subset = _filter_cases(cases, domain, split)
        if limit:
            rng = random.Random(SAMPLE_SEED)
            subset = rng.sample(subset, min(int(limit), len(subset)))
        if not subset:
            return JSONResponse({"error": f"No cases match domain={domain!r} split={split!r}."}, status_code=404)

        results = run_ablation([agent_id], subset)
        summary = summarize_ablation(results)
        return JSONResponse({"agent_id": agent_id, "domain": domain, "split": split, "n_cases": len(subset), "summary": summary})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


async def serve_index(request):
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


routes = [
    Route("/", serve_index),
    Route("/api/config", get_config, methods=["GET"]),
    Route("/api/config", update_config, methods=["POST"]),
    Route("/api/manifest", get_manifest, methods=["GET"]),
    Route("/api/cases", get_cases, methods=["GET"]),
    Route("/api/classify", classify_text, methods=["POST"]),
    Route("/api/simulate", simulate, methods=["POST"]),
    Route("/api/benchmark", run_benchmark, methods=["POST"]),
    Route("/api/ablation", run_ablation_endpoint, methods=["POST"]),
    Mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"),
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
]

app = Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    print("Starting PEAQ Server on http://localhost:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")

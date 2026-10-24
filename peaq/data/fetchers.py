"""Fetches and caches the raw benchmark data under data/raw (gitignored).
See ATTRIBUTION.md for each source's license.

tau-bench and AgentDojo are pinned to a commit so their parsers don't break
when the upstream format changes.
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import httpx

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

TAUBENCH_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
AGENTDOJO_COMMIT = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"

_AGENTHARM_BASE = "https://huggingface.co/datasets/ai-safety-institute/AgentHarm/raw/main/benchmark"
_INJECAGENT_BASE = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data"
_TOOLEMU_BASE = "https://raw.githubusercontent.com/ryoungj/ToolEmu/main/assets"
_TAUBENCH_BASE = f"https://raw.githubusercontent.com/sierra-research/tau-bench/{TAUBENCH_COMMIT}/tau_bench/envs"
_SWEBENCH_ROWS_API = "https://datasets-server.huggingface.co/rows"
_AGENTDOJO_BASE = f"https://raw.githubusercontent.com/ethz-spylab/agentdojo/{AGENTDOJO_COMMIT}/src/agentdojo/default_suites/v1"
AGENTDOJO_SUITES = ("banking", "slack", "travel", "workspace")


def _fetch_text(url: str, dest: Path, force: bool = False, timeout: float = 30.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    return dest


def fetch_agentharm(force: bool = False) -> Dict[str, Path]:
    out = {}
    files = {
        "harmful_test": "harmful_behaviors_test_public.json",
        "harmful_validation": "harmful_behaviors_validation.json",
        "benign_test": "benign_behaviors_test_public.json",
        "benign_validation": "benign_behaviors_validation.json",
        "license": "../LICENSE",
    }
    for key, name in files.items():
        url = f"{_AGENTHARM_BASE}/{name}"
        dest = RAW_DIR / "agentharm" / Path(name).name
        out[key] = _fetch_text(url, dest, force=force)
    return out


def fetch_injecagent(force: bool = False) -> Dict[str, Path]:
    out = {}
    files = ["test_cases_dh_base.json", "test_cases_ds_base.json", "tools.json"]
    for name in files:
        url = f"{_INJECAGENT_BASE}/{name}"
        dest = RAW_DIR / "injecagent" / name
        out[name] = _fetch_text(url, dest, force=force)
    return out


def fetch_toolemu(force: bool = False) -> Dict[str, Path]:
    dest = RAW_DIR / "toolemu" / "all_cases.json"
    return {"cases": _fetch_text(f"{_TOOLEMU_BASE}/all_cases.json", dest, force=force)}


def fetch_taubench(force: bool = False) -> Dict[str, Path]:
    out = {}
    files = {
        "retail_tasks": "retail/tasks_test.py",
        "retail_wiki": "retail/wiki.md",
        "airline_tasks": "airline/tasks_test.py",
        "airline_wiki": "airline/wiki.md",
    }
    for key, rel in files.items():
        url = f"{_TAUBENCH_BASE}/{rel}"
        dest = RAW_DIR / "taubench" / rel.replace("/", "_")
        out[key] = _fetch_text(url, dest, force=force)
    return out


def fetch_swebench(force: bool = False, limit: int = 300) -> Path:
    """Fetches a sample of SWE-bench rows (the full set is not needed)."""
    dest = RAW_DIR / "swebench" / f"sample_{limit}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    rows: List[dict] = []
    offset = 0
    page = 100
    with httpx.Client(timeout=30.0) as client:
        while len(rows) < limit:
            resp = client.get(_SWEBENCH_ROWS_API, params={
                "dataset": "princeton-nlp/SWE-bench",
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": min(page, limit - len(rows)),
            })
            resp.raise_for_status()
            data = resp.json()
            batch = [r["row"] for r in data.get("rows", [])]
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
    dest.write_text(json.dumps(rows), encoding="utf-8")
    return dest


def fetch_agentdojo(force: bool = False) -> Dict[str, Path]:
    """Fetches the v1 suites' task files (parsed with ast, never executed)
    and the license."""
    out = {}
    for suite in AGENTDOJO_SUITES:
        for fname in ("user_tasks.py", "injection_tasks.py"):
            url = f"{_AGENTDOJO_BASE}/{suite}/{fname}"
            dest = RAW_DIR / "agentdojo" / f"{suite}_{fname}"
            out[f"{suite}_{fname}"] = _fetch_text(url, dest, force=force)
    out["LICENSE"] = _fetch_text(
        f"https://raw.githubusercontent.com/ethz-spylab/agentdojo/{AGENTDOJO_COMMIT}/LICENSE",
        RAW_DIR / "agentdojo" / "LICENSE", force=force,
    )
    return out


FETCHERS: Dict[str, Callable[..., object]] = {
    "agentharm": fetch_agentharm,
    "injecagent": fetch_injecagent,
    "toolemu": fetch_toolemu,
    "taubench": fetch_taubench,
    "swebench": fetch_swebench,
    "agentdojo": fetch_agentdojo,
}


def fetch_all(force: bool = False) -> None:
    for name, fn in FETCHERS.items():
        print(f"Fetching {name}...")
        fn(force=force)
    print("Done. Raw data cached under", RAW_DIR)


if __name__ == "__main__":
    fetch_all()

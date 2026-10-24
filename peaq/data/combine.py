"""Merges the normalized benchmark sources, assigns a stratified train/test
split, and writes data/processed/cases.jsonl and manifest.json (both
gitignored, so run `python -m peaq.data.combine` locally)."""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from peaq.data.normalizers import normalize_all
from peaq.data.schema import UnifiedCase

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
CASES_PATH = PROCESSED_DIR / "cases.jsonl"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

TEST_FRACTION = 0.3
SPLIT_SEED = 20260817


def assign_splits(cases: List[UnifiedCase], test_fraction: float = TEST_FRACTION, seed: int = SPLIT_SEED) -> List[UnifiedCase]:
    """Seeded split stratified by (domain, source, label)."""
    strata: Dict[tuple, List[UnifiedCase]] = defaultdict(list)
    for c in cases:
        strata[(c.domain, c.source, c.ground_truth_label)].append(c)

    rng = random.Random(seed)
    out: List[UnifiedCase] = []
    for key, group in strata.items():
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_fraction)) if len(group) > 1 else 0
        for i, c in enumerate(group):
            split = "test" if i < n_test else "train"
            out.append(UnifiedCase(
                source=c.source, source_id=c.source_id, domain=c.domain,
                request_text=c.request_text, intended_actions=c.intended_actions,
                context=c.context, ground_truth_label=c.ground_truth_label,
                severity=c.severity, category=c.category,
                license_source=c.license_source, raw=c.raw, split=split,
            ))
    return out


def build_manifest(cases: List[UnifiedCase]) -> Dict:
    counts = Counter((c.domain, c.source, c.ground_truth_label, c.split) for c in cases)
    manifest = {
        "total_cases": len(cases),
        "by_source": dict(Counter(c.source for c in cases)),
        "by_domain": dict(Counter(c.domain for c in cases)),
        "by_label": dict(Counter(c.ground_truth_label for c in cases)),
        "by_severity": dict(Counter(c.severity for c in cases)),
        "by_split": dict(Counter(c.split for c in cases)),
        "by_domain_source_label_split": [
            {"domain": d, "source": s, "label": l, "split": sp, "count": n}
            for (d, s, l, sp), n in sorted(counts.items())
        ],
    }
    return manifest


def combine(force_fetch: bool = False, write: bool = True) -> List[UnifiedCase]:
    cases = normalize_all(force_fetch=force_fetch)
    cases = assign_splits(cases)

    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with CASES_PATH.open("w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c.to_dict(), default=str) + "\n")
        MANIFEST_PATH.write_text(json.dumps(build_manifest(cases), indent=2), encoding="utf-8")
        print(f"Wrote {len(cases)} cases to {CASES_PATH}")
        print(f"Wrote manifest to {MANIFEST_PATH}")
    return cases


def load_cases() -> List[UnifiedCase]:
    if not CASES_PATH.exists():
        raise FileNotFoundError(f"{CASES_PATH} does not exist -- run `python -m peaq.data.combine` first")
    cases = []
    with CASES_PATH.open(encoding="utf-8") as f:
        for line in f:
            cases.append(UnifiedCase.from_dict(json.loads(line)))
    return cases


if __name__ == "__main__":
    combine()

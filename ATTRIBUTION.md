# Data Attribution

PEAQ's evaluation dataset is built by fetching and normalizing 6 externally
published benchmarks at evaluation time (`python -m peaq.data.combine`).
**None of this data is committed to this repository** (`data/raw/` and
`data/processed/` are gitignored) — running the fetch/combine pipeline is a
required local setup step. This file records each source's license and any
notices required when redistributing derived results (tables, aggregate
statistics, or excerpted cases) from this evaluation.

## AgentHarm

- Source: `ai-safety-institute/AgentHarm` on Hugging Face (UK AI Safety
  Institute / Gray Swan AI).
- Files used: `benchmark/{harmful,benign}_behaviors_{test_public,validation}.json`.
- License: **MIT with an additional clause** — "We prohibit using the
  dataset and benchmark for purposes besides improving the safety and
  security of AI systems." This evaluation (building and validating a
  pre-execution agent safety gate) falls squarely within that permitted use.
  Any redistribution of AgentHarm-derived content must keep this clause and
  the copyright notice attached.
- Copyright (c) 2024 Gray Swan AI and UK AI Safety Institute.

## InjecAgent

- Source: `uiuc-kang-lab/InjecAgent` on GitHub.
- Files used: `data/test_cases_dh_base.json`, `data/test_cases_ds_base.json`,
  `data/tools.json`.
- License: MIT. Free to use, adapt, and redistribute with attribution.

## ToolEmu

- Source: `ryoungj/ToolEmu` on GitHub.
- Files used: `assets/all_cases.json`.
- License: **Apache-2.0** — redistribution of derived content must include
  the standard Apache-2.0 attribution notice.

## tau-bench

- Source: `sierra-research/tau-bench` on GitHub, pinned to commit
  `59a200c6d575d595120f1cb70fea53cef0632f6b` (see `peaq/data/fetchers.py` —
  pinned rather than tracking `main` so the tau-bench parser can't silently
  break if the upstream file format changes).
- Files used: `tau_bench/envs/{retail,airline}/{tasks_test.py,wiki.md}`.
- License: MIT.

## SWE-bench

- Source: `princeton-nlp/SWE-bench` on Hugging Face (canonical GitHub repo:
  `SWE-bench/SWE-bench`), fetched via the `datasets-server.huggingface.co`
  rows API.
- License: MIT.

## AgentDojo

- Source: `ethz-spylab/agentdojo` on GitHub (ETH Zurich SPY Lab), pinned to
  commit `089ed468cf3ed0322acc66b0211f26d9d90dbf60` (see
  `peaq/data/fetchers.py`'s `AGENTDOJO_COMMIT`).
- Files used: `src/agentdojo/default_suites/v1/{banking,slack,travel,workspace}/{user_tasks.py,injection_tasks.py}`
  — the paper's original 4 suites only, not the later attack/defense
  variant directories (`v1_1`, `v1_2`, ...), which reuse the same
  underlying tasks. Parsed statically via `ast` (never executed or
  imported as a package) — see `peaq/data/normalizers/agentdojo.py`'s
  module docstring for why.
- License: MIT.
- Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic,
  Luca Beurer-Kellner, Marc Fischer, Florian Tramèr.

## Reproducing the dataset

```bash
python -m peaq.data.fetchers   # cache raw source files under data/raw/
python -m peaq.data.combine    # normalize + stratified train/test split -> data/processed/
```

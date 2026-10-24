# PEAQ: Pre-Execution Agent Qualification

PEAQ is a deterministic checkpoint that decides whether an AI agent may start
an action at all, before it plans or calls any tool. Most agent safety work
intercepts at runtime, after the agent has already planned. PEAQ moves the
decision earlier and keeps it free of LLM judgment.

For each request the gate:

1. **Classifies** the request into an action and a criticality (low, medium or
   high), using keyword rules or an optional local Ollama model. Only the label
   passes on, never the model's reasoning.
2. **Resolves a trusted context** (order records, verification and approval
   flags) from data the request text cannot influence.
3. **Checks hard constraints**, typed rules per action such as "customer
   deletion needs human sign-off" or "data export needs a verified requester".
4. **Computes a trust score** from the agent's competence, policy adherence and
   adversarial robustness, weighted by criticality.
5. **Returns a decision**: `APPROVE`, `CONDITIONAL_APPROVE` (plan allowed, but
   runtime checks still run), `HUMAN_REVIEW` or `REJECT`.

## Evaluation

The gate is evaluated on 2,191 cases from six public benchmarks (AgentHarm,
InjecAgent, ToolEmu, tau-bench, SWE-bench, AgentDojo), spread over five agent
domains: customer service, software engineering, financial operations,
communications, and data and systems access. Four safety configurations are
compared: no protection, post-hoc detection, runtime authorization, and PEAQ.

Agent trust profiles come from published results (GPT-4o and GPT-3.5-turbo)
plus four synthetic archetypes for stress-testing. Per-tier approval thresholds
are set to the highest trust score among the real profiles.

On the held-out test split, the hard constraints alone catch 98.3% of bad
cases, and adding the trust score raises that only to 98.6% while increasing
false rejections (40% to 58%). The trust score contributes little for most of
this dataset. See [Limitations](#limitations).

## Setup

Requires Python 3.8 or newer.

```bash
pip install -r requirements.txt
```

Build the dataset (needs network access, and writes to the gitignored `data/`
folder):

```bash
python -m peaq.data.fetchers   # download the raw benchmark files
python -m peaq.data.combine    # normalize them into data/processed/cases.jsonl
```

Optionally, for the LLM-assisted classifier, install [Ollama](https://ollama.com)
and pull a model:

```bash
ollama pull qwen3:1.7b
```

Set `OLLAMA_MODEL` (in the environment or a `.env` file) to use a different
model. Without Ollama, classification uses the built-in keyword rules, and
nothing else depends on it. `python test_ollama.py` checks the connection.

## Usage

```bash
python runner.py                # full evaluation: ablation, threshold search, held-out test
python runner.py --subset 100   # faster run on a random subset
python server.py                # web dashboard
```

Run the tests:

```bash
python -m unittest tests/test_peaq.py tests/test_simulator.py tests/test_ablation.py \
    tests/test_threshold_search.py tests/test_agentdojo_normalizer.py
```

Tests that need the dataset are skipped until it has been built.

## Project structure

```
peaq/
  engine.py            gate decision, trust score, threshold calibration, classifiers
  domains/             per-domain actions, criticality and hard-constraint predicates
  profiles.py          agent trust profiles (2 from the literature, 4 synthetic)
  simulator.py         deterministic evaluation over the benchmark cases
  ablation.py          hard-constraints-only vs trust-score-only vs combined
  threshold_search.py  cross-validated search for the approval thresholds
  latency_model.py     simulated per-stage latency and an illustrative cost model
  agent.py             demo of a live Ollama agent behind the gate (not evaluated)
  data/                benchmark fetchers and normalizers
runner.py              runs the full evaluation pipeline
server.py, static/     web dashboard
tests/                 unit and integration tests
ATTRIBUTION.md         licenses for the benchmark datasets
```

## Metrics

- **Catch rate**: share of bad cases where the harmful action was blocked.
- **False-rejection rate**: share of benign cases wrongly blocked or sent to
  human review.
- **Policy violations and adversarial bypasses**: bad actions that ran anyway.
- **Latency** per pipeline stage. This is simulated, and the cost figures are
  illustrative placeholders, not real pricing.

## Limitations

- The false-rejection rate is high. With thresholds set from the real
  profiles, both literature-derived profiles score low enough that the gate
  rejects most benign requests, so PEAQ is currently more conservative than
  useful.
- Thresholds are calibrated on only two real model profiles. The top-scoring
  profile clears its own bar by construction, so that outcome is not
  independent evidence.
- Most bad cases in the dataset carry an empty context, so the hard constraints
  decide almost all of them. Grid-searching the high-criticality threshold has
  no effect for the same reason.
- Context flags for the benchmark cases are synthesized from each source's own
  fields, not recorded by a real deployment.
- The dashboard is partial. The benchmark and ablation views need the dataset
  built first, and the live agent demo covers only two domains.

## Use of AI

I used Claude Code to generate test cases and to help write and refactor parts of the code.

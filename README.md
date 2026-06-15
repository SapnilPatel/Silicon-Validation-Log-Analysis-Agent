# SVLA-Agent

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange)](https://github.com/astral-sh/ruff)

**Silicon Validation Log Analysis Agent** — an agentic AI debug assistant for
post-silicon validation logs. SVLA-Agent ingests large multi-source silicon test
logs, parses them into structured events, classifies failures, clusters recurring
failure families, retrieves similar historical incidents (RAG), and uses Claude to
generate explainable root-cause hypotheses with remediation steps — then writes
clean Markdown/JSON debug reports and an interactive dashboard.

> The whole pipeline runs **offline** out of the box (deterministic fallbacks for
> embeddings, retrieval and hypotheses). Add an `ANTHROPIC_API_KEY` to upgrade the
> hypothesis agent to Claude.

---

## What it does

```
raw logs ──▶ ingest ──▶ parse ──▶ classify ──▶ cluster ──▶ RAG ──▶ hypothesize ──▶ report
            (stream)  (Drain3)   (failures)   (HDBSCAN)  (Chroma)   (Claude)      (MD/JSON)
```

| Stage | Module | What happens |
|---|---|---|
| **Ingest** | `ingest/reader.py` | Discover & group per-source run files (`run_*.<src>.log`, gzip-aware), stream lines lazily |
| **Parse** | `parse/parser.py` | Regex line structure (ts / subsys / level / `k=v` / multi-line dumps) + Drain3 template mining |
| **Classify** | `classify/classifier.py` | Turn error/fatal events into prioritised `Failure` records (subsystem-aware rules) |
| **Embed** | `embeddings/embedder.py` | sentence-transformers embeddings (deterministic hashing fallback offline) |
| **Cluster** | `cluster/clusterer.py` | Group failures into families with `sklearn.cluster.HDBSCAN` |
| **RAG** | `rag/store.py` | Index/query historical failures in a persistent Chroma collection (cosine) |
| **Hypothesize** | `hypothesize/agent.py` | Claude tool-use for structured root-cause hypotheses; grounded rule-based fallback |
| **Report** | `report/reporter.py` | Jinja2 Markdown + JSON debug reports |
| **Evaluate** | `eval/metrics.py` | Detection P/R/F1, failure-typing P/R/F1, clustering quality, RAG recall@k |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full data flow and schemas.

---

## Quickstart

The project ships a virtualenv at `.svla/` with all dependencies installed.

```bash
# 1. Generate the synthetic silicon-log dataset (100 multi-source runs + labels)
.svla/bin/svla generate --runs 100

# 2. Run the full analysis pipeline -> writes reports/run_*.{json,md}
.svla/bin/svla process

# 3. Evaluate pipeline output against the ground-truth labels
.svla/bin/svla eval

# 4. Explore results in the dashboard
.svla/bin/streamlit run app/streamlit_app.py
```

> Not using the bundled venv? Install the package instead:
> `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,app]"`,
> then use `svla ...` directly.

A `Makefile` wraps the common flows: `make data`, `make process`, `make eval`,
`make app`, `make check`.

---

## CLI

```
svla generate   Generate the synthetic silicon-log dataset
svla process    Run the full analysis pipeline and write debug reports
svla index      (Re)build the RAG vector store of historical failures
svla eval       Evaluate pipeline output against ground-truth labels
svla version    Print the version
```

Useful flags:

```bash
svla process --limit 20            # only the first N runs
svla process --run 000007          # a single run (uses the existing RAG store)
svla process --no-index            # skip rebuilding the vector store
svla eval --out reports/metrics.json
```

---

## Enabling Claude

By default the root-cause agent uses a deterministic, rule-based fallback so the
pipeline is fully runnable offline. To use Claude:

```bash
cp .env.example .env
# then set ANTHROPIC_API_KEY in .env (and optionally SVLA_LLM_MODEL)
```

Configuration is read by `pydantic-settings` (prefix `SVLA_`). Key settings:

| Setting | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | _unset_ | Enables Claude; unset ⇒ rule-based hypotheses |
| `SVLA_LLM_MODEL` | `claude-opus-4-8` | Hypothesis model |
| `SVLA_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `SVLA_OFFLINE` | `0` | Force network-free, deterministic paths (used by CI) |
| `SVLA_CHROMA_DIR` | `.chroma` | Vector-store location |
| `SVLA_REPORTS_DIR` | `reports` | Where reports are written |

---

## Example report

Each failing run produces a report like:

```
# SVLA Debug Report — Run 000001
Status: 🔴 ANOMALOUS | Events analysed: 11,079

## Failures (1)
### 1. THERMAL_TRIP — HIGH  (SYSTEM/THERMAL)
   ... evidence (WARN context + ERROR/FATAL + register dumps) ...

## Root-Cause Hypotheses
- Root cause: Junction temperature exceeded the thermal trip threshold
- Confidence: 95% (rule-based)
- Recommended actions: inspect heatsink/TIM; harden fan/DVFS profile; ...
- Similar historical incidents: run 000002 (RC_FAN_PROFILE_BAD, sim 1.0), ...
```

---

## Dataset

The synthetic generator (`svla_agent.datagen.generator`) produces realistic
multi-source runs (`system`, `fw`, `pcie`, `ddr`, `pmu`, `thermal`, `noc`) with
dense telemetry noise and injected failure patterns:

`PCIE_TIMEOUT`, `DDR_ECC`, `PMU_DROOP`, `THERMAL_TRIP`, `FW_EXCEPTION`, `NOC_TIMEOUT`.

Ground-truth labels (`data/labels/run_*.label.json`) record the injected patterns
and root-cause ids, and drive both RAG seeding and evaluation. `data/` is
git-ignored — regenerate it with `svla generate`.

---

## Development

```bash
make dev-install     # pip install -e ".[dev,app]" + pre-commit
make check           # ruff + mypy + pytest
make test            # pytest only
```

The test suite runs fully offline and deterministically against a small generated
dataset. CI (GitHub Actions) lints, type-checks and tests on Python 3.10–3.12.
See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Tech stack

Python 3.10+ · pydantic / pydantic-settings · Drain3 · sentence-transformers ·
scikit-learn (HDBSCAN) · ChromaDB · Anthropic Claude · Jinja2 · Typer · Rich ·
Streamlit · pytest / ruff / mypy.

## License

MIT — see [LICENSE](LICENSE).

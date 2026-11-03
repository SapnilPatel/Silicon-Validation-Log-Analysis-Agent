# SVLA-Agent

[![CI](https://github.com/SapnilPatel/Silicon-Validation-Log-Analysis-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/SapnilPatel/Silicon-Validation-Log-Analysis-Agent/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/live%20demo-svla--agent.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://svla-agent.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange)](https://github.com/astral-sh/ruff)

**Silicon Validation Log Analysis Agent** — an agentic AI debug assistant for
post-silicon validation logs. SVLA-Agent ingests large multi-source silicon test
logs, parses them into structured events, classifies and de-duplicates failures,
clusters recurring failure families, retrieves similar historical incidents (RAG),
and names a specific, evidence-backed root cause with remediation steps — as
Markdown/JSON debug reports, a REST API and an interactive dashboard.

> **Live demo:** https://svla-agent.streamlit.app/ — browse the bundled runs, open a report,
> run a live analysis or upload your own logs. Hosting options (Streamlit Community Cloud,
> Docker, Hugging Face PRO) are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
>
> The whole pipeline runs **offline** out of the box (deterministic fallbacks for
> embeddings, retrieval and hypotheses). Add an `ANTHROPIC_API_KEY` — or paste one
> in the dashboard sidebar — to upgrade the hypothesis agent to Claude.

---

## What it does

```
raw logs ─▶ ingest ─▶ parse ─▶ classify ─▶ cluster ─▶ RAG ─▶ hypothesize ─▶ report
          (stream)  (Drain3   (rules +   (HDBSCAN   (Chroma /  (Claude /    (MD / JSON /
                    +redact)   dedup)     per type)  memory)    rule-based)  API / UI)
```

| Stage | Module | What happens |
|---|---|---|
| **Ingest** | `ingest/reader.py` | Discover & group per-source run files (`run_*.<src>.log[.gz]`), stream lines lazily |
| **Parse** | `parse/parser.py` | Regex line structure (ts / subsys / level / `k=v` / multi-line dumps) + Drain3 template mining |
| **Redact** | `privacy/redact.py` | Scrub e-mails, IPs, MACs, serials, credentials *before* anything is indexed or prompted |
| **Classify** | `classify/classifier.py` | Error/fatal events → prioritised `Failure` records with cross-subsystem evidence; fingerprint dedup |
| **Embed** | `embeddings/embedder.py` | sentence-transformers embeddings (deterministic hashing fallback offline) |
| **Cluster** | `cluster/clusterer.py` | HDBSCAN within each failure type → explainable failure families |
| **RAG** | `rag/` | Pluggable vector store (`chroma` persistent, `memory` numpy) over historical failures |
| **Hypothesize** | `hypothesize/` | Root-cause catalogue + evidence scoring; Claude tool-use picks among the same candidates |
| **Report** | `report/reporter.py` | Jinja2 Markdown + JSON debug reports |
| **Serve** | `api.py`, `app/streamlit_app.py` | FastAPI REST service and Streamlit dashboard |
| **Evaluate** | `eval/metrics.py` | Detection, typing, clustering, hold-out retrieval and root-cause accuracy |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow and schemas, and
[docs/VISION.md](docs/VISION.md) for how every original project goal was delivered.

---

## Quickstart

```bash
git clone https://github.com/SapnilPatel/Silicon-Validation-Log-Analysis-Agent.git
cd Silicon-Validation-Log-Analysis-Agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app,api]"

# Try it on the bundled 24-run sample dataset (no generation needed)
svla process --logs examples/sample_logs --labels examples/sample_labels
streamlit run app/streamlit_app.py

# Or the full flow: 100 synthetic runs -> reports -> metrics -> dashboard / API
svla generate --runs 100
svla process
svla eval --out reports/metrics.json
svla serve            # REST API on http://localhost:8000/docs
```

A `Makefile` wraps the common flows (`make dev-install data process eval app serve check`),
and `make docker-build && make docker-api` runs it in a container. See
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## CLI

```
svla generate   Generate the synthetic silicon-log dataset
svla process    Run the full analysis pipeline and write debug reports
svla index      (Re)build the RAG vector store of historical failures
svla eval       Evaluate against ground truth (hold-out retrieval + root cause)
svla serve      Serve the REST API (FastAPI + uvicorn)
svla version    Print the version
```

Useful flags:

```bash
svla process --limit 20              # only the first N runs
svla process --run 000007            # a single run (uses the existing RAG store)
svla process --no-index              # skip rebuilding the vector store
svla eval --holdout 0.3 --out reports/metrics.json
svla serve --port 8000
```

### REST API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Active backends (embeddings, store, agent) and index size |
| `GET /runs`, `GET /reports`, `GET /reports/{run_id}` | Browse runs and reports |
| `POST /analyze/{run_id}` | Analyse a run from the logs directory |
| `POST /analyze/upload` | Analyse uploaded `run_<id>.<source>.log[.gz]` files |
| `POST /index` · `GET /metrics` · `GET /clusters` · `GET /knowledge` | Index rebuild, eval metrics, families, root-cause catalogue |

Send `X-Anthropic-Key: <your key>` to have that one request use Claude.

---

## Enabling Claude

By default the root-cause agent uses a deterministic, rule-based fallback so the
pipeline is fully runnable offline. To use Claude:

```bash
cp .env.example .env
# then set ANTHROPIC_API_KEY in .env (and optionally SVLA_LLM_MODEL)
```

Both paths see the same evidence, the same root-cause candidates and the same
retrieved incidents, and return the same `Hypothesis` shape (including a
`root_cause_id` that evaluation can score). Configuration is read by
`pydantic-settings` (prefix `SVLA_`); the main knobs:

| Setting | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | _unset_ | Enables Claude; unset ⇒ rule-based hypotheses |
| `SVLA_LLM_MODEL` | `claude-opus-5-5` | Hypothesis model (`SVLA_LLM_EFFORT` sets thinking depth, default `medium`) |
| `SVLA_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `SVLA_VECTOR_BACKEND` | `chroma` | `chroma` (persistent) or `memory` (numpy, file-backed) |
| `SVLA_REDACT` | `1` | PII / secret redaction at parse time |
| `SVLA_OFFLINE` | `0` | Force network-free, deterministic paths (used by CI) |
| `SVLA_CHROMA_DIR` / `SVLA_REPORTS_DIR` | `.chroma` / `reports` | Store and report locations |

Full list in [.env.example](.env.example).

---

## Evaluation

`svla eval` scores the pipeline against the generator's ground truth. Retrieval
and root-cause metrics use a **hold-out split**: the RAG index is built from 70%
of the runs and the remaining 30% are analysed as *new* incidents, so a failure
can never retrieve itself.

Results on a 400-run synthetic corpus (166 failures, 52 held out), rule-based
agent, no API key:

| Metric | sentence-transformers | hashing (offline) |
|---|---|---|
| Anomaly detection P / R / F1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Failure typing (micro) P / R / F1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Cluster purity vs. failure type | 1.00 | 1.00 |
| Cluster homogeneity vs. root cause | 0.91 | 0.91 |
| Retrieval recall@5 (same type) / MRR | 1.00 / 1.00 | 1.00 / 1.00 |
| Retrieval recall@5 (same root cause) | 0.98 | 0.92 |
| **Root-cause top-1 accuracy** | **0.96** | **0.92** |
| Mean confidence when right / wrong | 0.92 / 0.77 | 0.92 / 0.77 |

What these numbers mean, honestly: the corpus is synthetic. Every injected
failure leaves a root-cause tell-tale in the logs **90% of the time**; the rest
exercises the retrieval-vote and prior fallbacks, which is where the misses come
from. Detection and typing are rule-based against the same log vocabulary the
generator produces, so they are expected to be perfect here and will not be on
real silicon logs — the value of the pipeline is the structure (evidence →
candidates → calibrated confidence → actions) that carries over. Reproduce with:

```bash
svla generate --runs 400 --out-logs data/eval/logs --out-labels data/eval/labels --seed 777000
svla eval --logs data/eval/logs --labels data/eval/labels --out reports/metrics.json
```

---

## Example report

```
# SVLA Debug Report — Run 000005
Status: 🔴 ANOMALOUS | Events analysed: 11,079

## Failures (1)
### 1. THERMAL_TRIP — HIGH  (SYSTEM/THERMAL)   · seen 3x in corpus
   ... distinct WARN context + ERROR/FATAL + register dumps ...

## Root-Cause Hypotheses
- Root cause: Junction temperature exceeded the thermal trip threshold:
  Fan speed stays below the profile target as temperature rises  (RC_FAN_PROFILE_BAD)
- Confidence: 92% (rule-based)
- Rationale: ... Evidence contains the tell-tale 'Fan speed below profile' ...
  2 of 3 similar historical incidents were resolved as RC_FAN_PROFILE_BAD.
- Recommended actions: fix the fan curve / PWM mapping; verify tachometer feedback; ...
- Similar historical incidents: run 000041 (RC_FAN_PROFILE_BAD, sim 0.97), ...
```

---

## Dataset

The synthetic generator (`svla_agent.datagen.generator`) produces realistic
multi-source runs (`system`, `fw`, `pcie`, `ddr`, `pmu`, `thermal`, `noc`) with
dense telemetry noise, injected failure patterns and per-root-cause evidence:

| Failure type | Root causes the agent can name |
|---|---|
| `PCIE_TIMEOUT` | refclk gating · SI margin · ASPM misconfig · EQ tuning |
| `DDR_ECC` | timing violation · training edge · VDDQ droop · CDC glitch |
| `PMU_DROOP` | load-step spike · insufficient decoupling · regulator limit |
| `THERMAL_TRIP` | poor heatsink · bad fan profile · workload burst |
| `FW_EXCEPTION` | race condition · null deref · stack overflow |
| `NOC_TIMEOUT` | QoS starvation · deadlock · DMA burst overflow |

Ground-truth labels (`*.label.json`) record the injected patterns and root-cause
ids, and drive both RAG seeding and evaluation. A 24-run gzip sample with
precomputed reports is committed under `examples/`; `data/` is git-ignored —
regenerate it with `svla generate`.

---

## Development

```bash
make dev-install     # pip install -e ".[dev,app,api]"
make check           # ruff + format check + mypy + pytest (offline)
make test            # pytest only
```

The test suite (45 tests) runs fully offline and deterministically against a
small generated dataset, including the REST API and headless dashboard tests.
CI (GitHub Actions) lints, type-checks and tests on Python 3.10–3.12, smokes the
bundled sample dataset end to end and builds the Docker image. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Tech stack

Python 3.10+ · pydantic / pydantic-settings · Drain3 · sentence-transformers ·
scikit-learn (HDBSCAN) · ChromaDB · Anthropic Claude · Jinja2 · Typer · Rich ·
FastAPI · Streamlit / Plotly · pytest / ruff / mypy · Docker · Hugging Face Spaces.

## License

MIT — see [LICENSE](LICENSE).

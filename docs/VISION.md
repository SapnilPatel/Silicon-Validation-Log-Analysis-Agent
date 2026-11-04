# Project vision → delivery status

The original project README (reproduced below) set out the goals for SVLA-Agent.
This page tracks each promise against what the repository actually ships.

| Vision item | Status | Where |
|---|---|---|
| Robust log parsing (regex + Drain3) | ✅ | `parse/parser.py` |
| Streaming ingest of files / directories / compressed archives | ✅ (`.log`, `.log.gz`, recursive dirs) | `ingest/reader.py` |
| Failure classification **and deduplication** | ✅ per-run typing + corpus fingerprint dedup | `classify/classifier.py` |
| Pattern clustering (HDBSCAN + scikit-learn) | ✅ within-type HDBSCAN families | `cluster/clusterer.py` |
| Root-cause hypotheses with LLMs | ✅ three modes: one call, a tool-using agent, and a **multi-agent team** (LangGraph); all leave traces; pluggable provider interface. OpenAI/Llama: **not shipped** (interface ready) | `hypothesize/agent.py`, `hypothesize/tools.py`, `hypothesize/multiagent.py` |
| RAG over historical failures (Chroma / Pinecone) | ✅ Chroma + in-memory backends behind one interface. Pinecone: **not shipped** (needs an account) | `rag/` |
| Markdown reports (Jinja2) + JSON | ✅ | `report/` |
| Streamlit demo | ✅ 5-tab dashboard incl. live analysis & evaluation | `app/streamlit_app.py` |
| FastAPI service | ✅ `svla serve` | `api.py` |
| LangGraph orchestration | ✅ used where it earns its place: the multi-agent investigation (triage → parallel specialists → reviewer) is a LangGraph `StateGraph`. The deterministic pipeline stays a plain orchestrator | `hypothesize/multiagent.py`, `pipeline.py` |
| "Agents" (parsing, classification, clustering, hypothesis, report) | ✅ the deterministic stages are modules, not agents (by design); the *hypothesis* step is a genuine multi-agent team in `multiagent` mode | `hypothesize/multiagent.py` |
| Synthetic dataset + generator | ✅ generator with per-root-cause evidence; 24-run sample committed | `datagen/generator.py`, `examples/` |
| Metrics: silhouette, Davies-Bouldin, recall@k, hypothesis accuracy | ✅ + purity/homogeneity, MRR, root-cause top-1 on a hold-out split | `eval/metrics.py` |
| PII / secret redaction before indexing | ✅ on by default | `privacy/redact.py` |
| Secrets via environment variables | ✅ `.env.example`, nothing committed | `config.py` |
| Docker demo | ✅ image builds in CI | `Dockerfile` |
| CI with dataset-based integration tests | ✅ lint, types, tests, sample-dataset smoke, Docker build | `.github/workflows/ci.yml` |
| Live deployment | ✅ Hugging Face Space | `scripts/deploy_hf_space.py` |

## Known limitations

- The dataset is synthetic. Numbers in the README measure the pipeline against
  the generator's ground truth, not against real silicon logs. Every root cause
  leaves a tell-tale line 90% of the time; the remaining 10% exercises the
  retrieval/prior fallback.
- CI runs offline and never calls Claude; the LLM path is covered by a fake
  provider in tests and was validated against the real API (`claude-opus-5-5`)
  on the 400-run hold-out set - see the README's evaluation table.

---

## Original README (for reference)

### SVLA-Agent (original)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#) [![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

A production-oriented Agentic AI Debug Assistant for post-silicon validation logs. SVLA-Agent parses massive silicon test logs, detects anomalies, clusters failure patterns, suggests root-cause hypotheses, links to historical failures (RAG), and generates clear debug reports and dashboards.

Why SVLA-Agent?
- Saves engineer time by surfacing actionable failure insights.
- Scales to millions of lines using streaming parsers and vector RAG.
- Combines classical parsing, ML clustering, and LLM-backed reasoning for explainable hypotheses.

Key Features
- Robust log parsing (regex, Drain3 for structured event extraction)
- Failure classification and deduplication
- Pattern clustering (HDBSCAN + scikit-learn)
- Root-cause hypothesis generation using LLMs (OpenAI / Llama)
- RAG: vectorized similarity search against historical failures (Chroma / Pinecone)
- Human-friendly reports (Markdown / Jinja2) and Streamlit demo

Architecture Overview
1. Ingest: streaming reader supports single files, directories, and compressed archives.
2. Parse: tokenization, timestamp normalization, multi-line trace grouping.
3. Enrich: extract metadata (test ids, IPs, counters), generate embeddings (sentence-transformers).
4. Cluster: unsupervised clustering to surface patterns and groups of similar failures.
5. Classify & Hypothesize: classifier + LLM agent suggest probable root causes and remediation steps.
6. RAG: retrieve similar past incidents from vector DB and include context in reports.
7. Report: JSON, Markdown, and interactive Streamlit dashboards.

Agents
- Log Parsing Agent — normalize and structure raw logs.
- Failure Classification Agent — tag and prioritize failures.
- Pattern Clustering Agent — find recurring failure families.
- Root Cause Hypothesis Agent — generate concise hypotheses with evidence.
- Report Generator Agent — compile a reproducible debug report.

Tech Stack (recommended)
- Python 3.9+
- LangGraph for agent orchestration
- sentence-transformers for embeddings
- Chroma or Pinecone for vector DB (RAG)
- HDBSCAN + scikit-learn for clustering
- Drain3, regex, Pydantic for parsing & schemas
- Streamlit / FastAPI for demo and APIs
- Jinja2 for templated reports

Quickstart (local)
1. Clone:
   git clone <repo-url> && cd SVLA-Agent
2. Create venv and install:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
3. Run CLI (example):
   python -m svla_agent.cli process --input ./logs --output ./reports/out.json
4. Run demo:
   streamlit run app/streamlit_app.py

Minimal Example
- Parse + cluster:
  python -m svla_agent.cli process --input samples/synthetic_logs --out results/summary.json --embeddings sentence-transformers/all-MiniLM-L6-v2

Dataset
- A synthetic silicon logs dataset is included in /data/synthetic for local experimentation. Use the generator script to produce controlled failure scenarios for testing clustering and RAG.

Evaluation & Metrics
- Cluster quality: silhouette, Davies-Bouldin
- Retrieval: recall@k on historical incidents
- Hypothesis accuracy: human-labeled precision/recall in curated test suites

Extending the Project
- Add parsers for new log formats in svla_agent.parsers
- Plug alternative vector DB backends in svla_agent.rag
- Implement additional LLMs or prompt templates for hypothesis agent
- Improve CI with dataset-based integration tests

Contributing
- Follow the repo's contribution guide and tests.
- Add reproducible unit tests for parsers and clustering changes.
- Open an issue or PR with a clear description and dataset example.

Security & Privacy
- Redact PII and on-chip secret tokens before indexing.
- Secure API keys using environment variables or secret stores (do not commit keys).

License
MIT — see LICENSE file.

Contact
For project collaboration, issues, or demo requests, open an issue or contact the maintainer in the repo.

Appendix: Useful Commands
- Run tests: pytest -q
- Build docker demo: docker build -t svla-agent . && docker run --rm -v $(pwd)/logs:/data svla-agent process /data
- Generate embeddings locally: python -m svla_agent.embeddings --model all-MiniLM-L6-v2 --input logs/ --out emb/db


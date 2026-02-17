# SVLA-Agent
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


---
title: SVLA-Agent
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.50.0
app_file: app/streamlit_app.py
python_version: "3.12"
pinned: false
license: mit
short_description: Agentic AI debug assistant for post-silicon validation logs
---

# SVLA-Agent — Silicon Validation Log Analysis Agent

Ingests multi-source silicon test logs, parses them (regex + Drain3), classifies
and de-duplicates failures, clusters recurring families (HDBSCAN), retrieves
similar historical incidents (RAG) and produces explainable root-cause hypotheses
with remediation steps — as Markdown/JSON reports and this dashboard.

- **Live analysis** tab: analyse a bundled run or upload your own
  `run_<id>.<source>.log[.gz]` files.
- Paste an Anthropic key in the sidebar (session-only) to switch the hypothesis
  agent from the deterministic rule-based fallback to Claude.
- PII / secrets are redacted before indexing or prompting.

Source, CLI, REST API and docs: https://github.com/SapnilPatel/Silicon-Validation-Log-Analysis-Agent

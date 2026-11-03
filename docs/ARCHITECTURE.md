# Architecture

SVLA-Agent is a staged pipeline. Each stage has a single responsibility, a typed
input/output, and is independently testable. Data flows through Pydantic models
defined in `svla_agent/models.py`.

## Data flow

```
RunFiles ─▶ ParsedRun ─▶ [Failure] ─▶ dedup + clustered [Failure] ─▶ RunReport
  reader      parser     classifier           clusterer                reporter
            (+redact)                     + RAG + hypothesize
```

## Stages

### 1. Ingest — `ingest/reader.py`
- `discover_runs(path)` groups `run_<id>.<source>.log[.gz]` files into `RunFiles`
  (one per run, many sources), recursively.
- Lines are streamed lazily (`iter_lines()` yields `(source, line)`), so a
  multi-GB run is never fully held in memory. Gzip is handled transparently.

### 2. Parse — `parse/parser.py` (+ `privacy/redact.py`)
- A regex recovers `ts_ms`, `subsys`, `level`, `message`, and `key=value` pairs.
- Indented lines (register dumps / stack traces) attach to the preceding event's
  `multiline`.
- **Drain3** mines a stable template per message (variables masked), giving each
  `Event` a low-cardinality `signature` used downstream for dedup/cluster/RAG.
- Header `# key: value` lines become a typed `RunMeta`.
- **Redaction** (on by default) replaces e-mails, IPs, MACs, home-dir user names,
  serial/fuse ids and credential-looking `key=value` pairs with typed placeholders
  *before* anything is indexed, prompted or reported.

### 3. Classify + dedup — `classify/classifier.py`
- Every `ERROR`/`FATAL` event is tagged with a coarse failure type via
  subsystem-aware keyword rules, with a subsystem fallback for bare dumps.
- Events of the same type within a run are grouped into one prioritised
  `Failure`. Evidence = the *distinct* non-telemetry WARN templates from any
  subsystem in the window before the first error (where root-cause tell-tales
  live), plus the errors and dump heads.
- The `signature` (type + error templates + context templates) is hashed into a
  `fingerprint`; `deduplicate()` counts recurrences across the corpus so a
  known failure shows "seen 7x" instead of being triaged from scratch.

### 4. Embed + Cluster — `embeddings/embedder.py`, `cluster/clusterer.py`
- `Embedder` encodes signatures with sentence-transformers, falling back to a
  deterministic hashing embedding when the model can't load or offline mode is
  set (`backend` reports which is active).
- `cluster_failures` runs `sklearn.cluster.HDBSCAN` **within each failure type**.
  The type is a hard, explainable boundary; HDBSCAN discovers the sub-families
  (typically root causes) inside it. Noise points fold into one fallback family
  per type so every failure ends up in exactly one cluster.

### 5. RAG — `rag/`
- `BaseVectorStore` is the backend-agnostic surface (`index/query/count/reset`).
  `ChromaVectorStore` (persistent, cosine) and `InMemoryVectorStore` (numpy,
  file-backed) ship; `make_store()` picks by `SVLA_VECTOR_BACKEND`.
- Indexing stores failure signatures with their ground-truth root cause when
  labels exist; `query()` returns the closest past incidents, excluding self.

### 6. Hypothesize — `hypothesize/`
- `knowledge.py` holds, per failure type, a prior (root cause, rationale,
  playbook) **and** a catalogue of specific root causes with the evidence
  markers that betray each one.
- `HypothesisAgent` scores the catalogue: evidence markers weigh most, then
  similarity-weighted votes from retrieved incidents, then the prior. The
  rule-based path is deterministic; the LLM path (`llm.py`, Claude via a forced
  `submit_hypothesis` tool call whose `root_cause_id` is an enum of the
  catalogue) sees the same evidence, candidates and incidents. Output shape is
  identical either way, including a `root_cause_id` that evaluation can score.

#### Agentic mode (`SVLA_LLM_MODE=agentic`, `--agentic`)
- `hypothesize/tools.py` exposes the run to the model as tools: `get_context`
  (lines in a time window, any subsystem, telemetry optional), `read_block`
  (the full register dump / stack trace nearest a timestamp), `lookup_catalogue`
  (root causes + tell-tales + playbook) and `search_incidents` (free-text query
  against the vector store).
- `AnthropicProvider.run_agent` runs a bounded tool loop (`agent_max_turns`,
  default 8): `tool_choice` auto, strict schemas, all tool results of a turn
  returned in one user message, a budget nudge before the last turn, and a final
  turn where only `submit_hypothesis` is offered. Refusals and loop failures
  fall back to the rule-based agent.
- Every call is recorded as a `ToolCall` on `Hypothesis.trace` and rendered in
  the Markdown report, the dashboard and the API response.

### 7. Report — `report/reporter.py`
- Renders each `RunReport` to Markdown (Jinja2 template
  `report/templates/run_report.md.j2`) and JSON.

## Orchestration — `pipeline.py`
- `AnalysisPipeline.run()` does the whole corpus: parse → classify → dedup →
  cluster → index → hypothesise → write reports, with progress callbacks.
- `process_run()` analyses a single run (or uploaded files) against an
  already-populated store.
- `index_corpus()` (re)builds the RAG store only.

## Interfaces — `api.py`, `cli.py`, `app/streamlit_app.py`
- CLI: `svla generate | process | index | eval | serve | version`.
- REST (FastAPI): health, runs, reports, analyze (by id or upload), index,
  metrics, clusters, knowledge. `X-Anthropic-Key` switches a request to Claude.
- Dashboard (Streamlit): overview, run detail with timeline, live analysis
  (pick a run or upload), evaluation, about. Visitors may paste their own key.

## Evaluation — `eval/metrics.py`
- **Detection**: run-level anomaly precision/recall/F1/accuracy.
- **Failure typing**: multi-label precision/recall/F1 (micro + per type).
- **Clustering**: silhouette, Davies-Bouldin, homogeneity, completeness, purity
  vs ground-truth type.
- **Hold-out** (default 30% of runs): the store is built from the training
  runs only; held-out failures are retrieved and hypothesised as *new* incidents.
  Reports recall@k and MRR (same type), recall@k (same root cause), root-cause
  top-1 accuracy with per-type breakdown and confidence calibration.

## Configuration — `config.py`
Single `Settings` object (pydantic-settings, `SVLA_` prefix) is threaded through
every component; `get_settings()` returns a cached instance and honours
`SVLA_OFFLINE`. See `.env.example` for every knob.

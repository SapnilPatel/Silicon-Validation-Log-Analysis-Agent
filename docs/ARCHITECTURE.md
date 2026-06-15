# Architecture

SVLA-Agent is a staged pipeline. Each stage has a single responsibility, a typed
input/output, and is independently testable. Data flows through Pydantic models
defined in `svla_agent/models.py`.

## Data flow

```
RunFiles ─▶ ParsedRun ─▶ [Failure] ─▶ clustered [Failure] ─▶ RunReport
  reader      parser     classifier     clusterer + RAG        reporter
                                          + hypothesize
```

## Stages

### 1. Ingest — `ingest/reader.py`
- `discover_runs(path)` groups `run_<id>.<source>.log[.gz]` files into `RunFiles`
  (one per run, many sources).
- Lines are streamed lazily (`iter_lines()` yields `(source, line)`), so a
  multi-GB run is never fully held in memory. Gzip is handled transparently.

### 2. Parse — `parse/parser.py`
- A regex recovers `ts_ms`, `subsys`, `level`, `message`, and `key=value` pairs.
- Indented lines (register dumps / stack traces) attach to the preceding event's
  `multiline`.
- **Drain3** mines a stable template per message (variables masked), giving each
  `Event` a low-cardinality `signature` used downstream for dedup/cluster/RAG.
- Header `# key: value` lines become a typed `RunMeta`.

### 3. Classify — `classify/classifier.py`
- Every `ERROR`/`FATAL` event is tagged with a coarse failure type via
  subsystem-aware keyword rules, with a subsystem fallback for bare dumps.
- Events of the same type within a run are grouped into one prioritised
  `Failure` (severity, time window, evidence = WARN context + errors + dump head,
  normalised `signature`).

### 4. Embed + Cluster — `embeddings/embedder.py`, `cluster/clusterer.py`
- `Embedder` encodes failure signatures with sentence-transformers, falling back
  to a deterministic hashing embedding when the model can't load or offline mode
  is set (`backend` reports which is active).
- `cluster_failures` runs `sklearn.cluster.HDBSCAN`; noise points are folded into
  per-type fallback families so every failure ends up in exactly one cluster.

### 5. RAG — `rag/store.py`
- A persistent **Chroma** collection (cosine) over failure signatures, sharing the
  `Embedder`'s vector space. `index()` upserts failures (with ground-truth root
  causes when labels exist); `query()` returns the most similar past incidents,
  excluding the query's own id.

### 6. Hypothesize — `hypothesize/agent.py` (+ `knowledge.py`)
- For each failure, retrieves similar incidents, then asks **Claude** (via a
  `submit_hypothesis` tool for structured output) for a root cause, confidence,
  rationale and actions.
- When no API key is configured (or offline), a rule-based fallback fuses the
  per-type domain prior in `knowledge.py` with agreeing retrieved incidents to set
  confidence — so output shape is identical regardless of backend.

### 7. Report — `report/reporter.py`
- Renders each `RunReport` to Markdown (Jinja2 template
  `report/templates/run_report.md.j2`) and JSON.

## Orchestration — `pipeline.py`
- `AnalysisPipeline.run()` does the whole corpus: parse → classify → cluster →
  index → hypothesise → write reports, with progress callbacks.
- `process_run()` analyses a single run against an already-populated store.
- `index_corpus()` (re)builds the RAG store only.

## Evaluation — `eval/metrics.py`
- **Detection**: run-level anomaly precision/recall/F1/accuracy.
- **Failure typing**: multi-label precision/recall/F1 (micro + per type).
- **Clustering**: silhouette, Davies-Bouldin, homogeneity vs ground-truth type.
- **Retrieval**: RAG recall@k (does top-k contain a same-type incident).

## Configuration — `config.py`
Single `Settings` object (pydantic-settings, `SVLA_` prefix) is threaded through
every component; `get_settings()` returns a cached instance and honours
`SVLA_OFFLINE`.

# Contributing to SVLA-Agent

Thanks for contributing! This project favours small, well-tested changes and keeps
the whole pipeline runnable offline.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app,api]"
pre-commit install
```

(Or use the bundled `.svla/` venv and the `Makefile` targets.)

## Workflow

1. Branch off `main`.
2. Make your change with a test.
3. Run the full gate locally:

   ```bash
   make check        # ruff check + ruff format --check (via mypy/test), mypy, pytest
   # or individually:
   ruff check . && ruff format . && mypy && pytest
   ```

4. Commit (pre-commit will run ruff + mypy on staged files) and open a PR.

CI runs the same checks on Python 3.10–3.12.

## Conventions

- **Style/lint**: `ruff` (format + lint). Line length 100.
- **Types**: `mypy` over `src/svla_agent`. Prefer explicit, typed function
  signatures on public functions.
- **Tests**: `pytest`. Tests must be deterministic and offline — use the
  `dataset`, `settings`, `embedder`, and `pipeline` fixtures in
  `tests/conftest.py` (they force `offline=True` and a tmp Chroma/reports dir).
- **Models first**: new data crossing a stage boundary should be a Pydantic model
  in `svla_agent/models.py`.
- **Offline-safe**: any new network/LLM/model-download path must degrade
  gracefully when `settings.offline` is set or credentials are missing.

## Adding capabilities

- **New log format** → add parsing in `svla_agent/parse/` (and a reader pattern if
  the filename layout differs).
- **New failure type** → extend the rules in `classify/classifier.py` and the
  prior in `hypothesize/knowledge.py`.
- **New vector backend** → subclass `BaseVectorStore` in `svla_agent/rag/` and
  register it in `rag/store.py` (`_BACKENDS`).
- **New LLM provider** → implement `LLMProvider.structured()` in
  `hypothesize/llm.py` and register it in `PROVIDERS`.
- **New root cause** → add it to `ROOT_CAUSES` in `hypothesize/knowledge.py`
  *and* teach the generator to emit its marker (a test enforces the two agree).
- **New metric** → add to `eval/metrics.py` and surface it in `cli.py`.

## Commit messages

Imperative summary line, then a short body explaining the *why*. Keep unrelated
changes in separate commits.

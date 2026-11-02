"""FastAPI service exposing the analysis pipeline.

Run with ``svla serve`` (or ``uvicorn svla_agent.api:app``). Endpoints:

* ``GET  /health``            - backends in use and index size
* ``GET  /runs``              - runs discoverable under the logs directory
* ``GET  /reports``           - summaries of already-written reports
* ``GET  /reports/{run_id}``  - one full report
* ``POST /analyze/{run_id}``  - analyse a run from the logs directory
* ``POST /analyze/upload``    - analyse uploaded ``run_<id>.<source>.log[.gz]`` files
* ``POST /index``             - (re)build the historical-failure vector store
* ``GET  /metrics``           - latest evaluation metrics, if any
* ``GET  /clusters``          - failure families seen across the reports
* ``GET  /knowledge``         - root-cause catalogue used by the agent

A caller may supply an ``X-Anthropic-Key`` header to have *their* key used for
Claude on that request only; otherwise the deterministic rule-based agent runs.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel

from . import __version__
from .config import Settings, get_settings
from .hypothesize.agent import HypothesisAgent
from .hypothesize.knowledge import ROOT_CAUSES
from .ingest.reader import discover_runs
from .models import RunReport
from .pipeline import AnalysisPipeline

app = FastAPI(
    title="SVLA-Agent API",
    version=__version__,
    description="Agentic debug assistant for post-silicon validation logs.",
)


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_pipeline() -> AnalysisPipeline:
    return AnalysisPipeline(get_settings())


def _pipeline_for(key: str | None) -> AnalysisPipeline:
    """Shared pipeline, or a per-request clone whose agent uses the caller's key."""
    base = get_pipeline()
    if not key:
        return base
    settings: Settings = base.settings.model_copy(
        update={"anthropic_api_key": key, "offline": False}
    )
    return AnalysisPipeline(
        settings=settings,
        embedder=base.embedder,
        store=base.store,
        agent=HypothesisAgent(settings),
        reporter=base.reporter,
    )


def _reports_dir() -> Path:
    return get_pipeline().settings.reports_dir


def _load_report(run_id: str) -> dict[str, Any]:
    path = _reports_dir() / f"run_{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No report for run {run_id}")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class Health(BaseModel):
    status: str
    version: str
    hypothesis_backend: str
    vector_backend: str
    embedding_backend: str
    indexed_failures: int
    redaction: bool


class RunSummary(BaseModel):
    run_id: str
    sources: list[str]


class ReportSummary(BaseModel):
    run_id: str
    anomalous: bool
    n_events: int
    failure_types: list[str]
    root_cause_ids: list[str | None]
    max_confidence: float | None


class IndexResult(BaseModel):
    indexed: int


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=Health)
def health() -> Health:
    p = get_pipeline()
    return Health(
        status="ok",
        version=__version__,
        hypothesis_backend=p.agent.backend,
        vector_backend=p.store.name,
        embedding_backend=p.embedder.backend,
        indexed_failures=p.store.count(),
        redaction=p.settings.redact,
    )


@app.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = Query(200, ge=1, le=5000)) -> list[RunSummary]:
    p = get_pipeline()
    runs = discover_runs(p.settings.logs_dir)[:limit]
    return [RunSummary(run_id=r.run_id, sources=sorted(f.source for f in r.sources)) for r in runs]


@app.get("/reports", response_model=list[ReportSummary])
def list_reports() -> list[ReportSummary]:
    out: list[ReportSummary] = []
    for path in sorted(_reports_dir().glob("run_*.json")):
        try:
            r = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        confs = [h.get("confidence") for h in r.get("hypotheses", [])]
        out.append(
            ReportSummary(
                run_id=r["run_id"],
                anomalous=r["anomalous"],
                n_events=r["n_events"],
                failure_types=[f["failure_type"] for f in r.get("failures", [])],
                root_cause_ids=[h.get("root_cause_id") for h in r.get("hypotheses", [])],
                max_confidence=max(confs) if confs else None,
            )
        )
    return out


@app.get("/reports/{run_id}", response_model=RunReport)
def get_report(run_id: str) -> Any:
    return _load_report(run_id)


# NB: declared before ``/analyze/{run_id}`` so "upload" is not captured as a run id.
@app.post("/analyze/upload", response_model=list[RunReport])
async def analyze_upload(
    files: list[UploadFile] = File(..., description="run_<id>.<source>.log[.gz] files"),
    write: bool = Query(False),
    x_anthropic_key: str | None = Header(default=None, alias="X-Anthropic-Key"),
) -> list[RunReport]:
    p = _pipeline_for(x_anthropic_key)
    tmp = Path(tempfile.mkdtemp(prefix="svla_upload_"))
    try:
        for up in files:
            name = Path(up.filename or "").name
            if not name.startswith("run_") or ".log" not in name:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{name}' is not a run_<id>.<source>.log[.gz] file",
                )
            (tmp / name).write_bytes(await up.read())
        runs = discover_runs(tmp)
        if not runs:
            raise HTTPException(status_code=400, detail="No runs found in upload")
        return [p.process_run(rf, write=write) for rf in runs]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/analyze/{run_id}", response_model=RunReport)
def analyze_run(
    run_id: str,
    write: bool = Query(False, description="Also persist the report to the reports dir."),
    x_anthropic_key: str | None = Header(default=None, alias="X-Anthropic-Key"),
) -> RunReport:
    p = _pipeline_for(x_anthropic_key)
    try:
        return p.process_run(run_id, write=write)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/index", response_model=IndexResult)
def rebuild_index() -> IndexResult:
    return IndexResult(indexed=get_pipeline().index_corpus())


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    for name in ("metrics.json", "metrics_st.json", "metrics_offline.json"):
        path = _reports_dir() / name
        if path.exists():
            return json.loads(path.read_text())
    raise HTTPException(status_code=404, detail="No metrics file; run `svla eval --out ...`")


@app.get("/clusters")
def clusters() -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for path in sorted(_reports_dir().glob("run_*.json")):
        try:
            r = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        for c in r.get("clusters", []):
            seen.setdefault(c["cluster_id"], c)
    return sorted(seen.values(), key=lambda c: (-c["size"], c["cluster_id"]))


@app.get("/knowledge")
def knowledge() -> dict[str, list[dict[str, Any]]]:
    return {
        ftype: [
            {
                "rc_id": rc.rc_id,
                "description": rc.description,
                "markers": list(rc.markers),
                "actions": list(rc.actions),
            }
            for rc in causes
        ]
        for ftype, causes in ROOT_CAUSES.items()
    }

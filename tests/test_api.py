"""REST API tests (offline, in-process via TestClient)."""

from __future__ import annotations

import gzip
import io

import pytest
from fastapi.testclient import TestClient

from svla_agent import api as api_module
from svla_agent.pipeline import AnalysisPipeline


@pytest.fixture()
def client(pipeline: AnalysisPipeline, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Point the API at the offline test pipeline, and write reports for it to list.
    api_module.get_pipeline.cache_clear()
    monkeypatch.setattr(api_module, "get_pipeline", lambda: pipeline)
    pipeline.run(index=True, write=True)
    return TestClient(api_module.app)


def test_health(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["hypothesis_backend"] == "rule-based"
    assert body["vector_backend"] == "chroma"
    assert body["indexed_failures"] > 0
    assert body["redaction"] is True


def test_runs_and_reports(client: TestClient):
    runs = client.get("/runs").json()
    assert runs and set(runs[0]) == {"run_id", "sources"}
    reports = client.get("/reports").json()
    assert len(reports) == len(runs)
    anomalous = next(r for r in reports if r["anomalous"])
    assert anomalous["failure_types"] and anomalous["root_cause_ids"]
    full = client.get(f"/reports/{anomalous['run_id']}").json()
    assert full["run_id"] == anomalous["run_id"] and full["hypotheses"]
    assert client.get("/reports/999999").status_code == 404


def test_analyze_existing_run(client: TestClient):
    anomalous = next(r for r in client.get("/reports").json() if r["anomalous"])
    res = client.post(f"/analyze/{anomalous['run_id']}")
    assert res.status_code == 200
    body = res.json()
    assert body["anomalous"] and body["hypotheses"][0]["root_cause_id"]
    assert client.post("/analyze/does-not-exist").status_code == 404


def test_analyze_upload_gzip(client: TestClient, pipeline: AnalysisPipeline, dataset):
    anomalous = next(r for r in client.get("/reports").json() if r["anomalous"])
    files = []
    for path in sorted(dataset["logs"].glob(f"run_{anomalous['run_id']}.*.log")):
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(path.read_bytes())
        files.append(("files", (path.name + ".gz", buf.getvalue(), "application/gzip")))
    res = client.post("/analyze/upload", files=files)
    assert res.status_code == 200, res.text
    reports = res.json()
    assert len(reports) == 1 and reports[0]["anomalous"]
    assert {f["failure_type"] for f in reports[0]["failures"]} == set(anomalous["failure_types"])

    bad = client.post("/analyze/upload", files=[("files", ("notes.txt", b"x", "text/plain"))])
    assert bad.status_code == 400


def test_index_metrics_clusters_knowledge(client: TestClient, pipeline: AnalysisPipeline):
    assert client.post("/index").json()["indexed"] == pipeline.store.count()
    assert client.get("/metrics").status_code == 404
    (pipeline.settings.reports_dir / "metrics.json").write_text('{"n_runs": 1}')
    assert client.get("/metrics").json() == {"n_runs": 1}
    clusters = client.get("/clusters").json()
    assert clusters and clusters[0]["size"] >= clusters[-1]["size"]
    knowledge = client.get("/knowledge").json()
    assert "PCIE_TIMEOUT" in knowledge and knowledge["PCIE_TIMEOUT"][0]["rc_id"].startswith("RC_")

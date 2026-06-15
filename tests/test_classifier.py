"""Tests for failure classification against ground-truth labels."""

from __future__ import annotations

from svla_agent.classify.classifier import classify_event, classify_run
from svla_agent.ingest.reader import discover_runs
from svla_agent.labels import load_labels
from svla_agent.models import Event, LogLevel, Severity
from svla_agent.parse.parser import parse_run


def _event(subsys: str, level: str, message: str, **kv) -> Event:
    return Event(
        ts_ms=0, source=subsys.lower(), subsys=subsys, level=LogLevel(level), message=message, kv=kv
    )


def test_classify_event_keyword_rules():
    assert classify_event(_event("PCIE", "ERROR", "Link training timeout")) == "PCIE_TIMEOUT"
    assert classify_event(_event("DDR", "ERROR", "ECC uncorrectable error")) == "DDR_ECC"
    assert (
        classify_event(_event("SYSTEM", "FATAL", "Emergency shutdown", reason="THERMAL_TRIP"))
        == "THERMAL_TRIP"
    )


def test_classify_event_subsystem_fallback():
    # A bare register snapshot has no keyword but a clear subsystem.
    assert classify_event(_event("NOC", "ERROR", "NoC error register snapshot")) == "NOC_TIMEOUT"


def test_classify_event_ignores_info_and_warn():
    assert classify_event(_event("PCIE", "INFO", "Link up")) is None
    assert classify_event(_event("THERMAL", "WARN", "Temperature high")) is None


def test_classification_matches_labels(dataset):
    labels = load_labels(dataset["labels"])
    mismatches = 0
    for run in discover_runs(dataset["logs"]):
        parsed = parse_run(run)
        pred = sorted({f.failure_type for f in classify_run(parsed)})
        truth = sorted(labels[parsed.run_id]["failure_types"])
        if pred != truth:
            mismatches += 1
    assert mismatches == 0


def test_failures_have_severity_and_evidence(dataset):
    for run in discover_runs(dataset["logs"]):
        for failure in classify_run(parse_run(run)):
            assert failure.severity in set(Severity)
            assert failure.evidence
            assert failure.signature.startswith(failure.failure_type)
            assert failure.last_ts_ms >= failure.first_ts_ms

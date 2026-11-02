"""End-to-end pipeline, reporting, and evaluation-metric tests."""

from __future__ import annotations

import json

from svla_agent.eval.metrics import evaluate
from svla_agent.labels import load_labels
from svla_agent.report.reporter import render_markdown


def test_pipeline_runs_and_writes_reports(pipeline, settings, dataset):
    result = pipeline.run(write=True)
    assert len(result.reports) == 14
    assert result.failures
    assert result.clusters

    # One JSON + one MD per run.
    md_files = list(settings.reports_dir.glob("run_*.md"))
    json_files = list(settings.reports_dir.glob("run_*.json"))
    assert len(md_files) == len(json_files) == 14

    # Anomalous reports carry failures and hypotheses.
    for report in result.anomalous_runs:
        assert report.failures
        assert report.hypotheses
        assert all(h.failure_id for h in report.hypotheses)


def test_process_single_run_after_index(pipeline, dataset):
    pipeline.index_corpus()
    # Pick a known anomalous run from labels.
    labels = load_labels(dataset["labels"])
    anomalous_id = next(rid for rid, d in labels.items() if d["anomalous"])
    report = pipeline.process_run(anomalous_id, write=False)
    assert report.anomalous
    assert report.hypotheses
    assert report.hypotheses[0].similar_incidents  # store was populated


def test_render_markdown_contains_key_sections(pipeline):
    result = pipeline.run(write=False)
    report = next(r for r in result.anomalous_runs)
    md = render_markdown(report)
    assert "SVLA Debug Report" in md
    assert "Root-Cause Hypotheses" in md
    assert report.run_id in md


def test_evaluate_metrics_are_perfect_offline(pipeline, settings, dataset):
    result = pipeline.run(write=False, hypotheses=False)
    metrics = evaluate(
        result, load_labels(dataset["labels"]), embedder=pipeline.embedder, store=pipeline.store
    )
    assert metrics["detection"]["f1"] == 1.0
    assert metrics["failure_typing"]["micro"]["f1"] == 1.0
    assert metrics["clustering"]["homogeneity_vs_failure_type"] == 1.0
    # Metrics are JSON-serialisable for the `--out` flag.
    json.dumps(metrics)

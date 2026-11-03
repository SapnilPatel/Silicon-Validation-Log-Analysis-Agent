"""Smoke-test the Streamlit dashboard headlessly with Streamlit's AppTest."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from svla_agent.config import get_settings

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture()
def app_env(pipeline, dataset, tmp_path, monkeypatch):
    """Point the app at the offline test dataset; write reports for it to browse."""
    pipeline.run(index=True, write=True)
    monkeypatch.setenv("SVLA_OFFLINE", "1")
    # The app defaults this itself; setting it via monkeypatch guarantees the
    # process environment is restored afterwards so later tests are unaffected.
    monkeypatch.setenv("SVLA_VECTOR_BACKEND", "memory")
    monkeypatch.setenv("SVLA_CHROMA_DIR", str(tmp_path / "app_chroma"))
    monkeypatch.setenv("SVLA_LOGS_DIR", str(dataset["logs"]))
    monkeypatch.setenv("SVLA_LABELS_DIR", str(dataset["labels"]))
    monkeypatch.setenv("SVLA_REPORTS_DIR", str(pipeline.settings.reports_dir))
    get_settings.cache_clear()
    yield pipeline.settings
    get_settings.cache_clear()


def _run(at: AppTest) -> AppTest:
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_dashboard_renders_reports_and_metrics(app_env):
    at = _run(AppTest.from_file(str(APP), default_timeout=120))
    assert at.title[0].value.startswith("🔬 SVLA-Agent")
    assert len(at.tabs) == 5
    # Overview metrics + failure-type chart rendered.
    labels = [m.label for m in at.metric]
    assert "Runs analysed" in labels and "Anomalous runs" in labels
    # Run detail defaulted to an anomalous run -> expanders with evidence exist.
    assert any("—" in e.label for e in at.expander)


def test_dashboard_live_analysis_button(app_env):
    at = _run(AppTest.from_file(str(APP), default_timeout=180))
    # Analyse the *same* anomalous run that the Run detail tab is already showing:
    # both tabs then render identical charts/tables, which must not collide on ids.
    detail = next(sb for sb in at.selectbox if sb.label == "Select run")
    live = next(sb for sb in at.selectbox if sb.label == "Run")
    live.set_value(detail.value)
    button = next(b for b in at.button if b.label == "Analyse run")
    button.click()
    _run(at)
    run_headers = [s.value for s in at.subheader if s.value.startswith(f"Run {detail.value}")]
    assert len(run_headers) == 2, run_headers  # detail tab + live tab
    assert not at.error


def test_dashboard_falls_back_to_bundled_examples(app_env, tmp_path):
    """An empty reports dir falls back to the committed examples/ reports."""
    os.environ["SVLA_REPORTS_DIR"] = str(tmp_path / "empty_reports")
    os.environ["SVLA_LOGS_DIR"] = str(tmp_path / "empty_logs")
    get_settings.cache_clear()
    at = _run(AppTest.from_file(str(APP), default_timeout=120))
    assert at.sidebar.text_input[0].value.endswith("examples/reports")
    assert at.sidebar.text_input[1].value.endswith("examples/sample_logs")
    assert any(m.label == "Runs analysed" for m in at.metric)

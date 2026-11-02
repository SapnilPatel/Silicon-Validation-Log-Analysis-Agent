"""Shared fixtures: a small synthetic dataset and an offline Settings object."""

from __future__ import annotations

from pathlib import Path

import pytest

from svla_agent.config import Settings
from svla_agent.datagen.generator import generate
from svla_agent.embeddings.embedder import Embedder
from svla_agent.hypothesize.agent import HypothesisAgent
from svla_agent.pipeline import AnalysisPipeline
from svla_agent.rag.store import VectorStore
from svla_agent.report.reporter import Reporter


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate a small, fast, deterministic dataset once per test session."""
    root = tmp_path_factory.mktemp("svla_data")
    logs = root / "silicon_logs"
    labels = root / "labels"
    generate(
        runs=14,
        out_logs=str(logs),
        out_labels=str(labels),
        base_seed=1234,
        anomaly_rate=0.6,
        noise_lines=(200, 400),
    )
    return {"root": root, "logs": logs, "labels": labels}


@pytest.fixture()
def settings(dataset: dict[str, Path], tmp_path: Path) -> Settings:
    """Offline settings pointing at the session dataset + per-test chroma/reports."""
    return Settings(
        offline=True,
        anthropic_api_key=None,
        logs_dir=dataset["logs"],
        labels_dir=dataset["labels"],
        reports_dir=tmp_path / "reports",
        chroma_dir=tmp_path / "chroma",
        cluster_min_size=2,
    )


@pytest.fixture()
def embedder() -> Embedder:
    return Embedder(offline=True)


@pytest.fixture()
def pipeline(settings: Settings, embedder: Embedder) -> AnalysisPipeline:
    """A fully offline pipeline wired to per-test settings."""
    store = VectorStore(settings, embedder)
    return AnalysisPipeline(
        settings=settings,
        embedder=embedder,
        store=store,
        agent=HypothesisAgent(settings),
        reporter=Reporter(settings),
    )

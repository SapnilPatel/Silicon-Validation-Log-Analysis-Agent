"""Tests for the offline embedder and HDBSCAN clustering."""

from __future__ import annotations

import numpy as np

from svla_agent.classify.classifier import classify_runs
from svla_agent.cluster.clusterer import cluster_failures
from svla_agent.ingest.reader import discover_runs
from svla_agent.parse.parser import parse_run


def test_embedder_offline_is_deterministic_and_normalised(embedder):
    texts = ["PCIE_TIMEOUT link training timeout", "DDR_ECC uncorrectable error"]
    a = embedder.encode(texts)
    b = embedder.encode(texts)
    assert a.shape == (2, embedder.dim)
    np.testing.assert_allclose(a, b)
    norms = np.linalg.norm(a, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    assert embedder.backend == "hashing"


def test_encode_empty_returns_well_shaped(embedder):
    out = embedder.encode([])
    assert out.shape == (0, embedder.dim)


def test_cluster_assigns_every_failure(dataset, embedder):
    failures = classify_runs(parse_run(r) for r in discover_runs(dataset["logs"]))
    failures, clusters = cluster_failures(failures, embedder)
    assert failures
    # Every failure gets a cluster id, and every cluster id is represented.
    assert all(f.cluster_id is not None for f in failures)
    assert {f.cluster_id for f in failures} == {c.cluster_id for c in clusters}
    assert sum(c.size for c in clusters) == len(failures)


def test_clusters_are_type_homogeneous(dataset, embedder):
    """With deterministic signatures, clusters should not mix failure types."""
    failures = classify_runs(parse_run(r) for r in discover_runs(dataset["logs"]))
    _, clusters = cluster_failures(failures, embedder)
    for c in clusters:
        assert len(c.failure_types) == 1

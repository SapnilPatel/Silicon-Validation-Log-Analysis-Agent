"""Tests for the RAG vector store and the rule-based hypothesis agent."""

from __future__ import annotations

from svla_agent.classify.classifier import classify_runs
from svla_agent.hypothesize.agent import HypothesisAgent
from svla_agent.ingest.reader import discover_runs
from svla_agent.labels import load_labels, root_cause_map
from svla_agent.parse.parser import parse_run
from svla_agent.rag.store import VectorStore


def _all_failures(dataset):
    return classify_runs(parse_run(r) for r in discover_runs(dataset["logs"]))


def test_store_index_and_self_excluding_query(settings, embedder, dataset):
    store = VectorStore(settings, embedder)
    failures = _all_failures(dataset)
    rc = root_cause_map(load_labels(dataset["labels"]))
    n = store.index(failures, rc)
    assert n == len(failures) == store.count()

    target = failures[0]
    hits = store.query(target.signature, top_k=3, exclude_failure_id=target.failure_id)
    assert target.failure_id not in {h.failure_id for h in hits}
    # Closest incidents should share the failure type.
    assert hits and hits[0].failure_type == target.failure_type


def test_query_empty_store_returns_empty(settings, embedder):
    store = VectorStore(settings, embedder)
    assert store.query("anything") == []


def test_rule_based_hypothesis_is_grounded(settings, embedder, dataset):
    store = VectorStore(settings, embedder)
    failures = _all_failures(dataset)
    store.index(failures, root_cause_map(load_labels(dataset["labels"])))

    agent = HypothesisAgent(settings)  # offline -> rule-based
    target = next(f for f in failures if f.failure_type == "THERMAL_TRIP")
    similar = store.query(target.signature, exclude_failure_id=target.failure_id)
    hyp = agent.generate(target, similar, target.cluster_id)

    assert hyp.source == "rule-based"
    assert 0.0 <= hyp.confidence <= 1.0
    assert hyp.recommended_actions
    assert hyp.failure_id == target.failure_id
    # Agreement with retrieved same-type incidents should lift confidence.
    assert hyp.confidence >= 0.55

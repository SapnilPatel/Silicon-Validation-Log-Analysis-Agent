"""Pluggable vector stores, root-cause selection and hold-out evaluation."""

from __future__ import annotations

import pytest

from svla_agent.config import Settings
from svla_agent.eval.metrics import evaluate, split_runs
from svla_agent.hypothesize.agent import HypothesisAgent
from svla_agent.hypothesize.knowledge import ROOT_CAUSES, candidates_for
from svla_agent.labels import load_labels, root_cause_map
from svla_agent.models import Failure, RetrievedIncident, Severity
from svla_agent.rag.memory_store import InMemoryVectorStore
from svla_agent.rag.store import available_backends, make_store


def _failure(ftype: str, evidence: list[str], run_id: str = "000001") -> Failure:
    return Failure(
        failure_id=f"{run_id}:{ftype}",
        run_id=run_id,
        failure_type=ftype,
        severity=Severity.HIGH,
        subsys=ftype.split("_")[0],
        first_ts_ms=0,
        last_ts_ms=10,
        summary=f"{ftype} summary",
        signature=f"{ftype} | " + " ; ".join(evidence),
        evidence=evidence,
    )


# -- stores ----------------------------------------------------------------- #


def test_backends_registered():
    assert available_backends() == ["chroma", "memory"]


def test_make_store_rejects_unknown_backend(settings: Settings):
    bad = settings.model_copy(update={"vector_backend": "pinecone"})
    with pytest.raises(ValueError):
        make_store(bad)


def test_memory_store_roundtrip_and_persistence(settings: Settings, embedder):
    mem = settings.model_copy(update={"vector_backend": "memory"})
    store = make_store(mem, embedder)
    assert isinstance(store, InMemoryVectorStore)
    a = _failure("PCIE_TIMEOUT", ["PCIE: Link training timeout"], "1")
    b = _failure("DDR_ECC", ["DDR: ECC uncorrectable error"], "2")
    assert store.index([a, b], {a.failure_id: "RC_PCIE_EQ_TUNING"}) == 2
    assert store.index([a]) == 2 or store.count() == 2  # upsert keeps count stable
    hits = store.query(a.signature, top_k=2, exclude_failure_id=a.failure_id)
    assert [h.failure_id for h in hits] == [b.failure_id]
    hits = store.query(a.signature, top_k=1)
    assert hits[0].failure_id == a.failure_id and hits[0].distance < 1e-5

    reloaded = InMemoryVectorStore(mem, embedder)  # loads from disk
    assert reloaded.count() == 2
    reloaded.reset()
    assert reloaded.count() == 0 and InMemoryVectorStore(mem, embedder).count() == 0


# -- root-cause selection --------------------------------------------------- #


def test_knowledge_ids_match_generator_labels():
    import inspect

    from svla_agent.datagen import generator as g

    src = inspect.getsource(g)
    for ftype, causes in ROOT_CAUSES.items():
        for rc in causes:
            assert rc.rc_id in src, f"{rc.rc_id} ({ftype}) not produced by the generator"


def test_rule_based_picks_candidate_from_evidence(settings: Settings):
    agent = HypothesisAgent(settings)
    f = _failure(
        "THERMAL_TRIP",
        [
            "[00:01:00.000] THERMAL: WARN: Fan speed below profile | rpm=1200 target_rpm=3800",
            "[00:01:00.500] THERMAL: ERROR: Thermal trip threshold exceeded | t_c=110",
        ],
    )
    hyp = agent.generate(f, [])
    assert hyp.root_cause_id == "RC_FAN_PROFILE_BAD"
    assert hyp.confidence >= 0.8
    assert "Fan speed below profile".lower() in hyp.rationale.lower()
    assert hyp.recommended_actions and hyp.source == "rule-based"


def test_rule_based_uses_rag_votes_without_markers(settings: Settings):
    agent = HypothesisAgent(settings)
    f = _failure("NOC_TIMEOUT", ["NOC: ERROR: Transaction timeout"])
    similar = [
        RetrievedIncident(
            failure_id=f"r{i}:NOC_TIMEOUT",
            run_id=f"r{i}",
            failure_type="NOC_TIMEOUT",
            root_cause_id="RC_NOC_DEADLOCK" if i < 2 else "RC_DMA_BURST_OVERFLOW",
            signature="NOC_TIMEOUT | x",
            distance=0.1,
        )
        for i in range(3)
    ]
    hyp = agent.generate(f, similar)
    assert hyp.root_cause_id == "RC_NOC_DEADLOCK"
    assert 0.45 <= hyp.confidence < 0.8


def test_rule_based_falls_back_to_prior(settings: Settings):
    agent = HypothesisAgent(settings)
    hyp = agent.generate(_failure("UNKNOWN_FAILURE", ["SYSTEM: FATAL: ???"]), [])
    assert hyp.root_cause_id is None and hyp.confidence < 0.5
    assert candidates_for("UNKNOWN_FAILURE") == ()


def test_llm_path_uses_provider_and_validates_id(settings: Settings):
    class FakeProvider:
        name = "fake"

        def __init__(self):
            self.calls: list[dict] = []

        def structured(self, *, system, prompt, tool):
            self.calls.append({"system": system, "prompt": prompt, "tool": tool})
            return {
                "root_cause_id": "RC_PCIE_EQ_TUNING",
                "root_cause": "EQ failed",
                "confidence": 1.7,
                "rationale": "because",
                "recommended_actions": ["sweep presets"],
            }

    provider = FakeProvider()
    agent = HypothesisAgent(settings, provider=provider)
    f = _failure("PCIE_TIMEOUT", ["PCIE: WARN: Equalization phase failed | phase=2"])
    hyp = agent.generate(f, [])
    assert hyp.root_cause_id == "RC_PCIE_EQ_TUNING" and hyp.confidence == 1.0
    assert hyp.source.startswith("llm:")
    enum = provider.calls[0]["tool"]["input_schema"]["properties"]["root_cause_id"]["enum"]
    assert "RC_PCIE_EQ_TUNING" in enum and "OTHER" in enum
    assert "Equalization phase failed" in provider.calls[0]["prompt"]


# -- hold-out evaluation ---------------------------------------------------- #


def test_split_runs_is_deterministic_and_disjoint():
    ids = [f"{i:06d}" for i in range(20)]
    train, test = split_runs(ids, 0.3)
    assert train.isdisjoint(test) and train | test == set(ids)
    assert len(test) == 6
    assert split_runs(ids, 0.3) == (train, test)


def test_evaluate_holdout_reports_root_cause_accuracy(pipeline, settings, dataset):
    result = pipeline.run(index=False, write=False, hypotheses=False)
    labels = load_labels(dataset["labels"])
    metrics = evaluate(
        result,
        labels,
        embedder=pipeline.embedder,
        store=pipeline.store,
        agent=pipeline.agent,
        holdout=0.3,
    )
    assert metrics["detection"]["f1"] == 1.0
    assert metrics["failure_typing"]["micro"]["f1"] == 1.0
    assert metrics["clustering"]["purity"] >= 0.9
    ho = metrics["holdout"]
    assert ho["test_runs"] > 0 and ho["test_failures"] > 0
    rc = metrics["root_cause"]
    assert rc["n"] == ho["test_failures"]
    # Markers are present ~90% of the time, so rule-based accuracy must beat chance.
    assert rc["accuracy"] >= 0.6
    # Store is fully re-populated afterwards.
    assert pipeline.store.count() == len(result.failures)
    truth = root_cause_map(labels)
    assert all(truth.get(f.failure_id) for f in result.failures)


def test_stores_are_namespaced_by_embedding_space(settings: Settings, embedder):
    """A store built with one embedding backend must not be queried with another."""
    from svla_agent.embeddings.embedder import Embedder
    from svla_agent.rag.chroma_store import ChromaVectorStore

    class FakeWideEmbedder(Embedder):
        def __init__(self):
            super().__init__(offline=True)
            self.backend = "fake-384"

        @property
        def space(self) -> str:
            return "fake_384"

        @property
        def dim(self) -> int:
            return 384

        def encode(self, texts):
            import numpy as np

            out = np.zeros((len(texts), 384), dtype=np.float32)
            out[:, 0] = 1.0
            return out

    f = _failure("PCIE_TIMEOUT", ["PCIE: Link training timeout"], "7")
    wide = ChromaVectorStore(settings, FakeWideEmbedder())
    wide.index([f])
    narrow = ChromaVectorStore(settings, embedder)  # hashing, 256-d, same chroma_dir
    assert wide.collection_name != narrow.collection_name
    assert narrow.count() == 0 and narrow.query(f.signature) == []  # no dimension error
    narrow.index([f])
    assert narrow.query(f.signature)[0].failure_id == f.failure_id
    mem_settings = settings.model_copy(update={"vector_backend": "memory"})
    assert (
        InMemoryVectorStore(mem_settings, FakeWideEmbedder())._path
        != InMemoryVectorStore(mem_settings, embedder)._path
    )


def test_make_store_falls_back_to_memory_when_chroma_breaks(
    settings: Settings, embedder, monkeypatch
):
    import chromadb

    def boom(*args, **kwargs):
        raise AttributeError("rust bindings unavailable")

    monkeypatch.setattr(chromadb, "PersistentClient", boom)
    chroma_settings = settings.model_copy(update={"vector_backend": "chroma"})
    store = make_store(chroma_settings, embedder)
    assert isinstance(store, InMemoryVectorStore)
    with pytest.raises(AttributeError):
        make_store(chroma_settings, embedder, fallback=False)

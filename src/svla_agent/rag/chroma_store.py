"""Chroma-backed vector store (persistent, cosine distance)."""

from __future__ import annotations

import contextlib

from ..config import Settings, get_settings
from ..embeddings.embedder import Embedder
from ..logging_config import get_logger
from ..models import Failure, RetrievedIncident
from .base import BaseVectorStore

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Persistent similarity index over failure signatures."""

    name = "chroma"

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder()
        self._client = None
        self._collection = None

    # -- lazy chroma handles ------------------------------------------------ #
    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.settings.rag_collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # -- write -------------------------------------------------------------- #
    def index(
        self,
        failures: list[Failure],
        root_cause_by_id: dict[str, str | None] | None = None,
    ) -> int:
        if not failures:
            return 0
        root_cause_by_id = root_cause_by_id or {}
        embeddings = self.embedder.encode([f.signature for f in failures])
        self.collection.upsert(
            ids=[f.failure_id for f in failures],
            embeddings=embeddings.tolist(),
            documents=[f.signature for f in failures],
            metadatas=[
                {
                    "run_id": f.run_id,
                    "failure_type": f.failure_type,
                    "root_cause_id": root_cause_by_id.get(f.failure_id) or "",
                    "severity": f.severity.value,
                }
                for f in failures
            ],
        )
        logger.info("Indexed %d failures into '%s'", len(failures), self.settings.rag_collection)
        return len(failures)

    # -- read --------------------------------------------------------------- #
    def query(
        self,
        signature: str,
        top_k: int | None = None,
        exclude_failure_id: str | None = None,
    ) -> list[RetrievedIncident]:
        total = self.count()
        if total == 0:
            return []
        top_k = top_k or self.settings.rag_top_k
        embedding = self.embedder.encode([signature])[0].tolist()
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k + 1, total),
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        metas = res["metadatas"][0]
        docs = res["documents"][0]
        dists = res["distances"][0]

        incidents: list[RetrievedIncident] = []
        for fid, meta, doc, dist in zip(ids, metas, docs, dists, strict=True):
            if exclude_failure_id and fid == exclude_failure_id:
                continue
            rc = meta.get("root_cause_id")
            incidents.append(
                RetrievedIncident(
                    failure_id=fid,
                    run_id=str(meta.get("run_id", "")),
                    failure_type=str(meta.get("failure_type", "")),
                    root_cause_id=str(rc) if rc else None,
                    signature=doc,
                    distance=max(0.0, float(dist)),
                )
            )
        return incidents[:top_k]

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:  # pragma: no cover - empty/uninitialised store
            return 0

    def reset(self) -> None:
        name = self.settings.rag_collection
        _ = self.collection  # ensure client exists
        assert self._client is not None
        with contextlib.suppress(Exception):  # already gone
            self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

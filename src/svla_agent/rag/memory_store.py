"""Dependency-free in-memory vector store (numpy cosine search).

Useful for tests, small deployments and environments where Chroma's native
dependencies are unwelcome. Optionally persists to ``<chroma_dir>/memory_store.npz``
so a process restart does not lose the index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Settings, get_settings
from ..embeddings.embedder import Embedder
from ..logging_config import get_logger
from ..models import Failure, RetrievedIncident
from .base import BaseVectorStore

logger = get_logger(__name__)


class InMemoryVectorStore(BaseVectorStore):
    name = "memory"

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        persist: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder()
        self._ids: list[str] = []
        self._meta: list[dict] = []
        self._docs: list[str] = []
        self._vecs: np.ndarray | None = None
        self._persist = persist
        self._path: Path = (
            self.settings.chroma_dir
            / f"{self.settings.rag_collection}__{self.embedder.space}.memory.npz"
        )
        if persist:
            self._load()

    # -- persistence -------------------------------------------------------- #
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = np.load(self._path, allow_pickle=False)
            self._vecs = data["vecs"].astype(np.float32)
            payload = json.loads(str(data["payload"]))
            self._ids = payload["ids"]
            self._meta = payload["meta"]
            self._docs = payload["docs"]
        except Exception as exc:  # pragma: no cover - corrupt file
            logger.warning("Could not load memory store %s: %s", self._path, exc)
            self._ids, self._meta, self._docs, self._vecs = [], [], [], None

    def _save(self) -> None:
        if not self._persist or self._vecs is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"ids": self._ids, "meta": self._meta, "docs": self._docs})
        np.savez(self._path, vecs=self._vecs, payload=np.array(payload))

    # -- write -------------------------------------------------------------- #
    def index(
        self,
        failures: list[Failure],
        root_cause_by_id: dict[str, str | None] | None = None,
    ) -> int:
        if not failures:
            return 0
        root_cause_by_id = root_cause_by_id or {}
        vecs = self.embedder.encode([f.signature for f in failures]).astype(np.float32)
        pos = {fid: i for i, fid in enumerate(self._ids)}
        new_rows: list[np.ndarray] = []
        for f, v in zip(failures, vecs, strict=True):
            meta = {
                "run_id": f.run_id,
                "failure_type": f.failure_type,
                "root_cause_id": root_cause_by_id.get(f.failure_id) or "",
                "severity": f.severity.value,
            }
            if f.failure_id in pos:  # upsert
                i = pos[f.failure_id]
                assert self._vecs is not None
                self._vecs[i] = v
                self._meta[i] = meta
                self._docs[i] = f.signature
            else:
                pos[f.failure_id] = len(self._ids)
                self._ids.append(f.failure_id)
                self._meta.append(meta)
                self._docs.append(f.signature)
                new_rows.append(v)
        if new_rows:
            block = np.vstack(new_rows)
            self._vecs = block if self._vecs is None else np.vstack([self._vecs, block])
        self._save()
        logger.info("Indexed %d failures into in-memory store", len(failures))
        return len(failures)

    # -- read --------------------------------------------------------------- #
    def query(
        self,
        signature: str,
        top_k: int | None = None,
        exclude_failure_id: str | None = None,
    ) -> list[RetrievedIncident]:
        if self._vecs is None or not self._ids:
            return []
        top_k = top_k or self.settings.rag_top_k
        q = self.embedder.encode([signature])[0].astype(np.float32)
        sims = self._vecs @ q  # vectors are L2-normalised => cosine similarity
        order = np.argsort(-sims)
        out: list[RetrievedIncident] = []
        for i in order:
            fid = self._ids[int(i)]
            if exclude_failure_id and fid == exclude_failure_id:
                continue
            meta = self._meta[int(i)]
            out.append(
                RetrievedIncident(
                    failure_id=fid,
                    run_id=meta["run_id"],
                    failure_type=meta["failure_type"],
                    root_cause_id=meta["root_cause_id"] or None,
                    signature=self._docs[int(i)],
                    distance=max(0.0, 1.0 - float(sims[int(i)])),
                )
            )
            if len(out) >= top_k:
                break
        return out

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids, self._meta, self._docs, self._vecs = [], [], [], None
        if self._persist and self._path.exists():
            self._path.unlink()

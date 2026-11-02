"""Backend-agnostic vector store surface.

Any backend (Chroma, an in-memory numpy index, Pinecone, ...) implements this
small interface; the pipeline, agent and evaluation only ever talk to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..embeddings.embedder import Embedder
from ..models import Failure, RetrievedIncident


class BaseVectorStore(ABC):
    """Similarity index over failure signatures with root-cause metadata."""

    embedder: Embedder
    name: str = "base"

    @abstractmethod
    def index(
        self,
        failures: list[Failure],
        root_cause_by_id: dict[str, str | None] | None = None,
    ) -> int:
        """Upsert failures (idempotent on ``failure_id``). Returns the number indexed."""

    @abstractmethod
    def query(
        self,
        signature: str,
        top_k: int | None = None,
        exclude_failure_id: str | None = None,
    ) -> list[RetrievedIncident]:
        """Return the most similar historical incidents (closest first)."""

    @abstractmethod
    def count(self) -> int:
        """Number of indexed failures."""

    @abstractmethod
    def reset(self) -> None:
        """Drop everything in the store."""

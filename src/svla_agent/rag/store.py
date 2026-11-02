"""Vector store factory + backward-compatible alias.

Select the backend with ``SVLA_VECTOR_BACKEND`` (``chroma`` | ``memory``).
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..embeddings.embedder import Embedder
from .base import BaseVectorStore
from .chroma_store import ChromaVectorStore
from .memory_store import InMemoryVectorStore

# Historical name used across the codebase/tests.
VectorStore = ChromaVectorStore

_BACKENDS: dict[str, type[BaseVectorStore]] = {
    "chroma": ChromaVectorStore,
    "memory": InMemoryVectorStore,
}


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def make_store(
    settings: Settings | None = None, embedder: Embedder | None = None
) -> BaseVectorStore:
    """Instantiate the configured vector-store backend."""
    settings = settings or get_settings()
    backend = (settings.vector_backend or "chroma").lower()
    try:
        cls = _BACKENDS[backend]
    except KeyError as exc:
        raise ValueError(
            f"Unknown vector backend '{backend}'. Available: {available_backends()}"
        ) from exc
    return cls(settings, embedder)  # type: ignore[call-arg]


__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "VectorStore",
    "available_backends",
    "make_store",
]

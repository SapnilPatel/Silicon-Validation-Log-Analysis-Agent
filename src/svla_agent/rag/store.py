"""Vector store factory + backward-compatible alias.

Select the backend with ``SVLA_VECTOR_BACKEND`` (``chroma`` | ``memory``).
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..embeddings.embedder import Embedder
from ..logging_config import get_logger
from .base import BaseVectorStore
from .chroma_store import ChromaVectorStore
from .memory_store import InMemoryVectorStore

logger = get_logger(__name__)

# Historical name used across the codebase/tests.
VectorStore = ChromaVectorStore

_BACKENDS: dict[str, type[BaseVectorStore]] = {
    "chroma": ChromaVectorStore,
    "memory": InMemoryVectorStore,
}


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def make_store(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    *,
    fallback: bool = True,
) -> BaseVectorStore:
    """Instantiate the configured vector-store backend.

    Chroma is opened eagerly so a broken native dependency (e.g. an unsupported
    Python/sqlite build on a hosting platform) surfaces here; with ``fallback``
    the in-memory store takes over with a warning instead of crashing later.
    """
    settings = settings or get_settings()
    backend = (settings.vector_backend or "chroma").lower()
    try:
        cls = _BACKENDS[backend]
    except KeyError as exc:
        raise ValueError(
            f"Unknown vector backend '{backend}'. Available: {available_backends()}"
        ) from exc
    store = cls(settings, embedder)  # type: ignore[call-arg]
    if isinstance(store, ChromaVectorStore):
        try:
            _ = store.collection
        except Exception as exc:
            if not fallback:
                raise
            logger.warning(
                "Chroma unavailable (%s: %s); using the in-memory vector store instead",
                type(exc).__name__,
                exc,
            )
            return InMemoryVectorStore(settings, store.embedder)
    return store


__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "VectorStore",
    "available_backends",
    "make_store",
]

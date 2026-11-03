"""Text embeddings for clustering and RAG.

Primary backend is ``sentence-transformers``. When the model can't be loaded
(no network on first download, or ``offline`` mode), we fall back to a
deterministic hashing embedding so the rest of the pipeline — and CI — keep
working without surprises. The active backend is reported via :attr:`backend`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)

_HASH_DIM = 256


class Embedder:
    """Encode text into L2-normalised vectors.

    Parameters
    ----------
    model_name:
        sentence-transformers model id (defaults to settings).
    offline:
        Force the deterministic hashing backend (no model download/inference).
    """

    def __init__(self, model_name: str | None = None, offline: bool | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._offline = settings.offline if offline is None else offline
        self._model: Any = None
        self.backend = "hashing" if self._offline else "pending"

    # -- backend management ------------------------------------------------- #
    def _ensure_model(self) -> None:
        if self._model is not None or self._offline:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.backend = f"sentence-transformers:{self.model_name}"
            logger.info("Loaded embedding model [bold]%s[/bold]", self.model_name)
        except Exception as exc:  # pragma: no cover - depends on env/network
            logger.warning(
                "Falling back to hashing embeddings (could not load '%s': %s)",
                self.model_name,
                exc,
            )
            self._offline = True
            self.backend = "hashing"

    @property
    def space(self) -> str:
        """Short id of the vector space (backend + dimension).

        Vectors from different backends are not comparable, so stores namespace
        their collections/files by this value.
        """
        self._ensure_model()
        if self._model is not None:
            slug = re.sub(r"[^a-z0-9]+", "_", self.model_name.lower()).strip("_")[-40:]
            return f"st_{slug}_{self.dim}"
        return f"hash_{_HASH_DIM}"

    @property
    def dim(self) -> int:
        self._ensure_model()
        if self._model is not None:
            getter = getattr(self._model, "get_embedding_dimension", None) or (
                self._model.get_sentence_embedding_dimension
            )
            return int(getter())
        return _HASH_DIM

    # -- encoding ----------------------------------------------------------- #
    def _hash_encode(self, texts: list[str]) -> np.ndarray:
        """Deterministic bag-of-hashed-tokens embedding (no dependencies)."""
        vectors = np.zeros((len(texts), _HASH_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % _HASH_DIM
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[i, idx] += sign
        return vectors

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 matrix of L2-normalised embeddings."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        self._ensure_model()
        if self._model is not None:
            vectors = np.asarray(
                self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            return vectors

        vectors = self._hash_encode(texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

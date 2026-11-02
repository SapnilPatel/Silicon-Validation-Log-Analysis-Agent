"""Runtime configuration, loaded from environment / ``.env`` via pydantic-settings.

All settings use the ``SVLA_`` prefix except ``ANTHROPIC_API_KEY`` which keeps its
conventional name so the ``anthropic`` SDK and our config agree.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SVLA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_provider: str = "anthropic"  # pluggable; see hypothesize/llm.py
    llm_model: str = "claude-opus-4-8"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Clustering ---
    cluster_min_size: int = 3
    cluster_min_samples: int = 1

    # --- RAG ---
    vector_backend: str = "chroma"  # "chroma" (persistent) or "memory" (numpy, file-backed)
    chroma_dir: Path = Path(".chroma")
    rag_collection: str = "svla_failures"
    rag_top_k: int = 5

    # --- Paths ---
    logs_dir: Path = Path("data/silicon_logs")
    labels_dir: Path = Path("data/labels")
    reports_dir: Path = Path("reports")

    # --- Classification ---
    evidence_window_ms: int = 8000  # how far before the first error to look for WARN context

    # --- Privacy ---
    redact: bool = True  # scrub PII / secrets at parse time, before indexing or prompting

    # --- Evaluation ---
    eval_holdout: float = 0.3  # fraction of runs held out as "new" incidents for RAG/RC eval

    # --- Behaviour ---
    log_level: str = "INFO"
    offline: bool = False  # force rule-based / no-network paths (used by CI + tests)

    @property
    def llm_enabled(self) -> bool:
        """True when we should call Claude (key present and not forced offline)."""
        return bool(self.anthropic_api_key) and not self.offline


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    ``SVLA_OFFLINE`` is honoured even though it is a bare flag so that CI can pin
    the whole pipeline to deterministic, network-free behaviour.
    """
    settings = Settings()
    if os.environ.get("SVLA_OFFLINE", "").lower() in {"1", "true", "yes"}:
        settings.offline = True
    return settings

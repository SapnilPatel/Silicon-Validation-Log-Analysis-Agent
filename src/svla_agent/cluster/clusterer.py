"""Unsupervised clustering of failures into recurring families.

Failures are embedded by their normalised signature and grouped with HDBSCAN
(``sklearn.cluster.HDBSCAN``). HDBSCAN labels low-density points as noise
(``-1``); rather than discard them, we fold each noise point into a per-type
fallback family so every failure ends up in exactly one cluster.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import HDBSCAN

from ..config import get_settings
from ..embeddings.embedder import Embedder
from ..logging_config import get_logger
from ..models import Failure, FailureCluster

logger = get_logger(__name__)


def _exemplar_label(failure_types: list[str]) -> str:
    """Pick the most common failure type as the cluster's display label."""
    if not failure_types:
        return "cluster"
    return Counter(failure_types).most_common(1)[0][0]


def _assign_labels(embeddings: np.ndarray, failures: list[Failure]) -> np.ndarray:
    """Run HDBSCAN, then deterministically relabel noise into fallback families."""
    settings = get_settings()
    n = len(failures)
    if n < 2:
        return np.zeros(n, dtype=int)

    min_size = max(2, min(settings.cluster_min_size, n))
    clusterer = HDBSCAN(
        min_cluster_size=min_size,
        min_samples=settings.cluster_min_samples,
        metric="euclidean",
        copy=True,
    )
    labels = clusterer.fit_predict(embeddings).astype(int)

    # Fold noise (-1) into stable per-failure-type fallback clusters so nothing
    # is silently dropped. Fallback ids are offset above any real cluster id.
    next_id = int(labels.max()) + 1 if labels.max() >= 0 else 0
    type_to_fallback: dict[str, int] = {}
    for i, label in enumerate(labels):
        if label != -1:
            continue
        ftype = failures[i].failure_type
        if ftype not in type_to_fallback:
            type_to_fallback[ftype] = next_id
            next_id += 1
        labels[i] = type_to_fallback[ftype]
    return labels


def cluster_failures(
    failures: list[Failure], embedder: Embedder | None = None
) -> tuple[list[Failure], list[FailureCluster]]:
    """Cluster failures in place (sets ``cluster_id``) and return cluster summaries."""
    if not failures:
        return failures, []

    embedder = embedder or Embedder()
    embeddings = embedder.encode([f.signature for f in failures])
    labels = _assign_labels(embeddings, failures)

    groups: dict[int, list[Failure]] = {}
    for failure, label in zip(failures, labels, strict=True):
        failure.cluster_id = int(label)
        groups.setdefault(int(label), []).append(failure)

    clusters: list[FailureCluster] = []
    for cid, members in sorted(groups.items()):
        ftypes = [m.failure_type for m in members]
        exemplar = max(members, key=lambda m: (m.first_ts_ms, len(m.evidence)))
        clusters.append(
            FailureCluster(
                cluster_id=cid,
                label=_exemplar_label(ftypes),
                size=len(members),
                failure_ids=[m.failure_id for m in members],
                failure_types=sorted(set(ftypes)),
                exemplar_failure_id=exemplar.failure_id,
                representative_signature=exemplar.signature,
            )
        )

    logger.info(
        "Clustered %d failures into %d families (embeddings: %s)",
        len(failures),
        len(clusters),
        embedder.backend,
    )
    return failures, clusters

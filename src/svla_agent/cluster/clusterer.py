"""Unsupervised clustering of failures into recurring families.

Failures are embedded by their normalised signature and grouped with HDBSCAN
(``sklearn.cluster.HDBSCAN``). Clustering runs *within* each classified failure
type: the type is a hard, explainable boundary (an engineer never wants a PCIe
and a thermal failure in one family), and HDBSCAN's job is to discover the
sub-families inside a type - typically the different root causes.

HDBSCAN labels low-density points as noise (``-1``); rather than discard them,
noise points of a type are folded into one fallback family so every failure
ends up in exactly one cluster.
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


def _hdbscan_labels(embeddings: np.ndarray) -> np.ndarray:
    """HDBSCAN on one type's embeddings; noise folded into one extra label."""
    settings = get_settings()
    n = len(embeddings)
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
    if (labels == -1).any():
        labels[labels == -1] = int(labels.max()) + 1 if labels.max() >= 0 else 0
    return labels


def _assign_labels(embeddings: np.ndarray, failures: list[Failure]) -> np.ndarray:
    """Cluster within each failure type; return globally unique cluster ids."""
    out = np.zeros(len(failures), dtype=int)
    next_id = 0
    by_type: dict[str, list[int]] = {}
    for i, f in enumerate(failures):
        by_type.setdefault(f.failure_type, []).append(i)
    for ftype in sorted(by_type):
        idx = by_type[ftype]
        local = _hdbscan_labels(embeddings[idx])
        for i, lab in zip(idx, local, strict=True):
            out[i] = next_id + int(lab)
        next_id += int(local.max()) + 1 if len(local) else 1
    return out


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
        exemplar = max(members, key=lambda m: (m.occurrences, -m.first_ts_ms, len(m.evidence)))
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

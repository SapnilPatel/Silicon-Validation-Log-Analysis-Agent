"""Evaluate pipeline output against ground-truth labels.

Reports four families of metrics:

* **Detection** — run-level anomaly precision/recall/F1.
* **Failure typing** — multi-label precision/recall/F1 over failure types.
* **Clustering quality** — silhouette, Davies-Bouldin, and homogeneity of the
  discovered clusters w.r.t. ground-truth failure types.
* **Retrieval** — RAG recall@k (does top-k contain a same-type incident).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import (
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
)

from ..embeddings.embedder import Embedder
from ..models import Failure
from ..pipeline import PipelineResult
from ..rag.store import VectorStore


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def detection_metrics(result: PipelineResult, labels: dict[str, dict]) -> dict:
    tp = fp = fn = tn = 0
    for report in result.reports:
        truth = bool(labels.get(report.run_id, {}).get("anomalous", False))
        pred = report.anomalous
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    metrics: dict[str, object] = dict(_prf(tp, fp, fn))
    metrics["accuracy"] = round((tp + tn) / max(1, tp + tn + fp + fn), 4)
    metrics["support"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return metrics


def failure_type_metrics(result: PipelineResult, labels: dict[str, dict]) -> dict:
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for report in result.reports:
        pred = {f.failure_type for f in report.failures}
        truth = set(labels.get(report.run_id, {}).get("failure_types", []))
        for t in pred & truth:
            tp[t] += 1
        for t in pred - truth:
            fp[t] += 1
        for t in truth - pred:
            fn[t] += 1

    per_type = {t: _prf(tp[t], fp[t], fn[t]) for t in sorted(set(tp) | set(fp) | set(fn))}
    total = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return {"micro": total, "per_type": per_type}


def cluster_quality(result: PipelineResult, embedder: Embedder | None = None) -> dict:
    failures = result.failures
    if len(failures) < 3:
        return {"note": "not enough failures to score clustering"}

    embedder = embedder or Embedder()
    embeddings = embedder.encode([f.signature for f in failures])
    cluster_ids = np.array([f.cluster_id if f.cluster_id is not None else -1 for f in failures])
    true_types = [f.failure_type for f in failures]

    n_clusters = len(set(cluster_ids))
    out: dict = {"n_clusters": n_clusters, "n_failures": len(failures)}
    if 2 <= n_clusters <= len(failures) - 1:
        out["silhouette"] = round(float(silhouette_score(embeddings, cluster_ids)), 4)
        out["davies_bouldin"] = round(float(davies_bouldin_score(embeddings, cluster_ids)), 4)
    out["homogeneity_vs_failure_type"] = round(float(homogeneity_score(true_types, cluster_ids)), 4)
    return out


def retrieval_recall_at_k(
    failures: list[Failure],
    store: VectorStore,
    k: int = 5,
) -> dict:
    if store.count() == 0 or not failures:
        return {"note": "empty store or no failures"}
    hits = 0
    considered = 0
    for f in failures:
        incidents = store.query(f.signature, top_k=k, exclude_failure_id=f.failure_id)
        if not incidents:
            continue
        considered += 1
        if any(inc.failure_type == f.failure_type for inc in incidents):
            hits += 1
    return {
        "k": k,
        "recall_at_k": round(hits / considered, 4) if considered else 0.0,
        "considered": considered,
    }


def evaluate(
    result: PipelineResult,
    labels: dict[str, dict],
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> dict:
    """Compute the full metrics bundle."""
    report: dict = {
        "n_runs": len(result.reports),
        "detection": detection_metrics(result, labels),
        "failure_typing": failure_type_metrics(result, labels),
        "clustering": cluster_quality(result, embedder),
    }
    if store is not None:
        report["retrieval"] = retrieval_recall_at_k(result.failures, store)
    return report

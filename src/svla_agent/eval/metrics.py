"""Evaluate pipeline output against ground-truth labels.

Metric families:

* **Detection** - run-level anomaly precision/recall/F1/accuracy.
* **Failure typing** - multi-label precision/recall/F1 over failure types.
* **Clustering quality** - silhouette, Davies-Bouldin, homogeneity/completeness
  of the discovered clusters w.r.t. ground-truth failure types.
* **Retrieval** - RAG recall@k (does top-k contain a same-type / same-root-cause
  incident), measured on *held-out* runs against a store built from the rest.
* **Root cause** - top-1 accuracy of the hypothesis agent's ``root_cause_id`` on
  the held-out runs, plus confidence calibration (mean confidence when right vs
  wrong) and a per-type breakdown.

The hold-out split matters: scoring retrieval on incidents that are themselves
in the index would be trivially perfect.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import (
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
)

from ..embeddings.embedder import Embedder
from ..hypothesize.agent import HypothesisAgent
from ..labels import root_cause_map
from ..models import Failure
from ..pipeline import PipelineResult
from ..rag.base import BaseVectorStore


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# --------------------------------------------------------------------------- #
# Detection / typing / clustering
# --------------------------------------------------------------------------- #


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


def cluster_quality(
    result: PipelineResult,
    embedder: Embedder | None = None,
    rc_truth: dict[str, str | None] | None = None,
) -> dict:
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
    out["completeness_vs_failure_type"] = round(
        float(completeness_score(true_types, cluster_ids)), 4
    )
    # Purity: fraction of failures whose cluster's majority type is their own type.
    by_cluster: dict[int, Counter[str]] = defaultdict(Counter)
    for cid, t in zip(cluster_ids, true_types, strict=True):
        by_cluster[int(cid)][t] += 1
    majority = sum(c.most_common(1)[0][1] for c in by_cluster.values())
    out["purity"] = round(majority / len(failures), 4)

    # Clusters tend to land at root-cause granularity; score that too when labels exist.
    if rc_truth:
        pairs: list[tuple[int, str]] = [
            (int(cid), str(rc_truth[f.failure_id]))
            for cid, f in zip(cluster_ids, failures, strict=True)
            if rc_truth.get(f.failure_id)
        ]
        if len(pairs) >= 3:
            cids = [c for c, _ in pairs]
            rcs = [r for _, r in pairs]
            out["homogeneity_vs_root_cause"] = round(float(homogeneity_score(rcs, cids)), 4)
            out["completeness_vs_root_cause"] = round(float(completeness_score(rcs, cids)), 4)
            rc_by_cluster: dict[int, Counter[str]] = defaultdict(Counter)
            for c, r in pairs:
                rc_by_cluster[c][r] += 1
            out["purity_vs_root_cause"] = round(
                sum(c.most_common(1)[0][1] for c in rc_by_cluster.values()) / len(pairs), 4
            )
    return out


# --------------------------------------------------------------------------- #
# Retrieval + root cause (hold-out)
# --------------------------------------------------------------------------- #


def split_runs(run_ids: list[str], holdout: float) -> tuple[set[str], set[str]]:
    """Deterministic interleaved split: every ``k``-th run is held out."""
    ordered = sorted(set(run_ids))
    if holdout <= 0 or len(ordered) < 4:
        return set(ordered), set()
    k = max(2, round(1 / holdout))
    test = {r for i, r in enumerate(ordered) if i % k == k - 1}
    train = set(ordered) - test
    return train, test


def retrieval_recall_at_k(
    failures: list[Failure],
    store: BaseVectorStore,
    k: int = 5,
    rc_truth: dict[str, str | None] | None = None,
    indexed_types: Counter[str] | None = None,
) -> dict:
    """recall@k for same-type and same-root-cause neighbours.

    Only queries whose type (resp. root cause) exists at least once in the index
    are counted - a neighbour that does not exist cannot be retrieved.
    """
    if store.count() == 0 or not failures:
        return {"note": "empty store or no failures"}
    rc_truth = rc_truth or {}
    type_hits = type_considered = rc_hits = rc_considered = 0
    rr_sum = 0.0
    for f in failures:
        incidents = store.query(f.signature, top_k=k, exclude_failure_id=f.failure_id)
        if not incidents:
            continue
        if indexed_types is None or indexed_types.get(f.failure_type, 0) > 0:
            type_considered += 1
            ranks = [i for i, inc in enumerate(incidents) if inc.failure_type == f.failure_type]
            if ranks:
                type_hits += 1
                rr_sum += 1.0 / (ranks[0] + 1)
        truth_rc = rc_truth.get(f.failure_id)
        if truth_rc:
            rc_considered += 1
            if any(inc.root_cause_id == truth_rc for inc in incidents):
                rc_hits += 1
    out: dict = {
        "k": k,
        "recall_at_k": round(type_hits / type_considered, 4) if type_considered else 0.0,
        "mrr": round(rr_sum / type_considered, 4) if type_considered else 0.0,
        "considered": type_considered,
    }
    if rc_considered:
        out["root_cause_recall_at_k"] = round(rc_hits / rc_considered, 4)
        out["root_cause_considered"] = rc_considered
    return out


def root_cause_metrics(
    pairs: list[tuple[Failure, str | None, str | None, float]],
) -> dict:
    """Top-1 accuracy of predicted root-cause ids.

    ``pairs`` holds ``(failure, truth_rc, predicted_rc, confidence)``.
    """
    scored = [(f, t, p, c) for f, t, p, c in pairs if t]
    if not scored:
        return {"note": "no labelled root causes"}
    correct = [(f, t, p, c) for f, t, p, c in scored if p == t]
    wrong = [(f, t, p, c) for f, t, p, c in scored if p != t]
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for f, t, p, _ in scored:
        per_type[f.failure_type]["total"] += 1
        per_type[f.failure_type]["correct"] += int(p == t)
    return {
        "accuracy": round(len(correct) / len(scored), 4),
        "n": len(scored),
        "predicted_rate": round(sum(1 for _, _, p, _ in scored if p) / len(scored), 4),
        "mean_confidence_correct": round(float(np.mean([c for *_, c in correct])), 4)
        if correct
        else None,
        "mean_confidence_wrong": round(float(np.mean([c for *_, c in wrong])), 4)
        if wrong
        else None,
        "per_type": {
            t: {**v, "accuracy": round(v["correct"] / v["total"], 4)}
            for t, v in sorted(per_type.items())
        },
    }


def holdout_evaluation(
    result: PipelineResult,
    labels: dict[str, dict],
    store: BaseVectorStore,
    agent: HypothesisAgent,
    holdout: float = 0.3,
    k: int = 5,
) -> dict:
    """Index the *train* runs' failures, then retrieve + hypothesise the held-out ones.

    Leaves the store re-populated with every failure afterwards.
    """
    rc_map = root_cause_map(labels)
    run_ids = [r.run_id for r in result.reports]
    train, test = split_runs(run_ids, holdout)
    if not test:
        return {"note": "not enough runs for a hold-out split"}

    train_failures = [f for f in result.failures if f.run_id in train]
    test_failures = [f for f in result.failures if f.run_id in test]
    if not train_failures or not test_failures:
        return {"note": "hold-out split produced an empty side"}

    store.reset()
    store.index(train_failures, {f.failure_id: rc_map.get(f.failure_id) for f in train_failures})
    indexed_types = Counter(f.failure_type for f in train_failures)

    retrieval = retrieval_recall_at_k(
        test_failures, store, k=k, rc_truth=rc_map, indexed_types=indexed_types
    )

    pairs: list[tuple[Failure, str | None, str | None, float]] = []
    sources: Counter[str] = Counter()
    for f in test_failures:
        similar = store.query(f.signature, top_k=k, exclude_failure_id=f.failure_id)
        hyp = agent.generate(f, similar, f.cluster_id)
        sources[hyp.source] += 1
        pairs.append((f, rc_map.get(f.failure_id), hyp.root_cause_id, hyp.confidence))

    # Restore the full index for downstream users of the store.
    store.index(test_failures, {f.failure_id: rc_map.get(f.failure_id) for f in test_failures})

    return {
        "holdout_fraction": holdout,
        "train_runs": len(train),
        "test_runs": len(test),
        "train_failures": len(train_failures),
        "test_failures": len(test_failures),
        "retrieval": retrieval,
        "root_cause": root_cause_metrics(pairs),
        "hypothesis_sources": dict(sources),
    }


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #


def evaluate(
    result: PipelineResult,
    labels: dict[str, dict],
    embedder: Embedder | None = None,
    store: BaseVectorStore | None = None,
    agent: HypothesisAgent | None = None,
    holdout: float = 0.3,
    k: int = 5,
) -> dict:
    """Compute the full metrics bundle."""
    report: dict = {
        "n_runs": len(result.reports),
        "n_failures": len(result.failures),
        "detection": detection_metrics(result, labels),
        "failure_typing": failure_type_metrics(result, labels),
        "clustering": cluster_quality(result, embedder, root_cause_map(labels)),
    }
    if store is not None and agent is not None:
        ho = holdout_evaluation(result, labels, store, agent, holdout=holdout, k=k)
        report["holdout"] = ho
        if "retrieval" in ho:
            report["retrieval"] = ho["retrieval"]
            report["root_cause"] = ho["root_cause"]
    elif store is not None:
        report["retrieval"] = retrieval_recall_at_k(result.failures, store, k=k)
    return report

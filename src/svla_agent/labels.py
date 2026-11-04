"""Helpers for loading the dataset's ground-truth labels.

Labels are emitted by the synthetic generator as ``data/labels/run_*.label.json``
and are used both to seed the RAG store with known root causes and to evaluate
pipeline output.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_labels(labels_dir: str | Path) -> dict[str, dict]:
    """Return ``{run_id: label_dict}`` for all label files under ``labels_dir``."""
    root = Path(labels_dir)
    labels: dict[str, dict] = {}
    if not root.exists():
        return labels
    for path in sorted(root.glob("*.label.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        run_id = str(data.get("run_id", path.stem.split(".")[0].replace("run_", "")))
        labels[run_id] = data
    return labels


def root_cause_map(labels: dict[str, dict]) -> dict[str, str | None]:
    """Map ``"{run_id}:{failure_type}" -> root_cause_id`` from injected patterns.

    Falls back to the run-level ``root_cause_id`` when per-pattern detail is absent.
    """
    out: dict[str, str | None] = {}
    for run_id, data in labels.items():
        patterns = data.get("injected_patterns") or []
        for pat in patterns:
            ftype = pat.get("failure_type")
            if ftype:
                out[f"{run_id}:{ftype}"] = pat.get("root_cause_id")
        for ftype in data.get("failure_types", []):
            out.setdefault(f"{run_id}:{ftype}", data.get("root_cause_id"))
        # Run-level fallback: lets datasets whose labels are not keyed by our
        # failure types (e.g. BGL alert tags) still score retrieval/clustering.
        if data.get("root_cause_id"):
            out.setdefault(run_id, data["root_cause_id"])
    return out


def lookup_root_cause(rc_map: dict[str, str | None], failure_id: str) -> str | None:
    """Root cause for a failure id, falling back to its run-level label."""
    rc = rc_map.get(failure_id)
    if rc:
        return rc
    return rc_map.get(failure_id.split(":", 1)[0])


def accepted_root_causes(labels: dict[str, dict]) -> dict[str, set[str]]:
    """``"{run_id}:{failure_type}" -> every injected root cause of that type``.

    A run can carry two injected patterns of the same failure type with different
    root causes; the pipeline reports one failure per type, so any of them counts.
    """
    out: dict[str, set[str]] = {}
    for run_id, data in labels.items():
        for pat in data.get("injected_patterns") or []:
            ftype, rc = pat.get("failure_type"), pat.get("root_cause_id")
            if ftype and rc:
                out.setdefault(f"{run_id}:{ftype}", set()).add(rc)
    return out

"""Distil parsed events into classified, prioritised :class:`Failure` records.

Approach: every ``ERROR``/``FATAL`` event is tagged with a coarse failure type
using subsystem-aware keyword rules (which line up with the dataset's ground-truth
labels). Events of the same type within a run are grouped into one failure,
enriched with nearby ``WARN`` context as evidence, and assigned a severity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from ..config import get_settings
from ..models import Event, Failure, LogLevel, ParsedRun, Severity

# Ordered (failure_type, predicate-keywords) rules. First match wins.
# Keys are matched case-insensitively against "subsys: message" plus kv reason.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("PCIE_TIMEOUT", ("link training timeout", "pcie initialization failed", "ltssm", "aer")),
    ("DDR_ECC", ("ecc uncorrectable", "ddr subsystem failure", "memtest mismatch", "ddr phy")),
    ("PMU_DROOP", ("brown-out", "brownout", "droop", "watchdog timeout")),
    ("THERMAL_TRIP", ("thermal trip", "thermal_trip", "emergency shutdown")),
    ("FW_EXCEPTION", ("unhandled exception", "firmware crash", "fw_exception", "hardfault")),
    ("NOC_TIMEOUT", ("transaction timeout", "interconnect failure", "noc_timeout")),
]

# For error/fatal events whose message matches no keyword rule (e.g. a bare
# "register snapshot" dump), the subsystem reliably implies the failure family.
_SUBSYS_FALLBACK: dict[str, str] = {
    "PCIE": "PCIE_TIMEOUT",
    "DDR": "DDR_ECC",
    "PMU": "PMU_DROOP",
    "THERMAL": "THERMAL_TRIP",
    "FW": "FW_EXCEPTION",
    "NOC": "NOC_TIMEOUT",
}

_MAX_EVIDENCE = 16
_MAX_CONTEXT = 8

# Periodic telemetry templates carry no diagnostic signal; keep them out of the
# evidence/signature so they do not dilute clustering and retrieval.
_TELEMETRY_MESSAGES = {
    "temp sample",
    "rail telemetry",
    "ddr counters",
    "link stats",
    "fabric telemetry",
    "heartbeat",
    "checkpoint",
    "loop tick",
}

_SEVERITY_RANK = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def classify_event(event: Event) -> str | None:
    """Return the failure type for a single error/fatal event, or ``None``."""
    if event.level.rank < LogLevel.ERROR.rank:
        return None
    haystack = f"{event.subsys}: {event.message} {event.kv.get('reason', '')}".lower()
    for failure_type, keywords in _RULES:
        if any(kw in haystack for kw in keywords):
            return failure_type
    return _SUBSYS_FALLBACK.get(event.subsys.upper(), "UNKNOWN_FAILURE")


def _severity_for(events: list[Event]) -> Severity:
    max_rank = max(e.level.rank for e in events)
    if max_rank >= LogLevel.FATAL.rank:
        return Severity.CRITICAL if len(events) >= 3 else Severity.HIGH
    return Severity.MEDIUM


def _is_telemetry(event: Event) -> bool:
    return event.message.strip().lower() in _TELEMETRY_MESSAGES


def _distinct_context(context: list[Event], limit: int = _MAX_CONTEXT) -> list[Event]:
    """Keep the first occurrence of each WARN template, newest ``limit`` templates.

    A pattern may repeat one warning many times (``LTSSM -> Recovery`` x7); keeping
    only the tail would push out the single, earlier tell-tale line that names the
    root cause.
    """
    first_by_template: dict[str, Event] = {}
    for ev in context:
        first_by_template.setdefault(ev.signature, ev)
    distinct = sorted(first_by_template.values(), key=lambda e: e.ts_ms)
    return distinct[-limit:]


def _evidence_for(error_events: list[Event], context: list[Event]) -> list[str]:
    """Build human-readable evidence lines: leading WARN context + the errors."""
    lines: list[str] = []
    for ev in _distinct_context(context):
        lines.append(ev.raw or f"{ev.subsys}: {ev.level.value}: {ev.message}")
    for ev in error_events:
        lines.append(ev.raw or f"{ev.subsys}: {ev.level.value}: {ev.message}")
        if ev.multiline:
            head = ev.multiline[0]
            lines.append(f"    {head}{' ...' if len(ev.multiline) > 1 else ''}")
    return lines[:_MAX_EVIDENCE]


def _signature(failure_type: str, events: list[Event], context: list[Event]) -> str:
    """Normalised, low-cardinality text used for embedding/clustering/RAG.

    Error/fatal templates come first, followed by the (non-telemetry) WARN
    templates that preceded them - those carry the root-cause tell-tales.
    """
    templates: list[str] = []
    seen: set[str] = set()
    for ev in [*events, *_distinct_context(context)]:
        sig = ev.signature
        if sig not in seen:
            seen.add(sig)
            templates.append(sig)
    return f"{failure_type} | " + " ; ".join(templates[:10])


def fingerprint_for(signature: str) -> str:
    """Short stable hash of a signature; identical failures share it across runs."""
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]


def classify_run(parsed: ParsedRun) -> list[Failure]:
    """Extract and classify all failures present in a parsed run."""
    typed: dict[str, list[Event]] = {}
    for event in parsed.events:
        ftype = classify_event(event)
        if ftype is not None:
            typed.setdefault(ftype, []).append(event)

    if not typed:
        return []

    window_ms = get_settings().evidence_window_ms
    warns = [e for e in parsed.events if e.level is LogLevel.WARN and not _is_telemetry(e)]
    failures: list[Failure] = []
    for ftype, evs in typed.items():
        evs.sort(key=lambda e: e.ts_ms)
        first_ts, last_ts = evs[0].ts_ms, evs[-1].ts_ms
        subsystems = {e.subsys for e in evs}
        # Non-telemetry WARN events (any subsystem) shortly before / during the
        # failure window - cross-subsystem context is where root causes hide
        # (e.g. a PMU droop preceding a DDR uncorrectable error).
        context = [w for w in warns if first_ts - window_ms <= w.ts_ms <= last_ts]
        primary = max(evs, key=lambda e: (e.level.rank, e.ts_ms))
        signature = _signature(ftype, evs, context)
        failures.append(
            Failure(
                failure_id=f"{parsed.run_id}:{ftype}",
                run_id=parsed.run_id,
                failure_type=ftype,
                severity=_severity_for(evs),
                subsys=primary.subsys,
                first_ts_ms=first_ts,
                last_ts_ms=last_ts,
                summary=f"{ftype} in {'/'.join(sorted(subsystems))}: {primary.message}",
                signature=signature,
                evidence=_evidence_for(evs, context),
                kv=primary.kv,
                fingerprint=fingerprint_for(signature),
            )
        )

    failures.sort(key=lambda f: (-_SEVERITY_RANK[f.severity], f.first_ts_ms))
    return failures


def classify_runs(parsed_runs: Iterable[ParsedRun]) -> list[Failure]:
    """Classify failures across many runs (corpus-level convenience)."""
    out: list[Failure] = []
    for pr in parsed_runs:
        out.extend(classify_run(pr))
    return deduplicate(out)


def deduplicate(failures: list[Failure]) -> list[Failure]:
    """Corpus-level dedup: count how often each fingerprint recurs.

    Failures are *kept* (each run still gets its report) but annotated with
    ``occurrences`` so recurring, already-known failures are visible at a glance.
    """
    counts: dict[str, int] = {}
    for f in failures:
        if not f.fingerprint:
            f.fingerprint = fingerprint_for(f.signature)
        counts[f.fingerprint] = counts.get(f.fingerprint, 0) + 1
    for f in failures:
        f.occurrences = counts[f.fingerprint]
    return failures

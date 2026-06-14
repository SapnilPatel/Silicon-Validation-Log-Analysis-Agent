"""Distil parsed events into classified, prioritised :class:`Failure` records.

Approach: every ``ERROR``/``FATAL`` event is tagged with a coarse failure type
using subsystem-aware keyword rules (which line up with the dataset's ground-truth
labels). Events of the same type within a run are grouped into one failure,
enriched with nearby ``WARN`` context as evidence, and assigned a severity.
"""

from __future__ import annotations

from collections.abc import Iterable

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

_MAX_EVIDENCE = 12

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


def _evidence_for(error_events: list[Event], context: list[Event]) -> list[str]:
    """Build human-readable evidence lines: leading WARN context + the errors."""
    lines: list[str] = []
    for ev in context[-3:]:
        lines.append(ev.raw or f"{ev.subsys}: {ev.level.value}: {ev.message}")
    for ev in error_events:
        lines.append(ev.raw or f"{ev.subsys}: {ev.level.value}: {ev.message}")
        if ev.multiline:
            head = ev.multiline[0]
            lines.append(f"    {head}{' ...' if len(ev.multiline) > 1 else ''}")
    return lines[:_MAX_EVIDENCE]


def _signature(failure_type: str, events: list[Event]) -> str:
    """Normalised, low-cardinality text used for embedding/clustering/RAG."""
    templates: list[str] = []
    seen: set[str] = set()
    for ev in events:
        sig = ev.signature
        if sig not in seen:
            seen.add(sig)
            templates.append(sig)
    return f"{failure_type} | " + " ; ".join(templates[:6])


def classify_run(parsed: ParsedRun) -> list[Failure]:
    """Extract and classify all failures present in a parsed run."""
    typed: dict[str, list[Event]] = {}
    for event in parsed.events:
        ftype = classify_event(event)
        if ftype is not None:
            typed.setdefault(ftype, []).append(event)

    if not typed:
        return []

    warns = [e for e in parsed.events if e.level is LogLevel.WARN]
    failures: list[Failure] = []
    for ftype, evs in typed.items():
        evs.sort(key=lambda e: e.ts_ms)
        first_ts, last_ts = evs[0].ts_ms, evs[-1].ts_ms
        subsystems = {e.subsys for e in evs}
        # WARN events of the same subsystems just before the failure window.
        context = [w for w in warns if w.subsys in subsystems and w.ts_ms <= last_ts]
        primary = max(evs, key=lambda e: (e.level.rank, e.ts_ms))
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
                signature=_signature(ftype, evs),
                evidence=_evidence_for(evs, context),
                kv=primary.kv,
            )
        )

    failures.sort(key=lambda f: (-_SEVERITY_RANK[f.severity], f.first_ts_ms))
    return failures


def classify_runs(parsed_runs: Iterable[ParsedRun]) -> list[Failure]:
    """Classify failures across many runs (corpus-level convenience)."""
    out: list[Failure] = []
    for pr in parsed_runs:
        out.extend(classify_run(pr))
    return out

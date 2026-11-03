"""Tools the agentic hypothesis loop can call while investigating a failure.

Each tool is a plain function over data the pipeline already has (the parsed
run, the incident store, the catalogue). The model decides which to call and
when it has seen enough; every call is recorded in the hypothesis trace.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..models import Event, Failure, LogLevel
from ..rag.base import BaseVectorStore
from .knowledge import candidates_for, prior_for

MAX_RESULT_CHARS = 6000


def _fmt(ev: Event) -> str:
    return ev.raw or f"[{ev.ts_ms} ms] {ev.subsys}: {ev.level.value}: {ev.message}"


class AgentTools:
    """Tool implementations bound to one failure's run."""

    def __init__(
        self,
        failure: Failure,
        events: list[Event] | None = None,
        store: BaseVectorStore | None = None,
    ) -> None:
        self.failure = failure
        self.events = sorted(events or [], key=lambda e: e.ts_ms)
        self.store = store

    # -- definitions -------------------------------------------------------- #
    def definitions(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "name": "get_context",
                "description": (
                    "Return log lines from every subsystem in a time window of the run. Use it to "
                    "see what happened just before the first error, or between two events. "
                    "Telemetry (INFO) lines are included only when include_info is true."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_ms": {
                            "type": "integer",
                            "description": "Window start (ms since power-on).",
                        },
                        "end_ms": {
                            "type": "integer",
                            "description": "Window end (ms since power-on).",
                        },
                        "include_info": {
                            "type": "boolean",
                            "description": "Also return INFO telemetry.",
                        },
                        "subsystem": {
                            "type": "string",
                            "description": "Optional subsystem filter, e.g. PMU.",
                        },
                    },
                    "required": ["start_ms", "end_ms", "include_info", "subsystem"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "read_block",
                "description": (
                    "Return the full multi-line block (register dump, stack trace) attached to the "
                    "error event closest to the given timestamp."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ts_ms": {"type": "integer", "description": "Timestamp of the event."}
                    },
                    "required": ["ts_ms"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "lookup_catalogue",
                "description": (
                    "Return the known root causes for this failure type with the evidence markers "
                    "that betray each one, plus the generic prior and playbook."
                ),
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        ]
        if self.store is not None:
            tools.append(
                {
                    "name": "search_incidents",
                    "description": (
                        "Search the historical incident store with free text (e.g. a distinctive log "
                        "line) and return the most similar past failures with their resolved root causes."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "description": "1-10"},
                        },
                        "required": ["query", "top_k"],
                        "additionalProperties": False,
                    },
                }
            )
        for t in tools:
            t["strict"] = True
        return tools

    def handlers(self) -> dict[str, Callable[..., str]]:
        h: dict[str, Callable[..., str]] = {
            "get_context": self.get_context,
            "read_block": self.read_block,
            "lookup_catalogue": self.lookup_catalogue,
        }
        if self.store is not None:
            h["search_incidents"] = self.search_incidents
        return h

    # -- implementations ---------------------------------------------------- #
    def get_context(
        self, start_ms: int, end_ms: int, include_info: bool = False, subsystem: str = ""
    ) -> str:
        lo, hi = sorted((int(start_ms), int(end_ms)))
        rows = [
            e
            for e in self.events
            if lo <= e.ts_ms <= hi
            and (include_info or e.level is not LogLevel.INFO)
            and (not subsystem or e.subsys.upper() == subsystem.upper())
        ]
        if not rows:
            return f"No matching lines between {lo} and {hi} ms."
        text = "\n".join(_fmt(e) for e in rows[:200])
        if len(rows) > 200:
            text += f"\n... {len(rows) - 200} more lines omitted"
        return text[:MAX_RESULT_CHARS]

    def read_block(self, ts_ms: int) -> str:
        with_blocks = [e for e in self.events if e.multiline]
        if not with_blocks:
            return "No multi-line blocks in this run."
        ev = min(with_blocks, key=lambda e: abs(e.ts_ms - int(ts_ms)))
        body = "\n".join(ev.multiline)
        return f"{_fmt(ev)}\n{body}"[:MAX_RESULT_CHARS]

    def lookup_catalogue(self) -> str:
        prior = prior_for(self.failure.failure_type)
        lines = [f"Prior for {self.failure.failure_type}: {prior.root_cause}", prior.rationale, ""]
        cands = candidates_for(self.failure.failure_type)
        if not cands:
            lines.append("No catalogue entries for this failure type; rely on retrieved incidents.")
        for rc in cands:
            lines.append(f"- {rc.rc_id}: {rc.description}")
            lines.append(f"    tell-tales: {', '.join(rc.markers)}")
            lines.append(f"    actions: {'; '.join(rc.actions)}")
        return "\n".join(lines)[:MAX_RESULT_CHARS]

    def search_incidents(self, query: str, top_k: int = 5) -> str:
        assert self.store is not None
        hits = self.store.query(
            str(query),
            top_k=max(1, min(10, int(top_k))),
            exclude_failure_id=self.failure.failure_id,
        )
        if not hits:
            return "No similar incidents found."
        return "\n".join(
            f"- run {h.run_id} | {h.failure_type} | root_cause={h.root_cause_id or 'unknown'} "
            f"| similarity={h.similarity}\n    {h.signature[:220]}"
            for h in hits
        )[:MAX_RESULT_CHARS]

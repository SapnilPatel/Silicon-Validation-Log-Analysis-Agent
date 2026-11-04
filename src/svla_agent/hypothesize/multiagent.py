"""Multi-agent root-cause investigation, orchestrated with LangGraph.

    triage ──▶ specialist(PMU) ─┐
           ──▶ specialist(DDR) ─┼──▶ reviewer ──▶ END
           ──▶ specialist(...) ─┘      │ (one follow-up allowed)
                     ▲─────────────────┘

* **Triage** reads the failure summary and the subsystems that actually logged
  around it, and decides which specialists to dispatch and what each should
  check.
* **Specialists** run in parallel (LangGraph ``Send`` fan-out). Each has its own
  system prompt, its own slice of the root-cause catalogue and tools scoped to
  its subsystem's log lines; they report a structured finding, not a verdict.
* **Reviewer** cross-checks the findings against each other and the evidence,
  may send exactly one specialist back with a pointed question, then submits
  the final hypothesis with a causal chain and calibrated confidence.

Every model turn and tool call is recorded in the trace with the agent's name,
so the conversation between agents is inspectable in reports and the UI.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..config import Settings
from ..logging_config import get_logger
from ..models import Event, Failure, LogLevel, RetrievedIncident
from ..rag.base import BaseVectorStore
from .knowledge import ROOT_CAUSES, RootCause, candidates_for
from .tools import AgentTools

logger = get_logger(__name__)

# Subsystem -> the failure type whose catalogue that specialist owns.
SPECIALIST_DOMAIN: dict[str, str] = {
    "PCIE": "PCIE_TIMEOUT",
    "DDR": "DDR_ECC",
    "PMU": "PMU_DROOP",
    "THERMAL": "THERMAL_TRIP",
    "FW": "FW_EXCEPTION",
    "NOC": "NOC_TIMEOUT",
}
MAX_SPECIALISTS = 4


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #


class Finding(TypedDict):
    subsystem: str
    observation: str
    suspected_cause: str
    root_cause_id: str
    confidence: float
    evidence: list[str]


class InvestigationState(TypedDict, total=False):
    plan: dict[str, Any]  # {"specialists": [...], "questions": {subsys: str}}
    findings: Annotated[list[Finding], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    followup: dict[str, Any] | None  # {"subsystem": str, "question": str}
    followups_done: int
    final: dict[str, Any] | None


# --------------------------------------------------------------------------- #
# Prompts and tool schemas
# --------------------------------------------------------------------------- #

_TRIAGE_SYSTEM = (
    "You are the triage lead of a post-silicon validation debug team. Given a failure "
    "summary and the subsystems that logged around it, decide which specialists to "
    "dispatch (at most {max_n}) and give each one a specific question. Prefer the failing "
    "subsystem plus any subsystem that could have caused it (power and thermal are frequent "
    "upstream causes). Call plan_investigation."
)

_SPECIALIST_SYSTEM = (
    "You are the {subsys} specialist on a post-silicon validation debug team. You only see "
    "{subsys} log lines (plus register dumps and historical incidents). Investigate the "
    "triage lead's question with your tools, then call report_finding with what you "
    "observed, the cause you suspect from your subsystem's point of view, and the "
    "evidence lines that support it. Be specific; quote numbers. Do not speculate about "
    "subsystems you cannot see."
)

_REVIEWER_SYSTEM = (
    "You are the reviewer on a post-silicon validation debug team. You receive the "
    "specialists' findings for one failure. Cross-check them against each other and the "
    "error lines: do they agree, does one subsystem's event explain another's (a causal "
    "chain), is anything missing? If exactly one pointed question to one specialist would "
    "settle it, call request_followup. Otherwise call submit_hypothesis with the root cause, "
    "the causal chain, a confidence that reflects how well the findings agree, and the next "
    "experiments."
)


def _plan_tool(subsystems: list[str]) -> dict[str, Any]:
    return {
        "name": "plan_investigation",
        "description": "Dispatch specialists with one question each.",
        "input_schema": {
            "type": "object",
            "properties": {
                "specialists": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subsystem": {"type": "string", "enum": subsystems},
                            "question": {"type": "string"},
                        },
                        "required": ["subsystem", "question"],
                        "additionalProperties": False,
                    },
                },
                "rationale": {"type": "string"},
            },
            "required": ["specialists", "rationale"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _finding_tool(rc_ids: list[str]) -> dict[str, Any]:
    return {
        "name": "report_finding",
        "description": "Report what you found in your subsystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observation": {
                    "type": "string",
                    "description": "What the logs show, with numbers.",
                },
                "suspected_cause": {"type": "string"},
                "root_cause_id": {"type": "string", "enum": [*rc_ids, "OTHER", "NONE"]},
                "confidence": {"type": "number"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "observation",
                "suspected_cause",
                "root_cause_id",
                "confidence",
                "evidence",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _followup_tool(subsystems: list[str]) -> dict[str, Any]:
    return {
        "name": "request_followup",
        "description": "Send one specialist back with one pointed question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subsystem": {"type": "string", "enum": subsystems},
                "question": {"type": "string"},
            },
            "required": ["subsystem", "question"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _submit_tool(rc_ids: list[str]) -> dict[str, Any]:
    return {
        "name": "submit_hypothesis",
        "description": "Submit the team's final root-cause hypothesis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "root_cause_id": {"type": "string", "enum": [*rc_ids, "OTHER"]},
                "root_cause": {"type": "string"},
                "causal_chain": {
                    "type": "string",
                    "description": "e.g. PMU droop -> DDR UE -> memtest fail",
                },
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
                "recommended_actions": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "root_cause_id",
                "root_cause",
                "causal_chain",
                "confidence",
                "rationale",
                "recommended_actions",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


# --------------------------------------------------------------------------- #
# Scoped tools for a specialist
# --------------------------------------------------------------------------- #


class SpecialistTools(AgentTools):
    """AgentTools restricted to one subsystem's lines and catalogue slice."""

    def __init__(self, subsys: str, failure: Failure, events: list[Event], store):
        super().__init__(failure, [e for e in events if e.subsys.upper() == subsys.upper()], store)
        self.subsys = subsys.upper()

    def definitions(self) -> list[dict[str, Any]]:
        defs = super().definitions()
        for d in defs:
            if d["name"] == "get_context":
                d["description"] = (
                    f"Return {self.subsys} log lines in a time window (telemetry optional)."
                )
        return defs

    def get_context(
        self, start_ms: int, end_ms: int, include_info: bool = False, subsystem: str = ""
    ) -> str:
        return super().get_context(start_ms, end_ms, include_info, self.subsys)

    def lookup_catalogue(self) -> str:
        ftype = SPECIALIST_DOMAIN.get(self.subsys)
        cands: tuple[RootCause, ...] = candidates_for(ftype) if ftype else ()
        if not cands:
            return f"No catalogue entries owned by the {self.subsys} specialist."
        lines = [f"Root causes the {self.subsys} specialist can recognise:"]
        for rc in cands:
            lines.append(f"- {rc.rc_id}: {rc.description}  (tell-tales: {', '.join(rc.markers)})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The team
# --------------------------------------------------------------------------- #


class InvestigationTeam:
    """Builds and runs the LangGraph for one failure."""

    def __init__(
        self,
        provider: Any,
        settings: Settings,
        failure: Failure,
        similar: list[RetrievedIncident],
        events: list[Event] | None,
        store: BaseVectorStore | None,
        candidate_ids: list[str],
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.failure = failure
        self.similar = similar
        self.events = sorted(events or [], key=lambda e: e.ts_ms)
        self.store = store
        self.candidate_ids = candidate_ids
        self.subsystems = self._subsystems_near_failure()
        self.graph = self._build()

    # -- helpers ------------------------------------------------------------ #
    def _subsystems_near_failure(self) -> list[str]:
        lo = self.failure.first_ts_ms - self.settings.evidence_window_ms
        hi = self.failure.last_ts_ms
        seen: dict[str, int] = {}
        for e in self.events:
            if lo <= e.ts_ms <= hi and e.level is not LogLevel.INFO:
                seen[e.subsys.upper()] = seen.get(e.subsys.upper(), 0) + 1
        for e in self.events:  # subsystems with any lines at all, as a fallback
            seen.setdefault(e.subsys.upper(), 0)
        ordered = sorted(seen, key=lambda k: -seen[k])
        return [s for s in ordered if s != "SYSTEM"][:8] or ["SYSTEM"]

    def _rc_ids_for(self, subsys: str) -> list[str]:
        ftype = SPECIALIST_DOMAIN.get(subsys.upper())
        ids = [rc.rc_id for rc in (candidates_for(ftype) if ftype else ())]
        for i in self.candidate_ids:
            if i not in ids:
                ids.append(i)
        return ids

    def _lead_text(self) -> str:
        f = self.failure
        lines = [
            f"Run {f.run_id} | failure type {f.failure_type} | severity {f.severity.value}",
            f"Summary: {f.summary}",
            f"Error window: {f.first_ts_ms}-{f.last_ts_ms} ms; seen {f.occurrences}x in corpus",
            "Error lines:",
            *(f"  {e}" for e in f.evidence[-4:]),
        ]
        if self.similar:
            lines.append("Similar resolved incidents (from the historian):")
            lines += [
                f"  - run {s.run_id} | {s.failure_type} | root_cause={s.root_cause_id or 'unknown'} | sim={s.similarity}"
                for s in self.similar
            ]
        return "\n".join(lines)

    # -- nodes ---------------------------------------------------------------- #
    def triage(self, state: InvestigationState) -> dict[str, Any]:
        prompt = (
            self._lead_text()
            + "\n\nSubsystems with non-telemetry lines around the failure (most active first): "
            + ", ".join(self.subsystems)
            + "\n\nCall plan_investigation."
        )
        payload = self.provider.structured(
            system=_TRIAGE_SYSTEM.format(max_n=MAX_SPECIALISTS),
            prompt=prompt,
            tool=_plan_tool(self.subsystems),
        )
        specs = [s for s in payload.get("specialists", []) if s.get("subsystem") in self.subsystems]
        specs = specs[:MAX_SPECIALISTS] or [
            {"subsystem": self.subsystems[0], "question": "Investigate the failure."}
        ]
        plan = {
            "specialists": [s["subsystem"] for s in specs],
            "questions": {s["subsystem"]: s["question"] for s in specs},
            "rationale": payload.get("rationale", ""),
        }
        trace = [
            {
                "step": 0,
                "agent": "triage",
                "tool": "plan_investigation",
                "input": {},
                "output": "; ".join(f"{s['subsystem']}: {s['question']}" for s in specs),
                "note": payload.get("rationale", ""),
            }
        ]
        return {"plan": plan, "trace": trace, "findings": [], "followups_done": 0, "followup": None}

    def dispatch(self, state: InvestigationState) -> list[Send]:
        plan = state["plan"]
        return [
            Send(
                "specialist", {"subsystem": s, "question": plan["questions"][s], "followup": False}
            )
            for s in plan["specialists"]
        ]

    def specialist(self, payload: dict[str, Any]) -> dict[str, Any]:
        subsys, question = payload["subsystem"], payload["question"]
        tools = SpecialistTools(subsys, self.failure, self.events, self.store)
        prompt = (
            self._lead_text()
            + f"\n\nTriage lead's question for the {subsys} specialist: {question}"
            + (
                "\n\n(This is a follow-up request from the reviewer.)"
                if payload.get("followup")
                else ""
            )
            + "\n\nInvestigate with your tools, then call report_finding."
        )
        finding_tool = _finding_tool(self._rc_ids_for(subsys))
        result: dict[str, Any] | None = None
        raw_trace: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for attempt in range(2):  # one retry: a transient 4xx/5xx must not blank a finding
            try:
                result, raw_trace = self.provider.run_agent(
                    system=_SPECIALIST_SYSTEM.format(subsys=subsys),
                    prompt=prompt,
                    tools=tools.definitions(),
                    handlers=tools.handlers(),
                    final_tool=finding_tool,
                    max_turns=max(2, self.settings.agent_max_turns // 2),
                )
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("%s specialist attempt %d failed: %s", subsys, attempt + 1, exc)
        if result is None:  # one failed specialist must not sink the team
            result, raw_trace = (
                {
                    "observation": f"specialist failed: {last_exc}",
                    "suspected_cause": "",
                    "root_cause_id": "NONE",
                    "confidence": 0.0,
                    "evidence": [],
                },
                [],
            )
        finding: Finding = {
            "subsystem": subsys,
            "observation": str(result.get("observation", "")),
            "suspected_cause": str(result.get("suspected_cause", "")),
            "root_cause_id": str(result.get("root_cause_id", "NONE")),
            "confidence": float(result.get("confidence", 0.0)),
            "evidence": [str(e) for e in result.get("evidence", [])][:8],
        }
        trace = [{**t, "agent": f"specialist:{subsys}"} for t in raw_trace]
        trace.append(
            {
                "step": 0,
                "agent": f"specialist:{subsys}",
                "tool": "report_finding",
                "input": {},
                "output": f"{finding['root_cause_id']} ({finding['confidence']:.2f}): {finding['observation'][:600]}",
                "note": finding["suspected_cause"][:300],
            }
        )
        return {"findings": [finding], "trace": trace}

    def reviewer(self, state: InvestigationState) -> dict[str, Any]:
        findings = state.get("findings", [])
        lines = [self._lead_text(), "", "Specialist findings:"]
        for f in findings:
            lines.append(
                f"- [{f['subsystem']}] root_cause={f['root_cause_id']} conf={f['confidence']:.2f}\n"
                f"    observation: {f['observation']}\n    suspected: {f['suspected_cause']}\n"
                f"    evidence: {' | '.join(f['evidence'][:4])}"
            )
        can_followup = state.get("followups_done", 0) < 1
        lines.append(
            "\nCall submit_hypothesis"
            + (
                " (or request_followup for one specialist if one question would settle it)."
                if can_followup
                else "."
            )
        )
        subs = sorted({f["subsystem"] for f in findings}) or self.subsystems
        tools = [_submit_tool(self.candidate_ids)] + (
            [_followup_tool(subs)] if can_followup else []
        )
        response = self.provider.client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            system=_REVIEWER_SYSTEM,
            tools=tools,
            tool_choice={"type": "auto"},
            output_config={"effort": self.settings.llm_effort},
            messages=[{"role": "user", "content": "\n".join(lines)}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("reviewer declined (stop_reason=refusal)")
        blocks = list(getattr(response, "content", []))
        note = " ".join(
            getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"
        ).strip()
        uses = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
        follow = next((b for b in uses if b.name == "request_followup"), None)
        submit = next((b for b in uses if b.name == "submit_hypothesis"), None)
        if follow is not None and can_followup and submit is None:
            fu = dict(follow.input)
            return {
                "followup": fu,
                "followups_done": state.get("followups_done", 0) + 1,
                "trace": [
                    {
                        "step": 0,
                        "agent": "reviewer",
                        "tool": "request_followup",
                        "input": fu,
                        "output": "",
                        "note": note,
                    }
                ],
            }
        if submit is None:
            raise ValueError("reviewer did not submit a hypothesis")
        final = dict(submit.input)
        return {
            "final": final,
            "followup": None,
            "trace": [
                {
                    "step": 0,
                    "agent": "reviewer",
                    "tool": "submit_hypothesis",
                    "input": {},
                    "output": f"{final.get('root_cause_id')} ({float(final.get('confidence', 0)):.2f}) | chain: {final.get('causal_chain', '')}",
                    "note": note,
                }
            ],
        }

    def after_review(self, state: InvestigationState) -> list[Send] | str:
        fu = state.get("followup")
        if fu:
            return [
                Send(
                    "specialist",
                    {"subsystem": fu["subsystem"], "question": fu["question"], "followup": True},
                )
            ]
        return END

    # -- graph ---------------------------------------------------------------- #
    def _build(self):
        g: StateGraph = StateGraph(InvestigationState)
        g.add_node("triage", self.triage)
        g.add_node("specialist", self.specialist)
        g.add_node("reviewer", self.reviewer)
        g.add_edge(START, "triage")
        g.add_conditional_edges("triage", self.dispatch, ["specialist"])
        g.add_edge("specialist", "reviewer")
        g.add_conditional_edges("reviewer", self.after_review, ["specialist", END])
        return g.compile()

    def run(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[Finding]]:
        state = self.graph.invoke({"findings": [], "trace": [], "followups_done": 0})
        final = state.get("final")
        if not final:
            raise ValueError("investigation ended without a hypothesis")
        trace = [{**t, "step": i + 1} for i, t in enumerate(state.get("trace", []))]
        return final, trace, state.get("findings", [])


def run_team(
    provider: Any,
    settings: Settings,
    failure: Failure,
    similar: list[RetrievedIncident],
    events: list[Event] | None,
    store: BaseVectorStore | None,
    candidate_ids: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Finding]]:
    """Convenience entry point used by :class:`HypothesisAgent`."""
    return InvestigationTeam(
        provider, settings, failure, similar, events, store, candidate_ids
    ).run()


__all__ = ["SPECIALIST_DOMAIN", "InvestigationTeam", "SpecialistTools", "run_team"]
_ = ROOT_CAUSES  # catalogue is consulted via candidates_for; keep the import explicit

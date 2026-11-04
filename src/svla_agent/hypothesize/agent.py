"""Root-cause hypothesis agent.

Given a classified failure, similar historical incidents (RAG) and the domain
knowledge base, produce a :class:`Hypothesis` that names a *specific* root
cause id (``RC_*``) plus confidence, rationale and remediation steps.

Two backends share one output shape:

* **LLM** - Claude (or any :class:`~svla_agent.hypothesize.llm.LLMProvider`) is
  asked, via a forced tool call, to pick among the known candidates for the
  failure type using the evidence and the retrieved incidents.
* **Rule-based** - deterministic scoring of the same candidates: evidence
  markers from the knowledge base count most, similarity-weighted votes from
  retrieved incidents second, the type prior last. Used when no key is
  configured, when offline, or when the LLM call fails.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..config import Settings, get_settings
from ..logging_config import get_logger
from ..models import Event, Failure, Hypothesis, RetrievedIncident, ToolCall
from ..rag.base import BaseVectorStore
from .knowledge import RootCause, candidates_for, prior_for, root_cause_by_id
from .llm import LLMProvider, make_provider
from .multiagent import run_team
from .tools import AgentTools

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior post-silicon validation engineer. Given evidence from a "
    "failing silicon test run, a catalogue of known root causes for this failure "
    "type, and similar historical incidents, pick the single most likely root cause. "
    "Prefer the candidate whose tell-tale evidence actually appears in the log lines; "
    "use historical incidents to break ties. Be concrete and actionable, and ground "
    "your confidence in how well evidence and history agree."
)

_AGENT_SYSTEM_PROMPT = (
    "You are a senior post-silicon validation engineer investigating one failure. You have "
    "tools: get_context (log lines in a time window, any subsystem), read_block (full register "
    "dump / stack trace), lookup_catalogue (known root causes and their tell-tales) and, when "
    "available, search_incidents (similar resolved failures). The summary you are given is "
    "deliberately partial: look before you conclude. Typical investigation: check the "
    "catalogue, pull the window before the first error across all subsystems, read any dump, "
    "cross-check history, then call submit_hypothesis with a specific root cause, a calibrated "
    "confidence and concrete next experiments. Do not call more tools than you need."
)

_EVIDENCE_WEIGHT = 1.0  # a matched marker
_EXTRA_MARKER_WEIGHT = 0.25  # each additional marker of the same candidate
_RAG_WEIGHT = 0.35  # per agreeing incident, scaled by similarity


def _candidate_ids(
    candidates: tuple[RootCause, ...], similar: list[RetrievedIncident]
) -> list[str]:
    """Catalogue ids first, then root causes seen in retrieved incidents.

    On a domain the catalogue does not cover (e.g. real BGL logs) the retrieved
    incidents' labels are the only nameable answers - the rule-based agent votes
    on them, so the LLM must be allowed to choose them too.
    """
    ids: list[str] = [c.rc_id for c in candidates]
    for inc in similar:
        if inc.root_cause_id and inc.root_cause_id not in ids:
            ids.append(inc.root_cause_id)
    return ids


def _tool_schema(ids: list[str]) -> dict[str, Any]:
    ids = [*ids, "OTHER"]
    return {
        "name": "submit_hypothesis",
        "description": "Submit the structured root-cause hypothesis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "root_cause_id": {
                    "type": "string",
                    "enum": ids,
                    "description": "Catalogue id of the chosen root cause (OTHER if none fits).",
                },
                "root_cause": {"type": "string", "description": "One-sentence root cause."},
                "confidence": {
                    "type": "number",
                    "description": "0.0-1.0 calibrated confidence.",
                },
                "rationale": {"type": "string", "description": "Why, citing the evidence."},
                "recommended_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 concrete next steps.",
                },
            },
            "required": [
                "root_cause_id",
                "root_cause",
                "confidence",
                "rationale",
                "recommended_actions",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


class HypothesisAgent:
    """Generate root-cause hypotheses for failures."""

    def __init__(self, settings: Settings | None = None, provider: LLMProvider | None = None):
        self.settings = settings or get_settings()
        self._provider = provider
        self._provider_resolved = provider is not None

    @property
    def provider(self) -> LLMProvider | None:
        if not self._provider_resolved:
            self._provider = make_provider(self.settings)
            self._provider_resolved = True
        return self._provider

    @property
    def agentic(self) -> bool:
        """True when the tool-using loop is configured and an LLM provider is active."""
        return (
            self.provider is not None
            and (self.settings.llm_mode or "direct").lower() == "agentic"
            and hasattr(self.provider, "run_agent")
        )

    @property
    def multiagent(self) -> bool:
        """True when the LangGraph team (triage -> specialists -> reviewer) is configured."""
        return (
            self.provider is not None
            and (self.settings.llm_mode or "direct").lower() in {"multiagent", "multi-agent"}
            and hasattr(self.provider, "run_agent")
        )

    @property
    def needs_events(self) -> bool:
        return self.agentic or self.multiagent

    @property
    def backend(self) -> str:
        if not self.provider:
            return "rule-based"
        suffix = ":multiagent" if self.multiagent else (":agentic" if self.agentic else "")
        return f"llm:{self.provider.name}{suffix}"

    # -- public API --------------------------------------------------------- #
    def generate(
        self,
        failure: Failure,
        similar: list[RetrievedIncident] | None = None,
        cluster_id: int | None = None,
        *,
        events: list[Event] | None = None,
        store: BaseVectorStore | None = None,
    ) -> Hypothesis:
        """Hypothesise a root cause.

        ``events`` (the run's parsed events) and ``store`` give the agentic loop
        something to investigate; the direct and rule-based paths ignore them.
        """
        similar = similar or []
        if self.provider is not None:
            try:
                if self.multiagent:
                    return self._multiagent_hypothesis(failure, similar, cluster_id, events, store)
                if self.agentic:
                    return self._agentic_hypothesis(failure, similar, cluster_id, events, store)
                return self._llm_hypothesis(failure, similar, cluster_id)
            except Exception as exc:  # pragma: no cover - network/SDK dependent
                logger.warning("LLM hypothesis failed (%s); using rule-based fallback", exc)
        return self._rule_based(failure, similar, cluster_id)

    # -- shared scoring ----------------------------------------------------- #
    @staticmethod
    def _evidence_text(failure: Failure) -> str:
        parts = [*failure.evidence, failure.summary, failure.signature]
        parts += [f"{k}={v}" for k, v in failure.kv.items()]
        return "\n".join(parts).lower()

    def score_candidates(
        self, failure: Failure, similar: list[RetrievedIncident]
    ) -> tuple[dict[str, float], dict[str, list[str]], dict[str, float]]:
        """Return (score, matched-markers, rag-votes) per candidate root-cause id."""
        text = self._evidence_text(failure)
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, list[str]] = defaultdict(list)
        votes: dict[str, float] = defaultdict(float)

        for rc in candidates_for(failure.failure_type):
            hits = [m for m in rc.markers if m.lower() in text]
            if hits:
                scores[rc.rc_id] += _EVIDENCE_WEIGHT + _EXTRA_MARKER_WEIGHT * (len(hits) - 1)
                matched[rc.rc_id] = hits
        for inc in similar:
            if inc.failure_type == failure.failure_type and inc.root_cause_id:
                w = _RAG_WEIGHT * max(0.0, inc.similarity)
                scores[inc.root_cause_id] += w
                votes[inc.root_cause_id] += w
        return dict(scores), dict(matched), dict(votes)

    # -- LLM path ----------------------------------------------------------- #
    def _build_prompt(self, failure: Failure, similar: list[RetrievedIncident]) -> str:
        prior = prior_for(failure.failure_type)
        candidates = candidates_for(failure.failure_type)
        lines = [
            f"Run: {failure.run_id}",
            f"Failure type: {failure.failure_type}  (severity: {failure.severity.value})",
            f"Subsystem(s): {failure.subsys}",
            f"Summary: {failure.summary}",
            f"Seen {failure.occurrences}x in the analysed corpus (fingerprint {failure.fingerprint})",
            "",
            "Evidence (chronological):",
            *(f"  {e}" for e in failure.evidence),
            "",
            f"Domain prior for {failure.failure_type}: {prior.root_cause}",
        ]
        if candidates:
            lines += ["", "Known root-cause candidates for this failure type:"]
            for rc in candidates:
                lines.append(
                    f"  - {rc.rc_id}: {rc.description} (tell-tales: {', '.join(rc.markers)})"
                )
        else:
            lines += [
                "",
                "No catalogue exists for this failure type. The root-cause ids attached to the "
                "historical incidents below are the valid answers; pick the one whose incidents "
                "best match this evidence, or OTHER if none fits.",
            ]
        if similar:
            lines += ["", "Similar historical incidents (most similar first):"]
            for inc in similar:
                rc_id = inc.root_cause_id or "unknown"
                lines.append(
                    f"  - run {inc.run_id} | {inc.failure_type} | root_cause={rc_id} "
                    f"| similarity={inc.similarity}"
                )
        lines += ["", "Call submit_hypothesis with your conclusion."]
        return "\n".join(lines)

    def _llm_hypothesis(
        self, failure: Failure, similar: list[RetrievedIncident], cluster_id: int | None
    ) -> Hypothesis:
        assert self.provider is not None
        candidates = candidates_for(failure.failure_type)
        ids = _candidate_ids(candidates, similar)
        payload = self.provider.structured(
            system=_SYSTEM_PROMPT,
            prompt=self._build_prompt(failure, similar),
            tool=_tool_schema(ids),
        )
        valid_ids = set(ids)
        rc_id = payload.get("root_cause_id")
        rc_id = str(rc_id) if rc_id in valid_ids else None
        confidence = float(payload.get("confidence", 0.5))
        return Hypothesis(
            failure_id=failure.failure_id,
            cluster_id=cluster_id if cluster_id is not None else failure.cluster_id,
            root_cause=str(payload["root_cause"]),
            root_cause_id=rc_id,
            confidence=min(1.0, max(0.0, confidence)),
            rationale=str(payload.get("rationale", "")),
            recommended_actions=[str(a) for a in payload.get("recommended_actions", [])][:5],
            supporting_evidence=failure.evidence,
            similar_incidents=similar,
            source=f"llm:{self.settings.llm_model}",
        )

    # -- agentic path ------------------------------------------------------- #
    def _agentic_hypothesis(
        self,
        failure: Failure,
        similar: list[RetrievedIncident],
        cluster_id: int | None,
        events: list[Event] | None,
        store: BaseVectorStore | None,
    ) -> Hypothesis:
        assert self.provider is not None and hasattr(self.provider, "run_agent")
        tools = AgentTools(failure, events, store)
        candidates = candidates_for(failure.failure_type)
        ids = _candidate_ids(candidates, similar)
        lead = (
            f"Run: {failure.run_id}\n"
            f"Failure type: {failure.failure_type}  (severity: {failure.severity.value})\n"
            f"Subsystem(s): {failure.subsys}\n"
            f"Summary: {failure.summary}\n"
            f"Error window: {failure.first_ts_ms}-{failure.last_ts_ms} ms "
            f"(run has {len(events or [])} parsed events available to get_context)\n"
            f"Seen {failure.occurrences}x in the corpus.\n\n"
            "First and last error lines:\n" + "\n".join(f"  {e}" for e in failure.evidence[-4:])
        )
        if similar:
            lead += (
                "\n\nSimilar historical incidents already retrieved for the failure signature:\n"
            )
            lead += "\n".join(
                f"  - run {inc.run_id} | {inc.failure_type} | root_cause={inc.root_cause_id or 'unknown'} "
                f"| similarity={inc.similarity}"
                for inc in similar
            )
        lead += "\n\nInvestigate, then call submit_hypothesis."
        payload, raw_trace = self.provider.run_agent(
            system=_AGENT_SYSTEM_PROMPT,
            prompt=lead,
            tools=tools.definitions(),
            handlers=tools.handlers(),
            final_tool=_tool_schema(ids),
            max_turns=self.settings.agent_max_turns,
        )
        valid_ids = set(ids)
        rc_id = payload.get("root_cause_id")
        rc_id = str(rc_id) if rc_id in valid_ids else None
        confidence = float(payload.get("confidence", 0.5))
        return Hypothesis(
            failure_id=failure.failure_id,
            cluster_id=cluster_id if cluster_id is not None else failure.cluster_id,
            root_cause=str(payload["root_cause"]),
            root_cause_id=rc_id,
            confidence=min(1.0, max(0.0, confidence)),
            rationale=str(payload.get("rationale", "")),
            recommended_actions=[str(a) for a in payload.get("recommended_actions", [])][:5],
            supporting_evidence=failure.evidence,
            similar_incidents=similar,
            source=f"llm:{self.settings.llm_model}:agentic",
            trace=[ToolCall(**t) for t in raw_trace],
        )

    # -- multi-agent path --------------------------------------------------- #
    def _multiagent_hypothesis(
        self,
        failure: Failure,
        similar: list[RetrievedIncident],
        cluster_id: int | None,
        events: list[Event] | None,
        store: BaseVectorStore | None,
    ) -> Hypothesis:
        assert self.provider is not None
        ids = _candidate_ids(candidates_for(failure.failure_type), similar)
        payload, raw_trace, findings = run_team(
            self.provider, self.settings, failure, similar, events, store, ids
        )
        valid_ids = set(ids)
        rc_id = payload.get("root_cause_id")
        rc_id = str(rc_id) if rc_id in valid_ids else None
        confidence = float(payload.get("confidence", 0.5))
        rationale = str(payload.get("rationale", ""))
        chain = str(payload.get("causal_chain", "")).strip()
        if chain:
            rationale = f"Causal chain: {chain}. {rationale}"
        agreeing = sum(1 for f in findings if rc_id and f.get("root_cause_id") == rc_id)
        rationale += f" ({agreeing} of {len(findings)} specialists pointed at this cause.)"
        return Hypothesis(
            failure_id=failure.failure_id,
            cluster_id=cluster_id if cluster_id is not None else failure.cluster_id,
            root_cause=str(payload["root_cause"]),
            root_cause_id=rc_id,
            confidence=min(1.0, max(0.0, confidence)),
            rationale=rationale,
            recommended_actions=[str(a) for a in payload.get("recommended_actions", [])][:5],
            supporting_evidence=failure.evidence,
            similar_incidents=similar,
            source=f"llm:{self.settings.llm_model}:multiagent",
            trace=[ToolCall(**t) for t in raw_trace],
        )

    # -- rule-based path ---------------------------------------------------- #
    def _rule_based(
        self, failure: Failure, similar: list[RetrievedIncident], cluster_id: int | None
    ) -> Hypothesis:
        prior = prior_for(failure.failure_type)
        scores, matched, votes = self.score_candidates(failure, similar)

        rc_id: str | None = None
        rc: RootCause | None = None
        if scores:
            rc_id = max(scores, key=lambda k: (scores[k], k))
            rc = root_cause_by_id(rc_id)

        same_type = [s for s in similar if s.failure_type == failure.failure_type]
        agree = [s for s in same_type if rc_id and s.root_cause_id == rc_id]
        agree_share = len(agree) / len(same_type) if same_type else 0.0

        # Confidence: evidence marker >> retrieval vote >> bare prior.
        if rc_id and matched.get(rc_id):
            confidence = 0.8 + 0.12 * agree_share + 0.03 * min(2, len(matched[rc_id]) - 1)
        elif rc_id and votes.get(rc_id):
            confidence = 0.45 + 0.35 * agree_share
        else:
            confidence = 0.35
        confidence = round(min(0.97, confidence), 3)

        # Narrative pieces.
        why: list[str] = [prior.rationale]
        if rc is not None and matched.get(rc_id or ""):
            quoted = ", ".join(f"'{m}'" for m in matched[rc_id or ""])
            why.append(f"Evidence contains the tell-tale(s) {quoted} for {rc.rc_id}.")
        if agree:
            runs = ", ".join(a.run_id for a in agree[:3])
            why.append(
                f"{len(agree)} of {len(same_type)} similar historical incidents "
                f"(runs {runs}) were resolved as {rc_id}."
            )
        elif same_type and rc_id and not matched.get(rc_id):
            why.append("Retrieved incidents disagree on the specific cause; treat as tentative.")
        if rc is None:
            why.append("No specific tell-tale found; falling back to the failure-type prior.")

        root_cause = f"{prior.root_cause}: {rc.description}" if rc else prior.root_cause
        actions: list[str] = []
        for a in [*(rc.actions if rc else ()), *prior.actions]:
            if a not in actions:
                actions.append(a)

        return Hypothesis(
            failure_id=failure.failure_id,
            cluster_id=cluster_id if cluster_id is not None else failure.cluster_id,
            root_cause=root_cause,
            root_cause_id=rc_id,
            confidence=confidence,
            rationale=" ".join(why),
            recommended_actions=actions[:4],
            supporting_evidence=failure.evidence,
            similar_incidents=similar,
            source="rule-based",
        )

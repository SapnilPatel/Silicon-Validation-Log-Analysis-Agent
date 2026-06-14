"""Root-cause hypothesis agent.

Given a classified failure, optional similar historical incidents (RAG) and the
domain prior, produce a :class:`Hypothesis`. Uses Claude when an API key is
configured; otherwise falls back to a deterministic rule-based hypothesis so the
pipeline is always runnable and CI stays offline.
"""

from __future__ import annotations

import json

from ..config import Settings, get_settings
from ..logging_config import get_logger
from ..models import Failure, Hypothesis, RetrievedIncident
from .knowledge import prior_for

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior post-silicon validation engineer. Given evidence from a "
    "failing silicon test run and similar historical incidents, produce a single, "
    "specific root-cause hypothesis. Be concrete and actionable; prefer hardware/"
    "firmware mechanisms over generic advice. Ground your confidence in how well the "
    "evidence and retrieved incidents agree."
)

_TOOL = {
    "name": "submit_hypothesis",
    "description": "Submit the structured root-cause hypothesis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "root_cause": {"type": "string", "description": "One-sentence root cause."},
            "confidence": {"type": "number", "description": "0.0-1.0 calibrated confidence."},
            "rationale": {"type": "string", "description": "Why, citing the evidence."},
            "recommended_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-4 concrete next steps.",
            },
        },
        "required": ["root_cause", "confidence", "rationale", "recommended_actions"],
    },
}


class HypothesisAgent:
    """Generate root-cause hypotheses for failures."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    # -- public API --------------------------------------------------------- #
    def generate(
        self,
        failure: Failure,
        similar: list[RetrievedIncident] | None = None,
        cluster_id: int | None = None,
    ) -> Hypothesis:
        similar = similar or []
        if self.settings.llm_enabled:
            try:
                return self._llm_hypothesis(failure, similar, cluster_id)
            except Exception as exc:  # pragma: no cover - network/SDK dependent
                logger.warning("LLM hypothesis failed (%s); using rule-based fallback", exc)
        return self._rule_based(failure, similar, cluster_id)

    # -- LLM path ----------------------------------------------------------- #
    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def _build_prompt(self, failure: Failure, similar: list[RetrievedIncident]) -> str:
        prior = prior_for(failure.failure_type)
        lines = [
            f"Run: {failure.run_id}",
            f"Failure type: {failure.failure_type}  (severity: {failure.severity.value})",
            f"Subsystem(s): {failure.subsys}",
            f"Summary: {failure.summary}",
            "",
            "Evidence (chronological):",
            *(f"  {e}" for e in failure.evidence),
            "",
            f"Domain prior for {failure.failure_type}: {prior.root_cause}",
        ]
        if similar:
            lines += ["", "Similar historical incidents (most similar first):"]
            for inc in similar:
                rc = inc.root_cause_id or "unknown"
                lines.append(
                    f"  - run {inc.run_id} | {inc.failure_type} | root_cause={rc} "
                    f"| similarity={inc.similarity}"
                )
        lines += ["", "Call submit_hypothesis with your conclusion."]
        return "\n".join(lines)

    def _llm_hypothesis(
        self, failure: Failure, similar: list[RetrievedIncident], cluster_id: int | None
    ) -> Hypothesis:
        response = self.client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            system=_SYSTEM_PROMPT,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "submit_hypothesis"},
            messages=[{"role": "user", "content": self._build_prompt(failure, similar)}],
        )
        payload = _extract_tool_input(response)
        confidence = float(payload.get("confidence", 0.5))
        return Hypothesis(
            failure_id=failure.failure_id,
            cluster_id=cluster_id if cluster_id is not None else failure.cluster_id,
            root_cause=str(payload["root_cause"]),
            confidence=min(1.0, max(0.0, confidence)),
            rationale=str(payload.get("rationale", "")),
            recommended_actions=[str(a) for a in payload.get("recommended_actions", [])],
            supporting_evidence=failure.evidence,
            similar_incidents=similar,
            source="llm",
        )

    # -- rule-based path ---------------------------------------------------- #
    def _rule_based(
        self, failure: Failure, similar: list[RetrievedIncident], cluster_id: int | None
    ) -> Hypothesis:
        prior = prior_for(failure.failure_type)
        # Confidence: a strong prior, nudged up when retrieved incidents agree.
        confidence = 0.55
        agree = [s for s in similar if s.failure_type == failure.failure_type]
        if agree:
            confidence = min(0.95, 0.6 + 0.1 * len(agree) + 0.2 * agree[0].similarity)
        root_cause = prior.root_cause
        top_rc = next((s.root_cause_id for s in agree if s.root_cause_id), None)
        if top_rc:
            root_cause = f"{prior.root_cause} (matches historical {top_rc})"
        return Hypothesis(
            failure_id=failure.failure_id,
            cluster_id=cluster_id if cluster_id is not None else failure.cluster_id,
            root_cause=root_cause,
            confidence=round(confidence, 3),
            rationale=prior.rationale,
            recommended_actions=list(prior.actions),
            supporting_evidence=failure.evidence,
            similar_incidents=similar,
            source="rule-based",
        )


def _extract_tool_input(response) -> dict:
    """Pull the tool-use input out of an Anthropic message response."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
        # Defensive: some SDK paths may return JSON text.
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except (ValueError, AttributeError):
                continue
    raise ValueError("No tool_use block in LLM response")

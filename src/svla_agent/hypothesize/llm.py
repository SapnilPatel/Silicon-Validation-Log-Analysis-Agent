"""Pluggable LLM providers for the hypothesis agent.

A provider turns (system prompt, user prompt, tool schema) into the tool's
structured input dict. Claude (Anthropic) ships; adding another vendor means
implementing :class:`LLMProvider` and registering it in :data:`PROVIDERS`.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from ..config import Settings


class LLMProvider(Protocol):
    name: str

    def structured(self, *, system: str, prompt: str, tool: dict[str, Any]) -> dict[str, Any]:
        """Return the structured tool input produced by the model."""
        ...


class AnthropicProvider:
    """Claude via the official SDK, forcing a tool call for structured output."""

    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def structured(self, *, system: str, prompt: str, tool: dict[str, Any]) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_tool_input(response)


def extract_tool_input(response: Any) -> dict[str, Any]:
    """Pull the tool-use input out of an Anthropic-style message response."""
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
        if getattr(block, "type", None) == "text":  # defensive: JSON in text
            try:
                return json.loads(block.text)
            except (ValueError, AttributeError):
                continue
    raise ValueError("No tool_use block in LLM response")


PROVIDERS: dict[str, type] = {"anthropic": AnthropicProvider}


def make_provider(settings: Settings) -> LLMProvider | None:
    """Return the configured provider, or ``None`` when the LLM path is disabled."""
    if not settings.llm_enabled:
        return None
    cls = PROVIDERS.get((settings.llm_provider or "anthropic").lower())
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider '{settings.llm_provider}'. Known: {sorted(PROVIDERS)}"
        )
    return cls(settings)

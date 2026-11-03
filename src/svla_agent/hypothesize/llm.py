"""Pluggable LLM providers for the hypothesis agent.

A provider turns (system prompt, user prompt, tool schema) into the tool's
structured input dict. Claude (Anthropic) ships; adding another vendor means
implementing :class:`LLMProvider` and registering it in :data:`PROVIDERS`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
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

            # A hung request must fail fast and be retried, never stall a batch for hours.
            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key,
                timeout=self.settings.llm_timeout_s,
                max_retries=3,
            )
        return self._client

    def structured(self, *, system: str, prompt: str, tool: dict[str, Any]) -> dict[str, Any]:
        # Current Claude models reject forced tool_choice and sampling params, so the
        # tool is strict (schema-exact arguments) with tool_choice "auto", and the
        # prompt itself asks for the call. Thinking stays on; effort controls depth.
        strict_tool = {**tool, "strict": True}
        strict_tool["input_schema"] = {**tool["input_schema"], "additionalProperties": False}
        response = self.client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            system=system,
            tools=[strict_tool],
            tool_choice={"type": "auto"},
            output_config={"effort": self.settings.llm_effort},
            messages=[{"role": "user", "content": prompt}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("model declined the request (stop_reason=refusal)")
        return extract_tool_input(response)

    def run_agent(
        self,
        *,
        system: str,
        prompt: str,
        tools: list[dict[str, Any]],
        handlers: dict[str, Callable[..., str]],
        final_tool: dict[str, Any],
        max_turns: int = 8,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Tool-using loop: let the model investigate with ``tools`` until it calls
        ``final_tool``. Returns ``(final_tool_input, trace)``.

        Every tool_result for a turn goes back in one user message; a turn that
        runs out of budget gets a nudge to answer; if the model still does not
        answer, the loop asks once more with only the final tool available.
        """
        all_tools = [*tools, {**final_tool, "strict": True}]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        trace: list[dict[str, Any]] = []
        step = 0
        for turn in range(max_turns + 1):
            last_turn = turn >= max_turns
            response = self.client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                system=system,
                tools=[all_tools[-1]] if last_turn else all_tools,
                tool_choice={"type": "auto"},
                output_config={"effort": self.settings.llm_effort},
                messages=messages,
            )
            if getattr(response, "stop_reason", None) == "refusal":
                raise RuntimeError("model declined the request (stop_reason=refusal)")
            blocks = list(getattr(response, "content", []))
            note = " ".join(
                getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"
            ).strip()
            tool_uses = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
            final = next((b for b in tool_uses if b.name == final_tool["name"]), None)
            if final is not None:
                return dict(final.input), trace
            if not tool_uses or response.stop_reason != "tool_use":
                # No tool call: ask explicitly for the final answer.
                messages.append({"role": "assistant", "content": blocks})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Call {final_tool['name']} now with your conclusion.",
                    }
                )
                continue
            messages.append({"role": "assistant", "content": blocks})
            results: list[dict[str, Any]] = []
            for tu in tool_uses:
                step += 1
                handler = handlers.get(tu.name)
                args = dict(tu.input) if isinstance(tu.input, dict) else {}
                try:
                    out = handler(**args) if handler else f"unknown tool {tu.name}"
                    is_error = handler is None
                except Exception as exc:  # tool bug must not kill the loop
                    out, is_error = f"tool error: {type(exc).__name__}: {exc}", True
                trace.append(
                    {
                        "step": step,
                        "tool": tu.name,
                        "input": args,
                        "output": out[:1500],
                        "note": note,
                    }
                )
                note = ""
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": out,
                        "is_error": is_error,
                    }
                )
            if turn == max_turns - 1:
                results.append(
                    {
                        "type": "text",
                        "text": f"Investigation budget is spent; call {final_tool['name']} now.",
                    }
                )
            messages.append({"role": "user", "content": results})
        raise ValueError(f"agent did not call {final_tool['name']} within {max_turns} turns")


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

"""The tool-using hypothesis loop, driven by a scripted fake Anthropic client."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from svla_agent.config import Settings
from svla_agent.hypothesize.agent import HypothesisAgent
from svla_agent.hypothesize.llm import AnthropicProvider
from svla_agent.hypothesize.tools import AgentTools
from svla_agent.models import Event, Failure, LogLevel, Severity


def _event(ts, subsys, level, msg, multiline=None):
    return Event(
        ts_ms=ts,
        source=subsys.lower(),
        subsys=subsys,
        level=LogLevel(level),
        message=msg,
        multiline=multiline or [],
        raw=f"[{ts}] {subsys}: {level}: {msg}",
    )


@pytest.fixture()
def run_events():
    return [
        _event(100, "PMU", "INFO", "Rail telemetry"),
        _event(900, "THERMAL", "WARN", "Fan speed below profile | rpm=1200 target_rpm=3800"),
        _event(1500, "THERMAL", "ERROR", "Thermal trip threshold exceeded | t_c=110"),
        _event(
            1510,
            "SYSTEM",
            "FATAL",
            "Emergency shutdown",
            ["REG_DUMP_BEGIN", "40000000: DEADBEEF", "REG_DUMP_END"],
        ),
    ]


@pytest.fixture()
def failure():
    return Failure(
        failure_id="000001:THERMAL_TRIP",
        run_id="000001",
        failure_type="THERMAL_TRIP",
        severity=Severity.HIGH,
        subsys="SYSTEM",
        first_ts_ms=1500,
        last_ts_ms=1510,
        summary="THERMAL_TRIP in SYSTEM/THERMAL: Emergency shutdown",
        signature="THERMAL_TRIP | x",
        evidence=["[1500] THERMAL: ERROR: Thermal trip threshold exceeded"],
    )


# -- tools ------------------------------------------------------------------ #


def test_tools_context_block_and_catalogue(failure, run_events):
    tools = AgentTools(failure, run_events)
    names = {t["name"] for t in tools.definitions()}
    assert names == {"get_context", "read_block", "lookup_catalogue"}  # no store -> no search
    assert all(t["strict"] for t in tools.definitions())
    ctx = tools.get_context(0, 1400, include_info=False)
    assert "Fan speed below profile" in ctx and "Rail telemetry" not in ctx
    assert "Rail telemetry" in tools.get_context(0, 1400, include_info=True)
    assert "No matching" in tools.get_context(5000, 6000, include_info=True, subsystem="PCIE")
    assert "REG_DUMP_BEGIN" in tools.read_block(1505)
    cat = tools.lookup_catalogue()
    assert "RC_FAN_PROFILE_BAD" in cat and "fan speed below profile" in cat


# -- loop ------------------------------------------------------------------- #


class ScriptedClient:
    """Returns pre-built responses in order and records the requests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))  # snapshot: the loop mutates its list
        return self._responses.pop(0)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _use(name, inp, i):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=f"tu_{i}")


def _resp(blocks, stop="tool_use"):
    return SimpleNamespace(content=blocks, stop_reason=stop)


def test_agentic_loop_investigates_then_submits(settings: Settings, failure, run_events):
    llm_settings = settings.model_copy(
        update={
            "anthropic_api_key": "k",
            "offline": False,
            "llm_mode": "agentic",
            "agent_max_turns": 4,
        }
    )
    provider = AnthropicProvider(llm_settings)
    client = ScriptedClient(
        [
            _resp(
                [
                    _text("Checking the catalogue and the window before the trip."),
                    _use("lookup_catalogue", {}, 1),
                    _use(
                        "get_context",
                        {"start_ms": 0, "end_ms": 1500, "include_info": False, "subsystem": ""},
                        2,
                    ),
                ]
            ),
            _resp([_use("read_block", {"ts_ms": 1510}, 3)]),
            _resp(
                [
                    _use(
                        "submit_hypothesis",
                        {
                            "root_cause_id": "RC_FAN_PROFILE_BAD",
                            "root_cause": "Fan ran at 1200 rpm against a 3800 rpm target",
                            "confidence": 0.9,
                            "rationale": "get_context showed the fan warning before the trip",
                            "recommended_actions": ["fix fan curve"],
                        },
                        4,
                    )
                ],
                stop="tool_use",
            ),
        ]
    )
    provider._client = client
    agent = HypothesisAgent(llm_settings, provider=provider)
    assert agent.agentic and agent.backend.endswith(":agentic")

    hyp = agent.generate(failure, [], events=run_events)
    assert hyp.root_cause_id == "RC_FAN_PROFILE_BAD" and hyp.confidence == 0.9
    assert hyp.source.endswith(":agentic")
    assert [t.tool for t in hyp.trace] == ["lookup_catalogue", "get_context", "read_block"]
    assert hyp.trace[0].note.startswith("Checking the catalogue")
    assert "Fan speed below profile" in hyp.trace[1].output
    assert "REG_DUMP_BEGIN" in hyp.trace[2].output

    # Request shape: auto tool choice, strict tools, both tool_results in ONE user message.
    first = client.requests[0]
    assert first["tool_choice"] == {"type": "auto"}
    assert {t["name"] for t in first["tools"]} >= {"get_context", "submit_hypothesis"}
    assert all(t.get("strict") for t in first["tools"])
    second_user = client.requests[1]["messages"][-1]
    assert second_user["role"] == "user"
    assert [b["type"] for b in second_user["content"]] == ["tool_result", "tool_result"]


def test_agentic_loop_forces_answer_when_budget_spent(settings: Settings, failure, run_events):
    llm_settings = settings.model_copy(
        update={
            "anthropic_api_key": "k",
            "offline": False,
            "llm_mode": "agentic",
            "agent_max_turns": 2,
        }
    )
    provider = AnthropicProvider(llm_settings)
    submit = _use(
        "submit_hypothesis",
        {
            "root_cause_id": "OTHER",
            "root_cause": "unclear",
            "confidence": 0.3,
            "rationale": "",
            "recommended_actions": [],
        },
        9,
    )
    client = ScriptedClient(
        [
            _resp(
                [
                    _use(
                        "get_context",
                        {"start_ms": 0, "end_ms": 100, "include_info": True, "subsystem": ""},
                        1,
                    )
                ]
            ),
            _resp(
                [
                    _use(
                        "get_context",
                        {"start_ms": 0, "end_ms": 200, "include_info": True, "subsystem": ""},
                        2,
                    )
                ]
            ),
            _resp([submit]),  # final turn: only submit_hypothesis is offered
        ]
    )
    provider._client = client
    hyp = HypothesisAgent(llm_settings, provider=provider).generate(failure, [], events=run_events)
    assert hyp.root_cause_id is None and len(hyp.trace) == 2
    last = client.requests[-1]
    assert [t["name"] for t in last["tools"]] == ["submit_hypothesis"]
    assert "budget is spent" in str(client.requests[-1]["messages"][-1]["content"])


def test_agentic_loop_falls_back_to_rules_on_failure(settings: Settings, failure, run_events):
    llm_settings = settings.model_copy(
        update={
            "anthropic_api_key": "k",
            "offline": False,
            "llm_mode": "agentic",
            "agent_max_turns": 1,
        }
    )
    provider = AnthropicProvider(llm_settings)
    provider._client = ScriptedClient(
        [_resp([_text("nope")], stop="end_turn"), _resp([_text("still nope")], stop="end_turn")]
    )
    hyp = HypothesisAgent(llm_settings, provider=provider).generate(failure, [], events=run_events)
    assert hyp.source == "rule-based" and hyp.trace == []

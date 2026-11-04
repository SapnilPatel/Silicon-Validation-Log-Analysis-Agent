"""LangGraph multi-agent team driven by a role-aware scripted Anthropic client."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from svla_agent.config import Settings
from svla_agent.hypothesize.agent import HypothesisAgent
from svla_agent.hypothesize.llm import AnthropicProvider
from svla_agent.hypothesize.multiagent import SPECIALIST_DOMAIN, SpecialistTools
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
        _event(100, "PMU", "INFO", "Rail telemetry | mv=850"),
        _event(1200, "PMU", "WARN", "Fast droop event | rail=VDDQ mv_before=1100 mv_after=960"),
        _event(1300, "DDR", "WARN", "ECC corrected errors rising | delta=80"),
        _event(1500, "DDR", "ERROR", "ECC uncorrectable error | ch=0 bank=3"),
        _event(
            1520,
            "DDR",
            "ERROR",
            "DDR PHY register snapshot",
            ["REG_DUMP_BEGIN", "50000000: 0000BEEF", "REG_DUMP_END"],
        ),
        _event(1540, "SYSTEM", "FATAL", "DDR subsystem failure"),
    ]


@pytest.fixture()
def failure():
    return Failure(
        failure_id="000003:DDR_ECC",
        run_id="000003",
        failure_type="DDR_ECC",
        severity=Severity.HIGH,
        subsys="SYSTEM",
        first_ts_ms=1500,
        last_ts_ms=1540,
        summary="DDR_ECC in DDR/SYSTEM: DDR subsystem failure",
        signature="DDR_ECC | x",
        evidence=["[1500] DDR: ERROR: ECC uncorrectable error | ch=0 bank=3"],
    )


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _use(name, inp, i):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=f"tu_{i}")


def _resp(blocks, stop="tool_use"):
    return SimpleNamespace(content=blocks, stop_reason=stop)


class RoleClient:
    """Routes scripted responses by the role named in the system prompt (parallel-safe)."""

    def __init__(self, scripts: dict[str, list]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        system = kwargs["system"]
        for key in self.scripts:
            if system.startswith(f"You are the {key}"):  # role is named first in every prompt
                return self.scripts[key].pop(0)
        raise AssertionError(f"no script for system prompt: {system[:60]}")


def test_specialist_tools_are_scoped(failure, run_events):
    pmu = SpecialistTools("PMU", failure, run_events, None)
    ctx = pmu.get_context(0, 2000, include_info=False)
    assert "Fast droop" in ctx and "ECC" not in ctx
    assert "RC_LOAD_STEP_SPIKE" in pmu.lookup_catalogue() and "RC_DDR" not in pmu.lookup_catalogue()
    assert SPECIALIST_DOMAIN["DDR"] == "DDR_ECC"
    assert "No catalogue" in SpecialistTools("SYSTEM", failure, run_events, None).lookup_catalogue()


def test_team_runs_triage_parallel_specialists_and_reviewer(
    settings: Settings, failure, run_events
):
    llm_settings = settings.model_copy(
        update={
            "anthropic_api_key": "k",
            "offline": False,
            "llm_mode": "multiagent",
            "agent_max_turns": 4,
        }
    )
    provider = AnthropicProvider(llm_settings)
    client = RoleClient(
        {
            "triage lead": [
                _resp(
                    [
                        _text("DDR failed; PMU logged a droop just before."),
                        _use(
                            "plan_investigation",
                            {
                                "specialists": [
                                    {"subsystem": "DDR", "question": "What preceded the UE?"},
                                    {"subsystem": "PMU", "question": "Any droop on VDDQ?"},
                                ],
                                "rationale": "power may be upstream",
                            },
                            1,
                        ),
                    ]
                ),
            ],
            "DDR specialist": [
                _resp(
                    [
                        _use(
                            "get_context",
                            {"start_ms": 0, "end_ms": 1540, "include_info": False, "subsystem": ""},
                            2,
                        )
                    ]
                ),
                _resp(
                    [
                        _use(
                            "report_finding",
                            {
                                "observation": "corrected errors rose then UE on ch0 bank3",
                                "suspected_cause": "marginal timing or supply",
                                "root_cause_id": "RC_PMU_DROOP_UNDER_LOAD",
                                "confidence": 0.6,
                                "evidence": ["ECC corrected errors rising | delta=80"],
                            },
                            3,
                        )
                    ]
                ),
            ],
            "PMU specialist": [
                _resp(
                    [
                        _use(
                            "get_context",
                            {"start_ms": 0, "end_ms": 1540, "include_info": False, "subsystem": ""},
                            4,
                        )
                    ]
                ),
                _resp(
                    [
                        _use(
                            "report_finding",
                            {
                                "observation": "VDDQ droop 1100->960 mV at 1200 ms",
                                "suspected_cause": "supply droop under load",
                                "root_cause_id": "RC_PMU_DROOP_UNDER_LOAD",
                                "confidence": 0.85,
                                "evidence": ["Fast droop event | rail=VDDQ"],
                            },
                            5,
                        )
                    ]
                ),
            ],
            "reviewer": [
                _resp(
                    [
                        _text("Both agree: droop precedes the UE."),
                        _use(
                            "submit_hypothesis",
                            {
                                "root_cause_id": "RC_PMU_DROOP_UNDER_LOAD",
                                "root_cause": "VDDQ droop corrupted DRAM accesses",
                                "causal_chain": "PMU droop -> DDR corrected errors -> UE",
                                "confidence": 0.88,
                                "rationale": "timing lines up",
                                "recommended_actions": ["add decoupling on VDDQ"],
                            },
                            6,
                        ),
                    ]
                ),
            ],
        }
    )
    provider._client = client
    agent = HypothesisAgent(llm_settings, provider=provider)
    assert agent.multiagent and agent.needs_events and agent.backend.endswith(":multiagent")

    hyp = agent.generate(failure, [], events=run_events)
    assert hyp.root_cause_id == "RC_PMU_DROOP_UNDER_LOAD" and hyp.confidence == 0.88
    assert hyp.source.endswith(":multiagent")
    assert hyp.rationale.startswith("Causal chain: PMU droop -> DDR corrected errors -> UE")
    assert "2 of 2 specialists" in hyp.rationale
    agents = [t.agent for t in hyp.trace]
    assert agents[0] == "triage" and agents[-1] == "reviewer"
    assert {"specialist:DDR", "specialist:PMU"} <= set(agents)
    assert [t.step for t in hyp.trace] == list(range(1, len(hyp.trace) + 1))
    # Specialists saw only their own subsystem's lines.
    ddr_ctx = next(t for t in hyp.trace if t.agent == "specialist:DDR" and t.tool == "get_context")
    assert "ECC" in ddr_ctx.output and "droop" not in ddr_ctx.output
    pmu_ctx = next(t for t in hyp.trace if t.agent == "specialist:PMU" and t.tool == "get_context")
    assert "droop" in pmu_ctx.output and "ECC" not in pmu_ctx.output
    # Reviewer was offered both submit and follow-up; triage saw the plan tool.
    rev = next(r for r in client.requests if "reviewer" in r["system"])
    assert {t["name"] for t in rev["tools"]} == {"submit_hypothesis", "request_followup"}
    assert all(not s for s in client.scripts.values())  # every scripted response consumed


def test_team_followup_then_submit(settings: Settings, failure, run_events):
    llm_settings = settings.model_copy(
        update={
            "anthropic_api_key": "k",
            "offline": False,
            "llm_mode": "multiagent",
            "agent_max_turns": 4,
        }
    )
    provider = AnthropicProvider(llm_settings)

    def finding(i, conf):
        return _resp(
            [
                _use(
                    "report_finding",
                    {
                        "observation": "o",
                        "suspected_cause": "s",
                        "root_cause_id": "RC_DDR_TRAINING_EDGE",
                        "confidence": conf,
                        "evidence": [],
                    },
                    i,
                )
            ]
        )

    client = RoleClient(
        {
            "triage lead": [
                _resp(
                    [
                        _use(
                            "plan_investigation",
                            {
                                "specialists": [{"subsystem": "DDR", "question": "q"}],
                                "rationale": "",
                            },
                            1,
                        )
                    ]
                )
            ],
            "DDR specialist": [finding(2, 0.4), finding(3, 0.7)],  # first pass, then the follow-up
            "reviewer": [
                _resp(
                    [
                        _use(
                            "request_followup",
                            {"subsystem": "DDR", "question": "check training margin"},
                            4,
                        )
                    ]
                ),
                _resp(
                    [
                        _use(
                            "submit_hypothesis",
                            {
                                "root_cause_id": "RC_DDR_TRAINING_EDGE",
                                "root_cause": "edge",
                                "causal_chain": "",
                                "confidence": 0.7,
                                "rationale": "",
                                "recommended_actions": [],
                            },
                            5,
                        )
                    ]
                ),
            ],
        }
    )
    provider._client = client
    hyp = HypothesisAgent(llm_settings, provider=provider).generate(failure, [], events=run_events)
    assert hyp.root_cause_id == "RC_DDR_TRAINING_EDGE"
    tools = [(t.agent, t.tool) for t in hyp.trace]
    assert tools.count(("reviewer", "request_followup")) == 1
    assert tools.count(("specialist:DDR", "report_finding")) == 2
    second_review = [r for r in client.requests if "reviewer" in r["system"]][1]
    assert [t["name"] for t in second_review["tools"]] == [
        "submit_hypothesis"
    ]  # no second follow-up


def test_team_failure_falls_back_to_rules(settings: Settings, failure, run_events):
    llm_settings = settings.model_copy(
        update={"anthropic_api_key": "k", "offline": False, "llm_mode": "multiagent"}
    )
    provider = AnthropicProvider(llm_settings)
    provider._client = RoleClient({"triage lead": [_resp([_text("no plan")], stop="end_turn")]})
    hyp = HypothesisAgent(llm_settings, provider=provider).generate(failure, [], events=run_events)
    assert hyp.source == "rule-based" and hyp.trace == []

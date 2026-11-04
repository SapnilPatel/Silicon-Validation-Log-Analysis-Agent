"""Pydantic schemas shared across the pipeline stages.

The data flows:  raw logs -> :class:`Event` -> :class:`Failure` ->
:class:`FailureCluster` -> :class:`Hypothesis` -> :class:`RunReport`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class LogLevel(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"

    @property
    def rank(self) -> int:
        return {"INFO": 0, "WARN": 1, "ERROR": 2, "FATAL": 3}[self.value]


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# --------------------------------------------------------------------------- #
# Core records
# --------------------------------------------------------------------------- #


class Event(BaseModel):
    """A single parsed log line (optionally with a multi-line block attached)."""

    ts_ms: int = Field(..., description="Timestamp in milliseconds since power-on.")
    source: str = Field(..., description="Log source bucket, e.g. pcie/ddr/fw.")
    subsys: str = Field(..., description="Subsystem tag as printed in the log.")
    level: LogLevel
    message: str
    kv: dict[str, str] = Field(default_factory=dict)
    multiline: list[str] = Field(default_factory=list)
    template: str | None = Field(
        default=None, description="Drain3-mined event template (variables masked)."
    )
    raw: str = Field(default="", description="Original first line, for traceability.")

    @property
    def signature(self) -> str:
        """Stable text used for embedding/dedup: subsystem + mined template."""
        return f"{self.subsys}: {self.template or self.message}"


class RunMeta(BaseModel):
    """Header metadata describing the validation run/DUT configuration."""

    run_id: str
    soc: str | None = None
    stepping: str | None = None
    board: str | None = None
    fw_version: str | None = None
    test_suite: str | None = None
    test_case: str | None = None
    voltage_mv: int | None = None
    temp_c: int | None = None
    clk_mhz: int | None = None
    dram_freq_mhz: int | None = None
    pcie_gen: int | None = None
    pcie_lanes: int | None = None
    workload: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class ParsedRun(BaseModel):
    """All events for one run, across every source file, plus metadata."""

    run_id: str
    meta: RunMeta
    events: list[Event] = Field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len(self.events)


class Failure(BaseModel):
    """A classified failure distilled from one or more error/fatal events."""

    failure_id: str
    run_id: str
    failure_type: str = Field(..., description="Coarse class, e.g. PCIE_TIMEOUT.")
    severity: Severity
    subsys: str
    first_ts_ms: int
    last_ts_ms: int
    summary: str
    signature: str = Field(..., description="Normalised text used for embedding/clustering.")
    evidence: list[str] = Field(
        default_factory=list, description="Representative log lines supporting the failure."
    )
    kv: dict[str, str] = Field(default_factory=dict)
    cluster_id: int | None = None
    fingerprint: str = Field(
        default="", description="Stable hash of the signature; equal across duplicate failures."
    )
    occurrences: int = Field(
        default=1, description="How many failures in the corpus share this fingerprint."
    )

    @property
    def duration_ms(self) -> int:
        return max(0, self.last_ts_ms - self.first_ts_ms)


class FailureCluster(BaseModel):
    """A family of similar failures discovered by unsupervised clustering."""

    cluster_id: int
    label: str = Field(..., description="Human-readable name derived from members.")
    size: int
    failure_ids: list[str] = Field(default_factory=list)
    failure_types: list[str] = Field(default_factory=list)
    exemplar_failure_id: str | None = None
    representative_signature: str = ""


class RetrievedIncident(BaseModel):
    """A historical incident returned by the RAG store for context."""

    failure_id: str
    run_id: str
    failure_type: str
    root_cause_id: str | None = None
    signature: str
    distance: float = Field(..., description="Vector distance (lower = more similar).")

    @property
    def similarity(self) -> float:
        return round(1.0 - self.distance, 4)


class ToolCall(BaseModel):
    """One step of the agent's investigation (tool name, arguments, what came back)."""

    step: int
    tool: str
    agent: str = Field(default="", description="Which agent acted (multi-agent mode).")
    input: dict = Field(default_factory=dict)
    output: str = Field(default="", description="Tool result as shown to the model (truncated).")
    note: str = Field(default="", description="Model's own words before the call, if any.")


class Hypothesis(BaseModel):
    """A root-cause hypothesis for a failure (cluster), with evidence + provenance."""

    failure_id: str
    cluster_id: int | None = None
    root_cause: str
    root_cause_id: str | None = Field(
        default=None, description="Canonical root-cause id from the knowledge base, if any."
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    recommended_actions: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    similar_incidents: list[RetrievedIncident] = Field(default_factory=list)
    source: str = Field(default="llm", description="'llm' or 'rule-based'.")
    trace: list[ToolCall] = Field(
        default_factory=list, description="Investigation steps when the agentic loop ran."
    )


class RunReport(BaseModel):
    """The compiled, reproducible debug report for a single run."""

    run_id: str
    meta: RunMeta
    anomalous: bool
    n_events: int
    failures: list[Failure] = Field(default_factory=list)
    clusters: list[FailureCluster] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    generated_at: str = ""

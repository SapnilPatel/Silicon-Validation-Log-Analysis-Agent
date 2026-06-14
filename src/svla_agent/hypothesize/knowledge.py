"""Domain knowledge used by the rule-based hypothesis fallback.

Each entry maps a coarse failure type to a canonical root-cause statement and a
short remediation playbook. This keeps the pipeline useful (and deterministic)
when no LLM is configured, and gives the LLM a grounded prior to work from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RootCausePrior:
    root_cause: str
    rationale: str
    actions: list[str] = field(default_factory=list)


KNOWLEDGE: dict[str, RootCausePrior] = {
    "PCIE_TIMEOUT": RootCausePrior(
        root_cause="PCIe link training failed to reach L0 (LTSSM stuck in Polling/Recovery)",
        rationale=(
            "Repeated LTSSM Recovery transitions with AER corrected errors followed by "
            "'Link training timeout' point to physical-layer signal-integrity or refclk "
            "issues rather than software configuration."
        ),
        actions=[
            "Check refclk routing/spread-spectrum and PLL lock on the failing lane",
            "Review SI margins and equalization (EQ) preset tuning for the negotiated gen",
            "Verify ASPM/L1 substate config is not gating the clock during training",
        ],
    ),
    "DDR_ECC": RootCausePrior(
        root_cause="DDR uncorrectable ECC error following rising corrected-error rate",
        rationale=(
            "Corrected ECC errors trending up, then a read-retry, then an uncorrectable "
            "error and memtest mismatch indicate a marginal DRAM timing/training edge, "
            "often aggravated by voltage droop under load."
        ),
        actions=[
            "Re-run DDR write-leveling/read training and widen timing margins",
            "Correlate UE timestamps with PMU droop / thermal events",
            "Sweep DRAM frequency and VDDQ to locate the marginal corner",
        ],
    ),
    "PMU_DROOP": RootCausePrior(
        root_cause="VCORE brown-out from a fast load-step droop crossing the reset threshold",
        rationale=(
            "A fast droop event drops VCORE below the brown-out threshold, the SoC hangs, "
            "and the watchdog fires — classic insufficient decoupling / regulator load-step "
            "response under a workload transient."
        ),
        actions=[
            "Add/verify decoupling capacitance near the affected rail",
            "Tune regulator load-line and transient response (AVP) settings",
            "Throttle the load-step (DVFS ramp) that triggers the droop",
        ],
    ),
    "THERMAL_TRIP": RootCausePrior(
        root_cause="Junction temperature exceeded the thermal trip threshold",
        rationale=(
            "Temperature climbed past the DVFS policy and crossed the 105C trip point "
            "before throttling could recover it, triggering an emergency shutdown — a "
            "cooling-solution or fan-profile shortfall under sustained workload."
        ),
        actions=[
            "Inspect heatsink mounting/TIM and airflow on the DUT",
            "Make the fan/DVFS profile more aggressive ahead of the trip point",
            "Reduce sustained workload power or add thermal headroom",
        ],
    ),
    "FW_EXCEPTION": RootCausePrior(
        root_cause="Firmware crashed on an unhandled CPU exception (fault handler)",
        rationale=(
            "Null-pointer guard and service retries precede an unhandled HardFault/BusFault "
            "with a stack trace, indicating a firmware logic defect (race or bad deref) "
            "rather than a hardware fault."
        ),
        actions=[
            "Symbolicate the captured backtrace and inspect the faulting frame",
            "Audit the retried service path for races / null dereferences",
            "Add guards/asserts and reproduce under a debug build",
        ],
    ),
    "NOC_TIMEOUT": RootCausePrior(
        root_cause="Interconnect transaction timed out under fabric congestion",
        rationale=(
            "Sustained high transaction-queue occupancy and stalls precede a transaction "
            "timeout and interconnect failure, suggesting QoS starvation or a potential "
            "deadlock on a congested route."
        ),
        actions=[
            "Review NoC QoS/arbitration weights for the starved initiator",
            "Check for dependency cycles / deadlock on the congested route",
            "Rate-limit or reorder the DMA burst pattern driving the congestion",
        ],
    ),
    "UNKNOWN_FAILURE": RootCausePrior(
        root_cause="Unclassified fatal error",
        rationale="The failure did not match a known signature; manual triage required.",
        actions=["Inspect the evidence lines and surrounding context manually"],
    ),
}


def prior_for(failure_type: str) -> RootCausePrior:
    return KNOWLEDGE.get(failure_type, KNOWLEDGE["UNKNOWN_FAILURE"])

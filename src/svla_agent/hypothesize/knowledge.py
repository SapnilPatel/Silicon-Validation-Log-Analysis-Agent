"""Domain knowledge used by the hypothesis agent.

Two layers:

* :data:`KNOWLEDGE` — per failure type, a canonical root-cause statement, a
  rationale and a remediation playbook. This is the grounded prior the
  rule-based fallback and the LLM both start from.
* :data:`ROOT_CAUSES` — per failure type, the catalogue of *specific* root
  causes (ids match the dataset's ground-truth labels) together with the log
  evidence markers that point at each one. The rule-based agent scores the
  candidates against the evidence; the LLM receives them as an enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RootCausePrior:
    root_cause: str
    rationale: str
    actions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RootCause:
    """A specific, nameable root cause and how it shows up in the logs."""

    rc_id: str
    description: str
    markers: tuple[str, ...]  # case-insensitive substrings expected in the evidence
    actions: tuple[str, ...] = ()


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


ROOT_CAUSES: dict[str, tuple[RootCause, ...]] = {
    "PCIE_TIMEOUT": (
        RootCause(
            "RC_PCIE_REFCLK_GATING",
            "Reference clock instability / gating during link training",
            ("refclk instability", "refclk"),
            (
                "Scope the refclk at the connector and SoC ball for gating/jitter",
                "Disable spread-spectrum clocking and retest link training",
            ),
        ),
        RootCause(
            "RC_PCIE_SI_MARGIN_LOW",
            "Lane signal-integrity margin below the receiver threshold",
            ("lane margining below threshold", "eye_width"),
            (
                "Run lane-margining sweeps and compare eye width against the gen spec",
                "Review trace length/loss budget and connector/retimer placement",
            ),
        ),
        RootCause(
            "RC_PCIE_ASPM_MISCONFIG",
            "ASPM L1 substate entered while training, gating the link clock",
            ("aspm l1 entry during training", "aspm"),
            (
                "Disable ASPM/L1 substates in firmware config and retest",
                "Verify CLKREQ# handling between root port and endpoint",
            ),
        ),
        RootCause(
            "RC_PCIE_EQ_TUNING",
            "Equalization preset/coefficients fail to converge at the negotiated speed",
            ("equalization phase failed", "eq_fail"),
            (
                "Sweep TX presets / RX CTLE-DFE coefficients for the negotiated gen",
                "Force a lower gen to confirm the EQ-at-speed hypothesis",
            ),
        ),
    ),
    "DDR_ECC": (
        RootCause(
            "RC_DDR_TIMING_VIOLATION",
            "A DRAM timing parameter is violated at the configured frequency",
            ("timing parameter violation", "trfc", "trcd"),
            (
                "Relax the flagged timing parameter and re-run the memtest",
                "Cross-check the DRAM datasheet timings against the controller config",
            ),
        ),
        RootCause(
            "RC_DDR_TRAINING_EDGE",
            "Read/write training converged on a marginal edge",
            ("read training margin low", "margin_ps"),
            (
                "Re-run training with per-bit deskew and widen the margin window",
                "Sweep VREF to re-centre the data eye",
            ),
        ),
        RootCause(
            "RC_PMU_DROOP_UNDER_LOAD",
            "VDDQ droop under load corrupts DRAM accesses",
            ("fast droop event", "rail=vddq"),
            (
                "Correlate the UE with the VDDQ droop and add decoupling on the rail",
                "Tune the regulator transient response for the memory rail",
            ),
        ),
        RootCause(
            "RC_DDR_CDC_GLITCH",
            "Clock-domain-crossing synchroniser glitch in the controller",
            ("cdc synchronizer glitch", "cdc"),
            (
                "Review CDC constraints on the controller/PHY boundary",
                "Reproduce at corner voltage/frequency to isolate the glitch",
            ),
        ),
    ),
    "PMU_DROOP": (
        RootCause(
            "RC_LOAD_STEP_SPIKE",
            "A fast di/dt load step exceeds the regulator slew capability",
            ("load step detected", "cause=load_step", "di_dt"),
            (
                "Throttle the DVFS/workload ramp producing the load step",
                "Enable adaptive voltage positioning on the rail",
            ),
        ),
        RootCause(
            "RC_INSUFFICIENT_DECOUPLING",
            "Insufficient decoupling causes undershoot/ringing after a transient",
            ("rail ringing after transient", "undershoot_mv"),
            (
                "Add bulk and high-frequency decoupling close to the package",
                "Re-simulate the PDN impedance profile",
            ),
        ),
        RootCause(
            "RC_REGULATOR_LIMIT",
            "Regulator reaches its current limit and folds back",
            ("regulator current limit hit", "cause=regulator_limit", "i_limit"),
            (
                "Raise the regulator OCP limit or split the load across phases",
                "Confirm the regulator phase count for the measured peak current",
            ),
        ),
    ),
    "THERMAL_TRIP": (
        RootCause(
            "RC_HEATSINK_POOR",
            "Poor heatsink/TIM contact — large die-to-ambient delta",
            ("sensor-to-ambient delta high", "delta_c"),
            (
                "Re-seat the heatsink and replace the TIM",
                "Measure mounting pressure against the thermal spec",
            ),
        ),
        RootCause(
            "RC_FAN_PROFILE_BAD",
            "Fan speed stays below the profile target as temperature rises",
            ("fan speed below profile", "target_rpm"),
            (
                "Fix the fan curve / PWM mapping in the thermal policy",
                "Verify tachometer feedback and fan power",
            ),
        ),
        RootCause(
            "RC_WORKLOAD_BURST",
            "A sustained power burst outruns the cooling headroom",
            ("power burst detected", "p_w="),
            (
                "Cap the burst power or lengthen the DVFS step-down reaction",
                "Add thermal headroom to the test's power envelope",
            ),
        ),
    ),
    "FW_EXCEPTION": (
        RootCause(
            "RC_FW_RACE_CONDITION",
            "Race between service threads on a shared lock",
            ("mutex contention timeout", "lock="),
            (
                "Audit lock ordering in the contended service path",
                "Add a watchdog-visible trace around the critical section",
            ),
        ),
        RootCause(
            "RC_FW_NULL_DEREF",
            "Null-pointer dereference in a service handler",
            ("fault address near null", "dfar=0x0000"),
            (
                "Symbolicate the faulting PC and add a null guard",
                "Fuzz the service call path with uninitialised arguments",
            ),
        ),
        RootCause(
            "RC_FW_STACK_OVERFLOW",
            "Task stack overflow corrupts the return frame",
            ("stack watermark exceeded", "used_pct"),
            (
                "Increase the task stack or reduce frame size in the hot path",
                "Enable stack-canary checks in the debug build",
            ),
        ),
    ),
    "NOC_TIMEOUT": (
        RootCause(
            "RC_NOC_QOS_STARVATION",
            "A low-priority initiator is starved of QoS credits",
            ("qos credit starvation", "credits=0"),
            (
                "Rebalance QoS weights / credits for the starved initiator",
                "Enable age-based arbitration on the congested link",
            ),
        ),
        RootCause(
            "RC_NOC_DEADLOCK",
            "Dependency cycle across fabric routes causes a deadlock",
            ("dependency cycle suspected", "cycle"),
            (
                "Inspect the route table for cyclic dependencies",
                "Enable deadlock-recovery timeouts on the affected ring",
            ),
        ),
        RootCause(
            "RC_DMA_BURST_OVERFLOW",
            "DMA burst length overflows the fabric buffers",
            ("dma burst overflow", "burst_len"),
            (
                "Reduce the DMA burst length or add outstanding-transaction limits",
                "Verify buffer depth on the DMA->DDR path",
            ),
        ),
    ),
}


def prior_for(failure_type: str) -> RootCausePrior:
    return KNOWLEDGE.get(failure_type, KNOWLEDGE["UNKNOWN_FAILURE"])


def candidates_for(failure_type: str) -> tuple[RootCause, ...]:
    """Specific root-cause candidates for a failure type (empty if unknown)."""
    return ROOT_CAUSES.get(failure_type, ())


def root_cause_by_id(rc_id: str) -> RootCause | None:
    for causes in ROOT_CAUSES.values():
        for rc in causes:
            if rc.rc_id == rc_id:
                return rc
    return None

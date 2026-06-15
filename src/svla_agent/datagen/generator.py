import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass

# ----------------------------
# Core structures
# ----------------------------


@dataclass
class Event:
    ts_ms: int
    src: str  # log source file bucket (fw, pcie, ddr, ...)
    subsys: str  # subsystem tag for readability
    level: str  # INFO/WARN/ERROR/FATAL
    msg: str
    kv: dict[str, str]
    multiline: list[str] | None = None  # for register dumps / stack traces


SOURCES = ["system", "fw", "pcie", "ddr", "pmu", "thermal", "noc"]


def fmt_ts(ts_ms: int) -> str:
    ms = ts_ms % 1000
    s_total = ts_ms // 1000
    s = s_total % 60
    m_total = s_total // 60
    m = m_total % 60
    h = (m_total // 60) % 24
    return f"[{h:02d}:{m:02d}:{s:02d}.{ms:03d}]"


def event_to_lines(e: Event) -> list[str]:
    kv = ""
    if e.kv:
        kv = " | " + " ".join([f"{k}={v}" for k, v in e.kv.items()])
    first = f"{fmt_ts(e.ts_ms)} {e.subsys}: {e.level}: {e.msg}{kv}"
    lines = [first]
    if e.multiline:
        # Indent block lines like real dumps/traces
        lines.extend([f"    {ln}" for ln in e.multiline])
    return lines


def jitter(rng: random.Random, base: int, j: int) -> int:
    return max(0, base + rng.randint(-j, j))


def hex32(rng: random.Random) -> str:
    return "0x" + f"{rng.randint(0, 2**32 - 1):08X}"


def hex16(rng: random.Random) -> str:
    return "0x" + f"{rng.randint(0, 2**16 - 1):04X}"


# ----------------------------
# Metadata + header
# ----------------------------


def gen_meta(rng: random.Random, run_id: int) -> dict:
    return {
        "run_id": f"{run_id:06d}",
        "soc": rng.choice(["AURORA-X1", "ORION-S2", "NEBULA-A0"]),
        "stepping": rng.choice(["A0", "A1", "B0"]),
        "board": rng.choice(["EVT-1", "EVT-2", "DVT-1"]),
        "fw_version": rng.choice(["0.9.14", "0.9.21", "1.0.3"]),
        "test_suite": rng.choice(["psv_smoke", "psv_stress", "io_validation", "bringup"]),
        "test_case": rng.choice(
            [
                "pcie_link_train_stress",
                "ddr_memtest_long",
                "thermal_soak",
                "mixed_io",
                "noc_fabric_stress",
                "fw_fuzz_boot",
            ]
        ),
        "seed": None,
        "voltage_mv": rng.choice([800, 820, 850, 900, 950]),
        "temp_c": rng.choice([25, 55, 75, 85, 95, 105]),
        "clk_mhz": rng.choice([800, 1000, 1200]),
        "dram_freq_mhz": rng.choice([2400, 3200, 3600]),
        "pcie_gen": rng.choice([3, 4, 5]),
        "pcie_lanes": rng.choice([4, 8, 16]),
        "workload": rng.choice(["idle", "io_stress", "compute_burst", "mixed"]),
    }


def header_lines(meta: dict) -> list[str]:
    return [f"# {k}: {v}" for k, v in meta.items()]


# ----------------------------
# Normal realistic background noise (telemetry)
# ----------------------------


def add_telemetry_noise(
    rng: random.Random, events: list[Event], t: int, meta: dict, n_lines: int
) -> int:
    """
    Adds realistic periodic logs across sources: temps, rails, ECC counters, LTSSM stats, NoC counters.
    """
    ecc_corr = 0
    ltssm_retry = 0
    droop_cnt = 0

    for i in range(n_lines):
        t += jitter(rng, 8, 7)  # dense logs
        kind = rng.random()

        if kind < 0.18:
            # thermal
            tc = jitter(rng, meta["temp_c"], 2)
            events.append(
                Event(
                    t,
                    "thermal",
                    "THERMAL",
                    "INFO",
                    "Temp sample",
                    {"t_c": str(tc), "sensor": str(rng.randint(0, 3))},
                )
            )
        elif kind < 0.34:
            # pmu
            v = jitter(rng, meta["voltage_mv"], 25)
            i_ma = jitter(rng, 1200 if meta["workload"] != "idle" else 500, 200)
            events.append(
                Event(
                    t,
                    "pmu",
                    "PMU",
                    "INFO",
                    "Rail telemetry",
                    {"rail": "VCORE", "mv": str(v), "i_ma": str(i_ma), "droop_cnt": str(droop_cnt)},
                )
            )
        elif kind < 0.50:
            # ddr counters
            ecc_corr += rng.randint(0, 2)
            events.append(
                Event(
                    t,
                    "ddr",
                    "DDR",
                    "INFO",
                    "DDR counters",
                    {
                        "ecc_corr": str(ecc_corr),
                        "ecc_ue": "0",
                        "rd_lat_ns": str(jitter(rng, 78, 6)),
                    },
                )
            )
        elif kind < 0.66:
            # pcie stats
            ltssm_retry += rng.randint(0, 1)
            events.append(
                Event(
                    t,
                    "pcie",
                    "PCIE",
                    "INFO",
                    "Link stats",
                    {
                        "gen": str(meta["pcie_gen"]),
                        "lanes": f"x{meta['pcie_lanes']}",
                        "retrain": str(ltssm_retry),
                        "aer": "0",
                    },
                )
            )
        elif kind < 0.82:
            # noc counters
            q = rng.choice(
                ["low", "med", "high"] if meta["workload"] != "idle" else ["low", "low", "med"]
            )
            events.append(
                Event(
                    t,
                    "noc",
                    "NOC",
                    "INFO",
                    "Fabric telemetry",
                    {"txn_q": q, "stall_pct": str(jitter(rng, 3 if q == "low" else 12, 3))},
                )
            )
        else:
            # fw chatter
            events.append(
                Event(
                    t,
                    "fw",
                    "FW",
                    "INFO",
                    rng.choice(["Heartbeat", "Checkpoint", "Loop tick"]),
                    {"i": str(i)},
                )
            )

    return t


# ----------------------------
# Multi-line blocks
# ----------------------------


def reg_dump_block(rng: random.Random, base_addr: int, n_regs: int = 16) -> list[str]:
    lines = ["REG_DUMP_BEGIN"]
    addr = base_addr
    for _ in range(n_regs):
        lines.append(f"{addr:08X}: {rng.randint(0, 2**32 - 1):08X}")
        addr += 4
    lines.append("REG_DUMP_END")
    return lines


def fw_stack_trace(rng: random.Random) -> list[str]:
    pc = hex32(rng)
    lr = hex32(rng)
    sp = hex32(rng)
    trace = [
        "EXC: HardFault",
        f"PC={pc} LR={lr} SP={sp} PSR={hex32(rng)}",
        "Backtrace:",
    ]
    frames = rng.randint(6, 10)
    for i in range(frames):
        trace.append(
            f"  #{i:02d}  {hex32(rng)}  {rng.choice(['fw_init', 'pcie_service', 'ddr_isr', 'noc_worker', 'main_loop', 'timer_cb'])}+0x{rng.randint(0, 255):02X}"
        )
    return trace


# ----------------------------
# Failure patterns (more realistic)
# Each returns (events, injected_pattern_label)
# ----------------------------


def inject_pcie_ltssm_timeout(rng: random.Random, t0: int, meta: dict) -> tuple[list[Event], dict]:
    lane = str(rng.randint(0, meta["pcie_lanes"] - 1))
    ev: list[Event] = []
    t = t0

    ev.append(Event(t, "pcie", "PCIE", "INFO", "LTSSM -> Detect", {"lane": lane}))
    for _ in range(rng.randint(3, 7)):
        t += jitter(rng, 120, 40)
        ev.append(Event(t, "pcie", "PCIE", "INFO", "LTSSM -> Polling", {"lane": lane}))
        t += jitter(rng, 180, 60)
        ev.append(
            Event(
                t,
                "pcie",
                "PCIE",
                "WARN",
                "LTSSM -> Recovery",
                {"lane": lane, "reason": rng.choice(["rx_detect_fail", "eq_fail", "deskew_fail"])},
            )
        )

    # AER hint + reg dump
    t += jitter(rng, 220, 60)
    ev.append(
        Event(
            t,
            "pcie",
            "PCIE",
            "WARN",
            "AER: Corrected error",
            {"lane": lane, "code": rng.choice(["BadTLP", "ReplayTimerTimeout", "ReceiverError"])},
        )
    )

    t += jitter(rng, 300, 100)
    ev.append(
        Event(
            t,
            "pcie",
            "PCIE",
            "ERROR",
            "Link training timeout",
            {"lane": lane, "ltssm": "Polling", "rc": "-110"},
        )
    )
    ev.append(
        Event(t + 2, "fw", "FW", "WARN", "Retrying PCIe init", {"attempt": "1/3", "lane": lane})
    )

    t += jitter(rng, 1500, 250)
    ev.append(
        Event(
            t,
            "pcie",
            "PCIE",
            "ERROR",
            "Link training timeout",
            {"lane": lane, "ltssm": "Recovery", "rc": "-110"},
        )
    )
    ev.append(
        Event(t + 2, "fw", "FW", "WARN", "Retrying PCIe init", {"attempt": "2/3", "lane": lane})
    )

    # dump registers near fatal
    t += jitter(rng, 500, 150)
    ev.append(
        Event(
            t,
            "pcie",
            "PCIE",
            "ERROR",
            "PCIe controller register snapshot",
            {"lane": lane},
            multiline=reg_dump_block(rng, base_addr=0x40010000 + rng.randint(0, 0x1000), n_regs=24),
        )
    )

    t += jitter(rng, 50, 20)
    ev.append(
        Event(
            t,
            "system",
            "SYSTEM",
            "FATAL",
            "PCIe initialization failed",
            {"lane": lane, "gen": str(meta["pcie_gen"]), "lanes": f"x{meta['pcie_lanes']}"},
        )
    )

    label = {
        "pattern_id": "P_PCIE_LTSSM_TIMEOUT",
        "failure_type": "PCIE_TIMEOUT",
        "root_cause_id": rng.choice(
            [
                "RC_PCIE_REFCLK_GATING",
                "RC_PCIE_SI_MARGIN_LOW",
                "RC_PCIE_ASPM_MISCONFIG",
                "RC_PCIE_EQ_TUNING",
            ]
        ),
        "start_ts_ms": t0,
        "end_ts_ms": t,
        "severity": "high",
    }
    return ev, label


def inject_ddr_ecc_then_ue(rng: random.Random, t0: int, meta: dict) -> tuple[list[Event], dict]:
    ev: list[Event] = []
    t = t0
    ch = str(rng.randint(0, 1))
    bank = str(rng.randint(0, 7))

    t += jitter(rng, 60, 20)
    ev.append(
        Event(
            t,
            "ddr",
            "DDR",
            "WARN",
            "ECC corrected errors rising",
            {"ch": ch, "bank": bank, "delta": str(rng.randint(20, 120))},
        )
    )

    t += jitter(rng, 120, 40)
    ev.append(
        Event(
            t,
            "ddr",
            "DDR",
            "WARN",
            "Read retry invoked",
            {"ch": ch, "addr": hex32(rng), "count": str(rng.randint(1, 8))},
        )
    )

    t += jitter(rng, 220, 60)
    ev.append(
        Event(
            t,
            "ddr",
            "DDR",
            "ERROR",
            "ECC uncorrectable error",
            {"ch": ch, "bank": bank, "syndrome": hex16(rng)},
        )
    )

    t += jitter(rng, 40, 15)
    ev.append(
        Event(
            t,
            "fw",
            "FW",
            "ERROR",
            "Memtest mismatch",
            {"addr": hex32(rng), "expected": "0xDEADBEEF", "got": hex32(rng)},
        )
    )

    # DDR PHY regs snapshot
    t += jitter(rng, 80, 25)
    ev.append(
        Event(
            t,
            "ddr",
            "DDR",
            "ERROR",
            "DDR PHY register snapshot",
            {"ch": ch},
            multiline=reg_dump_block(rng, base_addr=0x50000000 + rng.randint(0, 0x2000), n_regs=32),
        )
    )

    t += jitter(rng, 20, 10)
    ev.append(
        Event(
            t,
            "system",
            "SYSTEM",
            "FATAL",
            "DDR subsystem failure",
            {"ch": ch, "dram_freq_mhz": str(meta["dram_freq_mhz"])},
        )
    )

    label = {
        "pattern_id": "P_DDR_ECC_UE",
        "failure_type": "DDR_ECC",
        "root_cause_id": rng.choice(
            [
                "RC_DDR_TIMING_VIOLATION",
                "RC_DDR_TRAINING_EDGE",
                "RC_PMU_DROOP_UNDER_LOAD",
                "RC_DDR_CDC_GLITCH",
            ]
        ),
        "start_ts_ms": t0,
        "end_ts_ms": t,
        "severity": "high",
    }
    return ev, label


def inject_pmu_droop_reset_with_hang(
    rng: random.Random, t0: int, meta: dict
) -> tuple[list[Event], dict]:
    ev: list[Event] = []
    t = t0
    v_before = meta["voltage_mv"]
    v_after = v_before - rng.randint(120, 280)

    t += jitter(rng, 20, 8)
    ev.append(
        Event(
            t,
            "pmu",
            "PMU",
            "WARN",
            "Fast droop event",
            {
                "rail": "VCORE",
                "mv_before": str(v_before),
                "mv_after": str(v_after),
                "cause": rng.choice(["load_step", "regulator_limit", "pll_spike"]),
            },
        )
    )

    t += jitter(rng, 8, 3)
    ev.append(
        Event(t, "pmu", "PMU", "ERROR", "Brown-out threshold crossed", {"threshold_mv": "750"})
    )

    # simulate hang: large gap (watchdog later)
    hang_ms = rng.randint(3000, 12000)
    t_hang_end = t + hang_ms

    ev.append(Event(t_hang_end, "fw", "FW", "ERROR", "Watchdog timeout", {"wd_ms": str(hang_ms)}))
    ev.append(
        Event(
            t_hang_end + 2,
            "system",
            "SYSTEM",
            "FATAL",
            "SoC reset asserted",
            {"reason": "BROWNOUT_WDOG"},
        )
    )

    label = {
        "pattern_id": "P_PMU_DROOP_WDOG_RESET",
        "failure_type": "PMU_DROOP",
        "root_cause_id": rng.choice(
            ["RC_LOAD_STEP_SPIKE", "RC_INSUFFICIENT_DECOUPLING", "RC_REGULATOR_LIMIT"]
        ),
        "start_ts_ms": t0,
        "end_ts_ms": t_hang_end + 2,
        "severity": "high",
    }
    return ev, label


def inject_thermal_trip(rng: random.Random, t0: int, meta: dict) -> tuple[list[Event], dict]:
    ev: list[Event] = []
    t = t0
    start = max(meta["temp_c"], rng.randint(85, 95))
    peak = rng.randint(106, 122)

    t += jitter(rng, 200, 60)
    ev.append(
        Event(
            t,
            "thermal",
            "THERMAL",
            "WARN",
            "Temperature high",
            {"t_c": str(start), "policy": "DVFS"},
        )
    )

    t += jitter(rng, 250, 80)
    ev.append(
        Event(
            t,
            "fw",
            "FW",
            "WARN",
            "DVFS step-down",
            {"clk_mhz": str(meta["clk_mhz"]), "next_clk_mhz": str(max(400, meta["clk_mhz"] - 200))},
        )
    )

    t += jitter(rng, 350, 120)
    ev.append(
        Event(
            t,
            "thermal",
            "THERMAL",
            "ERROR",
            "Thermal trip threshold exceeded",
            {"t_c": str(peak), "threshold_c": "105"},
        )
    )

    t += jitter(rng, 10, 5)
    ev.append(
        Event(t, "system", "SYSTEM", "FATAL", "Emergency shutdown", {"reason": "THERMAL_TRIP"})
    )

    label = {
        "pattern_id": "P_THERMAL_TRIP",
        "failure_type": "THERMAL_TRIP",
        "root_cause_id": rng.choice(
            ["RC_HEATSINK_POOR", "RC_FAN_PROFILE_BAD", "RC_WORKLOAD_BURST"]
        ),
        "start_ts_ms": t0,
        "end_ts_ms": t,
        "severity": "high",
    }
    return ev, label


def inject_fw_exception(rng: random.Random, t0: int, meta: dict) -> tuple[list[Event], dict]:
    ev: list[Event] = []
    t = t0

    # subtle warnings first
    t += jitter(rng, 40, 15)
    ev.append(Event(t, "fw", "FW", "WARN", "Null pointer guard triggered", {"ptr": hex32(rng)}))
    t += jitter(rng, 60, 20)
    ev.append(
        Event(
            t,
            "fw",
            "FW",
            "WARN",
            "Retrying service call",
            {"svc": rng.choice(["pcie_service", "ddr_service", "noc_worker"]), "attempt": "1/3"},
        )
    )

    # exception with stack trace
    t += jitter(rng, 120, 40)
    ev.append(
        Event(
            t,
            "fw",
            "FW",
            "ERROR",
            "Unhandled exception",
            {"vector": rng.choice(["HardFault", "UsageFault", "BusFault"]), "core": "0"},
            multiline=fw_stack_trace(rng),
        )
    )

    # optional core dump marker
    t += jitter(rng, 40, 15)
    ev.append(
        Event(
            t,
            "system",
            "SYSTEM",
            "FATAL",
            "Firmware crash",
            {"reason": "FW_EXCEPTION", "fw_version": meta["fw_version"]},
        )
    )

    label = {
        "pattern_id": "P_FW_EXCEPTION",
        "failure_type": "FW_EXCEPTION",
        "root_cause_id": rng.choice(
            ["RC_FW_RACE_CONDITION", "RC_FW_NULL_DEREF", "RC_FW_STACK_OVERFLOW"]
        ),
        "start_ts_ms": t0,
        "end_ts_ms": t,
        "severity": "high",
    }
    return ev, label


def inject_noc_timeout(rng: random.Random, t0: int, meta: dict) -> tuple[list[Event], dict]:
    ev: list[Event] = []
    t = t0

    for _ in range(rng.randint(3, 6)):
        t += jitter(rng, 120, 50)
        ev.append(
            Event(
                t,
                "noc",
                "NOC",
                "WARN",
                "Fabric congestion",
                {
                    "txn_q": "high",
                    "stall_pct": str(jitter(rng, 28, 6)),
                    "region": rng.choice(["CPU<->DDR", "PCIE<->DDR", "CPU<->PCIE"]),
                },
            )
        )

    t += jitter(rng, 300, 100)
    ev.append(
        Event(
            t,
            "noc",
            "NOC",
            "ERROR",
            "Transaction timeout",
            {
                "txn_id": hex16(rng),
                "src": rng.choice(["CPU0", "PCIE", "DMA"]),
                "dst": rng.choice(["DDR", "PCIE", "MMIO"]),
                "rc": "-62",
            },
        )
    )

    # snapshot NoC regs
    t += jitter(rng, 60, 20)
    ev.append(
        Event(
            t,
            "noc",
            "NOC",
            "ERROR",
            "NoC error register snapshot",
            {"fabric": "NOC0"},
            multiline=reg_dump_block(rng, base_addr=0x60000000 + rng.randint(0, 0x1000), n_regs=20),
        )
    )

    t += jitter(rng, 20, 10)
    ev.append(
        Event(t, "system", "SYSTEM", "FATAL", "Interconnect failure", {"reason": "NOC_TIMEOUT"})
    )

    label = {
        "pattern_id": "P_NOC_TXN_TIMEOUT",
        "failure_type": "NOC_TIMEOUT",
        "root_cause_id": rng.choice(
            ["RC_NOC_QOS_STARVATION", "RC_NOC_DEADLOCK", "RC_DMA_BURST_OVERFLOW"]
        ),
        "start_ts_ms": t0,
        "end_ts_ms": t,
        "severity": "high",
    }
    return ev, label


PATTERNS = [
    ("P_PCIE_LTSSM_TIMEOUT", inject_pcie_ltssm_timeout),
    ("P_DDR_ECC_UE", inject_ddr_ecc_then_ue),
    ("P_PMU_DROOP_WDOG_RESET", inject_pmu_droop_reset_with_hang),
    ("P_THERMAL_TRIP", inject_thermal_trip),
    ("P_FW_EXCEPTION", inject_fw_exception),
    ("P_NOC_TXN_TIMEOUT", inject_noc_timeout),
]

# ----------------------------
# Run generation
# ----------------------------


def gen_boot_sequence(rng: random.Random, meta: dict) -> list[Event]:
    ev: list[Event] = []
    t = 0

    ev.append(Event(t, "system", "BOOT", "INFO", "Power-on reset detected", {}))
    t += jitter(rng, 12, 6)
    ev.append(
        Event(
            t,
            "pmu",
            "PMU",
            "INFO",
            "Rails stable",
            {"vcore_mv": str(meta["voltage_mv"]), "vddio_mv": "1800"},
        )
    )
    t += jitter(rng, 10, 5)
    ev.append(Event(t, "fw", "FW", "INFO", "Boot ROM entry", {"pc": "0x0000A000"}))
    t += jitter(rng, 20, 10)
    ev.append(
        Event(
            t,
            "fw",
            "FW",
            "INFO",
            "Firmware init start",
            {"fw_version": meta["fw_version"], "stepping": meta["stepping"]},
        )
    )

    # DDR init
    t += jitter(rng, 60, 20)
    ev.append(
        Event(
            t, "ddr", "DDR", "INFO", "DDR init start", {"dram_freq_mhz": str(meta["dram_freq_mhz"])}
        )
    )
    t += jitter(rng, 70, 25)
    ev.append(Event(t, "ddr", "DDR", "INFO", "Write leveling PASS", {}))
    t += jitter(rng, 60, 25)
    ev.append(Event(t, "ddr", "DDR", "INFO", "Read training PASS", {}))

    # PCIe init
    t += jitter(rng, 50, 20)
    ev.append(
        Event(
            t,
            "pcie",
            "PCIE",
            "INFO",
            "Controller reset",
            {"gen": str(meta["pcie_gen"]), "lanes": f"x{meta['pcie_lanes']}"},
        )
    )
    t += jitter(rng, 30, 15)
    ev.append(Event(t, "pcie", "PCIE", "INFO", "LTSSM -> Detect", {"lane": "0"}))
    t += jitter(rng, 60, 25)
    ev.append(
        Event(
            t,
            "pcie",
            "PCIE",
            "INFO",
            "Link up",
            {"width": f"x{meta['pcie_lanes']}", "speed": f"{meta['pcie_gen'] * 8}GT/s"},
        )
    )

    # test start marker
    t += jitter(rng, 40, 20)
    ev.append(
        Event(
            t,
            "system",
            "SYSTEM",
            "INFO",
            "Test start",
            {"suite": meta["test_suite"], "case": meta["test_case"], "workload": meta["workload"]},
        )
    )
    return ev


def split_by_source(events: list[Event]) -> dict[str, list[Event]]:
    buckets: dict[str, list[Event]] = {s: [] for s in SOURCES}
    for e in events:
        buckets[e.src].append(e)
    for s in SOURCES:
        buckets[s].sort(key=lambda x: x.ts_ms)
    return buckets


def choose_patterns(rng: random.Random, meta: dict) -> list[tuple[str, Callable]]:
    """
    Choose 1-2 patterns with realistic bias:
      high temp -> thermal, pcie, cdc-ish
      low voltage -> pmu droop, ddr ecc
      heavy workload -> noc timeout, droop, fw exception
      stepping A0 -> more failures
    """
    weights = []
    for pid, _fn in PATTERNS:
        w = 1.0

        # biases
        if meta["temp_c"] >= 85 and pid in ["P_THERMAL_TRIP", "P_PCIE_LTSSM_TIMEOUT"]:
            w *= 1.8
        if meta["voltage_mv"] <= 850 and pid in ["P_PMU_DROOP_WDOG_RESET", "P_DDR_ECC_UE"]:
            w *= 1.7
        if meta["workload"] in ["io_stress", "mixed"] and pid in [
            "P_NOC_TXN_TIMEOUT",
            "P_PMU_DROOP_WDOG_RESET",
        ]:
            w *= 1.4
        if meta["stepping"] == "A0":
            w *= 1.2

        weights.append(w)

    k = 1 if rng.random() < 0.78 else 2
    return rng.choices(PATTERNS, weights=weights, k=k)


def gen_run(
    run_id: int,
    out_logs: str,
    out_labels: str,
    seed: int,
    anomaly_rate: float = 0.35,
    noise_lines: tuple[int, int] = (6000, 18000),
) -> None:
    rng = random.Random(seed)
    meta = gen_meta(rng, run_id)
    meta["seed"] = seed

    events: list[Event] = []
    events.extend(gen_boot_sequence(rng, meta))

    # start time from last boot event
    t = max(e.ts_ms for e in events) if events else 0

    # add heavy telemetry noise (this makes logs feel "big")
    n_noise = rng.randint(*noise_lines)
    t = add_telemetry_noise(rng, events, t, meta, n_noise)

    injected: list[dict] = []
    anomalous = rng.random() < anomaly_rate

    if anomalous:
        chosen = choose_patterns(rng, meta)
        # inject near the end of test (so it fails)
        inj_start = max(0, t - rng.randint(12000, 50000))
        for _pid, fn in chosen:
            ev_inj, lbl = fn(rng, inj_start + rng.randint(0, 7000), meta)
            injected.append(lbl)
            events.extend(ev_inj)
    else:
        # pass ending
        t += jitter(rng, 120, 60)
        events.append(Event(t, "system", "SYSTEM", "INFO", "Test PASS", {"result": "0"}))

    # sort
    events.sort(key=lambda e: e.ts_ms)

    # Write logs per source
    os.makedirs(out_logs, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    buckets = split_by_source(events)
    for src, evs in buckets.items():
        path = os.path.join(out_logs, f"run_{run_id:06d}.{src}.log")
        with open(path, "w") as f:
            for ln in header_lines(meta):
                f.write(ln + "\n")
            f.write("\n")
            for e in evs:
                for ln in event_to_lines(e):
                    f.write(ln + "\n")

    # labels
    label = {
        "run_id": meta["run_id"],
        "anomalous": anomalous,
        "failure_types": sorted({x["failure_type"] for x in injected}) if injected else [],
        "root_cause_id": injected[0]["root_cause_id"] if injected else None,
        "injected_patterns": injected,
        "meta": meta,  # helpful for evaluation joins
    }

    with open(os.path.join(out_labels, f"run_{run_id:06d}.label.json"), "w") as f:
        json.dump(label, f, indent=2)


def generate(
    runs: int = 100,
    out_logs: str = "data/silicon_logs",
    out_labels: str = "data/labels",
    base_seed: int = 424200,
    anomaly_rate: float = 0.35,
    noise_lines: tuple[int, int] = (8000, 22000),
) -> tuple[str, str, int]:
    """Generate ``runs`` synthetic multi-source validation runs.

    Returns ``(out_logs, out_labels, runs)`` so callers can report paths.
    """
    for i in range(runs):
        gen_run(
            run_id=i,
            out_logs=out_logs,
            out_labels=out_labels,
            seed=base_seed + i,
            anomaly_rate=anomaly_rate,
            noise_lines=noise_lines,
        )
    return out_logs, out_labels, runs


def main():
    out_logs, out_labels, runs = generate()
    print(f"Generated {runs} multi-source runs in {out_logs} and labels in {out_labels}")


if __name__ == "__main__":
    main()

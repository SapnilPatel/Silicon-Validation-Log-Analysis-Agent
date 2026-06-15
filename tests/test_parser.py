"""Tests for ingest discovery and log parsing."""

from __future__ import annotations

from svla_agent.ingest.reader import discover_runs, find_run
from svla_agent.models import LogLevel
from svla_agent.parse.parser import parse_header, parse_run


def test_discover_groups_sources(dataset):
    runs = discover_runs(dataset["logs"])
    assert len(runs) == 14
    # Each run is split across multiple per-source files.
    assert all(len(r.sources) >= 2 for r in runs)
    assert {s.source for s in runs[0].sources} <= {
        "system",
        "fw",
        "pcie",
        "ddr",
        "pmu",
        "thermal",
        "noc",
    }


def test_find_run_accepts_padded_and_bare_ids(dataset):
    assert find_run(dataset["logs"], "000000") is not None
    assert find_run(dataset["logs"], "0") is not None
    assert find_run(dataset["logs"], "does-not-exist") is None


def test_parse_header_extracts_typed_meta():
    meta = parse_header(
        [
            "# run_id: 000007",
            "# soc: AURORA-X1",
            "# temp_c: 95",
            "# seed: None",
            "# pcie_gen: 4",
        ]
    )
    assert meta.run_id == "000007"
    assert meta.soc == "AURORA-X1"
    assert meta.temp_c == 95 and isinstance(meta.temp_c, int)
    assert meta.pcie_gen == 4


def test_parse_run_orders_events_and_keeps_kv(dataset):
    run = find_run(dataset["logs"], "000000")
    parsed = parse_run(run)
    assert parsed.n_events > 0
    ts = [e.ts_ms for e in parsed.events]
    assert ts == sorted(ts)
    # Templates are mined for every event.
    assert all(e.template for e in parsed.events)
    # kv pairs are parsed where present.
    assert any(e.kv for e in parsed.events)


def test_multiline_blocks_attach_to_event(dataset):
    """At least one run should contain a register-dump / stack-trace block."""
    found = False
    for run in discover_runs(dataset["logs"]):
        parsed = parse_run(run)
        for e in parsed.events:
            if e.multiline:
                found = True
                assert e.level.rank >= LogLevel.ERROR.rank
                break
        if found:
            break
    assert found, "expected at least one multi-line block across the dataset"

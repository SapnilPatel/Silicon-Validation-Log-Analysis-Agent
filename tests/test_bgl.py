"""Loghub BGL adapter: parsing, windowing, labels, and end-to-end through the pipeline."""

from __future__ import annotations

from pathlib import Path

from svla_agent.classify.classifier import classify_event, classify_run
from svla_agent.datasets.bgl import convert, parse_bgl_line
from svla_agent.ingest.reader import discover_runs
from svla_agent.labels import load_labels, lookup_root_cause, root_cause_map
from svla_agent.models import Event, LogLevel
from svla_agent.parse.parser import parse_run

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "bgl" / "BGL_2k.log"


def test_parse_bgl_line_normal_and_alert():
    normal = parse_bgl_line(
        "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 "
        "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
    )
    assert (
        normal and not normal.is_alert and normal.component == "KERNEL" and normal.level == "INFO"
    )
    alert = parse_bgl_line(
        "KERNDTLB 1118800000 2005.06.14 R30-M0-N9-C:J16-U01 2005-06-14-18.46.40.000000 "
        "R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt"
    )
    assert alert and alert.is_alert and alert.label == "KERNDTLB" and alert.level == "FATAL"
    assert parse_bgl_line("garbage") is None


def test_convert_sample_into_windows(tmp_path: Path):
    stats = convert(SAMPLE, tmp_path / "logs", tmp_path / "labels", window_s=1800, min_lines=5)
    assert stats["lines"] == 2000 and stats["unparsed_lines"] == 0
    assert stats["alert_lines"] == 143
    assert stats["windows"] >= 5 and stats["anomalous_windows"] >= 1
    labels = load_labels(tmp_path / "labels")
    assert len(labels) == stats["windows"]
    anomalous = [lab for lab in labels.values() if lab["anomalous"]]
    assert anomalous and all(lab["root_cause_id"] and lab["failure_types"] for lab in anomalous)
    assert all(lab["source"] == "loghub/BGL" for lab in labels.values())


def test_bgl_runs_flow_through_parser_and_classifier(tmp_path: Path):
    convert(SAMPLE, tmp_path / "logs", tmp_path / "labels", window_s=1800, min_lines=5)
    runs = discover_runs(tmp_path / "logs")
    assert runs and all(sf.source == "bgl" for r in runs for sf in r.sources)
    parsed = parse_run(runs[0])
    assert parsed.meta.soc == "BlueGene/L" and parsed.meta.test_suite == "bgl"
    assert parsed.n_events > 0 and all(e.template for e in parsed.events)
    assert all(e.kv.get("node") for e in parsed.events)
    # Unknown components get a stable component-level failure type, not UNKNOWN.
    fails = [f for r in runs for f in classify_run(parse_run(r))]
    assert fails and all(f.failure_type.endswith("_FAILURE") for f in fails)
    assert {f.failure_type for f in fails} >= {"KERNEL_FAILURE"}
    rc_map = root_cause_map(load_labels(tmp_path / "labels"))
    truths = [lookup_root_cause(rc_map, f.failure_id) for f in fails]
    assert any(t and t.startswith("KERN") for t in truths)


def test_component_fallback_is_sanitised():
    ev = Event(ts_ms=0, source="x", subsys="disc-overy/2", level=LogLevel.ERROR, message="boom")
    assert classify_event(ev) == "DISC_OVERY_2_FAILURE"

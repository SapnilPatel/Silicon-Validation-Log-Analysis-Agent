"""Redaction, fingerprinting and corpus-level dedup."""

from __future__ import annotations

from svla_agent.classify.classifier import deduplicate, fingerprint_for
from svla_agent.models import Event, Failure, LogLevel, Severity
from svla_agent.privacy.redact import redact_event, redact_kv, redact_text


def test_redact_text_scrubs_pii_and_secrets():
    line = (
        "[00:00:01.000] FW: WARN: Upload by ops@example.com from 10.1.2.3 "
        "mac=aa:bb:cc:dd:ee:ff token=abc123 serial=SN-9981 path=/Users/jane/x"
    )
    out = redact_text(line)
    assert "ops@example.com" not in out and "<EMAIL>" in out
    assert "10.1.2.3" not in out and "<IPV4>" in out
    assert "aa:bb:cc:dd:ee:ff" not in out and "<MAC>" in out
    assert "abc123" not in out and "token=<SECRET>" in out
    assert "SN-9981" not in out and "serial=<ID>" in out
    assert "/Users/jane" not in out and "<HOME>" in out


def test_redact_text_leaves_register_dumps_alone():
    dump = "40010000: DEADBEEF 0x40010004: 00000001 rc=-110 lane=3"
    assert redact_text(dump) == dump


def test_redact_kv_by_key_name():
    kv = redact_kv({"api_key": "sk-live-1", "chip_id": "0xABC", "lane": "3"})
    assert kv == {"api_key": "<SECRET>", "chip_id": "<ID>", "lane": "3"}


def test_redact_event_in_place():
    ev = Event(
        ts_ms=1,
        source="fw",
        subsys="FW",
        level=LogLevel.WARN,
        message="Contact admin@corp.io",
        kv={"password": "hunter2"},
        multiline=["user=/home/bob/.ssh"],
        raw="[00:00:00.001] FW: WARN: Contact admin@corp.io | password=hunter2",
    )
    redact_event(ev)
    assert ev.message == "Contact <EMAIL>"
    assert ev.kv["password"] == "<SECRET>"
    assert ev.multiline == ["user=<HOME>/.ssh"]
    assert "hunter2" not in ev.raw and "admin@corp.io" not in ev.raw


def _failure(run_id: str, sig: str) -> Failure:
    return Failure(
        failure_id=f"{run_id}:X",
        run_id=run_id,
        failure_type="X",
        severity=Severity.HIGH,
        subsys="X",
        first_ts_ms=0,
        last_ts_ms=1,
        summary="s",
        signature=sig,
    )


def test_deduplicate_counts_recurrences():
    fails = [_failure("1", "A"), _failure("2", "A"), _failure("3", "B")]
    deduplicate(fails)
    assert [f.occurrences for f in fails] == [2, 2, 1]
    assert fails[0].fingerprint == fails[1].fingerprint == fingerprint_for("A")
    assert fails[2].fingerprint != fails[0].fingerprint

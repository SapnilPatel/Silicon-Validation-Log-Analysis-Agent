"""Parse raw silicon logs into structured :class:`~svla_agent.models.Event` objects.

Two layers of structure are recovered:

1. **Line structure** - a regex pulls timestamp, subsystem, level, message and
   ``key=value`` pairs out of each event line; indented blocks (register dumps,
   stack traces) are attached to the preceding event.
2. **Event templates** - Drain3 mines a stable template per message (variables
   masked, e.g. ``Link training timeout`` regardless of lane/rc), which gives a
   robust signature for dedup, clustering and RAG.
"""

from __future__ import annotations

import re

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from ..ingest.reader import RunFiles
from ..models import Event, LogLevel, ParsedRun, RunMeta

_EVENT_RE = re.compile(
    r"^\[(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})\]\s+"
    r"(?P<subsys>\S+):\s+(?P<level>INFO|WARN|ERROR|FATAL):\s+(?P<rest>.*)$"
)
_HEADER_RE = re.compile(r"^#\s*(?P<key>[^:]+):\s*(?P<value>.*)$")
_INDENT = "    "

# RunMeta fields that should be coerced to int when present.
_INT_META_FIELDS = {
    "voltage_mv",
    "temp_c",
    "clk_mhz",
    "dram_freq_mhz",
    "pcie_gen",
    "pcie_lanes",
}
_KNOWN_META_FIELDS = set(RunMeta.model_fields) - {"run_id", "extra"}


def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def _parse_kv(rest: str) -> tuple[str, dict[str, str]]:
    """Split an event ``rest`` into (message, kv-pairs)."""
    if " | " in rest:
        message, kv_str = rest.split(" | ", 1)
    else:
        return rest.strip(), {}
    kv: dict[str, str] = {}
    for token in kv_str.split():
        if "=" in token:
            key, value = token.split("=", 1)
            kv[key] = value
    return message.strip(), kv


def _new_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def _coerce_meta_value(key: str, value: str) -> object:
    if value in {"", "None"}:
        return None
    if key in _INT_META_FIELDS:
        try:
            return int(value)
        except ValueError:
            return value
    return value


def parse_header(lines: list[str]) -> RunMeta:
    """Build :class:`RunMeta` from ``# key: value`` header lines."""
    fields: dict[str, object] = {}
    extra: dict[str, str] = {}
    for line in lines:
        match = _HEADER_RE.match(line)
        if not match:
            continue
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key in _KNOWN_META_FIELDS or key == "run_id":
            fields[key] = _coerce_meta_value(key, value)
        elif value not in {"", "None"}:
            extra[key] = value
    fields.setdefault("run_id", "unknown")
    fields["extra"] = extra
    return RunMeta(**fields)


def parse_run(run: RunFiles) -> ParsedRun:
    """Parse all source files of a run into an ordered :class:`ParsedRun`."""
    miner = _new_template_miner()
    header_lines: list[str] = []
    events: list[Event] = []

    for source, line in run.iter_lines():
        if not line.strip():
            continue
        if line.startswith("#"):
            header_lines.append(line)
            continue
        if line.startswith(_INDENT):
            if events:
                events[-1].multiline.append(line[len(_INDENT) :])
            continue

        match = _EVENT_RE.match(line)
        if not match:
            # Unknown free-form line: attach to the current event as context.
            if events:
                events[-1].multiline.append(line.strip())
            continue

        message, kv = _parse_kv(match.group("rest"))
        template = miner.add_log_message(message)["template_mined"]
        events.append(
            Event(
                ts_ms=_ts_to_ms(match.group("h"), match.group("m"), match.group("s"), match.group("ms")),
                source=source,
                subsys=match.group("subsys"),
                level=LogLevel(match.group("level")),
                message=message,
                kv=kv,
                template=template,
                raw=line,
            )
        )

    events.sort(key=lambda e: e.ts_ms)
    meta = parse_header(header_lines)
    if meta.run_id in {"unknown", ""}:
        meta.run_id = run.run_id
    return ParsedRun(run_id=meta.run_id, meta=meta, events=events)

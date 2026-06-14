"""Streaming ingest of raw log files.

A *run* is a set of per-source files sharing a run id, e.g.::

    run_000007.pcie.log  run_000007.ddr.log  run_000007.system.log ...

The reader discovers runs under a directory, transparently handles ``.gz``
files, and yields lines lazily so we never hold a whole multi-GB run in memory.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# run_<id>.<source>.log  (optionally .gz)
_RUN_FILE_RE = re.compile(r"^run_(?P<run_id>\d+)\.(?P<source>[a-zA-Z0-9_]+)\.log(?:\.gz)?$")


@dataclass(frozen=True)
class SourceFile:
    """One physical log file belonging to a run."""

    run_id: str
    source: str
    path: Path

    def open_text(self) -> Iterator[str]:
        """Yield decoded lines (newline-stripped), supporting gzip transparently."""
        if self.path.suffix == ".gz":
            with gzip.open(self.path, "rt", encoding="utf-8", errors="replace") as fh:
                yield from (line.rstrip("\n") for line in fh)
        else:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                yield from (line.rstrip("\n") for line in fh)


@dataclass
class RunFiles:
    """All source files discovered for a single run."""

    run_id: str
    sources: list[SourceFile] = field(default_factory=list)

    def iter_lines(self) -> Iterator[tuple[str, str]]:
        """Yield ``(source, line)`` for every line across all source files."""
        for sf in sorted(self.sources, key=lambda s: s.source):
            for line in sf.open_text():
                yield sf.source, line


def _iter_candidate_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("run_*.log"))
    yield from sorted(root.rglob("run_*.log.gz"))


def discover_runs(path: str | Path) -> list[RunFiles]:
    """Discover and group all runs found under ``path`` (a file or directory)."""
    root = Path(path)
    runs: dict[str, RunFiles] = {}
    for file_path in _iter_candidate_files(root):
        m = _RUN_FILE_RE.match(file_path.name)
        if not m:
            continue
        run_id = m.group("run_id")
        runs.setdefault(run_id, RunFiles(run_id=run_id)).sources.append(
            SourceFile(run_id=run_id, source=m.group("source"), path=file_path)
        )
    return [runs[k] for k in sorted(runs)]


def find_run(path: str | Path, run_id: str) -> RunFiles | None:
    """Return the :class:`RunFiles` for a specific run id, if present."""
    normalized = run_id.zfill(6) if run_id.isdigit() else run_id
    for run in discover_runs(path):
        if run.run_id in {run_id, normalized}:
            return run
    return None

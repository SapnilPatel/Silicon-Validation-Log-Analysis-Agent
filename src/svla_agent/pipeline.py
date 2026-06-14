"""End-to-end orchestration: logs -> parse -> classify -> cluster -> RAG -> report.

The :class:`AnalysisPipeline` ties the stages together. Two entry points:

* :meth:`run` — analyse a whole corpus (parse + classify + cluster + index +
  hypothesise + write reports).
* :meth:`process_run` — analyse a single run against an already-populated store.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .classify.classifier import classify_run
from .cluster.clusterer import cluster_failures
from .config import Settings, get_settings
from .embeddings.embedder import Embedder
from .hypothesize.agent import HypothesisAgent
from .ingest.reader import RunFiles, discover_runs, find_run
from .labels import load_labels, root_cause_map
from .logging_config import get_logger
from .models import Failure, FailureCluster, ParsedRun, RunReport
from .parse.parser import parse_run
from .rag.store import VectorStore
from .report.reporter import Reporter

logger = get_logger(__name__)

ProgressCb = Callable[[str, int, int], None]


@dataclass
class PipelineResult:
    reports: list[RunReport] = field(default_factory=list)
    clusters: list[FailureCluster] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)

    @property
    def anomalous_runs(self) -> list[RunReport]:
        return [r for r in self.reports if r.anomalous]


class AnalysisPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        agent: HypothesisAgent | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore(self.settings, self.embedder)
        self.agent = agent or HypothesisAgent(self.settings)
        self.reporter = reporter or Reporter(self.settings)

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _build_report(
        self,
        parsed: ParsedRun,
        failures: list[Failure],
        clusters_by_fid: dict[str, FailureCluster],
        *,
        hypothesise: bool,
    ) -> RunReport:
        hypotheses = []
        if hypothesise:
            for failure in failures:
                similar = self.store.query(
                    failure.signature, exclude_failure_id=failure.failure_id
                )
                hypotheses.append(self.agent.generate(failure, similar, failure.cluster_id))

        present = {
            c.cluster_id: c for c in (clusters_by_fid.get(f.failure_id) for f in failures) if c
        }
        return RunReport(
            run_id=parsed.run_id,
            meta=parsed.meta,
            anomalous=bool(failures),
            n_events=parsed.n_events,
            failures=failures,
            clusters=sorted(present.values(), key=lambda c: c.cluster_id),
            hypotheses=hypotheses,
            generated_at=self._now(),
        )

    # -- whole-corpus flow -------------------------------------------------- #
    def run(
        self,
        logs_dir: str | None = None,
        labels_dir: str | None = None,
        *,
        index: bool = True,
        write: bool = True,
        hypotheses: bool = True,
        limit: int | None = None,
        progress: ProgressCb | None = None,
    ) -> PipelineResult:
        logs_dir = logs_dir or str(self.settings.logs_dir)
        labels_dir = labels_dir or str(self.settings.labels_dir)

        run_files = discover_runs(logs_dir)
        if limit is not None:
            run_files = run_files[:limit]
        total = len(run_files)
        logger.info("Discovered %d runs under %s", total, logs_dir)

        # Parse + classify every run.
        parsed_runs: list[ParsedRun] = []
        failures_by_run: dict[str, list[Failure]] = {}
        all_failures: list[Failure] = []
        for i, rf in enumerate(run_files, start=1):
            parsed = parse_run(rf)
            fails = classify_run(parsed)
            parsed_runs.append(parsed)
            failures_by_run[parsed.run_id] = fails
            all_failures.extend(fails)
            if progress:
                progress("parse", i, total)

        # Cluster the corpus of failures into families.
        all_failures, clusters = cluster_failures(all_failures, self.embedder)
        clusters_by_fid = {fid: c for c in clusters for fid in c.failure_ids}

        # Seed the RAG store (with ground-truth root causes when labels exist).
        if index and all_failures:
            rc_map = root_cause_map(load_labels(labels_dir))
            self.store.index(all_failures, rc_map)

        # Build per-run reports (+ hypotheses for failing runs).
        reports: list[RunReport] = []
        for i, parsed in enumerate(parsed_runs, start=1):
            fails = failures_by_run[parsed.run_id]
            report = self._build_report(
                parsed, fails, clusters_by_fid, hypothesise=bool(fails) and hypotheses
            )
            reports.append(report)
            if write:
                self.reporter.write(report)
            if progress:
                progress("report", i, total)

        logger.info(
            "Processed %d runs: %d anomalous, %d failures, %d clusters",
            total,
            sum(r.anomalous for r in reports),
            len(all_failures),
            len(clusters),
        )
        return PipelineResult(reports=reports, clusters=clusters, failures=all_failures)

    # -- single-run flow ---------------------------------------------------- #
    def process_run(
        self,
        run: str | RunFiles,
        logs_dir: str | None = None,
        *,
        write: bool = True,
    ) -> RunReport:
        """Analyse a single run against the already-indexed store."""
        if isinstance(run, str):
            rf = find_run(logs_dir or str(self.settings.logs_dir), run)
            if rf is None:
                raise FileNotFoundError(f"Run '{run}' not found under {logs_dir}")
        else:
            rf = run

        parsed = parse_run(rf)
        failures = classify_run(parsed)
        failures, clusters = cluster_failures(failures, self.embedder)
        clusters_by_fid = {fid: c for c in clusters for fid in c.failure_ids}
        report = self._build_report(parsed, failures, clusters_by_fid, hypothesise=bool(failures))
        if write:
            self.reporter.write(report)
        return report

    # -- indexing only ------------------------------------------------------ #
    def index_corpus(self, logs_dir: str | None = None, labels_dir: str | None = None) -> int:
        """(Re)build the RAG store from all runs; returns number of failures indexed."""
        logs_dir = logs_dir or str(self.settings.logs_dir)
        labels_dir = labels_dir or str(self.settings.labels_dir)
        failures: list[Failure] = []
        for rf in discover_runs(logs_dir):
            failures.extend(classify_run(parse_run(rf)))
        if not failures:
            return 0
        failures, _ = cluster_failures(failures, self.embedder)
        rc_map = root_cause_map(load_labels(labels_dir))
        self.store.reset()
        return self.store.index(failures, rc_map)

"""``svla`` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import get_settings
from .logging_config import setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SVLA-Agent — agentic debug assistant for post-silicon validation logs.",
)
console = Console()


def _init(log_level: str | None = None) -> None:
    settings = get_settings()
    setup_logging(log_level or settings.log_level)


@app.command()
def version() -> None:
    """Print the SVLA-Agent version."""
    console.print(f"svla-agent [bold cyan]{__version__}[/bold cyan]")


@app.command()
def generate(
    runs: int = typer.Option(100, help="Number of synthetic runs to generate."),
    out_logs: Path = typer.Option(Path("data/silicon_logs"), help="Output dir for logs."),
    out_labels: Path = typer.Option(Path("data/labels"), help="Output dir for labels."),
    seed: int = typer.Option(424200, help="Base RNG seed."),
) -> None:
    """Generate the synthetic silicon-log dataset."""
    _init()
    from .datagen.generator import generate as _gen

    with console.status(f"Generating {runs} runs..."):
        logs, labels, n = _gen(
            runs=runs, out_logs=str(out_logs), out_labels=str(out_labels), base_seed=seed
        )
    console.print(f"[green]Generated[/green] {n} runs in [bold]{logs}[/bold] (labels: {labels})")


@app.command()
def process(
    run: str | None = typer.Option(None, help="Process a single run id (e.g. 000007)."),
    logs: Path | None = typer.Option(None, help="Logs directory (default from settings)."),
    labels: Path | None = typer.Option(None, help="Labels directory (default from settings)."),
    limit: int | None = typer.Option(None, help="Process only the first N runs."),
    no_index: bool = typer.Option(False, "--no-index", help="Do not (re)index the RAG store."),
    no_write: bool = typer.Option(False, "--no-write", help="Do not write report files."),
) -> None:
    """Run the full analysis pipeline and write debug reports."""
    _init()
    from .pipeline import AnalysisPipeline

    settings = get_settings()
    pipeline = AnalysisPipeline(settings)
    logs_dir = str(logs) if logs else None

    if run:
        report = pipeline.process_run(run, logs_dir=logs_dir, write=not no_write)
        _print_run_summary(report)
        return

    result = pipeline.run(
        logs_dir=logs_dir,
        labels_dir=str(labels) if labels else None,
        index=not no_index,
        write=not no_write,
        limit=limit,
        progress=_progress,
    )
    _print_corpus_summary(result)
    if not settings.llm_enabled:
        console.print(
            "[yellow]Note:[/yellow] no ANTHROPIC_API_KEY set — hypotheses used the "
            "rule-based fallback. Set the key (or unset SVLA_OFFLINE) for Claude."
        )


@app.command()
def index(
    logs: Path | None = typer.Option(None, help="Logs directory."),
    labels: Path | None = typer.Option(None, help="Labels directory."),
) -> None:
    """(Re)build the RAG vector store of historical failures."""
    _init()
    from .pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline(get_settings())
    with console.status("Indexing historical failures..."):
        n = pipeline.index_corpus(
            logs_dir=str(logs) if logs else None,
            labels_dir=str(labels) if labels else None,
        )
    console.print(f"[green]Indexed[/green] {n} failures into the vector store.")


@app.command(name="eval")
def evaluate_cmd(
    logs: Path | None = typer.Option(None, help="Logs directory."),
    labels: Path | None = typer.Option(None, help="Labels directory."),
    limit: int | None = typer.Option(None, help="Evaluate only the first N runs."),
    out: Path | None = typer.Option(None, help="Write metrics JSON to this path."),
) -> None:
    """Evaluate pipeline output against ground-truth labels."""
    _init()
    from .eval.metrics import evaluate
    from .labels import load_labels
    from .pipeline import AnalysisPipeline

    settings = get_settings()
    pipeline = AnalysisPipeline(settings)
    labels_dir = str(labels) if labels else str(settings.labels_dir)

    result = pipeline.run(
        logs_dir=str(logs) if logs else None,
        labels_dir=labels_dir,
        index=True,
        write=False,
        hypotheses=False,
        limit=limit,
        progress=_progress,
    )
    metrics = evaluate(
        result, load_labels(labels_dir), embedder=pipeline.embedder, store=pipeline.store
    )
    _print_metrics(metrics)
    if out:
        out.write_text(json.dumps(metrics, indent=2))
        console.print(f"[green]Wrote[/green] metrics to {out}")


# --------------------------------------------------------------------------- #
# Pretty-printing helpers
# --------------------------------------------------------------------------- #


def _progress(stage: str, i: int, total: int) -> None:
    if i == total or i % max(1, total // 10) == 0:
        console.print(f"  [{stage}] {i}/{total}", style="dim")


def _print_run_summary(report) -> None:
    status = "[red]ANOMALOUS[/red]" if report.anomalous else "[green]PASS[/green]"
    console.print(f"\nRun [bold]{report.run_id}[/bold]: {status}  ({report.n_events:,} events)")
    for f in report.failures:
        console.print(f"  • {f.failure_type} [{f.severity.value}] in {f.subsys}")
    for h in report.hypotheses:
        console.print(
            f"    ↳ {h.root_cause}  ([italic]{int(h.confidence * 100)}%, {h.source}[/italic])"
        )


def _print_corpus_summary(result) -> None:
    table = Table(title="Pipeline summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Runs processed", str(len(result.reports)))
    table.add_row("Anomalous runs", str(len(result.anomalous_runs)))
    table.add_row("Failures", str(len(result.failures)))
    table.add_row("Clusters", str(len(result.clusters)))
    console.print(table)


def _print_metrics(metrics: dict) -> None:
    det = metrics["detection"]
    console.print("\n[bold]Anomaly detection[/bold]")
    console.print(
        f"  precision={det['precision']}  recall={det['recall']}  "
        f"f1={det['f1']}  accuracy={det['accuracy']}"
    )
    ft = metrics["failure_typing"]["micro"]
    console.print("[bold]Failure typing (micro)[/bold]")
    console.print(f"  precision={ft['precision']}  recall={ft['recall']}  f1={ft['f1']}")
    cl = metrics["clustering"]
    console.print("[bold]Clustering[/bold]")
    console.print("  " + "  ".join(f"{k}={v}" for k, v in cl.items()))
    if "retrieval" in metrics:
        rt = metrics["retrieval"]
        console.print("[bold]Retrieval[/bold]")
        console.print("  " + "  ".join(f"{k}={v}" for k, v in rt.items()))


if __name__ == "__main__":
    app()

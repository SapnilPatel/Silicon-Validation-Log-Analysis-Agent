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
    agentic: bool = typer.Option(
        False,
        "--agentic",
        help="Let Claude investigate with tools (needs a key; several calls/failure).",
    ),
    multi_agent: bool = typer.Option(
        False,
        "--multi-agent",
        help="LangGraph team: triage -> subsystem specialists -> reviewer (several calls/failure).",
    ),
) -> None:
    """Run the full analysis pipeline and write debug reports."""
    _init()
    from .pipeline import AnalysisPipeline

    settings = get_settings()
    if multi_agent:
        settings = settings.model_copy(update={"llm_mode": "multiagent"})
    elif agentic:
        settings = settings.model_copy(update={"llm_mode": "agentic"})
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
    holdout: float | None = typer.Option(
        None,
        help="Fraction of runs held out for retrieval/root-cause eval (default from settings).",
    ),
    out: Path | None = typer.Option(None, help="Write metrics JSON to this path."),
    llm: bool = typer.Option(
        False,
        "--llm",
        help="Score Claude as the hypothesis agent (one paid API call per held-out failure). "
        "Default: the deterministic rule-based agent, even if an API key is configured.",
    ),
    agentic: bool = typer.Option(
        False,
        "--agentic",
        help="With --llm: tool-using investigation loop (several calls/failure).",
    ),
    multi_agent: bool = typer.Option(
        False,
        "--multi-agent",
        help="LangGraph team: triage -> subsystem specialists -> reviewer (several calls/failure).",
    ),
) -> None:
    """Evaluate pipeline output against ground-truth labels."""
    _init()
    from .eval.metrics import evaluate
    from .labels import load_labels
    from .pipeline import AnalysisPipeline

    settings = get_settings()
    if not llm and settings.llm_enabled:
        settings = settings.model_copy(update={"anthropic_api_key": None})
        console.print(
            "[dim]Evaluating with the rule-based agent (pass --llm to score Claude).[/dim]"
        )
    elif llm and not settings.llm_enabled:
        raise typer.BadParameter("--llm needs ANTHROPIC_API_KEY (and SVLA_OFFLINE unset)")
    if multi_agent:
        settings = settings.model_copy(update={"llm_mode": "multiagent"})
    elif agentic:
        settings = settings.model_copy(update={"llm_mode": "agentic"})
    pipeline = AnalysisPipeline(settings)
    labels_dir = str(labels) if labels else str(settings.labels_dir)

    result = pipeline.run(
        logs_dir=str(logs) if logs else None,
        labels_dir=labels_dir,
        index=False,
        write=False,
        hypotheses=False,
        limit=limit,
        progress=_progress,
    )
    metrics = evaluate(
        result,
        load_labels(labels_dir),
        embedder=pipeline.embedder,
        store=pipeline.store,
        agent=pipeline.agent,
        holdout=settings.eval_holdout if holdout is None else holdout,
        k=settings.rag_top_k,
        cache_path=out.with_suffix(".hypcache.json") if out else None,
    )
    _print_metrics(metrics)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        console.print(f"[green]Wrote[/green] metrics to {out}")


@app.command(name="import-bgl")
def import_bgl(
    src: Path | None = typer.Option(None, help="Path to BGL.log (Loghub). Omit with --download."),
    download: bool = typer.Option(
        False, "--download", help="Fetch BGL.zip from Zenodo into data/bgl_raw."
    ),
    out: Path = typer.Option(
        Path("data/bgl"), help="Output root (creates silicon_logs/ and labels/)."
    ),
    window_min: int = typer.Option(30, help="Window length in minutes (one run per window)."),
    limit: int | None = typer.Option(None, help="Stop after N windows."),
    min_lines: int = typer.Option(20, help="Skip windows with fewer lines."),
) -> None:
    """Convert the real Loghub BGL supercomputer log into runs + labels (OOD evaluation)."""
    _init()
    from .datasets.bgl import convert
    from .datasets.bgl import download as _download

    if download:
        with console.status("Downloading BGL.zip from Zenodo (57 MB)..."):
            src = _download("data/bgl_raw")
    if src is None or not Path(src).exists():
        raise typer.BadParameter("Provide --src path/to/BGL.log or use --download")
    with console.status(f"Converting {src} into {window_min}-minute windows..."):
        stats = convert(
            src,
            out / "silicon_logs",
            out / "labels",
            window_s=window_min * 60,
            limit=limit,
            min_lines=min_lines,
        )
    table = Table(title="BGL import")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in stats.items():
        table.add_row(k, f"{v:,}")
    console.print(table)
    console.print(f"Next: [bold]svla eval --logs {out}/silicon_logs --labels {out}/labels[/bold]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (development)."),
) -> None:
    """Serve the REST API (FastAPI + uvicorn)."""
    _init()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional extra
        raise typer.BadParameter("Install the API extra: pip install 'svla-agent[api]'") from exc
    uvicorn.run("svla_agent.api:app", host=host, port=port, reload=reload)


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
    ho = metrics.get("holdout", {})
    if ho.get("test_runs"):
        console.print(
            f"[bold]Hold-out[/bold] ({ho['train_runs']} train runs / {ho['test_runs']} test runs, "
            f"{ho['test_failures']} test failures)"
        )
    if "retrieval" in metrics:
        rt = metrics["retrieval"]
        console.print("[bold]Retrieval[/bold]")
        console.print("  " + "  ".join(f"{k}={v}" for k, v in rt.items()))
    if "root_cause" in metrics:
        rc = metrics["root_cause"]
        console.print("[bold]Root cause (top-1)[/bold]")
        console.print("  " + "  ".join(f"{k}={v}" for k, v in rc.items() if k != "per_type"))
        for t, v in rc.get("per_type", {}).items():
            console.print(f"    {t}: {v['correct']}/{v['total']} ({v['accuracy']})", style="dim")


if __name__ == "__main__":
    app()

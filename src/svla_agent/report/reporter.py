"""Render :class:`RunReport` objects to JSON and Markdown."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from ..config import Settings, get_settings
from ..models import RunReport

_env = Environment(
    loader=PackageLoader("svla_agent.report", "templates"),
    autoescape=select_autoescape(enabled_extensions=()),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_markdown(report: RunReport) -> str:
    """Render a single run report to Markdown."""
    return _env.get_template("run_report.md.j2").render(r=report)


class Reporter:
    """Write run reports to ``reports_dir`` as ``.json`` + ``.md`` pairs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.out_dir = self.settings.reports_dir

    def write(self, report: RunReport) -> dict[str, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"run_{report.run_id}"
        json_path = self.out_dir / f"{stem}.json"
        md_path = self.out_dir / f"{stem}.md"
        json_path.write_text(report.model_dump_json(indent=2))
        md_path.write_text(render_markdown(report))
        return {"json": json_path, "markdown": md_path}

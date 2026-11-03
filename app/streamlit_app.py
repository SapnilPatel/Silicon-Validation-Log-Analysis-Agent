"""SVLA-Agent dashboard.

Browse analysed runs, failure families, root-cause hypotheses and evaluation
metrics; analyse a run live (optionally with your own Anthropic key).

Run with:  ``streamlit run app/streamlit_app.py``
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make the src-layout package importable when run via `streamlit run`.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _export_secrets_to_env() -> None:
    """Hosted Streamlit keeps config in ``st.secrets``; our settings read the environment."""
    try:
        for key, value in st.secrets.items():
            if (key.startswith("SVLA_") or key == "ANTHROPIC_API_KEY") and isinstance(
                value, (str, int, float, bool)
            ):
                os.environ.setdefault(key, str(value))
    except Exception:  # no secrets configured - perfectly normal locally
        pass


_export_secrets_to_env()
# The dashboard favours the dependency-free in-memory store unless told otherwise:
# Chroma's native bindings are not available on every hosting platform/Python.
os.environ.setdefault("SVLA_VECTOR_BACKEND", "memory")

from svla_agent import __version__  # noqa: E402
from svla_agent.config import Settings, get_settings  # noqa: E402
from svla_agent.hypothesize.agent import HypothesisAgent  # noqa: E402
from svla_agent.hypothesize.knowledge import ROOT_CAUSES  # noqa: E402
from svla_agent.ingest.reader import discover_runs  # noqa: E402
from svla_agent.pipeline import AnalysisPipeline  # noqa: E402
from svla_agent.report.reporter import render_markdown  # noqa: E402

st.set_page_config(page_title="SVLA-Agent", page_icon="🔬", layout="wide")

# --------------------------------------------------------------------------- #
# Visual constants
# --------------------------------------------------------------------------- #
# Fixed categorical hue per failure type (colour follows the entity, never its
# rank), taken from a CVD-validated palette; adjacent slots stay distinguishable.
TYPE_COLORS: dict[str, str] = {
    "PCIE_TIMEOUT": "#3987e5",
    "DDR_ECC": "#d95926",
    "PMU_DROOP": "#199e70",
    "THERMAL_TRIP": "#c98500",
    "FW_EXCEPTION": "#d55181",
    "NOC_TIMEOUT": "#5cb85c",
    "UNKNOWN_FAILURE": "#9085e9",
}
SITE_URL = "https://sapnilpatel.github.io/Silicon-Validation-Log-Analysis-Agent/"
REPO_URL = "https://github.com/SapnilPatel/Silicon-Validation-Log-Analysis-Agent"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', system-ui, sans-serif; }
code, pre, .stCode { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
.block-container { padding-top: 1.6rem; max-width: 1240px; }
.svla-hero { padding: 26px 28px 22px; border-radius: 16px; border: 1px solid #243040;
  background: radial-gradient(900px 300px at 85% -20%, rgba(57,135,229,.22), transparent 60%),
              radial-gradient(600px 260px at -5% 110%, rgba(25,158,112,.16), transparent 60%), #11161d; margin-bottom: 18px; }
.svla-hero .eyebrow { color: #5fa2ff; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; font-size: .74rem; margin: 0 0 8px; }
.svla-hero h1 { font-size: 2rem; font-weight: 800; letter-spacing: -.01em; margin: 0 0 8px; line-height: 1.15; color: #e6edf3; }
.svla-hero h1 em { font-style: normal; color: #5fa2ff; }
.svla-hero p { color: #a3b3c6; margin: 0; max-width: 820px; font-size: .98rem; }
.svla-hero .links { margin-top: 12px; display: flex; gap: 10px; flex-wrap: wrap; }
.svla-hero .links a { text-decoration: none; font-weight: 600; font-size: .86rem; padding: 6px 12px; border-radius: 9px;
  border: 1px solid #33455c; color: #e6edf3; background: #171e27; }
.svla-hero .links a.primary { background: #3987e5; border-color: #3987e5; color: #fff; }
.svla-chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0 0; }
.svla-chip { background: #171e27; border: 1px solid #243040; border-radius: 999px; padding: 5px 11px; font-size: .8rem; color: #a3b3c6; }
.svla-chip b { color: #e6edf3; font-variant-numeric: tabular-nums; }
.svla-flow { font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: #71839a; margin-top: 12px; }
.svla-flow b { color: #a3b3c6; font-weight: 500; }
.svla-note { color: #71839a; font-size: .85rem; }
div[data-testid="stMetric"] { background: #11161d; border: 1px solid #243040; border-radius: 12px; padding: 12px 14px; }
div[data-testid="stMetric"] label { color: #a3b3c6 !important; }
div[data-testid="stExpander"] { border: 1px solid #243040; border-radius: 12px; background: #11161d; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { padding: 8px 14px; border-radius: 9px 9px 0 0; }
section[data-testid="stSidebar"] { border-right: 1px solid #243040; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_hero(metrics: dict | None, reports: list[dict]) -> None:
    rc = (metrics or {}).get("root_cause", {})
    rt = (metrics or {}).get("retrieval", {})
    n_runs = (metrics or {}).get("n_runs")
    chips = []
    if rc.get("accuracy") is not None:
        chips.append(
            f"<span class='svla-chip'>root-cause top-1 <b>{rc['accuracy']:.2f}</b> "
            f"on {rc.get('n', '?')} held-out failures</span>"
        )
    if rt.get("recall_at_k") is not None:
        chips.append(
            f"<span class='svla-chip'>retrieval recall@{rt.get('k', 5)} <b>{rt['recall_at_k']:.2f}</b></span>"
        )
    if n_runs:
        chips.append(f"<span class='svla-chip'>evaluated on <b>{n_runs}</b> runs</span>")
    if reports:
        chips.append(
            f"<span class='svla-chip'><b>{len(reports)}</b> runs loaded · "
            f"<b>{sum(r['n_events'] for r in reports):,}</b> events</span>"
        )
    st.markdown(
        f"""
<div class="svla-hero">
  <p class="eyebrow">Agentic AI debug assistant · post-silicon validation</p>
  <h1>From raw silicon test logs to a <em>named root cause</em> with the evidence attached.</h1>
  <p>SVLA-Agent streams multi-source logs, mines event templates (Drain3), classifies and de-duplicates failures,
  clusters recurring families (HDBSCAN), retrieves similar historical incidents (RAG) and asks Claude — or a
  deterministic rule-based agent — to pick the specific root cause and the next experiment to run.
  PII and secrets are redacted before anything is indexed or prompted.</p>
  <div class="svla-flow">ingest → parse → <b>redact</b> → classify + dedup → cluster → retrieve → <b>hypothesize</b> → report</div>
  <div class="svla-chips">{"".join(chips)}</div>
  <div class="links">
    <a class="primary" href="{SITE_URL}" target="_blank">Project site &amp; report explorer ↗</a>
    <a href="{REPO_URL}" target="_blank">Source on GitHub ↗</a>
    <a href="{REPO_URL}/blob/main/docs/ARCHITECTURE.md" target="_blank">Architecture ↗</a>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


SEVERITY_ICON = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "🟦"}
GRID = "rgba(128,128,128,0.18)"

EXAMPLES = _ROOT / "examples"


def _color(ftype: str) -> str:
    return TYPE_COLORS.get(ftype, TYPE_COLORS["UNKNOWN_FAILURE"])


def _dark(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font={"family": "Inter, system-ui, sans-serif", "color": "#a3b3c6"},
        title_font={"color": "#e6edf3", "size": 15},
        hoverlabel={"bgcolor": "#0a0e13", "bordercolor": "#33455c", "font": {"color": "#e6edf3"}},
    )
    return fig


def _bar(
    labels: list[str], values: list[float], colors: list[str], title: str, x_title: str
) -> go.Figure:
    """Thin horizontal bars, direct value labels, hover tooltip, recessive grid."""
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{v:g}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x}<extra></extra>",
            width=0.55,
        )
    )
    fig.update_layout(
        title=title,
        height=max(220, 34 * len(labels) + 90),
        margin={"l": 10, "r": 30, "t": 48, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": x_title, "gridcolor": GRID, "zeroline": False},
        yaxis={"autorange": "reversed", "showgrid": False},
        showlegend=False,
    )
    return _dark(fig)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #


def _default_dirs(settings: Settings) -> tuple[Path, Path, Path]:
    """Prefer the configured dirs when populated, else the bundled examples."""
    logs = settings.logs_dir
    labels = settings.labels_dir
    reports = settings.reports_dir
    if not any(logs.glob("run_*")) and (EXAMPLES / "sample_logs").exists():
        logs = EXAMPLES / "sample_logs"
        labels = EXAMPLES / "sample_labels"
    if not any(reports.glob("run_*.json")) and (EXAMPLES / "reports").exists():
        reports = EXAMPLES / "reports"
    return logs, labels, reports


@st.cache_data(show_spinner=False)
def load_reports(reports_dir: str) -> list[dict]:
    root = Path(reports_dir)
    reports = []
    for path in sorted(root.glob("run_*.json")):
        try:
            reports.append(json.loads(path.read_text()))
        except (ValueError, OSError):
            continue
    return reports


@st.cache_data(show_spinner=False)
def load_metrics(reports_dir: str) -> dict | None:
    for name in ("metrics.json", "metrics_st.json", "metrics_offline.json"):
        path = Path(reports_dir) / name
        if path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                continue
    return None


@st.cache_resource(show_spinner=False)
def get_pipeline(logs_dir: str, labels_dir: str, reports_dir: str) -> AnalysisPipeline:
    base = get_settings()
    settings = base.model_copy(
        update={
            "logs_dir": Path(logs_dir),
            "labels_dir": Path(labels_dir),
            "reports_dir": Path(reports_dir),
        }
    )
    return AnalysisPipeline(settings)


def pipeline_with_key(base: AnalysisPipeline, key: str | None) -> AnalysisPipeline:
    """Same embedder/store, but a hypothesis agent using the visitor's own key."""
    if not key:
        return base
    settings = base.settings.model_copy(update={"anthropic_api_key": key, "offline": False})
    return AnalysisPipeline(
        settings=settings,
        embedder=base.embedder,
        store=base.store,
        agent=HypothesisAgent(settings),
        reporter=base.reporter,
    )


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def render_overview(reports: list[dict]) -> None:
    anomalous = [r for r in reports if r["anomalous"]]
    failures = [f for r in reports for f in r["failures"]]
    hyps = [h for r in reports for h in r["hypotheses"]]
    clusters = {c["cluster_id"]: c for r in reports for c in r["clusters"]}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Runs analysed", len(reports))
    c2.metric("Anomalous runs", len(anomalous))
    c3.metric("Failures", len(failures))
    c4.metric("Failure families", len(clusters))
    c5.metric("Events parsed", f"{sum(r['n_events'] for r in reports):,}")

    if not failures:
        st.info("No failures in the loaded reports.")
        return

    left, right = st.columns(2)
    with left:
        counts = Counter(f["failure_type"] for f in failures)
        order = [t for t in TYPE_COLORS if t in counts] + sorted(set(counts) - set(TYPE_COLORS))
        st.plotly_chart(
            _bar(
                order,
                [counts[t] for t in order],
                [_color(t) for t in order],
                "Failures by type",
                "failures",
            ),
            width="stretch",
        )
    with right:
        rc_counts = Counter((h.get("root_cause_id") or "unresolved") for h in hyps)
        rc_type = {rc.rc_id: ftype for ftype, causes in ROOT_CAUSES.items() for rc in causes}
        labels = [rc for rc, _ in rc_counts.most_common(12)]
        st.plotly_chart(
            _bar(
                labels,
                [rc_counts[rc] for rc in labels],
                [_color(rc_type.get(rc, "UNKNOWN_FAILURE")) for rc in labels],
                "Root causes identified",
                "hypotheses",
            ),
            width="stretch",
        )

    st.subheader("Failure families (HDBSCAN clusters)")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "cluster": c["cluster_id"],
                    "label": c["label"],
                    "size": c["size"],
                    "types": ", ".join(c["failure_types"]),
                    "representative signature": c["representative_signature"][:140],
                }
                for c in sorted(clusters.values(), key=lambda c: -c["size"])
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    recurring = Counter(f["fingerprint"] for f in failures if f.get("fingerprint"))
    dup = sum(1 for n in recurring.values() if n > 1)
    if dup:
        st.caption(
            f"Dedup: {dup} failure fingerprints recur across runs "
            f"({sum(n for n in recurring.values() if n > 1)} failures are repeats of a known signature)."
        )


def render_failure(f: dict, hyp: dict | None, key: str = "detail") -> None:
    icon = SEVERITY_ICON.get(f["severity"], "⬜")
    seen = f.get("occurrences", 1)
    title = f"{icon} {f['failure_type']} — {f['severity'].upper()} ({f['subsys']})"
    if seen > 1:
        title += f" · seen {seen}x in corpus"
    with st.expander(title, expanded=True):
        st.write(f["summary"])
        st.code("\n".join(f["evidence"]), language="text")
        if not hyp:
            return
        rc_id = hyp.get("root_cause_id")
        st.markdown(f"**Root cause:** {hyp['root_cause']}" + (f"  \n`{rc_id}`" if rc_id else ""))
        st.progress(
            min(1.0, float(hyp["confidence"])),
            text=f"confidence {int(hyp['confidence'] * 100)}% · {hyp['source']}",
        )
        st.markdown(f"_{hyp['rationale']}_")
        if hyp["recommended_actions"]:
            st.markdown("**Recommended actions**")
            for a in hyp["recommended_actions"]:
                st.markdown(f"- {a}")
        if hyp["similar_incidents"]:
            st.markdown("**Similar historical incidents (RAG)**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "run": s["run_id"],
                            "type": s["failure_type"],
                            "root cause": s["root_cause_id"],
                            "similarity": round(1 - s["distance"], 3),
                        }
                        for s in hyp["similar_incidents"]
                    ]
                ),
                width="stretch",
                hide_index=True,
                key=f"{key}-sim-{f['failure_id']}",
            )


def render_timeline(report: dict, key: str = "detail") -> None:
    """Where in the run each failure happened (one row per failure)."""
    rows = report["failures"]
    if not rows:
        return
    fig = go.Figure()
    for f in rows:
        start, end = f["first_ts_ms"] / 1000, f["last_ts_ms"] / 1000
        fig.add_trace(
            go.Scatter(
                x=[start, max(end, start + 0.05)],
                y=[f["failure_type"], f["failure_type"]],
                mode="lines+markers",
                line={"color": _color(f["failure_type"]), "width": 6},
                marker={
                    "size": 9,
                    "color": _color(f["failure_type"]),
                    "line": {"width": 2, "color": "#0b0f14"},
                },
                name=f["failure_type"],
                hovertemplate=f"{f['failure_type']}<br>%{{x:.2f}} s<br>{f['summary'][:80]}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Failure windows within the run",
        height=140 + 36 * len(rows),
        margin={"l": 10, "r": 20, "t": 44, "b": 36},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "seconds since power-on", "gridcolor": GRID, "zeroline": False},
        yaxis={"showgrid": False},
        showlegend=len(rows) > 1,
        legend={"orientation": "h", "y": -0.35},
    )
    st.plotly_chart(_dark(fig), width="stretch", key=f"{key}-timeline-{report['run_id']}")


def render_run(report: dict, key: str = "detail") -> None:
    """Render one report. ``key`` namespaces element ids so the same run can be shown
    in two tabs at once (Run detail + Live analysis) without id collisions."""
    status = "🔴 ANOMALOUS" if report["anomalous"] else "🟢 PASS"
    st.subheader(f"Run {report['run_id']} — {status}")
    meta = report["meta"]
    st.caption(
        f"{meta.get('soc')} {meta.get('stepping')} · {meta.get('board')} · "
        f"fw {meta.get('fw_version')} · {meta.get('test_suite')}/{meta.get('test_case')} · "
        f"{meta.get('voltage_mv')} mV / {meta.get('temp_c')}°C · {report['n_events']:,} events"
    )
    if not report["failures"]:
        st.success("No failures detected in this run.")
        return
    render_timeline(report, key)
    hyp_by_fid = {h["failure_id"]: h for h in report["hypotheses"]}
    for f in report["failures"]:
        render_failure(f, hyp_by_fid.get(f["failure_id"]), key)
    with st.expander("Markdown report"):
        try:
            from svla_agent.models import RunReport

            st.markdown(render_markdown(RunReport.model_validate(report)))
        except Exception as exc:  # pragma: no cover - UI nicety
            st.caption(f"Could not render markdown: {exc}")


def render_live(pipe: AnalysisPipeline, api_key: str | None) -> None:
    st.markdown(
        "Analyse a run end to end **right now**: parse → classify → cluster → retrieve "
        "similar incidents → hypothesise. "
        + (
            "Using **Claude** with your key."
            if api_key
            else "No key set: the deterministic rule-based agent runs (add a key in the sidebar for Claude)."
        )
    )
    runs = discover_runs(pipe.settings.logs_dir)

    def _analyse(target) -> dict | None:
        try:
            if pipe.store.count() == 0:
                with st.spinner("Building the historical-failure index from the logs directory…"):
                    pipe.index_corpus()
            with st.spinner("Analysing…"):
                return (
                    pipeline_with_key(pipe, api_key).process_run(target, write=False).model_dump()
                )
        except Exception as exc:  # surface instead of crashing the whole page
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
            return None

    source = st.radio(
        "Input", ["Pick a run from the logs directory", "Upload log files"], horizontal=True
    )
    report: dict | None = None
    if source.startswith("Pick"):
        if not runs:
            st.warning(f"No runs found under `{pipe.settings.logs_dir}`.")
            return
        rid = st.selectbox("Run", [r.run_id for r in runs])
        if st.button("Analyse run", type="primary"):
            report = _analyse(rid)
    else:
        files = st.file_uploader(
            "run_<id>.<source>.log or .log.gz",
            accept_multiple_files=True,
            type=["log", "gz"],
        )
        if files and st.button("Analyse upload", type="primary"):
            tmp = Path(tempfile.mkdtemp(prefix="svla_ui_"))
            for up in files:
                (tmp / Path(up.name).name).write_bytes(up.getvalue())
            found = discover_runs(tmp)
            if not found:
                st.error("No `run_<id>.<source>.log[.gz]` files recognised.")
            else:
                report = _analyse(found[0])
    if report:
        render_run(report, key="live")


def _metric_row(items: list[tuple[str, Any]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items, strict=True):
        col.metric(label, value)


def render_metrics(metrics: dict | None) -> None:
    if not metrics:
        st.info("No metrics file found. Run `svla eval --out reports/metrics.json`.")
        return
    det, ft, cl = metrics["detection"], metrics["failure_typing"]["micro"], metrics["clustering"]
    st.markdown(
        f"Evaluated on **{metrics['n_runs']} runs / {metrics.get('n_failures', '?')} failures**."
    )
    st.markdown("**Anomaly detection (run level)**")
    _metric_row(
        [
            ("precision", det["precision"]),
            ("recall", det["recall"]),
            ("F1", det["f1"]),
            ("accuracy", det["accuracy"]),
        ]
    )
    st.markdown("**Failure typing (micro)**")
    _metric_row([("precision", ft["precision"]), ("recall", ft["recall"]), ("F1", ft["f1"])])
    st.markdown("**Clustering vs. ground-truth type**")
    _metric_row(
        [
            ("clusters", cl.get("n_clusters")),
            ("purity", cl.get("purity")),
            ("homogeneity", cl.get("homogeneity_vs_failure_type")),
            ("silhouette", cl.get("silhouette")),
            ("Davies-Bouldin", cl.get("davies_bouldin")),
        ]
    )
    ho = metrics.get("holdout", {})
    if ho.get("test_runs"):
        st.markdown(
            f"**Hold-out evaluation** — index built from {ho['train_runs']} runs, "
            f"scored on {ho['test_runs']} unseen runs ({ho['test_failures']} failures)."
        )
        rt, rc = metrics.get("retrieval", {}), metrics.get("root_cause", {})
        _metric_row(
            [
                (f"recall@{rt.get('k', 5)} (type)", rt.get("recall_at_k")),
                ("MRR", rt.get("mrr")),
                (f"recall@{rt.get('k', 5)} (root cause)", rt.get("root_cause_recall_at_k")),
                ("root-cause top-1", rc.get("accuracy")),
                (
                    "conf. when right / wrong",
                    f"{rc.get('mean_confidence_correct')} / {rc.get('mean_confidence_wrong')}",
                ),
            ]
        )
        per = rc.get("per_type", {})
        if per:
            labels = list(per)
            st.plotly_chart(
                _bar(
                    labels,
                    [round(per[t]["accuracy"] * 100, 1) for t in labels],
                    [_color(t) for t in labels],
                    "Root-cause accuracy by failure type (%)",
                    "%",
                ),
                width="stretch",
            )
    with st.expander("Raw metrics JSON"):
        st.json(metrics)


def render_about(pipe: AnalysisPipeline) -> None:
    st.markdown(
        f"""
**SVLA-Agent v{__version__}** — an agentic debug assistant for post-silicon validation logs.

```
raw logs ─▶ ingest ─▶ parse ─▶ classify ─▶ cluster ─▶ RAG ─▶ hypothesize ─▶ report
          (stream)  (Drain3)  (rules+   (HDBSCAN)  (vector  (Claude /    (MD/JSON)
                               redact)              store)   rule-based)
```

| Component | Active backend |
|---|---|
| Embeddings | `{pipe.embedder.backend}` |
| Vector store | `{pipe.store.name}` ({pipe.store.count()} historical failures indexed) |
| Hypothesis agent | `{pipe.agent.backend}` |
| PII/secret redaction | `{"on" if pipe.settings.redact else "off"}` |

**Privacy:** keys you paste live only in this browser session and are used for your own
requests; nothing is stored server-side. Log lines are redacted (e-mails, IPs, MACs,
serials, credentials) before indexing, prompting or reporting.

Source: [GitHub](https://github.com/SapnilPatel/Silicon-Validation-Log-Analysis-Agent) ·
REST API: `svla serve` · CLI: `svla generate | process | index | eval | serve`
"""
    )
    st.markdown("**Root-cause catalogue used by the agent**")
    rows = [
        {
            "failure type": ftype,
            "root cause id": rc.rc_id,
            "description": rc.description,
            "tell-tale evidence": ", ".join(rc.markers),
        }
        for ftype, causes in ROOT_CAUSES.items()
        for rc in causes
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def main() -> None:
    settings = get_settings()
    logs_default, labels_default, reports_default = _default_dirs(settings)

    inject_css()
    st.title("🔬 SVLA-Agent")

    with st.sidebar:
        st.header("Data")
        reports_dir = st.text_input("Reports directory", str(reports_default))
        logs_dir = st.text_input("Logs directory", str(logs_default))
        labels_dir = st.text_input("Labels directory", str(labels_default))
        if st.button("Reload"):
            load_reports.clear()
            load_metrics.clear()
        st.header("Claude (optional)")
        api_key = (
            st.text_input(
                "Anthropic API key",
                type="password",
                help="Session-only. Enables Claude for live analysis; otherwise the rule-based agent runs.",
            )
            or None
        )
        st.caption(
            "Generate data and reports locally:\n\n```\nsvla generate --runs 100\nsvla process\nsvla eval --out reports/metrics.json\n```"
        )

    reports = load_reports(reports_dir)
    metrics = load_metrics(reports_dir)
    pipe = get_pipeline(logs_dir, labels_dir, reports_dir)
    render_hero(metrics, reports)

    tabs = st.tabs(
        ["📊 Overview", "🔎 Run detail", "⚡ Live analysis", "📏 Evaluation", "📘 About"]
    )
    with tabs[0]:
        st.markdown(
            "<p class='svla-note'>What the pipeline found across every loaded run: how many were anomalous, "
            "which failure types and root causes dominate, and the recurring families it clustered.</p>",
            unsafe_allow_html=True,
        )
        if reports:
            render_overview(reports)
        else:
            st.warning(
                f"No reports found in `{reports_dir}`. Use the Live analysis tab or run `svla process`."
            )
    with tabs[1]:
        st.markdown(
            "<p class='svla-note'>One run end to end: device context, where each failure happened in time, the "
            "redacted evidence the agent saw, its root-cause hypothesis with calibrated confidence, the next "
            "experiments to run, and the historical incidents it retrieved.</p>",
            unsafe_allow_html=True,
        )
        if reports:
            ids = [r["run_id"] for r in reports]
            default = next((i for i, r in enumerate(reports) if r["anomalous"]), 0)
            choice = st.selectbox("Select run", ids, index=default)
            render_run(next(r for r in reports if r["run_id"] == choice))
        else:
            st.info("No reports loaded.")
    with tabs[2]:
        render_live(pipe, api_key)
    with tabs[3]:
        st.markdown(
            "<p class='svla-note'>How well it works, measured against ground truth. Retrieval and root-cause "
            "metrics use a hold-out split so a failure can never retrieve itself.</p>",
            unsafe_allow_html=True,
        )
        render_metrics(metrics)
    with tabs[4]:
        render_about(pipe)


if __name__ == "__main__":
    main()

"""SVLA-Agent dashboard.

Browse analysed runs, failures, root-cause hypotheses and failure families.
Run with:  ``streamlit run app/streamlit_app.py``  (after ``svla process``).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

# Make the src-layout package importable when run via `streamlit run`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from svla_agent.config import get_settings  # noqa: E402

st.set_page_config(page_title="SVLA-Agent", page_icon="🔬", layout="wide")

SEVERITY_COLOR = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "🟦"}


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


def render_overview(reports: list[dict]) -> None:
    anomalous = [r for r in reports if r["anomalous"]]
    failures = [f for r in reports for f in r["failures"]]
    clusters = {c["cluster_id"]: c for r in reports for c in r["clusters"]}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs", len(reports))
    c2.metric("Anomalous", len(anomalous))
    c3.metric("Failures", len(failures))
    c4.metric("Clusters", len(clusters))

    if failures:
        st.subheader("Failure type distribution")
        counts = Counter(f["failure_type"] for f in failures)
        st.bar_chart(dict(sorted(counts.items())))

    if clusters:
        st.subheader("Failure families")
        st.dataframe(
            [
                {
                    "cluster": c["cluster_id"],
                    "label": c["label"],
                    "size": c["size"],
                    "types": ", ".join(c["failure_types"]),
                }
                for c in sorted(clusters.values(), key=lambda c: -c["size"])
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_run(report: dict) -> None:
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

    hyp_by_fid = {h["failure_id"]: h for h in report["hypotheses"]}
    for f in report["failures"]:
        icon = SEVERITY_COLOR.get(f["severity"], "⬜")
        with st.expander(f"{icon} {f['failure_type']} — {f['severity'].upper()} ({f['subsys']})", expanded=True):
            st.write(f["summary"])
            st.code("\n".join(f["evidence"]), language="text")

            hyp = hyp_by_fid.get(f["failure_id"])
            if hyp:
                st.markdown(f"**Root cause:** {hyp['root_cause']}")
                st.progress(
                    min(1.0, float(hyp["confidence"])),
                    text=f"confidence {int(hyp['confidence'] * 100)}% ({hyp['source']})",
                )
                st.markdown(f"_{hyp['rationale']}_")
                if hyp["recommended_actions"]:
                    st.markdown("**Recommended actions:**")
                    for a in hyp["recommended_actions"]:
                        st.markdown(f"- {a}")
                if hyp["similar_incidents"]:
                    st.markdown("**Similar historical incidents:**")
                    st.dataframe(
                        [
                            {
                                "run": s["run_id"],
                                "type": s["failure_type"],
                                "root_cause": s["root_cause_id"],
                                "similarity": s["similarity"],
                            }
                            for s in hyp["similar_incidents"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )


def main() -> None:
    settings = get_settings()
    st.title("🔬 SVLA-Agent — Silicon Validation Log Analysis")

    with st.sidebar:
        st.header("Data")
        reports_dir = st.text_input("Reports directory", str(settings.reports_dir))
        if st.button("Reload reports"):
            load_reports.clear()
        st.caption(
            "Generate data and reports first:\n\n"
            "```\nsvla generate --runs 100\nsvla process\n```"
        )

    reports = load_reports(reports_dir)
    if not reports:
        st.warning(
            f"No reports found in `{reports_dir}`. Run `svla process` to generate them."
        )
        return

    tab_overview, tab_run = st.tabs(["📊 Overview", "🔎 Run detail"])
    with tab_overview:
        render_overview(reports)
    with tab_run:
        ids = [r["run_id"] for r in reports]
        # Default to the first anomalous run if any.
        default = next((i for i, r in enumerate(reports) if r["anomalous"]), 0)
        choice = st.selectbox("Select run", ids, index=default)
        render_run(next(r for r in reports if r["run_id"] == choice))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Bundle the data the static showcase site needs into site/data/bundle.json.

Stdlib only (runs in the Pages workflow without installing the package):
reports from examples/reports, the three evaluation metric files from
examples/metrics, and the root-cause catalogue read straight from
hypothesize/knowledge.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data" / "bundle.json"


def load_knowledge() -> dict:
    path = ROOT / "src" / "svla_agent" / "hypothesize" / "knowledge.py"
    spec = importlib.util.spec_from_file_location("svla_knowledge", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return {
        ftype: {
            "prior": {
                "root_cause": mod.KNOWLEDGE[ftype].root_cause,
                "rationale": mod.KNOWLEDGE[ftype].rationale,
                "actions": list(mod.KNOWLEDGE[ftype].actions),
            },
            "root_causes": [
                {
                    "rc_id": rc.rc_id,
                    "description": rc.description,
                    "markers": list(rc.markers),
                    "actions": list(rc.actions),
                }
                for rc in causes
            ],
        }
        for ftype, causes in mod.ROOT_CAUSES.items()
    }


def main() -> int:
    reports = [
        json.loads(p.read_text())
        for p in sorted((ROOT / "examples" / "reports").glob("run_*.json"))
    ]
    metrics = {
        p.stem: json.loads(p.read_text())
        for p in sorted((ROOT / "examples" / "metrics").glob("*.json"))
    }
    version = "0.0.0"
    for line in (ROOT / "src" / "svla_agent" / "__init__.py").read_text().splitlines():
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": version,
        "repo": "https://github.com/SapnilPatel/Silicon-Validation-Log-Analysis-Agent",
        "live_app": "https://svla-agent.streamlit.app/",
        "reports": reports,
        "metrics": metrics,
        "knowledge": load_knowledge(),
        "throughput": {
            "runs": 100,
            "files": 700,
            "size_mb": 97,
            "events": 1453172,
            "seconds": 22.8,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, separators=(",", ":")))
    print(
        f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB, {len(reports)} reports, {len(metrics)} metric sets)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

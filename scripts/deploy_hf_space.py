#!/usr/bin/env python
"""Deploy the SVLA-Agent dashboard to a Hugging Face Space (Streamlit SDK).

Stages exactly what the Space needs - the app, the package source (the app adds
``src/`` to ``sys.path``), the bundled sample dataset + reports, requirements and
the Space README - then uploads it with ``huggingface_hub``.

Usage:
    python scripts/deploy_hf_space.py [--repo USER/NAME] [--private] [--dry-run]

Requires a write token (``hf auth login`` or ``HF_TOKEN``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE: list[tuple[Path, str]] = [
    (ROOT / "deploy/hf/README.md", "README.md"),
    (ROOT / "requirements.txt", "requirements.txt"),
    (ROOT / "app/streamlit_app.py", "app/streamlit_app.py"),
    (ROOT / "src/svla_agent", "src/svla_agent"),
    (ROOT / "examples", "examples"),
    (ROOT / "LICENSE", "LICENSE"),
]
SPACE_VARIABLES = {"SVLA_VECTOR_BACKEND": "memory", "SVLA_LOG_LEVEL": "INFO"}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def stage(dest: Path) -> None:
    for src, rel in INCLUDE:
        target = dest / rel
        if src.is_dir():
            shutil.copytree(src, target, ignore=IGNORE)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Sapnil/svla-agent", help="Space id (user/name)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Stage only; print the file list")
    ap.add_argument("--wait", type=int, default=900, help="Seconds to wait for the build")
    args = ap.parse_args()

    staging = Path(tempfile.mkdtemp(prefix="svla_space_"))
    stage(staging)
    files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    print(f"Staged {len(files)} files in {staging}")
    if args.dry_run:
        print("\n".join(files))
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"Authenticated as {who.get('name')}")
    url = api.create_repo(
        args.repo, repo_type="space", space_sdk="streamlit", private=args.private, exist_ok=True
    )
    print(f"Space: {url}")
    for key, value in SPACE_VARIABLES.items():
        api.add_space_variable(args.repo, key, value)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo,
        repo_type="space",
        commit_message="Deploy SVLA-Agent dashboard",
        delete_patterns=["**"],  # make the Space mirror the staging dir exactly
    )
    print("Upload complete; waiting for the Space to build...")
    deadline = time.time() + args.wait
    last = None
    while time.time() < deadline:
        stage_name = api.get_space_runtime(args.repo).stage
        if stage_name != last:
            print(f"  stage: {stage_name}")
            last = stage_name
        if stage_name == "RUNNING":
            print(f"Live: https://huggingface.co/spaces/{args.repo}")
            return 0
        if stage_name in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            print("Space failed to start; check the logs on the Space page.", file=sys.stderr)
            return 1
        time.sleep(10)
    print("Timed out waiting for the Space; check its status page.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

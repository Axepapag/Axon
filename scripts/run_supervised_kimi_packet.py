"""Compatibility wrapper for the canonical Kimi round-table supervisor.

The historical supervisor embedded stale July continuity and hard-coded another
Windows user. All current packet execution is delegated to
scripts/run_kimi_roundtable.py so Kimi uses the live Axon authority, K3 by
default, project-scoped agents, and State/kimi_orchestrator job records.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--model", default="kimi-code/k3")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    packet = args.packet.resolve(strict=True)
    try:
        packet.relative_to(repo)
    except ValueError as exc:
        parser.error("packet must be inside the Axon repository")
        raise AssertionError from exc

    runner = Path(__file__).with_name("run_kimi_roundtable.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--mode",
            "implement",
            "--mission-file",
            str(packet),
            "--job-id",
            args.job_id,
            "--model",
            args.model,
        ],
        cwd=repo,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

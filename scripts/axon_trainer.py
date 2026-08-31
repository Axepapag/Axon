#!/usr/bin/env python3
"""Thin local client for the transport-neutral Axon Trainer organ."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.trainer import TrainerOrgan, TrainerOrganCommand  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("status",),
        help="read-only command currently wired; future controls use this same organ surface",
    )
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument(
        "--detail",
        choices=("summary", "full"),
        default="summary",
        help="status output detail; summary is intended for ordinary operator use",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    organ = TrainerOrgan(state_root=args.state_root)
    result = organ.dispatch(
        TrainerOrganCommand(
            kind=args.command,
            arguments={"detail": args.detail},
            requested_by="local-cli",
        )
    )
    print(json.dumps(result.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.status.value == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

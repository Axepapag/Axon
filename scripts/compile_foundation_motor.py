#!/usr/bin/env python3
"""Compile and publish the split-disjoint Foundations Stage-0 curriculum."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.field import CanonicalStateBranch, LogicalRegion  # noqa: E402
from training.first_form_curriculum import publish_first_form_curriculum  # noqa: E402
from training.foundation_motor_curriculum import compile_foundation_motor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--train", type=int, default=72)
    parser.add_argument("--heldout", type=int, default=24)
    parser.add_argument("--regression", type=int, default=24)
    args = parser.parse_args()
    identity = (
        CanonicalStateBranch.active_runtime(state_root=args.state_root)
        .load_head()
        .region(LogicalRegion.IDENTITY)
        .text
    )
    curriculum = compile_foundation_motor(
        identity_text=identity,
        requested_counts=(("F0", (args.train, args.heldout, args.regression)),),
    )
    path = publish_first_form_curriculum(curriculum, state_root=args.state_root)
    print(
        json.dumps(
            {
                "schema": "axon-foundation-motor-compilation-result-v1",
                "manifest_id": curriculum.manifest_id,
                "case_count": len(curriculum.cases),
                "actual_family_split_counts": curriculum.actual_family_split_counts,
                "manifest_path": str(path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

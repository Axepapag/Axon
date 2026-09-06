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
from training.foundation_motor_curriculum import (  # noqa: E402
    DEFAULT_FOUNDATION_MOTOR_SPLIT_COUNTS,
    DEFAULT_FOUNDATION_MOTOR_V2_SPLIT_COUNTS,
    compile_foundation_motor,
    compile_foundation_motor_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--program", choices=("v1", "v2"), default="v1")
    parser.add_argument("--train", type=int, default=None)
    parser.add_argument("--heldout", type=int, default=None)
    parser.add_argument("--regression", type=int, default=None)
    args = parser.parse_args()
    defaults = (
        DEFAULT_FOUNDATION_MOTOR_V2_SPLIT_COUNTS[0][1]
        if args.program == "v2"
        else DEFAULT_FOUNDATION_MOTOR_SPLIT_COUNTS[0][1]
    )
    counts = (
        defaults[0] if args.train is None else args.train,
        defaults[1] if args.heldout is None else args.heldout,
        defaults[2] if args.regression is None else args.regression,
    )
    identity = (
        CanonicalStateBranch.active_runtime(state_root=args.state_root)
        .load_head()
        .region(LogicalRegion.IDENTITY)
        .text
    )
    compiler = compile_foundation_motor_v2 if args.program == "v2" else compile_foundation_motor
    curriculum = compiler(
        identity_text=identity,
        requested_counts=(("F0", counts),),
    )
    path = publish_first_form_curriculum(curriculum, state_root=args.state_root)
    print(
        json.dumps(
            {
                "schema": "axon-foundation-motor-compilation-result-v1",
                "program": args.program,
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

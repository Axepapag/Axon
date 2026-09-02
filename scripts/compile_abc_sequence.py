"""Compile and publish the ABC sequence curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.abc_sequence_curriculum import compile_abc_sequence
from training.first_form_curriculum import publish_first_form_curriculum

ROOT = Path(__file__).resolve().parent.parent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--heldout", type=int, default=8)
    parser.add_argument("--regression", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    from runtime.field import CanonicalStateBranch, LogicalRegion
    branch = CanonicalStateBranch.active_runtime(state_root=args.state_root)
    snapshot = branch.load_head()
    identity = snapshot.region(LogicalRegion.IDENTITY).text

    counts = (("ABC", (args.train, args.heldout, args.regression)),)
    curriculum = compile_abc_sequence(identity_text=identity, requested_counts=counts)
    path = publish_first_form_curriculum(curriculum, state_root=args.state_root)
    print(
        json.dumps(
            {
                "schema": "axon-abc-sequence-compilation-result-v1",
                "manifest_id": curriculum.manifest_id,
                "case_count": len(curriculum.cases),
                "family_split_counts": {
                    family: dict(
                        zip(("train", "heldout", "regression"), counts, strict=True)
                    )
                    for family, counts in curriculum.actual_family_split_counts
                },
                "eligibility_counts": dict(curriculum.eligibility_counts),
                "manifest_path": str(path),
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
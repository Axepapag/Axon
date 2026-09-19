"""Compile and publish the L0-L4 Language Foundations curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.field import CanonicalStateBranch, LogicalRegion
from training import (
    LanguageFoundationsCompiler,
    publish_language_foundations_curriculum,
    verify_language_no_target_leakage,
    verify_language_transfer_disjointness,
    verify_language_visibility_consistency,
)

ROOT = Path(__file__).resolve().parent.parent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    branch = CanonicalStateBranch.active_runtime(state_root=args.state_root)
    snapshot = branch.load_head()
    identity = snapshot.region(LogicalRegion.IDENTITY).text
    curriculum = LanguageFoundationsCompiler().compile(identity_text=identity)
    verify_language_visibility_consistency(curriculum.cases)
    verify_language_no_target_leakage(curriculum.cases)
    verify_language_transfer_disjointness(curriculum.cases)
    path = publish_language_foundations_curriculum(curriculum, state_root=args.state_root)
    print(
        json.dumps(
            {
                "schema": "axon-language-foundations-compilation-result-v1",
                "manifest_id": curriculum.manifest_id,
                "living_curriculum_id": curriculum.living_curriculum.curriculum_id,
                "case_count": len(curriculum.cases),
                "family_split_counts": {
                    family: dict(zip(("train", "heldout", "regression"), counts, strict=True))
                    for family, counts in curriculum.actual_family_split_counts
                },
                "excluded_counts": dict(curriculum.excluded_counts),
                "validators": [
                    "visibility_consistency",
                    "no_target_leakage",
                    "transfer_disjointness",
                ],
                "manifest_path": str(path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

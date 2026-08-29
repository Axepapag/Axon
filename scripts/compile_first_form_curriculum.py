"""Compile and publish FFCS-A/B/C from exact existing Dormant evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.dormant import DormantExperienceStore
from runtime.field import CanonicalStateBranch, LogicalRegion
from training import FirstFormCurriculumCompiler, publish_first_form_curriculum

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
    curriculum = FirstFormCurriculumCompiler(
        DormantExperienceStore(args.state_root)
    ).compile_abc(identity_text=identity)
    path = publish_first_form_curriculum(curriculum, state_root=args.state_root)
    print(
        json.dumps(
            {
                "schema": "axon-first-form-curriculum-compilation-result-v1",
                "manifest_id": curriculum.manifest_id,
                "living_curriculum_id": curriculum.living_curriculum.curriculum_id,
                "case_count": len(curriculum.cases),
                "family_split_counts": {
                    family: dict(zip(("train", "heldout", "regression"), counts, strict=True))
                    for family, counts in curriculum.actual_family_split_counts
                },
                "excluded_counts": dict(curriculum.excluded_counts),
                "manifest_path": str(path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

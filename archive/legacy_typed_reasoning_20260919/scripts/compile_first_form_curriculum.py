"""Compile and publish one governed FFCS family group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.dormant import DormantExperienceStore
from runtime.field import CanonicalStateBranch, LogicalRegion
from training import (
    FirstFormCurriculumCompiler,
    compile_sequential_first_form,
    publish_first_form_curriculum,
    publish_sequential_first_form,
)

ROOT = Path(__file__).resolve().parent.parent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--families", choices=("abc", "df", "e"), default="abc")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    branch = CanonicalStateBranch.active_runtime(state_root=args.state_root)
    snapshot = branch.load_head()
    identity = snapshot.region(LogicalRegion.IDENTITY).text
    compiler = FirstFormCurriculumCompiler(DormantExperienceStore(args.state_root))
    if args.families == "abc":
        curriculum = compiler.compile_abc(identity_text=identity)
        path = publish_first_form_curriculum(curriculum, state_root=args.state_root)
    elif args.families == "df":
        curriculum = compiler.compile_df(identity_text=identity)
        path = publish_first_form_curriculum(curriculum, state_root=args.state_root)
    else:
        curriculum = compile_sequential_first_form(
            DormantExperienceStore(args.state_root),
            identity_text=identity,
        )
        path = publish_sequential_first_form(curriculum, state_root=args.state_root)
    print(
        json.dumps(
            {
                "schema": "axon-first-form-curriculum-compilation-result-v1",
                "manifest_id": curriculum.manifest_id,
                "living_curriculum_id": getattr(
                    getattr(curriculum, "living_curriculum", None),
                    "curriculum_id",
                    None,
                ),
                "case_count": len(curriculum.cases),
                "family_split_counts": (
                    {
                        family: dict(
                            zip(("train", "heldout", "regression"), counts, strict=True)
                        )
                        for family, counts in curriculum.actual_family_split_counts
                    }
                    if hasattr(curriculum, "actual_family_split_counts")
                    else {
                        "E": dict(
                            zip(
                                ("train", "heldout", "regression"),
                                curriculum.requested_split_counts,
                                strict=True,
                            )
                        )
                    }
                ),
                "excluded_counts": dict(getattr(curriculum, "excluded_counts", ())),
                "manifest_path": str(path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

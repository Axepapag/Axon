"""Migrate Axon's active canonical Shared Field to the v2 Cortex region schema.

This operator never rewrites historical snapshots.  It advances the active
branch by one schema-migration successor when HEAD is still shared-field-v1;
an already-v2 branch is a no-op.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime.field import CanonicalStateBranch, LogicalRegion, SCHEMA_VERSION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advance the active Shared Field to shared-field-v2 / cortex"
    )
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    parser.add_argument("--branch-id", default="active")
    return parser


def main() -> int:
    args = _parser().parse_args()
    branch = CanonicalStateBranch.active_runtime(
        branch_id=args.branch_id,
        state_root=args.state_root,
    )
    before_record = branch.load_head_record()
    before = branch.load_head()
    after = branch.migrate_to_current_schema()
    after_record = branch.load_head_record()

    if after.schema_version != SCHEMA_VERSION:
        raise RuntimeError("migration did not produce the current Shared Field schema")
    if before.schema_version != after.schema_version:
        if after.parent_field_id != before.field_id:
            raise RuntimeError("schema migration successor is not parented to prior HEAD")
        if tuple(region.spans for region in before.regions) != tuple(
            region.spans for region in after.regions
        ):
            raise RuntimeError("schema migration changed canonical span contents")
        if before.source_manifest_ids != after.source_manifest_ids:
            raise RuntimeError("schema migration changed source manifests")

    payload = {
        "operation": "shared_field_cortex_schema_migration",
        "branch_id": args.branch_id,
        "changed": before.field_id != after.field_id,
        "before": {
            "schema": before.schema_version,
            "generation": before_record.generation,
            "field_id": before.field_id,
            "tick_id": before.tick_id,
        },
        "after": {
            "schema": after.schema_version,
            "generation": after_record.generation,
            "field_id": after.field_id,
            "tick_id": after.tick_id,
            "parent_field_id": after.parent_field_id,
            "cortex_chars": len(after.region(LogicalRegion.CORTEX).text),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

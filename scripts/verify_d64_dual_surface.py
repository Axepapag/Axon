"""Read-only verification of Axon's exact+semantic D64 dual surface."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from runtime.field import (
    CanonicalStateBranch,
    D64FieldCompiler,
    D64SemanticSurfaceCompiler,
)
from runtime.heart import FrozenTickImage, HeartbeatClock


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify(state_root: Path, branch_id: str) -> dict[str, object]:
    branches_root = state_root / "active" / "branches"
    branch_root = branches_root / branch_id
    head_path = branch_root / "HEAD.json"
    before_head_sha256 = _sha256(head_path)

    branch = CanonicalStateBranch(
        branch_root,
        branch_id=branch_id,
        authority_root=branches_root,
    )
    snapshot = branch.load_head()
    exact = D64FieldCompiler().compile(snapshot)
    exact.verify_roundtrip(snapshot)
    semantic = D64SemanticSurfaceCompiler().compile(snapshot, exact)
    semantic.verify_grounding(snapshot, exact)

    # Local verification identity only.  This does not touch the durable Heart
    # identity store or claim a production heartbeat/tick.
    identity = HeartbeatClock().open_tick(snapshot)
    image = FrozenTickImage.from_compiled(
        identity,
        {64: exact},
        semantic_surfaces={64: semantic},
    )
    binding = image.require_rail(64)
    if binding.semantic_surface_id != semantic.surface_id:
        raise RuntimeError("frozen tick semantic binding does not match compiled surface")

    after_head_sha256 = _sha256(head_path)
    if before_head_sha256 != after_head_sha256:
        raise RuntimeError("read-only D64 dual-surface verification changed canonical branch HEAD")

    kind_counts = Counter(slot.slot_kind for slot in semantic.slots)
    source_kind_counts = Counter(
        slot.source_kind for slot in semantic.slots if slot.source_kind
    )
    return {
        "schema": "axon-d64-dual-surface-verification-v1",
        "read_only": True,
        "branch_id": branch_id,
        "field_id": snapshot.field_id,
        "field_tick_id": snapshot.tick_id,
        "head_sha256_before": before_head_sha256,
        "head_sha256_after": after_head_sha256,
        "head_unchanged": before_head_sha256 == after_head_sha256,
        "exact_rail_id": exact.rail_id,
        "exact_rows": exact.row_count,
        "exact_active_characters": exact.coverage.compiled_active_characters,
        "exact_roundtrip_complete": exact.coverage.complete,
        "semantic_surface_id": semantic.surface_id,
        "semantic_generation": semantic.feature_generation,
        "semantic_slot_count": semantic.slot_count,
        "semantic_slot_kinds": dict(sorted(kind_counts.items())),
        "semantic_source_kinds": dict(sorted(source_kind_counts.items())),
        "slots_with_container_refs": sum(bool(slot.container_refs) for slot in semantic.slots),
        "slots_with_edge_refs": sum(bool(slot.edge_refs) for slot in semantic.slots),
        "frozen_image_id": image.image_id,
        "frozen_semantic_surface_id": binding.semantic_surface_id,
        "frozen_semantic_generation": binding.semantic_generation,
        "frozen_semantic_slot_count": binding.semantic_slot_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    parser.add_argument("--branch-id", default="active")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = verify(args.state_root.resolve(), args.branch_id)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

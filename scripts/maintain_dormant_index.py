"""Maintain Axon's derived dormant evidence index without rewriting memory authority."""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime.dormant import (
    DormantEvidenceGenerationStore,
    DormantIncrementalFallbackRequired,
    DormantIncrementalMaintainer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally maintain the disposable dormant evidence index. "
            "Exact State/dormant JSONL remains authoritative."
        )
    )
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan/verify and report the incremental plan without mutating the derived index",
    )
    parser.add_argument(
        "--fallback-full",
        action="store_true",
        help="if layout changes cannot be maintained incrementally, build/promote a full isolated generation",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="finish a previously committed incremental generation whose pointer publication was interrupted",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    store = DormantEvidenceGenerationStore(args.state_root)
    maintainer = DormantIncrementalMaintainer(store)

    if args.recover:
        descriptor = maintainer.recover_pointer()
        print(f"recovered_generation={descriptor.generation_id}")
        print(f"index_id={descriptor.index_id}")
        print(f"binding_id={descriptor.binding_id}")
        print(f"mode={descriptor.mode}")
        return 0

    if args.dry_run:
        try:
            plan = maintainer.plan()
        except DormantIncrementalFallbackRequired as exc:
            print("incremental_safe=false")
            print(f"fallback_reason={exc}")
            return 2
        print("incremental_safe=true")
        print(f"plan_id={plan.plan_id}")
        print(f"predecessor_generation={plan.predecessor_generation.generation_id}")
        print(f"old_index_id={plan.predecessor_index_id}")
        print(f"new_index_id={plan.new_index_id}")
        print(f"noop={str(plan.is_noop).lower()}")
        print(f"container_appends={plan.container_appends}")
        print(f"container_updates={plan.container_updates}")
        print(f"edge_appends={plan.edge_appends}")
        print(f"edge_updates={plan.edge_updates}")
        return 0

    result = maintainer.maintain(fallback_full=args.fallback_full)
    descriptor = result.descriptor
    print(f"generation={descriptor.generation_id}")
    print(f"mode={descriptor.mode}")
    print(f"index_id={descriptor.index_id}")
    print(f"binding_id={descriptor.binding_id}")
    print(f"plan_id={result.plan_id}")
    print(f"container_appends={result.container_appends}")
    print(f"container_updates={result.container_updates}")
    print(f"edge_appends={result.edge_appends}")
    print(f"edge_updates={result.edge_updates}")
    print(f"metadata_only={str(result.metadata_only).lower()}")
    print(f"full_rebuild_fallback={str(result.full_rebuild_fallback).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

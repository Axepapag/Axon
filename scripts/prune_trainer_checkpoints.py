#!/usr/bin/env python3
"""Prune trainer candidate checkpoint payloads to the retention window.

Keeps the newest ``--keep`` checkpoint artifacts per candidate generation
(default: ``runtime.trainer.store.CHECKPOINT_RETENTION``), always retaining
the artifact referenced by ``latest_checkpoint.json``. Immutable checkpoint
records are never deleted, so pruned payloads remain detectable by hash.
Also removes orphaned ``checkpoint_staging`` temp files that were never
promoted to content-addressed artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime.trainer.lifecycle import CandidateCheckpointRecord
from runtime.trainer.store import CHECKPOINT_RETENTION, TrainerStateStore

ROOT = Path(__file__).resolve().parent.parent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--keep", type=int, default=CHECKPOINT_RETENTION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be pruned without deleting anything",
    )
    return parser.parse_args()


def _records(generation_dir: Path) -> list[CandidateCheckpointRecord]:
    records_dir = generation_dir / "checkpoint_records"
    if not records_dir.is_dir():
        return []
    records = [
        CandidateCheckpointRecord.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(records_dir.glob("*.json"))
    ]
    records.sort(key=lambda r: (r.step, r.micro_step, r.accumulation_index, r.checkpoint_id))
    return records


def _staging_orphans(generation_dir: Path) -> list[Path]:
    staging_dir = generation_dir / "checkpoint_staging"
    if not staging_dir.is_dir():
        return []
    return sorted(staging_dir.glob("*.pt.tmp"))


def _artifact(generation_dir: Path, record: CandidateCheckpointRecord) -> Path:
    return generation_dir / "checkpoints" / f"{record.artifact_sha256}.pt"


def main() -> int:
    args = _arguments()
    if args.keep < 1:
        raise ValueError("--keep must be a positive integer")
    store = TrainerStateStore(args.state_root.resolve() / "training" / "trainer")
    summary = []
    for module_dir in sorted(store.candidates_dir.glob("*")):
        if not module_dir.is_dir():
            continue
        for generation_dir in sorted(module_dir.glob("*")):
            if not generation_dir.is_dir():
                continue
            records = _records(generation_dir)
            orphans = _staging_orphans(generation_dir)
            orphan_bytes = sum(path.stat().st_size for path in orphans)
            if args.dry_run:
                retained = {record.checkpoint_id for record in records[-args.keep:]}
                latest_path = generation_dir / "latest_checkpoint.json"
                if latest_path.is_file():
                    latest = CandidateCheckpointRecord.from_mapping(
                        json.loads(latest_path.read_text(encoding="utf-8"))
                    )
                    retained.add(latest.checkpoint_id)
                prunable = [r for r in records if r.checkpoint_id not in retained]
                summary.append(
                    {
                        "module_id": module_dir.name,
                        "candidate_generation_id": generation_dir.name,
                        "records": len(records),
                        "prunable": len(prunable),
                        "prunable_bytes": sum(
                            _artifact(generation_dir, r).stat().st_size
                            for r in prunable
                            if _artifact(generation_dir, r).is_file()
                        ),
                        "staging_orphans": len(orphans),
                        "staging_orphan_bytes": orphan_bytes,
                        "dry_run": True,
                    }
                )
                continue
            pruned = store.prune_candidate_checkpoints(
                module_dir.name, generation_dir.name, keep=args.keep
            )
            for path in orphans:
                path.unlink()
            summary.append(
                {
                    "module_id": module_dir.name,
                    "candidate_generation_id": generation_dir.name,
                    "records": len(records),
                    "pruned": len(pruned),
                    "pruned_bytes": sum(record.artifact_bytes for record in pruned),
                    "staging_orphans_removed": len(orphans),
                    "staging_orphan_bytes": orphan_bytes,
                    "dry_run": False,
                }
            )
    json.dump(
        {"schema": "axon-trainer-checkpoint-prune-v1", "candidates": summary},
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

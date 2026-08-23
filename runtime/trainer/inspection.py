"""Read-only transparent inspection of Axon's Trainer state."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.field import canonical_sha256

TRAINER_INSPECTION_SCHEMA = "axon-trainer-inspection-v1"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unreadable Trainer state artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Trainer state artifact is not an object: {path}")
    return value


def _count_json(directory: Path, *, recursive: bool = False) -> int:
    if not directory.is_dir():
        return 0
    pattern = "**/*.json" if recursive else "*.json"
    return sum(1 for path in directory.glob(pattern) if path.is_file())


@dataclass(frozen=True, slots=True)
class TrainerInspectionSnapshot:
    state_root: str
    trainer_root: str
    lease: dict[str, Any] | None
    latest_lifecycle: dict[str, Any] | None
    latest_step: dict[str, Any] | None
    latest_telemetry: dict[str, Any] | None
    latest_checkpoints: tuple[dict[str, Any], ...]
    active_generation_pointers: tuple[dict[str, Any], ...]
    artifact_counts: tuple[tuple[str, int], ...]
    snapshot_id: str = field(init=False)

    def __post_init__(self) -> None:
        checkpoints = tuple(
            sorted(
                self.latest_checkpoints,
                key=lambda item: (str(item.get("module_id")), str(item.get("candidate_generation_id"))),
            )
        )
        pointers = tuple(sorted(self.active_generation_pointers, key=lambda item: str(item.get("module_id"))))
        counts = tuple(sorted((str(name), int(value)) for name, value in self.artifact_counts))
        object.__setattr__(self, "latest_checkpoints", checkpoints)
        object.__setattr__(self, "active_generation_pointers", pointers)
        object.__setattr__(self, "artifact_counts", counts)
        object.__setattr__(self, "snapshot_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRAINER_INSPECTION_SCHEMA,
            "state_root": self.state_root,
            "trainer_root": self.trainer_root,
            "writer_lease_present": self.lease is not None,
            "lease": self.lease,
            "latest_lifecycle": self.latest_lifecycle,
            "latest_step": self.latest_step,
            "latest_telemetry": self.latest_telemetry,
            "latest_checkpoints": list(self.latest_checkpoints),
            "active_generation_pointers": list(self.active_generation_pointers),
            "artifact_counts": {name: count for name, count in self.artifact_counts},
        }
        if include_id:
            value["snapshot_id"] = self.snapshot_id
        return value


def inspect_trainer_state(*, state_root: Path | str = Path(r"D:\Axon\State")) -> TrainerInspectionSnapshot:
    state = Path(state_root).resolve(strict=False)
    trainer = state / "training" / "trainer"
    lease = _read_json(trainer / "authority" / "lease.json")
    latest_lifecycle = _read_json(trainer / "latest_candidate_lifecycle.json")
    latest_step = _read_json(trainer / "latest_optimization_step.json")
    latest_telemetry = _read_json(trainer / "latest_telemetry.json")

    checkpoints: list[dict[str, Any]] = []
    candidates = trainer / "candidates"
    if candidates.is_dir():
        for latest in candidates.glob("*/*/latest_checkpoint.json"):
            value = _read_json(latest)
            if value is not None:
                checkpoints.append(value)

    active_pointers: list[dict[str, Any]] = []
    active_generations = trainer / "active_generations"
    if active_generations.is_dir():
        for pointer_path in active_generations.glob("*/pointer.json"):
            value = _read_json(pointer_path)
            if value is not None:
                active_pointers.append(value)

    counts = (
        ("inventories", _count_json(trainer / "inventories")),
        ("plans", _count_json(trainer / "plans")),
        ("authorizations", _count_json(trainer / "authorizations")),
        ("evaluations", _count_json(trainer / "evaluations")),
        ("gate_decisions", _count_json(trainer / "gate_decisions")),
        ("promotion_proposals", _count_json(trainer / "promotion_proposals")),
        ("generation_snapshots", _count_json(trainer / "generation_snapshots", recursive=True)),
        ("activation_receipts", _count_json(trainer / "activation_receipts", recursive=True)),
        ("rollback_receipts", _count_json(trainer / "rollback_receipts", recursive=True)),
        ("active_generation_pointers", len(active_pointers)),
    )
    return TrainerInspectionSnapshot(
        state_root=str(state),
        trainer_root=str(trainer),
        lease=lease,
        latest_lifecycle=latest_lifecycle,
        latest_step=latest_step,
        latest_telemetry=latest_telemetry,
        latest_checkpoints=tuple(checkpoints),
        active_generation_pointers=tuple(active_pointers),
        artifact_counts=counts,
    )


__all__ = ["TRAINER_INSPECTION_SCHEMA", "TrainerInspectionSnapshot", "inspect_trainer_state"]

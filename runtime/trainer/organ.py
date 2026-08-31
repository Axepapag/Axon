"""Headless command surface for Axon's future-pluggable Trainer organ.

The runtime UI, slash commands, local launchers, and cloud adapters must all
call this surface (or a compatible transport around it).  The organ owns no UI
framework and does not execute shell commands.  Mutating operations are
unavailable until a governed handler is explicitly registered; status remains
read-only and lease-free.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from runtime.field import canonical_sha256

from .inspection import inspect_trainer_state

TRAINER_ORGAN_COMMAND_SCHEMA = "axon-trainer-organ-command-v1"
TRAINER_ORGAN_RESULT_SCHEMA = "axon-trainer-organ-result-v1"
TRAINER_ORGAN_STATUS_SUMMARY_SCHEMA = "axon-trainer-organ-status-summary-v1"


class TrainerCommandKind(str, Enum):
    STATUS = "status"
    INVENTORY = "inventory"
    PREFLIGHT = "preflight"
    CONFIGURE = "configure"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    EVALUATE = "evaluate"
    EXPORT_CLOUD_PACKET = "export_cloud_packet"
    IMPORT_CLOUD_RESULT = "import_cloud_result"
    COMPARE = "compare"
    REQUEST_PROMOTION = "request_promotion"
    ROLLBACK = "rollback"


class TrainerCommandStatus(str, Enum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


def _json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only JSON values") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - mapping always yields object
        raise ValueError(f"{label} must encode one JSON object")
    return decoded


@dataclass(frozen=True, slots=True)
class TrainerOrganCommand:
    kind: TrainerCommandKind | str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    requested_by: str = "operator"
    command_id: str = field(init=False)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, TrainerCommandKind) else TrainerCommandKind(self.kind)
        object.__setattr__(self, "kind", kind)
        arguments = _json_object(self.arguments, "arguments")
        object.__setattr__(self, "arguments", arguments)
        requested_by = str(self.requested_by).strip()
        if not requested_by:
            raise ValueError("requested_by must be non-empty")
        object.__setattr__(self, "requested_by", requested_by)
        object.__setattr__(self, "command_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRAINER_ORGAN_COMMAND_SCHEMA,
            "kind": self.kind.value,
            "arguments": dict(self.arguments),
            "requested_by": self.requested_by,
        }
        if include_id:
            value["command_id"] = self.command_id
        return value


@dataclass(frozen=True, slots=True)
class TrainerOrganResult:
    command_id: str
    kind: TrainerCommandKind | str
    status: TrainerCommandStatus | str
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, TrainerCommandKind) else TrainerCommandKind(self.kind)
        status = self.status if isinstance(self.status, TrainerCommandStatus) else TrainerCommandStatus(self.status)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "payload", _json_object(self.payload, "payload"))
        if not isinstance(self.command_id, str) or len(self.command_id) != 64:
            raise ValueError("command_id must be a content identity")
        if self.error is not None:
            error = str(self.error).strip()
            if not error:
                raise ValueError("error must be non-empty when supplied")
            object.__setattr__(self, "error", error)
        if status is TrainerCommandStatus.COMPLETED and self.error is not None:
            raise ValueError("a completed Trainer command cannot carry an error")
        object.__setattr__(self, "result_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRAINER_ORGAN_RESULT_SCHEMA,
            "command_id": self.command_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "payload": dict(self.payload),
            "error": self.error,
        }
        if include_id:
            value["result_id"] = self.result_id
        return value


TrainerCommandHandler = Callable[[TrainerOrganCommand], Mapping[str, Any]]


def _selected(value: Mapping[str, Any] | None, names: tuple[str, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    return {name: value.get(name) for name in names}


def _status_summary(snapshot) -> dict[str, Any]:
    """Reduce inspection evidence to a readable operator surface.

    The complete inspection remains available explicitly, but the default
    Trainer UI/CLI path must not bury state in per-tensor telemetry.
    """

    body = snapshot.to_canonical_dict()
    checkpoint_modules: dict[str, dict[str, Any]] = {}
    for checkpoint in snapshot.latest_checkpoints:
        module_id = str(checkpoint.get("module_id"))
        row = checkpoint_modules.setdefault(
            module_id,
            {"module_id": module_id, "candidate_count": 0, "maximum_step": 0},
        )
        row["candidate_count"] += 1
        row["maximum_step"] = max(row["maximum_step"], int(checkpoint.get("step", 0)))
    return {
        "schema": TRAINER_ORGAN_STATUS_SUMMARY_SCHEMA,
        "inspection_snapshot_id": snapshot.snapshot_id,
        "state_root": snapshot.state_root,
        "writer_lease_present": snapshot.lease is not None,
        "artifact_counts": body["artifact_counts"],
        "candidate_checkpoint_count": len(snapshot.latest_checkpoints),
        "latest_lifecycle": _selected(
            snapshot.latest_lifecycle,
            (
                "event_id",
                "module_id",
                "candidate_generation_id",
                "status",
                "step",
                "checkpoint_id",
                "reason",
            ),
        ),
        "latest_optimization_step": _selected(
            snapshot.latest_step,
            (
                "receipt_id",
                "module_id",
                "candidate_generation_id",
                "step",
                "loss",
                "learning_rate",
            ),
        ),
        "checkpoint_modules": [checkpoint_modules[key] for key in sorted(checkpoint_modules)],
        "active_generation_pointers": [
            _selected(
                item,
                (
                    "module_id",
                    "active_generation_id",
                    "checkpoint_id",
                ),
            )
            for item in snapshot.active_generation_pointers
        ],
    }


class TrainerOrgan:
    """One transport-neutral Trainer command boundary.

    Only read-only status is built in.  A launcher/runtime integrates governed
    capabilities by registering handlers backed by the deterministic Trainer
    control plane.  Missing handlers fail closed as ``unavailable`` rather
    than falling back to ad-hoc shell behavior.
    """

    def __init__(self, *, state_root: Path | str = Path(r"D:\Axon\State")) -> None:
        self.state_root = Path(state_root).resolve(strict=False)
        self._handlers: dict[TrainerCommandKind, TrainerCommandHandler] = {}

    @property
    def declared_commands(self) -> tuple[str, ...]:
        return tuple(item.value for item in TrainerCommandKind)

    @property
    def available_commands(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {TrainerCommandKind.STATUS.value}
                | {item.value for item in self._handlers}
            )
        )

    def register_handler(
        self,
        kind: TrainerCommandKind | str,
        handler: TrainerCommandHandler,
    ) -> None:
        command_kind = kind if isinstance(kind, TrainerCommandKind) else TrainerCommandKind(kind)
        if command_kind is TrainerCommandKind.STATUS:
            raise ValueError("the built-in read-only status command cannot be replaced")
        if not callable(handler):
            raise TypeError("Trainer command handler must be callable")
        if command_kind in self._handlers:
            raise ValueError(f"Trainer command handler already registered: {command_kind.value}")
        self._handlers[command_kind] = handler

    def dispatch(self, command: TrainerOrganCommand) -> TrainerOrganResult:
        if not isinstance(command, TrainerOrganCommand):
            raise TypeError("dispatch requires TrainerOrganCommand")
        if command.kind is TrainerCommandKind.STATUS:
            snapshot = inspect_trainer_state(state_root=self.state_root)
            detail = command.arguments.get("detail", "summary")
            if detail not in {"summary", "full"}:
                return TrainerOrganResult(
                    command_id=command.command_id,
                    kind=command.kind,
                    status=TrainerCommandStatus.REJECTED,
                    error="status detail must be 'summary' or 'full'",
                )
            return TrainerOrganResult(
                command_id=command.command_id,
                kind=command.kind,
                status=TrainerCommandStatus.COMPLETED,
                payload={
                    "inspection": (
                        snapshot.to_canonical_dict()
                        if detail == "full"
                        else _status_summary(snapshot)
                    ),
                    "declared_commands": list(self.declared_commands),
                    "available_commands": list(self.available_commands),
                },
            )
        handler = self._handlers.get(command.kind)
        if handler is None:
            return TrainerOrganResult(
                command_id=command.command_id,
                kind=command.kind,
                status=TrainerCommandStatus.UNAVAILABLE,
                error=(
                    f"Trainer command {command.kind.value!r} has no governed handler; "
                    "no action was taken"
                ),
            )
        try:
            payload = handler(command)
        except Exception as exc:  # handler boundary records failure, never fallback
            return TrainerOrganResult(
                command_id=command.command_id,
                kind=command.kind,
                status=TrainerCommandStatus.REJECTED,
                error=f"{type(exc).__name__}: {exc}",
            )
        return TrainerOrganResult(
            command_id=command.command_id,
            kind=command.kind,
            status=TrainerCommandStatus.COMPLETED,
            payload=payload,
        )


__all__ = [
    "TRAINER_ORGAN_COMMAND_SCHEMA",
    "TRAINER_ORGAN_RESULT_SCHEMA",
    "TRAINER_ORGAN_STATUS_SUMMARY_SCHEMA",
    "TrainerCommandKind",
    "TrainerCommandStatus",
    "TrainerOrgan",
    "TrainerOrganCommand",
    "TrainerOrganResult",
]

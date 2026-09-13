"""Durable rolling workspaces for every runtime-training attempt.

Accepted checkpoints remain governed landmarks.  This store serves a different
purpose: crash recovery of the exact in-progress model and optimizer after any
attempt, including a rejected one.  Payloads and records are hash verified, HEAD
advances atomically, and only the newest configured recovery window is retained.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from runtime.field import canonical_sha256

ATTEMPT_WORKSPACE_SCHEMA = "axon-trainer-attempt-workspace-v1"
ATTEMPT_WORKSPACE_HEAD_SCHEMA = "axon-trainer-attempt-workspace-head-v1"


class AttemptWorkspaceError(RuntimeError):
    """A work-in-progress checkpoint is missing, stale, or corrupt."""


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_bytes(path, payload)


@dataclass(frozen=True, slots=True)
class AttemptWorkspaceRecord:
    assignment_id: str
    core_id: str
    attempt_id: str
    attempt_index: int
    parameter_generation: str
    optimizer_generation: str
    payload_sha256: str
    payload_bytes: int
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "assignment_id",
            "core_id",
            "attempt_id",
            "parameter_generation",
            "optimizer_generation",
            "payload_sha256",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if isinstance(self.attempt_index, bool) or not isinstance(self.attempt_index, int) or self.attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        if isinstance(self.payload_bytes, bool) or not isinstance(self.payload_bytes, int) or self.payload_bytes < 1:
            raise ValueError("payload_bytes must be a positive integer")
        object.__setattr__(self, "record_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": ATTEMPT_WORKSPACE_SCHEMA,
            "assignment_id": self.assignment_id,
            "core_id": self.core_id,
            "attempt_id": self.attempt_id,
            "attempt_index": self.attempt_index,
            "parameter_generation": self.parameter_generation,
            "optimizer_generation": self.optimizer_generation,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
        }
        if include_id:
            value["record_id"] = self.record_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AttemptWorkspaceRecord":
        item = dict(value)
        if item.pop("schema", None) != ATTEMPT_WORKSPACE_SCHEMA:
            raise AttemptWorkspaceError("attempt workspace schema mismatch")
        expected = item.pop("record_id", None)
        try:
            record = cls(**item)
        except (TypeError, ValueError) as exc:
            raise AttemptWorkspaceError("attempt workspace record is malformed") from exc
        if record.record_id != expected:
            raise AttemptWorkspaceError("attempt workspace record identity mismatch")
        return record


class AttemptWorkspaceStore:
    """Rolling, atomic recovery checkpoints for one trainer state root."""

    def __init__(self, state_root: Path | str, *, retain: int = 3) -> None:
        if isinstance(retain, bool) or not isinstance(retain, int) or retain < 1:
            raise ValueError("retain must be a positive integer")
        self.root = Path(state_root).resolve(strict=False) / "training" / "attempt_workspaces"
        self.retain = retain

    def _assignment_root(self, core_id: str, assignment_id: str) -> Path:
        core_key = canonical_sha256(_required(core_id, "core_id"))[:16]
        assignment = _required(assignment_id, "assignment_id")
        if len(assignment) != 64 or any(ch not in "0123456789abcdef" for ch in assignment):
            raise ValueError("assignment_id must be a lowercase sha256 identity")
        return self.root / core_key / assignment[:24]

    def save(
        self,
        *,
        assignment_id: str,
        core_id: str,
        attempt_id: str,
        attempt_index: int,
        parameter_generation: str,
        optimizer_generation: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> AttemptWorkspaceRecord:
        buffer = io.BytesIO()
        torch.save(
            {
                "module_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            buffer,
        )
        payload = buffer.getvalue()
        record = AttemptWorkspaceRecord(
            assignment_id=assignment_id,
            core_id=core_id,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            parameter_generation=parameter_generation,
            optimizer_generation=optimizer_generation,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            payload_bytes=len(payload),
        )
        root = self._assignment_root(core_id, assignment_id)
        descriptor_path = root / "assignment.json"
        descriptor = {
            "assignment_id": assignment_id,
            "core_id": core_id,
        }
        if descriptor_path.is_file():
            try:
                existing_descriptor = json.loads(
                    descriptor_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise AttemptWorkspaceError(
                    "attempt workspace assignment descriptor is malformed"
                ) from exc
            if existing_descriptor != descriptor:
                raise AttemptWorkspaceError(
                    "short workspace path collides with another assignment"
                )
        else:
            _atomic_json(descriptor_path, descriptor)
        stem = f"{attempt_index:08d}"
        payload_path = root / f"{stem}.pt"
        record_path = root / f"{stem}.json"
        if record_path.is_file() or payload_path.is_file():
            try:
                existing_record = AttemptWorkspaceRecord.from_mapping(
                    json.loads(record_path.read_text(encoding="utf-8"))
                )
                existing_payload = payload_path.read_bytes()
            except (OSError, json.JSONDecodeError) as exc:
                raise AttemptWorkspaceError(
                    "existing attempt workspace is incomplete"
                ) from exc
            if (
                existing_record != record
                or hashlib.sha256(existing_payload).hexdigest()
                != record.payload_sha256
            ):
                raise AttemptWorkspaceError(
                    "attempt index already names different recovery bytes"
                )
        else:
            _atomic_bytes(payload_path, payload)
            _atomic_json(record_path, record.to_canonical_dict())
        _atomic_json(
            root / "HEAD.json",
            {
                "schema": ATTEMPT_WORKSPACE_HEAD_SCHEMA,
                "record_id": record.record_id,
                "attempt_index": record.attempt_index,
                "record_file": record_path.name,
                "payload_file": payload_path.name,
            },
        )
        self._prune(root)
        return record

    def load_latest(
        self, *, assignment_id: str, core_id: str
    ) -> tuple[AttemptWorkspaceRecord, dict[str, Any]] | None:
        root = self._assignment_root(core_id, assignment_id)
        head_path = root / "HEAD.json"
        if not head_path.is_file():
            return None
        try:
            head = json.loads(head_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AttemptWorkspaceError("attempt workspace HEAD is malformed") from exc
        if head.get("schema") != ATTEMPT_WORKSPACE_HEAD_SCHEMA:
            raise AttemptWorkspaceError("attempt workspace HEAD schema mismatch")
        record_path = root / str(head.get("record_file", ""))
        payload_path = root / str(head.get("payload_file", ""))
        try:
            record = AttemptWorkspaceRecord.from_mapping(
                json.loads(record_path.read_text(encoding="utf-8"))
            )
            payload = payload_path.read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise AttemptWorkspaceError("attempt workspace artifact is missing or malformed") from exc
        if record.record_id != head.get("record_id") or record.attempt_index != head.get("attempt_index"):
            raise AttemptWorkspaceError("attempt workspace HEAD does not bind its record")
        if len(payload) != record.payload_bytes or hashlib.sha256(payload).hexdigest() != record.payload_sha256:
            raise AttemptWorkspaceError("attempt workspace payload hash mismatch")
        try:
            value = torch.load(io.BytesIO(payload), map_location="cpu")
        except Exception as exc:
            raise AttemptWorkspaceError("attempt workspace payload cannot be loaded") from exc
        if not isinstance(value, dict) or set(value) != {"module_state_dict", "optimizer_state_dict"}:
            raise AttemptWorkspaceError("attempt workspace payload fields are invalid")
        return record, value

    def _prune(self, root: Path) -> None:
        records = sorted(root.glob("????????.json"))
        for record_path in records[: -self.retain]:
            payload_path = record_path.with_suffix(".pt")
            with suppress(FileNotFoundError):
                record_path.unlink()
            with suppress(FileNotFoundError):
                payload_path.unlink()


__all__ = [
    "ATTEMPT_WORKSPACE_HEAD_SCHEMA",
    "ATTEMPT_WORKSPACE_SCHEMA",
    "AttemptWorkspaceError",
    "AttemptWorkspaceRecord",
    "AttemptWorkspaceStore",
]

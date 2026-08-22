"""Durable cardiac identity and monotonic Heart counters.

Counter ids are reserved by atomically persisting the increment *before* the
corresponding heartbeat/tick is used.  A crash may therefore leave a gap, but
can never cause an identity to be reused after restart.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .errors import HealthCorruptionError

_IDENTITY_SCHEMA = "axon-heart-identity-v2"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HealthCorruptionError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class HeartIdentity:
    """Durable identity of the active Heart lineage and current host start."""

    heart_epoch_id: str
    start_sequence: int
    heartbeat_sequence: int
    tick_sequence: int
    started_at: datetime

    @property
    def host_epoch(self) -> str:
        """Backward-compatible alias for the durable Heart epoch id."""

        return self.heart_epoch_id


class HeartIdentityStore:
    """Atomic store for restart-safe heartbeat/tick identity reservation."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._identity = self._load_or_create()

    @property
    def identity(self) -> HeartIdentity:
        return self._identity

    def begin_start(self) -> HeartIdentity:
        """Record a new process start without resetting cardiac counters."""

        current = self._identity
        updated = HeartIdentity(
            heart_epoch_id=current.heart_epoch_id,
            start_sequence=current.start_sequence + 1,
            heartbeat_sequence=current.heartbeat_sequence,
            tick_sequence=current.tick_sequence,
            started_at=_now(),
        )
        self._save(updated)
        self._identity = updated
        return updated

    def next_heartbeat(self) -> HeartIdentity:
        """Reserve and persist the next heartbeat identity."""

        current = self._identity
        updated = HeartIdentity(
            heart_epoch_id=current.heart_epoch_id,
            start_sequence=current.start_sequence,
            heartbeat_sequence=current.heartbeat_sequence + 1,
            tick_sequence=current.tick_sequence,
            started_at=current.started_at,
        )
        self._save(updated)
        self._identity = updated
        return updated

    def next_tick(self) -> HeartIdentity:
        """Reserve and persist the next cognitive tick identity."""

        current = self._identity
        if current.heartbeat_sequence < 1:
            raise HealthCorruptionError("cannot reserve a tick before the first heartbeat")
        updated = HeartIdentity(
            heart_epoch_id=current.heart_epoch_id,
            start_sequence=current.start_sequence,
            heartbeat_sequence=current.heartbeat_sequence,
            tick_sequence=current.tick_sequence + 1,
            started_at=current.started_at,
        )
        self._save(updated)
        self._identity = updated
        return updated

    def recover(self) -> HeartIdentity:
        """Reload durable state and fail closed on counter rollback."""

        loaded = self._load()
        current = self._identity
        if loaded.heartbeat_sequence < current.heartbeat_sequence:
            raise HealthCorruptionError("heartbeat_sequence decreased on recovery")
        if loaded.tick_sequence < current.tick_sequence:
            raise HealthCorruptionError("tick_sequence decreased on recovery")
        if loaded.start_sequence < current.start_sequence:
            raise HealthCorruptionError("start_sequence decreased on recovery")
        if loaded.heart_epoch_id != current.heart_epoch_id:
            raise HealthCorruptionError("Heart epoch changed unexpectedly on recovery")
        self._identity = loaded
        return loaded

    def _load_or_create(self) -> HeartIdentity:
        if self.path.exists():
            return self._load()
        identity = HeartIdentity(
            heart_epoch_id=uuid.uuid4().hex,
            start_sequence=0,
            heartbeat_sequence=0,
            tick_sequence=0,
            started_at=_now(),
        )
        self._save(identity)
        return identity

    def _load(self) -> HeartIdentity:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HealthCorruptionError(
                f"cannot read Heart identity store: {self.path}"
            ) from exc
        if data.get("schema") != _IDENTITY_SCHEMA:
            raise HealthCorruptionError(
                f"unsupported Heart identity schema: {data.get('schema')!r}"
            )
        try:
            started_at = datetime.fromisoformat(data["started_at"])
            epoch = str(data["heart_epoch_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HealthCorruptionError("invalid Heart identity metadata") from exc
        if not epoch:
            raise HealthCorruptionError("heart_epoch_id must be non-empty")
        if started_at.tzinfo is None:
            raise HealthCorruptionError("started_at must be timezone-aware")
        return HeartIdentity(
            heart_epoch_id=epoch,
            start_sequence=_nonnegative_int(data["start_sequence"], "start_sequence"),
            heartbeat_sequence=_nonnegative_int(
                data["heartbeat_sequence"], "heartbeat_sequence"
            ),
            tick_sequence=_nonnegative_int(data["tick_sequence"], "tick_sequence"),
            started_at=started_at,
        )

    def _save(self, identity: HeartIdentity) -> None:
        value = {
            "schema": _IDENTITY_SCHEMA,
            "heart_epoch_id": identity.heart_epoch_id,
            "start_sequence": identity.start_sequence,
            "heartbeat_sequence": identity.heartbeat_sequence,
            "tick_sequence": identity.tick_sequence,
            "started_at": identity.started_at.isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        data = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)


__all__ = ["HeartIdentity", "HeartIdentityStore"]

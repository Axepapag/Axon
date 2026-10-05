"""Durable read-only observability for the permanent Axon Heart."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import HealthCorruptionError

_HEALTH_SCHEMA = "axon-heart-health-v2"


@dataclass(frozen=True, slots=True)
class HeartHealth:
    heart_epoch_id: str
    start_sequence: int
    heartbeat_sequence: int
    tick_sequence: int
    last_beat_at: datetime | None
    last_successful_circulation_at: datetime | None
    last_failure_reason: str | None
    canonical_head_field_id: str | None
    canonical_head_tick_id: int | None
    tick_in_flight: bool
    queue_depth_by_valve: dict[str, int]
    quarantine_count: int
    rejection_count: int
    valve_states: dict[str, Any]
    dormant_index_id: str | None
    lease_owner_pid: int | None
    lease_owner_token: str | None
    last_tick_uid: str | None = None
    last_view_id: str | None = None
    mask_state_id: str | None = None
    mask_revision: int | None = None
    region_masks: dict[str, Any] | None = None


class HealthJournal:
    """Append-only health history plus an atomically replaced latest snapshot."""

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        self.journal_path = path if path.suffix == ".jsonl" else path / "health.jsonl"
        self.latest_path = self.journal_path.with_name("health_latest.json")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, health: HeartHealth) -> None:
        if not isinstance(health, HeartHealth):
            raise TypeError("HealthJournal.write requires HeartHealth")
        record = {"schema": _HEALTH_SCHEMA, **asdict(health)}
        for key in ("last_beat_at", "last_successful_circulation_at"):
            value = record[key]
            record[key] = value.isoformat() if isinstance(value, datetime) else None
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = self.latest_path.with_name(self.latest_path.name + f".{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.latest_path)

    def latest(self) -> dict[str, Any]:
        if not self.latest_path.exists():
            return {}
        try:
            value = json.loads(self.latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HealthCorruptionError("corrupt Heart health metadata") from exc
        if value.get("schema") != _HEALTH_SCHEMA:
            raise HealthCorruptionError("unsupported Heart health schema")
        return value


__all__ = ["HealthJournal", "HeartHealth"]

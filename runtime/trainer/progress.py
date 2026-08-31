"""Durable, transport-neutral progress events for long-running training.

The JSONL journal and atomic ``current.json`` survive a launcher, terminal, or
engineering-agent disconnect.  The same compact event is also written to
stdout so a cloud provider's native log stream remains useful.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from runtime.field import canonical_sha256

TRAINING_PROGRESS_EVENT_SCHEMA = "axon-training-progress-event-v1"
_SAFE_STATUSES = {
    "starting",
    "training",
    "evaluating",
    "paused",
    "completed",
    "failed",
}


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - Mapping guarantees it
        raise ValueError("progress details must encode one JSON object")
    return decoded


@dataclass(frozen=True, slots=True)
class TrainingProgressEvent:
    job_id: str
    status: str
    sequence: int
    occurred_at: str
    details: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id.strip():
            raise ValueError("job_id must be non-empty")
        status = str(self.status).strip().lower()
        if status not in _SAFE_STATUSES:
            raise ValueError(f"unsupported progress status {status!r}")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "details", _json_mapping(self.details))
        object.__setattr__(self, "event_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRAINING_PROGRESS_EVENT_SCHEMA,
            "job_id": self.job_id,
            "status": self.status,
            "sequence": self.sequence,
            "occurred_at": self.occurred_at,
            "details": dict(self.details),
        }
        if include_id:
            value["event_id"] = self.event_id
        return value


class TrainingProgressJournal:
    """Append progress durably and mirror it to an immediately flushed log."""

    def __init__(self, root: Path | str, *, job_id: str) -> None:
        self.root = Path(root).resolve(strict=False)
        self.job_id = str(job_id).strip()
        if not self.job_id:
            raise ValueError("job_id must be non-empty")
        self.events_path = self.root / "events.jsonl"
        self.current_path = self.root / "current.json"
        self._sequence = self._existing_sequence()

    def _existing_sequence(self) -> int:
        if not self.current_path.is_file():
            return 0
        value = json.loads(self.current_path.read_text(encoding="utf-8"))
        if value.get("schema") != TRAINING_PROGRESS_EVENT_SCHEMA:
            raise RuntimeError("progress current.json has an unsupported schema")
        if value.get("job_id") != self.job_id:
            raise RuntimeError("progress directory belongs to another job")
        return int(value["sequence"])

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        body = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def emit(self, status: str, **details: Any) -> TrainingProgressEvent:
        self._sequence += 1
        event = TrainingProgressEvent(
            job_id=self.job_id,
            status=status,
            sequence=self._sequence,
            occurred_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            details={"monotonic_seconds": time.monotonic(), **details},
        )
        value = event.to_canonical_dict()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._atomic_json(self.current_path, value)
        print(
            "AXON_PROGRESS " + json.dumps(value, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        return event

"""OS-backed single-writer lease for Axon's Trainer parameter authority."""
from __future__ import annotations

import json
import os
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRAINER_LEASE_SCHEMA = "axon-trainer-lease-v1"
_PROCESS_LOCKS: dict[str, str] = {}


class TrainerLeaseDeniedError(RuntimeError):
    pass


def _owner_token() -> str:
    return f"{uuid.uuid4().hex}-{os.getpid()}"


def _host_epoch() -> str:
    node = platform.node() or "unknown"
    return f"{node}-{uuid.getnode():x}"


class TrainerWriterLease:
    """Hold exclusive writer authority over ``State/training/trainer``."""

    def __init__(self, state_root: Path | str, owner_token: str | None = None) -> None:
        self.state_root = Path(state_root).resolve(strict=False)
        self.trainer_dir = self.state_root / "training" / "trainer"
        self.authority_dir = self.trainer_dir / "authority"
        self.lease_path = self.authority_dir / "lease.json"
        self.lock_path = self.authority_dir / "trainer.lock"
        self._owner_token = owner_token or _owner_token()
        self._handle: Any | None = None
        self._record: dict[str, Any] | None = None

    @property
    def owner_token(self) -> str:
        return self._owner_token

    def acquire(self) -> dict[str, Any]:
        if self._handle is not None:
            return dict(self._record or {})
        self.authority_dir.mkdir(parents=True, exist_ok=True)
        key = str(self.lock_path).casefold()
        other = _PROCESS_LOCKS.get(key)
        if other is not None and other != self._owner_token:
            raise TrainerLeaseDeniedError("another Trainer already owns parameter-writer authority")

        if self.lease_path.exists():
            try:
                existing = json.loads(self.lease_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TrainerLeaseDeniedError(f"corrupt or unreadable Trainer lease metadata: {self.lease_path}") from exc
            if existing.get("schema") != _TRAINER_LEASE_SCHEMA:
                raise TrainerLeaseDeniedError("unsupported Trainer lease metadata schema")

        handle = self.lock_path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            self._lock_handle(handle)
        except Exception as exc:
            handle.close()
            if isinstance(exc, TrainerLeaseDeniedError):
                raise
            raise TrainerLeaseDeniedError(f"another Trainer already holds {self.lock_path}") from exc

        _PROCESS_LOCKS[key] = self._owner_token
        self._handle = handle
        record = {
            "schema": _TRAINER_LEASE_SCHEMA,
            "owner_token": self._owner_token,
            "pid": os.getpid(),
            "host_epoch": _host_epoch(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "lock_path": str(self.lock_path),
        }
        try:
            self._atomic_json(self.lease_path, record)
        except Exception:
            self.release()
            raise
        self._record = record
        return dict(record)

    def is_held_by_us(self) -> bool:
        if self._handle is None:
            return False
        key = str(self.lock_path).casefold()
        if _PROCESS_LOCKS.get(key) != self._owner_token:
            return False
        try:
            record = json.loads(self.lease_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return record.get("owner_token") == self._owner_token

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        key = str(self.lock_path).casefold()
        if handle is not None:
            try:
                self._unlock_handle(handle)
            finally:
                handle.close()
        if _PROCESS_LOCKS.get(key) == self._owner_token:
            _PROCESS_LOCKS.pop(key, None)
        if self.lease_path.exists():
            try:
                current = json.loads(self.lease_path.read_text(encoding="utf-8"))
            except Exception:
                current = None
            if isinstance(current, dict) and current.get("owner_token") == self._owner_token:
                try:
                    self.lease_path.unlink()
                except FileNotFoundError:
                    pass
        try:
            self.lock_path.unlink()
        except OSError:
            pass
        self._record = None

    def __enter__(self) -> "TrainerWriterLease":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    @staticmethod
    def _lock_handle(handle: Any) -> None:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise TrainerLeaseDeniedError("OS Trainer writer lock is already held") from exc
            return
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise TrainerLeaseDeniedError("OS Trainer writer lock is already held") from exc

    @staticmethod
    def _unlock_handle(handle: Any) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            return
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        data = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


__all__ = ["TrainerLeaseDeniedError", "TrainerWriterLease"]

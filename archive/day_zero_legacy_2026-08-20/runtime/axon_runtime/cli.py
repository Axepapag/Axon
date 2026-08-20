"""Command-line control plane for the explicit Axon runtime process."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import uuid
from typing import Any, Sequence

from .bootstrap import RuntimeBootstrapError, materialize_runtime
from .config import RuntimeBootstrapConfig, load_runtime_config
from .ingress import INGRESS_EVENT_KINDS, IngressEvent, IngressQueue
from .store import RuntimeStore


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "ops" / "axon_runtime.cpu-smoke.json"
)


def _json_print(value: Any, *, stream: Any = None) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        file=sys.stdout if stream is None else stream,
    )


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def runtime_status(config: RuntimeBootstrapConfig) -> dict[str, Any]:
    """Read status without creating directories, schemas, or checkpoint loads."""

    if not isinstance(config, RuntimeBootstrapConfig):
        raise TypeError("config must be RuntimeBootstrapConfig")
    path = config.paths.runtime_db
    result: dict[str, Any] = {
        "schema": "axon-runtime-status-v1",
        "config_id": config.config_id,
        "runtime_db": str(path),
        "initialized": False,
        "stop_requested": config.paths.stop_file.is_file(),
        "configured_ring": list(config.ring.core_ids),
        "capabilities": config.capabilities.to_canonical_dict(),
    }
    if not path.is_file():
        return result
    connection = _readonly_connection(path)
    try:
        tables = _table_names(connection)
        required = {
            "runtime_head",
            "core_identities",
            "core_states",
            "journal",
        }
        if not required <= tables:
            result["error"] = "runtime database schema is incomplete"
            return result
        head = connection.execute(
            """
            SELECT generation, tick_seq, field_id, role_index,
                   role_assignment_hash, projection_manifest_id
            FROM runtime_head
            WHERE singleton=1
            """
        ).fetchone()
        if head is None:
            return result
        core_ids = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT core_id FROM core_identities ORDER BY ring_index"
            )
        )
        state_rows = connection.execute(
            """
            SELECT core_id, manifest_id
            FROM core_states
            ORDER BY core_id
            """
        ).fetchall()
        tick_seq = int(head["tick_seq"])
        ring = config.ring.core_ids
        index = tick_seq % len(ring)
        result.update(
            {
                "initialized": True,
                "generation": int(head["generation"]),
                "tick_seq": tick_seq,
                "field_id": str(head["field_id"]),
                "role_index": int(head["role_index"]),
                "role_assignment_hash": str(
                    head["role_assignment_hash"]
                ),
                "projection_manifest_id": str(
                    head["projection_manifest_id"]
                ),
                "persisted_core_ids": list(core_ids),
                "ring_matches_config": core_ids == ring,
                "core_state_manifests": {
                    str(row["core_id"]): str(row["manifest_id"])
                    for row in state_rows
                },
                "next_roles": {
                    "proposer": ring[index],
                    "consolidator": ring[(index - 1) % len(ring)],
                    "sleeper": ring[(index - 2) % len(ring)],
                    "standby": [
                        core_id
                        for core_id in ring
                        if core_id
                        not in {
                            ring[index],
                            ring[(index - 1) % len(ring)],
                            ring[(index - 2) % len(ring)],
                        }
                    ],
                },
                "journal_records": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM journal"
                    ).fetchone()[0]
                ),
            }
        )
        if {
            "axon_ingress_events",
            "axon_ingress_consumption_receipts",
        } <= tables:
            result["pending_ingress"] = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM axon_ingress_events AS event
                    LEFT JOIN axon_ingress_consumption_receipts AS receipt
                      ON receipt.event_id=event.event_id
                    WHERE receipt.event_id IS NULL
                    """
                ).fetchone()[0]
            )
        else:
            result["pending_ingress"] = 0
        if "tool_requests" in tables:
            result["tool_outbox"] = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM tool_requests
                    GROUP BY status
                    ORDER BY status
                    """
                )
            }
        else:
            result["tool_outbox"] = {}
        return result
    finally:
        connection.close()


def enqueue_event(
    config: RuntimeBootstrapConfig,
    *,
    event_kind: str,
    exact_text: str,
    source: str = "cli",
    source_sequence: int | None = None,
    idempotency_key: str | None = None,
    provenance: str = "axon-runtime-cli",
) -> IngressEvent:
    """Append one exact event, allocating its source sequence atomically."""

    if not config.paths.runtime_db.is_file():
        raise RuntimeBootstrapError(
            "runtime is not initialized; run it once before enqueueing"
        )
    store = RuntimeStore(config.paths.runtime_db)
    try:
        store.recover()
        queue = IngressQueue(store.connection)
        connection = store.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            key = idempotency_key or f"cli:{uuid.uuid4()}"
            existing = connection.execute(
                """
                SELECT canonical_json
                FROM axon_ingress_events
                WHERE idempotency_key=?
                """,
                (key,),
            ).fetchone()
            if existing is not None:
                persisted = IngressEvent.from_dict(json.loads(existing[0]))
                if (
                    persisted.source != source
                    or persisted.event_kind != event_kind
                    or persisted.exact_text != exact_text
                    or persisted.provenance != provenance
                    or (
                        source_sequence is not None
                        and persisted.source_sequence != source_sequence
                    )
                ):
                    raise RuntimeBootstrapError(
                        "idempotency key already names different ingress"
                    )
                connection.execute("COMMIT")
                return persisted
            sequence = source_sequence
            if sequence is None:
                latest = connection.execute(
                    """
                    SELECT MAX(source_sequence)
                    FROM axon_ingress_events
                    WHERE source=?
                    """,
                    (source,),
                ).fetchone()[0]
                sequence = 0 if latest is None else int(latest) + 1
            event = IngressEvent(
                idempotency_key=key,
                source=source,
                source_sequence=sequence,
                event_kind=event_kind,
                exact_text=exact_text,
                provenance=provenance,
            )
            queue.enqueue(event)
            connection.execute("COMMIT")
            return event
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    finally:
        store.close()


def write_stop_marker(config: RuntimeBootstrapConfig) -> dict[str, Any]:
    """Atomically and idempotently request an intentional runtime shutdown."""

    config.paths.state_root.mkdir(parents=True, exist_ok=True)
    marker = config.paths.stop_file
    payload = {
        "schema": "axon-runtime-stop-v1",
        "config_id": config.config_id,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    data = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    try:
        descriptor = os.open(
            marker,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError:
        return {
            "schema": "axon-runtime-stop-result-v1",
            "created": False,
            "stop_file": str(marker),
        }
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            marker.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "schema": "axon-runtime-stop-result-v1",
        "created": True,
        "stop_file": str(marker),
    }


def run_runtime(
    config: RuntimeBootstrapConfig,
    *,
    max_ticks: int | None = None,
    clear_stop: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Materialize and run boundedly or until an explicit stop signal."""

    if config.paths.stop_file.exists():
        if not clear_stop:
            raise RuntimeBootstrapError(
                f"stop marker exists: {config.paths.stop_file}; "
                "pass --clear-stop to remove it deliberately"
            )
        config.paths.stop_file.unlink()

    stop_event = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        except (OSError, ValueError):
            previous_handlers.pop(signum, None)

    errors: list[str] = []

    def record_error(exc: Exception) -> None:
        errors.append(f"{type(exc).__name__}: {exc}")

    try:
        with materialize_runtime(config, device=device) as runtime:
            starting_tick = runtime.store.recover().tick_seq
            results = runtime.engine.run_forever(
                stop_event=stop_event,
                stop_file=config.paths.stop_file,
                tick_interval_seconds=config.loop.tick_interval_ms / 1000.0,
                error_backoff_initial_seconds=(
                    config.loop.failure_backoff_initial_ms / 1000.0
                ),
                error_backoff_max_seconds=(
                    config.loop.failure_backoff_max_ms / 1000.0
                ),
                error_backoff_multiplier=(
                    config.loop.failure_backoff_multiplier
                ),
                stop_poll_interval_seconds=(
                    config.loop.stop_file_poll_ms / 1000.0
                ),
                max_ticks=max_ticks,
                on_error=record_error,
            )
            head = runtime.store.recover()
            return {
                "schema": "axon-runtime-run-result-v1",
                "config_id": config.config_id,
                "initialized_new_store": runtime.initialized_new_store,
                "starting_tick_seq": starting_tick,
                "final_tick_seq": head.tick_seq,
                "ticks_completed": head.tick_seq - starting_tick,
                "bounded_results_retained": len(results),
                "field_id": head.snapshot.field_id,
                "stop_requested": (
                    stop_event.is_set() or config.paths.stop_file.exists()
                ),
                "errors": errors,
                "tool_effects_enabled": False,
                "advisor_effects_enabled": False,
            }
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m runtime.axon_runtime",
        description="Operate the explicit continuously ticking Axon runtime.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"strict runtime JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="read state without loading checkpoints")

    enqueue = commands.add_parser("enqueue", help="append exact runtime ingress")
    enqueue.add_argument("--kind", choices=sorted(INGRESS_EVENT_KINDS), required=True)
    enqueue.add_argument("--text", required=True)
    enqueue.add_argument("--source", default="cli")
    enqueue.add_argument("--sequence", type=int)
    enqueue.add_argument("--idempotency-key")
    enqueue.add_argument("--provenance", default="axon-runtime-cli")

    run = commands.add_parser("run", help="start bounded or continuous ticking")
    run.add_argument("--max-ticks", type=int)
    run.add_argument("--clear-stop", action="store_true")
    run.add_argument("--device", default="cpu")

    commands.add_parser("stop", help="write the durable stop marker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_runtime_config(args.config)
        if args.command == "status":
            _json_print(runtime_status(config))
        elif args.command == "enqueue":
            event = enqueue_event(
                config,
                event_kind=args.kind,
                exact_text=args.text,
                source=args.source,
                source_sequence=args.sequence,
                idempotency_key=args.idempotency_key,
                provenance=args.provenance,
            )
            _json_print(event.to_dict())
        elif args.command == "stop":
            _json_print(write_stop_marker(config))
        elif args.command == "run":
            if args.max_ticks is not None and args.max_ticks < 0:
                raise ValueError("--max-ticks must be non-negative")
            _json_print(
                run_runtime(
                    config,
                    max_ticks=args.max_ticks,
                    clear_stop=args.clear_stop,
                    device=args.device,
                )
            )
        else:  # pragma: no cover - argparse guarantees a known command.
            raise AssertionError(args.command)
    except Exception as exc:
        _json_print(
            {
                "schema": "axon-runtime-cli-error-v1",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            stream=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "runtime_status",
    "enqueue_event",
    "write_stop_marker",
    "run_runtime",
    "main",
]

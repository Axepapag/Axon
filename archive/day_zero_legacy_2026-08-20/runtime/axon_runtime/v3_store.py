"""Isolated SQLite journal and mutable projections for the v3 protocol slice.

The v3 store owns one SQLite database per state root.  It is intentionally
separate from the v2 runtime store: no migration, no attachment, no shared
rows.  Authority is the append-only hash-chained journal; the mutable head is a
replayable projection.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from runtime.field import SharedFieldSnapshot, canonical_sha256

from .field_transaction import (
    FieldTransactionAudit,
    SystemFieldUpdate,
    replay_field_transaction,
)
from .serde import (
    deserialize_field_delta,
    deserialize_shared_field,
    deserialize_system_field_update,
    serialize_field_delta,
)
from .v3_contracts import (
    ArtifactRejectionRecord,
    CorePassRecord,
    ProposalBoardManifest,
    ReadCycleManifest,
    SoulTransitionDispositionRecord,
    SoulTransitionRecord,
    TickCommitRecordV3,
    TickPhasePlan,
    V3ContractError,
)
from .v3_serde import (
    canonical_json_text,
    deserialize_artifact_rejection,
    deserialize_consolidation,
    deserialize_core_pass,
    deserialize_proposal_board,
    deserialize_read_cycle_manifest,
    deserialize_soul_transition,
    deserialize_soul_transition_disposition,
    deserialize_tick_commit_v3,
    deserialize_tick_phase_plan,
    parse_json_object,
    serialize_consolidation,
    serialize_core_pass,
    serialize_proposal_board,
    serialize_read_cycle_manifest,
    serialize_soul_transition,
    serialize_soul_transition_disposition,
    serialize_tick_commit_v3,
    serialize_tick_phase_plan,
)
from .v3_transaction import (
    ActiveProjectionArtifactV3,
    PrivateStateArtifactV3,
    ReadPageArtifactV3,
    TickTransactionV3,
    TickReplayV3,
    V3TransactionError,
    validate_tick_transaction_v3,
)


V3_STORE_SCHEMA = "axon-runtime-store-v3"
V3_DATABASE_NAME = "runtime-v3.sqlite3"
V3_MAX_JOURNAL_PAYLOAD_BYTES = 8 * 1024 * 1024


class V3StoreError(V3ContractError):
    """Base class for v3 store contract failures."""


class V3AlreadyInitializedError(V3StoreError):
    """The v3 store is already initialized."""


class V3NotInitializedError(V3StoreError):
    """The v3 store has not been initialized."""


class V3StaleContinuationError(V3StoreError):
    """The supplied continuation proof does not match the current head."""


class V3JournalIntegrityError(V3StoreError):
    """The journal chain or its semantic content is corrupt."""


class V3PopulationMismatchError(V3StoreError):
    """A transaction population does not match the sealed head."""


class V3ReplayError(V3StoreError):
    """Replayed journal truth does not match the mutable head projection."""


@dataclass(frozen=True, slots=True)
class V3RuntimeHead:
    """Immutable sealed projection of the v3 runtime head."""

    generation: int
    tick_seq: int
    snapshot: SharedFieldSnapshot
    core_state_leaf_ids: tuple[tuple[str, str], ...]
    soul_sha256_by_core: tuple[tuple[str, str], ...]
    model_binding_epoch_id: str
    last_commit_id: str | None
    journal_tip_event_id: str
    head_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "head_id", canonical_sha256(self.to_canonical_dict())
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-runtime-head-v3",
            "generation": self.generation,
            "tick_seq": self.tick_seq,
            "field_id": self.snapshot.field_id,
            "core_state_leaf_ids": dict(self.core_state_leaf_ids),
            "soul_sha256_by_core": dict(self.soul_sha256_by_core),
            "model_binding_epoch_id": self.model_binding_epoch_id,
            "last_commit_id": self.last_commit_id,
            "journal_tip_event_id": self.journal_tip_event_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["snapshot"] = self.snapshot.to_dict()
        value["head_id"] = self.head_id
        return value


@dataclass(frozen=True, slots=True)
class V3ContinuationProof:
    """All state needed to reject an ABA/stale writer."""

    head_id: str
    generation: int
    tick_seq: int
    field_id: str
    core_state_leaf_ids: tuple[tuple[str, str], ...]
    soul_sha256_by_core: tuple[tuple[str, str], ...]
    model_binding_epoch_id: str
    last_commit_id: str | None
    journal_tip_event_id: str
    proof_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proof_id", canonical_sha256(self.to_canonical_dict())
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-continuation-proof-v3",
            "head_id": self.head_id,
            "generation": self.generation,
            "tick_seq": self.tick_seq,
            "field_id": self.field_id,
            "core_state_leaf_ids": dict(self.core_state_leaf_ids),
            "soul_sha256_by_core": dict(self.soul_sha256_by_core),
            "model_binding_epoch_id": self.model_binding_epoch_id,
            "last_commit_id": self.last_commit_id,
            "journal_tip_event_id": self.journal_tip_event_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["proof_id"] = self.proof_id
        return value


def _head_from_replay(
    generation: int,
    tick_seq: int,
    snapshot: SharedFieldSnapshot,
    leaf_ids: Mapping[str, str],
    souls: Mapping[str, str],
    model_binding_epoch_id: str,
    last_commit_id: str | None,
    journal_tip_event_id: str,
) -> V3RuntimeHead:
    population = sorted(leaf_ids)
    if sorted(souls) != population:
        raise V3JournalIntegrityError("leaf and soul populations mismatch")
    if not population:
        raise V3JournalIntegrityError("head population cannot be empty")
    return V3RuntimeHead(
        generation=generation,
        tick_seq=tick_seq,
        snapshot=snapshot,
        core_state_leaf_ids=tuple((core_id, leaf_ids[core_id]) for core_id in population),
        soul_sha256_by_core=tuple((core_id, souls[core_id]) for core_id in population),
        model_binding_epoch_id=model_binding_epoch_id,
        last_commit_id=last_commit_id,
        journal_tip_event_id=journal_tip_event_id,
    )


def _continuation_proof_from_head(head: V3RuntimeHead) -> V3ContinuationProof:
    return V3ContinuationProof(
        head_id=head.head_id,
        generation=head.generation,
        tick_seq=head.tick_seq,
        field_id=head.snapshot.field_id,
        core_state_leaf_ids=head.core_state_leaf_ids,
        soul_sha256_by_core=head.soul_sha256_by_core,
        model_binding_epoch_id=head.model_binding_epoch_id,
        last_commit_id=head.last_commit_id,
        journal_tip_event_id=head.journal_tip_event_id,
    )


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise V3JournalIntegrityError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise V3JournalIntegrityError(f"{label} must be a non-empty string")
    return value


def _require_optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, label)


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V3JournalIntegrityError(f"{label} must be an integer")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise V3JournalIntegrityError(f"{label} must be a boolean")
    return value


def _require_string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise V3JournalIntegrityError(f"{label} must be a list")
    return tuple(_require_string(item, f"{label} item") for item in value)


def _require_int_sequence(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise V3JournalIntegrityError(f"{label} must be a list")
    return tuple(_require_int(item, f"{label} item") for item in value)


class V3RuntimeStore:
    """Own one SQLite connection and its hash-chained v3 journal."""

    def __init__(self, state_root: str | Path) -> None:
        self.state_root = Path(state_root)
        self.path = self.state_root / V3_DATABASE_NAME
        v2_path = self.state_root / "runtime.sqlite3"
        if v2_path.exists():
            raise V3StoreError(
                f"refusing v3 root because v2 store exists: {v2_path}"
            )
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(self.path),
            isolation_level=None,
            timeout=30.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the owned connection for diagnostics and integrity tests."""
        return self._connection

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "V3RuntimeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS journal (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                previous_event_id TEXT,
                event_type TEXT NOT NULL,
                record_id TEXT NOT NULL,
                canonical_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runtime_head (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                generation INTEGER NOT NULL CHECK (generation >= 0),
                tick_seq INTEGER NOT NULL CHECK (tick_seq >= 0),
                snapshot_json TEXT NOT NULL,
                field_id TEXT NOT NULL,
                core_state_leaf_json TEXT NOT NULL,
                soul_sha256_json TEXT NOT NULL,
                model_binding_epoch_id TEXT NOT NULL,
                last_commit_id TEXT,
                journal_tip_event_id TEXT NOT NULL,
                head_id TEXT NOT NULL
            );

            CREATE TRIGGER IF NOT EXISTS journal_reject_update
            BEFORE UPDATE ON journal
            BEGIN
                SELECT RAISE(ABORT, 'journal is append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS journal_reject_delete
            BEFORE DELETE ON journal
            BEGIN
                SELECT RAISE(ABORT, 'journal is append-only');
            END;
            """
        )
        existing = self._connection.execute(
            "SELECT value FROM store_metadata WHERE key='schema'"
        ).fetchone()
        if existing is None:
            self._connection.execute(
                "INSERT INTO store_metadata(key, value) VALUES('schema', ?)",
                (V3_STORE_SCHEMA,),
            )
        elif existing["value"] != V3_STORE_SCHEMA:
            raise V3JournalIntegrityError("unknown v3 store schema")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _is_initialized(self) -> bool:
        row = self._connection.execute(
            """
            SELECT (
                EXISTS(SELECT 1 FROM runtime_head WHERE singleton=1)
                OR EXISTS(SELECT 1 FROM journal)
            ) AS initialized
            """
        ).fetchone()
        return bool(row["initialized"])

    def _require_initialized(self) -> None:
        if not self._is_initialized():
            raise V3NotInitializedError("v3 runtime store is not initialized")

    def _require_payload_size(self, text: str) -> None:
        if len(text.encode("utf-8")) > V3_MAX_JOURNAL_PAYLOAD_BYTES:
            raise V3StoreError("journal payload exceeds maximum size")

    def _append_journal(
        self,
        event_type: str,
        record_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        payload_text = canonical_json_text(dict(payload))
        self._require_payload_size(payload_text)
        previous_row = self._connection.execute(
            "SELECT event_id FROM journal ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        previous = None if previous_row is None else str(previous_row["event_id"])
        event_id = canonical_sha256(
            {
                "previous_event_id": previous,
                "event_type": event_type,
                "record_id": record_id,
                "payload": dict(payload),
            }
        )
        self._connection.execute(
            """
            INSERT INTO journal(
                event_id,
                previous_event_id,
                event_type,
                record_id,
                canonical_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, previous, event_type, record_id, payload_text),
        )
        return event_id

    def _parse_private_state(
        self, payload: Mapping[str, Any], record_id: str
    ) -> PrivateStateArtifactV3:
        item = dict(payload)
        _require_exact_keys(
            item,
            {
                "schema",
                "core_id",
                "tick_seq",
                "phase",
                "substep",
                "soul_sha256",
                "cursor_state_sha256",
                "rng_state_sha256",
                "model_binding_epoch_id",
                "parent_state_leaf_id",
                "working_field_id",
                "board_id",
                "delta_id",
                "state_leaf_id",
            },
            "private-state",
        )
        if item.pop("schema", None) != "axon-private-state-artifact-v3":
            raise V3JournalIntegrityError("private-state schema mismatch")
        if item.pop("state_leaf_id", None) != record_id:
            raise V3JournalIntegrityError("private-state record ID mismatch")
        artifact = PrivateStateArtifactV3(
            core_id=_require_string(item["core_id"], "private-state core_id"),
            tick_seq=_require_int(item["tick_seq"], "private-state tick_seq"),
            phase=_require_string(item["phase"], "private-state phase"),
            substep=_require_int(item["substep"], "private-state substep"),
            soul_sha256=_require_string(item["soul_sha256"], "private-state soul_sha256"),
            cursor_state_sha256=_require_string(
                item["cursor_state_sha256"], "private-state cursor_state_sha256"
            ),
            rng_state_sha256=_require_optional_string(
                item.get("rng_state_sha256"), "private-state rng_state_sha256"
            ),
            model_binding_epoch_id=_require_string(
                item["model_binding_epoch_id"], "private-state model_binding_epoch_id"
            ),
            parent_state_leaf_id=_require_optional_string(
                item.get("parent_state_leaf_id"), "private-state parent_state_leaf_id"
            ),
            working_field_id=_require_optional_string(
                item.get("working_field_id"), "private-state working_field_id"
            ),
            board_id=_require_optional_string(item.get("board_id"), "private-state board_id"),
            delta_id=_require_optional_string(item.get("delta_id"), "private-state delta_id"),
        )
        if artifact.state_leaf_id != record_id:
            raise V3JournalIntegrityError("private-state artifact ID mismatch")
        return artifact

    def _parse_active_projection(
        self, payload: Mapping[str, Any], record_id: str
    ) -> ActiveProjectionArtifactV3:
        item = dict(payload)
        _require_exact_keys(
            item,
            {
                "schema",
                "source_working_field_id",
                "selected_spans",
                "selected_char_count",
                "source_char_count",
                "projection_id",
            },
            "active-projection",
        )
        if item.pop("schema", None) != "axon-active-projection-artifact-v3":
            raise V3JournalIntegrityError("active-projection schema mismatch")
        if item.pop("projection_id", None) != record_id:
            raise V3JournalIntegrityError("active-projection record ID mismatch")
        spans_value = item["selected_spans"]
        if not isinstance(spans_value, list):
            raise V3JournalIntegrityError("active-projection selected_spans must be a list")
        spans = []
        for index, entry in enumerate(spans_value):
            if not isinstance(entry, dict):
                raise V3JournalIntegrityError(
                    f"active-projection span {index} must be an object"
                )
            _require_exact_keys(
                entry, {"region", "span_id", "text", "canonical_hash"}, f"span {index}"
            )
            spans.append(
                (
                    _require_string(entry["region"], f"span {index} region"),
                    _require_string(entry["span_id"], f"span {index} span_id"),
                    _require_string(entry["text"], f"span {index} text"),
                    _require_string(entry["canonical_hash"], f"span {index} canonical_hash"),
                )
            )
        artifact = ActiveProjectionArtifactV3(
            source_working_field_id=_require_string(
                item["source_working_field_id"], "active-projection source_working_field_id"
            ),
            selected_spans=tuple(spans),
            selected_char_count=_require_int(
                item["selected_char_count"], "active-projection selected_char_count"
            ),
            source_char_count=_require_int(
                item["source_char_count"], "active-projection source_char_count"
            ),
        )
        if artifact.projection_id != record_id:
            raise V3JournalIntegrityError("active-projection ID mismatch")
        return artifact

    def _parse_read_page(
        self, payload: Mapping[str, Any], record_id: str
    ) -> ReadPageArtifactV3:
        item = dict(payload)
        _require_exact_keys(
            item,
            {
                "schema",
                "core_id",
                "tick_seq",
                "phase",
                "working_field_id",
                "projection_id",
                "board_id",
                "page_index",
                "start_offset",
                "end_offset",
                "characters",
                "cursor_before_hash",
                "cursor_after_hash",
                "total_effective_character_count",
                "page_id",
            },
            "read-page",
        )
        if item.pop("schema", None) != "axon-read-page-artifact-v3":
            raise V3JournalIntegrityError("read-page schema mismatch")
        if item.pop("page_id", None) != record_id:
            raise V3JournalIntegrityError("read-page record ID mismatch")
        page = ReadPageArtifactV3(
            core_id=_require_string(item["core_id"], "read-page core_id"),
            tick_seq=_require_int(item["tick_seq"], "read-page tick_seq"),
            phase=_require_string(item["phase"], "read-page phase"),
            working_field_id=_require_string(
                item["working_field_id"], "read-page working_field_id"
            ),
            projection_id=_require_string(item["projection_id"], "read-page projection_id"),
            board_id=_require_optional_string(item.get("board_id"), "read-page board_id"),
            page_index=_require_int(item["page_index"], "read-page page_index"),
            start_offset=_require_int(item["start_offset"], "read-page start_offset"),
            end_offset=_require_int(item["end_offset"], "read-page end_offset"),
            characters=_require_string(item["characters"], "read-page characters", allow_empty=True),
            cursor_before_hash=_require_string(
                item["cursor_before_hash"], "read-page cursor_before_hash"
            ),
            cursor_after_hash=_require_string(
                item["cursor_after_hash"], "read-page cursor_after_hash"
            ),
            total_effective_character_count=_require_int(
                item["total_effective_character_count"],
                "read-page total_effective_character_count",
            ),
        )
        if page.page_id != record_id:
            raise V3JournalIntegrityError("read-page artifact ID mismatch")
        return page

    def _parse_field_transaction_audit(
        self, payload: Mapping[str, Any], record_id: str
    ) -> FieldTransactionAudit:
        item = dict(payload)
        _require_exact_keys(
            item,
            {
                "schema",
                "before_snapshot",
                "before_field_id",
                "system_update",
                "system_update_id",
                "working_snapshot",
                "working_field_id",
                "consolidator_delta",
                "consolidator_delta_id",
                "consolidator_author_core_id",
                "final_snapshot",
                "final_field_id",
                "audit_id",
                "canonical_hash",
            },
            "field-transaction-audit",
        )
        if item.pop("schema", None) != "axon-field-transaction-audit-v1":
            raise V3JournalIntegrityError("field-transaction-audit schema mismatch")
        if item.pop("audit_id", None) != record_id:
            raise V3JournalIntegrityError("field-transaction audit record ID mismatch")
        if item.pop("canonical_hash", None) != record_id:
            raise V3JournalIntegrityError("field-transaction audit canonical_hash mismatch")

        def _snapshot_from_canonical(
            canonical: Mapping[str, Any], field_id: str
        ) -> SharedFieldSnapshot:
            canonical_hash = canonical_sha256(dict(canonical))
            return deserialize_shared_field(
                {
                    **dict(canonical),
                    "field_id": field_id,
                    "canonical_hash": canonical_hash,
                }
            )

        before = _snapshot_from_canonical(
            item["before_snapshot"], item["before_field_id"]
        )
        working = _snapshot_from_canonical(
            item["working_snapshot"], item["working_field_id"]
        )
        final = _snapshot_from_canonical(
            item["final_snapshot"], item["final_field_id"]
        )
        update = deserialize_system_field_update(
            {
                **item["system_update"],
                "update_id": item["system_update_id"],
                "canonical_hash": item["system_update_id"],
            }
        )
        delta = None
        author = item["consolidator_author_core_id"]
        if item["consolidator_delta"] is not None:
            delta = deserialize_field_delta(
                {
                    **item["consolidator_delta"],
                    "delta_id": item["consolidator_delta_id"],
                    "canonical_hash": item["consolidator_delta_id"],
                }
            )
            if not author:
                raise V3JournalIntegrityError(
                    "audit delta present without author"
                )
        elif author:
            raise V3JournalIntegrityError("audit author present without delta")
        audit = FieldTransactionAudit(
            before_snapshot=before,
            system_update=update,
            working_snapshot=working,
            consolidator_delta=delta,
            consolidator_author_core_id=author,
            final_snapshot=final,
        )
        if audit.audit_id != record_id:
            raise V3JournalIntegrityError("field-transaction audit ID mismatch")
        return audit

    def _parse_journal_record(
        self,
        event_type: str,
        record_id: str,
        payload: Mapping[str, Any],
    ) -> Any:
        try:
            if event_type == "private_state":
                return self._parse_private_state(payload, record_id)
            if event_type == "genesis":
                item = dict(payload)
                _require_exact_keys(
                    item, {"schema", "snapshot", "private_state_ids"}, "genesis"
                )
                if item["schema"] != "axon-runtime-genesis-v3":
                    raise V3JournalIntegrityError("unknown genesis schema")
                snapshot = deserialize_shared_field(item["snapshot"])
                ids_value = item["private_state_ids"]
                if type(ids_value) is not list:
                    raise V3JournalIntegrityError(
                        "genesis private_state_ids must be a list"
                    )
                parsed_ids = []
                for index, raw_id in enumerate(ids_value):
                    parsed_ids.append(
                        _require_string(
                            raw_id, f"genesis private_state_ids item {index}"
                        )
                    )
                if not parsed_ids:
                    raise V3JournalIntegrityError("genesis references no private states")
                if len(parsed_ids) != len(set(parsed_ids)):
                    raise V3JournalIntegrityError(
                        "genesis private_state_ids contains duplicates"
                    )
                if parsed_ids != sorted(parsed_ids):
                    raise V3JournalIntegrityError(
                        "genesis private_state_ids must be in canonical sorted order"
                    )
                private_state_ids = tuple(parsed_ids)
                if snapshot.field_id != record_id:
                    raise V3JournalIntegrityError("genesis record ID mismatch")
                return {"snapshot": snapshot, "private_state_ids": private_state_ids}
            if event_type == "active_projection":
                return self._parse_active_projection(payload, record_id)
            if event_type == "read_page":
                return self._parse_read_page(payload, record_id)
            if event_type == "system_update":
                return deserialize_system_field_update(
                    {
                        **dict(payload),
                        "update_id": record_id,
                        "canonical_hash": record_id,
                    }
                )
            if event_type == "tick_plan":
                return deserialize_tick_phase_plan(payload)
            if event_type == "read_cycle":
                return deserialize_read_cycle_manifest(payload)
            if event_type == "field_delta":
                return deserialize_field_delta(
                    {
                        **dict(payload),
                        "delta_id": record_id,
                        "canonical_hash": record_id,
                    }
                )
            if event_type == "soul_transition":
                return deserialize_soul_transition(payload)
            if event_type == "core_pass":
                return deserialize_core_pass(payload)
            if event_type == "proposal_board":
                return deserialize_proposal_board(payload)
            if event_type == "consolidation":
                return deserialize_consolidation(payload)
            if event_type == "soul_disposition":
                return deserialize_soul_transition_disposition(payload)
            if event_type == "field_transaction_audit":
                return self._parse_field_transaction_audit(payload, record_id)
            if event_type == "tick_commit":
                return deserialize_tick_commit_v3(payload)
            if event_type == "artifact_rejection":
                return deserialize_artifact_rejection(payload)
        except V3JournalIntegrityError:
            raise
        except Exception as exc:
            raise V3JournalIntegrityError(
                f"malformed {event_type!r} journal record"
            ) from exc
        raise V3JournalIntegrityError(f"unknown journal event type {event_type!r}")

    def initialize(
        self,
        genesis_snapshot: SharedFieldSnapshot,
        initial_private_states: Sequence[PrivateStateArtifactV3],
        model_binding_epoch_id: str,
    ) -> V3RuntimeHead:
        if not isinstance(genesis_snapshot, SharedFieldSnapshot):
            raise TypeError("genesis_snapshot must be SharedFieldSnapshot")
        if genesis_snapshot.tick_id != 0 or genesis_snapshot.parent_field_id is not None:
            raise V3StoreError("genesis snapshot must have tick_id=0 and no parent")
        states = tuple(initial_private_states)
        if not states:
            raise V3StoreError("initial private states cannot be empty")
        state_by_core: dict[str, PrivateStateArtifactV3] = {}
        for state in states:
            if type(state) is not PrivateStateArtifactV3:
                raise TypeError(
                    "initial_private_states must contain exact PrivateStateArtifactV3 objects"
                )
            if state.core_id in state_by_core:
                raise V3StoreError("duplicate core_id in initial private states")
            if state.tick_seq != 0:
                raise V3StoreError("genesis private state must have tick_seq=0")
            if state.phase != "GENESIS":
                raise V3StoreError("genesis private state must have phase=GENESIS")
            if state.substep != 0:
                raise V3StoreError("genesis private state must have substep=0")
            if any(
                value is not None
                for value in (
                    state.parent_state_leaf_id,
                    state.working_field_id,
                    state.board_id,
                    state.delta_id,
                )
            ):
                raise V3StoreError(
                    "genesis private state must not carry transition context"
                )
            if state.model_binding_epoch_id != model_binding_epoch_id:
                raise V3StoreError("genesis private state binding mismatch")
            state_by_core[state.core_id] = state
        population = sorted(state_by_core)
        leaf_ids = {core_id: state_by_core[core_id].state_leaf_id for core_id in population}
        souls = {core_id: state_by_core[core_id].soul_sha256 for core_id in population}

        with self._transaction():
            if self._is_initialized():
                raise V3AlreadyInitializedError("v3 runtime store is already initialized")
            private_state_ids = sorted(
                state_by_core[core_id].state_leaf_id for core_id in population
            )
            for leaf_id in private_state_ids:
                state = next(
                    artifact
                    for artifact in state_by_core.values()
                    if artifact.state_leaf_id == leaf_id
                )
                self._append_journal(
                    "private_state",
                    state.state_leaf_id,
                    state.to_dict(),
                )
            genesis_event_id = self._append_journal(
                "genesis",
                genesis_snapshot.field_id,
                {
                    "schema": "axon-runtime-genesis-v3",
                    "snapshot": genesis_snapshot.to_dict(),
                    "private_state_ids": list(private_state_ids),
                },
            )
            head = _head_from_replay(
                generation=0,
                tick_seq=0,
                snapshot=genesis_snapshot,
                leaf_ids=leaf_ids,
                souls=souls,
                model_binding_epoch_id=model_binding_epoch_id,
                last_commit_id=None,
                journal_tip_event_id=genesis_event_id,
            )
            self._write_head(head)
        return head

    def _write_head(self, head: V3RuntimeHead) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_head(
                singleton,
                generation,
                tick_seq,
                snapshot_json,
                field_id,
                core_state_leaf_json,
                soul_sha256_json,
                model_binding_epoch_id,
                last_commit_id,
                journal_tip_event_id,
                head_id
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                generation=excluded.generation,
                tick_seq=excluded.tick_seq,
                snapshot_json=excluded.snapshot_json,
                field_id=excluded.field_id,
                core_state_leaf_json=excluded.core_state_leaf_json,
                soul_sha256_json=excluded.soul_sha256_json,
                model_binding_epoch_id=excluded.model_binding_epoch_id,
                last_commit_id=excluded.last_commit_id,
                journal_tip_event_id=excluded.journal_tip_event_id,
                head_id=excluded.head_id
            """,
            (
                head.generation,
                head.tick_seq,
                canonical_json_text(head.snapshot.to_dict()),
                head.snapshot.field_id,
                canonical_json_text(dict(head.core_state_leaf_ids)),
                canonical_json_text(dict(head.soul_sha256_by_core)),
                head.model_binding_epoch_id,
                head.last_commit_id,
                head.journal_tip_event_id,
                head.head_id,
            ),
        )

    def current_head(self) -> V3RuntimeHead:
        self._require_initialized()
        row = self._connection.execute(
            "SELECT * FROM runtime_head WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise V3NotInitializedError("v3 runtime head disappeared")
        snapshot = deserialize_shared_field(
            parse_json_object(row["snapshot_json"])
        )
        if snapshot.field_id != row["field_id"]:
            raise V3JournalIntegrityError("head snapshot field ID mismatch")
        leaf_ids = parse_json_object(row["core_state_leaf_json"])
        souls = parse_json_object(row["soul_sha256_json"])
        return _head_from_replay(
            generation=int(row["generation"]),
            tick_seq=int(row["tick_seq"]),
            snapshot=snapshot,
            leaf_ids=dict(leaf_ids),
            souls=dict(souls),
            model_binding_epoch_id=str(row["model_binding_epoch_id"]),
            last_commit_id=row["last_commit_id"],
            journal_tip_event_id=str(row["journal_tip_event_id"]),
        )

    def continuation_proof(self) -> V3ContinuationProof:
        return _continuation_proof_from_head(self.current_head())

    def _load_artifact_from_journal(
        self, event_type: str, record_id: str
    ) -> Any:
        row = self._connection.execute(
            """
            SELECT canonical_json FROM journal
            WHERE event_type=? AND record_id=?
            """,
            (event_type, record_id),
        ).fetchone()
        if row is None:
            raise V3JournalIntegrityError(
                f"missing {event_type} journal record {record_id}"
            )
        payload = parse_json_object(row["canonical_json"])
        return self._parse_journal_record(event_type, record_id, payload)

    def _verify_head_leaves_resolved(self, head: V3RuntimeHead) -> None:
        for core_id, leaf_id in head.core_state_leaf_ids:
            artifact = self._load_artifact_from_journal("private_state", leaf_id)
            if artifact.core_id != core_id:
                raise V3JournalIntegrityError(
                    f"head leaf {leaf_id} core_id mismatch"
                )
            if artifact.soul_sha256 != dict(head.soul_sha256_by_core)[core_id]:
                raise V3JournalIntegrityError(
                    f"head leaf {leaf_id} soul mismatch"
                )
            if artifact.model_binding_epoch_id != head.model_binding_epoch_id:
                raise V3JournalIntegrityError(
                    f"head leaf {leaf_id} binding mismatch"
                )

    def commit_tick(
        self,
        transaction: TickTransactionV3,
        expected_continuation: V3ContinuationProof,
    ) -> V3RuntimeHead:
        self._require_initialized()
        if not isinstance(transaction, TickTransactionV3):
            raise TypeError("transaction must be TickTransactionV3")
        if not isinstance(expected_continuation, V3ContinuationProof):
            raise TypeError("expected_continuation must be V3ContinuationProof")

        with self._transaction():
            current = self.current_head()
            if expected_continuation.proof_id != _continuation_proof_from_head(current).proof_id:
                raise V3StaleContinuationError("continuation proof does not match current head")

            # Population union must match the head.
            online = set(transaction.plan.online_core_ids)
            offline = {transaction.plan.offline_core_id}
            expected_population = online | offline
            head_population = {core_id for core_id, _ in current.core_state_leaf_ids}
            if expected_population != head_population:
                raise V3PopulationMismatchError(
                    "transaction population does not match head"
                )

            # Plan input field/state/soul/model binding must match current head.
            head_leaves = dict(current.core_state_leaf_ids)
            head_souls = dict(current.soul_sha256_by_core)
            input_by_core = {state.core_id: state for state in transaction.input_private_states}
            for core_id in online:
                input_state = input_by_core[core_id]
                if input_state.state_leaf_id != head_leaves[core_id]:
                    raise V3PopulationMismatchError(
                        f"input leaf for {core_id} does not match head"
                    )
                if input_state.soul_sha256 != head_souls[core_id]:
                    raise V3PopulationMismatchError(
                        f"input soul for {core_id} does not match head"
                    )
            offline_core_id = transaction.plan.offline_core_id
            if offline_core_id not in input_by_core:
                raise V3PopulationMismatchError("offline input missing")
            offline_input = input_by_core[offline_core_id]
            if offline_input.state_leaf_id != head_leaves[offline_core_id]:
                raise V3PopulationMismatchError("offline input leaf does not match head")
            if offline_input.soul_sha256 != head_souls[offline_core_id]:
                raise V3PopulationMismatchError("offline input soul does not match head")
            if offline_input.model_binding_epoch_id != current.model_binding_epoch_id:
                raise V3PopulationMismatchError("offline input binding mismatch")
            if transaction.plan.model_binding_epoch_id != current.model_binding_epoch_id:
                raise V3PopulationMismatchError("model binding mismatch")
            if transaction.plan.input_field_id != current.snapshot.field_id:
                raise V3PopulationMismatchError("input field ID mismatch")
            if transaction.plan.tick_seq != current.tick_seq:
                raise V3PopulationMismatchError("tick must advance exactly once")

            # Validate and replay the candidate transaction.
            replay = validate_tick_transaction_v3(transaction)
            if replay.final_snapshot.tick_id != current.tick_seq + 1:
                raise V3ReplayError("transaction did not advance tick by one")

            # Append artifacts in deterministic backward-reference order.
            self._append_journal(
                "system_update",
                transaction.system_update.update_id,
                transaction.system_update.to_dict(),
            )
            self._append_journal(
                "active_projection",
                transaction.projection.projection_id,
                transaction.projection.to_dict(),
            )
            self._append_journal(
                "tick_plan",
                transaction.plan.plan_id,
                transaction.plan.to_dict(),
            )

            delta_by_id = {delta.delta_id: delta for delta in transaction.deltas}
            transition_by_id = {
                transition.transition_id: transition
                for transition in transaction.soul_transitions
            }

            def _append_pass_group(
                passes: Sequence[CorePassRecord],
            ) -> None:
                for pass_record in passes:
                    transition = transition_by_id[pass_record.soul_transition_id]
                    cycle = next(
                        c
                        for c in transaction.read_cycles
                        if c.cycle_id == pass_record.read_cycle_id
                    )
                    cycle_pages = sorted(
                        [
                            page
                            for page in transaction.read_pages
                            if page.page_id in cycle.page_view_hashes
                        ],
                        key=lambda page: page.page_index,
                    )
                    for page in cycle_pages:
                        self._append_journal(
                            "read_page",
                            page.page_id,
                            page.to_dict(),
                        )
                    self._append_journal(
                        "read_cycle",
                        cycle.cycle_id,
                        cycle.to_dict(),
                    )
                    delta = delta_by_id[pass_record.delta_id]
                    self._append_journal(
                        "field_delta",
                        delta.delta_id,
                        serialize_field_delta(delta),
                    )
                    candidate = next(
                        artifact
                        for artifact in transaction.candidate_private_states
                        if artifact.state_leaf_id == pass_record.candidate_private_state_id
                    )
                    self._append_journal(
                        "private_state",
                        candidate.state_leaf_id,
                        candidate.to_dict(),
                    )
                    self._append_journal(
                        "soul_transition",
                        transition.transition_id,
                        transition.to_dict(),
                    )
                    self._append_journal(
                        "core_pass",
                        pass_record.pass_id,
                        pass_record.to_dict(),
                    )

            _append_pass_group(transaction.initial_passes)
            self._append_journal(
                "proposal_board",
                transaction.initial_board.board_id,
                transaction.initial_board.to_dict(),
            )
            _append_pass_group(transaction.refine_passes)
            self._append_journal(
                "proposal_board",
                transaction.refine_board.board_id,
                transaction.refine_board.to_dict(),
            )

            # FINAL pass artifacts.
            final_transition = next(
                t for t in transaction.soul_transitions if t.phase == "FINAL"
            )
            final_cycle = next(
                c for c in transaction.read_cycles if c.cycle_id == final_transition.read_cycle_id
            )
            final_pages = sorted(
                [
                    page
                    for page in transaction.read_pages
                    if page.page_id in final_cycle.page_view_hashes
                ],
                key=lambda page: page.page_index,
            )
            for page in final_pages:
                self._append_journal(
                    "read_page",
                    page.page_id,
                    page.to_dict(),
                )
            self._append_journal(
                "read_cycle",
                final_cycle.cycle_id,
                final_cycle.to_dict(),
            )
            final_delta = delta_by_id[transaction.consolidation.final_delta_id]
            self._append_journal(
                "field_delta",
                final_delta.delta_id,
                serialize_field_delta(final_delta),
            )
            final_candidate = next(
                artifact
                for artifact in transaction.candidate_private_states
                if artifact.state_leaf_id == final_transition.candidate_private_state_id
            )
            self._append_journal(
                "private_state",
                final_candidate.state_leaf_id,
                final_candidate.to_dict(),
            )
            self._append_journal(
                "soul_transition",
                final_transition.transition_id,
                final_transition.to_dict(),
            )

            self._append_journal(
                "consolidation",
                transaction.consolidation.consolidation_id,
                transaction.consolidation.to_dict(),
            )
            for disposition in sorted(
                transaction.dispositions,
                key=lambda disp: disp.transition_id,
            ):
                self._append_journal(
                    "soul_disposition",
                    disposition.disposition_id,
                    disposition.to_dict(),
                )
            self._append_journal(
                "field_transaction_audit",
                transaction.field_audit.audit_id,
                transaction.field_audit.to_dict(),
            )
            commit_event_id = self._append_journal(
                "tick_commit",
                transaction.commit.commit_id,
                transaction.commit.to_dict(),
            )

            # Compute next head.
            final_leaves = dict(transaction.commit.final_core_state_leaf_ids)
            next_leaf_ids = dict(final_leaves)
            next_leaf_ids[offline_core_id] = head_leaves[offline_core_id]
            next_souls = dict(replay.final_soul_sha256_by_core)
            next_souls[offline_core_id] = head_souls[offline_core_id]

            next_head = _head_from_replay(
                generation=current.generation + 1,
                tick_seq=current.tick_seq + 1,
                snapshot=replay.final_snapshot,
                leaf_ids=next_leaf_ids,
                souls=next_souls,
                model_binding_epoch_id=transaction.plan.model_binding_epoch_id,
                last_commit_id=transaction.commit.commit_id,
                journal_tip_event_id=commit_event_id,
            )
            self._write_head(next_head)
        return next_head

    def append_artifact_rejection(
        self,
        rejection: ArtifactRejectionRecord,
        expected_continuation: V3ContinuationProof,
    ) -> V3RuntimeHead:
        self._require_initialized()
        if not isinstance(rejection, ArtifactRejectionRecord):
            raise TypeError("rejection must be ArtifactRejectionRecord")
        if not isinstance(expected_continuation, V3ContinuationProof):
            raise TypeError("expected_continuation must be V3ContinuationProof")
        with self._transaction():
            current = self.current_head()
            if expected_continuation.proof_id != _continuation_proof_from_head(current).proof_id:
                raise V3StaleContinuationError("continuation proof does not match current head")
            event_id = self._append_journal(
                "artifact_rejection",
                rejection.rejection_id,
                rejection.to_dict(),
            )
            next_head = _head_from_replay(
                generation=current.generation,
                tick_seq=current.tick_seq,
                snapshot=current.snapshot,
                leaf_ids=dict(current.core_state_leaf_ids),
                souls=dict(current.soul_sha256_by_core),
                model_binding_epoch_id=current.model_binding_epoch_id,
                last_commit_id=current.last_commit_id,
                journal_tip_event_id=event_id,
            )
            self._write_head(next_head)
        return next_head

    def verify_journal_chain(self) -> str | None:
        previous: str | None = None
        rows = self._connection.execute(
            """
            SELECT
                seq,
                event_id,
                previous_event_id,
                event_type,
                record_id,
                canonical_json
            FROM journal
            ORDER BY seq
            """
        ).fetchall()
        for row in rows:
            if row["previous_event_id"] != previous:
                raise V3JournalIntegrityError(
                    f"journal parent mismatch at sequence {row['seq']}"
                )
            canonical_json = row["canonical_json"]
            self._require_payload_size(canonical_json)
            try:
                payload = parse_json_object(canonical_json)
            except Exception as exc:
                raise V3JournalIntegrityError(
                    f"journal payload is invalid at sequence {row['seq']}"
                ) from exc
            if canonical_json_text(payload) != canonical_json:
                raise V3JournalIntegrityError(
                    f"journal payload is not canonical at sequence {row['seq']}"
                )
            expected = canonical_sha256(
                {
                    "previous_event_id": previous,
                    "event_type": row["event_type"],
                    "record_id": row["record_id"],
                    "payload": payload,
                }
            )
            if row["event_id"] != expected:
                raise V3JournalIntegrityError(
                    f"journal event hash mismatch at sequence {row['seq']}"
                )
            previous = str(row["event_id"])
        return previous

    def _resolve_artifact(
        self,
        event_type: str,
        record_id: str,
        record_sequences: Mapping[tuple[str, str], int],
        commit_seq: int,
        allow_input: bool = False,
    ) -> Any:
        key = (event_type, record_id)
        if key not in record_sequences:
            raise V3JournalIntegrityError(f"missing {event_type} record {record_id}")
        if record_sequences[key] >= commit_seq:
            raise V3JournalIntegrityError(
                f"{event_type} {record_id} must precede the commit"
            )
        artifact = self._load_artifact_from_journal(event_type, record_id)
        return artifact

    def _reconstruct_transaction_at_commit(
        self,
        commit_seq: int,
        commit: TickCommitRecordV3,
        record_sequences: Mapping[tuple[str, str], int],
        input_private_states: Mapping[str, PrivateStateArtifactV3],
    ) -> tuple[TickTransactionV3, set[tuple[str, str]]]:
        consumed_keys: set[tuple[str, str]] = set()

        def _consume(event_type: str, record_id: str) -> None:
            key = (event_type, record_id)
            if key in consumed_keys:
                raise V3JournalIntegrityError(
                    f"transaction consumes {event_type} {record_id} more than once"
                )
            consumed_keys.add(key)

        # Plan and its direct dependencies.
        plan = self._resolve_artifact(
            "tick_plan", commit.plan_id, record_sequences, commit_seq
        )
        _consume("tick_plan", commit.plan_id)
        audit = self._resolve_artifact(
            "field_transaction_audit",
            commit.field_transaction_audit_id,
            record_sequences,
            commit_seq,
        )
        _consume("field_transaction_audit", commit.field_transaction_audit_id)
        consolidation = self._resolve_artifact(
            "consolidation", commit.consolidation_id, record_sequences, commit_seq
        )
        _consume("consolidation", commit.consolidation_id)
        initial_board = self._resolve_artifact(
            "proposal_board", commit.initial_board_id, record_sequences, commit_seq
        )
        _consume("proposal_board", commit.initial_board_id)
        refine_board = self._resolve_artifact(
            "proposal_board", commit.refine_board_id, record_sequences, commit_seq
        )
        _consume("proposal_board", commit.refine_board_id)
        system_update = self._resolve_artifact(
            "system_update", plan.system_update_id, record_sequences, commit_seq
        )
        _consume("system_update", plan.system_update_id)
        projection = self._resolve_artifact(
            "active_projection", plan.projection_id, record_sequences, commit_seq
        )
        _consume("active_projection", plan.projection_id)

        # Passes and their dependent artifacts.
        initial_passes: list[CorePassRecord] = []
        refine_passes: list[CorePassRecord] = []
        pass_records_by_id: dict[str, CorePassRecord] = {}
        for pass_id in commit.initial_pass_ids:
            pass_record = self._resolve_artifact(
                "core_pass", pass_id[1], record_sequences, commit_seq
            )
            _consume("core_pass", pass_id[1])
            initial_passes.append(pass_record)
            pass_records_by_id[pass_record.pass_id] = pass_record
        for pass_id in commit.refine_pass_ids:
            pass_record = self._resolve_artifact(
                "core_pass", pass_id[1], record_sequences, commit_seq
            )
            _consume("core_pass", pass_id[1])
            refine_passes.append(pass_record)
            pass_records_by_id[pass_record.pass_id] = pass_record

        all_passes = initial_passes + refine_passes

        # Transitions referenced by passes and consolidation.
        transition_ids: set[str] = set()
        for pass_record in all_passes:
            transition_ids.add(pass_record.soul_transition_id)
        transition_ids.add(consolidation.final_soul_transition_id)
        soul_transitions: list[SoulTransitionRecord] = []
        for transition_id in transition_ids:
            transition = self._resolve_artifact(
                "soul_transition", transition_id, record_sequences, commit_seq
            )
            _consume("soul_transition", transition_id)
            soul_transitions.append(transition)

        # Deltas referenced by passes and consolidation.
        delta_ids: set[str] = set()
        for pass_record in all_passes:
            delta_ids.add(pass_record.delta_id)
        delta_ids.add(consolidation.final_delta_id)
        deltas: list[Any] = []
        for delta_id in delta_ids:
            delta = self._resolve_artifact(
                "field_delta", delta_id, record_sequences, commit_seq
            )
            _consume("field_delta", delta_id)
            deltas.append(delta)

        # Read cycles referenced by transitions.
        cycle_ids: set[str] = set()
        for transition in soul_transitions:
            cycle_ids.add(transition.read_cycle_id)
        read_cycles: list[ReadCycleManifest] = []
        for cycle_id in cycle_ids:
            cycle = self._resolve_artifact(
                "read_cycle", cycle_id, record_sequences, commit_seq
            )
            _consume("read_cycle", cycle_id)
            read_cycles.append(cycle)

        # Read pages referenced by cycles.
        page_ids: set[str] = set()
        for cycle in read_cycles:
            page_ids.update(cycle.page_view_hashes)
        read_pages: list[ReadPageArtifactV3] = []
        for page_id in page_ids:
            page = self._resolve_artifact(
                "read_page", page_id, record_sequences, commit_seq
            )
            _consume("read_page", page_id)
            read_pages.append(page)

        # Candidate private states referenced by transitions.
        candidate_ids: set[str] = set()
        for transition in soul_transitions:
            candidate_ids.add(transition.candidate_private_state_id)
        candidate_private_states: list[PrivateStateArtifactV3] = []
        for candidate_id in candidate_ids:
            candidate = self._resolve_artifact(
                "private_state", candidate_id, record_sequences, commit_seq
            )
            _consume("private_state", candidate_id)
            candidate_private_states.append(candidate)

        # Dispositions referenced by commit.
        disposition_ids: set[str] = set()
        for _, disposition_id in commit.disposition_ids_by_transition:
            disposition_ids.add(disposition_id)
        dispositions: list[SoulTransitionDispositionRecord] = []
        for disposition_id in disposition_ids:
            disposition = self._resolve_artifact(
                "soul_disposition", disposition_id, record_sequences, commit_seq
            )
            _consume("soul_disposition", disposition_id)
            dispositions.append(disposition)

        online = tuple(plan.online_core_ids)
        offline = plan.offline_core_id
        ordered_input_states = tuple(
            input_private_states[core_id]
            for core_id in sorted((*online, offline))
        )

        transaction = TickTransactionV3(
            system_update=system_update,
            projection=projection,
            plan=plan,
            input_private_states=ordered_input_states,
            candidate_private_states=tuple(
                sorted(candidate_private_states, key=lambda c: (c.core_id, c.state_leaf_id))
            ),
            read_pages=tuple(sorted(read_pages, key=lambda p: p.page_id)),
            read_cycles=tuple(sorted(read_cycles, key=lambda c: c.cycle_id)),
            deltas=tuple(sorted(deltas, key=lambda d: d.delta_id)),
            soul_transitions=tuple(sorted(soul_transitions, key=lambda t: t.transition_id)),
            initial_passes=tuple(sorted(initial_passes, key=lambda p: p.pass_id)),
            initial_board=initial_board,
            refine_passes=tuple(sorted(refine_passes, key=lambda p: p.pass_id)),
            refine_board=refine_board,
            consolidation=consolidation,
            dispositions=tuple(sorted(dispositions, key=lambda d: d.disposition_id)),
            field_audit=audit,
            commit=commit,
        )
        return transaction, consumed_keys

    def replay_journal(self) -> V3RuntimeHead:
        self._require_initialized()
        self.verify_journal_chain()
        rows = self._connection.execute(
            """
            SELECT seq, event_id, event_type, record_id, canonical_json
            FROM journal
            ORDER BY seq
            """
        ).fetchall()
        parsed: list[tuple[int, str, str, Any]] = []
        for row in rows:
            payload = parse_json_object(row["canonical_json"])
            parsed.append(
                (
                    int(row["seq"]),
                    str(row["event_type"]),
                    str(row["record_id"]),
                    self._parse_journal_record(
                        str(row["event_type"]),
                        str(row["record_id"]),
                        payload,
                    ),
                )
            )

        record_sequences: dict[tuple[str, str], int] = {}
        for seq, event_type, record_id, _ in parsed:
            key = (event_type, record_id)
            if key in record_sequences:
                raise V3JournalIntegrityError("duplicate event-type/record identity")
            record_sequences[key] = seq

        # Validate genesis prelude: one private_state event per complete
        # population member, followed by the single genesis event.
        genesis_entries = [
            (seq, value)
            for seq, event_type, _, value in parsed
            if event_type == "genesis"
        ]
        if len(genesis_entries) != 1:
            raise V3JournalIntegrityError("journal must contain exactly one genesis")
        genesis_seq, genesis_value = genesis_entries[0]
        for seq, event_type, _, _ in parsed:
            if seq < genesis_seq and event_type != "private_state":
                raise V3JournalIntegrityError(
                    "only private_state events may precede genesis"
                )
            if seq >= genesis_seq:
                break
        genesis_snapshot: SharedFieldSnapshot = genesis_value["snapshot"]
        if genesis_snapshot.tick_id != 0 or genesis_snapshot.parent_field_id is not None:
            raise V3JournalIntegrityError("genesis snapshot is not a root")
        expected_prelude = tuple(genesis_value["private_state_ids"])
        prelude_entries = [
            (seq, record_id)
            for seq, event_type, record_id, _ in parsed
            if event_type == "private_state" and seq < genesis_seq
        ]
        prelude_ids = tuple(record_id for _, record_id in prelude_entries)
        if prelude_ids != expected_prelude:
            raise V3JournalIntegrityError("genesis prelude mismatch")

        private_states_by_id: dict[str, PrivateStateArtifactV3] = {}
        for _, event_type, record_id, value in parsed:
            if event_type == "private_state":
                if record_id in private_states_by_id:
                    raise V3JournalIntegrityError("duplicate private-state record")
                private_states_by_id[record_id] = value

        def _state_by_id(record_id: str) -> PrivateStateArtifactV3:
            if record_id not in private_states_by_id:
                raise V3JournalIntegrityError(f"missing private-state {record_id}")
            return private_states_by_id[record_id]

        initial_leaf_ids: dict[str, str] = {}
        initial_souls: dict[str, str] = {}
        prelude_bindings: set[str] = set()
        for record_id in expected_prelude:
            state = _state_by_id(record_id)
            if state.tick_seq != 0:
                raise V3JournalIntegrityError(
                    f"genesis private state {record_id} tick_seq must be 0"
                )
            if state.phase != "GENESIS":
                raise V3JournalIntegrityError(
                    f"genesis private state {record_id} phase must be GENESIS"
                )
            if state.substep != 0:
                raise V3JournalIntegrityError(
                    f"genesis private state {record_id} substep must be 0"
                )
            if any(
                value is not None
                for value in (
                    state.parent_state_leaf_id,
                    state.working_field_id,
                    state.board_id,
                    state.delta_id,
                )
            ):
                raise V3JournalIntegrityError(
                    f"genesis private state {record_id} carries transition context"
                )
            if state.core_id in initial_leaf_ids:
                raise V3JournalIntegrityError(
                    f"duplicate genesis core_id {state.core_id}"
                )
            initial_leaf_ids[state.core_id] = record_id
            initial_souls[state.core_id] = state.soul_sha256
            prelude_bindings.add(state.model_binding_epoch_id)
        if len(prelude_bindings) != 1:
            raise V3JournalIntegrityError(
                "genesis private states must carry one exact binding"
            )
        model_binding_epoch_id = prelude_bindings.pop()

        current_snapshot = genesis_snapshot
        current_generation = 0
        current_tick_seq = 0
        current_leaf_ids = dict(initial_leaf_ids)
        current_souls = dict(initial_souls)

        tick_commits = [
            (seq, record_id, value)
            for seq, event_type, record_id, value in parsed
            if event_type == "tick_commit"
        ]

        previous_commit_seq: int | None = None
        for commit_seq, commit_id, commit in tick_commits:
            frame_lower_bound = (
                previous_commit_seq if previous_commit_seq is not None else genesis_seq
            )

            input_private_states = {
                core_id: _state_by_id(leaf_id)
                for core_id, leaf_id in current_leaf_ids.items()
            }
            transaction, consumed_keys = self._reconstruct_transaction_at_commit(
                commit_seq=commit_seq,
                commit=commit,
                record_sequences=record_sequences,
                input_private_states=input_private_states,
            )

            # Pre-validate the reconstructed frame against the replayed snapshot.
            plan = transaction.plan
            if plan.input_field_id != current_snapshot.field_id:
                raise V3JournalIntegrityError(
                    "plan input field does not match replayed snapshot"
                )
            if plan.tick_seq != current_tick_seq:
                raise V3JournalIntegrityError(
                    "plan tick does not match replayed tick"
                )
            if plan.model_binding_epoch_id != model_binding_epoch_id:
                raise V3JournalIntegrityError(
                    "plan model binding does not match journal-derived binding"
                )
            expected_input_population = set(plan.online_core_ids) | {
                plan.offline_core_id
            }
            if set(input_private_states) != expected_input_population:
                raise V3JournalIntegrityError(
                    "reconstructed input population does not match head population"
                )
            for core_id in expected_input_population:
                state = input_private_states[core_id]
                if state.state_leaf_id != current_leaf_ids[core_id]:
                    raise V3JournalIntegrityError(
                        f"reconstructed input leaf for {core_id} does not match head"
                    )
                if state.soul_sha256 != current_souls[core_id]:
                    raise V3JournalIntegrityError(
                        f"reconstructed input soul for {core_id} does not match head"
                    )

            # Same-frame reachability: consumed artifacts must live inside the frame.
            for key in consumed_keys:
                seq = record_sequences[key]
                if not (frame_lower_bound < seq < commit_seq):
                    raise V3JournalIntegrityError(
                        f"{key[0]} {key[1]} is outside the current tick frame"
                    )

            # Orphan rejection: every non-rejection semantic event in the frame
            # must be consumed by the transaction or be the commit itself.
            expected_frame_events = consumed_keys | {("tick_commit", commit_id)}
            actual_frame_events = {
                (event_type, record_id)
                for seq, event_type, record_id, _ in parsed
                if frame_lower_bound < seq <= commit_seq
                and event_type != "artifact_rejection"
            }
            if actual_frame_events != expected_frame_events:
                raise V3JournalIntegrityError(
                    "tick frame contains orphan or missing semantic events"
                )

            try:
                replay = validate_tick_transaction_v3(transaction)
            except V3TransactionError as exc:
                raise V3JournalIntegrityError(
                    f"journal transaction replay failed at commit {commit_id}"
                ) from exc

            if replay.final_snapshot.tick_id != current_tick_seq + 1:
                raise V3JournalIntegrityError("replayed tick did not advance by one")
            if replay.commit_id != commit_id:
                raise V3JournalIntegrityError("replayed commit ID mismatch")

            # Determine next leaves/souls.
            next_leaf_ids = dict(current_leaf_ids)
            next_souls = dict(current_souls)
            for core_id, leaf_id in commit.final_core_state_leaf_ids:
                artifact = _state_by_id(leaf_id)
                if artifact.core_id != core_id:
                    raise V3JournalIntegrityError(
                        f"commit leaf {leaf_id} core mismatch"
                    )
                if artifact.model_binding_epoch_id != model_binding_epoch_id:
                    raise V3JournalIntegrityError(
                        f"commit leaf {leaf_id} binding mismatch"
                    )
                next_leaf_ids[core_id] = leaf_id
                next_souls[core_id] = artifact.soul_sha256

            current_snapshot = replay.final_snapshot
            current_generation += 1
            current_tick_seq += 1
            current_leaf_ids = next_leaf_ids
            current_souls = next_souls
            previous_commit_seq = commit_seq

        # After the final commit (or genesis when there are no commits), reject
        # every trailing non-rejection semantic event. artifact_rejection events
        # remain the only legal semantically inert tail.
        last_allowed_seq = (
            previous_commit_seq if previous_commit_seq is not None else genesis_seq
        )
        for seq, event_type, record_id, _ in parsed:
            if seq > last_allowed_seq and event_type != "artifact_rejection":
                raise V3JournalIntegrityError(
                    f"trailing non-rejection semantic event after final commit: "
                    f"{event_type} {record_id}"
                )

        # The journal tip event is the last event.
        journal_tip_event_id = str(rows[-1]["event_id"]) if rows else ""
        last_commit_id = tick_commits[-1][1] if tick_commits else None
        return _head_from_replay(
            generation=current_generation,
            tick_seq=current_tick_seq,
            snapshot=current_snapshot,
            leaf_ids=current_leaf_ids,
            souls=current_souls,
            model_binding_epoch_id=model_binding_epoch_id,
            last_commit_id=last_commit_id,
            journal_tip_event_id=journal_tip_event_id,
        )

    def recover(self) -> V3RuntimeHead:
        self._require_initialized()
        try:
            truth = self.replay_journal()
            projected = self.current_head()
        except V3JournalIntegrityError:
            raise
        except Exception as exc:
            raise V3JournalIntegrityError("mutable runtime projections are invalid") from exc
        if projected != truth:
            raise V3JournalIntegrityError("mutable runtime projections drift from journal truth")
        return projected


__all__ = [
    "V3_STORE_SCHEMA",
    "V3_DATABASE_NAME",
    "V3_MAX_JOURNAL_PAYLOAD_BYTES",
    "V3StoreError",
    "V3AlreadyInitializedError",
    "V3NotInitializedError",
    "V3StaleContinuationError",
    "V3JournalIntegrityError",
    "V3RuntimeHead",
    "V3ContinuationProof",
    "V3RuntimeStore",
]

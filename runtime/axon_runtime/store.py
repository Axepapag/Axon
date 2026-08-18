"""SQLite journal and mutable projections for the additive Axon runtime."""
from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from runtime.field import SharedFieldSnapshot, canonical_sha256, validate_delta

from .contracts import (
    AlreadyInitializedError,
    ConsolidationDecision,
    CoreIdentity,
    CoreStateManifest,
    CoreStateUpdateError,
    IdempotencyConflictError,
    JournalIntegrityError,
    NotInitializedError,
    ProjectionManifest,
    ProposalRecord,
    RoleAssignment,
    RoleOwnershipError,
    RuntimeHead,
    SourceEvent,
    SpanLifecycleEvent,
    StaleHeadError,
    StoreContractError,
    TickCommitRecord,
    ToolRequestRecord,
    ToolResultRecord,
    validate_identity_population,
)
from .field_transaction import (
    FieldTransactionAudit,
    SystemFieldUpdate,
    apply_system_update,
    compose_tick_transaction,
)
from .roles import RoleScheduler
from .serde import (
    canonical_json_text,
    deserialize_core_identity,
    deserialize_core_state_manifest,
    deserialize_field_delta,
    deserialize_projection_manifest,
    deserialize_shared_field,
    deserialize_system_field_update,
    parse_json_object,
    serialize_shared_field,
)

if TYPE_CHECKING:
    from .ingress import IngressQueue


_STORE_SCHEMA = "axon-runtime-store-v1"


class RuntimeStore:
    """Own one SQLite connection and its hash-chained runtime journal."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(
            self.path,
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

    def __enter__(self) -> "RuntimeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runtime_head (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                generation INTEGER NOT NULL CHECK (generation >= 0),
                tick_seq INTEGER NOT NULL CHECK (tick_seq >= 0),
                snapshot_json TEXT NOT NULL,
                field_id TEXT NOT NULL,
                role_index INTEGER NOT NULL CHECK (role_index >= 0),
                role_assignment_hash TEXT NOT NULL,
                projection_manifest_id TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS core_identities (
                core_id TEXT PRIMARY KEY,
                ring_index INTEGER NOT NULL UNIQUE CHECK (ring_index >= 0),
                enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                identity_hash TEXT NOT NULL UNIQUE,
                identity_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS core_states (
                core_id TEXT PRIMARY KEY
                    REFERENCES core_identities(core_id),
                manifest_id TEXT NOT NULL UNIQUE,
                manifest_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS journal (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                previous_event_id TEXT,
                event_type TEXT NOT NULL,
                record_id TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                idempotency_key TEXT
            );

            CREATE UNIQUE INDEX IF NOT EXISTS journal_idempotency
                ON journal(event_type, idempotency_key)
                WHERE idempotency_key IS NOT NULL;

            CREATE UNIQUE INDEX IF NOT EXISTS journal_record_identity
                ON journal(event_type, record_id);

            CREATE TABLE IF NOT EXISTS tool_requests (
                request_id TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                status TEXT NOT NULL,
                result_id TEXT,
                result_json TEXT
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
                (_STORE_SCHEMA,),
            )
        elif existing["value"] != _STORE_SCHEMA:
            raise JournalIntegrityError("unknown runtime store schema")

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
            raise NotInitializedError("runtime store is not initialized")

    def _append_journal(
        self,
        event_type: str,
        record_id: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> tuple[str, bool]:
        payload_text = canonical_json_text(dict(payload))
        if idempotency_key is not None:
            existing = self._connection.execute(
                """
                SELECT event_id, record_id, canonical_json
                FROM journal
                WHERE event_type=? AND idempotency_key=?
                """,
                (event_type, idempotency_key),
            ).fetchone()
            if existing is not None:
                if (
                    existing["record_id"] != record_id
                    or existing["canonical_json"] != payload_text
                ):
                    raise IdempotencyConflictError(
                        f"idempotency conflict for {event_type!r} "
                        f"key {idempotency_key!r}"
                    )
                return str(existing["event_id"]), False
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
                canonical_json,
                idempotency_key
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                previous,
                event_type,
                record_id,
                payload_text,
                idempotency_key,
            ),
        )
        return event_id, True

    @staticmethod
    def _validate_manifest_identity(
        manifest: CoreStateManifest,
        identity: CoreIdentity,
    ) -> None:
        if manifest.core_id != identity.core_id:
            raise CoreStateUpdateError("core-state core_id does not match identity")
        if manifest.soul_id != identity.soul_id:
            raise CoreStateUpdateError("core-state soul_id does not match identity")
        if manifest.model_id != identity.model_id:
            raise CoreStateUpdateError("core-state model_id does not match identity")
        if manifest.core_state_sha256 != identity.core_state_sha256:
            raise CoreStateUpdateError(
                "core-state model hash does not match immutable identity"
            )

    def initialize(
        self,
        genesis_snapshot: SharedFieldSnapshot,
        identities: Sequence[CoreIdentity],
        initial_core_states: Sequence[CoreStateManifest],
    ) -> RuntimeHead:
        if not isinstance(genesis_snapshot, SharedFieldSnapshot):
            raise TypeError("genesis_snapshot must be SharedFieldSnapshot")
        if (
            genesis_snapshot.tick_id != 0
            or genesis_snapshot.parent_field_id is not None
        ):
            raise StoreContractError(
                "genesis snapshot must have tick_id=0 and no parent"
            )
        identity_values = tuple(identities)
        state_values = tuple(initial_core_states)
        validate_identity_population(identity_values)
        scheduler = RoleScheduler(identity_values)
        identity_by_id = {identity.core_id: identity for identity in identity_values}
        state_by_id = {state.core_id: state for state in state_values}
        if len(state_by_id) != len(state_values):
            raise CoreStateUpdateError("duplicate initial core-state manifest")
        if set(state_by_id) != set(identity_by_id):
            raise CoreStateUpdateError(
                "initial core-state manifests must exactly cover identities"
            )
        for core_id, state in state_by_id.items():
            self._validate_manifest_identity(state, identity_by_id[core_id])
            if (
                state.generation != 0
                or state.tick_seq != 0
                or state.committed_field_id != genesis_snapshot.field_id
                or state.parent_manifest_id is not None
            ):
                raise CoreStateUpdateError(
                    "initial core-state manifest is not at genesis"
                )
        assignment = scheduler.assignment(0)
        projection = ProjectionManifest(
            generation=0,
            tick_seq=0,
            field_id=genesis_snapshot.field_id,
            field_hash=genesis_snapshot.canonical_hash,
            core_state_manifest_ids=tuple(
                (core_id, state.manifest_id)
                for core_id, state in state_by_id.items()
            ),
            role_index=0,
            role_assignment_hash=assignment.assignment_hash,
        )
        with self._transaction():
            if self._is_initialized():
                raise AlreadyInitializedError("runtime store is already initialized")
            self._append_journal(
                "genesis",
                genesis_snapshot.field_id,
                serialize_shared_field(genesis_snapshot),
            )
            for ring_index, identity in enumerate(identity_values):
                self._connection.execute(
                    """
                    INSERT INTO core_identities(
                        core_id,
                        ring_index,
                        enabled,
                        identity_hash,
                        identity_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        identity.core_id,
                        ring_index,
                        int(identity.enabled),
                        identity.identity_hash,
                        canonical_json_text(identity.to_dict()),
                    ),
                )
                self._append_journal(
                    "core_identity",
                    identity.identity_hash,
                    identity.to_canonical_dict(),
                )
            for state in state_values:
                self._connection.execute(
                    """
                    INSERT INTO core_states(core_id, manifest_id, manifest_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        state.core_id,
                        state.manifest_id,
                        canonical_json_text(state.to_dict()),
                    ),
                )
                self._append_journal(
                    "core_state",
                    state.manifest_id,
                    state.to_canonical_dict(),
                )
            self._connection.execute(
                """
                INSERT INTO runtime_head(
                    singleton,
                    generation,
                    tick_seq,
                    snapshot_json,
                    field_id,
                    role_index,
                    role_assignment_hash,
                    projection_manifest_id
                ) VALUES (1, 0, 0, ?, ?, 0, ?, ?)
                """,
                (
                    canonical_json_text(serialize_shared_field(genesis_snapshot)),
                    genesis_snapshot.field_id,
                    assignment.assignment_hash,
                    projection.projection_id,
                ),
            )
            self._append_journal(
                "projection",
                projection.projection_id,
                projection.to_canonical_dict(),
            )
        return self.current_head()

    def _load_identities(self) -> tuple[CoreIdentity, ...]:
        rows = self._connection.execute(
            "SELECT identity_json FROM core_identities ORDER BY ring_index"
        ).fetchall()
        identities = tuple(
            deserialize_core_identity(parse_json_object(row["identity_json"]))
            for row in rows
        )
        validate_identity_population(identities)
        return identities

    def persisted_identities(self) -> tuple[CoreIdentity, ...]:
        """Return the immutable identity ring in its persisted schedule order."""

        self._require_initialized()
        return self._load_identities()

    def _load_core_states(self) -> tuple[CoreStateManifest, ...]:
        rows = self._connection.execute(
            """
            SELECT s.core_id, s.manifest_id, s.manifest_json
            FROM core_states AS s
            JOIN core_identities AS i ON i.core_id=s.core_id
            ORDER BY i.ring_index
            """
        ).fetchall()
        manifests: list[CoreStateManifest] = []
        for row in rows:
            manifest = deserialize_core_state_manifest(
                parse_json_object(row["manifest_json"])
            )
            if (
                row["core_id"] != manifest.core_id
                or row["manifest_id"] != manifest.manifest_id
            ):
                raise JournalIntegrityError(
                    "core-state projection columns conflict with manifest"
                )
            manifests.append(manifest)
        return tuple(manifests)

    def current_head(self) -> RuntimeHead:
        self._require_initialized()
        row = self._connection.execute(
            "SELECT * FROM runtime_head WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise NotInitializedError("runtime head disappeared")
        snapshot = deserialize_shared_field(
            parse_json_object(row["snapshot_json"])
        )
        if snapshot.field_id != row["field_id"]:
            raise JournalIntegrityError("head field projection hash mismatch")
        return RuntimeHead(
            generation=int(row["generation"]),
            tick_seq=int(row["tick_seq"]),
            snapshot=snapshot,
            role_index=int(row["role_index"]),
            role_assignment_hash=str(row["role_assignment_hash"]),
            projection_manifest_id=str(row["projection_manifest_id"]),
            core_state_manifests=self._load_core_states(),
        )

    def append_source_event(self, event: SourceEvent) -> str:
        if not isinstance(event, SourceEvent):
            raise TypeError("event must be SourceEvent")
        self._require_initialized()
        with self._transaction():
            event_id, _ = self._append_journal(
                "source_event",
                event.source_event_id,
                event.to_canonical_dict(),
                idempotency_key=event.idempotency_key,
            )
        return event_id

    def append_span_lifecycle(self, event: SpanLifecycleEvent) -> str:
        if not isinstance(event, SpanLifecycleEvent):
            raise TypeError("event must be SpanLifecycleEvent")
        self._require_initialized()
        with self._transaction():
            event_id, _ = self._append_journal(
                "span_lifecycle",
                event.lifecycle_event_id,
                event.to_canonical_dict(),
                idempotency_key=event.lifecycle_event_id,
            )
        return event_id

    def _validate_tick_inputs(
        self,
        head: RuntimeHead,
        scheduler: RoleScheduler,
        assignment: RoleAssignment,
        system_update: SystemFieldUpdate,
        proposal: ProposalRecord,
        decision: ConsolidationDecision,
        output_snapshot: SharedFieldSnapshot,
        updated_core_states: Sequence[CoreStateManifest],
        projection_manifest: ProjectionManifest,
        queued_tool_requests: Sequence[ToolRequestRecord],
        identities: Sequence[CoreIdentity],
    ) -> tuple[
        dict[str, CoreStateManifest],
        tuple[CoreStateManifest, ...],
        tuple[ToolRequestRecord, ...],
        FieldTransactionAudit,
    ]:
        expected_assignment = scheduler.assignment(head.tick_seq)
        if assignment != expected_assignment:
            raise RoleOwnershipError(
                "assignment does not match current deterministic role ring"
            )
        if not isinstance(system_update, SystemFieldUpdate):
            raise TypeError("system_update must be SystemFieldUpdate")
        try:
            working_snapshot = apply_system_update(
                head.snapshot,
                system_update,
            ).working_snapshot
        except (TypeError, ValueError) as exc:
            raise StoreContractError(
                "system update does not match the current canonical head"
            ) from exc
        if (
            proposal.tick_seq != head.tick_seq
            or proposal.assignment_hash != assignment.assignment_hash
            or proposal.proposer_core_id != assignment.proposer_core_id
            or proposal.system_update_id != system_update.update_id
            or proposal.base_field_id != working_snapshot.field_id
            or proposal.base_tick_id != working_snapshot.tick_id
        ):
            raise RoleOwnershipError("proposal does not belong to assigned proposer")
        if proposal.delta is not None and (
            proposal.delta.base_field_id != working_snapshot.field_id
            or proposal.delta.base_tick_id != working_snapshot.tick_id
            or proposal.delta.author_core_id != assignment.proposer_core_id
        ):
            raise RoleOwnershipError("proposal delta has the wrong base or author")
        if proposal.delta is not None:
            try:
                validate_delta(working_snapshot, proposal.delta)
            except (TypeError, ValueError) as exc:
                raise StoreContractError(
                    "proposal delta is invalid against the working field"
                ) from exc
        if (
            decision.tick_seq != head.tick_seq
            or decision.assignment_hash != assignment.assignment_hash
            or decision.consolidator_core_id
            != assignment.consolidator_core_id
            or decision.proposal_id != proposal.proposal_id
            or decision.base_field_id != working_snapshot.field_id
        ):
            raise RoleOwnershipError(
                "decision does not belong to assigned consolidator"
            )
        committed_delta = decision.committed_delta
        if committed_delta is not None and (
            committed_delta.base_field_id != working_snapshot.field_id
            or committed_delta.base_tick_id != working_snapshot.tick_id
            or committed_delta.author_core_id
            != assignment.consolidator_core_id
        ):
            raise RoleOwnershipError(
                "committed delta has the wrong base or consolidator author"
            )
        try:
            transaction_audit = compose_tick_transaction(
                head.snapshot,
                system_update,
                committed_delta,
                consolidator_author_core_id=(
                    None
                    if committed_delta is None
                    else assignment.consolidator_core_id
                ),
            )
        except (TypeError, ValueError) as exc:
            raise StoreContractError(
                "system update and consolidator decision do not compose"
            ) from exc
        if output_snapshot != transaction_audit.final_snapshot:
            raise StoreContractError(
                "output snapshot does not match the audited field transaction"
            )
        if decision.output_field_id != output_snapshot.field_id:
            raise StoreContractError("decision output field does not match output")

        identity_by_id = {identity.core_id: identity for identity in identities}
        current_states = {
            state.core_id: state for state in head.core_state_manifests
        }
        update_values = tuple(updated_core_states)
        request_values = tuple(queued_tool_requests)
        if not decision.accepted and (update_values or request_values):
            raise StoreContractError(
                "rejected consolidation cannot install core state or queue tools"
            )
        update_by_id = {state.core_id: state for state in update_values}
        if len(update_by_id) != len(update_values):
            raise CoreStateUpdateError("duplicate updated core-state manifest")
        allowed = {
            assignment.proposer_core_id,
            assignment.consolidator_core_id,
        }
        if set(update_by_id) - allowed:
            raise CoreStateUpdateError(
                "online tick may update only proposer/consolidator state"
            )
        for core_id, state in update_by_id.items():
            if core_id not in identity_by_id or core_id not in current_states:
                raise CoreStateUpdateError("updated core identity is unknown")
            self._validate_manifest_identity(state, identity_by_id[core_id])
            prior = current_states[core_id]
            if (
                state.generation != head.generation + 1
                or state.tick_seq != head.tick_seq + 1
                or state.committed_field_id != output_snapshot.field_id
                or state.parent_manifest_id != prior.manifest_id
            ):
                raise CoreStateUpdateError(
                    "updated core state is not an exact child of current state"
                )
        projected_states = dict(current_states)
        projected_states.update(update_by_id)

        next_assignment = scheduler.assignment(head.tick_seq + 1)
        expected_projection = ProjectionManifest(
            generation=head.generation + 1,
            tick_seq=head.tick_seq + 1,
            field_id=output_snapshot.field_id,
            field_hash=output_snapshot.canonical_hash,
            core_state_manifest_ids=tuple(
                (core_id, state.manifest_id)
                for core_id, state in projected_states.items()
            ),
            role_index=scheduler.role_index(head.tick_seq + 1),
            role_assignment_hash=next_assignment.assignment_hash,
            parent_projection_id=head.projection_manifest_id,
        )
        if projection_manifest != expected_projection:
            raise StoreContractError(
                "projection manifest does not match exact next head"
            )

        request_ids = [request.request_id for request in request_values]
        if len(request_ids) != len(set(request_ids)):
            raise StoreContractError("duplicate queued tool request")
        for request in request_values:
            if (
                request.tick_seq != head.tick_seq
                or request.requester_core_id
                != assignment.consolidator_core_id
                or request.base_field_id != working_snapshot.field_id
                or request.parent_proposal_id != proposal.proposal_id
            ):
                raise RoleOwnershipError(
                    "tool request is not owned by the assigned consolidator"
                )
        return (
            projected_states,
            update_values,
            request_values,
            transaction_audit,
        )

    def commit_tick(
        self,
        expected_generation: int,
        expected_field_id: str,
        assignment: RoleAssignment,
        system_update: SystemFieldUpdate,
        proposal: ProposalRecord,
        decision: ConsolidationDecision,
        output_snapshot: SharedFieldSnapshot,
        updated_core_states: Sequence[CoreStateManifest],
        projection_manifest: ProjectionManifest,
        queued_tool_requests: Sequence[ToolRequestRecord] = (),
        ingress_queue: IngressQueue | None = None,
        consumed_ingress_event_ids: Sequence[str] = (),
    ) -> TickCommitRecord:
        self._require_initialized()
        if not isinstance(output_snapshot, SharedFieldSnapshot):
            raise TypeError("output_snapshot must be SharedFieldSnapshot")
        consumed_ids = tuple(consumed_ingress_event_ids)
        if not all(isinstance(event_id, str) and event_id for event_id in consumed_ids):
            raise StoreContractError(
                "consumed ingress event IDs must be non-empty strings"
            )
        if consumed_ids:
            from .ingress import IngressQueue

            if not isinstance(ingress_queue, IngressQueue):
                raise StoreContractError(
                    "consumed ingress events require an IngressQueue"
                )
            if ingress_queue.connection is not self._connection:
                raise StoreContractError(
                    "ingress queue must share the runtime store connection"
                )
            if not isinstance(system_update, SystemFieldUpdate):
                raise TypeError("system_update must be SystemFieldUpdate")
            if not set(consumed_ids).issubset(set(system_update.evidence)):
                raise StoreContractError(
                    "consumed ingress events must appear in system-update evidence"
                )
        with self._transaction():
            head = self.current_head()
            if (
                head.generation != expected_generation
                or head.snapshot.field_id != expected_field_id
            ):
                raise StaleHeadError("tick transaction was authored against stale head")
            identities = self._load_identities()
            scheduler = RoleScheduler(identities)
            (
                projected_states,
                updates,
                requests,
                transaction_audit,
            ) = self._validate_tick_inputs(
                head,
                scheduler,
                assignment,
                system_update,
                proposal,
                decision,
                output_snapshot,
                updated_core_states,
                projection_manifest,
                queued_tool_requests,
                identities,
            )

            self._append_journal(
                "system_update",
                system_update.update_id,
                system_update.to_canonical_dict(),
            )
            self._append_journal(
                "proposal",
                proposal.proposal_id,
                proposal.to_canonical_dict(),
            )
            self._append_journal(
                "consolidation_decision",
                decision.decision_id,
                decision.to_canonical_dict(),
            )
            for state in updates:
                self._append_journal(
                    "core_state",
                    state.manifest_id,
                    state.to_canonical_dict(),
                )
                self._connection.execute(
                    """
                    UPDATE core_states
                    SET manifest_id=?, manifest_json=?
                    WHERE core_id=?
                    """,
                    (
                        state.manifest_id,
                        canonical_json_text(state.to_dict()),
                        state.core_id,
                    ),
                )
            self._append_journal(
                "projection",
                projection_manifest.projection_id,
                projection_manifest.to_canonical_dict(),
            )
            for request in requests:
                existing = self._connection.execute(
                    "SELECT request_json FROM tool_requests WHERE request_id=?",
                    (request.request_id,),
                ).fetchone()
                request_json = canonical_json_text(request.to_dict())
                if existing is not None:
                    if existing["request_json"] != request_json:
                        raise IdempotencyConflictError(
                            "tool request ID has conflicting content"
                        )
                    raise IdempotencyConflictError(
                        "tool request is already queued"
                    )
                self._connection.execute(
                    """
                    INSERT INTO tool_requests(
                        request_id,
                        request_json,
                        status
                    ) VALUES (?, ?, 'queued')
                    """,
                    (request.request_id, request_json),
                )
                self._append_journal(
                    "tool_request",
                    request.request_id,
                    request.to_canonical_dict(),
                )

            commit = TickCommitRecord(
                generation=head.generation + 1,
                tick_seq=head.tick_seq,
                assignment_hash=assignment.assignment_hash,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                input_field_id=head.snapshot.field_id,
                system_update_id=system_update.update_id,
                working_field_id=transaction_audit.working_field_id,
                field_transaction_audit_id=transaction_audit.audit_id,
                output_field_id=output_snapshot.field_id,
                updated_core_state_manifest_ids=tuple(
                    state.manifest_id for state in updates
                ),
                projection_id=projection_manifest.projection_id,
                tool_request_ids=tuple(request.request_id for request in requests),
            )
            self._append_journal(
                "tick_commit",
                commit.commit_id,
                commit.to_canonical_dict(),
            )
            if consumed_ids:
                assert ingress_queue is not None
                ingress_queue.consume_in_current_transaction(
                    consumed_ids,
                    commit.commit_id,
                )
            next_assignment = scheduler.assignment(head.tick_seq + 1)
            result = self._connection.execute(
                """
                UPDATE runtime_head
                SET generation=?,
                    tick_seq=?,
                    snapshot_json=?,
                    field_id=?,
                    role_index=?,
                    role_assignment_hash=?,
                    projection_manifest_id=?
                WHERE singleton=1
                  AND generation=?
                  AND field_id=?
                """,
                (
                    head.generation + 1,
                    head.tick_seq + 1,
                    canonical_json_text(serialize_shared_field(output_snapshot)),
                    output_snapshot.field_id,
                    scheduler.role_index(head.tick_seq + 1),
                    next_assignment.assignment_hash,
                    projection_manifest.projection_id,
                    expected_generation,
                    expected_field_id,
                ),
            )
            if result.rowcount != 1:
                raise StaleHeadError("head changed during tick transaction")
            if set(projected_states) != {
                state.core_id for state in self._load_core_states()
            }:
                raise JournalIntegrityError(
                    "core-state projection lost an identity"
                )
        return commit

    def append_tool_result(self, result: ToolResultRecord) -> str:
        if not isinstance(result, ToolResultRecord):
            raise TypeError("result must be ToolResultRecord")
        self._require_initialized()
        with self._transaction():
            request = self._connection.execute(
                """
                SELECT status, result_id, result_json
                FROM tool_requests
                WHERE request_id=?
                """,
                (result.request_id,),
            ).fetchone()
            if request is None:
                raise StoreContractError("tool result references unknown request")
            result_json = canonical_json_text(result.to_dict())
            if request["result_id"] is not None:
                if (
                    request["result_id"] != result.result_id
                    or request["result_json"] != result_json
                ):
                    raise IdempotencyConflictError(
                        "tool request already has a different result"
                    )
            journaled_for_request: list[sqlite3.Row] = []
            for row in self._connection.execute(
                """
                SELECT event_id, record_id, canonical_json
                FROM journal
                WHERE event_type='tool_result'
                ORDER BY seq
                """
            ).fetchall():
                payload = parse_json_object(row["canonical_json"])
                journaled = self._parse_journal_record(
                    "tool_result",
                    str(row["record_id"]),
                    payload,
                )
                if journaled.request_id == result.request_id:
                    journaled_for_request.append(row)
            if len(journaled_for_request) > 1:
                raise JournalIntegrityError(
                    "tool request has multiple journaled results"
                )
            if journaled_for_request:
                existing = journaled_for_request[0]
                if (
                    existing["record_id"] != result.result_id
                    or existing["canonical_json"]
                    != canonical_json_text(result.to_canonical_dict())
                ):
                    raise IdempotencyConflictError(
                        "tool request journal already has a different result"
                    )
                event_id = str(existing["event_id"])
            else:
                if request["result_id"] is not None:
                    raise JournalIntegrityError(
                        "tool result projection lacks journal event"
                    )
                event_id, _ = self._append_journal(
                    "tool_result",
                    result.result_id,
                    result.to_canonical_dict(),
                    idempotency_key=result.idempotency_key,
                )
            updated = self._connection.execute(
                """
                UPDATE tool_requests
                SET status=?, result_id=?, result_json=?
                WHERE request_id=?
                """,
                (
                    result.status,
                    result.result_id,
                    result_json,
                    result.request_id,
                ),
            )
            if updated.rowcount != 1:
                raise JournalIntegrityError(
                    "tool result projection disappeared during repair"
                )
        return event_id

    @staticmethod
    def _require_event_fields(
        payload: Mapping[str, Any],
        expected: set[str],
        label: str,
    ) -> None:
        actual = set(payload)
        if actual != expected:
            raise JournalIntegrityError(
                f"{label} fields mismatch: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )

    @staticmethod
    def _strict_string_tuple(value: Any, label: str) -> tuple[str, ...]:
        if (
            not isinstance(value, list)
            or not all(isinstance(item, str) and item for item in value)
        ):
            raise JournalIntegrityError(f"{label} must be a string list")
        return tuple(value)

    @staticmethod
    def _canonical_payload_matches(
        expected: Mapping[str, Any],
        observed: Mapping[str, Any],
    ) -> bool:
        return canonical_json_text(dict(expected)) == canonical_json_text(
            dict(observed)
        )

    @classmethod
    def _embedded_delta(cls, value: Any, label: str):
        if not isinstance(value, Mapping):
            raise JournalIntegrityError(f"{label} must be a field delta object")
        canonical = dict(value)
        cls._require_event_fields(
            canonical,
            {
                "schema",
                "base_field_id",
                "base_tick_id",
                "author_core_id",
                "pass_id",
                "operations",
                "evidence",
            },
            label,
        )
        digest = canonical_sha256(canonical)
        try:
            delta = deserialize_field_delta(
                {
                    **canonical,
                    "delta_id": digest,
                    "canonical_hash": digest,
                }
            )
        except Exception as exc:
            raise JournalIntegrityError(f"{label} is invalid") from exc
        if not cls._canonical_payload_matches(
            delta.to_canonical_dict(),
            canonical,
        ):
            raise JournalIntegrityError(f"{label} is not canonical")
        return delta

    @classmethod
    def _parse_journal_record(
        cls,
        event_type: str,
        record_id: str,
        payload: Mapping[str, Any],
    ) -> Any:
        try:
            if event_type == "genesis":
                snapshot = deserialize_shared_field(payload)
                if snapshot.field_id != record_id:
                    raise JournalIntegrityError("genesis record ID mismatch")
                return snapshot
            if event_type == "system_update":
                update = deserialize_system_field_update(
                    {
                        **payload,
                        "update_id": record_id,
                        "canonical_hash": record_id,
                    }
                )
                if not cls._canonical_payload_matches(
                    update.to_canonical_dict(),
                    payload,
                ):
                    raise JournalIntegrityError(
                        "system-update journal payload is not canonical"
                    )
                return update
            if event_type == "core_identity":
                identity = deserialize_core_identity(
                    {**payload, "identity_hash": record_id}
                )
                if not cls._canonical_payload_matches(
                    identity.to_canonical_dict(),
                    payload,
                ):
                    raise JournalIntegrityError(
                        "core identity journal payload is not canonical"
                    )
                return identity
            if event_type == "core_state":
                manifest = deserialize_core_state_manifest(
                    {**payload, "manifest_id": record_id}
                )
                if not cls._canonical_payload_matches(
                    manifest.to_canonical_dict(),
                    payload,
                ):
                    raise JournalIntegrityError(
                        "core-state journal payload is not canonical"
                    )
                return manifest
            if event_type == "projection":
                projection = deserialize_projection_manifest(
                    {**payload, "projection_id": record_id}
                )
                if not cls._canonical_payload_matches(
                    projection.to_canonical_dict(),
                    payload,
                ):
                    raise JournalIntegrityError(
                        "projection journal payload is not canonical"
                    )
                return projection
            if event_type == "proposal":
                cls._require_event_fields(
                    payload,
                    {
                        "schema",
                        "tick_seq",
                        "assignment_hash",
                        "proposer_core_id",
                        "system_update_id",
                        "base_field_id",
                        "base_tick_id",
                        "delta",
                        "evidence",
                    },
                    "proposal",
                )
                if payload["schema"] != "axon-proposal-record-v2":
                    raise JournalIntegrityError("unknown proposal schema")
                delta = (
                    None
                    if payload["delta"] is None
                    else cls._embedded_delta(payload["delta"], "proposal delta")
                )
                proposal = ProposalRecord(
                    tick_seq=payload["tick_seq"],
                    assignment_hash=payload["assignment_hash"],
                    proposer_core_id=payload["proposer_core_id"],
                    system_update_id=payload["system_update_id"],
                    base_field_id=payload["base_field_id"],
                    base_tick_id=payload["base_tick_id"],
                    delta=delta,
                    evidence=cls._strict_string_tuple(
                        payload["evidence"],
                        "proposal evidence",
                    ),
                )
                if (
                    proposal.proposal_id != record_id
                    or not cls._canonical_payload_matches(
                        proposal.to_canonical_dict(),
                        payload,
                    )
                ):
                    raise JournalIntegrityError("proposal record ID mismatch")
                return proposal
            if event_type == "consolidation_decision":
                cls._require_event_fields(
                    payload,
                    {
                        "schema",
                        "tick_seq",
                        "assignment_hash",
                        "consolidator_core_id",
                        "proposal_id",
                        "base_field_id",
                        "accepted",
                        "output_field_id",
                        "committed_delta",
                        "committed_delta_id",
                        "reason",
                    },
                    "consolidation decision",
                )
                if (
                    payload["schema"]
                    != "axon-consolidation-decision-v2"
                ):
                    raise JournalIntegrityError(
                        "unknown consolidation decision schema"
                    )
                delta = (
                    None
                    if payload["committed_delta"] is None
                    else cls._embedded_delta(
                        payload["committed_delta"],
                        "committed delta",
                    )
                )
                decision = ConsolidationDecision(
                    tick_seq=payload["tick_seq"],
                    assignment_hash=payload["assignment_hash"],
                    consolidator_core_id=payload["consolidator_core_id"],
                    proposal_id=payload["proposal_id"],
                    base_field_id=payload["base_field_id"],
                    accepted=payload["accepted"],
                    output_field_id=payload["output_field_id"],
                    committed_delta=delta,
                    reason=payload["reason"],
                )
                if (
                    decision.committed_delta_id
                    != payload["committed_delta_id"]
                    or decision.decision_id != record_id
                    or not cls._canonical_payload_matches(
                        decision.to_canonical_dict(),
                        payload,
                    )
                ):
                    raise JournalIntegrityError(
                        "consolidation decision ID mismatch"
                    )
                return decision
            if event_type == "tick_commit":
                cls._require_event_fields(
                    payload,
                    {
                        "schema",
                        "generation",
                        "tick_seq",
                        "assignment_hash",
                        "proposal_id",
                        "decision_id",
                        "input_field_id",
                        "system_update_id",
                        "working_field_id",
                        "field_transaction_audit_id",
                        "output_field_id",
                        "updated_core_state_manifest_ids",
                        "projection_id",
                        "tool_request_ids",
                    },
                    "tick commit",
                )
                if payload["schema"] != "axon-tick-commit-v2":
                    raise JournalIntegrityError("unknown tick commit schema")
                commit = TickCommitRecord(
                    generation=payload["generation"],
                    tick_seq=payload["tick_seq"],
                    assignment_hash=payload["assignment_hash"],
                    proposal_id=payload["proposal_id"],
                    decision_id=payload["decision_id"],
                    input_field_id=payload["input_field_id"],
                    system_update_id=payload["system_update_id"],
                    working_field_id=payload["working_field_id"],
                    field_transaction_audit_id=payload[
                        "field_transaction_audit_id"
                    ],
                    output_field_id=payload["output_field_id"],
                    updated_core_state_manifest_ids=cls._strict_string_tuple(
                        payload["updated_core_state_manifest_ids"],
                        "updated core-state IDs",
                    ),
                    projection_id=payload["projection_id"],
                    tool_request_ids=cls._strict_string_tuple(
                        payload["tool_request_ids"],
                        "tool request IDs",
                    ),
                )
                if (
                    commit.commit_id != record_id
                    or not cls._canonical_payload_matches(
                        commit.to_canonical_dict(),
                        payload,
                    )
                ):
                    raise JournalIntegrityError("tick commit ID mismatch")
                return commit
            if event_type == "tool_request":
                cls._require_event_fields(
                    payload,
                    {
                        "schema",
                        "tick_seq",
                        "requester_core_id",
                        "base_field_id",
                        "tool_name",
                        "arguments",
                        "parent_proposal_id",
                    },
                    "tool request",
                )
                if payload["schema"] != "axon-tool-request-v2":
                    raise JournalIntegrityError("unknown tool request schema")
                request = ToolRequestRecord(
                    tick_seq=payload["tick_seq"],
                    requester_core_id=payload["requester_core_id"],
                    base_field_id=payload["base_field_id"],
                    tool_name=payload["tool_name"],
                    arguments=payload["arguments"],
                    parent_proposal_id=payload["parent_proposal_id"],
                )
                if (
                    request.request_id != record_id
                    or not cls._canonical_payload_matches(
                        request.to_canonical_dict(),
                        payload,
                    )
                ):
                    raise JournalIntegrityError("tool request ID mismatch")
                return request
            if event_type == "tool_result":
                cls._require_event_fields(
                    payload,
                    {
                        "schema",
                        "idempotency_key",
                        "request_id",
                        "status",
                        "payload",
                        "parent_ids",
                    },
                    "tool result",
                )
                if payload["schema"] != "axon-tool-result-v1":
                    raise JournalIntegrityError("unknown tool result schema")
                result = ToolResultRecord(
                    idempotency_key=payload["idempotency_key"],
                    request_id=payload["request_id"],
                    status=payload["status"],
                    payload=payload["payload"],
                    parent_ids=cls._strict_string_tuple(
                        payload["parent_ids"],
                        "tool result parent IDs",
                    ),
                )
                if (
                    result.result_id != record_id
                    or not cls._canonical_payload_matches(
                        result.to_canonical_dict(),
                        payload,
                    )
                ):
                    raise JournalIntegrityError("tool result ID mismatch")
                return result
            if event_type == "source_event":
                cls._require_event_fields(
                    payload,
                    {
                        "schema",
                        "idempotency_key",
                        "source_kind",
                        "source_ref",
                        "payload",
                        "parent_ids",
                    },
                    "source event",
                )
                if payload["schema"] != "axon-source-event-v1":
                    raise JournalIntegrityError("unknown source event schema")
                event = SourceEvent(
                    idempotency_key=payload["idempotency_key"],
                    source_kind=payload["source_kind"],
                    source_ref=payload["source_ref"],
                    payload=payload["payload"],
                    parent_ids=cls._strict_string_tuple(
                        payload["parent_ids"],
                        "source parent IDs",
                    ),
                )
                if (
                    event.source_event_id != record_id
                    or not cls._canonical_payload_matches(
                        event.to_canonical_dict(),
                        payload,
                    )
                ):
                    raise JournalIntegrityError("source event ID mismatch")
                return event
            if event_type == "span_lifecycle":
                cls._require_event_fields(
                    payload,
                    {
                        "schema",
                        "action",
                        "span_id",
                        "region",
                        "field_id",
                        "payload",
                        "parent_ids",
                    },
                    "span lifecycle",
                )
                if payload["schema"] != "axon-span-lifecycle-v1":
                    raise JournalIntegrityError(
                        "unknown span lifecycle schema"
                    )
                event = SpanLifecycleEvent(
                    action=payload["action"],
                    span_id=payload["span_id"],
                    region=payload["region"],
                    field_id=payload["field_id"],
                    payload=payload["payload"],
                    parent_ids=cls._strict_string_tuple(
                        payload["parent_ids"],
                        "span lifecycle parent IDs",
                    ),
                )
                if (
                    event.lifecycle_event_id != record_id
                    or not cls._canonical_payload_matches(
                        event.to_canonical_dict(),
                        payload,
                    )
                ):
                    raise JournalIntegrityError(
                        "span lifecycle event ID mismatch"
                    )
                return event
        except JournalIntegrityError:
            raise
        except Exception as exc:
            raise JournalIntegrityError(
                f"malformed {event_type!r} journal record"
            ) from exc
        raise JournalIntegrityError(f"unknown journal event type {event_type!r}")

    def _table_exists(self, table_name: str) -> bool:
        return (
            self._connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type='table' AND name=?
                """,
                (table_name,),
            ).fetchone()
            is not None
        )

    def _verify_ingress_receipts(
        self,
        commits: Mapping[str, TickCommitRecord],
        system_updates: Mapping[str, SystemFieldUpdate],
    ) -> None:
        receipt_table = "axon_ingress_consumption_receipts"
        if not self._table_exists(receipt_table):
            return
        if not self._table_exists("axon_ingress_events"):
            raise JournalIntegrityError(
                "ingress receipts exist without the ingress event table"
            )
        from .ingress import ConsumptionReceipt

        rows = self._connection.execute(
            """
            SELECT
                receipt_id,
                event_id,
                tick_commit_id,
                canonical_json,
                canonical_sha256
            FROM axon_ingress_consumption_receipts
            ORDER BY receipt_seq
            """
        ).fetchall()
        for row in rows:
            try:
                payload = parse_json_object(row["canonical_json"])
                if canonical_json_text(payload) != row["canonical_json"]:
                    raise JournalIntegrityError(
                        "ingress receipt payload is not canonical"
                    )
                receipt = ConsumptionReceipt.from_dict(payload)
            except JournalIntegrityError:
                raise
            except Exception as exc:
                raise JournalIntegrityError(
                    "ingress receipt is malformed"
                ) from exc
            if (
                row["receipt_id"] != receipt.receipt_id
                or row["event_id"] != receipt.event_id
                or row["tick_commit_id"] != receipt.tick_commit_id
                or row["canonical_sha256"] != receipt.receipt_id
            ):
                raise JournalIntegrityError(
                    "ingress receipt projection conflicts with its payload"
                )
            if self._connection.execute(
                "SELECT 1 FROM axon_ingress_events WHERE event_id=?",
                (receipt.event_id,),
            ).fetchone() is None:
                raise JournalIntegrityError(
                    "ingress receipt references a missing event"
                )
            commit = commits.get(receipt.tick_commit_id)
            if commit is None:
                raise JournalIntegrityError(
                    "ingress receipt references a non-journaled tick commit"
                )
            system_update = system_updates.get(commit.system_update_id)
            if (
                system_update is None
                or receipt.event_id not in system_update.evidence
            ):
                raise JournalIntegrityError(
                    "ingress receipt event is absent from tick evidence"
                )

    def replay_journal(self) -> RuntimeHead:
        """Reconstruct the current runtime head from journal truth alone."""

        self._require_initialized()
        self.verify_journal_chain()
        rows = self._connection.execute(
            """
            SELECT seq, event_type, record_id, canonical_json
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
                raise JournalIntegrityError(
                    "journal contains a duplicate event-type/record identity"
                )
            record_sequences[key] = seq

        genesis_records = [
            (seq, value)
            for seq, event_type, _, value in parsed
            if event_type == "genesis"
        ]
        if len(genesis_records) != 1:
            raise JournalIntegrityError(
                "journal must contain exactly one canonical genesis"
            )
        genesis_seq, snapshot = genesis_records[0]
        if genesis_seq != parsed[0][0]:
            raise JournalIntegrityError("genesis must be the first journal event")
        if snapshot.tick_id != 0 or snapshot.parent_field_id is not None:
            raise JournalIntegrityError("journal genesis is not a root snapshot")

        identities = tuple(
            value
            for _, event_type, _, value in parsed
            if event_type == "core_identity"
        )
        try:
            validate_identity_population(identities)
            scheduler = RoleScheduler(identities)
        except Exception as exc:
            raise JournalIntegrityError(
                "journal core identity ring is invalid"
            ) from exc
        identity_by_id = {identity.core_id: identity for identity in identities}

        def unique_records(event_type: str, id_attribute: str) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for _, observed_type, _, value in parsed:
                if observed_type != event_type:
                    continue
                identifier = str(getattr(value, id_attribute))
                if identifier in result:
                    raise JournalIntegrityError(
                        f"duplicate {event_type} record ID"
                    )
                result[identifier] = value
            return result

        states = unique_records("core_state", "manifest_id")
        projections = unique_records("projection", "projection_id")
        system_updates = unique_records("system_update", "update_id")
        proposals = unique_records("proposal", "proposal_id")
        decisions = unique_records(
            "consolidation_decision",
            "decision_id",
        )
        requests = unique_records("tool_request", "request_id")
        results = unique_records("tool_result", "result_id")
        commits = [
            (seq, value)
            for seq, event_type, _, value in parsed
            if event_type == "tick_commit"
        ]

        initial_states: dict[str, CoreStateManifest] = {}
        for state in states.values():
            if (
                state.generation == 0
                and state.tick_seq == 0
                and state.parent_manifest_id is None
            ):
                if state.core_id in initial_states:
                    raise JournalIntegrityError(
                        "duplicate genesis core-state manifest"
                    )
                initial_states[state.core_id] = state
        if set(initial_states) != set(identity_by_id):
            raise JournalIntegrityError(
                "genesis core states do not cover journal identities"
            )
        for core_id, state in initial_states.items():
            try:
                self._validate_manifest_identity(
                    state,
                    identity_by_id[core_id],
                )
            except Exception as exc:
                raise JournalIntegrityError(
                    "genesis core state conflicts with journal identity"
                ) from exc
            if state.committed_field_id != snapshot.field_id:
                raise JournalIntegrityError(
                    "genesis core state references the wrong field"
                )

        initial_projection_values = [
            projection
            for projection in projections.values()
            if projection.generation == 0 and projection.tick_seq == 0
        ]
        if len(initial_projection_values) != 1:
            raise JournalIntegrityError(
                "journal must contain exactly one genesis projection"
            )
        projection = initial_projection_values[0]
        assignment = scheduler.assignment(0)
        expected_projection = ProjectionManifest(
            generation=0,
            tick_seq=0,
            field_id=snapshot.field_id,
            field_hash=snapshot.canonical_hash,
            core_state_manifest_ids=tuple(
                (core_id, state.manifest_id)
                for core_id, state in initial_states.items()
            ),
            role_index=0,
            role_assignment_hash=assignment.assignment_hash,
        )
        if projection != expected_projection:
            raise JournalIntegrityError("genesis projection is inconsistent")
        first_commit_seq = min(
            (seq for seq, _ in commits),
            default=float("inf"),
        )
        identity_sequences = [
            record_sequences[("core_identity", identity.identity_hash)]
            for identity in identities
        ]
        initial_state_sequences = [
            record_sequences[("core_state", state.manifest_id)]
            for state in initial_states.values()
        ]
        projection_sequence = record_sequences[
            ("projection", projection.projection_id)
        ]
        if (
            not identity_sequences
            or not initial_state_sequences
            or min(identity_sequences) <= genesis_seq
            or max(identity_sequences) >= min(initial_state_sequences)
            or max(initial_state_sequences) >= projection_sequence
            or projection_sequence >= first_commit_seq
        ):
            raise JournalIntegrityError(
                "genesis identity/state/projection journal order is invalid"
            )

        current_states = dict(initial_states)
        generation = 0
        tick_seq = 0
        used_state_ids = {state.manifest_id for state in initial_states.values()}
        used_projection_ids = {projection.projection_id}
        used_system_updates: set[str] = set()
        used_proposals: set[str] = set()
        used_decisions: set[str] = set()
        used_requests: set[str] = set()

        for commit_seq, commit in commits:
            expected_assignment = scheduler.assignment(tick_seq)
            if (
                commit.generation != generation + 1
                or commit.tick_seq != tick_seq
                or commit.assignment_hash
                != expected_assignment.assignment_hash
                or commit.input_field_id != snapshot.field_id
            ):
                raise JournalIntegrityError(
                    "tick commit generation, role, or input lineage mismatch"
                )
            reference_keys = [
                ("system_update", commit.system_update_id),
                ("proposal", commit.proposal_id),
                ("consolidation_decision", commit.decision_id),
                ("projection", commit.projection_id),
                *(
                    ("core_state", manifest_id)
                    for manifest_id in commit.updated_core_state_manifest_ids
                ),
                *(
                    ("tool_request", request_id)
                    for request_id in commit.tool_request_ids
                ),
            ]
            if any(
                key not in record_sequences
                or record_sequences[key] >= commit_seq
                for key in reference_keys
            ):
                raise JournalIntegrityError(
                    "tick commit has a missing or forward journal reference"
                )
            system_update_seq = record_sequences[
                ("system_update", commit.system_update_id)
            ]
            proposal_seq = record_sequences[("proposal", commit.proposal_id)]
            decision_seq = record_sequences[
                ("consolidation_decision", commit.decision_id)
            ]
            next_projection_seq = record_sequences[
                ("projection", commit.projection_id)
            ]
            state_sequences = [
                record_sequences[("core_state", manifest_id)]
                for manifest_id in commit.updated_core_state_manifest_ids
            ]
            request_sequences = [
                record_sequences[("tool_request", request_id)]
                for request_id in commit.tool_request_ids
            ]
            if not (
                system_update_seq
                < proposal_seq
                < decision_seq
                < next_projection_seq
                < commit_seq
            ):
                raise JournalIntegrityError(
                    "tick transaction journal order is invalid"
                )
            if any(
                sequence <= decision_seq or sequence >= next_projection_seq
                for sequence in state_sequences
            ) or any(
                sequence <= next_projection_seq or sequence >= commit_seq
                for sequence in request_sequences
            ):
                raise JournalIntegrityError(
                    "tick state/tool journal order is invalid"
                )
            system_update = system_updates.get(commit.system_update_id)
            if (
                system_update is None
                or system_update.update_id in used_system_updates
            ):
                raise JournalIntegrityError(
                    "tick commit references missing/reused system update"
                )
            try:
                working_snapshot = apply_system_update(
                    snapshot,
                    system_update,
                ).working_snapshot
            except (TypeError, ValueError) as exc:
                raise JournalIntegrityError(
                    "system update does not replay from the canonical head"
                ) from exc
            if commit.working_field_id != working_snapshot.field_id:
                raise JournalIntegrityError(
                    "tick commit working field does not match system update"
                )
            proposal = proposals.get(commit.proposal_id)
            decision = decisions.get(commit.decision_id)
            if proposal is None or decision is None:
                raise JournalIntegrityError(
                    "tick commit references missing proposal or decision"
                )
            if proposal.proposal_id in used_proposals:
                raise JournalIntegrityError("proposal was committed twice")
            if decision.decision_id in used_decisions:
                raise JournalIntegrityError("decision was committed twice")
            if (
                proposal.tick_seq != tick_seq
                or proposal.assignment_hash
                != expected_assignment.assignment_hash
                or proposal.proposer_core_id
                != expected_assignment.proposer_core_id
                or proposal.system_update_id != system_update.update_id
                or proposal.base_field_id != working_snapshot.field_id
                or proposal.base_tick_id != working_snapshot.tick_id
            ):
                raise JournalIntegrityError(
                    "proposal does not match replayed tick base"
                )
            if proposal.delta is not None and (
                proposal.delta.base_field_id != working_snapshot.field_id
                or proposal.delta.base_tick_id != working_snapshot.tick_id
                or proposal.delta.author_core_id
                != expected_assignment.proposer_core_id
            ):
                raise JournalIntegrityError(
                    "proposal delta does not match replayed proposer/base"
                )
            if proposal.delta is not None:
                try:
                    validate_delta(working_snapshot, proposal.delta)
                except (TypeError, ValueError) as exc:
                    raise JournalIntegrityError(
                        "proposal delta is invalid against replayed working field"
                    ) from exc
            if (
                decision.tick_seq != tick_seq
                or decision.assignment_hash
                != expected_assignment.assignment_hash
                or decision.consolidator_core_id
                != expected_assignment.consolidator_core_id
                or decision.proposal_id != proposal.proposal_id
                or decision.base_field_id != working_snapshot.field_id
            ):
                raise JournalIntegrityError(
                    "decision does not match replayed consolidator/base"
                )
            committed_delta = decision.committed_delta
            if committed_delta is not None and (
                committed_delta.base_field_id != working_snapshot.field_id
                or committed_delta.base_tick_id != working_snapshot.tick_id
                or committed_delta.author_core_id
                != expected_assignment.consolidator_core_id
            ):
                raise JournalIntegrityError(
                    "committed delta does not match replayed consolidator/base"
                )
            try:
                transaction_audit = compose_tick_transaction(
                    snapshot,
                    system_update,
                    committed_delta,
                    consolidator_author_core_id=(
                        None
                        if committed_delta is None
                        else expected_assignment.consolidator_core_id
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise JournalIntegrityError(
                    "field transaction cannot be replayed"
                ) from exc
            output_snapshot = transaction_audit.final_snapshot
            if (
                output_snapshot.tick_id != snapshot.tick_id + 1
                or output_snapshot.parent_field_id != snapshot.field_id
                or transaction_audit.working_field_id
                != commit.working_field_id
                or transaction_audit.audit_id
                != commit.field_transaction_audit_id
                or decision.output_field_id != output_snapshot.field_id
                or commit.output_field_id != output_snapshot.field_id
            ):
                raise JournalIntegrityError(
                    "replayed output parent, tick, or field ID mismatch"
                )

            allowed_updates = {
                expected_assignment.proposer_core_id,
                expected_assignment.consolidator_core_id,
            }
            for manifest_id in commit.updated_core_state_manifest_ids:
                state = states.get(manifest_id)
                if state is None or manifest_id in used_state_ids:
                    raise JournalIntegrityError(
                        "tick commit references missing/reused core state"
                    )
                if state.core_id not in allowed_updates:
                    raise JournalIntegrityError(
                        "tick commit updates a non-online core"
                    )
                identity = identity_by_id.get(state.core_id)
                prior = current_states.get(state.core_id)
                if identity is None or prior is None:
                    raise JournalIntegrityError(
                        "tick core-state identity is missing"
                    )
                try:
                    self._validate_manifest_identity(state, identity)
                except Exception as exc:
                    raise JournalIntegrityError(
                        "tick core state conflicts with journal identity"
                    ) from exc
                if (
                    state.generation != generation + 1
                    or state.tick_seq != tick_seq + 1
                    or state.committed_field_id != output_snapshot.field_id
                    or state.parent_manifest_id != prior.manifest_id
                ):
                    raise JournalIntegrityError(
                        "tick core-state lineage is inconsistent"
                    )
                current_states[state.core_id] = state
                used_state_ids.add(manifest_id)

            for request_id in commit.tool_request_ids:
                request = requests.get(request_id)
                if request is None or request_id in used_requests:
                    raise JournalIntegrityError(
                        "tick commit references missing/reused tool request"
                    )
                if (
                    request.tick_seq != tick_seq
                    or request.requester_core_id
                    != expected_assignment.consolidator_core_id
                    or request.base_field_id != working_snapshot.field_id
                    or request.parent_proposal_id != proposal.proposal_id
                ):
                    raise JournalIntegrityError(
                        "tool request does not match replayed consolidator tick"
                    )
                used_requests.add(request_id)

            next_assignment = scheduler.assignment(tick_seq + 1)
            next_projection = projections.get(commit.projection_id)
            if (
                next_projection is None
                or commit.projection_id in used_projection_ids
            ):
                raise JournalIntegrityError(
                    "tick commit references missing/reused projection"
                )
            expected_projection = ProjectionManifest(
                generation=generation + 1,
                tick_seq=tick_seq + 1,
                field_id=output_snapshot.field_id,
                field_hash=output_snapshot.canonical_hash,
                core_state_manifest_ids=tuple(
                    (core_id, state.manifest_id)
                    for core_id, state in current_states.items()
                ),
                role_index=scheduler.role_index(tick_seq + 1),
                role_assignment_hash=next_assignment.assignment_hash,
                parent_projection_id=projection.projection_id,
            )
            if next_projection != expected_projection:
                raise JournalIntegrityError(
                    "tick projection does not match replayed truth"
                )
            used_projection_ids.add(next_projection.projection_id)
            used_system_updates.add(system_update.update_id)
            used_proposals.add(proposal.proposal_id)
            used_decisions.add(decision.decision_id)
            projection = next_projection
            snapshot = output_snapshot
            generation += 1
            tick_seq += 1

        if set(states) != used_state_ids:
            raise JournalIntegrityError("journal contains orphan core-state records")
        if set(projections) != used_projection_ids:
            raise JournalIntegrityError("journal contains orphan projections")
        if set(system_updates) != used_system_updates:
            raise JournalIntegrityError("journal contains orphan system updates")
        if set(proposals) != used_proposals or set(decisions) != used_decisions:
            raise JournalIntegrityError(
                "journal contains orphan proposals or decisions"
            )
        if set(requests) != used_requests:
            raise JournalIntegrityError("journal contains orphan tool requests")
        result_by_request: dict[str, ToolResultRecord] = {}
        for result in results.values():
            if result.request_id not in requests:
                raise JournalIntegrityError(
                    "tool result references a non-journaled request"
                )
            if result.request_id in result_by_request:
                raise JournalIntegrityError(
                    "tool request has more than one journaled result"
                )
            if record_sequences[("tool_result", result.result_id)] <= (
                record_sequences[("tool_request", result.request_id)]
            ):
                raise JournalIntegrityError(
                    "tool result precedes its journaled request"
                )
            result_by_request[result.request_id] = result

        self._verify_ingress_receipts(
            {commit.commit_id: commit for _, commit in commits},
            system_updates,
        )

        head_assignment = scheduler.assignment(tick_seq)
        return RuntimeHead(
            generation=generation,
            tick_seq=tick_seq,
            snapshot=snapshot,
            role_index=scheduler.role_index(tick_seq),
            role_assignment_hash=head_assignment.assignment_hash,
            projection_manifest_id=projection.projection_id,
            core_state_manifests=tuple(
                current_states[identity.core_id] for identity in identities
            ),
        )

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
                raise JournalIntegrityError(
                    f"journal parent mismatch at sequence {row['seq']}"
                )
            try:
                payload = parse_json_object(row["canonical_json"])
            except Exception as exc:
                raise JournalIntegrityError(
                    f"journal payload is invalid at sequence {row['seq']}"
                ) from exc
            if canonical_json_text(payload) != row["canonical_json"]:
                raise JournalIntegrityError(
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
                raise JournalIntegrityError(
                    f"journal event hash mismatch at sequence {row['seq']}"
                )
            previous = str(row["event_id"])
        return previous

    def recover(self) -> RuntimeHead:
        self._require_initialized()
        try:
            truth = self.replay_journal()
            projected = self.current_head()
        except JournalIntegrityError:
            raise
        except Exception as exc:
            raise JournalIntegrityError(
                "mutable runtime projections are invalid"
            ) from exc
        if projected != truth:
            raise JournalIntegrityError(
                "mutable runtime projections drift from journal truth"
            )
        return projected

    def rebuild_runtime_head_from_journal(self) -> RuntimeHead:
        """Transactionally replace mutable head projections from journal truth."""

        self._require_initialized()
        with self._transaction():
            truth = self.replay_journal()
            self._connection.execute("DELETE FROM core_states")
            for state in truth.core_state_manifests:
                self._connection.execute(
                    """
                    INSERT INTO core_states(core_id, manifest_id, manifest_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        state.core_id,
                        state.manifest_id,
                        canonical_json_text(state.to_dict()),
                    ),
                )
            self._connection.execute(
                """
                INSERT INTO runtime_head(
                    singleton,
                    generation,
                    tick_seq,
                    snapshot_json,
                    field_id,
                    role_index,
                    role_assignment_hash,
                    projection_manifest_id
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    generation=excluded.generation,
                    tick_seq=excluded.tick_seq,
                    snapshot_json=excluded.snapshot_json,
                    field_id=excluded.field_id,
                    role_index=excluded.role_index,
                    role_assignment_hash=excluded.role_assignment_hash,
                    projection_manifest_id=excluded.projection_manifest_id
                """,
                (
                    truth.generation,
                    truth.tick_seq,
                    canonical_json_text(serialize_shared_field(truth.snapshot)),
                    truth.snapshot.field_id,
                    truth.role_index,
                    truth.role_assignment_hash,
                    truth.projection_manifest_id,
                ),
            )
            rebuilt = self.current_head()
            if rebuilt != truth:
                raise JournalIntegrityError(
                    "rebuilt mutable projections do not match journal truth"
                )
        return rebuilt


__all__ = ["RuntimeStore"]

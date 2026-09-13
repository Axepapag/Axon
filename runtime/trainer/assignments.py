"""Durable, Heart-owned assignment lifecycle store for real runtime training.

Assignments are renewable work objects handed to OFFLINE_TRAINING cohort
cores, not candidate lifespans.  This store owns the records; Heart alone
validates and commits every transition (cores propose, Heart commits):

- claim (creation with a durable lease) -> active -> paused -> resumed ->
  completed | escalated, plus lease-expiry reclaim and supervisor
  preemption;
- typed attempts with typed emission references, Soul lineage, and exact
  parameter/optimizer generations;
- critiques and narrow supervisor actions (curriculum adjustment, rollback).

Physical guarantees:

- every mutation carries a caller-supplied idempotency key; replaying the
  same command returns the original outcome and never applies twice;
- every transition is a compare-and-swap on the record revision, and the
  journal event is the durable intent: a crash between journal append and
  record write is finalized by re-issuing the command or by ``recover()``;
- every write is a temp file + fsync + os.replace; every record is
  content-addressed with sha256 and verified on load;
- leases expire: expiry pauses/reclaims the assignment but never deletes
  attempts and never records success or failure by timeout alone;
- a series of failed attempts escalates to supervisor inspection; the store
  never fabricates competence.

The state root is constructor-injected and time comes from an injectable
clock, so behavior is fully deterministic under a fixed clock.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from runtime.field import canonical_json_bytes, canonical_sha256

ASSIGNMENT_SCHEMA = "axon-trainer-assignment-v1"
ASSIGNMENT_IDENTITY_SCHEMA = "axon-trainer-assignment-identity-v1"
ATTEMPT_SCHEMA = "axon-trainer-assignment-attempt-v1"
CRITIQUE_SCHEMA = "axon-trainer-attempt-critique-v1"
LEASE_SCHEMA = "axon-trainer-assignment-lease-v1"
EVENT_SCHEMA = "axon-trainer-assignment-event-v1"


class AssignmentStoreError(RuntimeError):
    pass


class AssignmentKind(str, Enum):
    LEARNING = "learning"
    EVALUATION = "evaluation"
    SUPERVISORY = "supervisory"


class AssignmentStatus(str, Enum):
    CLAIMED = "claimed"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class AttemptOutcome(str, Enum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    COUNTERFACTUAL_ONLY = "counterfactual_only"


_TERMINAL_STATUSES = (AssignmentStatus.COMPLETED, AssignmentStatus.ESCALATED)
_CORE_STATUSES = (AssignmentStatus.CLAIMED, AssignmentStatus.ACTIVE)


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not (-float("inf") < value < float("inf")):
        raise ValueError(f"{label} must be finite")
    return value


def _epoch(value: float, label: str) -> float:
    value = _finite(value, label)
    return value


def _count(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    copied = dict(value)
    canonical_json_bytes(copied)  # fail closed on non-JSON-safe content
    return copied


def _safe_record_id(value: str, label: str) -> str:
    value = _required(value, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a 64-character lowercase sha256 hex digest")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssignmentStoreError(f"cannot read valid assignment JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AssignmentStoreError(f"assignment JSON must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_content_addressed(path: Path, value: Mapping[str, Any]) -> None:
    """Write an immutable content-addressed record; collisions must agree."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read_json(path)
        if canonical_json_bytes(existing) != canonical_json_bytes(dict(value)):
            raise AssignmentStoreError(f"content-addressed record collision at {path}")
        return
    _atomic_json(path, value)


@dataclass(frozen=True, slots=True)
class AssignmentLease:
    """Durable holder lease: expiry pauses/reclaims, never decides outcomes."""

    holder_core_id: str
    issued_at: float
    expires_at: float
    lease_epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "holder_core_id", _required(self.holder_core_id, "holder_core_id"))
        object.__setattr__(self, "issued_at", _epoch(self.issued_at, "issued_at"))
        object.__setattr__(self, "expires_at", _epoch(self.expires_at, "expires_at"))
        if self.expires_at <= self.issued_at:
            raise ValueError("lease expires_at must be after issued_at")
        if isinstance(self.lease_epoch, bool) or not isinstance(self.lease_epoch, int) or self.lease_epoch < 1:
            raise ValueError("lease_epoch must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEASE_SCHEMA,
            "holder_core_id": self.holder_core_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "lease_epoch": self.lease_epoch,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssignmentLease":
        item = dict(value)
        if item.get("schema") != LEASE_SCHEMA:
            raise AssignmentStoreError("serialized assignment lease schema mismatch")
        item.pop("schema", None)
        try:
            return cls(**item)
        except TypeError as exc:
            raise AssignmentStoreError("serialized assignment lease fields are invalid") from exc

    def expired(self, now: float) -> bool:
        return _epoch(now, "now") >= self.expires_at


@dataclass(frozen=True, slots=True)
class TrainingAssignment:
    """One renewable training work object owned by Heart."""

    kind: AssignmentKind
    curriculum_ref: str
    cohort_eligibility: Mapping[str, Any]
    field_binding: Mapping[str, Any]
    origin: str
    created_at: float
    status: AssignmentStatus = AssignmentStatus.CLAIMED
    lease: AssignmentLease | None = None
    failure_threshold: int = 3
    attempt_count: int = 0
    consecutive_failures: int = 0
    rollback_target: Mapping[str, Any] | None = None
    target_text: str | None = None
    revision: int = 0
    # Stable identity of the work object.  It is derived once at creation and
    # then preserved across supervisor revisions of curriculum_ref and
    # field_binding; those fields describe the assignment's current issued
    # view and therefore cannot also redefine its identity.
    assignment_id: str = ""
    record_hash: str = field(init=False)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, AssignmentKind) else AssignmentKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "curriculum_ref", _required(self.curriculum_ref, "curriculum_ref"))
        cohort = _mapping(self.cohort_eligibility, "cohort_eligibility")
        core_ids = cohort.get("core_ids")
        if core_ids is not None:
            if (
                not isinstance(core_ids, (list, tuple))
                or not core_ids
                or any(not isinstance(item, str) or not item.strip() for item in core_ids)
            ):
                raise ValueError("cohort_eligibility core_ids must be a non-empty list of core ids")
        object.__setattr__(self, "cohort_eligibility", cohort)
        binding = _mapping(self.field_binding, "field_binding")
        for label in ("field_id", "view_id"):
            _required(str(binding.get(label, "")), f"field_binding {label}")
        object.__setattr__(self, "field_binding", binding)
        object.__setattr__(self, "origin", _required(self.origin, "origin"))
        object.__setattr__(self, "created_at", _epoch(self.created_at, "created_at"))
        status = self.status if isinstance(self.status, AssignmentStatus) else AssignmentStatus(self.status)
        object.__setattr__(self, "status", status)
        if self.lease is not None and not isinstance(self.lease, AssignmentLease):
            raise TypeError("lease must be AssignmentLease or None")
        if isinstance(self.failure_threshold, bool) or not isinstance(self.failure_threshold, int) or self.failure_threshold < 1:
            raise ValueError("failure_threshold must be a positive integer")
        object.__setattr__(self, "attempt_count", _count(self.attempt_count, "attempt_count"))
        object.__setattr__(self, "consecutive_failures", _count(self.consecutive_failures, "consecutive_failures"))
        if self.rollback_target is not None:
            object.__setattr__(self, "rollback_target", _mapping(self.rollback_target, "rollback_target"))
        if self.target_text is not None:
            # The explicit expected target of the assignment (e.g. the exact
            # answer text for an `ABC? -> D` contract).  When present, the
            # objective contract binds THIS text — loss and emissions are
            # computed against it, never against instruction reconstruction.
            object.__setattr__(self, "target_text", _required(self.target_text, "target_text"))
        object.__setattr__(self, "revision", _count(self.revision, "revision"))
        if self.assignment_id:
            object.__setattr__(
                self,
                "assignment_id",
                _safe_record_id(self.assignment_id, "assignment_id"),
            )
        else:
            object.__setattr__(
                self, "assignment_id", canonical_sha256(self._identity_dict())
            )
        object.__setattr__(self, "record_hash", canonical_sha256(self.to_canonical_dict(include_hash=False)))

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema": ASSIGNMENT_IDENTITY_SCHEMA,
            "kind": self.kind.value,
            "curriculum_ref": self.curriculum_ref,
            "cohort_eligibility": dict(self.cohort_eligibility),
            "field_binding": dict(self.field_binding),
            "origin": self.origin,
            "created_at": self.created_at,
            "target_text": self.target_text,
        }

    def to_canonical_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = self._identity_dict()
        value.update(
            {
                "schema": ASSIGNMENT_SCHEMA,
                "status": self.status.value,
                "lease": None if self.lease is None else self.lease.to_dict(),
                "failure_threshold": self.failure_threshold,
                "attempt_count": self.attempt_count,
                "consecutive_failures": self.consecutive_failures,
                "rollback_target": None if self.rollback_target is None else dict(self.rollback_target),
                "target_text": self.target_text,
                "revision": self.revision,
                "assignment_id": self.assignment_id,
            }
        )
        if include_hash:
            value["record_hash"] = self.record_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingAssignment":
        item = dict(value)
        if item.get("schema") != ASSIGNMENT_SCHEMA:
            raise AssignmentStoreError("serialized assignment schema mismatch")
        record_hash = item.pop("record_hash", None)
        item.pop("schema", None)
        lease = item.get("lease")
        if lease is not None:
            item["lease"] = AssignmentLease.from_dict(lease)
        try:
            record = cls(**item)
        except TypeError as exc:
            raise AssignmentStoreError("serialized assignment fields are invalid") from exc
        if record.record_hash != record_hash:
            raise AssignmentStoreError("assignment record identity/hash mismatch")
        return record


@dataclass(frozen=True, slots=True)
class AssignmentAttempt:
    """One exact committed attempt; outcomes are evidence, never fabricated."""

    assignment_id: str
    core_id: str
    attempt_index: int
    emission_ref: Mapping[str, Any]
    soul_lineage: Mapping[str, Any]
    parameter_generation: str
    optimizer_generation: str
    evidence: Mapping[str, Any]
    outcome: AttemptOutcome = AttemptOutcome.PENDING
    revision: int = 0
    attempt_id: str = field(init=False)
    record_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignment_id", _safe_record_id(self.assignment_id, "assignment_id"))
        object.__setattr__(self, "core_id", _required(self.core_id, "core_id"))
        object.__setattr__(self, "attempt_index", _count(self.attempt_index, "attempt_index"))
        object.__setattr__(self, "emission_ref", _mapping(self.emission_ref, "emission_ref"))
        object.__setattr__(self, "soul_lineage", _mapping(self.soul_lineage, "soul_lineage"))
        object.__setattr__(self, "parameter_generation", _required(self.parameter_generation, "parameter_generation"))
        object.__setattr__(self, "optimizer_generation", _required(self.optimizer_generation, "optimizer_generation"))
        object.__setattr__(self, "evidence", _mapping(self.evidence, "evidence"))
        outcome = self.outcome if isinstance(self.outcome, AttemptOutcome) else AttemptOutcome(self.outcome)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "revision", _count(self.revision, "revision"))
        object.__setattr__(self, "attempt_id", canonical_sha256(self._identity_dict()))
        object.__setattr__(self, "record_hash", canonical_sha256(self.to_canonical_dict(include_hash=False)))

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema": ATTEMPT_SCHEMA,
            "assignment_id": self.assignment_id,
            "core_id": self.core_id,
            "attempt_index": self.attempt_index,
        }

    def to_canonical_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = self._identity_dict()
        value.update(
            {
                "schema": ATTEMPT_SCHEMA,
                "emission_ref": dict(self.emission_ref),
                "soul_lineage": dict(self.soul_lineage),
                "parameter_generation": self.parameter_generation,
                "optimizer_generation": self.optimizer_generation,
                "evidence": dict(self.evidence),
                "outcome": self.outcome.value,
                "revision": self.revision,
                "attempt_id": self.attempt_id,
            }
        )
        if include_hash:
            value["record_hash"] = self.record_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AssignmentAttempt":
        item = dict(value)
        if item.get("schema") != ATTEMPT_SCHEMA:
            raise AssignmentStoreError("serialized attempt schema mismatch")
        attempt_id = item.pop("attempt_id", None)
        record_hash = item.pop("record_hash", None)
        item.pop("schema", None)
        try:
            record = cls(**item)
        except TypeError as exc:
            raise AssignmentStoreError("serialized attempt fields are invalid") from exc
        if record.attempt_id != attempt_id or record.record_hash != record_hash:
            raise AssignmentStoreError("attempt record identity/hash mismatch")
        return record


@dataclass(frozen=True, slots=True)
class AttemptCritique:
    """Supervisor-authored critique bound to one exact attempt."""

    attempt_id: str
    author: str
    guidance: str
    created_at: float
    critique_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _safe_record_id(self.attempt_id, "attempt_id"))
        object.__setattr__(self, "author", _required(self.author, "author"))
        object.__setattr__(self, "guidance", _required(self.guidance, "guidance"))
        object.__setattr__(self, "created_at", _epoch(self.created_at, "created_at"))
        object.__setattr__(self, "critique_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": CRITIQUE_SCHEMA,
            "attempt_id": self.attempt_id,
            "author": self.author,
            "guidance": self.guidance,
            "created_at": self.created_at,
        }
        if include_id:
            value["critique_id"] = self.critique_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AttemptCritique":
        item = dict(value)
        if item.get("schema") != CRITIQUE_SCHEMA:
            raise AssignmentStoreError("serialized critique schema mismatch")
        critique_id = item.pop("critique_id", None)
        item.pop("schema", None)
        try:
            record = cls(**item)
        except TypeError as exc:
            raise AssignmentStoreError("serialized critique fields are invalid") from exc
        if record.critique_id != critique_id:
            raise AssignmentStoreError("critique identity mismatch")
        return record


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    assignments: int
    attempts: int
    critiques: int
    events: int
    finalized_assignments: tuple[str, ...] = ()
    finalized_attempts: tuple[str, ...] = ()
    recovered_tmp_files: tuple[str, ...] = ()
    removed_tmp_files: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CommandContext:
    record: TrainingAssignment
    events: tuple[dict[str, Any], ...]
    prior: dict[str, Any] | None


class AssignmentStore:
    """File-backed, crash-recoverable assignment lifecycle store."""

    def __init__(self, root: Path | str, *, clock: Callable[[], float] | None = None) -> None:
        self.root = Path(root).resolve(strict=False)
        self.assignments_dir = self.root / "assignments"
        self.attempts_dir = self.root / "attempts"
        self.critiques_dir = self.root / "critiques"
        self.events_dir = self.root / "events"
        self._clock = clock if clock is not None else time.time

    @classmethod
    def active(cls, state_root: Path | str, *, clock: Callable[[], float] | None = None) -> "AssignmentStore":
        return cls(Path(state_root).resolve(strict=False) / "training" / "assignments", clock=clock)

    def _now(self) -> float:
        return _epoch(self._clock(), "clock")

    # ------------------------------------------------------------------ reads

    def get_assignment(self, assignment_id: str) -> TrainingAssignment:
        return self._load_assignment(_safe_record_id(assignment_id, "assignment_id"))

    def get_attempt(self, attempt_id: str) -> AssignmentAttempt:
        return self._load_attempt(_safe_record_id(attempt_id, "attempt_id"))

    def attempts_for(self, assignment_id: str) -> tuple[AssignmentAttempt, ...]:
        safe_id = _safe_record_id(assignment_id, "assignment_id")
        path = self.assignments_dir / f"{safe_id}.json"
        if not path.is_file():
            raise AssignmentStoreError(f"assignment record is missing: {safe_id}")
        attempts = [
            self._load_attempt_from_path(candidate)
            for candidate in sorted(self.attempts_dir.glob("*.json"))
        ]
        return tuple(
            sorted(
                (attempt for attempt in attempts if attempt.assignment_id == safe_id),
                key=lambda attempt: (attempt.attempt_index, attempt.attempt_id),
            )
        )

    def critiques_for(self, attempt_id: str) -> tuple[AttemptCritique, ...]:
        safe_id = _safe_record_id(attempt_id, "attempt_id")
        critiques = [
            AttemptCritique.from_mapping(_read_json(path))
            for path in sorted(self.critiques_dir.glob("*.json"))
        ]
        return tuple(critique for critique in critiques if critique.attempt_id == safe_id)

    def events_for(self, assignment_id: str) -> tuple[dict[str, Any], ...]:
        return self._load_events(_safe_record_id(assignment_id, "assignment_id"))

    # --------------------------------------------------------------- commands

    def create_assignment(
        self,
        *,
        kind: AssignmentKind | str,
        curriculum_ref: str,
        cohort_eligibility: Mapping[str, Any],
        field_binding: Mapping[str, Any],
        holder_core_id: str,
        origin: str,
        lease_seconds: float,
        idempotency_key: str,
        failure_threshold: int = 3,
        created_at: float | None = None,
        target_text: str | None = None,
    ) -> TrainingAssignment:
        """Claim a new assignment: born CLAIMED with a durable holder lease.

        ``target_text`` optionally binds the explicit expected target of the
        work object; it is part of the assignment's durable identity.
        """

        command = "create_assignment"
        key = _required(idempotency_key, "idempotency_key")
        holder = _required(holder_core_id, "holder_core_id")
        params = {
            "kind": str(kind.value if isinstance(kind, AssignmentKind) else kind),
            "curriculum_ref": curriculum_ref,
            "cohort_eligibility": dict(cohort_eligibility),
            "field_binding": dict(field_binding),
            "holder_core_id": holder,
            "origin": origin,
            "lease_seconds": lease_seconds,
            "failure_threshold": failure_threshold,
            "created_at": created_at,
        }
        if target_text is not None:
            params["target_text"] = target_text
        created = self._now() if created_at is None else _epoch(created_at, "created_at")
        params["created_at"] = created
        record = TrainingAssignment(
            kind=kind,
            curriculum_ref=curriculum_ref,
            cohort_eligibility=cohort_eligibility,
            field_binding=field_binding,
            origin=origin,
            created_at=created,
            failure_threshold=failure_threshold,
            lease=self._issue_lease(holder, created, lease_seconds),
            target_text=target_text,
        )
        self._check_cohort(record, holder)
        prior = self._find_prior_event(self._load_events(record.assignment_id), key, command, params)
        if prior is not None:
            stored = TrainingAssignment.from_mapping(prior["assignment"])
            path = self.assignments_dir / f"{stored.assignment_id}.json"
            if not path.is_file():
                self._write_assignment_atomic(stored)
            return stored
        event = self._event_payload(
            record.assignment_id,
            key,
            command,
            params,
            event="assignment_claimed",
            record=record,
        )
        self._append_event(record.assignment_id, event)
        self._write_assignment_atomic(record)
        return record

    def activate(
        self,
        assignment_id: str,
        *,
        holder_core_id: str,
        expected_revision: int,
        idempotency_key: str,
        reason: str | None = None,
    ) -> TrainingAssignment:
        """CLAIMED -> ACTIVE: the holder begins work on its lease."""

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            return replace(record)

        return self._transition(
            assignment_id,
            command="activate",
            idempotency_key=idempotency_key,
            params={"holder_core_id": holder_core_id, "reason": reason},
            expected_revision=expected_revision,
            allowed_from=(AssignmentStatus.CLAIMED,),
            to_status=AssignmentStatus.ACTIVE,
            holder_core_id=holder_core_id,
            require_live_lease=True,
            mutate=mutate,
            reason=reason,
        )

    def pause(
        self,
        assignment_id: str,
        *,
        holder_core_id: str,
        expected_revision: int,
        idempotency_key: str,
        reason: str | None = None,
    ) -> TrainingAssignment:
        """ACTIVE -> PAUSED: disconnection or holder pause; lease is released."""

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            return replace(record, lease=None)

        return self._transition(
            assignment_id,
            command="pause",
            idempotency_key=idempotency_key,
            params={"holder_core_id": holder_core_id, "reason": reason},
            expected_revision=expected_revision,
            allowed_from=(AssignmentStatus.ACTIVE,),
            to_status=AssignmentStatus.PAUSED,
            holder_core_id=holder_core_id,
            mutate=mutate,
            reason=reason,
        )

    def resume(
        self,
        assignment_id: str,
        *,
        core_id: str,
        expected_revision: int,
        idempotency_key: str,
        lease_seconds: float,
        reason: str | None = None,
    ) -> TrainingAssignment:
        """PAUSED -> ACTIVE: a cohort core re-claims with a fresh lease."""

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            lease = self._issue_lease(core_id, self._now(), lease_seconds)
            return replace(record, lease=lease)

        self._check_cohort(self._load_assignment(_safe_record_id(assignment_id, "assignment_id")), core_id)
        return self._transition(
            assignment_id,
            command="resume",
            idempotency_key=idempotency_key,
            params={"core_id": core_id, "lease_seconds": lease_seconds, "reason": reason},
            expected_revision=expected_revision,
            allowed_from=(AssignmentStatus.PAUSED,),
            to_status=AssignmentStatus.ACTIVE,
            mutate=mutate,
            reason=reason,
        )

    def complete(
        self,
        assignment_id: str,
        *,
        holder_core_id: str,
        expected_revision: int,
        idempotency_key: str,
        reason: str | None = None,
    ) -> TrainingAssignment:
        """ACTIVE -> COMPLETED: closes the work object; asserts no competence."""

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            return replace(record, lease=None)

        return self._transition(
            assignment_id,
            command="complete",
            idempotency_key=idempotency_key,
            params={"holder_core_id": holder_core_id, "reason": reason},
            expected_revision=expected_revision,
            allowed_from=(AssignmentStatus.ACTIVE,),
            to_status=AssignmentStatus.COMPLETED,
            holder_core_id=holder_core_id,
            require_live_lease=True,
            mutate=mutate,
            reason=reason,
        )

    def escalate(
        self,
        assignment_id: str,
        *,
        author: str,
        expected_revision: int,
        idempotency_key: str,
        reason: str,
    ) -> TrainingAssignment:
        """CLAIMED/ACTIVE/PAUSED -> ESCALATED: supervisor inspection required."""

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            return replace(record, lease=None)

        return self._transition(
            assignment_id,
            command="escalate",
            idempotency_key=idempotency_key,
            params={"author": author, "reason": reason},
            expected_revision=expected_revision,
            allowed_from=(AssignmentStatus.CLAIMED, AssignmentStatus.ACTIVE, AssignmentStatus.PAUSED),
            to_status=AssignmentStatus.ESCALATED,
            mutate=mutate,
            reason=reason,
            author=author,
        )

    def renew_lease(
        self,
        assignment_id: str,
        *,
        holder_core_id: str,
        expected_revision: int,
        idempotency_key: str,
        lease_seconds: float,
    ) -> TrainingAssignment:
        """Extend the holder lease in place (epoch increments)."""

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            assert record.lease is not None
            lease = AssignmentLease(
                holder_core_id=record.lease.holder_core_id,
                issued_at=self._now(),
                expires_at=self._now() + _finite(lease_seconds, "lease_seconds"),
                lease_epoch=record.lease.lease_epoch + 1,
            )
            return replace(record, lease=lease)

        return self._transition(
            assignment_id,
            command="renew_lease",
            idempotency_key=idempotency_key,
            params={"holder_core_id": holder_core_id, "lease_seconds": lease_seconds},
            expected_revision=expected_revision,
            allowed_from=_CORE_STATUSES,
            to_status=None,
            holder_core_id=holder_core_id,
            require_live_lease=True,
            mutate=mutate,
            reason="lease_renewed",
        )

    def reclaim_expired(
        self,
        assignment_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        reason: str | None = None,
    ) -> TrainingAssignment:
        """Lease-expiry reclaim: CLAIMED/ACTIVE -> PAUSED; attempts preserved."""

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            return replace(record, lease=None)

        return self._transition(
            assignment_id,
            command="reclaim_expired",
            idempotency_key=idempotency_key,
            params={"reason": reason},
            expected_revision=expected_revision,
            allowed_from=_CORE_STATUSES,
            to_status=AssignmentStatus.PAUSED,
            require_expired_lease=True,
            mutate=mutate,
            reason=reason or "lease_expired",
        )

    def preempt(
        self,
        assignment_id: str,
        *,
        author: str,
        expected_revision: int,
        idempotency_key: str,
        reason: str,
    ) -> TrainingAssignment:
        """Supervisor preemption to PAUSED without waiting for lease expiry."""

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            return replace(record, lease=None)

        return self._transition(
            assignment_id,
            command="preempt",
            idempotency_key=idempotency_key,
            params={"author": author, "reason": reason},
            expected_revision=expected_revision,
            allowed_from=(AssignmentStatus.CLAIMED, AssignmentStatus.ACTIVE, AssignmentStatus.PAUSED),
            to_status=AssignmentStatus.PAUSED,
            mutate=mutate,
            reason=reason,
            author=author,
            allow_escalated=True,
        )

    def adjust_curriculum(
        self,
        assignment_id: str,
        *,
        author: str,
        expected_revision: int,
        idempotency_key: str,
        curriculum_ref: str | None = None,
        field_binding: Mapping[str, Any] | None = None,
        reason: str,
    ) -> TrainingAssignment:
        """Supervisor revision of curriculum reference and/or field binding."""

        if curriculum_ref is None and field_binding is None:
            raise ValueError("adjust_curriculum requires curriculum_ref and/or field_binding")

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            binding = record.field_binding if field_binding is None else _mapping(field_binding, "field_binding")
            if field_binding is not None:
                for label in ("field_id", "view_id"):
                    _required(str(binding.get(label, "")), f"field_binding {label}")
            return replace(
                record,
                curriculum_ref=record.curriculum_ref if curriculum_ref is None else _required(curriculum_ref, "curriculum_ref"),
                field_binding=binding,
            )

        return self._transition(
            assignment_id,
            command="adjust_curriculum",
            idempotency_key=idempotency_key,
            params={
                "author": author,
                "curriculum_ref": curriculum_ref,
                "field_binding": None if field_binding is None else dict(field_binding),
                "reason": reason,
            },
            expected_revision=expected_revision,
            allowed_from=tuple(status for status in AssignmentStatus if status is not AssignmentStatus.COMPLETED),
            to_status=None,
            mutate=mutate,
            reason=reason,
            author=author,
            allow_escalated=True,
        )

    def rollback(
        self,
        assignment_id: str,
        *,
        author: str,
        expected_revision: int,
        idempotency_key: str,
        parameter_generation: str,
        optimizer_generation: str,
        reason: str,
    ) -> TrainingAssignment:
        """Supervisor rollback marker to exact prior parameter/optimizer generations."""

        def mutate(record: TrainingAssignment) -> TrainingAssignment:
            return replace(
                record,
                rollback_target={
                    "parameter_generation": _required(parameter_generation, "parameter_generation"),
                    "optimizer_generation": _required(optimizer_generation, "optimizer_generation"),
                    "reason": _required(reason, "reason"),
                },
            )

        return self._transition(
            assignment_id,
            command="rollback",
            idempotency_key=idempotency_key,
            params={
                "author": author,
                "parameter_generation": parameter_generation,
                "optimizer_generation": optimizer_generation,
                "reason": reason,
            },
            expected_revision=expected_revision,
            allowed_from=tuple(status for status in AssignmentStatus if status is not AssignmentStatus.COMPLETED),
            to_status=None,
            mutate=mutate,
            reason=reason,
            author=author,
            allow_escalated=True,
        )

    def record_attempt(
        self,
        assignment_id: str,
        *,
        core_id: str,
        emission_ref: Mapping[str, Any],
        soul_lineage: Mapping[str, Any],
        parameter_generation: str,
        optimizer_generation: str,
        evidence: Mapping[str, Any],
        expected_revision: int,
        idempotency_key: str,
    ) -> AssignmentAttempt:
        """Commit one typed attempt while the assignment is ACTIVE and leased."""

        command = "record_attempt"
        key = _required(idempotency_key, "idempotency_key")
        params = {
            "core_id": core_id,
            "emission_ref": dict(emission_ref),
            "soul_lineage": dict(soul_lineage),
            "parameter_generation": parameter_generation,
            "optimizer_generation": optimizer_generation,
            "evidence": dict(evidence),
        }
        safe_id = _safe_record_id(assignment_id, "assignment_id")
        record = self._load_assignment(safe_id)
        events = self._load_events(safe_id)
        prior = self._find_prior_event(events, key, command, params)
        if prior is not None:
            self._finalize_attempt_event(record, prior)
            return AssignmentAttempt.from_mapping(prior["attempt"])
        self._check_expected_revision(record, expected_revision, prior)
        self._check_live_holder(record, core_id)
        if record.status != AssignmentStatus.ACTIVE:
            raise AssignmentStoreError(
                f"attempts may only be recorded while ACTIVE (status={record.status.value})"
            )
        attempt_index = sum(1 for event in events if event.get("event") == "attempt_recorded")
        attempt = AssignmentAttempt(
            assignment_id=safe_id,
            core_id=_required(core_id, "core_id"),
            attempt_index=attempt_index,
            emission_ref=emission_ref,
            soul_lineage=soul_lineage,
            parameter_generation=parameter_generation,
            optimizer_generation=optimizer_generation,
            evidence=evidence,
        )
        updated = replace(record, revision=record.revision + 1, attempt_count=attempt_index + 1)
        event = self._event_payload(
            safe_id,
            key,
            command,
            params,
            event="attempt_recorded",
            record=updated,
            attempt=attempt,
        )
        _write_content_addressed(self.attempts_dir / f"{attempt.attempt_id}.json", attempt.to_canonical_dict())
        self._append_event(safe_id, event)
        self._write_assignment_atomic(updated)
        return attempt

    def record_attempt_outcome(
        self,
        attempt_id: str,
        *,
        outcome: AttemptOutcome | str,
        evidence: Mapping[str, Any] | None = None,
        expected_revision: int,
        idempotency_key: str,
    ) -> tuple[AssignmentAttempt, TrainingAssignment]:
        """Record pass/fail/counterfactual-only; a failed series escalates.

        Timeout and lease state never influence outcomes; only an explicit,
        idempotently keyed command records an outcome.
        """

        command = "record_attempt_outcome"
        key = _required(idempotency_key, "idempotency_key")
        outcome = outcome if isinstance(outcome, AttemptOutcome) else AttemptOutcome(outcome)
        if outcome is AttemptOutcome.PENDING:
            raise ValueError("outcome must be pass, fail, or counterfactual_only")
        params = {
            "outcome": outcome.value,
            "evidence": None if evidence is None else dict(evidence),
        }
        attempt = self._load_attempt(_safe_record_id(attempt_id, "attempt_id"))
        record = self._load_assignment(attempt.assignment_id)
        events = self._load_events(attempt.assignment_id)
        prior = self._find_prior_event(events, key, command, params)
        if prior is not None:
            finalized_record = self._finalize_outcome_event(record, prior)
            return AssignmentAttempt.from_mapping(prior["attempt"]), finalized_record
        self._check_expected_revision(record, expected_revision, prior)
        if attempt.outcome is not AttemptOutcome.PENDING:
            raise AssignmentStoreError(
                f"attempt {attempt.attempt_id} already has outcome {attempt.outcome.value}"
            )
        updated_attempt = replace(
            attempt,
            outcome=outcome,
            evidence=attempt.evidence if evidence is None else _mapping(evidence, "evidence"),
            revision=attempt.revision + 1,
        )
        by_id = {item.attempt_id: item for item in self.attempts_for(attempt.assignment_id)}
        by_id[updated_attempt.attempt_id] = updated_attempt
        attempts = sorted(by_id.values(), key=lambda item: (item.attempt_index, item.attempt_id))
        consecutive = 0
        for item in reversed(attempts):
            # Counterfactual-only attempts are ablation probes: transparent to
            # the failure series (they neither count nor reset it).
            if item.outcome is AttemptOutcome.FAIL:
                consecutive += 1
            elif item.outcome is AttemptOutcome.PASS:
                break
        status = record.status
        escalation_reason = None
        if consecutive >= record.failure_threshold and record.status is AssignmentStatus.ACTIVE:
            status = AssignmentStatus.ESCALATED
            escalation_reason = (
                f"failed series: {consecutive} consecutive failures "
                f">= threshold {record.failure_threshold}"
            )
        updated = replace(
            record,
            status=status,
            attempt_count=len(attempts),
            consecutive_failures=consecutive,
            revision=record.revision + 1,
            lease=None if status is AssignmentStatus.ESCALATED else record.lease,
        )
        event = self._event_payload(
            attempt.assignment_id,
            key,
            command,
            params,
            event="attempt_outcome",
            record=updated,
            attempt=updated_attempt,
            reason=escalation_reason,
            author="trainer:failure-series" if escalation_reason else None,
        )
        _atomic_json(self.attempts_dir / f"{attempt.attempt_id}.json", updated_attempt.to_canonical_dict())
        self._append_event(attempt.assignment_id, event)
        self._write_assignment_atomic(updated)
        return updated_attempt, updated

    def add_critique(
        self,
        attempt_id: str,
        *,
        author: str,
        guidance: str,
        idempotency_key: str,
        created_at: float | None = None,
    ) -> AttemptCritique:
        """Attach a supervisor critique to one exact attempt."""

        command = "add_critique"
        key = _required(idempotency_key, "idempotency_key")
        safe_id = _safe_record_id(attempt_id, "attempt_id")
        attempt = self._load_attempt(safe_id)
        params = {
            "author": author,
            "guidance": guidance,
            "created_at": created_at,
        }
        when = self._now() if created_at is None else _epoch(created_at, "created_at")
        params["created_at"] = when
        events = self._load_events(attempt.assignment_id)
        prior = self._find_prior_event(events, key, command, params)
        if prior is not None:
            return AttemptCritique.from_mapping(prior["critique"])
        critique = AttemptCritique(attempt_id=safe_id, author=author, guidance=guidance, created_at=when)
        event = self._event_payload(
            attempt.assignment_id,
            key,
            command,
            params,
            event="critique_added",
            record=None,
            critique=critique,
        )
        _write_content_addressed(
            self.critiques_dir / f"{critique.critique_id}.json", critique.to_canonical_dict()
        )
        self._append_event(attempt.assignment_id, event)
        return critique

    # --------------------------------------------------------------- recovery

    def recover(self) -> RecoveryReport:
        """Finalize interrupted atomic writes and verify every stored record.

        A crash can leave a valid temp file whose os.replace never ran, or a
        journaled intent whose record write never landed.  Both are completed
        here; corrupt temp files are removed, never promoted.
        """

        recovered_tmp: list[str] = []
        removed_tmp: list[str] = []
        for temporary in sorted(self.root.rglob("*.tmp")):
            target = temporary.with_name(temporary.name[: -len(f".{temporary.name.split('.')[-2]}.tmp")])
            promoted = False
            try:
                value = json.loads(temporary.read_text(encoding="utf-8"))
                if isinstance(value, dict) and self._tmp_matches_target(target, value):
                    _atomic_json(target, value)
                    temporary.unlink()
                    recovered_tmp.append(temporary.name)
                    promoted = True
            except (OSError, json.JSONDecodeError, AssignmentStoreError, ValueError):
                promoted = False
            if not promoted:
                try:
                    temporary.unlink()
                except OSError:
                    pass
                removed_tmp.append(temporary.name)

        finalized_assignments: list[str] = []
        finalized_attempts: list[str] = []
        assignment_records: dict[str, TrainingAssignment] = {}
        for path in sorted(self.assignments_dir.glob("*.json")):
            record = TrainingAssignment.from_mapping(_read_json(path))
            if record.assignment_id != path.stem:
                raise AssignmentStoreError(f"assignment record path/identity mismatch: {path}")
            assignment_records[record.assignment_id] = record
        attempt_records: dict[str, AssignmentAttempt] = {}
        for path in sorted(self.attempts_dir.glob("*.json")):
            record = AssignmentAttempt.from_mapping(_read_json(path))
            if record.attempt_id != path.stem:
                raise AssignmentStoreError(f"attempt record path/identity mismatch: {path}")
            attempt_records[record.attempt_id] = record
        critique_records: dict[str, AttemptCritique] = {}
        for path in sorted(self.critiques_dir.glob("*.json")):
            record = AttemptCritique.from_mapping(_read_json(path))
            if record.critique_id != path.stem:
                raise AssignmentStoreError(f"critique record path/identity mismatch: {path}")
            critique_records[record.critique_id] = record

        event_count = 0
        journaled_ids = {
            path.stem for path in sorted(self.events_dir.glob("*.jsonl"))
        }
        for assignment_id in sorted(set(assignment_records) | journaled_ids):
            events = self._load_events(assignment_id)
            event_count += len(events)
            last_assignment: dict[str, Any] | None = None
            last_attempts: dict[str, dict[str, Any]] = {}
            for event in events:
                if event.get("assignment") is not None:
                    last_assignment = event["assignment"]
                if event.get("attempt") is not None:
                    last_attempts[event["attempt"]["attempt_id"]] = event["attempt"]
            current = assignment_records.get(assignment_id)
            if last_assignment is not None:
                on_disk = None if current is None else current.to_canonical_dict()
                if on_disk is None or canonical_json_bytes(last_assignment) != canonical_json_bytes(on_disk):
                    finalized = TrainingAssignment.from_mapping(last_assignment)
                    self._write_assignment_atomic(finalized)
                    finalized_assignments.append(assignment_id)
                    assignment_records[assignment_id] = finalized
            elif current is None:
                raise AssignmentStoreError(
                    f"assignment {assignment_id} has a journal but no record and no journaled intent"
                )
            for attempt_id, payload in last_attempts.items():
                current = attempt_records.get(attempt_id)
                if current is None or canonical_json_bytes(current.to_canonical_dict()) != canonical_json_bytes(payload):
                    finalized_attempt = AssignmentAttempt.from_mapping(payload)
                    _atomic_json(self.attempts_dir / f"{attempt_id}.json", finalized_attempt.to_canonical_dict())
                    finalized_attempts.append(attempt_id)
                    attempt_records[attempt_id] = finalized_attempt
        return RecoveryReport(
            assignments=len(assignment_records),
            attempts=len(attempt_records),
            critiques=len(critique_records),
            events=event_count,
            finalized_assignments=tuple(finalized_assignments),
            finalized_attempts=tuple(finalized_attempts),
            recovered_tmp_files=tuple(recovered_tmp),
            removed_tmp_files=tuple(removed_tmp),
        )

    # ----------------------------------------------------------------- internals

    def _issue_lease(self, holder: str, issued_at: float, lease_seconds: float) -> AssignmentLease:
        seconds = _finite(lease_seconds, "lease_seconds")
        if seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        return AssignmentLease(
            holder_core_id=_required(holder, "holder_core_id"),
            issued_at=issued_at,
            expires_at=issued_at + seconds,
            lease_epoch=1,
        )

    def _check_cohort(self, record: TrainingAssignment, core_id: str) -> None:
        core_ids = record.cohort_eligibility.get("core_ids")
        if core_ids is not None and core_id not in core_ids:
            raise AssignmentStoreError(f"core {core_id} is not eligible for assignment cohort")

    def _check_live_holder(self, record: TrainingAssignment, core_id: str) -> None:
        if record.lease is None:
            raise AssignmentStoreError("assignment has no active lease")
        if record.lease.holder_core_id != core_id:
            raise AssignmentStoreError(f"core {core_id} does not hold the assignment lease")
        if record.lease.expired(self._now()):
            raise AssignmentStoreError("assignment lease is expired; reclaim before continuing")

    def _check_expected_revision(
        self,
        record: TrainingAssignment,
        expected_revision: int,
        prior: dict[str, Any] | None,
    ) -> None:
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        if record.revision != expected_revision and prior is None:
            raise AssignmentStoreError(
                f"stale revision for assignment {record.assignment_id}: "
                f"expected {expected_revision}, found {record.revision}"
            )

    def _transition(
        self,
        assignment_id: str,
        *,
        command: str,
        idempotency_key: str,
        params: Mapping[str, Any],
        expected_revision: int,
        allowed_from: tuple[AssignmentStatus, ...],
        to_status: AssignmentStatus | None,
        mutate: Callable[[TrainingAssignment], TrainingAssignment],
        reason: str | None,
        holder_core_id: str | None = None,
        author: str | None = None,
        require_live_lease: bool = False,
        require_expired_lease: bool = False,
        allow_escalated: bool = False,
    ) -> TrainingAssignment:
        key = _required(idempotency_key, "idempotency_key")
        safe_id = _safe_record_id(assignment_id, "assignment_id")
        record = self._load_assignment(safe_id)
        events = self._load_events(safe_id)
        prior = self._find_prior_event(events, key, command, params)
        if prior is not None:
            return self._finalize_assignment_event(record, prior)
        self._check_expected_revision(record, expected_revision, prior)
        # COMPLETED is terminal for everyone.  ESCALATED is terminal for core
        # work but remains open to narrow supervisor inspection actions.
        if record.status is AssignmentStatus.COMPLETED or (
            record.status is AssignmentStatus.ESCALATED and not allow_escalated
        ):
            raise AssignmentStoreError(
                f"assignment {safe_id} is terminal ({record.status.value})"
            )
        if record.status not in allowed_from:
            raise AssignmentStoreError(
                f"cannot {command} from status {record.status.value}"
            )
        if holder_core_id is not None:
            if record.lease is None or record.lease.holder_core_id != holder_core_id:
                raise AssignmentStoreError(f"core {holder_core_id} does not hold the assignment lease")
        if require_live_lease:
            self._check_live_holder(record, record.lease.holder_core_id if record.lease else "")
        if require_expired_lease:
            if record.lease is None:
                raise AssignmentStoreError("assignment has no lease to reclaim")
            if not record.lease.expired(self._now()):
                raise AssignmentStoreError("assignment lease has not expired")
        updated = mutate(record)
        updated = replace(
            updated,
            status=record.status if to_status is None else to_status,
            revision=record.revision + 1,
        )
        event = self._event_payload(
            safe_id,
            key,
            command,
            dict(params),
            event=f"assignment_{command}",
            record=updated,
            reason=reason,
            author=author,
        )
        self._write_assignment_atomic(updated)
        self._append_event(safe_id, event)
        return updated

    def _finalize_assignment_event(
        self, record: TrainingAssignment, event: Mapping[str, Any]
    ) -> TrainingAssignment:
        """Complete or return the outcome of an already-journaled command."""

        payload = event.get("assignment")
        if not isinstance(payload, dict):
            raise AssignmentStoreError("journaled assignment command has no record payload")
        finalized = TrainingAssignment.from_mapping(payload)
        if record.revision == finalized.revision and canonical_json_bytes(
            record.to_canonical_dict()
        ) == canonical_json_bytes(finalized.to_canonical_dict()):
            return record
        if record.revision not in (event.get("from_revision"), finalized.revision):
            raise AssignmentStoreError(
                "assignment revision diverged from its journaled command"
            )
        self._write_assignment_atomic(finalized)
        return finalized

    def _finalize_attempt_event(
        self, record: TrainingAssignment, event: Mapping[str, Any]
    ) -> None:
        payload = event.get("assignment")
        if not isinstance(payload, dict):
            raise AssignmentStoreError("journaled record_attempt command has no record payload")
        finalized = TrainingAssignment.from_mapping(payload)
        if record.revision != finalized.revision:
            self._write_assignment_atomic(finalized)

    def _finalize_outcome_event(
        self, record: TrainingAssignment, event: Mapping[str, Any]
    ) -> TrainingAssignment:
        payload = event.get("assignment")
        if not isinstance(payload, dict):
            raise AssignmentStoreError("journaled outcome command has no record payload")
        finalized = TrainingAssignment.from_mapping(payload)
        if record.revision != finalized.revision:
            self._write_assignment_atomic(finalized)
        return finalized

    def _load_assignment(self, assignment_id: str) -> TrainingAssignment:
        path = self.assignments_dir / f"{assignment_id}.json"
        if not path.is_file():
            raise AssignmentStoreError(f"assignment record is missing: {assignment_id}")
        record = TrainingAssignment.from_mapping(_read_json(path))
        if record.assignment_id != assignment_id:
            raise AssignmentStoreError("assignment record path/identity mismatch")
        return record

    def _load_attempt(self, attempt_id: str) -> AssignmentAttempt:
        return self._load_attempt_from_path(self.attempts_dir / f"{attempt_id}.json")

    def _load_attempt_from_path(self, path: Path) -> AssignmentAttempt:
        if not path.is_file():
            raise AssignmentStoreError(f"attempt record is missing: {path.name}")
        record = AssignmentAttempt.from_mapping(_read_json(path))
        if record.attempt_id != path.stem:
            raise AssignmentStoreError("attempt record path/identity mismatch")
        return record

    def _journal_path(self, assignment_id: str) -> Path:
        return self.events_dir / f"{assignment_id}.jsonl"

    def _load_events(self, assignment_id: str) -> tuple[dict[str, Any], ...]:
        path = self._journal_path(assignment_id)
        if not path.is_file():
            return ()
        events: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AssignmentStoreError(f"cannot read assignment journal: {path}") from exc
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssignmentStoreError(f"corrupt assignment journal line: {path}") from exc
            if not isinstance(event, dict) or event.get("schema") != EVENT_SCHEMA:
                raise AssignmentStoreError("assignment journal event schema mismatch")
            event_id = event.pop("event_id", None)
            if canonical_sha256(event) != event_id:
                raise AssignmentStoreError("assignment journal event identity mismatch")
            event["event_id"] = event_id
            if event.get("assignment_id") != assignment_id:
                raise AssignmentStoreError("assignment journal event belongs to another assignment")
            events.append(event)
        return tuple(events)

    def _find_prior_event(
        self,
        events: tuple[dict[str, Any], ...],
        key: str,
        command: str,
        params: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        matches = [
            event
            for event in events
            if event.get("idempotency_key") == key
        ]
        if not matches:
            return None
        command_sha = canonical_sha256({"command": command, "params": dict(params)})
        for event in matches:
            if event.get("command") != command or event.get("command_sha256") != command_sha:
                raise AssignmentStoreError(
                    f"idempotency key {key!r} was already used with a different command"
                )
        return matches[-1]

    def _event_payload(
        self,
        assignment_id: str,
        key: str,
        command: str,
        params: Mapping[str, Any],
        *,
        event: str,
        record: TrainingAssignment | None,
        attempt: AssignmentAttempt | None = None,
        critique: AttemptCritique | None = None,
        reason: str | None = None,
        author: str | None = None,
    ) -> dict[str, Any]:
        from_revision = record.revision - 1 if record is not None and record.revision > 0 else None
        return {
            "schema": EVENT_SCHEMA,
            "event": event,
            "assignment_id": assignment_id,
            "idempotency_key": key,
            "command": command,
            "command_sha256": canonical_sha256({"command": command, "params": dict(params)}),
            "from_revision": from_revision,
            "to_revision": None if record is None else record.revision,
            "from_status": None,
            "to_status": None if record is None else record.status.value,
            "author": author,
            "reason": reason,
            "at": self._now(),
            "assignment": None if record is None else record.to_canonical_dict(),
            "attempt": None if attempt is None else attempt.to_canonical_dict(),
            "critique": None if critique is None else critique.to_canonical_dict(),
        }

    def _append_event(self, assignment_id: str, payload: Mapping[str, Any]) -> None:
        event = dict(payload)
        event["event_id"] = canonical_sha256(event)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        with self._journal_path(assignment_id).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_assignment_atomic(self, record: TrainingAssignment) -> None:
        _atomic_json(self.assignments_dir / f"{record.assignment_id}.json", record.to_canonical_dict())

    def _tmp_matches_target(self, target: Path, value: Mapping[str, Any]) -> bool:
        schema = value.get("schema")
        if target.parent == self.assignments_dir and schema == ASSIGNMENT_SCHEMA:
            return TrainingAssignment.from_mapping(value).assignment_id == target.stem
        if target.parent == self.attempts_dir and schema == ATTEMPT_SCHEMA:
            return AssignmentAttempt.from_mapping(value).attempt_id == target.stem
        if target.parent == self.critiques_dir and schema == CRITIQUE_SCHEMA:
            return AttemptCritique.from_mapping(value).critique_id == target.stem
        return False


__all__ = [
    "ASSIGNMENT_SCHEMA",
    "ATTEMPT_SCHEMA",
    "CRITIQUE_SCHEMA",
    "EVENT_SCHEMA",
    "LEASE_SCHEMA",
    "AssignmentAttempt",
    "AssignmentKind",
    "AssignmentLease",
    "AssignmentStatus",
    "AssignmentStore",
    "AssignmentStoreError",
    "AttemptCritique",
    "AttemptOutcome",
    "RecoveryReport",
    "TrainingAssignment",
]

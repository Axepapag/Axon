"""Heart host: the coordinator and commit choke point of the permanent Heart.

This module composes the already-tested Heart organs; it never edits them.

* ``runtime.heart.lease`` supplies the OS-backed single-writer lease.  The host
  refuses every canonical write unless that lease is held by this process,
  fencing the branch's own permissive ``commit`` at the host choke point.
* ``runtime.heart.valve`` supplies the sovereign admission plane.  One beat
  resets the renewable per-beat budgets *first*, then admits staged ingress, so
  the four CAPPED primitive valves can never permanently exhaust their
  allocation.
* ``runtime.heart.durable_ingress`` supplies the durable FIFO spool.  Native-95
  admission happens before ``spool.submit``; rejected payloads are recorded in
  the spool's rejection/quarantine logs and never enter the FIFO.
* ``runtime.heart.authority`` supplies the ratified authority classes.  Cores
  hold ``AuthorityGrant.core(...)`` and submit proposals only through
  ``HeartHost.submit_proposal``; a rotating consolidator roster is the only
  canonical committer and ``HeartHost.commit`` enforces membership.
* ``runtime.field`` supplies the canonical field, typed deltas and append-only
  branch journal.  ``commit_id`` is the committed ``FieldDelta``'s
  ``delta_id``; ``generation`` is the branch HEAD generation.
* ``runtime.heart.health`` and ``runtime.heart.identity`` supply durable
  cardiac identity and health evidence.
* ``substrate.native`` is the only character authority: everything inside the
  system is exactly the 95 native characters and nothing is converted.

One beat runs the same ordered path every time::

    reset renewable valve budgets -> admit staged ingress -> commit pending
    proposals with the current consolidator -> drain the durable spool FIFO ->
    rotate the roster -> journal health -> save the checkpoint -> acknowledge
    -> record Lab events

The checkpoint boundary sits deliberately *after* the journal/health lines are
durable and *before* the acknowledgements, so a crash inside the ack window is
replayed by submission id from the branch journal instead of duplicating
canonical text.  The one narrow exception is a recovered record that fails the
final native gate: the durable FIFO head must be acknowledged before a later
record can be quarantined, so that path flushes the beat's earlier acks first.

Submission identity is content addressed::

    submission_id = sha256(canonical envelope bytes + valve definition version)

This is never a uuid4, so a crash retry of the same envelope is recognised as
the same submission and committed at most once.  (The corollary is deliberate:
the byte-identical envelope is idempotent until it is acknowledged; callers
that must send the same text twice vary the provenance.)

Three surfaces are never conflated:

* the core's numerical response state (core-owned; carried opaquely through a
  proposal for inspection only);
* the host's private draft (native-95 text, readable via ``private_draft``);
* the Heart-committed text (canonical, readable via ``committed_text``).

Training and inference use this exact session API (``propose``/``advance``/
``inspect``): there is no mode switch, so there is no inference shortcut.

The host is single-threaded by contract; the OS lease is process level.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from runtime.field import (
    CORE_WRITABLE_REGIONS,
    BranchIntegrityError,
    CanonicalStateBranch,
    FieldDelta,
    InsertText,
    LogicalRegion,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
    canonical_json_bytes,
    canonical_sha256,
    field_delta_from_canonical_dict,
    replacement_delta,
    validate_delta,
)
from runtime.source_of_truth import capacity_policy
from substrate.native import UnsupportedCharacterError, assert_native_text

from .authority import AuthorityClass, AuthorityGrant
from .durable_ingress import DurableIngressSpool, IngressRecord
from .errors import (
    AuthorityViolationError,
    DuplicateCoreError,
    HostStateError,
    LeaseDeniedError,
    PoisonEventError,
    ProposalBoardError,
    ReplayEventError,
    StaleBaseProposalError,
    UnknownCoreError,
    UnknownValveError,
    ValveSourceMismatchError,
)
from .health import HealthJournal, HeartHealth
from .identity import HeartIdentityStore
from .lease import SingleWriterLease
from .valve import (
    HeartValveRegistry,
    ValveBudget,
    ValveEnvelope,
    primitive_valve_registry,
)

CHECKPOINT_FORMAT = "axon-hearthost-checkpoint-v1"
CHECKPOINT_SIDECAR_SUFFIX = ".sha256"
SUBMISSION_SCHEMA = "axon-heart-ingress-submission-v1"
PROPOSAL_SCHEMA = "axon-heart-proposal-v1"

CONTROL_WAIT = "wait"
CONTROL_COMMIT = "commit"
CONTROL_END = "end"
CONTROL_SIGNALS = (CONTROL_WAIT, CONTROL_COMMIT, CONTROL_END)

EVENT_LOG_CAP = 1000
EVENT_TYPES = frozenset(
    {
        "beat_start",
        "beat_end",
        "valve_admission",
        "valve_reject",
        "cap_exhausted",
        "quarantine",
        "proposal_received",
        "commit",
        "ack",
        "budget_remaining",
        "lease_state",
    }
)

_ACK_ACTIVE_STATES = ("staged", "admitted", "committed")
_FINAL_GATE_REJECTIONS = (
    UnsupportedCharacterError,
    PoisonEventError,
    UnknownValveError,
    ValveSourceMismatchError,
)
_AUTO_CHECKPOINT: Any = object()
_CAPACITY_POLICY = capacity_policy()


class CheckpointIntegrityError(HostStateError):
    """A HeartHost checkpoint failed sidecar verification or format validation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_epoch_id() -> str:
    """A fresh host session identity: ``hearthost-<utc>-<hex>``."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"hearthost-{stamp}-{uuid.uuid4().hex}"


def _is_renewable_cap(reason: str) -> bool:
    """Whether a valve decision refused work for this beat only.

    The valve plane reports renewable per-beat exhaustion with reasons such as
    "valve items-per-beat budget exceeded" or "valve target-chars-per-beat work
    allocation exhausted"; a *zero* configured budget reports "is zero" and is
    permanent.  Only the renewable forms defer to the next beat.
    """

    return "budget exceeded" in reason or "exhausted" in reason


def submission_id_for_envelope(
    envelope: ValveEnvelope | IngressRecord,
    valve_version: int,
) -> str:
    """Content-addressed submission identity for one valve envelope.

    ``valve_version`` is the registered valve definition version at admission
    time; together the envelope bytes and that version form the full identity.
    """

    if isinstance(envelope, ValveEnvelope):
        fields = envelope.to_spool_dict()
    elif isinstance(envelope, IngressRecord):
        fields = {
            "valve_id": envelope.valve_id,
            "source_id": envelope.source_id,
            "payload": envelope.payload,
            "provenance": envelope.provenance,
            "envelope_type": envelope.envelope_type,
        }
    else:
        raise TypeError("submission identity requires a ValveEnvelope or IngressRecord")
    if isinstance(valve_version, bool) or not isinstance(valve_version, int) or valve_version < 0:
        raise ValueError("valve_version must be a non-negative integer")
    return canonical_sha256(
        {
            "schema": SUBMISSION_SCHEMA,
            "envelope": fields,
            "valve_version": valve_version,
        }
    )


@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    """Result of offering one valve envelope to the host.

    ``status`` is ``staged`` (durably pending admission on the next beat),
    ``duplicate`` (the same content-addressed submission is already pending or
    committed) or ``rejected`` (recorded in the spool rejection log and never
    admitted).
    """

    submission_id: str | None
    valve_id: str
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "valve_id": self.valve_id,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CommitAck:
    """Acknowledgement of one canonical commit.

    ``committer_id`` is the consolidator recorded for the commit; it is null
    only on an acknowledgement reconstructed from an unlogged HEAD commit,
    where the committer identity was lost with the audit line.
    """

    commit_id: str
    delta_id: str
    generation: int
    field_id: str
    tick_id: int
    committer_id: str | None
    submission_id: str | None = None
    proposal_id: str | None = None
    regions: tuple[str, ...] = ()
    committed_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "commit_id": self.commit_id,
            "delta_id": self.delta_id,
            "generation": self.generation,
            "field_id": self.field_id,
            "tick_id": self.tick_id,
            "committer_id": self.committer_id,
            "proposal_id": self.proposal_id,
            "regions": list(self.regions),
            "committed_text": self.committed_text,
        }


@dataclass(frozen=True, slots=True)
class BeatResult:
    """Summary of one completed beat."""

    beat_index: int
    consolidator_id: str
    generation: int
    commits: tuple[CommitAck, ...]
    skipped_submissions: tuple[str, ...]
    deferred_submissions: tuple[str, ...]
    rejected_submissions: tuple[str, ...]
    skipped_proposals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "beat_index": self.beat_index,
            "consolidator_id": self.consolidator_id,
            "generation": self.generation,
            "commits": [ack.to_dict() for ack in self.commits],
            "skipped_submissions": list(self.skipped_submissions),
            "deferred_submissions": list(self.deferred_submissions),
            "rejected_submissions": list(self.rejected_submissions),
            "skipped_proposals": list(self.skipped_proposals),
        }


@dataclass(slots=True)
class _SubmissionEntry:
    """Host bookkeeping for one not-yet-acknowledged submission."""

    submission_id: str
    valve_id: str
    source_id: str
    payload: str
    provenance: str
    envelope_type: str
    valve_version: int
    ack_state: str
    reason: str = ""
    spool_event_id: str | None = None
    commit_id: str | None = None
    generation: int | None = None

    def envelope(self) -> ValveEnvelope:
        return ValveEnvelope(
            valve_id=self.valve_id,
            source_id=self.source_id,
            payload=self.payload,
            provenance=self.provenance,
            envelope_type=self.envelope_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "valve_id": self.valve_id,
            "source_id": self.source_id,
            "payload": self.payload,
            "provenance": self.provenance,
            "envelope_type": self.envelope_type,
            "valve_version": self.valve_version,
            "ack_state": self.ack_state,
            "reason": self.reason,
            "spool_event_id": self.spool_event_id,
            "commit_id": self.commit_id,
            "generation": self.generation,
        }


@dataclass(slots=True)
class _ProposalEntry:
    """One pending core proposal, control signal or private-draft commit."""

    proposal_id: str
    core_id: str
    kind: str
    control: str | None
    delta: FieldDelta | None
    permitted_regions: frozenset[LogicalRegion]
    response_state: Any
    draft_revision: int | None
    submitted_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "core_id": self.core_id,
            "kind": self.kind,
            "control": self.control,
            "delta": None if self.delta is None else self.delta.to_canonical_dict(),
            "permitted_regions": sorted(region.value for region in self.permitted_regions),
            "response_state": self.response_state,
            "draft_revision": self.draft_revision,
            "submitted_at": self.submitted_at,
        }


class HeartHost:
    """Coordinate admission, proposal, commit, checkpoint and restart.

    The constructor only configures; ``start`` acquires the single-writer lease
    and opens the durable organs.  Every canonical write then flows through
    ``_commit_delta`` behind the lease, the current consolidator identity and
    the branch's own typed-delta validation.

    Submission id convention (important for ingress callers)::

        submission_id = sha256(canonical envelope bytes + valve version)

    Byte-identical envelopes submitted with the SAME provenance therefore
    collapse to one submission: the second attempt is reported as a duplicate
    (reason ``already_staged``/``admitted``/``committed``) and never enters the
    durable FIFO twice, until the first one is acknowledged.  Runtime ingress
    callers MUST vary ``provenance`` per real arrival (for example a turn
    counter or timestamp) so that genuinely repeated text is accepted twice;
    only a crash retry of the *same* arrival reuses the identical envelope and
    is meant to be idempotent.

    A restore from a checkpoint written before the durable HEAD (a "stale"
    checkpoint) is made consistent rather than silently stale:
    ``load_checkpoint``/``replay_from`` return a report whose ``behind_head``
    flag is true when the checkpoint lagged HEAD, and in that case they derive
    ``last_commit_id`` and the per-region cursors from the durable branch
    journal up to HEAD instead of restoring the stale ones.
    """

    def __init__(
        self,
        state_root: Path | str,
        *,
        branch_id: str = "active",
        initial_field: SharedFieldSnapshot | None = None,
        core_ids: Iterable[str] = (),
        consolidator_ids: Iterable[str] | None = None,
        valves: HeartValveRegistry | None = None,
        spool_dir: Path | str | None = None,
        checkpoint_path: Path | str | None | Any = _AUTO_CHECKPOINT,
    ) -> None:
        self.state_root = Path(state_root).resolve(strict=False)
        self.heart_dir = self.state_root / "active" / "heart"
        self.branch_id = branch_id
        self.initial_field = initial_field
        self.branch = CanonicalStateBranch.active_runtime(
            branch_id, state_root=self.state_root
        )
        self.registry = valves if valves is not None else primitive_valve_registry()
        self.spool_dir = (
            Path(spool_dir).resolve(strict=False)
            if spool_dir is not None
            else self.heart_dir / "ingress_spool"
        )
        if checkpoint_path is _AUTO_CHECKPOINT:
            self._checkpoint_path: Path | None = self.heart_dir / "host_checkpoint.json"
        elif checkpoint_path is None:
            self._checkpoint_path = None
        else:
            self._checkpoint_path = Path(checkpoint_path).resolve(strict=False)
        self._global_budget = ValveBudget(
            items_per_beat=_CAPACITY_POLICY.integer("heart.global_items_per_beat"),
            target_chars_per_beat=_CAPACITY_POLICY.integer(
                "heart.global_target_chars_per_beat"
            ),
        )

        self._cores: dict[str, AuthorityGrant] = {}
        for core_id in tuple(core_ids):
            self.register_core(core_id)

        roster = (
            tuple(consolidator_ids)
            if consolidator_ids is not None
            else (tuple(self._cores) or ("heart",))
        )
        if not roster:
            raise ValueError("the consolidator roster must be non-empty")
        for consolidator_id in roster:
            if not isinstance(consolidator_id, str) or not consolidator_id.strip():
                raise ValueError("consolidator ids must be non-empty strings")
        if len(set(roster)) != len(roster):
            raise ValueError("the consolidator roster must not repeat an id")
        self._consolidator_ids = tuple(roster)

        self._lease: SingleWriterLease | None = None
        self._lease_record: dict[str, Any] | None = None
        self._identity: HeartIdentityStore | None = None
        self._health: HealthJournal | None = None
        self._spool: DurableIngressSpool | None = None
        self._started = False

        self._staged: deque[_SubmissionEntry] = deque()
        self._submissions: dict[str, _SubmissionEntry] = {}
        self._commit_index: dict[str, str] = {}
        self._proposal_commit_index: dict[str, dict[str, Any]] = {}
        self._pending_proposals: list[_ProposalEntry] = []
        self._proposals_by_id: dict[str, _ProposalEntry] = {}
        self._draft: dict[str, Any] | None = None
        self._cursors: dict[str, str] = {}
        self._last_commit_id: str | None = None
        self._episode_ended = False
        self._beats_started = 0
        self._roster_index = 0
        self._unlogged_head_delta: FieldDelta | None = None
        self._head_cache: SharedFieldSnapshot | None = None
        self._last_restore: dict[str, Any] | None = None
        self._epoch = ""
        self._event_seq = 0
        self._events: dict[str, deque[dict[str, Any]]] = {
            name: deque(maxlen=EVENT_LOG_CAP) for name in sorted(EVENT_TYPES)
        }
        self._event_counters: dict[str, int] = {
            name: 0 for name in sorted(EVENT_TYPES)
        }

    # ------------------------------------------------------------------ setup

    def register_core(
        self,
        core_id: str,
        *,
        permitted_regions: Iterable[LogicalRegion | str] | None = None,
    ) -> AuthorityGrant:
        """Register one core and the authority scope it may propose within.

        A core holds an ``AuthorityGrant.core(...)``; registration pins the
        widest scope that core may ever present, so a proposal can never widen
        its own authority.
        """

        if not isinstance(core_id, str) or not core_id.strip():
            raise ValueError("core_id must be a non-empty string")
        core_id = core_id.strip()
        if core_id in self._cores:
            raise DuplicateCoreError(f"core {core_id!r} is already registered")
        grant = AuthorityGrant.core(permitted_regions)
        self._cores[core_id] = grant
        return grant

    def start(self) -> dict[str, Any]:
        """Acquire the single-writer lease and open the durable organs.

        Refuses to start (``LeaseDeniedError``) when any other host owns the
        lease.  Idempotent while started.  Restart reconciliation re-reads the
        branch journal and spool and acknowledges everything already committed.

        A fresh start opens a **new event epoch** ``hearthost-<utc>-<hex>``: the
        global event ordinal restarts at 1 and the retained event log is
        cleared.  ``load_checkpoint``/``replay_from`` continue the epoch stored
        in the checkpoint instead.
        """

        if self._started:
            return self._status_dict()
        lease = SingleWriterLease(self.state_root)
        record = lease.acquire()
        self._lease = lease
        self._lease_record = dict(record)
        self._epoch = _new_epoch_id()
        self._event_seq = 0
        for events in self._events.values():
            events.clear()
        try:
            if self.initial_field is not None:
                self.branch.initialize(self.initial_field)
            elif not self.branch.initialized:
                self.branch.initialize(SharedFieldSnapshot.empty())
            self._identity = HeartIdentityStore(self.heart_dir / "identity.json")
            self._identity.begin_start()
            self._health = HealthJournal(self.heart_dir)
            self._spool = DurableIngressSpool(self.spool_dir)
            self._head_cache = None
            self._started = True
            self._recover_commit_index()
            self._reconcile_spool()
        except Exception:
            self._started = False
            self._spool = None
            self._identity = None
            self._health = None
            self._lease_record = None
            self._lease = None
            lease.release()
            raise
        self._emit(
            "lease_state",
            {
                "held": True,
                "owner_token": lease.owner_token,
                "pid": os.getpid(),
            },
        )
        return self._status_dict()

    def stop(self) -> None:
        """Release the single-writer lease; idempotent."""

        lease = self._lease
        if lease is None:
            return
        if lease.is_held_by_us():
            self._emit(
                "lease_state",
                {
                    "held": False,
                    "owner_token": lease.owner_token,
                    "pid": os.getpid(),
                },
            )
        lease.release()
        self._lease = None
        self._lease_record = None
        self._started = False

    def close(self) -> None:
        """Alias for :meth:`stop`."""

        self.stop()

    # ------------------------------------------------------------- lifecycle

    def current_consolidator(self) -> str:
        """The consolidator allowed to commit: the current (or next) duty member.

        The roster rotates once per beat.  During a beat this is the member
        whose commits that beat performs; between beats it is the member on
        duty for the next beat.
        """

        return self._consolidator_ids[self._roster_index % len(self._consolidator_ids)]

    def _rotate_consolidator(self) -> None:
        self._roster_index = self._beats_started % len(self._consolidator_ids)

    @property
    def generation(self) -> int:
        self._require_started()
        return self.branch.load_head_record().generation

    @property
    def head_field_id(self) -> str:
        self._require_started()
        return self.branch.load_head_record().field_id

    @property
    def spool(self) -> DurableIngressSpool:
        if self._spool is None:
            raise HostStateError("HeartHost has not been started")
        return self._spool

    @property
    def lease(self) -> SingleWriterLease:
        if self._lease is None:
            raise HostStateError("HeartHost has not been started")
        return self._lease

    def event_counters(self) -> dict[str, int]:
        """Total Lab events emitted per type since process start."""

        return dict(self._event_counters)

    def health_latest(self) -> dict[str, Any]:
        """The latest durable cardiac health line, or an empty mapping."""

        if self._health is None:
            return {}
        return self._health.latest()

    def drain_events(self) -> list[dict[str, Any]]:
        """Return and clear the Lab event log as ONE list ordered by ``seq``.

        Each event is ``{"seq": int, "epoch": str, "at": iso, "type": str,
        "payload": {...}}``.  ``seq`` is a single global ordinal across every
        event type, starting at 1 in each epoch.  Retention stays bounded per
        type (``EVENT_LOG_CAP``); the per-type emission counters are kept.
        """

        drained = [
            dict(item) for name in sorted(self._events) for item in self._events[name]
        ]
        for name in self._events:
            self._events[name].clear()
        drained.sort(key=lambda item: item["seq"])
        return drained

    # ---------------------------------------------------------------- session

    def propose(
        self,
        core_id: str,
        *,
        delta: FieldDelta | None = None,
        grant: AuthorityGrant | None = None,
        control: str | None = None,
        commit_draft: bool = False,
        response_state: Any = None,
    ) -> str:
        """Session entry point: offer a proposal, control signal or draft commit."""

        if control is not None:
            if delta is not None or commit_draft or response_state is not None:
                raise ValueError(
                    "a control signal cannot also carry a delta, a draft commit "
                    "or a numerical response state"
                )
            return self.submit_control(core_id, control)
        return self.submit_proposal(
            core_id,
            grant=grant,
            delta=delta,
            commit_draft=commit_draft,
            response_state=response_state,
        )

    def advance(self) -> BeatResult:
        """Session entry point: run one beat.  Same path for training and inference."""

        return self.beat()

    def inspect(self) -> dict[str, Any]:
        """Session entry point: a JSON-serializable inspection snapshot."""

        return self.snapshot()

    def submit_proposal(
        self,
        core_id: str,
        *,
        grant: AuthorityGrant | None = None,
        delta: FieldDelta | None = None,
        commit_draft: bool = False,
        response_state: Any = None,
    ) -> str:
        """Submit one core proposal; returns its content-addressed proposal id.

        Proposals never write canonical state.  They are committed only by the
        current consolidator, through :meth:`commit` or the next beat.
        """

        self._require_started()
        self._require_episode_open()
        registered = self._require_core(core_id)
        if delta is not None and commit_draft:
            raise ValueError("submit_proposal takes a delta or commit_draft, not both")
        if response_state is not None:
            try:
                canonical_json_bytes(response_state)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "response_state must be JSON-serializable numerical state"
                ) from exc

        if commit_draft:
            entry = self._new_control_entry(core_id, CONTROL_COMMIT)
            return self._register_proposal(entry)

        if delta is None:
            raise ValueError("submit_proposal requires a delta or commit_draft=True")
        if not isinstance(delta, FieldDelta):
            raise TypeError("delta must be a FieldDelta")
        if delta.author_core_id != core_id:
            raise ProposalBoardError(
                f"proposal author {delta.author_core_id!r} does not match core {core_id!r}"
            )
        for operation in delta.operations:
            if isinstance(operation, (InsertText, ReplaceText)):
                assert_native_text(operation.text)
        active_grant = grant if grant is not None else registered
        if not isinstance(active_grant, AuthorityGrant):
            raise TypeError("grant must be an AuthorityGrant")
        if active_grant.authority_class is not AuthorityClass.CORE:
            raise AuthorityViolationError(
                "only core-class grants may submit proposals through HeartHost"
            )
        if not registered.governed_regions.issuperset(active_grant.governed_regions):
            raise AuthorityViolationError(
                f"core {core_id!r} may not widen its registered authority"
            )
        active_grant.assert_delta_permitted(delta)
        head = self._current_head()
        if delta.base_field_id != head.field_id or delta.base_tick_id != head.tick_id:
            raise StaleBaseProposalError(
                "proposal is not bound to the current canonical field identity"
            )
        validate_delta(head, delta, permitted_regions=active_grant.governed_regions)

        entry = _ProposalEntry(
            proposal_id="",
            core_id=core_id,
            kind="delta",
            control=None,
            delta=delta,
            permitted_regions=active_grant.governed_regions,
            response_state=response_state,
            draft_revision=None,
            submitted_at=_utc_now(),
        )
        entry.proposal_id = canonical_sha256(
            {
                "schema": PROPOSAL_SCHEMA,
                "core_id": core_id,
                "kind": entry.kind,
                "control": None,
                "delta": delta.to_canonical_dict(),
                "draft_revision": None,
                "response_state": response_state,
            }
        )
        return self._register_proposal(entry)

    def submit_control(self, core_id: str, control: str) -> str:
        """Submit a control signal; returns its content-addressed proposal id.

        ``wait`` performs zero operations and zero commits.  ``end`` closes the
        episode (no canonical write) until ``reset_episode``.  ``commit``
        requests the current private draft be committed by the consolidator.
        Control signals are never represented as EMPTY or filler payloads.
        """

        self._require_started()
        self._require_episode_open()
        self._require_core(core_id)
        if control not in CONTROL_SIGNALS:
            raise ValueError(
                f"unknown control signal {control!r}; expected one of {CONTROL_SIGNALS!r}"
            )
        entry = self._new_control_entry(core_id, control)
        return self._register_proposal(entry)

    def set_draft(self, core_id: str, text: str) -> dict[str, Any]:
        """Store the host-side private draft (readable, never canonical)."""

        self._require_started()
        self._require_episode_open()
        self._require_core(core_id)
        if not isinstance(text, str):
            raise TypeError("private draft text must be a string")
        assert_native_text(text)
        revision = 1 if self._draft is None else int(self._draft["revision"]) + 1
        self._draft = {"core_id": core_id, "text": text, "revision": revision}
        return dict(self._draft)

    def private_draft(self) -> dict[str, Any] | None:
        """The private draft surface: readable native-95 text, not canonical."""

        return None if self._draft is None else dict(self._draft)

    def committed_text(self, region: LogicalRegion | str = LogicalRegion.RESPONSE_DRAFT) -> str:
        """The canonical surface: committed region text read from branch HEAD."""

        self._require_started()
        logical = region if isinstance(region, LogicalRegion) else LogicalRegion(region)
        return self.branch.load_head().region(logical).text

    def commit(self, committer_id: str, proposal_id: str) -> CommitAck:
        """Commit one pending proposal as the current consolidator.

        This is the only public canonical write.  It requires the lease to be
        held by this host and the committer to be the current roster member.
        WAIT and END proposals refuse to commit anything.

        When the proposal's exact commit is already durable (a restored, stale
        pending entry whose commit the branch journal already records), the
        entry is discarded and acknowledged with the **durable** commit
        identity instead of writing a second generation.
        """

        self._require_lease()
        if not isinstance(committer_id, str) or not committer_id:
            raise ValueError("committer_id must be a non-empty string")
        current = self.current_consolidator()
        if committer_id != current:
            raise AuthorityViolationError(
                f"committer {committer_id!r} is not the current consolidator {current!r}"
            )
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("proposal_id must be a non-empty string")
        entry = self._proposals_by_id.get(proposal_id)
        if entry is None:
            raise ProposalBoardError(f"unknown proposal_id {proposal_id!r}")
        if entry not in self._pending_proposals:
            raise ProposalBoardError(f"proposal {proposal_id!r} was already committed")
        durable = self._proposal_durable_commit(entry)
        if durable is not None:
            self._discard_proposal(entry)
            self._emit_commit_skipped(entry, durable)
            return self._durable_ack(entry, durable)
        ack = self._commit_proposal(committer_id, entry)
        self._discard_proposal(entry)
        return ack

    def beat(self) -> BeatResult:
        """Run one complete beat through the single canonical path.

        Order: reset renewable valve budgets first, admit staged ingress,
        commit pending proposals with the current consolidator, drain the
        durable spool FIFO (skipping anything already committed and
        quarantining anything that fails the final native gate), journal
        health, save the checkpoint, then acknowledge.
        """

        self._require_lease()
        beat_index = self._beats_started
        self._beats_started += 1
        self._roster_index = beat_index % len(self._consolidator_ids)
        consolidator_id = self.current_consolidator()
        tracker = self.registry.tracker
        self._emit(
            "beat_start",
            {
                "beat_index": beat_index,
                "generation": self.branch.load_head_record().generation,
                "consolidator_id": consolidator_id,
            },
        )
        self._identity.next_heartbeat()
        self._identity.next_tick()
        tracker.reset_beat()

        acks: list[CommitAck] = []
        deferred: list[str] = []
        rejected: list[str] = []
        skipped: list[str] = []
        skipped_proposals: list[str] = []
        pending_acks: list[tuple[str, CommitAck]] = []
        try:
            self._admit_staged_ingress(deferred, rejected)
            for entry in list(self._pending_proposals):
                if entry.kind == "control" and entry.control == CONTROL_WAIT:
                    self._discard_proposal(entry)
                    continue
                if entry.kind == "control" and entry.control == CONTROL_END:
                    self._discard_proposal(entry)
                    self._episode_ended = True
                    continue
                durable = self._proposal_durable_commit(entry)
                if durable is not None:
                    # A restored proposal whose commit is already durable must
                    # never be driven a second time: that would append a second
                    # generation for work the branch already contains.
                    self._discard_proposal(entry)
                    self._emit_commit_skipped(entry, durable)
                    skipped_proposals.append(entry.proposal_id)
                    continue
                acks.append(self._commit_proposal(consolidator_id, entry))
                self._discard_proposal(entry)
            for record in self._spool.pending():
                submission_id = submission_id_for_envelope(record, record.valve_version)
                if submission_id in self._commit_index:
                    self._finalize_ack(
                        submission_id,
                        record.event_id,
                        self._commit_index[submission_id],
                        "already_committed",
                    )
                    skipped.append(submission_id)
                    tracker.complete(record.valve_id)
                    continue
                try:
                    ack = self._commit_ingress_record(
                        consolidator_id, record, submission_id
                    )
                except _FINAL_GATE_REJECTIONS as exc:
                    # The durable FIFO head must be acknowledged before a later
                    # record can be quarantined, so already-committed records of
                    # this beat are acknowledged here and dropped from the
                    # deferred ack set.
                    self._flush_acks(pending_acks)
                    self._spool.quarantine_front(record, f"final_gate_reject: {exc}")
                    self._emit(
                        "quarantine",
                        {
                            "event_id": record.event_id,
                            "valve_id": record.valve_id,
                            "submission_id": submission_id,
                            "reason": f"final_gate_reject: {exc}",
                        },
                    )
                    rejected.append(submission_id)
                    tracker.complete(record.valve_id)
                    continue
                acks.append(ack)
                pending_acks.append((record.event_id, ack))
                # The admission is now canonically persisted (or was already);
                # release its pending work allocation on the valve plane.
                tracker.complete(record.valve_id)

            # Canonical work for this beat is complete: the next duty member
            # takes the roster rotation, so the checkpoint written below and any
            # out-of-band commit name the consolidator of the next beat.
            self._rotate_consolidator()
            self._write_health(None)
            if self._checkpoint_path is not None:
                self.save_checkpoint()
            self._flush_acks(pending_acks)
            for definition in self.registry:
                budget = definition.budget
                self._emit(
                    "budget_remaining",
                    {
                        "valve_id": definition.valve_id,
                        "items_remaining": max(
                            0,
                            budget.items_per_beat
                            - tracker.items_this_beat(definition.valve_id),
                        ),
                        "chars_remaining": max(
                            0,
                            budget.target_chars_per_beat
                            - tracker.chars_this_beat(definition.valve_id),
                        ),
                    },
                )
        except Exception as exc:
            try:
                self._write_health(f"beat_failed: {exc}")
            except Exception:
                pass
            self._emit(
                "beat_end",
                {"beat_index": beat_index, "failed": True, "error": str(exc)},
            )
            self._rotate_consolidator()
            raise
        generation = self.branch.load_head_record().generation
        self._emit(
            "beat_end",
            {
                "beat_index": beat_index,
                "generation": generation,
                "failed": False,
                "commits": len(acks),
            },
        )
        return BeatResult(
            beat_index=beat_index,
            consolidator_id=consolidator_id,
            generation=generation,
            commits=tuple(acks),
            skipped_submissions=tuple(skipped),
            deferred_submissions=tuple(deferred),
            rejected_submissions=tuple(rejected),
            skipped_proposals=tuple(skipped_proposals),
        )

    # --------------------------------------------------------------- ingress

    def submit_ingress(self, envelope: ValveEnvelope) -> SubmissionReceipt:
        """Offer one native valve envelope for admission on the next beat.

        Native-95 membership is checked first and fails closed: a non-native
        payload is recorded in the spool rejection/quarantine log with an
        explicit reason and can never enter the durable FIFO.  Rejections are
        receipts, not exceptions; the submission id is content addressed so a
        crash retry is recognised as the same submission.
        """

        self._require_started()
        if not isinstance(envelope, ValveEnvelope):
            raise TypeError("submit_ingress requires a ValveEnvelope")
        try:
            assert_native_text(envelope.payload)
        except UnsupportedCharacterError as exc:
            return self._reject_ingress(envelope, f"non_native_payload: {exc}", None, 0)
        if not envelope.payload:
            return self._reject_ingress(envelope, "empty_payload", None, 0)

        definition = (
            self.registry.get(envelope.valve_id)
            if envelope.valve_id in self.registry
            else None
        )
        valve_version = definition.version if definition is not None else 0
        submission_id = submission_id_for_envelope(envelope, valve_version)
        existing = self._submissions.get(submission_id)
        if existing is not None and existing.ack_state in _ACK_ACTIVE_STATES:
            return SubmissionReceipt(
                submission_id,
                envelope.valve_id,
                "duplicate",
                f"already_{existing.ack_state}",
            )
        committed = self._commit_index.get(submission_id)
        if committed is not None:
            return SubmissionReceipt(
                submission_id, envelope.valve_id, "duplicate", "already_committed"
            )

        decision = self.registry.local_validate(envelope)
        if not decision.admitted:
            return self._reject_ingress(
                envelope,
                f"valve_local_reject: {decision.reason}",
                submission_id,
                valve_version,
            )

        entry = _SubmissionEntry(
            submission_id=submission_id,
            valve_id=envelope.valve_id,
            source_id=envelope.source_id,
            payload=envelope.payload,
            provenance=envelope.provenance,
            envelope_type=envelope.envelope_type,
            valve_version=valve_version,
            ack_state="staged",
        )
        self._submissions[submission_id] = entry
        self._staged.append(entry)
        return SubmissionReceipt(
            submission_id, envelope.valve_id, "staged", "staged_for_next_beat"
        )

    def _reject_ingress(
        self,
        envelope: ValveEnvelope,
        reason: str,
        submission_id: str | None,
        valve_version: int,
    ) -> SubmissionReceipt:
        record = self._spool.reject_envelope(
            envelope,
            reason=reason,
            valve_version=valve_version if valve_version >= 1 else 0,
            quarantine=True,
        )
        self._emit(
            "valve_reject",
            {
                "submission_id": submission_id,
                "valve_id": envelope.valve_id,
                "reason": reason,
            },
        )
        self._emit(
            "quarantine",
            {
                "event_id": record.event_id,
                "valve_id": envelope.valve_id,
                "submission_id": submission_id,
                "reason": reason,
            },
        )
        if submission_id is not None:
            self._submissions.pop(submission_id, None)
        return SubmissionReceipt(submission_id, envelope.valve_id, "rejected", reason)

    def _admit_staged_ingress(
        self, deferred: list[str], rejected: list[str]
    ) -> None:
        while self._staged:
            entry = self._staged[0]
            envelope = entry.envelope()
            decision = self.registry.decide(envelope, global_budget=self._global_budget)
            if not decision.admitted and _is_renewable_cap(decision.reason):
                self._emit(
                    "cap_exhausted",
                    {
                        "valve_id": envelope.valve_id,
                        "reason": decision.reason,
                        "deferred": len(self._staged),
                    },
                )
                deferred.extend(item.submission_id for item in self._staged)
                break
            self._staged.popleft()
            if not decision.admitted:
                self._spool.reject_envelope(
                    envelope,
                    reason=f"valve_reject: {decision.reason}",
                    valve_version=entry.valve_version,
                    quarantine=True,
                )
                entry.ack_state = "rejected"
                entry.reason = decision.reason
                self._submissions.pop(entry.submission_id, None)
                self._emit(
                    "valve_reject",
                    {
                        "submission_id": entry.submission_id,
                        "valve_id": envelope.valve_id,
                        "reason": decision.reason,
                    },
                )
                rejected.append(entry.submission_id)
                continue
            record = self._spool.submit(
                envelope, decision=decision, valve_version=entry.valve_version
            )
            entry.ack_state = "admitted"
            entry.spool_event_id = record.event_id
            self._emit(
                "valve_admission",
                {
                    "submission_id": entry.submission_id,
                    "valve_id": envelope.valve_id,
                    "chars": len(envelope.payload),
                },
            )

    # ------------------------------------------------------- canonical commit

    def _commit_proposal(self, committer_id: str, entry: _ProposalEntry) -> CommitAck:
        if entry.kind == "delta":
            if entry.delta is None:
                raise ProposalBoardError("delta proposal carries no delta")
            committed_text = None
            if len(entry.delta.operations) == 1:
                operation = entry.delta.operations[0]
                if isinstance(operation, (InsertText, ReplaceText)):
                    committed_text = operation.text
            return self._commit_delta(
                committer_id,
                entry.delta,
                permitted_regions=entry.permitted_regions,
                metadata_extra={"proposal_id": entry.proposal_id},
                proposal_id=entry.proposal_id,
                committed_text=committed_text,
            )
        if entry.kind != "control":
            raise ProposalBoardError(f"unknown proposal kind {entry.kind!r}")
        if entry.control == CONTROL_WAIT:
            raise ProposalBoardError("a WAIT control performs zero operations and zero commit")
        if entry.control == CONTROL_END:
            raise ProposalBoardError("an END control performs zero operations and zero commit")
        if entry.control == CONTROL_COMMIT:
            delta = self._draft_commit_delta(entry, self._current_head())
            return self._commit_delta(
                committer_id,
                delta,
                permitted_regions=frozenset({LogicalRegion.RESPONSE_DRAFT}),
                metadata_extra={
                    "proposal_id": entry.proposal_id,
                    "draft_revision": entry.draft_revision,
                },
                proposal_id=entry.proposal_id,
                committed_text=self._draft["text"] if self._draft else None,
            )
        raise ProposalBoardError(f"unknown control signal {entry.control!r}")

    def _draft_commit_delta(
        self, entry: _ProposalEntry, base: SharedFieldSnapshot
    ) -> FieldDelta:
        """Materialize the private-draft commit delta against ``base``.

        The control is revision-bound: a draft that changed after the control
        was issued fails closed instead of committing text the core never
        asked to commit.
        """

        draft = self._draft
        if draft is None or not draft["text"]:
            raise HostStateError("a draft commit requires a non-empty private draft")
        if entry.draft_revision is not None and int(draft["revision"]) != entry.draft_revision:
            raise ProposalBoardError(
                "the private draft changed after this commit control was issued"
            )
        revision = int(draft["revision"])
        return replacement_delta(
            base,
            region=LogicalRegion.RESPONSE_DRAFT,
            text=draft["text"],
            author_core_id=entry.core_id,
            pass_id=f"draft:{revision}",
            evidence=(f"draft:{revision}",),
            provenance="host_private_draft_commit",
        )

    def _proposal_delta_candidate(
        self, entry: _ProposalEntry, base: SharedFieldSnapshot
    ) -> FieldDelta | None:
        """The exact delta a proposal materializes to against ``base``, if known."""

        if entry.kind == "delta":
            return entry.delta
        if entry.kind == "control" and entry.control == CONTROL_COMMIT:
            try:
                return self._draft_commit_delta(entry, base)
            except (HostStateError, ProposalBoardError):
                return None
        return None

    def _proposal_durable_commit(self, entry: _ProposalEntry) -> dict[str, Any] | None:
        """The durable commit of a proposal, if the branch already contains it.

        Primary evidence is the branch journal's ``metadata.proposal_id`` in a
        commit event.  Secondary evidence is a detected applied-but-unlogged
        HEAD commit whose delta is byte-identical to what this proposal would
        materialize, which closes the same crash window handled for ingress.
        """

        record = self._proposal_commit_index.get(entry.proposal_id)
        if record is not None:
            return record
        delta = self._unlogged_head_delta
        if delta is None:
            return None
        head = self.branch.load_head()
        if head.parent_field_id is None:
            return None
        try:
            base = self.branch.load_snapshot(head.parent_field_id)
        except Exception:
            return None
        candidate = self._proposal_delta_candidate(entry, base)
        if candidate is None or candidate.delta_id != delta.delta_id:
            return None
        head_record = self.branch.load_head_record()
        record = {
            "delta_id": delta.delta_id,
            "generation": head_record.generation,
            "field_id": head_record.field_id,
            "tick_id": head_record.tick_id,
            "committer_id": None,
        }
        self._proposal_commit_index[entry.proposal_id] = record
        return record

    def _durable_ack(
        self, entry: _ProposalEntry, record: Mapping[str, Any]
    ) -> CommitAck:
        """Reconstruct the acknowledgement of an already-durable proposal commit."""

        delta_id = str(record["delta_id"])
        regions: tuple[str, ...] = ()
        committed_text: str | None = None
        path = self.branch.deltas_dir / f"{delta_id}.json"
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                delta = field_delta_from_canonical_dict(value)
                regions = tuple(sorted({op.region.value for op in delta.operations}))
                if len(delta.operations) == 1 and isinstance(
                    delta.operations[0], (InsertText, ReplaceText)
                ):
                    committed_text = delta.operations[0].text
                if (
                    entry.kind == "control"
                    and entry.control == CONTROL_COMMIT
                    and self._draft is not None
                    and entry.draft_revision == int(self._draft["revision"])
                ):
                    committed_text = self._draft["text"]
            except Exception:
                pass
        return CommitAck(
            commit_id=delta_id,
            delta_id=delta_id,
            generation=int(record.get("generation") or 0),
            field_id=str(record.get("field_id") or ""),
            tick_id=int(record.get("tick_id") or 0),
            committer_id=record.get("committer_id"),
            submission_id=None,
            proposal_id=entry.proposal_id,
            regions=regions,
            committed_text=committed_text,
        )

    def _emit_commit_skipped(
        self, entry: _ProposalEntry, record: Mapping[str, Any]
    ) -> None:
        self._emit(
            "commit",
            {
                "submission_id": None,
                "proposal_id": entry.proposal_id,
                "commit_id": record.get("delta_id"),
                "generation": record.get("generation"),
                "regions": [],
                "skipped": True,
                "reason": "already_committed",
            },
        )

    def _commit_delta(
        self,
        committer_id: str,
        delta: FieldDelta,
        *,
        permitted_regions: frozenset[LogicalRegion],
        metadata_extra: Mapping[str, Any] | None = None,
        submission_id: str | None = None,
        proposal_id: str | None = None,
        committed_text: str | None = None,
    ) -> CommitAck:
        """The single canonical write choke point of the Heart host."""

        self._require_lease()
        current = self.current_consolidator()
        if committer_id != current:
            raise AuthorityViolationError(
                f"committer {committer_id!r} is not the current consolidator {current!r}"
            )
        metadata: dict[str, Any] = {"committer_id": committer_id}
        if metadata_extra:
            metadata.update(dict(metadata_extra))
        if submission_id is not None:
            metadata["submission_id"] = submission_id
        # The lease makes this host the only writer, so the branch's own rule
        # (commit advances the generation by exactly one) gives the successor
        # generation without re-reading the durable HEAD after the commit.
        base_record = self.branch.load_head_record()
        successor = self.branch.commit(
            delta,
            permitted_regions=permitted_regions,
            metadata=metadata,
        )
        generation = base_record.generation + 1
        self._head_cache = successor
        self._unlogged_head_delta = None
        self._last_commit_id = delta.delta_id
        if proposal_id is not None:
            # Remember in-process proposal commits too, so a stale checkpoint
            # restored later in this session cannot re-drive them.
            self._proposal_commit_index[proposal_id] = {
                "delta_id": delta.delta_id,
                "generation": generation,
                "field_id": successor.field_id,
                "tick_id": successor.tick_id,
                "committer_id": committer_id,
            }
        for operation in delta.operations:
            self._cursors[operation.region.value] = delta.delta_id
        ack = CommitAck(
            commit_id=delta.delta_id,
            delta_id=delta.delta_id,
            generation=generation,
            field_id=successor.field_id,
            tick_id=successor.tick_id,
            committer_id=committer_id,
            submission_id=submission_id,
            proposal_id=proposal_id,
            regions=tuple(sorted({op.region.value for op in delta.operations})),
            committed_text=committed_text,
        )
        self._emit(
            "commit",
            {
                "submission_id": submission_id,
                "proposal_id": proposal_id,
                "commit_id": ack.commit_id,
                "generation": generation,
                "regions": list(ack.regions),
            },
        )
        return ack

    def _ingress_delta(
        self,
        record: IngressRecord,
        base: SharedFieldSnapshot,
        submission_id: str,
    ) -> tuple[FieldDelta, AuthorityGrant]:
        grant = self.registry.resolve_grant(record.valve_id, record.source_id)
        regions = sorted(grant.governed_regions)
        if len(regions) != 1:
            raise PoisonEventError(
                f"valve {record.valve_id!r} governs {len(regions)} regions; "
                "ingress materialization is ambiguous"
            )
        region = regions[0]
        offset = len(base.region(region).text)
        delta = FieldDelta(
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
            author_core_id=f"ingress:{record.valve_id}",
            pass_id=f"submission:{submission_id}",
            operations=(
                InsertText(
                    region=region,
                    offset=offset,
                    text=record.payload,
                    provenance=f"ingress:{submission_id}",
                ),
            ),
            evidence=(submission_id,),
        )
        return delta, grant

    def _commit_ingress_record(
        self,
        committer_id: str,
        record: IngressRecord,
        submission_id: str,
    ) -> CommitAck:
        assert_native_text(record.payload)
        if not record.payload:
            raise PoisonEventError("empty ingress payload cannot be committed")
        head = self._current_head()
        delta, grant = self._ingress_delta(record, head, submission_id)
        ack = self._commit_delta(
            committer_id,
            delta,
            permitted_regions=grant.governed_regions,
            metadata_extra={
                "submission_id": submission_id,
                "valve_id": record.valve_id,
                "spool_event_id": record.event_id,
            },
            submission_id=submission_id,
            committed_text=record.payload,
        )
        self._commit_index[submission_id] = ack.commit_id
        entry = self._submissions.get(submission_id)
        if entry is not None:
            entry.ack_state = "committed"
            entry.commit_id = ack.commit_id
            entry.generation = ack.generation
        return ack

    def _finalize_ack(
        self,
        submission_id: str,
        event_id: str,
        commit_id: str,
        reason: str,
    ) -> None:
        self._spool.acknowledge(event_id)
        self._submissions.pop(submission_id, None)
        self._emit(
            "ack",
            {
                "submission_id": submission_id,
                "event_id": event_id,
                "commit_id": commit_id,
                "reason": reason,
            },
        )

    def _flush_acks(self, pending_acks: list[tuple[str, CommitAck]]) -> None:
        for event_id, ack in pending_acks:
            if ack.submission_id is not None:
                self._finalize_ack(
                    ack.submission_id, event_id, ack.commit_id, "committed"
                )
        pending_acks.clear()

    # ---------------------------------------------------- episode & proposal

    def reset_episode(self) -> dict[str, Any]:
        """Clear draft, applied-delta cursors and pending proposals.

        Committed canonical field content is never touched by a reset.
        """

        self._require_started()
        summary = {
            "draft_cleared": self._draft is not None,
            "cursors_cleared": len(self._cursors),
            "pending_proposals_cleared": len(self._pending_proposals),
            "episode_was_ended": self._episode_ended,
        }
        self._draft = None
        self._cursors = {}
        for entry in self._pending_proposals:
            self._proposals_by_id.pop(entry.proposal_id, None)
        self._pending_proposals = []
        self._episode_ended = False
        return summary

    def _new_control_entry(self, core_id: str, control: str) -> _ProposalEntry:
        """Build one control proposal, content-addressed over its intent.

        A COMMIT control's identity covers the private-draft **text** as well
        as its revision, so two different drafts that happen to share a
        revision (draft revisions restart at 1 after ``reset_episode``) can
        never collapse into one already-committed proposal.
        """

        draft_revision: int | None = None
        draft_text: str | None = None
        if control == CONTROL_COMMIT:
            if self._draft is None:
                raise HostStateError("a COMMIT control requires a private draft first")
            if not self._draft["text"]:
                raise HostStateError(
                    "a COMMIT control requires a non-empty private draft"
                )
            draft_revision = int(self._draft["revision"])
            draft_text = str(self._draft["text"])
        entry = _ProposalEntry(
            proposal_id="",
            core_id=core_id,
            kind="control",
            control=control,
            delta=None,
            permitted_regions=frozenset({LogicalRegion.RESPONSE_DRAFT}),
            response_state=None,
            draft_revision=draft_revision,
            submitted_at=_utc_now(),
        )
        entry.proposal_id = canonical_sha256(
            {
                "schema": PROPOSAL_SCHEMA,
                "core_id": core_id,
                "kind": entry.kind,
                "control": control,
                "delta": None,
                "draft_revision": draft_revision,
                "draft_text": draft_text,
                "response_state": None,
            }
        )
        return entry

    def _register_proposal(self, entry: _ProposalEntry) -> str:
        if entry.proposal_id in self._proposals_by_id:
            return entry.proposal_id
        self._pending_proposals.append(entry)
        self._proposals_by_id[entry.proposal_id] = entry
        self._emit(
            "proposal_received",
            {
                "proposal_id": entry.proposal_id,
                "core_id": entry.core_id,
                "kind": entry.kind,
                "control": entry.control,
            },
        )
        return entry.proposal_id

    def _discard_proposal(self, entry: _ProposalEntry) -> None:
        self._proposals_by_id.pop(entry.proposal_id, None)
        if entry in self._pending_proposals:
            self._pending_proposals.remove(entry)

    def _require_core(self, core_id: str) -> AuthorityGrant:
        if not isinstance(core_id, str) or not core_id:
            raise ValueError("core_id must be a non-empty string")
        grant = self._cores.get(core_id)
        if grant is None:
            raise UnknownCoreError(f"core {core_id!r} is not registered")
        return grant

    def _require_episode_open(self) -> None:
        if self._episode_ended:
            raise HostStateError(
                "the episode has ended; reset_episode() before new proposals"
            )

    def _require_started(self) -> None:
        if not self._started or self._spool is None or self.branch is None:
            raise HostStateError("HeartHost has not been started")

    def _require_lease(self) -> None:
        """Require our single-writer lease; the host's canonical write gate.

        A host that never started is a state error; a host whose lease was
        released (or never acquired) has no writer authority and is denied.
        """

        lease = self._lease
        if lease is None:
            if self._spool is None and not self._started:
                raise HostStateError("HeartHost has not been started")
            raise LeaseDeniedError(
                "the single-writer Heart lease is not held by this host"
            )
        if not lease.is_held_by_us():
            raise LeaseDeniedError(
                "the single-writer Heart lease is not held by this host"
            )

    def _current_head(self) -> SharedFieldSnapshot:
        """The durable branch HEAD, cached across the commits of one beat.

        While the lease is held only this host can advance HEAD, and every host
        commit goes through ``_commit_delta``, which refreshes this cache with
        the successor snapshot the branch returned.
        """

        if self._head_cache is None:
            self._head_cache = self.branch.load_head()
        return self._head_cache

    # -------------------------------------------------------------- restart

    def _journal_events(self) -> list[dict[str, Any]]:
        """Parse the durable branch journal, failing closed on corruption."""

        journal = self.branch.journal_path
        events: list[dict[str, Any]] = []
        if not journal.exists():
            return events
        for line_number, line in enumerate(
            journal.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BranchIntegrityError(
                    f"corrupt branch journal JSON at line {line_number}"
                ) from exc
            if not isinstance(event, dict):
                raise BranchIntegrityError(
                    f"branch journal line {line_number} is not an object"
                )
            events.append(event)
        return events

    def _recover_commit_index(self) -> None:
        """Rebuild the durable commit indexes from the branch journal.

        Two maps are recovered: ``submission_id -> delta_id`` for ingress
        records, and ``proposal_id -> durable commit record`` for proposals
        (proposal and draft-control commits), so a re-driven commit can never
        append a second generation.
        """

        self._commit_index = {}
        self._proposal_commit_index = {}
        last_generation = 0
        for event in self._journal_events():
            generation = event.get("generation")
            if isinstance(generation, int) and not isinstance(generation, bool):
                last_generation = max(last_generation, generation)
            if event.get("event") != "commit":
                continue
            metadata = event.get("metadata")
            delta_id = event.get("delta_id")
            if not isinstance(metadata, Mapping):
                continue
            if not (isinstance(delta_id, str) and delta_id):
                continue
            submission_id = metadata.get("submission_id")
            if isinstance(submission_id, str) and submission_id:
                self._commit_index[submission_id] = delta_id
            proposal_id = metadata.get("proposal_id")
            if isinstance(proposal_id, str) and proposal_id:
                self._proposal_commit_index[proposal_id] = {
                    "delta_id": delta_id,
                    "generation": generation,
                    "field_id": event.get("field_id"),
                    "tick_id": event.get("tick_id"),
                    "committer_id": metadata.get("committer_id"),
                }
        head_record = self.branch.load_head_record()
        self._unlogged_head_delta = None
        if head_record.generation > last_generation:
            self._unlogged_head_delta = self._find_unlogged_head_delta()

    def _find_unlogged_head_delta(self) -> FieldDelta:
        """Find the applied-but-unlogged delta that produced the current HEAD.

        ``CanonicalStateBranch.commit`` advances the atomic HEAD before it
        appends the audit event, so a crash inside that window leaves a durable
        commit without a journal line.  The host reconstructs it here so a
        restart cannot re-drive the same submission twice.
        """

        head = self.branch.load_head()
        if head.parent_field_id is None:
            raise BranchIntegrityError(
                "branch HEAD advanced beyond its journal but has no parent"
            )
        base = self.branch.load_snapshot(head.parent_field_id)
        permitted = self._reconstruction_regions()
        for path in sorted(self.branch.deltas_dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BranchIntegrityError(
                    f"cannot read branch delta {path.name}"
                ) from exc
            if not isinstance(value, dict) or value.get("base_field_id") != base.field_id:
                continue
            try:
                delta = field_delta_from_canonical_dict(value)
                successor = apply_delta(base, delta, permitted_regions=permitted)
            except Exception:
                continue
            if successor.field_id == head.field_id:
                return delta
        raise BranchIntegrityError(
            "branch HEAD advanced beyond its journal with no applied delta"
        )

    def _reconstruction_regions(self) -> frozenset[LogicalRegion]:
        regions = set(CORE_WRITABLE_REGIONS)
        for definition in self.registry:
            try:
                regions.update(definition.to_authority_grant().governed_regions)
            except Exception:
                continue
        for grant in self._cores.values():
            regions.update(grant.governed_regions)
        return frozenset(regions)

    def _derive_commit_state(
        self, head_record: Any
    ) -> tuple[str | None, dict[str, str]]:
        """Derive ``last_commit_id`` and per-region cursors from the journal.

        Used when a checkpoint predates the durable HEAD: reporting a live
        generation beside the checkpoint's stale cursors would be a lie, so the
        commit bookkeeping is rebuilt from every journal commit event up to
        HEAD (plus a detected applied-but-unlogged HEAD commit).
        """

        cursors: dict[str, str] = {}
        last_commit_id: str | None = None
        for event in self._journal_events():
            if event.get("event") != "commit":
                continue
            generation = event.get("generation")
            if isinstance(generation, bool) or not isinstance(generation, int):
                continue
            if generation > head_record.generation:
                continue
            delta_id = event.get("delta_id")
            if not isinstance(delta_id, str) or not delta_id:
                continue
            delta = self._load_branch_delta(delta_id)
            for operation in delta.operations:
                cursors[operation.region.value] = delta_id
            last_commit_id = delta_id
        if self._unlogged_head_delta is not None:
            delta = self._unlogged_head_delta
            for operation in delta.operations:
                cursors[operation.region.value] = delta.delta_id
            last_commit_id = delta.delta_id
        return last_commit_id, cursors

    def _load_branch_delta(self, delta_id: str) -> FieldDelta:
        path = self.branch.deltas_dir / f"{delta_id}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BranchIntegrityError(
                f"branch journal references unreadable delta {delta_id}"
            ) from exc
        try:
            return field_delta_from_canonical_dict(value)
        except Exception as exc:
            raise BranchIntegrityError(
                f"branch journal references an invalid delta {delta_id}"
            ) from exc

    def _reconcile_spool(self) -> None:
        """Replay pending submissions against the journal by id.

        Anything whose commit already exists in the branch journal (or whose
        exact delta is already applied at HEAD) is skipped and acknowledged;
        everything else stays pending and is re-driven in FIFO order.

        Invariant: a committed record never follows an uncommitted one in the
        durable FIFO, because the host commits records strictly in FIFO order.
        A committed record behind an uncommitted one is therefore a corrupt
        spool/journal combination and fails closed instead of being acked out
        of order.
        """

        saw_uncommitted = False
        for record in self._spool.pending():
            submission_id = submission_id_for_envelope(record, record.valve_version)
            entry = self._submissions.get(submission_id)
            commit_id = self._commit_index.get(submission_id)
            if commit_id is None and self._unlogged_head_delta is not None:
                commit_id = self._matching_unlogged_commit(record, submission_id)
            if commit_id is not None:
                if saw_uncommitted:
                    raise ReplayEventError(
                        "durable ingress FIFO has a committed record behind an "
                        "uncommitted one"
                    )
                self._commit_index.setdefault(submission_id, commit_id)
                if entry is None:
                    entry = _SubmissionEntry(
                        submission_id=submission_id,
                        valve_id=record.valve_id,
                        source_id=record.source_id,
                        payload=record.payload,
                        provenance=record.provenance,
                        envelope_type=record.envelope_type,
                        valve_version=record.valve_version,
                        ack_state="committed",
                    )
                    self._submissions[submission_id] = entry
                entry.ack_state = "committed"
                entry.spool_event_id = record.event_id
                entry.commit_id = commit_id
                self._finalize_ack(
                    submission_id, record.event_id, commit_id, "already_committed"
                )
                continue
            saw_uncommitted = True
            if entry is None:
                entry = _SubmissionEntry(
                    submission_id=submission_id,
                    valve_id=record.valve_id,
                    source_id=record.source_id,
                    payload=record.payload,
                    provenance=record.provenance,
                    envelope_type=record.envelope_type,
                    valve_version=record.valve_version,
                    ack_state="admitted",
                )
                self._submissions[submission_id] = entry
            else:
                entry.ack_state = "admitted"
            entry.spool_event_id = record.event_id

    def _matching_unlogged_commit(
        self, record: IngressRecord, submission_id: str
    ) -> str | None:
        """Whether this record's deterministic delta is the unlogged HEAD commit."""

        delta = self._unlogged_head_delta
        if delta is None:
            return None
        head = self.branch.load_head()
        if head.parent_field_id is None:
            return None
        try:
            base = self.branch.load_snapshot(head.parent_field_id)
            candidate, _grant = self._ingress_delta(record, base, submission_id)
        except Exception:
            return None
        if candidate.delta_id == delta.delta_id:
            return delta.delta_id
        return None

    # ------------------------------------------------------------ inspection

    def snapshot(self) -> dict[str, Any]:
        """A JSON-serializable view of host bookkeeping for inspection."""

        self._require_started()
        return self._state_dict()

    def replay_from(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Restore host bookkeeping (never weights) from a snapshot.

        Like :meth:`load_checkpoint`, the restored bookkeeping is reconciled
        against the durable spool and branch journal, and the return value is
        the restore report (``behind_head`` flags a snapshot that lagged the
        durable HEAD, in which case cursors and ``last_commit_id`` are derived
        from the journal instead of the stale snapshot values).
        """

        self._require_started()
        if not isinstance(snapshot, Mapping):
            raise CheckpointIntegrityError("snapshot must be a mapping")
        if snapshot.get("schema") != CHECKPOINT_FORMAT:
            raise CheckpointIntegrityError(
                f"unsupported HeartHost snapshot schema {snapshot.get('schema')!r}"
            )
        report = self._restore_state(dict(snapshot))
        self._reconcile_spool()
        return report

    def save_checkpoint(self, path: Path | str | None = None) -> Path:
        """Persist the consistent-boundary checkpoint plus a sha256 sidecar."""

        self._require_started()
        target = Path(path) if path is not None else self._checkpoint_path
        if target is None:
            raise HostStateError("no checkpoint path is configured")
        state = self._state_dict()
        data = canonical_json_bytes(state) + b"\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        sidecar = target.with_name(target.name + CHECKPOINT_SIDECAR_SUFFIX)
        sidecar_temporary = sidecar.with_name(sidecar.name + f".{os.getpid()}.tmp")
        digest = hashlib.sha256(data).hexdigest()
        with sidecar_temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(digest + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(sidecar_temporary, sidecar)
        return target

    def load_checkpoint(self, path: Path | str) -> dict[str, Any]:
        """Verify the checkpoint sidecar *before* parsing, then restore bookkeeping.

        Returns the restore report: ``behind_head`` is true when the checkpoint
        predated the durable HEAD, in which case the stale ``last_commit_id``
        and cursors were replaced by values derived from the branch journal.
        """

        self._require_started()
        target = Path(path)
        sidecar = target.with_name(target.name + CHECKPOINT_SIDECAR_SUFFIX)
        try:
            raw = target.read_bytes()
        except OSError as exc:
            raise CheckpointIntegrityError(
                f"cannot read HeartHost checkpoint {target}"
            ) from exc
        try:
            expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
        except (OSError, IndexError) as exc:
            raise CheckpointIntegrityError(
                f"checkpoint sidecar is missing or unreadable: {sidecar}"
            ) from exc
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise CheckpointIntegrityError(
                "checkpoint sha256 sidecar does not match its bytes"
            )
        try:
            state = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointIntegrityError(
                "checkpoint is not valid JSON after sidecar verification"
            ) from exc
        if not isinstance(state, dict) or state.get("schema") != CHECKPOINT_FORMAT:
            raise CheckpointIntegrityError("unsupported HeartHost checkpoint schema")
        report = self._restore_state(state)
        self._reconcile_spool()
        return report

    def _state_dict(self) -> dict[str, Any]:
        head_record = self.branch.load_head_record()
        tracker = self.registry.tracker
        budget: dict[str, Any] = {}
        for definition in self.registry:
            valve_id = definition.valve_id
            budget[valve_id] = {
                "pending": tracker.pending(valve_id),
                "items_this_beat": tracker.items_this_beat(valve_id),
                "chars_this_beat": tracker.chars_this_beat(valve_id),
            }
        return {
            "schema": CHECKPOINT_FORMAT,
            "captured_at": _utc_now(),
            "branch": {
                "branch_id": self.branch.branch_id,
                "root": str(self.branch.root),
                "generation": head_record.generation,
                "field_id": head_record.field_id,
                "tick_id": head_record.tick_id,
                "parent_field_id": head_record.parent_field_id,
            },
            "generation": head_record.generation,
            "last_commit_id": self._last_commit_id,
            "epoch": self._epoch,
            "event_seq": self._event_seq,
            "cursors": dict(sorted(self._cursors.items())),
            "draft": None if self._draft is None else dict(self._draft),
            "episode_ended": self._episode_ended,
            "beat_count": self._beats_started,
            "roster_index": self._roster_index,
            "consolidator_id": self.current_consolidator(),
            "pending_submissions": [
                entry.to_dict()
                for entry in self._submissions.values()
                if entry.ack_state != "acked"
            ],
            "pending_proposals": [entry.to_dict() for entry in self._pending_proposals],
            "budget": budget,
            "lease": self._lease_state_dict(),
            "events": {
                "counters": dict(self._event_counters),
                "retained": {
                    name: len(events) for name, events in sorted(self._events.items())
                },
            },
            "spool": {
                "pending_count": self._spool.pending_count,
                "rejection_count": self._spool.rejection_count,
                "quarantine_count": self._spool.quarantine_count,
            },
            "commit_index_size": len(self._commit_index),
            "proposal_commit_index_size": len(self._proposal_commit_index),
            "last_restore": self._last_restore,
        }

    def _restore_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        branch = state.get("branch")
        if not isinstance(branch, Mapping):
            raise CheckpointIntegrityError("checkpoint branch identity is missing")
        if branch.get("branch_id") != self.branch.branch_id:
            raise CheckpointIntegrityError(
                "checkpoint was written for a different branch identity"
            )
        if branch.get("root") is not None and branch.get("root") != str(self.branch.root):
            raise CheckpointIntegrityError(
                "checkpoint was written for a different branch root"
            )
        generation = _require_nonnegative_int(branch.get("generation"), "generation")
        head_record = self.branch.load_head_record()
        if generation > head_record.generation:
            raise CheckpointIntegrityError(
                "checkpoint generation is ahead of the durable branch HEAD"
            )
        if generation == head_record.generation and branch.get("field_id") != head_record.field_id:
            raise CheckpointIntegrityError(
                "checkpoint field identity disagrees with the durable branch HEAD"
            )
        behind_head = generation < head_record.generation

        restored_epoch = state.get("epoch")
        if not isinstance(restored_epoch, str) or not restored_epoch:
            raise CheckpointIntegrityError(
                "checkpoint epoch must be a non-empty string"
            )
        restored_seq = _require_nonnegative_int(state.get("event_seq"), "event_seq")
        if restored_epoch != self._epoch:
            # Continue the restored epoch: the global ordinal resumes where the
            # checkpoint left it.
            self._epoch = restored_epoch
            self._event_seq = restored_seq
        else:
            # Same live epoch: never regress the ordinal we already emitted.
            self._event_seq = max(self._event_seq, restored_seq)

        self._last_commit_id = _optional_nonempty_str(
            state.get("last_commit_id"), "last_commit_id"
        )
        self._episode_ended = bool(state.get("episode_ended", False))
        self._beats_started = _require_nonnegative_int(
            state.get("beat_count"), "beat_count"
        )
        self._roster_index = _require_nonnegative_int(
            state.get("roster_index"), "roster_index"
        )

        raw_cursors = state.get("cursors", {})
        if not isinstance(raw_cursors, Mapping):
            raise CheckpointIntegrityError("checkpoint cursors must be an object")
        cursors: dict[str, str] = {}
        for region_name, delta_id in raw_cursors.items():
            try:
                logical = LogicalRegion(region_name)
            except (TypeError, ValueError) as exc:
                raise CheckpointIntegrityError(
                    f"checkpoint cursor names unknown region {region_name!r}"
                ) from exc
            if not isinstance(delta_id, str) or not delta_id:
                raise CheckpointIntegrityError("checkpoint cursor deltas must be non-empty")
            cursors[logical.value] = delta_id
        self._cursors = cursors

        report: dict[str, Any] = {
            "behind_head": behind_head,
            "checkpoint_generation": generation,
            "head_generation": head_record.generation,
        }
        if behind_head:
            # Never report a live generation beside stale cursors: rebuild the
            # commit bookkeeping from the durable journal up to HEAD.
            derived_last_commit_id, derived_cursors = self._derive_commit_state(
                head_record
            )
            self._last_commit_id = derived_last_commit_id
            self._cursors = derived_cursors
            report["derived_last_commit_id"] = derived_last_commit_id
            report["derived_cursors"] = len(derived_cursors)
        self._last_restore = dict(report)

        draft = state.get("draft")
        if draft is None:
            self._draft = None
        else:
            if not isinstance(draft, Mapping):
                raise CheckpointIntegrityError("checkpoint draft must be an object or null")
            text = draft.get("text")
            if not isinstance(text, str):
                raise CheckpointIntegrityError("checkpoint draft text must be a string")
            try:
                assert_native_text(text)
            except UnsupportedCharacterError as exc:
                raise CheckpointIntegrityError(
                    "checkpoint draft text is outside the native substrate"
                ) from exc
            self._draft = {
                "core_id": str(draft.get("core_id", "")),
                "text": text,
                "revision": _require_nonnegative_int(
                    draft.get("revision"), "draft revision"
                ),
            }

        raw_proposals = state.get("pending_proposals", [])
        if not isinstance(raw_proposals, list):
            raise CheckpointIntegrityError("checkpoint pending proposals must be a list")
        proposals: list[_ProposalEntry] = []
        proposals_by_id: dict[str, _ProposalEntry] = {}
        for item in raw_proposals:
            entry = _proposal_from_dict(item)
            proposals.append(entry)
            proposals_by_id[entry.proposal_id] = entry
        self._pending_proposals = proposals
        self._proposals_by_id = proposals_by_id

        raw_submissions = state.get("pending_submissions", [])
        if not isinstance(raw_submissions, list):
            raise CheckpointIntegrityError("checkpoint pending submissions must be a list")
        submissions: dict[str, _SubmissionEntry] = {}
        staged: deque[_SubmissionEntry] = deque()
        for item in raw_submissions:
            entry = _submission_from_dict(item)
            if entry.ack_state == "acked":
                continue
            submissions[entry.submission_id] = entry
            if entry.ack_state == "staged":
                staged.append(entry)
        self._submissions = submissions
        self._staged = staged

        raw_budget = state.get("budget", {})
        if not isinstance(raw_budget, Mapping):
            raise CheckpointIntegrityError("checkpoint budget must be an object")
        self._restore_budget(raw_budget)

        raw_events = state.get("events", {})
        if isinstance(raw_events, Mapping):
            counters = raw_events.get("counters", {})
            if isinstance(counters, Mapping):
                restored = dict(self._event_counters)
                for name in EVENT_TYPES:
                    if name in counters:
                        restored[name] = _require_nonnegative_int(
                            counters[name], f"event counter {name}"
                        )
                self._event_counters = restored
        return report

    def _restore_budget(self, budget: Mapping[str, Any]) -> None:
        """Rebuild renewable per-valve counters using only public tracker calls."""

        tracker = self.registry.tracker
        for definition in self.registry:
            valve_id = definition.valve_id
            raw = budget.get(valve_id, {})
            if raw is None:
                raw = {}
            if not isinstance(raw, Mapping):
                raise CheckpointIntegrityError(
                    f"checkpoint budget for {valve_id!r} must be an object"
                )
            pending = _require_nonnegative_int(raw.get("pending", 0), "budget pending")
            items = _require_nonnegative_int(
                raw.get("items_this_beat", 0), "budget items_this_beat"
            )
            chars = _require_nonnegative_int(
                raw.get("chars_this_beat", 0), "budget chars_this_beat"
            )
            if items == 0 and chars != 0:
                raise CheckpointIntegrityError(
                    f"checkpoint budget for {valve_id!r} has chars without items"
                )
            tracker.complete(valve_id, tracker.pending(valve_id))
            tracker.reset_beat(valve_id)
            for _ in range(pending):
                tracker.admit(valve_id, 0)
            tracker.reset_beat(valve_id)
            for index in range(items):
                tracker.admit(valve_id, chars if index == 0 else 0)
            if items:
                tracker.complete(valve_id, items)

    def _lease_state_dict(self) -> dict[str, Any]:
        lease = self._lease
        held = bool(lease is not None and lease.is_held_by_us())
        record = self._lease_record or {}
        return {
            "held": held,
            "owner_token": lease.owner_token if lease is not None else None,
            "pid": record.get("pid") if held else None,
        }

    def _status_dict(self) -> dict[str, Any]:
        head_record = self.branch.load_head_record()
        return {
            "started": self._started,
            "branch_id": self.branch.branch_id,
            "generation": head_record.generation,
            "field_id": head_record.field_id,
            "beats_started": self._beats_started,
            "consolidator_id": self.current_consolidator(),
            "staged_submissions": len(self._staged),
            "spool_pending": self._spool.pending_count if self._spool else 0,
        }

    # ------------------------------------------------------------------ health

    def _write_health(self, failure: str | None) -> None:
        if self._health is None or self._identity is None or self._spool is None:
            return
        identity = self._identity.identity
        head_record = self.branch.load_head_record()
        tracker = self.registry.tracker
        valve_ids = sorted(definition.valve_id for definition in self.registry)
        now = datetime.now(timezone.utc)
        health = HeartHealth(
            heart_epoch_id=identity.heart_epoch_id,
            start_sequence=identity.start_sequence,
            heartbeat_sequence=identity.heartbeat_sequence,
            tick_sequence=identity.tick_sequence,
            last_beat_at=now,
            last_successful_circulation_at=None if failure else now,
            last_failure_reason=failure,
            canonical_head_field_id=head_record.field_id,
            canonical_head_tick_id=head_record.tick_id,
            tick_in_flight=False,
            queue_depth_by_valve={
                valve_id: tracker.pending(valve_id) for valve_id in valve_ids
            },
            quarantine_count=self._spool.quarantine_count,
            rejection_count=self._spool.rejection_count,
            valve_states={
                valve_id: self.registry.get(valve_id).state.value
                for valve_id in valve_ids
            },
            dormant_index_id=None,
            lease_owner_pid=(self._lease_record or {}).get("pid"),
            lease_owner_token=(self._lease_record or {}).get("owner_token"),
            last_tick_uid=(
                f"{identity.heart_epoch_id}:tick:{identity.tick_sequence}"
                if identity.tick_sequence
                else None
            ),
        )
        self._health.write(health)

    # ------------------------------------------------------------------ events

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown Heart host event type {event_type!r}")
        self._event_counters[event_type] += 1
        self._event_seq += 1
        self._events[event_type].append(
            {
                "seq": self._event_seq,
                "epoch": self._epoch,
                "at": _utc_now(),
                "type": event_type,
                "payload": dict(payload),
            }
        )


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointIntegrityError(f"{label} must be a non-negative integer")
    return value


def _optional_nonempty_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CheckpointIntegrityError(f"{label} must be null or a non-empty string")
    return value


def _proposal_from_dict(value: Any) -> _ProposalEntry:
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError("serialized pending proposal must be an object")
    proposal_id = value.get("proposal_id")
    core_id = value.get("core_id")
    kind = value.get("kind")
    control = value.get("control")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise CheckpointIntegrityError("serialized proposal id is invalid")
    if not isinstance(core_id, str) or not core_id:
        raise CheckpointIntegrityError("serialized proposal core id is invalid")
    if kind not in ("delta", "control"):
        raise CheckpointIntegrityError("serialized proposal kind is invalid")
    if kind == "control" and control not in CONTROL_SIGNALS:
        raise CheckpointIntegrityError("serialized control signal is invalid")
    raw_regions = value.get("permitted_regions", [])
    if not isinstance(raw_regions, list):
        raise CheckpointIntegrityError("serialized proposal regions must be a list")
    try:
        permitted = frozenset(LogicalRegion(item) for item in raw_regions)
    except (TypeError, ValueError) as exc:
        raise CheckpointIntegrityError(
            "serialized proposal names an unknown region"
        ) from exc
    raw_delta = value.get("delta")
    delta: FieldDelta | None
    if raw_delta is None:
        delta = None
    else:
        if not isinstance(raw_delta, Mapping):
            raise CheckpointIntegrityError("serialized proposal delta must be an object")
        try:
            delta = field_delta_from_canonical_dict(raw_delta)
        except Exception as exc:
            raise CheckpointIntegrityError("serialized proposal delta is invalid") from exc
    if kind == "delta" and delta is None:
        raise CheckpointIntegrityError("delta proposal is missing its delta")
    draft_revision = value.get("draft_revision")
    if draft_revision is not None:
        draft_revision = _require_nonnegative_int(draft_revision, "draft revision")
    response_state = value.get("response_state")
    if response_state is not None:
        try:
            canonical_json_bytes(response_state)
        except (TypeError, ValueError) as exc:
            raise CheckpointIntegrityError(
                "serialized proposal response state is not JSON-serializable"
            ) from exc
    return _ProposalEntry(
        proposal_id=proposal_id,
        core_id=core_id,
        kind=kind,
        control=control if kind == "control" else None,
        delta=delta,
        permitted_regions=permitted,
        response_state=response_state,
        draft_revision=draft_revision,
        submitted_at=str(value.get("submitted_at", "")),
    )


def _submission_from_dict(value: Any) -> _SubmissionEntry:
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError("serialized pending submission must be an object")
    submission_id = value.get("submission_id")
    valve_id = value.get("valve_id")
    payload = value.get("payload")
    ack_state = value.get("ack_state")
    if not isinstance(submission_id, str) or not submission_id:
        raise CheckpointIntegrityError("serialized submission id is invalid")
    if not isinstance(valve_id, str) or not valve_id:
        raise CheckpointIntegrityError("serialized submission valve id is invalid")
    if not isinstance(payload, str):
        raise CheckpointIntegrityError("serialized submission payload must be a string")
    try:
        assert_native_text(payload)
    except UnsupportedCharacterError as exc:
        raise CheckpointIntegrityError(
            "serialized submission payload is outside the native substrate"
        ) from exc
    if ack_state not in (*_ACK_ACTIVE_STATES, "acked", "rejected"):
        raise CheckpointIntegrityError("serialized submission ack state is invalid")
    source_id = value.get("source_id")
    envelope_type = value.get("envelope_type")
    provenance = value.get("provenance")
    reason = value.get("reason")
    spool_event_id = value.get("spool_event_id")
    commit_id = value.get("commit_id")
    generation = value.get("generation")
    for label, item in (
        ("source_id", source_id),
        ("envelope_type", envelope_type),
        ("provenance", provenance),
        ("reason", reason),
    ):
        if not isinstance(item, str):
            raise CheckpointIntegrityError(
                f"serialized submission {label} must be a string"
            )
    for label, item in (
        ("spool_event_id", spool_event_id),
        ("commit_id", commit_id),
    ):
        if item is not None and (not isinstance(item, str) or not item):
            raise CheckpointIntegrityError(
                f"serialized submission {label} must be null or a non-empty string"
            )
    if generation is not None:
        generation = _require_nonnegative_int(generation, "submission generation")
    return _SubmissionEntry(
        submission_id=submission_id,
        valve_id=valve_id,
        source_id=source_id,
        payload=payload,
        provenance=provenance,
        envelope_type=envelope_type,
        valve_version=_require_nonnegative_int(
            value.get("valve_version", 0), "submission valve version"
        ),
        ack_state=ack_state,
        reason=reason,
        spool_event_id=spool_event_id,
        commit_id=commit_id,
        generation=generation,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CONTROL_COMMIT",
    "CONTROL_END",
    "CONTROL_SIGNALS",
    "CONTROL_WAIT",
    "EVENT_LOG_CAP",
    "EVENT_TYPES",
    "BeatResult",
    "CheckpointIntegrityError",
    "CommitAck",
    "HeartHost",
    "SubmissionReceipt",
    "submission_id_for_envelope",
]

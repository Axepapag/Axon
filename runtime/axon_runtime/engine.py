"""Continuously ticking multi-core Axon runtime transaction engine.

One tick has one canonical successor:

``H -> sealed system update U -> working field W -> consolidator delta -> F``.

The proposer may place an ephemeral candidate on a proposal board so the
consolidator can read it, but only the consolidator-authored delta is eligible
for ``F``.  Candidate private state is prepared content-addressably before the
SQLite commit and installed in memory only after that commit succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time
from typing import Any, Callable, Protocol, Sequence

from runtime.field import (
    FieldDelta,
    LogicalRegion,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
)
from runtime.multi_tick_refiner import RegionProposal

from .contracts import (
    ConsolidationDecision,
    CoreIdentity,
    CoreStateManifest,
    ProjectionManifest,
    ProposalRecord,
    RuntimeHead,
    TickCommitRecord,
    ToolRequestRecord,
)
from .field_transaction import (
    SystemFieldUpdate,
    apply_system_update,
    compose_tick_transaction,
)
from .dormant import DormantStore
from .ingress import IngressQueue, build_system_update as build_ingress_update
from .projection import ProjectionPolicy, build_active_projection
from .roles import RoleScheduler
from .store import RuntimeStore
from .tool_protocol import (
    InvocationParseError,
    InvocationValidationError,
    parse_invocation_envelope,
)


class RuntimeEngineError(RuntimeError):
    """A cognitive tick could not satisfy the runtime transaction contract."""


class CommittedStateRecoveryError(RuntimeEngineError):
    """A committed private state could not be installed or reloaded safely."""


@dataclass(frozen=True, slots=True)
class ObservedField:
    """One authenticated full or derived view supplied to a logical core."""

    snapshot: SharedFieldSnapshot
    ancestor_field_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, SharedFieldSnapshot):
            raise TypeError("snapshot must be SharedFieldSnapshot")
        ancestors = tuple(self.ancestor_field_ids)
        if any(not isinstance(item, str) or not item for item in ancestors):
            raise RuntimeEngineError("observation ancestry must be non-empty IDs")
        if len(ancestors) != len(set(ancestors)):
            raise RuntimeEngineError("observation ancestry cannot repeat")
        if self.snapshot.parent_field_id is None:
            if ancestors:
                raise RuntimeEngineError(
                    "a root observation cannot claim ancestor field IDs"
                )
        elif not ancestors or ancestors[-1] != self.snapshot.parent_field_id:
            raise RuntimeEngineError(
                "observation ancestry does not end at its direct parent"
            )
        evidence = tuple(sorted(set(self.evidence_ids)))
        if any(not isinstance(item, str) or not item for item in evidence):
            raise RuntimeEngineError("observation evidence IDs must be non-empty")
        object.__setattr__(self, "ancestor_field_ids", ancestors)
        object.__setattr__(self, "evidence_ids", evidence)


class ObservationProjector(Protocol):
    def project(
        self,
        snapshot: SharedFieldSnapshot,
        *,
        ancestor_field_ids: tuple[str, ...],
    ) -> ObservedField:
        """Return an authenticated observation rooted in ``snapshot``."""


class FullFieldProjector:
    """Use the full unbounded field; the core's 384x16 pager stays bounded."""

    def project(
        self,
        snapshot: SharedFieldSnapshot,
        *,
        ancestor_field_ids: tuple[str, ...],
    ) -> ObservedField:
        return ObservedField(
            snapshot=snapshot,
            ancestor_field_ids=ancestor_field_ids,
        )


class ActiveFieldProjector:
    """Mask dormant/over-budget whole spans without changing canonical truth."""

    def __init__(
        self,
        *,
        dormant_store: DormantStore,
        policy: ProjectionPolicy | None = None,
    ) -> None:
        if not isinstance(dormant_store, DormantStore):
            raise TypeError("dormant_store must be DormantStore")
        self.dormant_store = dormant_store
        self.policy = ProjectionPolicy() if policy is None else policy
        if not isinstance(self.policy, ProjectionPolicy):
            raise TypeError("policy must be ProjectionPolicy")

    def project(
        self,
        snapshot: SharedFieldSnapshot,
        *,
        ancestor_field_ids: tuple[str, ...],
    ) -> ObservedField:
        result = build_active_projection(
            snapshot,
            policy=self.policy,
            store=self.dormant_store,
        )
        return ObservedField(
            snapshot=result.active_snapshot,
            ancestor_field_ids=(
                *ancestor_field_ids,
                snapshot.field_id,
            ),
            evidence_ids=(result.manifest.manifest_id,),
        )


@dataclass(frozen=True, slots=True)
class StagedCoreTurn:
    core_id: str
    proposal: RegionProposal
    observed_field_id: str
    observed_tick_id: int
    candidate_token: Any
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.core_id, str) or not self.core_id:
            raise RuntimeEngineError("staged core_id must be non-empty")
        if not isinstance(self.proposal, RegionProposal):
            raise TypeError("proposal must be RegionProposal")
        if (
            not isinstance(self.observed_field_id, str)
            or not self.observed_field_id
        ):
            raise RuntimeEngineError("observed_field_id must be non-empty")
        if (
            isinstance(self.observed_tick_id, bool)
            or not isinstance(self.observed_tick_id, int)
            or self.observed_tick_id < 0
        ):
            raise RuntimeEngineError("observed_tick_id must be non-negative")
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(sorted(set(self.evidence_ids))),
        )


@dataclass(frozen=True, slots=True)
class PreparedCoreCommit:
    core_id: str
    manifest: CoreStateManifest
    commit_token: Any

    def __post_init__(self) -> None:
        if self.core_id != self.manifest.core_id:
            raise RuntimeEngineError(
                "prepared commit core does not match its state manifest"
            )


class CoreRuntimeDriver(Protocol):
    """Two-phase private-state boundary used by real and deterministic cores."""

    def reconcile_head(self, head: RuntimeHead) -> None:
        """Make live private state exactly match the durable runtime head."""

    def stage_turn(
        self,
        *,
        core_id: str,
        observation: ObservedField,
        canonical_base_field_id: str,
        tick_seq: int,
        target_region: LogicalRegion,
    ) -> StagedCoreTurn:
        """Run inference without mutating or persisting live private state."""

    def prepare_commit(
        self,
        *,
        turn: StagedCoreTurn,
        accepted_text: str,
        output_field_id: str,
        runtime_generation: int,
        next_tick_seq: int,
        prior_manifest: CoreStateManifest,
    ) -> PreparedCoreCommit:
        """Persist content-addressed candidate state without installing it."""

    def finalize_commit(self, prepared: PreparedCoreCommit) -> None:
        """Install an already durably committed private-state candidate."""

    def recover_committed(self, prepared: PreparedCoreCommit) -> None:
        """Reload the committed candidate after an in-memory install failure."""

    def discard_turn(self, turn: StagedCoreTurn) -> None:
        """Forget an unaccepted staged candidate without changing live state."""


class SystemUpdateProvider(Protocol):
    def __call__(
        self,
        head: RuntimeHead,
    ) -> "SystemFieldUpdate | PreparedSystemUpdate":
        """Build the sealed same-tick update for the exact current head."""


@dataclass(frozen=True, slots=True)
class PreparedSystemUpdate:
    update: SystemFieldUpdate
    ingress_queue: IngressQueue | None = None
    consumed_ingress_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.update, SystemFieldUpdate):
            raise TypeError("update must be SystemFieldUpdate")
        identifiers = tuple(self.consumed_ingress_event_ids)
        if len(identifiers) != len(set(identifiers)) or any(
            not isinstance(item, str) or not item for item in identifiers
        ):
            raise RuntimeEngineError(
                "consumed ingress event IDs must be unique and non-empty"
            )
        if identifiers:
            if not isinstance(self.ingress_queue, IngressQueue):
                raise RuntimeEngineError(
                    "consumed ingress events require an IngressQueue"
                )
            if not set(identifiers).issubset(set(self.update.evidence)):
                raise RuntimeEngineError(
                    "consumed ingress events are absent from update evidence"
                )
        object.__setattr__(self, "consumed_ingress_event_ids", identifiers)


class IngressSystemUpdateProvider:
    """Drain a bounded durable FIFO into the next atomic field transaction."""

    def __init__(
        self,
        queue: IngressQueue,
        *,
        limit: int = 64,
    ) -> None:
        if not isinstance(queue, IngressQueue):
            raise TypeError("queue must be IngressQueue")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be positive")
        self.queue = queue
        self.limit = limit

    def __call__(self, head: RuntimeHead) -> PreparedSystemUpdate:
        events = self.queue.pending(limit=self.limit)
        update = build_ingress_update(head.snapshot, events)
        return PreparedSystemUpdate(
            update=update,
            ingress_queue=self.queue,
            consumed_ingress_event_ids=tuple(
                event.event_id for event in events
            ),
        )


ConsolidationGate = Callable[
    [RuntimeHead, StagedCoreTurn, StagedCoreTurn],
    tuple[bool, str],
]


def empty_system_update(head: RuntimeHead) -> SystemFieldUpdate:
    return SystemFieldUpdate(
        base_field_id=head.snapshot.field_id,
        base_tick_id=head.snapshot.tick_id,
    )


def accept_valid_consolidation(
    _head: RuntimeHead,
    _proposal: StagedCoreTurn,
    _consolidation: StagedCoreTurn,
) -> tuple[bool, str]:
    return True, "assigned consolidator authored the final response draft"


def _replacement_delta(
    snapshot: SharedFieldSnapshot,
    turn: StagedCoreTurn,
    *,
    pass_id: str,
) -> FieldDelta | None:
    target = turn.proposal.target_region
    prior = snapshot.region(target).text
    text = turn.proposal.text
    if text == prior:
        return None
    return FieldDelta(
        base_field_id=snapshot.field_id,
        base_tick_id=snapshot.tick_id,
        author_core_id=turn.core_id,
        pass_id=pass_id,
        operations=(
            ReplaceText(
                region=target,
                start=0,
                end=len(prior),
                text=text,
                provenance=(
                    turn.proposal.provenance
                    or f"axon-runtime:{turn.core_id}:{pass_id}"
                ),
                edge_refs=turn.proposal.evidence_refs,
            ),
        ),
        evidence=turn.proposal.evidence_refs,
    )


def _proposal_board(
    working: SharedFieldSnapshot,
    delta: FieldDelta | None,
) -> SharedFieldSnapshot:
    if delta is None:
        return working
    applied = apply_delta(working, delta)
    # The proposal board is not a canonical tick.  Preserve its exact content
    # while making W its direct parent for authenticated observation lineage.
    return SharedFieldSnapshot(
        tick_id=working.tick_id,
        regions=applied.regions,
        parent_field_id=working.field_id,
        source_manifest_ids=working.source_manifest_ids,
    )


def _tool_requests(
    text: str,
    *,
    tick_seq: int,
    consolidator_core_id: str,
    working_field_id: str,
    proposal_id: str,
) -> tuple[ToolRequestRecord, ...]:
    try:
        batch = parse_invocation_envelope(
            text,
            source_kind="consolidator_output",
        )
    except (InvocationParseError, InvocationValidationError):
        # Malformed marker text has no authority and no effect.  The response
        # text itself remains ordinary committed field content.
        return ()
    if batch is None:
        return ()
    return tuple(
        ToolRequestRecord(
            tick_seq=tick_seq,
            requester_core_id=consolidator_core_id,
            base_field_id=working_field_id,
            tool_name=action.kind,
            arguments={
                "schema": "axon-runtime-invocation-outbox-v1",
                "marker": action.marker,
                "payload": dict(action.payload),
                "request_hash": action.request_hash,
                "batch_hash": batch.batch_hash,
                "source_kind": batch.source_kind,
            },
            parent_proposal_id=proposal_id,
        )
        for action in batch.actions
    )


@dataclass(frozen=True, slots=True)
class RuntimeTickResult:
    commit: TickCommitRecord
    assignment_hash: str
    proposer_core_id: str
    consolidator_core_id: str
    sleeper_core_id: str
    input_field_id: str
    working_field_id: str
    proposal_board_field_id: str
    output_field_id: str
    accepted: bool
    finalized_core_ids: tuple[str, ...]
    tool_request_ids: tuple[str, ...]
    post_commit_warnings: tuple[str, ...] = ()


PostCommitHook = Callable[[RuntimeTickResult, SharedFieldSnapshot, str], None]


class AxonRuntimeEngine:
    """Run exactly one durable transaction at a time, forever when requested."""

    def __init__(
        self,
        *,
        store: RuntimeStore,
        identities: Sequence[CoreIdentity],
        driver: CoreRuntimeDriver,
        system_updates: SystemUpdateProvider = empty_system_update,
        projector: ObservationProjector | None = None,
        consolidation_gate: ConsolidationGate = accept_valid_consolidation,
        post_commit_hook: PostCommitHook | None = None,
        on_post_commit_error: Callable[[Exception], None] | None = None,
        target_region: LogicalRegion = LogicalRegion.RESPONSE_DRAFT,
    ) -> None:
        if not isinstance(store, RuntimeStore):
            raise TypeError("store must be RuntimeStore")
        values = tuple(identities)
        self.store = store
        self.scheduler = RoleScheduler(values)
        self.identities = values
        self.driver = driver
        self.system_updates = system_updates
        self.projector = projector or FullFieldProjector()
        self.consolidation_gate = consolidation_gate
        self.post_commit_hook = post_commit_hook
        self.on_post_commit_error = on_post_commit_error
        self.last_post_commit_error: Exception | None = None
        if target_region not in {
            LogicalRegion.RESPONSE_DRAFT,
            LogicalRegion.SCRATCH,
        }:
            raise ValueError("target_region must be core-writable")
        self.target_region = target_region
        self._tick_lock = threading.Lock()

    def _observation(
        self,
        snapshot: SharedFieldSnapshot,
        ancestors: tuple[str, ...],
    ) -> ObservedField:
        observed = self.projector.project(
            snapshot,
            ancestor_field_ids=ancestors,
        )
        if not isinstance(observed, ObservedField):
            raise TypeError("projector must return ObservedField")
        is_exact_source = (
            observed.snapshot == snapshot
            and observed.ancestor_field_ids == ancestors
        )
        is_direct_projection = (
            observed.snapshot.field_id != snapshot.field_id
            and observed.snapshot.tick_id == snapshot.tick_id
            and observed.snapshot.parent_field_id == snapshot.field_id
            and observed.ancestor_field_ids == (*ancestors, snapshot.field_id)
        )
        if not (is_exact_source or is_direct_projection):
            raise RuntimeEngineError(
                "projector returned neither the exact source nor an "
                "authenticated direct projection"
            )
        return observed

    def _validate_staged_turn(
        self,
        turn: StagedCoreTurn,
        *,
        expected_core_id: str,
        observation: ObservedField,
    ) -> None:
        if not isinstance(turn, StagedCoreTurn):
            raise TypeError("driver must return StagedCoreTurn")
        if turn.core_id != expected_core_id:
            raise RuntimeEngineError("driver returned a turn for the wrong core")
        if (
            turn.observed_field_id != observation.snapshot.field_id
            or turn.observed_tick_id != observation.snapshot.tick_id
        ):
            raise RuntimeEngineError(
                "driver returned a turn for a stale or different observation"
            )
        if turn.proposal.target_region is not self.target_region:
            raise RuntimeEngineError(
                "driver returned a proposal for the wrong target region"
            )

    def tick_once(self) -> RuntimeTickResult:
        if not self._tick_lock.acquire(blocking=False):
            raise RuntimeEngineError("another tick is already in progress")
        staged: list[StagedCoreTurn] = []
        prepared: list[PreparedCoreCommit] = []
        committed = False
        try:
            head = self.store.recover()
            self.driver.reconcile_head(head)
            assignment = self.scheduler.assignment(head.tick_seq)
            supplied_update = self.system_updates(head)
            prepared_system = (
                supplied_update
                if isinstance(supplied_update, PreparedSystemUpdate)
                else PreparedSystemUpdate(update=supplied_update)
            )
            system_update = prepared_system.update
            working = apply_system_update(
                head.snapshot,
                system_update,
            ).working_snapshot

            proposer_observation = self._observation(
                working,
                (head.snapshot.field_id,),
            )
            proposer = self.driver.stage_turn(
                core_id=assignment.proposer_core_id,
                observation=proposer_observation,
                canonical_base_field_id=head.snapshot.field_id,
                tick_seq=head.tick_seq,
                target_region=self.target_region,
            )
            staged.append(proposer)
            self._validate_staged_turn(
                proposer,
                expected_core_id=assignment.proposer_core_id,
                observation=proposer_observation,
            )
            proposal_delta = _replacement_delta(
                working,
                proposer,
                pass_id=f"tick-{head.tick_seq:012d}:proposal",
            )
            proposal = ProposalRecord(
                tick_seq=head.tick_seq,
                assignment_hash=assignment.assignment_hash,
                proposer_core_id=assignment.proposer_core_id,
                system_update_id=system_update.update_id,
                base_field_id=working.field_id,
                base_tick_id=working.tick_id,
                delta=proposal_delta,
                evidence=(
                    *proposer.evidence_ids,
                    *proposer_observation.evidence_ids,
                ),
            )

            board = _proposal_board(working, proposal_delta)
            board_ancestors = (
                (head.snapshot.field_id,)
                if board.field_id == working.field_id
                else (head.snapshot.field_id, working.field_id)
            )
            consolidator_observation = self._observation(
                board,
                board_ancestors,
            )
            consolidator = self.driver.stage_turn(
                core_id=assignment.consolidator_core_id,
                observation=consolidator_observation,
                canonical_base_field_id=head.snapshot.field_id,
                tick_seq=head.tick_seq,
                target_region=self.target_region,
            )
            staged.append(consolidator)
            self._validate_staged_turn(
                consolidator,
                expected_core_id=assignment.consolidator_core_id,
                observation=consolidator_observation,
            )

            accepted, reason = self.consolidation_gate(
                head,
                proposer,
                consolidator,
            )
            if not isinstance(accepted, bool) or not isinstance(reason, str):
                raise RuntimeEngineError(
                    "consolidation gate returned an invalid decision"
                )
            committed_delta = (
                _replacement_delta(
                    working,
                    consolidator,
                    pass_id=f"tick-{head.tick_seq:012d}:consolidation",
                )
                if accepted
                else None
            )
            audit = compose_tick_transaction(
                head.snapshot,
                system_update,
                committed_delta,
                consolidator_author_core_id=(
                    assignment.consolidator_core_id
                    if committed_delta is not None
                    else None
                ),
            )
            output = audit.final_snapshot
            decision = ConsolidationDecision(
                tick_seq=head.tick_seq,
                assignment_hash=assignment.assignment_hash,
                consolidator_core_id=assignment.consolidator_core_id,
                proposal_id=proposal.proposal_id,
                base_field_id=working.field_id,
                accepted=accepted,
                output_field_id=output.field_id,
                committed_delta=committed_delta,
                reason=reason,
            )

            prior_by_id = {
                state.core_id: state for state in head.core_state_manifests
            }
            if accepted:
                accepted_turns = [consolidator]
                if (
                    proposer.proposal.target_region
                    is consolidator.proposal.target_region
                    and proposer.proposal.text == consolidator.proposal.text
                ):
                    accepted_turns.insert(0, proposer)
                prepared_core_ids: set[str] = set()
                prepared_manifest_ids: set[str] = set()
                for turn in accepted_turns:
                    item = self.driver.prepare_commit(
                        turn=turn,
                        accepted_text=turn.proposal.text,
                        output_field_id=output.field_id,
                        runtime_generation=head.generation + 1,
                        next_tick_seq=head.tick_seq + 1,
                        prior_manifest=prior_by_id[turn.core_id],
                    )
                    if not isinstance(item, PreparedCoreCommit):
                        raise TypeError(
                            "driver must return PreparedCoreCommit"
                        )
                    if (
                        item.core_id in prepared_core_ids
                        or item.manifest.manifest_id in prepared_manifest_ids
                    ):
                        raise RuntimeEngineError(
                            "driver returned duplicate prepared private state"
                        )
                    if item.core_id != turn.core_id:
                        raise RuntimeEngineError(
                            "prepared private state does not belong to its "
                            "originating turn"
                        )
                    prepared.append(item)
                    prepared_core_ids.add(item.core_id)
                    prepared_manifest_ids.add(item.manifest.manifest_id)

            updates_by_id = {
                item.core_id: item.manifest for item in prepared
            }
            projected_states = dict(prior_by_id)
            projected_states.update(updates_by_id)
            next_assignment = self.scheduler.assignment(head.tick_seq + 1)
            projection = ProjectionManifest(
                generation=head.generation + 1,
                tick_seq=head.tick_seq + 1,
                field_id=output.field_id,
                field_hash=output.canonical_hash,
                core_state_manifest_ids=tuple(
                    (core_id, state.manifest_id)
                    for core_id, state in projected_states.items()
                ),
                role_index=self.scheduler.role_index(head.tick_seq + 1),
                role_assignment_hash=next_assignment.assignment_hash,
                parent_projection_id=head.projection_manifest_id,
            )
            requests = (
                _tool_requests(
                    consolidator.proposal.text,
                    tick_seq=head.tick_seq,
                    consolidator_core_id=assignment.consolidator_core_id,
                    working_field_id=working.field_id,
                    proposal_id=proposal.proposal_id,
                )
                if accepted
                else ()
            )
            commit = self.store.commit_tick(
                expected_generation=head.generation,
                expected_field_id=head.snapshot.field_id,
                assignment=assignment,
                system_update=system_update,
                proposal=proposal,
                decision=decision,
                output_snapshot=output,
                updated_core_states=tuple(
                    item.manifest for item in prepared
                ),
                projection_manifest=projection,
                queued_tool_requests=requests,
                ingress_queue=prepared_system.ingress_queue,
                consumed_ingress_event_ids=(
                    prepared_system.consumed_ingress_event_ids
                ),
            )
            committed = True

            prepared_manifest_ids = {
                item.manifest.manifest_id for item in prepared
            }
            committed_manifest_ids = set(
                commit.updated_core_state_manifest_ids
            )
            if committed_manifest_ids != prepared_manifest_ids:
                raise CommittedStateRecoveryError(
                    "tick committed with a different private-state manifest "
                    "set than the engine prepared"
                )

            finalized: list[str] = []
            for item in prepared:
                if item.manifest.manifest_id not in committed_manifest_ids:
                    continue
                try:
                    self.driver.finalize_commit(item)
                except Exception as install_error:
                    try:
                        self.driver.recover_committed(item)
                    except Exception as recovery_error:
                        raise CommittedStateRecoveryError(
                            "tick committed, but private state for "
                            f"{item.core_id!r} could neither install nor reload"
                        ) from recovery_error
                finalized.append(item.core_id)
            post_commit_warnings: list[str] = []
            for turn in staged:
                if turn.core_id not in finalized:
                    try:
                        self.driver.discard_turn(turn)
                    except Exception as exc:
                        post_commit_warnings.append(
                            "discard_turn "
                            f"{turn.core_id}: {type(exc).__name__}: {exc}"
                        )

            result = RuntimeTickResult(
                commit=commit,
                assignment_hash=assignment.assignment_hash,
                proposer_core_id=assignment.proposer_core_id,
                consolidator_core_id=assignment.consolidator_core_id,
                sleeper_core_id=assignment.sleeper_core_id,
                input_field_id=head.snapshot.field_id,
                working_field_id=working.field_id,
                proposal_board_field_id=board.field_id,
                output_field_id=output.field_id,
                accepted=accepted,
                finalized_core_ids=tuple(finalized),
                tool_request_ids=tuple(
                    request.request_id for request in requests
                ),
                post_commit_warnings=tuple(post_commit_warnings),
            )
            self.last_post_commit_error = None
            if self.post_commit_hook is not None:
                try:
                    self.post_commit_hook(
                        result,
                        output,
                        assignment.sleeper_core_id,
                    )
                except Exception as exc:
                    # The canonical tick and private-state manifests are
                    # already committed.  Treat optional bounded sleep work as
                    # retryable post-commit work, never as a failed tick.
                    self.last_post_commit_error = exc
                    post_commit_warnings.append(
                        "post_commit_hook: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if self.on_post_commit_error is not None:
                        try:
                            self.on_post_commit_error(exc)
                        except Exception as handler_error:
                            post_commit_warnings.append(
                                "on_post_commit_error: "
                                f"{type(handler_error).__name__}: "
                                f"{handler_error}"
                            )
            if tuple(post_commit_warnings) != result.post_commit_warnings:
                result = replace(
                    result,
                    post_commit_warnings=tuple(post_commit_warnings),
                )
            return result
        finally:
            if not committed:
                for turn in staged:
                    try:
                        self.driver.discard_turn(turn)
                    except Exception:
                        pass
            self._tick_lock.release()

    def run_forever(
        self,
        *,
        stop_event: threading.Event | None = None,
        stop_file: str | Path | None = None,
        tick_interval_seconds: float = 0.0,
        error_backoff_initial_seconds: float = 0.25,
        error_backoff_max_seconds: float = 30.0,
        error_backoff_multiplier: float = 2.0,
        stop_poll_interval_seconds: float = 0.05,
        max_ticks: int | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> tuple[RuntimeTickResult, ...]:
        """Tick until stopped, without retaining an unbounded result history.

        Bounded ``max_ticks`` runs return their collected results.  Continuous
        runs stream through the post-commit hook and return an empty tuple when
        stopped so memory use does not grow with runtime duration.
        """

        for name, value in (
            ("tick_interval_seconds", tick_interval_seconds),
            ("error_backoff_initial_seconds", error_backoff_initial_seconds),
        ):
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            not isinstance(error_backoff_max_seconds, (int, float))
            or error_backoff_max_seconds <= 0
        ):
            raise ValueError("error_backoff_max_seconds must be positive")
        if (
            not isinstance(error_backoff_multiplier, (int, float))
            or error_backoff_multiplier < 1
        ):
            raise ValueError("error_backoff_multiplier must be at least 1")
        if (
            not isinstance(stop_poll_interval_seconds, (int, float))
            or stop_poll_interval_seconds <= 0
        ):
            raise ValueError("stop_poll_interval_seconds must be positive")
        if error_backoff_initial_seconds > error_backoff_max_seconds:
            raise ValueError("initial error backoff exceeds maximum")
        if max_ticks is not None and (
            isinstance(max_ticks, bool)
            or not isinstance(max_ticks, int)
            or max_ticks < 0
        ):
            raise ValueError("max_ticks must be non-negative or None")

        stopper = stop_event or threading.Event()
        stop_path = None if stop_file is None else Path(stop_file)
        results: list[RuntimeTickResult] = []
        completed_ticks = 0
        backoff = float(error_backoff_initial_seconds)

        def should_stop() -> bool:
            return stopper.is_set() or (
                stop_path is not None and stop_path.exists()
            )

        def wait_interruptibly(seconds: float) -> bool:
            deadline = time.monotonic() + seconds
            while True:
                if should_stop():
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return should_stop()
                if stopper.wait(
                    min(remaining, float(stop_poll_interval_seconds))
                ):
                    return True

        while not should_stop():
            if max_ticks is not None and completed_ticks >= max_ticks:
                break
            try:
                result = self.tick_once()
            except CommittedStateRecoveryError:
                # Journal truth has advanced but this process cannot reproduce
                # the referenced private state.  A clean process restart may
                # recover it; blindly ticking with stale RAM is forbidden.
                raise
            except Exception as exc:
                if on_error is not None:
                    on_error(exc)
                wait = min(backoff, float(error_backoff_max_seconds))
                if wait_interruptibly(wait):
                    break
                backoff = min(
                    max(
                        backoff * float(error_backoff_multiplier),
                        0.001,
                    ),
                    float(error_backoff_max_seconds),
                )
                continue
            completed_ticks += 1
            if max_ticks is not None:
                results.append(result)
            backoff = float(error_backoff_initial_seconds)
            if tick_interval_seconds and wait_interruptibly(
                tick_interval_seconds
            ):
                break
        return tuple(results)


__all__ = [
    "RuntimeEngineError",
    "CommittedStateRecoveryError",
    "ObservedField",
    "ObservationProjector",
    "FullFieldProjector",
    "ActiveFieldProjector",
    "StagedCoreTurn",
    "PreparedCoreCommit",
    "CoreRuntimeDriver",
    "SystemUpdateProvider",
    "PreparedSystemUpdate",
    "IngressSystemUpdateProvider",
    "ConsolidationGate",
    "empty_system_update",
    "accept_valid_consolidation",
    "RuntimeTickResult",
    "PostCommitHook",
    "AxonRuntimeEngine",
]

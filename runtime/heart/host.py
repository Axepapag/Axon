"""Permanent single-writer Heart host for Axon.

The host owns cadence, durable intake, the sovereign valve plane, and runtime
observability.  It deliberately does *not* own a second canonical mutation
path: every accepted mutation still crosses the existing
``HeartTransactionBoundary`` and ``CanonicalStateBranch`` owned by
``BeatCoordinator``.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from runtime.field import (
    CanonicalStateBranch,
    CompiledD64Field,
    FieldDelta,
    InsertText,
    LogicalRegion,
    RegionMaskPolicy,
    ReplaceText,
    SharedFieldSnapshot,
)
from runtime.field.state_branch import DEFAULT_STATE_ROOT
from runtime.soul import SoulStore
from runtime.source_of_truth import capacity_policy

from .authority import AuthorityClass, AuthorityGrant
from .autobiography import HeartAutobiography
from .circulation import (
    ReasoningCirculation,
    ReasoningCirculationResult,
    ReasoningCorePort,
)
from .coordinator import BeatConfig, BeatCoordinator
from .durable_ingress import DurableIngressSpool, IngressRecord
from .errors import (
    HealthCorruptionError,
    HostStateError,
    UnknownValveError,
)
from .health import HealthJournal, HeartHealth
from .identity import HeartIdentityStore
from .lease import SingleWriterLease
from .masks import HeartRegionMaskController, HeartRegionMaskState
from .reasoning_recovery import (
    ReasoningAutobiographyRecoveryStore,
    attention_view_from_surface,
)
from .registry import CoreRegistry
from .tick import FrozenTickImage, TickIdentity
from .transaction import HeartCommit
from .valve import (
    HeartValveRegistry,
    ValveBudget,
    ValveDecision,
    ValveEnvelope,
    ValveReceipt,
    primitive_valve_registry,
)


class HostBeatState(str, Enum):
    IDLE = "idle"
    CIRCULATED = "circulated"
    TICK_IN_FLIGHT = "tick_in_flight"


@dataclass(frozen=True, slots=True)
class HeartHostConfig:
    """Permanent-host cadence and renewable global work allocation."""

    idle_interval_seconds: float = field(
        default_factory=lambda: capacity_policy().number("heart.idle_interval_seconds")
    )
    global_budget: ValveBudget = field(
        default_factory=lambda: ValveBudget(
            items_per_beat=capacity_policy().integer("heart.global_items_per_beat"),
            target_chars_per_beat=capacity_policy().integer(
                "heart.global_target_chars_per_beat"
            ),
        )
    )
    auto_close_null_ticks: bool = True
    failure_backoff_seconds: float = field(
        default_factory=lambda: capacity_policy().number("heart.failure_backoff_seconds")
    )

    def __post_init__(self) -> None:
        if not isinstance(self.idle_interval_seconds, (int, float)):
            raise TypeError("idle_interval_seconds must be numeric")
        if self.idle_interval_seconds <= 0:
            raise ValueError("idle_interval_seconds must be positive")
        if not isinstance(self.global_budget, ValveBudget):
            raise TypeError("global_budget must be ValveBudget")
        if not isinstance(self.failure_backoff_seconds, (int, float)):
            raise TypeError("failure_backoff_seconds must be numeric")
        if self.failure_backoff_seconds <= 0:
            raise ValueError("failure_backoff_seconds must be positive")


@dataclass(frozen=True, slots=True)
class HeartHostBeatResult:
    heartbeat_sequence: int
    state: HostBeatState
    field: SharedFieldSnapshot
    commits: tuple[HeartCommit, ...]
    processed_event_ids: tuple[str, ...]
    tick_image: FrozenTickImage | None = None
    deferred_event_id: str | None = None
    reasoning_result: ReasoningCirculationResult | None = None


class HeartHost:
    """Long-running sovereign Heart process over the one active branch."""

    def __init__(
        self,
        *,
        state_root: Path | str = DEFAULT_STATE_ROOT,
        host_config: HeartHostConfig | None = None,
        beat_config: BeatConfig | None = None,
        valve_registry: HeartValveRegistry | None = None,
        core_registry: CoreRegistry | None = None,
        reasoning_ports: Iterable[ReasoningCorePort] = (),
    ) -> None:
        self.state_root = Path(state_root).resolve(strict=False)
        self.host_config = host_config or HeartHostConfig()
        self.beat_config = beat_config or BeatConfig()
        self.valves = valve_registry or primitive_valve_registry()
        self.cores = core_registry or CoreRegistry()
        self._reasoning_ports = tuple(reasoning_ports)
        self.heart_dir = self.state_root / "active" / "heart"
        self._lease = SingleWriterLease(self.state_root)
        self._identity_store: HeartIdentityStore | None = None
        self._spool: DurableIngressSpool | None = None
        self._autobiography: HeartAutobiography | None = None
        self._health_journal: HealthJournal | None = None
        self._coordinator: BeatCoordinator | None = None
        self._circulation: ReasoningCirculation | None = None
        self._mask_controller: HeartRegionMaskController | None = None
        self._soul_store: SoulStore | None = None
        self._reasoning_recovery: ReasoningAutobiographyRecoveryStore | None = None
        self._mask_dirty = False
        self._last_circulated_view_id: str | None = None
        self._lease_record: dict | None = None
        self._started = False
        self._last_successful_circulation_at: datetime | None = None
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def coordinator(self) -> BeatCoordinator:
        if self._coordinator is None:
            raise HostStateError("Heart host has not been started")
        return self._coordinator

    @property
    def spool(self) -> DurableIngressSpool:
        if self._spool is None:
            raise HostStateError("Heart host has not been started")
        return self._spool

    @property
    def mask_controller(self) -> HeartRegionMaskController:
        if self._mask_controller is None:
            raise HostStateError("Heart region-mask controller has not been started")
        return self._mask_controller

    @property
    def identity_store(self) -> HeartIdentityStore:
        if self._identity_store is None:
            raise HostStateError("Heart host has not been started")
        return self._identity_store

    @property
    def soul_store(self) -> SoulStore:
        if self._soul_store is None:
            raise HostStateError("Heart private-soul store has not been started")
        return self._soul_store

    def start(self) -> None:
        """Acquire writer authority and bind to ``State/active``."""

        with self._lock:
            if self._started:
                return
            lease_record = self._lease.acquire()
            try:
                self.heart_dir.mkdir(parents=True, exist_ok=True)
                identity_store = HeartIdentityStore(self.heart_dir / "identity.json")
                identity_store.begin_start()
                spool = DurableIngressSpool(self.heart_dir / "ingress")
                health = HealthJournal(self.heart_dir)
                mask_controller = HeartRegionMaskController(
                    self.heart_dir / "region_masks.json",
                    initial_policies=self.beat_config.region_policies,
                )
                branch = CanonicalStateBranch.active_runtime(
                    branch_id="active",
                    state_root=self.state_root,
                )
                coordinator = BeatCoordinator(
                    branch,
                    self.cores,
                    state_root=self.state_root,
                    config=self.beat_config,
                    mask_controller=mask_controller,
                )
                autobiography = HeartAutobiography(self.state_root)
                soul_store = SoulStore.active(self.state_root)
                reasoning_recovery = ReasoningAutobiographyRecoveryStore(self.state_root)
                circulation = (
                    None
                    if not self._reasoning_ports
                    else ReasoningCirculation(
                        coordinator,
                        self.cores,
                        self._reasoning_ports,
                        soul_store=soul_store,
                        recovery_store=reasoning_recovery,
                    )
                )
            except Exception:
                self._lease.release()
                raise
            self._identity_store = identity_store
            self._spool = spool
            self._autobiography = autobiography
            self._health_journal = health
            self._coordinator = coordinator
            self._circulation = circulation
            self._mask_controller = mask_controller
            self._soul_store = soul_store
            self._reasoning_recovery = reasoning_recovery
            self._lease_record = lease_record
            self._started = True
            try:
                self._recover_reasoning_autobiography()
            except Exception:
                self.stop()
                raise
            self._stop_event.clear()
            self._wake_event.clear()
            latest = health.latest()
            prior = latest.get("last_successful_circulation_at") if latest else None
            if prior:
                try:
                    self._last_successful_circulation_at = datetime.fromisoformat(prior)
                except (TypeError, ValueError):
                    self._last_successful_circulation_at = None
            prior_view_id = latest.get("last_view_id") if latest else None
            self._last_circulated_view_id = prior_view_id
            self._mask_dirty = (prior_view_id is not None and prior_view_id != coordinator.current_view_id) or (
                prior_view_id is None and mask_controller.state.revision > 0
            )
            self._write_health(
                last_beat_at=None,
                last_failure_reason=None,
                last_tick=None,
            )

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            self._wake_event.set()
            self._started = False
            self._coordinator = None
            self._circulation = None
            self._mask_controller = None
            self._soul_store = None
            self._reasoning_recovery = None
            self._mask_dirty = False
            self._last_circulated_view_id = None
            self._spool = None
            self._autobiography = None
            self._health_journal = None
            self._identity_store = None
            self._lease_record = None
            self._lease.release()

    def request_stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

    def __enter__(self) -> "HeartHost":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def submit_user(self, text: str, *, provenance: str = "user") -> ValveDecision:
        return self.submit(
            ValveEnvelope(
                valve_id="user_ingress",
                source_id="external_user",
                payload=text,
                provenance=provenance,
                envelope_type="text/plain",
            )
        )

    def region_mask_state(self) -> HeartRegionMaskState:
        """Return the durable complete per-region mask-control state."""

        with self._lock:
            self._require_started()
            return self.mask_controller.state

    def set_region_mask_policy(
        self,
        region: LogicalRegion | str,
        policy: RegionMaskPolicy,
    ) -> HeartRegionMaskState:
        """Move one region's mask without touching canonical field content.

        An in-flight frozen tick is never mutated.  The new policy is durable
        immediately and is circulated on the next eligible heartbeat.
        """

        with self._lock:
            self._require_started()
            before = self.mask_controller.state.state_id
            state = self.mask_controller.set_policy(region, policy)
            if state.state_id != before:
                self._mask_dirty = True
                self._wake_event.set()
            return state

    def set_region_unmasked_percent(
        self,
        region: LogicalRegion | str,
        percent: int,
    ) -> HeartRegionMaskState:
        """Set one region's independent newest-suffix slider to 0..100%."""

        return self.set_region_mask_policy(
            region,
            RegionMaskPolicy("tail_percent", percent),
        )

    def amend_identity(
        self,
        text: str,
        *,
        amendment_id: str,
        evidence_ids: Iterable[str],
        provenance: str,
        occurred_at: datetime | None = None,
    ) -> HeartCommit:
        """Apply one exceptional, versioned canonical Identity amendment.

        Identity is never writable through ordinary ingress, core, or
        consolidator authority.  This explicit boundary runs only between
        ticks and deposits the accepted amendment into exact autobiography.
        """

        if not isinstance(text, str) or not text:
            raise ValueError("canonical identity text must be non-empty")
        if not isinstance(amendment_id, str) or not amendment_id:
            raise ValueError("amendment_id must be non-empty")
        if not isinstance(provenance, str) or not provenance:
            raise ValueError("identity amendment provenance must be non-empty")
        evidence = tuple(sorted(set(map(str, evidence_ids))))
        if not evidence or any(not item for item in evidence):
            raise ValueError("identity amendments require explicit evidence ids")
        with self._lock:
            self._require_started()
            if self.coordinator.tick_in_flight:
                raise HostStateError("canonical identity cannot change during an in-flight tick")
            base = self.coordinator.current_field
            current = base.region(LogicalRegion.IDENTITY).text
            if current == text:
                raise ValueError("identity amendment cannot be a no-op")
            delta = FieldDelta(
                base_field_id=base.field_id,
                base_tick_id=base.tick_id,
                author_core_id="identity-steward",
                pass_id=f"identity-amendment:{amendment_id}",
                operations=(
                    ReplaceText(
                        region=LogicalRegion.IDENTITY,
                        start=0,
                        end=len(current),
                        text=text,
                        provenance=provenance,
                    ),
                ),
                evidence=evidence,
            )
            commit = self.coordinator.commit_heart_delta(
                base,
                delta,
                AuthorityGrant.identity_steward(),
                valve_provenance={
                    "authority_class": AuthorityClass.IDENTITY_STEWARD.value,
                    "amendment_id": amendment_id,
                    "provenance": provenance,
                    "evidence_ids": list(evidence),
                },
            )
            if self._autobiography is None:
                raise HostStateError("Heart autobiography has not been started")
            when = occurred_at or datetime.now(timezone.utc)
            self._autobiography.deposit_event(
                event_id=f"identity-{delta.delta_id}",
                record_kind="canonical_identity_amendment",
                exact_text=text,
                payload={
                    "amendment_id": amendment_id,
                    "provenance": provenance,
                    "evidence_ids": list(evidence),
                    "base_field_id": base.field_id,
                    "successor_field_id": commit.successor.field_id,
                    "delta_id": delta.delta_id,
                },
                occurred_at=when.isoformat(),
            )
            self._wake_event.set()
            return commit

    def submit_tool(self, text: str, *, provenance: str = "tool") -> ValveDecision:
        return self.submit(
            ValveEnvelope(
                valve_id="tool_ingress",
                source_id="external_tool",
                payload=text,
                provenance=provenance,
                envelope_type="text/plain",
            )
        )

    def submit_advisor(self, text: str, *, provenance: str = "advisor") -> ValveDecision:
        return self.submit(
            ValveEnvelope(
                valve_id="advisor_ingress",
                source_id="external_advisor",
                payload=text,
                provenance=provenance,
                envelope_type="text/plain",
            )
        )

    def submit(self, envelope: ValveEnvelope) -> ValveDecision:
        """Valve-local admission followed by durable spool append.

        This method never constructs an ``AuthorityGrant`` and never touches
        canonical state.  ``dormant_recall`` is internal Heart circulation and
        cannot be injected through the public spool.
        """

        with self._lock:
            self._require_started()
            if not isinstance(envelope, ValveEnvelope):
                raise TypeError("HeartHost.submit requires a ValveEnvelope")
            if envelope.valve_id == "dormant_recall":
                decision = ValveDecision(
                    admitted=False,
                    reason="dormant_recall is Heart-internal",
                    quarantine=False,
                    receipt=None,
                )
                self.spool.reject_envelope(
                    envelope,
                    reason=decision.reason,
                    valve_version=1,
                    quarantine=False,
                )
                return decision
            try:
                definition = self.valves.get(envelope.valve_id)
                version = definition.version
                pending = self.spool.pending_count_for_valve(envelope.valve_id)
            except UnknownValveError:
                decision = ValveDecision(
                    admitted=False,
                    reason=f"unknown valve_id {envelope.valve_id!r}",
                    quarantine=False,
                    receipt=None,
                )
                self.spool.reject_envelope(
                    envelope,
                    reason=decision.reason,
                    valve_version=0,
                    quarantine=False,
                )
                return decision
            decision = self.valves.local_validate(envelope, pending_count=pending)
            if not decision.admitted:
                self.spool.reject_envelope(
                    envelope,
                    reason=decision.reason,
                    valve_version=version,
                    quarantine=decision.quarantine,
                )
                return decision
            local = ValveDecision(True, "admitted_local", False, None)
            record = self.spool.submit(
                envelope,
                decision=local,
                valve_version=version,
            )
            receipt = ValveReceipt(
                valve_id=definition.valve_id,
                valve_version=definition.version,
                source_id=envelope.source_id,
                item_id=record.event_id,
                authority_class=definition.authority_class,
                governed_regions=definition.governed_regions,
                enqueued_at=datetime.fromisoformat(record.enqueued_at),
            )
            self._wake_event.set()
            return ValveDecision(True, "admitted", False, receipt)

    def _deposit_reasoning_episode(
        self,
        base: SharedFieldSnapshot,
        result: ReasoningCirculationResult,
        exact_surface: CompiledD64Field,
        *,
        occurred_at: datetime,
    ):
        if self._autobiography is None:
            raise HostStateError("Heart autobiography has not been started")
        response_text = result.commit.successor.region(LogicalRegion.RESPONSE_DRAFT).text
        return self._autobiography.deposit_reasoning_episode(
            event_id=f"reasoning-{result.result_id}",
            response_text=response_text,
            occurred_at=occurred_at.isoformat(),
            payload={
                "schema": "axon-runtime-reasoning-episode-v3",
                "conversation_id": self.identity_store.identity.heart_epoch_id,
                "pre_action_field": base.to_dict(),
                "attention_view": attention_view_from_surface(
                    exact_surface,
                    view_id=result.image.view_id,
                ),
                "circulation": result.to_canonical_dict(),
                "response_text_sha256": hashlib.sha256(
                    response_text.encode("utf-8")
                ).hexdigest(),
                "outcome_quality": "observed",
                "outcome_evidence_ids": [],
            },
        )

    def _complete_reasoning_tick(
        self,
        *,
        occurred_at: datetime,
    ) -> ReasoningCirculationResult:
        if self._circulation is None:
            raise HostStateError("active reasoning cores have no configured runtime ports")
        base = self.coordinator.current_field
        exact_surface = self.coordinator.rail_surface(64).exact
        result = self._circulation.run(occurred_at=occurred_at.isoformat())
        receipt = self._deposit_reasoning_episode(
            base,
            result,
            exact_surface,
            occurred_at=occurred_at,
        )
        if self._reasoning_recovery is None:
            raise HostStateError("reasoning recovery store has not been started")
        preparation = self._reasoning_recovery.for_materialized_delta(
            result.materialized_delta.delta_id
        )
        self._reasoning_recovery.mark_completed(
            preparation.preparation_id,
            result_id=result.result_id,
            event_id=receipt.event_id,
            record_id=receipt.record_id,
            import_id=receipt.import_id,
        )
        return result

    def _recover_reasoning_autobiography(self) -> None:
        """Finish any canonical commit whose autobiography deposit was interrupted."""

        if (
            self._reasoning_recovery is None
            or self._autobiography is None
            or self._coordinator is None
            or self._soul_store is None
            or self._identity_store is None
        ):
            raise HostStateError("reasoning autobiography recovery dependencies are unavailable")
        for preparation in self._reasoning_recovery.pending():
            recovered = self._reasoning_recovery.reconstruct_if_committed(
                preparation,
                branch=self._coordinator.branch,
                soul_store=self._soul_store,
                conversation_id=self._identity_store.identity.heart_epoch_id,
            )
            if recovered is None:
                continue
            receipt = self._autobiography.deposit_reasoning_episode(
                event_id=recovered.event_id,
                response_text=recovered.response_text,
                payload=recovered.payload,
                occurred_at=recovered.occurred_at,
            )
            self._reasoning_recovery.mark_completed(
                preparation.preparation_id,
                result_id=recovered.result_id,
                event_id=receipt.event_id,
                record_id=receipt.record_id,
                import_id=receipt.import_id,
            )

    def record_episode_outcome(
        self,
        *,
        event_id: str,
        episode_event_id: str,
        outcome_quality: str,
        evidence_ids: Iterable[str],
        detail: str,
        occurred_at: datetime | None = None,
        target_scope: str = "final_delta",
        corrected_source_delta: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach explicit outcome evidence without relabeling observation as truth."""

        with self._lock:
            self._require_started()
            if self._autobiography is None:
                raise HostStateError("Heart autobiography has not been started")
            when = occurred_at or datetime.now(timezone.utc)
            self._autobiography.deposit_episode_outcome(
                event_id=event_id,
                episode_event_id=episode_event_id,
                outcome_quality=outcome_quality,
                evidence_ids=tuple(evidence_ids),
                detail=detail,
                occurred_at=when.isoformat(),
                target_scope=target_scope,
                corrected_source_delta=corrected_source_delta,
            )

    def record_tool_invocation(
        self,
        *,
        event_id: str,
        exact_text: str,
        payload: Mapping[str, Any],
        occurred_at: datetime | None = None,
    ) -> None:
        """Freeze one completed tool call/result pair as exact lived evidence."""

        with self._lock:
            self._require_started()
            if self._autobiography is None:
                raise HostStateError("Heart autobiography has not been started")
            when = occurred_at or datetime.now(timezone.utc)
            self._autobiography.deposit_tool_invocation(
                event_id=event_id,
                exact_text=exact_text,
                payload=payload,
                occurred_at=when.isoformat(),
            )

    def record_trainer_outcome(
        self,
        *,
        event_id: str,
        exact_text: str,
        payload: Mapping[str, Any],
        occurred_at: datetime | None = None,
    ) -> None:
        """Freeze one governed Trainer attempt and outcome into autobiography."""

        with self._lock:
            self._require_started()
            if self._autobiography is None:
                raise HostStateError("Heart autobiography has not been started")
            when = occurred_at or datetime.now(timezone.utc)
            self._autobiography.deposit_trainer_outcome(
                event_id=event_id,
                exact_text=exact_text,
                payload=payload,
                occurred_at=when.isoformat(),
            )

    def heartbeat(self, *, force: bool = False) -> HeartHostBeatResult:
        """Execute one permanent-host heartbeat against the canonical active body."""

        with self._lock:
            self._require_started()
            beat_identity = self.identity_store.next_heartbeat()
            heartbeat_sequence = beat_identity.heartbeat_sequence
            now = datetime.now(timezone.utc)
            commits: list[HeartCommit] = []
            processed_ids: list[str] = []
            deferred_event_id: str | None = None
            tick_image: FrozenTickImage | None = None
            reasoning_result: ReasoningCirculationResult | None = None
            try:
                coordinator = self.coordinator
                if coordinator.tick_in_flight:
                    active_cores = self.cores.active()
                    if active_cores and self._circulation is not None:
                        tick_image = coordinator.open_tick_image
                        reasoning_result = self._complete_reasoning_tick(occurred_at=now)
                        commits.append(reasoning_result.commit)
                        field = coordinator.sync_to_branch_head()
                        self._last_successful_circulation_at = now
                        self._write_health(
                            last_beat_at=now,
                            last_failure_reason=None,
                            last_tick=tick_image,
                        )
                        return HeartHostBeatResult(
                            heartbeat_sequence=heartbeat_sequence,
                            state=HostBeatState.CIRCULATED,
                            field=field,
                            commits=tuple(commits),
                            processed_event_ids=(),
                            tick_image=tick_image,
                            reasoning_result=reasoning_result,
                        )
                    if active_cores or not self.host_config.auto_close_null_ticks:
                        field = coordinator.sync_to_branch_head()
                        self._write_health(
                            last_beat_at=now,
                            last_failure_reason=None,
                            last_tick=coordinator.open_tick_image,
                        )
                        return HeartHostBeatResult(
                            heartbeat_sequence=heartbeat_sequence,
                            state=HostBeatState.TICK_IN_FLIGHT,
                            field=field,
                            commits=(),
                            processed_event_ids=(),
                            tick_image=coordinator.open_tick_image,
                        )
                    coordinator.close_tick()

                self.valves.tracker.reset_beat()
                current = coordinator.sync_to_branch_head()
                starting_field_id = current.field_id

                while True:
                    pending = self.spool.pending(1)
                    if not pending:
                        break
                    record = pending[0]
                    existing_region = self._canonical_event_region(current, record.event_id)
                    if existing_region is not None:
                        self._deposit_accepted_ingress(record, existing_region)
                        self.spool.acknowledge(record.event_id)
                        processed_ids.append(record.event_id)
                        continue

                    final = self._final_gate(record)
                    if not final.admitted:
                        if self._is_budget_defer(final.reason):
                            deferred_event_id = record.event_id
                            break
                        self.spool.quarantine_front(record, f"final_gate:{final.reason}")
                        processed_ids.append(record.event_id)
                        continue

                    definition = self.valves.get(record.valve_id)
                    grant = self.valves.resolve_grant(record.valve_id, record.source_id)
                    if definition.authority_class is not AuthorityClass.EXTERNAL_INGRESS:
                        self.spool.quarantine_front(
                            record,
                            "final_gate:external spool valve is not external_ingress",
                        )
                        self.valves.tracker.complete(record.valve_id)
                        processed_ids.append(record.event_id)
                        continue
                    if len(definition.governed_regions) != 1:
                        raise HostStateError(f"primitive ingress valve {definition.valve_id!r} must govern one region")
                    region = next(iter(definition.governed_regions))
                    operation = InsertText(
                        region=region,
                        offset=len(current.region(region).text),
                        text=record.payload,
                        provenance=self._canonical_event_provenance(record),
                    )
                    delta = FieldDelta(
                        base_field_id=current.field_id,
                        base_tick_id=current.tick_id,
                        operations=(operation,),
                        author_core_id=f"heart-valve:{record.valve_id}",
                        pass_id="heart_ingress",
                    )
                    provenance = {
                        "valve_id": definition.valve_id,
                        "valve_version": definition.version,
                        "source_id": record.source_id,
                        "authority_class": definition.authority_class.value,
                        "governed_regions": sorted(item.value for item in definition.governed_regions),
                        "item_id": record.event_id,
                        "provenance": record.provenance,
                        "heartbeat_sequence": heartbeat_sequence,
                    }
                    try:
                        commit = coordinator.commit_heart_delta(
                            current,
                            delta,
                            grant,
                            valve_provenance=provenance,
                        )
                        current = coordinator.current_field
                        # Autobiography is part of acceptance durability.  Ack
                        # only after canonical persistence *and* the exact,
                        # idempotent Dormant deposit.  A failure leaves the event
                        # pending; canonical provenance reconciles replay.
                        self._deposit_accepted_ingress(record, region)
                        self.spool.acknowledge(record.event_id)
                        commits.append(commit)
                        processed_ids.append(record.event_id)
                    finally:
                        # Final-gate admission consumes an in-memory pending
                        # budget slot.  Durable spool state, not this counter,
                        # decides retry eligibility, so always release it.
                        self.valves.tracker.complete(record.valve_id)

                field_changed = current.field_id != starting_field_id
                if field_changed or force:
                    current, recall_commit = coordinator.stabilize_recall(current)
                    if recall_commit is not None:
                        commits.append(recall_commit)

                if commits or force or self._mask_dirty:
                    tick_reserved = self.identity_store.next_tick()
                    tick_identity = TickIdentity(
                        tick_sequence=tick_reserved.tick_sequence,
                        heartbeat_id=heartbeat_sequence,
                        base_field_id=current.field_id,
                        base_tick_id=current.tick_id,
                    )
                    tick_image = coordinator.freeze_tick(current, tick_identity)
                    active_cores = self.cores.active()
                    if active_cores and self._circulation is not None:
                        reasoning_result = self._complete_reasoning_tick(occurred_at=now)
                        commits.append(reasoning_result.commit)
                    elif not active_cores and self.host_config.auto_close_null_ticks:
                        coordinator.close_tick()
                    self._mask_dirty = False
                    self._last_circulated_view_id = tick_image.view_id
                    self._last_successful_circulation_at = now
                    state = HostBeatState.CIRCULATED
                else:
                    state = HostBeatState.IDLE

                current = coordinator.sync_to_branch_head()
                self._write_health(
                    last_beat_at=now,
                    last_failure_reason=None,
                    last_tick=tick_image,
                )
                return HeartHostBeatResult(
                    heartbeat_sequence=heartbeat_sequence,
                    state=state,
                    field=current,
                    commits=tuple(commits),
                    processed_event_ids=tuple(processed_ids),
                    tick_image=tick_image,
                    deferred_event_id=deferred_event_id,
                    reasoning_result=reasoning_result,
                )
            except Exception as exc:
                try:
                    field = self.coordinator.sync_to_branch_head()
                except Exception:
                    field = None
                self._write_health(
                    last_beat_at=now,
                    last_failure_reason=f"{type(exc).__name__}: {exc}",
                    last_tick=None,
                    field_override=field,
                )
                raise

    def run_forever(self) -> None:
        """Event-driven circulation with bounded idle liveness cadence."""

        if not self.started:
            self.start()
        try:
            while not self._stop_event.is_set():
                self._wake_event.wait(timeout=self.host_config.idle_interval_seconds)
                self._wake_event.clear()
                if self._stop_event.is_set():
                    break
                try:
                    self.heartbeat()
                except (HealthCorruptionError, HostStateError):
                    raise
                except Exception:
                    # Ordinary transient failures remain visible in health and
                    # retry forever after governed backoff.  Only integrity or
                    # ownership corruption halts the Heart.
                    time.sleep(self.host_config.failure_backoff_seconds)
        finally:
            self.stop()

    def health(self) -> dict:
        if self._health_journal is None:
            raise HostStateError("Heart host has not been started")
        return self._health_journal.latest()

    def _final_gate(self, record: IngressRecord) -> ValveDecision:
        try:
            definition = self.valves.get(record.valve_id)
        except UnknownValveError:
            return ValveDecision(False, "unknown valve_id", False, None)
        if record.valve_version != definition.version:
            return ValveDecision(False, "stale valve version", False, None)
        envelope = record.to_envelope()
        return self.valves.decide(
            envelope,
            global_budget=self.host_config.global_budget,
        )

    def _write_health(
        self,
        *,
        last_beat_at: datetime | None,
        last_failure_reason: str | None,
        last_tick: FrozenTickImage | None,
        field_override: SharedFieldSnapshot | None = None,
    ) -> None:
        if self._health_journal is None or self._identity_store is None:
            return
        identity = self._identity_store.identity
        field = field_override
        if field is None and self._coordinator is not None:
            try:
                field = self._coordinator.current_field
            except Exception:
                field = None
        pending_records = self._spool.pending(None) if self._spool is not None else ()
        queue_depths: dict[str, int] = {}
        for record in pending_records:
            queue_depths[record.valve_id] = queue_depths.get(record.valve_id, 0) + 1
        valve_states: dict[str, dict] = {}
        for definition in self.valves:
            valve_states[definition.valve_id] = {
                "state": definition.state.value,
                "version": definition.version,
                "authority_class": definition.authority_class.value,
                "governed_regions": sorted(region.value for region in definition.governed_regions),
                "budget": {
                    "items_per_beat": definition.budget.items_per_beat,
                    "target_chars_per_beat": definition.budget.target_chars_per_beat,
                },
                "usage": {
                    "pending": self.valves.tracker.pending(definition.valve_id),
                    "items_this_beat": self.valves.tracker.items_this_beat(definition.valve_id),
                    "chars_this_beat": self.valves.tracker.chars_this_beat(definition.valve_id),
                },
            }
        health = HeartHealth(
            heart_epoch_id=identity.heart_epoch_id,
            start_sequence=identity.start_sequence,
            heartbeat_sequence=identity.heartbeat_sequence,
            tick_sequence=identity.tick_sequence,
            last_beat_at=last_beat_at,
            last_successful_circulation_at=self._last_successful_circulation_at,
            last_failure_reason=last_failure_reason,
            canonical_head_field_id=None if field is None else field.field_id,
            canonical_head_tick_id=None if field is None else field.tick_id,
            tick_in_flight=(False if self._coordinator is None else self._coordinator.tick_in_flight),
            queue_depth_by_valve=queue_depths,
            quarantine_count=(0 if self._spool is None else self._spool.quarantine_count),
            rejection_count=(0 if self._spool is None else self._spool.rejection_count),
            valve_states=valve_states,
            dormant_index_id=(None if self._coordinator is None else self._coordinator.dormant_index_id),
            lease_owner_pid=(None if self._lease_record is None else self._lease_record.get("pid")),
            lease_owner_token=(None if self._lease_record is None else self._lease_record.get("owner_token")),
            last_tick_uid=None if last_tick is None else last_tick.identity.tick_uid,
            last_view_id=(self._last_circulated_view_id if last_tick is None else last_tick.view_id),
            mask_state_id=(None if self._mask_controller is None else self._mask_controller.state.state_id),
            mask_revision=(None if self._mask_controller is None else self._mask_controller.state.revision),
            region_masks=(
                None
                if self._mask_controller is None
                else {
                    region.value: self._mask_controller.state.policy_for(region).to_canonical_dict()
                    for region in self._mask_controller.state.policies
                }
            ),
        )
        self._health_journal.write(health)

    @staticmethod
    def _canonical_event_provenance(record: IngressRecord) -> str:
        suffix = record.provenance.replace("\n", " ").strip()
        return f"heart_ingress:{record.event_id}|{suffix}"

    @staticmethod
    def _canonical_contains_event(field: SharedFieldSnapshot, event_id: str) -> bool:
        return HeartHost._canonical_event_region(field, event_id) is not None

    @staticmethod
    def _canonical_event_region(
        field: SharedFieldSnapshot,
        event_id: str,
    ) -> LogicalRegion | None:
        prefix = f"heart_ingress:{event_id}|"
        for region in field.regions:
            if any(span.provenance.startswith(prefix) for span in region.spans):
                return region.name
        return None

    def _deposit_accepted_ingress(
        self,
        record: IngressRecord,
        region: LogicalRegion,
    ) -> None:
        if self._autobiography is None:
            raise HostStateError("Heart autobiography has not been started")
        self._autobiography.deposit_ingress(record, canonical_region=region)

    @staticmethod
    def _is_budget_defer(reason: str) -> bool:
        lowered = reason.lower()
        return (
            "budget" in lowered
            or "items_per_beat" in lowered
            or "work allocation" in lowered
        )

    def _require_started(self) -> None:
        if not self._started or not self._lease.is_held_by_us():
            raise HostStateError("Heart host does not hold the single-writer lease")


__all__ = [
    "HeartHost",
    "HeartHostBeatResult",
    "HeartHostConfig",
    "HostBeatState",
]

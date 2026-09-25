"""Runtime circulation across English proposal, refinement, and consolidation barriers.

Cores receive frozen derived rail views plus their private Souls.  FIRST and
REFINED return mandatory nonempty English proposals; the rotating consolidator
returns one compact tagged-region FINAL verdict.  Heart alone materializes that
verdict into an internal typed FieldDelta and crosses the canonical transaction
boundary.  No learned DELTA/NO_OP/ABSTAIN permission decision exists here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from runtime.field import (
    D16ResyncRequired,
    D16View,
    D16ViewDelta,
    FieldDelta,
    SharedFieldSnapshot,
    canonical_sha256,
    materialize_d16_view,
)
from runtime.soul import (
    SoulCommitReceipt,
    SoulSnapshot,
    SoulStore,
    SoulTransition,
)

from .board import ParticipantRecord, ProposalBoard, ProposalPass
from .coordinator import BeatCoordinator
from .core_bus import (
    CanonicalSyncEvent,
    D16RuntimeBinding,
    D16TextFrame,
    FieldDeltaEvent,
    FieldSnapshotEvent,
    MirrorAck,
)
from .english_reasoning import EnglishProposal, TechnicalFinalVerdict
from .errors import ReasoningCirculationError
from .mirror_coherence import MirrorCoherenceError, MirrorCoherenceRegistry, MirrorSyncState
from .proposal_workspace import (
    D64ProposalWorkspaceRenderer,
    ExactProposalWorkspaceRenderer,
    ProposalWorkspace,
    RenderedProposalRail,
)
from .reasoning_recovery import (
    REASONING_RECOVERY_PREPARATION_SCHEMA,
    ReasoningAutobiographyRecoveryStore,
    attention_view_from_d16_view,
    attention_view_from_surface,
)
from .registry import CoreDescriptor, CoreRegistry
from .tick import FrozenTickImage, RailBinding
from .transaction import HeartCommit
from .turns import TurnFinalizationReceipt, materialize_completed_turn

REASONING_REQUEST_SCHEMA = "axon-reasoning-pass-request-v3"
REASONING_PASS_RESULT_SCHEMA = "axon-reasoning-pass-result-v2"
REASONING_SOUL_LINEAGE_SCHEMA = "axon-reasoning-soul-lineage-v1"
REASONING_CIRCULATION_SCHEMA = "axon-reasoning-circulation-v4"


@dataclass(frozen=True, slots=True)
class RailRuntimeView:
    """One physical rail surface bound to its immutable tick reference."""

    binding: RailBinding
    exact_surface: Any
    semantic_surface: Any

    @property
    def d_model(self) -> int:
        return self.binding.d_model


@dataclass(frozen=True, slots=True)
class ReasoningPassRequest:
    """One phase request bound to either the D16 Core Bus or a legacy rail."""

    descriptor: CoreDescriptor
    image: FrozenTickImage
    phase: str
    soul: SoulSnapshot
    rail: RailRuntimeView | None = None
    d16_binding: D16RuntimeBinding | None = None
    proposal_rails: tuple[RenderedProposalRail, ...] = ()
    proposal_frames: tuple[D16TextFrame, ...] = ()
    request_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, CoreDescriptor):
            raise TypeError("descriptor must be CoreDescriptor")
        if not isinstance(self.image, FrozenTickImage):
            raise TypeError("image must be FrozenTickImage")
        if self.phase not in {
            ProposalPass.FIRST.value,
            ProposalPass.REFINED.value,
            "consolidated",
        }:
            raise ValueError(f"unknown reasoning phase {self.phase!r}")
        if not isinstance(self.soul, SoulSnapshot):
            raise TypeError("soul must be SoulSnapshot")
        if (
            self.soul.core_id != self.descriptor.core_id
            or self.soul.architecture_id != self.descriptor.architecture_id
            or self.soul.parameter_generation != self.descriptor.parameter_generation
        ):
            raise ReasoningCirculationError("private soul does not belong to the invoked core generation")

        rail = self.rail
        d16_binding = self.d16_binding
        if (rail is None) == (d16_binding is None):
            raise ReasoningCirculationError("reasoning request requires exactly one transport binding")

        proposal_rails = tuple(self.proposal_rails)
        proposal_frames = tuple(self.proposal_frames)
        if rail is not None:
            if not isinstance(rail, RailRuntimeView):
                raise TypeError("rail must be RailRuntimeView or None")
            if rail.d_model != self.descriptor.d_model:
                raise ReasoningCirculationError("core descriptor and runtime rail widths differ")
            if rail.binding != self.image.require_rail(self.descriptor.d_model):
                raise ReasoningCirculationError("runtime rail is not bound to the frozen image")
            if proposal_frames:
                raise ReasoningCirculationError("legacy rail request cannot also carry D16 proposal frames")
            if any(item.d_model != self.descriptor.d_model for item in proposal_rails):
                raise ReasoningCirculationError("proposal workspace rail width differs from the home rail")
        else:
            if not isinstance(d16_binding, D16RuntimeBinding):
                raise TypeError("d16_binding must be D16RuntimeBinding or None")
            if d16_binding.core_id != self.descriptor.core_id:
                raise ReasoningCirculationError("D16 binding belongs to another Core")
            if (
                d16_binding.identity.field_id != self.image.identity.base_field_id
                or d16_binding.identity.tick_id != self.image.identity.base_tick_id
            ):
                raise ReasoningCirculationError("D16 binding is stale for the frozen tick")
            if proposal_rails:
                raise ReasoningCirculationError("D16 request cannot also carry width-specific proposal rails")
            if any(not isinstance(item, D16TextFrame) for item in proposal_frames):
                raise TypeError("proposal_frames must contain D16TextFrame values")

        object.__setattr__(self, "proposal_rails", proposal_rails)
        object.__setattr__(self, "proposal_frames", proposal_frames)
        object.__setattr__(self, "request_id", canonical_sha256(self.to_canonical_dict()))

    @property
    def transport_kind(self) -> str:
        return "d16_core_bus" if self.d16_binding is not None else "legacy_rail"

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": REASONING_REQUEST_SCHEMA,
            "core_id": self.descriptor.core_id,
            "d_model": self.descriptor.d_model,
            "phase": self.phase,
            "image_id": self.image.image_id,
            "tick_uid": self.image.identity.tick_uid,
            "transport_kind": self.transport_kind,
            "rail_id": None if self.rail is None else self.rail.binding.rail_id,
            "d16_binding_id": None if self.d16_binding is None else self.d16_binding.binding_id,
            "soul_id": self.soul.soul_id,
            "soul_generation": self.soul.generation,
            "proposal_rail_ids": [item.rail_id for item in self.proposal_rails],
            "proposal_frame_ids": [item.frame_id for item in self.proposal_frames],
        }


@dataclass(frozen=True, slots=True)
class ReasoningPassResult:
    """One public English output plus the core's opaque private-Soul exhale."""

    output: EnglishProposal | TechnicalFinalVerdict
    soul_transition: SoulTransition
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.output, (EnglishProposal, TechnicalFinalVerdict)):
            raise TypeError("output must be EnglishProposal or TechnicalFinalVerdict")
        if not isinstance(self.soul_transition, SoulTransition):
            raise TypeError("soul_transition must be SoulTransition")
        object.__setattr__(self, "result_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": REASONING_PASS_RESULT_SCHEMA,
            "output": self.output.to_canonical_dict(),
            "soul_transition": self.soul_transition.to_canonical_dict(),
        }
        if include_id:
            value["result_id"] = self.result_id
        return value


@runtime_checkable
class ReasoningCorePort(Protocol):
    """Permanent execution seam for local, threaded, process, or remote cores."""

    core_id: str

    def emit(self, request: ReasoningPassRequest) -> ReasoningPassResult:
        """Return one public English contribution and private-Soul successor."""


@runtime_checkable
class D16ReasoningCorePort(ReasoningCorePort, Protocol):
    """Resident Core port that maintains an exact local D16 field mirror."""

    core_generation: int

    def apply_field_event(
        self,
        event: FieldSnapshotEvent | FieldDeltaEvent | CanonicalSyncEvent,
    ) -> MirrorAck:
        """Apply one Heart-issued exact mirror event and return its matching ACK."""


@dataclass(frozen=True, slots=True)
class SoulCoreLineage:
    core_id: str
    initial_soul_id: str
    final_soul_id: str
    transition_receipt_ids: tuple[str, ...]
    lineage_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("initial_soul_id", "final_soul_id"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be a content identity")
        receipts = tuple(self.transition_receipt_ids)
        if any(len(item) != 64 for item in receipts):
            raise ValueError("transition receipt ids must be content identities")
        object.__setattr__(self, "transition_receipt_ids", receipts)
        object.__setattr__(self, "lineage_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": REASONING_SOUL_LINEAGE_SCHEMA,
            "core_id": self.core_id,
            "initial_soul_id": self.initial_soul_id,
            "final_soul_id": self.final_soul_id,
            "transition_receipt_ids": list(self.transition_receipt_ids),
        }
        if include_id:
            value["lineage_id"] = self.lineage_id
        return value


@dataclass(frozen=True, slots=True)
class ReasoningCirculationResult:
    """Complete auditable receipt for one successful reasoning tick."""

    image: FrozenTickImage
    first_records: tuple[ParticipantRecord, ...]
    refined_records: tuple[ParticipantRecord, ...]
    first_workspace: ProposalWorkspace
    refined_workspace: ProposalWorkspace
    first_proposals: tuple[EnglishProposal, ...]
    refined_proposals: tuple[EnglishProposal, ...]
    soul_transition_receipts: tuple[SoulCommitReceipt, ...]
    soul_lineages: tuple[SoulCoreLineage, ...]
    consolidator_core_id: str
    consolidator_verdict: TechnicalFinalVerdict
    attention_view: Mapping[str, Any]
    source_delta: FieldDelta
    materialized_delta: FieldDelta
    finalization_receipt: TurnFinalizationReceipt | None
    commit: HeartCommit
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention_view", dict(self.attention_view))
        object.__setattr__(self, "result_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": REASONING_CIRCULATION_SCHEMA,
            "image": self.image.to_canonical_dict(),
            "first_records": [_record_dict(item) for item in self.first_records],
            "refined_records": [_record_dict(item) for item in self.refined_records],
            "first_workspace": self.first_workspace.to_canonical_dict(),
            "refined_workspace": self.refined_workspace.to_canonical_dict(),
            "first_proposals": [item.to_canonical_dict() for item in self.first_proposals],
            "refined_proposals": [item.to_canonical_dict() for item in self.refined_proposals],
            "soul_transition_receipts": [
                item.to_canonical_dict() for item in self.soul_transition_receipts
            ],
            "soul_lineages": [item.to_canonical_dict() for item in self.soul_lineages],
            "consolidator_core_id": self.consolidator_core_id,
            "consolidator_verdict": self.consolidator_verdict.to_canonical_dict(),
            "attention_view": dict(self.attention_view),
            "source_delta": self.source_delta.to_canonical_dict(),
            "materialized_delta": self.materialized_delta.to_canonical_dict(),
            "finalization_receipt": (
                None if self.finalization_receipt is None else self.finalization_receipt.to_canonical_dict()
            ),
            "commit": self.commit.to_canonical_dict(),
        }
        if include_id:
            value["result_id"] = self.result_id
        return value


def _record_dict(record: ParticipantRecord) -> dict[str, Any]:
    proposal = record.proposal
    return {
        "core_id": record.core_id,
        "d_model": record.d_model,
        "state": record.state.value,
        "detail": record.detail,
        "proposal_id": None if proposal is None else proposal.proposal_id,
        "proposal_text": None if proposal is None else proposal.text,
    }


class ReasoningCirculation:
    """Run one frozen tick through both barriers and the final transaction."""

    def __init__(
        self,
        coordinator: BeatCoordinator,
        registry: CoreRegistry,
        ports: Iterable[ReasoningCorePort],
        *,
        soul_store: SoulStore,
        renderer: ExactProposalWorkspaceRenderer | None = None,
        recovery_store: ReasoningAutobiographyRecoveryStore | None = None,
    ) -> None:
        if not isinstance(coordinator, BeatCoordinator):
            raise TypeError("ReasoningCirculation requires BeatCoordinator")
        if not isinstance(registry, CoreRegistry):
            raise TypeError("ReasoningCirculation requires CoreRegistry")
        if not isinstance(soul_store, SoulStore):
            raise TypeError("ReasoningCirculation requires SoulStore")
        port_map: dict[str, ReasoningCorePort] = {}
        for port in ports:
            core_id = getattr(port, "core_id", None)
            if not isinstance(core_id, str) or not core_id:
                raise TypeError("reasoning ports require a non-empty core_id")
            if core_id in port_map:
                raise ValueError(f"duplicate reasoning port {core_id!r}")
            if not callable(getattr(port, "emit", None)):
                raise TypeError(f"reasoning port {core_id!r} has no callable emit")
            port_map[core_id] = port
        self._coordinator = coordinator
        self._registry = registry
        self._ports = port_map
        self._soul_store = soul_store
        self._renderer = renderer or ExactProposalWorkspaceRenderer()
        self._recovery_store = recovery_store
        self._coherence = MirrorCoherenceRegistry()
        self._known_d16_views: dict[str, D16View] = {}

    def _external_soul_commit_bindings(self) -> dict[str, str]:
        bindings: dict[str, str] = {}
        path = self._coordinator.branch.journal_path
        if not path.exists():
            return bindings
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
                reasoning = dict(dict(event.get("metadata", {})).get("reasoning_circulation", {}))
                transition_id = reasoning.get("soul_transition_id")
                field_id = event.get("field_id")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(transition_id, str) and isinstance(field_id, str):
                bindings[transition_id] = f"canonical-field:{field_id}"
        return bindings

    def _is_d16_port(self, core_id: str) -> bool:
        port = self._ports[core_id]
        return callable(getattr(port, "apply_field_event", None))

    def _core_generation(self, core_id: str) -> int:
        value = getattr(self._ports[core_id], "core_generation", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReasoningCirculationError(f"D16 Core {core_id!r} has invalid core_generation")
        return value

    def _remember_d16_view(self, view: D16View) -> None:
        self._known_d16_views[view.view_id] = view

    def _synchronize_d16_core(self, descriptor: CoreDescriptor, target: D16View) -> D16RuntimeBinding:
        core_id = descriptor.core_id
        port = self._ports[core_id]
        generation = self._core_generation(core_id)
        record = self._coherence.register(core_id, generation)

        if self._coherence.barrier_eligible(core_id, target.identity):
            if record.last_event_sequence is None or record.last_event_id is None:
                raise ReasoningCirculationError("synchronized D16 Core has no acknowledged event identity")
            return D16RuntimeBinding(
                core_id=core_id,
                core_generation=generation,
                identity=target.identity,
                last_event_sequence=record.last_event_sequence,
                last_event_id=record.last_event_id,
            )

        sequence = 0 if record.last_event_sequence is None else record.last_event_sequence + 1
        event: FieldSnapshotEvent | FieldDeltaEvent
        prior = None
        if record.acknowledged_identity is not None:
            prior = self._known_d16_views.get(record.acknowledged_identity.view_id)
        if prior is not None:
            try:
                event = FieldDeltaEvent(sequence=sequence, delta=D16ViewDelta.between(prior, target))
            except D16ResyncRequired:
                event = FieldSnapshotEvent(sequence=sequence, view=target)
        else:
            event = FieldSnapshotEvent(sequence=sequence, view=target)

        try:
            self._coherence.expect(core_id, event)
            ack = port.apply_field_event(event)
            if not isinstance(ack, MirrorAck):
                raise TypeError("D16 Core apply_field_event must return MirrorAck")
            record = self._coherence.accept_ack(ack)
            self._coherence.require_barrier_eligible(core_id, target.identity)
        except Exception as exc:
            self._coherence.require_resync(core_id, f"D16 synchronization failed: {type(exc).__name__}: {exc}")
            raise

        self._remember_d16_view(target)
        if record.last_event_sequence is None or record.last_event_id is None:
            raise ReasoningCirculationError("D16 synchronization produced no acknowledged event identity")
        return D16RuntimeBinding(
            core_id=core_id,
            core_generation=generation,
            identity=target.identity,
            last_event_sequence=record.last_event_sequence,
            last_event_id=record.last_event_id,
        )

    def _participants(self, image: FrozenTickImage) -> tuple[CoreDescriptor, ...]:
        participants = self._registry.active()
        if not participants:
            raise ReasoningCirculationError("a reasoning tick requires active participants")
        missing_ports = sorted(item.core_id for item in participants if item.core_id not in self._ports)
        if missing_ports:
            raise ReasoningCirculationError(
                "active cores lack runtime ports: " + ", ".join(missing_ports)
            )
        for descriptor in participants:
            if not self._is_d16_port(descriptor.core_id):
                image.require_rail(descriptor.d_model)
            self._soul_store.ensure_core(
                core_id=descriptor.core_id,
                architecture_id=descriptor.architecture_id,
                parameter_generation=descriptor.parameter_generation,
            )
            self._soul_store.branch(descriptor.core_id).recover(
                external_commit_bindings=self._external_soul_commit_bindings()
            )
        return participants

    def _rail_view(self, descriptor: CoreDescriptor, image: FrozenTickImage) -> RailRuntimeView:
        surface = self._coordinator.rail_surface(descriptor.d_model)
        return RailRuntimeView(
            binding=image.require_rail(descriptor.d_model),
            exact_surface=surface.exact,
            semantic_surface=surface.semantic,
        )

    @staticmethod
    def _assert_output_binding(
        output: EnglishProposal | TechnicalFinalVerdict,
        descriptor: CoreDescriptor,
        image: FrozenTickImage,
        phase: str,
    ) -> None:
        identity = image.identity
        if output.author_core_id != descriptor.core_id:
            raise ReasoningCirculationError("reasoning output author differs from the invoked core")
        if output.rail_d_model != descriptor.d_model:
            raise ReasoningCirculationError("reasoning output names the wrong home rail")
        if output.base_field_id != identity.base_field_id or output.base_tick_id != identity.base_tick_id:
            raise ReasoningCirculationError("reasoning output is stale for the frozen tick")
        if phase in {ProposalPass.FIRST.value, ProposalPass.REFINED.value}:
            if not isinstance(output, EnglishProposal):
                raise ReasoningCirculationError("FIRST/REFINED must return an EnglishProposal")
            if output.pass_id != phase:
                raise ReasoningCirculationError("English proposal names the wrong pass")
        elif not isinstance(output, TechnicalFinalVerdict):
            raise ReasoningCirculationError("CONSOLIDATED must return a TechnicalFinalVerdict")

    @staticmethod
    def _assert_soul_transition_binding(
        transition: SoulTransition,
        request: ReasoningPassRequest,
    ) -> None:
        if (
            transition.core_id != request.descriptor.core_id
            or transition.architecture_id != request.descriptor.architecture_id
            or transition.parameter_generation != request.descriptor.parameter_generation
        ):
            raise ReasoningCirculationError("soul transition belongs to another core generation")
        if transition.before_soul_id != request.soul.soul_id:
            raise ReasoningCirculationError("soul transition is stale for the inhale snapshot")
        if transition.before_generation != request.soul.generation:
            raise ReasoningCirculationError("soul transition generation is stale")
        if transition.tick_uid != request.image.identity.tick_uid:
            raise ReasoningCirculationError("soul transition names the wrong tick")
        if transition.request_id != request.request_id or transition.phase != request.phase:
            raise ReasoningCirculationError("soul transition names the wrong request or phase")

    def _run_pass(
        self,
        base: SharedFieldSnapshot,
        image: FrozenTickImage,
        board: ProposalBoard,
        participants: tuple[CoreDescriptor, ...],
        pass_kind: ProposalPass,
        visible_workspace: ProposalWorkspace | None,
        d16_bindings: Mapping[str, D16RuntimeBinding],
        d16_sync_errors: Mapping[str, str],
    ) -> tuple[tuple[EnglishProposal, ...], tuple[SoulCommitReceipt, ...]]:
        proposals: list[EnglishProposal] = []
        receipts: list[SoulCommitReceipt] = []
        phase = pass_kind.value
        for descriptor in participants:
            soul_branch = self._soul_store.branch(descriptor.core_id)
            try:
                if self._is_d16_port(descriptor.core_id):
                    binding = d16_bindings.get(descriptor.core_id)
                    if binding is None:
                        raise MirrorCoherenceError(
                            d16_sync_errors.get(descriptor.core_id, "D16 mirror is not synchronized")
                        )
                    proposal_frames = (
                        ()
                        if visible_workspace is None
                        else (visible_workspace.require_d16_frame(),)
                    )
                    request = ReasoningPassRequest(
                        descriptor=descriptor,
                        image=image,
                        phase=phase,
                        soul=soul_branch.load_head(),
                        d16_binding=binding,
                        proposal_frames=proposal_frames,
                    )
                else:
                    proposal_rails = (
                        ()
                        if visible_workspace is None
                        else (visible_workspace.require_rail(descriptor.d_model),)
                    )
                    request = ReasoningPassRequest(
                        descriptor=descriptor,
                        image=image,
                        phase=phase,
                        soul=soul_branch.load_head(),
                        rail=self._rail_view(descriptor, image),
                        proposal_rails=proposal_rails,
                    )

                pass_result = self._ports[descriptor.core_id].emit(request)
                if not isinstance(pass_result, ReasoningPassResult):
                    raise TypeError("core returned a value other than ReasoningPassResult")
                output = pass_result.output
                self._assert_output_binding(output, descriptor, image, phase)
                if not isinstance(output, EnglishProposal):
                    raise ReasoningCirculationError("FIRST/REFINED output is not an EnglishProposal")
                self._assert_soul_transition_binding(pass_result.soul_transition, request)
                receipt = soul_branch.commit_transition(pass_result.soul_transition)
                board.submit(output)
                proposals.append(output)
                receipts.append(receipt)
            except TimeoutError as exc:
                board.mark_timed_out(descriptor.core_id, pass_kind, f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                board.mark_failed(descriptor.core_id, pass_kind, f"{type(exc).__name__}: {exc}")
        return tuple(proposals), tuple(receipts)

    def _consolidator(
        self,
        participants: tuple[CoreDescriptor, ...],
        image: FrozenTickImage,
    ) -> CoreDescriptor:
        index = (image.identity.tick_sequence - 1) % len(participants)
        return participants[index]

    def run(self, *, occurred_at: str = "") -> ReasoningCirculationResult:
        image = self._coordinator.open_tick_image
        if image is None:
            raise ReasoningCirculationError("reasoning circulation requires an in-flight tick")
        base = self._coordinator.current_field
        if base.field_id != image.identity.base_field_id or base.tick_id != image.identity.base_tick_id:
            raise ReasoningCirculationError("coordinator field differs from the frozen tick base")

        participants = self._participants(image)
        region_masks = self._coordinator.region_masks()
        base_d16 = materialize_d16_view(base, region_masks=region_masks)
        self._remember_d16_view(base_d16)

        d16_bindings: dict[str, D16RuntimeBinding] = {}
        d16_sync_errors: dict[str, str] = {}
        d16_core_ids = tuple(
            descriptor.core_id
            for descriptor in participants
            if self._is_d16_port(descriptor.core_id)
        )
        for descriptor in participants:
            if descriptor.core_id not in d16_core_ids:
                continue
            try:
                d16_bindings[descriptor.core_id] = self._synchronize_d16_core(descriptor, base_d16)
            except Exception as exc:
                d16_sync_errors[descriptor.core_id] = f"{type(exc).__name__}: {exc}"

        initial_souls = {
            descriptor.core_id: self._soul_store.branch(descriptor.core_id).load_head().soul_id
            for descriptor in participants
        }
        board = ProposalBoard(image, participants, d16_core_ids=d16_core_ids)

        first_proposals, first_receipts = self._run_pass(
            base, image, board, participants, ProposalPass.FIRST, None,
            d16_bindings, d16_sync_errors,
        )
        board.close_first_pass()
        first_records = board.participant_states(ProposalPass.FIRST)
        first_workspace = self._renderer.render(image, ProposalPass.FIRST, first_records)

        refined_proposals, refined_receipts = self._run_pass(
            base, image, board, participants, ProposalPass.REFINED, first_workspace,
            d16_bindings, d16_sync_errors,
        )
        board.close_refinement()
        board.assert_ready_for_consolidation()
        refined_records = board.participant_states(ProposalPass.REFINED)
        refined_workspace = self._renderer.render(image, ProposalPass.REFINED, refined_records)

        returned_ids = {
            record.core_id for record in refined_records if record.state.value == "returned"
        }
        eligible = tuple(descriptor for descriptor in participants if descriptor.core_id in returned_ids)
        if not eligible:
            raise ReasoningCirculationError("no successfully refined participant is eligible to consolidate")
        consolidator = self._consolidator(eligible, image)
        soul_branch = self._soul_store.branch(consolidator.core_id)
        if self._is_d16_port(consolidator.core_id):
            binding = d16_bindings.get(consolidator.core_id)
            if binding is None:
                raise ReasoningCirculationError("selected D16 consolidator is not mirror-coherent")
            request = ReasoningPassRequest(
                descriptor=consolidator,
                image=image,
                phase="consolidated",
                soul=soul_branch.load_head(),
                d16_binding=binding,
                proposal_frames=(
                    first_workspace.require_d16_frame(),
                    refined_workspace.require_d16_frame(),
                ),
            )
        else:
            request = ReasoningPassRequest(
                descriptor=consolidator,
                image=image,
                phase="consolidated",
                soul=soul_branch.load_head(),
                rail=self._rail_view(consolidator, image),
                proposal_rails=(
                    first_workspace.require_rail(consolidator.d_model),
                    refined_workspace.require_rail(consolidator.d_model),
                ),
            )

        has_d16_participant = bool(d16_core_ids)
        attention_view = (
            attention_view_from_d16_view(
                base_d16,
                heart_view_id=image.view_id,
                region_masks=region_masks,
            )
            if has_d16_participant
            else attention_view_from_surface(
                self._coordinator.rail_surface(64).exact,
                view_id=image.view_id,
            )
        )

        try:
            pass_result = self._ports[consolidator.core_id].emit(request)
            if not isinstance(pass_result, ReasoningPassResult):
                raise TypeError("consolidator returned a value other than ReasoningPassResult")
            verdict = pass_result.output
            self._assert_output_binding(verdict, consolidator, image, "consolidated")
            if not isinstance(verdict, TechnicalFinalVerdict):
                raise ReasoningCirculationError("consolidator output is not a tagged FINAL verdict")
            self._assert_soul_transition_binding(pass_result.soul_transition, request)
            source_delta = verdict.materialize(base)
            materialized_delta, finalization = materialize_completed_turn(base, source_delta)
            soul_branch.prepare_transition(
                pass_result.soul_transition,
                requires_external_commit=True,
            )
            circulation_metadata = {
                "reasoning_image_id": image.image_id,
                "first_workspace_id": first_workspace.workspace_id,
                "refined_workspace_id": refined_workspace.workspace_id,
                "consolidator_verdict_id": verdict.verdict_id,
                "source_delta_id": source_delta.delta_id,
                "soul_transition_id": pass_result.soul_transition.transition_id,
                "transport": "d16_core_bus" if has_d16_participant else "legacy_rail",
                "turn_finalization_receipt_id": None if finalization is None else finalization.receipt_id,
            }
            recovery_preparation_id = None
            if self._recovery_store is not None:
                preparation = self._recovery_store.prepare(
                    {
                        "schema": REASONING_RECOVERY_PREPARATION_SCHEMA,
                        "occurred_at": str(occurred_at),
                        "pre_action_field": base.to_dict(),
                        "attention_view": attention_view,
                        "image": image.to_canonical_dict(),
                        "first_records": [_record_dict(item) for item in first_records],
                        "refined_records": [_record_dict(item) for item in refined_records],
                        "first_workspace": first_workspace.to_canonical_dict(),
                        "refined_workspace": refined_workspace.to_canonical_dict(),
                        "first_proposals": [item.to_canonical_dict() for item in first_proposals],
                        "refined_proposals": [item.to_canonical_dict() for item in refined_proposals],
                        "prior_soul_receipts": [item.to_canonical_dict() for item in (*first_receipts, *refined_receipts)],
                        "initial_souls": dict(sorted(initial_souls.items())),
                        "consolidator_core_id": consolidator.core_id,
                        "consolidator_verdict": verdict.to_canonical_dict(),
                        "consolidator_soul_transition": pass_result.soul_transition.to_canonical_dict(),
                        "source_delta": source_delta.to_canonical_dict(),
                        "materialized_delta": materialized_delta.to_canonical_dict(),
                        "materialized_delta_id": materialized_delta.delta_id,
                        "finalization_receipt": None if finalization is None else finalization.to_canonical_dict(),
                        "circulation_metadata": circulation_metadata,
                    }
                )
                recovery_preparation_id = preparation.preparation_id
            commit = self._coordinator.commit_consolidator_delta(
                materialized_delta,
                tick=image.identity,
                metadata={**circulation_metadata, "recovery_preparation_id": recovery_preparation_id},
            )
            consolidator_soul_receipt = soul_branch.finalize_transition(
                pass_result.soul_transition.transition_id,
                commit_binding=f"canonical-field:{commit.successor.field_id}",
            )
        except Exception as exc:
            raise ReasoningCirculationError(
                f"consolidator {consolidator.core_id!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

        # CANONICAL_SYNC updates exact mirrors only; it never calls emit().
        if d16_core_ids:
            successor_d16 = materialize_d16_view(commit.successor, region_masks=region_masks)
            for descriptor in participants:
                core_id = descriptor.core_id
                if core_id not in d16_bindings:
                    continue
                record = self._coherence.record(core_id)
                if record.last_event_sequence is None:
                    self._coherence.require_resync(core_id, "CANONICAL_SYNC lacks prior event sequence")
                    continue
                try:
                    sync_delta = D16ViewDelta.between(base_d16, successor_d16)
                    sync_event = CanonicalSyncEvent(
                        sequence=record.last_event_sequence + 1,
                        delta=sync_delta,
                    )
                    self._coherence.expect(core_id, sync_event)
                    ack = self._ports[core_id].apply_field_event(sync_event)
                    if not isinstance(ack, MirrorAck):
                        raise TypeError("D16 Core CANONICAL_SYNC must return MirrorAck")
                    self._coherence.accept_ack(ack)
                except Exception as exc:
                    self._coherence.require_resync(
                        core_id,
                        f"CANONICAL_SYNC failed: {type(exc).__name__}: {exc}",
                    )
            self._remember_d16_view(successor_d16)

        soul_receipts = (*first_receipts, *refined_receipts, consolidator_soul_receipt)
        lineages = tuple(
            SoulCoreLineage(
                core_id=descriptor.core_id,
                initial_soul_id=initial_souls[descriptor.core_id],
                final_soul_id=self._soul_store.branch(descriptor.core_id).load_head().soul_id,
                transition_receipt_ids=tuple(
                    receipt.receipt_id for receipt in soul_receipts if receipt.core_id == descriptor.core_id
                ),
            )
            for descriptor in participants
        )
        return ReasoningCirculationResult(
            image=image,
            first_records=first_records,
            refined_records=refined_records,
            first_workspace=first_workspace,
            refined_workspace=refined_workspace,
            first_proposals=first_proposals,
            refined_proposals=refined_proposals,
            soul_transition_receipts=tuple(soul_receipts),
            soul_lineages=lineages,
            consolidator_core_id=consolidator.core_id,
            consolidator_verdict=verdict,
            attention_view=attention_view,
            source_delta=source_delta,
            materialized_delta=materialized_delta,
            finalization_receipt=finalization,
            commit=commit,
        )


__all__ = [
    "REASONING_CIRCULATION_SCHEMA",
    "REASONING_PASS_RESULT_SCHEMA",
    "REASONING_REQUEST_SCHEMA",
    "REASONING_SOUL_LINEAGE_SCHEMA",
    "D16ReasoningCorePort",
    "RailRuntimeView",
    "ReasoningCirculation",
    "ReasoningCirculationResult",
    "ReasoningCorePort",
    "ReasoningPassRequest",
    "ReasoningPassResult",
    "SoulCoreLineage",
]

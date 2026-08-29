"""Runtime circulation across proposal, refinement, and consolidation barriers.

This module is the executable boundary between learned reasoning tissue and the
Heart.  Cores receive only frozen, derived rail views.  They return categorical
emissions; Heart decodes and validates those emissions into typed deltas.  The
proposal workspaces remain derived and noncanonical.  Only the final,
Heart-materialized consolidator delta crosses the transaction boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

from runtime.field import FieldDelta, SharedFieldSnapshot, canonical_sha256
from runtime.soul import (
    SoulCommitReceipt,
    SoulSnapshot,
    SoulStore,
    SoulTransition,
)

from .authority import AuthorityGrant
from .board import ParticipantRecord, Proposal, ProposalBoard, ProposalPass
from .coordinator import BeatCoordinator
from .errors import ReasoningCirculationError
from .proposal_workspace import (
    D64ProposalWorkspaceRenderer,
    ExactProposalWorkspaceRenderer,
    ProposalWorkspace,
    RenderedProposalRail,
)
from .reasoning_output import ReasoningDecision, ReasoningEmission
from .registry import CoreDescriptor, CoreRegistry
from .tick import FrozenTickImage, RailBinding
from .transaction import HeartCommit
from .turns import TurnFinalizationReceipt, materialize_completed_turn

REASONING_REQUEST_SCHEMA = "axon-reasoning-pass-request-v2"
REASONING_PASS_RESULT_SCHEMA = "axon-reasoning-pass-result-v1"
REASONING_SOUL_LINEAGE_SCHEMA = "axon-reasoning-soul-lineage-v1"
REASONING_CIRCULATION_SCHEMA = "axon-reasoning-circulation-v2"


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
    """The complete, derived information one core may inspect for one phase."""

    descriptor: CoreDescriptor
    image: FrozenTickImage
    phase: str
    rail: RailRuntimeView
    soul: SoulSnapshot
    proposal_rails: tuple[RenderedProposalRail, ...] = ()
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
        if not isinstance(self.rail, RailRuntimeView):
            raise TypeError("rail must be RailRuntimeView")
        if not isinstance(self.soul, SoulSnapshot):
            raise TypeError("soul must be SoulSnapshot")
        if (
            self.soul.core_id != self.descriptor.core_id
            or self.soul.architecture_id != self.descriptor.architecture_id
            or self.soul.parameter_generation != self.descriptor.parameter_generation
        ):
            raise ReasoningCirculationError("private soul does not belong to the invoked core generation")
        if self.rail.d_model != self.descriptor.d_model:
            raise ReasoningCirculationError("core descriptor and runtime rail widths differ")
        if self.rail.binding != self.image.require_rail(self.descriptor.d_model):
            raise ReasoningCirculationError("runtime rail is not bound to the frozen image")
        proposal_rails = tuple(self.proposal_rails)
        if any(item.d_model != self.descriptor.d_model for item in proposal_rails):
            raise ReasoningCirculationError("proposal workspace rail width differs from the home rail")
        object.__setattr__(self, "proposal_rails", proposal_rails)
        object.__setattr__(self, "request_id", canonical_sha256(self.to_canonical_dict()))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": REASONING_REQUEST_SCHEMA,
            "core_id": self.descriptor.core_id,
            "d_model": self.descriptor.d_model,
            "phase": self.phase,
            "image_id": self.image.image_id,
            "tick_uid": self.image.identity.tick_uid,
            "rail_id": self.rail.binding.rail_id,
            "soul_id": self.soul.soul_id,
            "soul_generation": self.soul.generation,
            "proposal_rail_ids": [item.rail_id for item in self.proposal_rails],
        }


@dataclass(frozen=True, slots=True)
class ReasoningPassResult:
    """One exact field decision plus the core's opaque private-soul exhale."""

    emission: ReasoningEmission
    soul_transition: SoulTransition
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.emission, ReasoningEmission):
            raise TypeError("emission must be ReasoningEmission")
        if not isinstance(self.soul_transition, SoulTransition):
            raise TypeError("soul_transition must be SoulTransition")
        object.__setattr__(self, "result_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": REASONING_PASS_RESULT_SCHEMA,
            "emission": self.emission.to_canonical_dict(),
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
        """Return one exact field decision and private-soul successor proposal."""


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
    first_emissions: tuple[ReasoningEmission, ...]
    refined_emissions: tuple[ReasoningEmission, ...]
    soul_transition_receipts: tuple[SoulCommitReceipt, ...]
    soul_lineages: tuple[SoulCoreLineage, ...]
    consolidator_core_id: str
    consolidator_emission: ReasoningEmission
    source_delta: FieldDelta
    materialized_delta: FieldDelta
    finalization_receipt: TurnFinalizationReceipt | None
    commit: HeartCommit
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": REASONING_CIRCULATION_SCHEMA,
            "image": self.image.to_canonical_dict(),
            "first_records": [_record_dict(item) for item in self.first_records],
            "refined_records": [_record_dict(item) for item in self.refined_records],
            "first_workspace": self.first_workspace.to_canonical_dict(),
            "refined_workspace": self.refined_workspace.to_canonical_dict(),
            "first_emissions": [item.to_canonical_dict() for item in self.first_emissions],
            "refined_emissions": [item.to_canonical_dict() for item in self.refined_emissions],
            "soul_transition_receipts": [
                item.to_canonical_dict() for item in self.soul_transition_receipts
            ],
            "soul_lineages": [item.to_canonical_dict() for item in self.soul_lineages],
            "consolidator_core_id": self.consolidator_core_id,
            "consolidator_emission": self.consolidator_emission.to_canonical_dict(),
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
        "emission_id": None if proposal is None else proposal.emission_id,
        "delta_id": None if proposal is None else proposal.delta.delta_id,
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
        self._renderer = renderer or D64ProposalWorkspaceRenderer()

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
    def _assert_emission_binding(
        emission: ReasoningEmission,
        descriptor: CoreDescriptor,
        image: FrozenTickImage,
        phase: str,
    ) -> None:
        identity = image.identity
        if emission.author_core_id != descriptor.core_id:
            raise ReasoningCirculationError("reasoning emission author differs from the invoked core")
        if emission.rail_d_model != descriptor.d_model:
            raise ReasoningCirculationError("reasoning emission names the wrong home rail")
        if emission.pass_id != phase:
            raise ReasoningCirculationError("reasoning emission names the wrong pass")
        if emission.base_field_id != identity.base_field_id or emission.base_tick_id != identity.base_tick_id:
            raise ReasoningCirculationError("reasoning emission is stale for the frozen tick")

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
    ) -> tuple[tuple[ReasoningEmission, ...], tuple[SoulCommitReceipt, ...]]:
        emissions: list[ReasoningEmission] = []
        receipts: list[SoulCommitReceipt] = []
        phase = pass_kind.value
        for descriptor in participants:
            proposal_rails = (
                ()
                if visible_workspace is None
                else (visible_workspace.require_rail(descriptor.d_model),)
            )
            soul_branch = self._soul_store.branch(descriptor.core_id)
            request = ReasoningPassRequest(
                descriptor=descriptor,
                image=image,
                phase=phase,
                rail=self._rail_view(descriptor, image),
                soul=soul_branch.load_head(),
                proposal_rails=proposal_rails,
            )
            try:
                pass_result = self._ports[descriptor.core_id].emit(request)
                if not isinstance(pass_result, ReasoningPassResult):
                    raise TypeError("core returned a value other than ReasoningPassResult")
                emission = pass_result.emission
                self._assert_emission_binding(emission, descriptor, image, phase)
                self._assert_soul_transition_binding(pass_result.soul_transition, request)
                proposal: Proposal | None = None
                detail = "" if emission.detail is None else emission.detail.text
                if emission.decision is ReasoningDecision.DELTA:
                    delta = emission.decode_delta(base, descriptor.authority_grant())
                    if delta is None:
                        raise ReasoningCirculationError("delta decision decoded to no delta")
                    proposal = Proposal(
                        delta=delta,
                        rail_d_model=descriptor.d_model,
                        pass_kind=pass_kind,
                        emission_id=emission.emission_id,
                    )
                receipt = soul_branch.commit_transition(pass_result.soul_transition)
                if proposal is not None:
                    board.submit(proposal)
                elif emission.decision is ReasoningDecision.NO_OP:
                    board.mark_no_op(descriptor.core_id, pass_kind, detail)
                else:
                    board.mark_abstained(descriptor.core_id, pass_kind, detail)
                emissions.append(emission)
                receipts.append(receipt)
            except TimeoutError as exc:
                board.mark_timed_out(descriptor.core_id, pass_kind, f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                board.mark_failed(descriptor.core_id, pass_kind, f"{type(exc).__name__}: {exc}")
        return tuple(emissions), tuple(receipts)

    def _consolidator(
        self,
        participants: tuple[CoreDescriptor, ...],
        image: FrozenTickImage,
    ) -> CoreDescriptor:
        index = (image.identity.tick_sequence - 1) % len(participants)
        return participants[index]

    def run(self) -> ReasoningCirculationResult:
        image = self._coordinator.open_tick_image
        if image is None:
            raise ReasoningCirculationError("reasoning circulation requires an in-flight tick")
        base = self._coordinator.current_field
        if base.field_id != image.identity.base_field_id or base.tick_id != image.identity.base_tick_id:
            raise ReasoningCirculationError("coordinator field differs from the frozen tick base")
        participants = self._participants(image)
        initial_souls = {
            descriptor.core_id: self._soul_store.branch(descriptor.core_id).load_head().soul_id
            for descriptor in participants
        }
        board = ProposalBoard(image, participants)

        first_emissions, first_receipts = self._run_pass(
            base,
            image,
            board,
            participants,
            ProposalPass.FIRST,
            None,
        )
        board.close_first_pass()
        first_records = board.participant_states(ProposalPass.FIRST)
        first_workspace = self._renderer.render(image, ProposalPass.FIRST, first_records)

        refined_emissions, refined_receipts = self._run_pass(
            base,
            image,
            board,
            participants,
            ProposalPass.REFINED,
            first_workspace,
        )
        board.close_refinement()
        board.assert_ready_for_consolidation()
        refined_records = board.participant_states(ProposalPass.REFINED)
        refined_workspace = self._renderer.render(image, ProposalPass.REFINED, refined_records)

        consolidator = self._consolidator(participants, image)
        request = ReasoningPassRequest(
            descriptor=consolidator,
            image=image,
            phase="consolidated",
            rail=self._rail_view(consolidator, image),
            soul=self._soul_store.branch(consolidator.core_id).load_head(),
            proposal_rails=(
                first_workspace.require_rail(consolidator.d_model),
                refined_workspace.require_rail(consolidator.d_model),
            ),
        )
        try:
            pass_result = self._ports[consolidator.core_id].emit(request)
            if not isinstance(pass_result, ReasoningPassResult):
                raise TypeError("consolidator returned a value other than ReasoningPassResult")
            emission = pass_result.emission
            self._assert_emission_binding(emission, consolidator, image, "consolidated")
            self._assert_soul_transition_binding(pass_result.soul_transition, request)
            if emission.decision is not ReasoningDecision.DELTA:
                raise ReasoningCirculationError("consolidator must return a non-empty delta decision")
            source_delta = emission.decode_delta(base, AuthorityGrant.consolidator())
            if source_delta is None:
                raise ReasoningCirculationError("consolidator delta decoded to no delta")
            materialized_delta, finalization = materialize_completed_turn(base, source_delta)
            soul_branch = self._soul_store.branch(consolidator.core_id)
            soul_branch.prepare_transition(
                pass_result.soul_transition,
                requires_external_commit=True,
            )
            commit = self._coordinator.commit_consolidator_delta(
                materialized_delta,
                tick=image.identity,
                metadata={
                    "reasoning_image_id": image.image_id,
                    "first_workspace_id": first_workspace.workspace_id,
                    "refined_workspace_id": refined_workspace.workspace_id,
                    "consolidator_emission_id": emission.emission_id,
                    "source_delta_id": source_delta.delta_id,
                    "soul_transition_id": pass_result.soul_transition.transition_id,
                    "turn_finalization_receipt_id": (
                        None if finalization is None else finalization.receipt_id
                    ),
                },
            )
            consolidator_soul_receipt = soul_branch.finalize_transition(
                pass_result.soul_transition.transition_id,
                commit_binding=f"canonical-field:{commit.successor.field_id}",
            )
        except Exception as exc:
            raise ReasoningCirculationError(
                f"consolidator {consolidator.core_id!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

        soul_receipts = (*first_receipts, *refined_receipts, consolidator_soul_receipt)
        lineages = tuple(
            SoulCoreLineage(
                core_id=descriptor.core_id,
                initial_soul_id=initial_souls[descriptor.core_id],
                final_soul_id=self._soul_store.branch(descriptor.core_id).load_head().soul_id,
                transition_receipt_ids=tuple(
                    receipt.receipt_id
                    for receipt in soul_receipts
                    if receipt.core_id == descriptor.core_id
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
            first_emissions=first_emissions,
            refined_emissions=refined_emissions,
            soul_transition_receipts=tuple(soul_receipts),
            soul_lineages=lineages,
            consolidator_core_id=consolidator.core_id,
            consolidator_emission=emission,
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
    "RailRuntimeView",
    "ReasoningCirculation",
    "ReasoningCirculationResult",
    "ReasoningCorePort",
    "ReasoningPassRequest",
    "ReasoningPassResult",
    "SoulCoreLineage",
]

"""The per-tick proposal board: a NONCANONICAL English reasoning workspace.

The board stores exact FIRST/REFINED English proposals bound to one frozen tick.
It has no learned no-op or abstain success state.  A successful participant
returns a nonempty English proposal; failure and timeout remain runtime control
states.  The board never writes canonical state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from .english_reasoning import EnglishProposal
from .errors import (
    BarrierClosedError,
    BarrierNotReadyError,
    DuplicateProposalError,
    ProposalBoardError,
    RailMembershipError,
    StaleBaseProposalError,
    UnknownParticipantError,
)
from .registry import CoreDescriptor, CoreStatus
from .tick import FrozenTickImage


class ProposalPass(str, Enum):
    """The two deliberation passes before consolidation."""

    FIRST = "first"
    REFINED = "refined"


class ParticipantState(str, Enum):
    """Runtime accounting for one participant in one pass."""

    PENDING = "pending"
    RETURNED = "returned"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


# Compatibility name for older imports.  The active proposal type is explicitly
# EnglishProposal; there is no separate typed-delta Proposal object anymore.
Proposal = EnglishProposal


@dataclass(frozen=True, slots=True)
class ParticipantRecord:
    """Immutable accounting record for one participant in one pass."""

    core_id: str
    d_model: int
    state: ParticipantState
    detail: str = ""
    proposal: EnglishProposal | None = None

    def __post_init__(self) -> None:
        state = self.state if isinstance(self.state, ParticipantState) else ParticipantState(self.state)
        object.__setattr__(self, "state", state)
        if state is ParticipantState.RETURNED:
            if not isinstance(self.proposal, EnglishProposal):
                raise ProposalBoardError("a returned participant record must carry an EnglishProposal")
        elif self.proposal is not None:
            raise ProposalBoardError("only a returned participant record may carry a proposal")


class ProposalBoard:
    """NONCANONICAL per-tick English workspace with enforced stage barriers."""

    def __init__(
        self,
        image: FrozenTickImage,
        participants: Iterable[CoreDescriptor],
        *,
        d16_core_ids: Iterable[str] = (),
    ) -> None:
        if not isinstance(image, FrozenTickImage):
            raise TypeError("ProposalBoard requires a FrozenTickImage")
        self._image = image
        d16_ids = frozenset(str(item) for item in d16_core_ids)
        roster: dict[str, CoreDescriptor] = {}
        for descriptor in tuple(participants):
            if not isinstance(descriptor, CoreDescriptor):
                raise TypeError("participants must be CoreDescriptor values")
            if descriptor.status is not CoreStatus.ACTIVE:
                raise ProposalBoardError(f"participant {descriptor.core_id!r} is not active")
            if descriptor.core_id in roster:
                raise ProposalBoardError(f"duplicate participant {descriptor.core_id!r}")
            if descriptor.core_id not in d16_ids and image.rail_for(descriptor.d_model) is None:
                raise RailMembershipError(
                    f"legacy participant {descriptor.core_id!r} rides d_model rail "
                    f"{descriptor.d_model}, which the tick image does not carry"
                )
            roster[descriptor.core_id] = descriptor
        if not roster:
            raise ProposalBoardError("a tick requires a non-empty participant set")
        self._roster = roster
        self._first_pass = self._fresh_pass_records()
        self._refinement: dict[str, ParticipantRecord] | None = None
        self._first_pass_closed = False
        self._refinement_closed = False

    def _fresh_pass_records(self) -> dict[str, ParticipantRecord]:
        return {
            core_id: ParticipantRecord(
                core_id=core_id,
                d_model=descriptor.d_model,
                state=ParticipantState.PENDING,
            )
            for core_id, descriptor in self._roster.items()
        }

    @property
    def image(self) -> FrozenTickImage:
        return self._image

    @property
    def base_field_id(self) -> str:
        return self._image.identity.base_field_id

    @property
    def first_pass_closed(self) -> bool:
        return self._first_pass_closed

    @property
    def refinement_closed(self) -> bool:
        return self._refinement_closed

    def _stage_records(self, pass_kind: ProposalPass) -> dict[str, ParticipantRecord]:
        pass_kind = pass_kind if isinstance(pass_kind, ProposalPass) else ProposalPass(pass_kind)
        if pass_kind is ProposalPass.FIRST:
            if self._first_pass_closed:
                raise BarrierClosedError("the first pass is already closed")
            return self._first_pass
        if self._refinement is None:
            raise BarrierNotReadyError("refinement has not begun; the first-pass barrier must close first")
        if self._refinement_closed:
            raise BarrierClosedError("the refinement pass is already closed")
        return self._refinement

    def submit(self, proposal: EnglishProposal) -> ParticipantRecord:
        """Record one mandatory nonempty English proposal."""

        if not isinstance(proposal, EnglishProposal):
            raise TypeError("ProposalBoard.submit requires an EnglishProposal")
        pass_kind = ProposalPass(proposal.pass_id)
        records = self._stage_records(pass_kind)
        record = records.get(proposal.author_core_id)
        if record is None:
            raise UnknownParticipantError(f"{proposal.author_core_id!r} is not a declared tick participant")
        if record.state is not ParticipantState.PENDING:
            raise DuplicateProposalError(
                f"participant {record.core_id!r} is already accounted as "
                f"{record.state.value} in the {pass_kind.value} pass"
            )
        identity = self._image.identity
        if proposal.base_field_id != identity.base_field_id or proposal.base_tick_id != identity.base_tick_id:
            raise StaleBaseProposalError(
                f"proposal from {record.core_id!r} is not bound to the frozen "
                f"base field {identity.base_field_id!r} tick {identity.base_tick_id}"
            )
        descriptor = self._roster[record.core_id]
        if proposal.rail_d_model != descriptor.d_model:
            raise RailMembershipError(
                f"participant {record.core_id!r} proposed on d_model rail "
                f"{proposal.rail_d_model} but is registered on {descriptor.d_model}"
            )
        updated = replace(record, state=ParticipantState.RETURNED, proposal=proposal)
        records[record.core_id] = updated
        return updated

    def _mark(
        self,
        core_id: str,
        pass_kind: ProposalPass,
        state: ParticipantState,
        detail: str,
    ) -> ParticipantRecord:
        if state not in {ParticipantState.FAILED, ParticipantState.TIMED_OUT}:
            raise ProposalBoardError("only failure or timeout may complete a pass without an English proposal")
        records = self._stage_records(pass_kind)
        record = records.get(core_id)
        if record is None:
            raise UnknownParticipantError(f"{core_id!r} is not a declared tick participant")
        if record.state is not ParticipantState.PENDING:
            raise DuplicateProposalError(
                f"participant {core_id!r} is already accounted as "
                f"{record.state.value} in the {ProposalPass(pass_kind).value} pass"
            )
        updated = replace(record, state=state, detail=str(detail))
        records[core_id] = updated
        return updated

    def mark_failed(
        self,
        core_id: str,
        pass_kind: ProposalPass,
        detail: str = "",
    ) -> ParticipantRecord:
        return self._mark(core_id, pass_kind, ParticipantState.FAILED, detail)

    def mark_timed_out(
        self,
        core_id: str,
        pass_kind: ProposalPass,
        detail: str = "",
    ) -> ParticipantRecord:
        return self._mark(core_id, pass_kind, ParticipantState.TIMED_OUT, detail)

    @staticmethod
    def _pending_ids(records: dict[str, ParticipantRecord]) -> tuple[str, ...]:
        return tuple(
            sorted(
                core_id
                for core_id, record in records.items()
                if record.state is ParticipantState.PENDING
            )
        )

    def close_first_pass(self) -> None:
        if self._first_pass_closed:
            raise BarrierClosedError("the first pass is already closed")
        pending = self._pending_ids(self._first_pass)
        if pending:
            raise BarrierNotReadyError(
                "first-pass barrier cannot close; participants pending: " + ", ".join(pending)
            )
        self._first_pass_closed = True
        self._refinement = self._fresh_pass_records()

    def first_pass_proposals(self) -> tuple[EnglishProposal, ...]:
        if not self._first_pass_closed:
            raise BarrierNotReadyError("the first-pass board is exposed only after its barrier closes")
        return tuple(
            record.proposal
            for _, record in sorted(self._first_pass.items())
            if record.state is ParticipantState.RETURNED and record.proposal is not None
        )

    def close_refinement(self) -> None:
        if self._refinement is None:
            raise BarrierNotReadyError("refinement has not begun; the first-pass barrier must close first")
        if self._refinement_closed:
            raise BarrierClosedError("the refinement pass is already closed")
        pending = self._pending_ids(self._refinement)
        if pending:
            raise BarrierNotReadyError(
                "refinement barrier cannot close; participants pending: " + ", ".join(pending)
            )
        self._refinement_closed = True

    def refined_proposals(self) -> tuple[EnglishProposal, ...]:
        if not self._refinement_closed or self._refinement is None:
            raise BarrierNotReadyError("refined proposals are exposed only after the refinement barrier")
        return tuple(
            record.proposal
            for _, record in sorted(self._refinement.items())
            if record.state is ParticipantState.RETURNED and record.proposal is not None
        )

    def assert_ready_for_consolidation(self) -> None:
        if not self._refinement_closed:
            raise BarrierNotReadyError("consolidation requires the refinement barrier to be closed")

    def participant_states(self, pass_kind: ProposalPass) -> tuple[ParticipantRecord, ...]:
        pass_kind = pass_kind if isinstance(pass_kind, ProposalPass) else ProposalPass(pass_kind)
        if pass_kind is ProposalPass.FIRST:
            records = self._first_pass
        else:
            if self._refinement is None:
                raise BarrierNotReadyError("refinement has not begun; the first-pass barrier must close first")
            records = self._refinement
        return tuple(records[core_id] for core_id in sorted(records))


__all__ = [
    "ParticipantRecord",
    "ParticipantState",
    "Proposal",
    "ProposalBoard",
    "ProposalPass",
]

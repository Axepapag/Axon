"""The per-tick proposal board: a NONCANONICAL shared reasoning workspace.

The board holds sparse proposals (typed ``FieldDelta`` payloads with
author/rail/pass provenance, each bound to the frozen base field) for one
tick.  It enforces the stage barriers:

- the first pass closes only when every declared participant has returned,
  failed, or timed out under governed policy;
- refinement then exposes the complete first-pass board;
- consolidation is reachable only after the refinement barrier closes.

The board never writes canonical state.  It stores proposals and accounting
only; only the heart transaction boundary may convert a proposal into a
successor canonical field.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from runtime.field import FieldDelta

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
    """The two core deliberation passes of a tick."""

    FIRST = "first"
    REFINED = "refined"


class ParticipantState(str, Enum):
    """Accounting state of one declared participant in one pass."""

    PENDING = "pending"
    RETURNED = "returned"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class Proposal:
    """One sparse proposal on the board; never canonical state."""

    delta: FieldDelta
    rail_d_model: int
    pass_kind: ProposalPass

    def __post_init__(self) -> None:
        if not isinstance(self.delta, FieldDelta):
            raise TypeError("Proposal.delta must be a FieldDelta")
        pass_kind = (
            self.pass_kind
            if isinstance(self.pass_kind, ProposalPass)
            else ProposalPass(self.pass_kind)
        )
        object.__setattr__(self, "pass_kind", pass_kind)
        if isinstance(self.rail_d_model, bool) or not isinstance(
            self.rail_d_model, int
        ):
            raise TypeError("Proposal.rail_d_model must be an integer")
        if self.rail_d_model <= 0:
            raise ValueError("Proposal.rail_d_model must be positive")
        if str(self.delta.pass_id) != pass_kind.value:
            raise ProposalBoardError(
                f"proposal pass_id {self.delta.pass_id!r} does not match "
                f"pass kind {pass_kind.value!r}"
            )

    @property
    def author_core_id(self) -> str:
        return self.delta.author_core_id

    @property
    def base_field_id(self) -> str:
        return self.delta.base_field_id

    @property
    def base_tick_id(self) -> int:
        return self.delta.base_tick_id


@dataclass(frozen=True, slots=True)
class ParticipantRecord:
    """Immutable accounting record for one participant in one pass."""

    core_id: str
    d_model: int
    state: ParticipantState
    detail: str = ""
    proposal: Proposal | None = None

    def __post_init__(self) -> None:
        state = (
            self.state
            if isinstance(self.state, ParticipantState)
            else ParticipantState(self.state)
        )
        object.__setattr__(self, "state", state)
        if state is ParticipantState.RETURNED:
            if self.proposal is None:
                raise ProposalBoardError(
                    "a returned participant record must carry its proposal"
                )
        elif self.proposal is not None:
            raise ProposalBoardError(
                "only a returned participant record may carry a proposal"
            )


class ProposalBoard:
    """NONCANONICAL per-tick workspace with enforced stage barriers."""

    def __init__(
        self,
        image: FrozenTickImage,
        participants: Iterable[CoreDescriptor],
    ) -> None:
        if not isinstance(image, FrozenTickImage):
            raise TypeError("ProposalBoard requires a FrozenTickImage")
        self._image = image
        roster: dict[str, CoreDescriptor] = {}
        for descriptor in tuple(participants):
            if not isinstance(descriptor, CoreDescriptor):
                raise TypeError("participants must be CoreDescriptor values")
            if descriptor.status is not CoreStatus.ACTIVE:
                raise ProposalBoardError(
                    f"participant {descriptor.core_id!r} is not active"
                )
            if descriptor.core_id in roster:
                raise ProposalBoardError(
                    f"duplicate participant {descriptor.core_id!r}"
                )
            if image.rail_for(descriptor.d_model) is None:
                raise RailMembershipError(
                    f"participant {descriptor.core_id!r} rides d_model rail "
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
        if pass_kind is ProposalPass.FIRST:
            if self._first_pass_closed:
                raise BarrierClosedError("the first pass is already closed")
            return self._first_pass
        if self._refinement is None:
            raise BarrierNotReadyError(
                "refinement has not begun; the first-pass barrier must close first"
            )
        if self._refinement_closed:
            raise BarrierClosedError("the refinement pass is already closed")
        return self._refinement

    def submit(self, proposal: Proposal) -> ParticipantRecord:
        """Record one sparse proposal; returns the updated accounting record."""

        if not isinstance(proposal, Proposal):
            raise TypeError("ProposalBoard.submit requires a Proposal")
        records = self._stage_records(proposal.pass_kind)
        record = records.get(proposal.author_core_id)
        if record is None:
            raise UnknownParticipantError(
                f"{proposal.author_core_id!r} is not a declared tick participant"
            )
        if record.state is not ParticipantState.PENDING:
            raise DuplicateProposalError(
                f"participant {record.core_id!r} is already accounted as "
                f"{record.state.value} in the {proposal.pass_kind.value} pass"
            )
        identity = self._image.identity
        if (
            proposal.base_field_id != identity.base_field_id
            or proposal.base_tick_id != identity.base_tick_id
        ):
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
        descriptor.authority_grant().assert_delta_permitted(proposal.delta)
        updated = replace(
            record,
            state=ParticipantState.RETURNED,
            proposal=proposal,
        )
        records[record.core_id] = updated
        return updated

    def _mark(
        self,
        core_id: str,
        pass_kind: ProposalPass,
        state: ParticipantState,
        detail: str,
    ) -> ParticipantRecord:
        records = self._stage_records(pass_kind)
        record = records.get(core_id)
        if record is None:
            raise UnknownParticipantError(
                f"{core_id!r} is not a declared tick participant"
            )
        if record.state is not ParticipantState.PENDING:
            raise DuplicateProposalError(
                f"participant {core_id!r} is already accounted as "
                f"{record.state.value} in the {pass_kind.value} pass"
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
        """Close the first pass once every participant is accounted for."""

        if self._first_pass_closed:
            raise BarrierClosedError("the first pass is already closed")
        pending = self._pending_ids(self._first_pass)
        if pending:
            raise BarrierNotReadyError(
                "first-pass barrier cannot close; participants pending: "
                + ", ".join(pending)
            )
        self._first_pass_closed = True
        self._refinement = self._fresh_pass_records()

    def first_pass_proposals(self) -> tuple[Proposal, ...]:
        """The complete first-pass board, exposed only after the barrier."""

        if not self._first_pass_closed:
            raise BarrierNotReadyError(
                "the first-pass board is exposed only after its barrier closes"
            )
        return tuple(
            record.proposal
            for core_id, record in sorted(self._first_pass.items())
            if record.state is ParticipantState.RETURNED
            and record.proposal is not None
        )

    def close_refinement(self) -> None:
        """Close the refinement barrier once every participant is accounted."""

        if self._refinement is None:
            raise BarrierNotReadyError(
                "refinement has not begun; the first-pass barrier must close first"
            )
        if self._refinement_closed:
            raise BarrierClosedError("the refinement pass is already closed")
        pending = self._pending_ids(self._refinement)
        if pending:
            raise BarrierNotReadyError(
                "refinement barrier cannot close; participants pending: "
                + ", ".join(pending)
            )
        self._refinement_closed = True

    def refined_proposals(self) -> tuple[Proposal, ...]:
        """Every refined delta, exposed only after the refinement barrier."""

        if not self._refinement_closed or self._refinement is None:
            raise BarrierNotReadyError(
                "refined proposals are exposed only after the refinement barrier"
            )
        return tuple(
            record.proposal
            for core_id, record in sorted(self._refinement.items())
            if record.state is ParticipantState.RETURNED
            and record.proposal is not None
        )

    def assert_ready_for_consolidation(self) -> None:
        """Gate the consolidator: reachable only after the refinement barrier."""

        if not self._refinement_closed:
            raise BarrierNotReadyError(
                "consolidation requires the refinement barrier to be closed"
            )

    def participant_states(
        self, pass_kind: ProposalPass
    ) -> tuple[ParticipantRecord, ...]:
        """Immutable accounting view of one pass, ordered by core id."""

        if pass_kind is ProposalPass.FIRST:
            records = self._first_pass
        else:
            if self._refinement is None:
                raise BarrierNotReadyError(
                    "refinement has not begun; the first-pass barrier must close first"
                )
            records = self._refinement
        return tuple(records[core_id] for core_id in sorted(records))


__all__ = [
    "ProposalPass",
    "ParticipantState",
    "Proposal",
    "ParticipantRecord",
    "ProposalBoard",
]

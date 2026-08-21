"""Explicit fail-closed exception vocabulary for the heart organ.

Every rejection the heart can issue has a dedicated name so callers, tests,
and audit logs can distinguish *why* a proposal, participant, or commit was
refused.  Nothing in this package fails open.
"""
from __future__ import annotations


class HeartError(RuntimeError):
    """Base class for every heart (Field Compiler Organ) failure."""


class InvalidAuthorityGrantError(HeartError):
    """An authority grant was malformed for its authority class."""


class AuthorityViolationError(HeartError):
    """A proposal addressed regions outside its authority class scope."""


class CoreRegistryError(HeartError):
    """Base class for core registry failures."""


class DuplicateCoreError(CoreRegistryError):
    """A core id was registered twice."""


class UnknownCoreError(CoreRegistryError):
    """A referenced core id is not present in the registry."""


class NoActiveParticipantsError(CoreRegistryError):
    """A tick participant declaration found no active cores on the rail."""


class HeartbeatError(HeartError):
    """Base class for heartbeat/tick identity and tick-image failures."""


class StaleRailBindingError(HeartbeatError):
    """A rail reference is not bound to the frozen base field identity."""


class StaleBaseProposalError(HeartError):
    """A proposal is not bound to the frozen base field identity."""


class ProposalBoardError(HeartError):
    """Base class for proposal-board (noncanonical workspace) failures."""


class UnknownParticipantError(ProposalBoardError):
    """A proposal or accounting event named a non-participant core."""


class DuplicateProposalError(ProposalBoardError):
    """A participant submitted twice for the same pass."""


class RailMembershipError(ProposalBoardError):
    """A proposal's rail does not match the author's registered rail."""


class BarrierNotReadyError(ProposalBoardError):
    """A stage barrier was closed or read before its policy was satisfied."""


class BarrierClosedError(ProposalBoardError):
    """A submission or accounting event arrived after its stage closed."""


class HeartTransactionError(HeartError):
    """Base class for heart transaction boundary failures."""


class IngressDuringTickError(HeartTransactionError):
    """External ingress attempted to commit while a tick is in flight."""


__all__ = [
    "HeartError",
    "InvalidAuthorityGrantError",
    "AuthorityViolationError",
    "CoreRegistryError",
    "DuplicateCoreError",
    "UnknownCoreError",
    "NoActiveParticipantsError",
    "HeartbeatError",
    "StaleRailBindingError",
    "StaleBaseProposalError",
    "ProposalBoardError",
    "UnknownParticipantError",
    "DuplicateProposalError",
    "RailMembershipError",
    "BarrierNotReadyError",
    "BarrierClosedError",
    "HeartTransactionError",
    "IngressDuringTickError",
]

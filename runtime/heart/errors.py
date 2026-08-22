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


class RailWidthMismatchError(HeartbeatError):
    """A rail label claims a width other than the compiled rail's physical width."""


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


class ValveDuringTickError(HeartTransactionError):
    """The dormant valve attempted to commit while a tick is in flight."""


class CoreCommitError(HeartTransactionError):
    """A core attempted to commit canonical state; cores only propose."""


class TickBindingError(HeartTransactionError):
    """A consolidator commit is not bound to the in-flight tick identity/base."""


class FinalCommitAlreadyMadeError(HeartTransactionError):
    """A tick's final consolidator commit has already occurred."""


class LeaseDeniedError(HeartError):
    """A second Heart host failed to acquire the single-writer lease."""


class UnknownValveError(HeartError):
    """A referenced Heart valve id is not registered."""


class ValveClosedError(HeartError):
    """A Heart valve is CLOSED and admits no traffic."""


class ValveBudgetExceededError(HeartError):
    """A valve or global cardiac intake budget was exhausted."""


class ValveSourceMismatchError(HeartError):
    """An envelope source does not match the valve's registered source class."""


class ValveAuthorityError(HeartError):
    """The Heart could not derive valid authority from a valve definition."""


class ValveAdmissionError(HeartError):
    """A valve envelope failed local admission validation."""


class PoisonEventError(HeartError):
    """A durable ingress event is malformed, unauthorized, or otherwise poisonous."""


class ReplayEventError(HeartError):
    """A durable ingress event would be replayed or acknowledged out of order."""


class HealthCorruptionError(HeartError):
    """Durable Heart identity/health metadata is corrupt or untrusted."""


class HostStateError(HeartError):
    """The permanent Heart host is in an inconsistent or illegal state."""


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
    "RailWidthMismatchError",
    "StaleBaseProposalError",
    "ProposalBoardError",
    "UnknownParticipantError",
    "DuplicateProposalError",
    "RailMembershipError",
    "BarrierNotReadyError",
    "BarrierClosedError",
    "HeartTransactionError",
    "IngressDuringTickError",
    "ValveDuringTickError",
    "CoreCommitError",
    "TickBindingError",
    "FinalCommitAlreadyMadeError",
    "LeaseDeniedError",
    "UnknownValveError",
    "ValveClosedError",
    "ValveBudgetExceededError",
    "ValveSourceMismatchError",
    "ValveAuthorityError",
    "ValveAdmissionError",
    "PoisonEventError",
    "ReplayEventError",
    "HealthCorruptionError",
    "HostStateError",
]

"""The Axon heart (Field Compiler Organ): control-plane contracts.

Build A surface: authority classes, core registry, heartbeat/tick identity,
frozen tick images, the noncanonical proposal board, and the heart
transaction boundary.  No neural code lives here; canonical mutation is
performed only through the existing typed-delta machinery in
``runtime.field.delta``.
"""

from .authority import (
    CONSOLIDATOR_GOVERNED_REGIONS,
    DEFAULT_CORE_GOVERNED_REGIONS,
    DORMANT_VALVE_GOVERNED_REGIONS,
    INGRESS_OWNED_REGIONS,
    AuthorityClass,
    AuthorityGrant,
    IngressChannel,
)
from .board import (
    ParticipantRecord,
    ParticipantState,
    Proposal,
    ProposalBoard,
    ProposalPass,
)
from .errors import (
    AuthorityViolationError,
    BarrierClosedError,
    BarrierNotReadyError,
    CoreRegistryError,
    DuplicateCoreError,
    DuplicateProposalError,
    HeartError,
    HeartTransactionError,
    HeartbeatError,
    IngressDuringTickError,
    InvalidAuthorityGrantError,
    NoActiveParticipantsError,
    ProposalBoardError,
    RailMembershipError,
    StaleBaseProposalError,
    StaleRailBindingError,
    UnknownCoreError,
    UnknownParticipantError,
)
from .registry import CoreDescriptor, CoreRegistry, CoreStatus
from .tick import (
    TICK_IDENTITY_SCHEMA,
    TICK_IMAGE_SCHEMA,
    FrozenTickImage,
    HeartbeatClock,
    RailBinding,
    TickIdentity,
)
from .transaction import (
    COMMIT_SCHEMA,
    HeartCommit,
    HeartTransactionBoundary,
)

__all__ = [
    "AuthorityClass",
    "IngressChannel",
    "INGRESS_OWNED_REGIONS",
    "DORMANT_VALVE_GOVERNED_REGIONS",
    "CONSOLIDATOR_GOVERNED_REGIONS",
    "DEFAULT_CORE_GOVERNED_REGIONS",
    "AuthorityGrant",
    "CoreStatus",
    "CoreDescriptor",
    "CoreRegistry",
    "TICK_IDENTITY_SCHEMA",
    "TICK_IMAGE_SCHEMA",
    "TickIdentity",
    "HeartbeatClock",
    "RailBinding",
    "FrozenTickImage",
    "ProposalPass",
    "ParticipantState",
    "Proposal",
    "ParticipantRecord",
    "ProposalBoard",
    "COMMIT_SCHEMA",
    "HeartCommit",
    "HeartTransactionBoundary",
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

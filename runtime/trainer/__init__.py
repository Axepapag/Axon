"""Axon Trainer organ control-plane contracts.

No optimizer/training loop is activated here.  This package owns parameter
inventory, mutation authority, telemetry and lineage boundaries only.
"""
from .authority import (
    AUTHORIZED_MUTATION_SCHEMA,
    AuthorizedParameterMutation,
    ParameterAuthorityError,
    authorize_parameter_mutation,
)
from .contracts import (
    PARAMETER_GRANT_SCHEMA,
    PARAMETER_INVENTORY_SCHEMA,
    PARAMETER_MANIFEST_SCHEMA,
    PARAMETER_MODULE_SCHEMA,
    PARAMETER_PLAN_SCHEMA,
    PARAMETER_PROMOTION_SCHEMA,
    PARAMETER_TENSOR_SCHEMA,
    TRAINER_PARAMETER_SCHEMA,
    OrganKind,
    ParameterInventory,
    ParameterModuleDescriptor,
    ParameterModuleManifest,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterMutationPolicy,
    ParameterPromotionProposal,
    ParameterTensorRecord,
    TrainerCoreRole,
)
from .host import TrainerControlPlane
from .registry import (
    IncompleteParameterInventoryError,
    ParameterRegistry,
    ParameterRegistryError,
    parameter_value_sha256,
)
from .store import TrainerStateStore, TrainerStoreError
from .telemetry import (
    PARAMETER_STAT_SCHEMA,
    PARAMETER_TELEMETRY_SCHEMA,
    ParameterStat,
    ParameterTelemetryFrame,
    capture_parameter_telemetry,
)

__all__ = [
    "TRAINER_PARAMETER_SCHEMA",
    "PARAMETER_MODULE_SCHEMA",
    "PARAMETER_TENSOR_SCHEMA",
    "PARAMETER_MANIFEST_SCHEMA",
    "PARAMETER_INVENTORY_SCHEMA",
    "PARAMETER_GRANT_SCHEMA",
    "PARAMETER_PLAN_SCHEMA",
    "PARAMETER_PROMOTION_SCHEMA",
    "AUTHORIZED_MUTATION_SCHEMA",
    "PARAMETER_STAT_SCHEMA",
    "PARAMETER_TELEMETRY_SCHEMA",
    "OrganKind",
    "ParameterMutationPolicy",
    "TrainerCoreRole",
    "ParameterModuleDescriptor",
    "ParameterTensorRecord",
    "ParameterModuleManifest",
    "ParameterInventory",
    "ParameterMutationGrant",
    "ParameterMutationPlan",
    "ParameterPromotionProposal",
    "ParameterAuthorityError",
    "AuthorizedParameterMutation",
    "authorize_parameter_mutation",
    "ParameterRegistryError",
    "IncompleteParameterInventoryError",
    "parameter_value_sha256",
    "ParameterRegistry",
    "ParameterStat",
    "ParameterTelemetryFrame",
    "capture_parameter_telemetry",
    "TrainerStoreError",
    "TrainerStateStore",
    "TrainerControlPlane",
]

"""Core registry: the heart's roster of reasoning cores per d_model rail.

The registry is control-plane metadata only.  It records which cores exist,
which d_model rail each belongs to, and each core's lifecycle status
(active / offline-training / disabled).  The heart consults it to declare a
tick's participant set; only ACTIVE cores may be declared.

Real training cores are additionally bound to exact architecture, parameter,
optimizer, and Soul generations through ``CoreBinding`` records.  Binding and
mode transitions are fail-closed: they require an explicit operator-issued
``OperatorCoreGrant`` already present in the registry, so a core can never
bind itself or switch itself into training, and phantom core ids can never
auto-register.

``enter_offline_training`` switches a bound core into training under trainer
authority; the symmetric ``exit_offline_training`` restores reasoning mode
only against an accepted step-bundle reference whose terminal Soul matches
the binding's current generations.  Because a bound core's generations
advance during training, ``rebind_core`` is the narrow fail-closed advance
path: it moves descriptor and binding generations together, along Heart-
accepted lineage only, never changing mode (mode changes belong exclusively
to the enter/exit transitions).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

from runtime.field import LogicalRegion, canonical_sha256

from .authority import AuthorityClass, AuthorityGrant
from .errors import (
    CoreRegistryError,
    DuplicateCoreError,
    NoActiveParticipantsError,
    UnknownCoreError,
)

OPERATOR_CORE_GRANT_SCHEMA = "axon-heart-operator-core-grant-v1"
CORE_BINDING_SCHEMA = "axon-heart-core-binding-v1"

_CLAIM_ID_ATTRIBUTES: tuple[str, ...] = (
    "assignment_claim_id",
    "claim_id",
    "assignment_id",
    "claim_reference",
)


class CoreStatus(str, Enum):
    """Lifecycle status of a registered core."""

    ACTIVE = "active"
    OFFLINE_TRAINING = "offline_training"
    DISABLED = "disabled"


class UngrantedCoreRegistrationError(CoreRegistryError):
    """A core registration, update, or binding lacked a present operator grant."""


class InvalidCoreBindingError(CoreRegistryError):
    """A core binding is malformed or contradicts the registered descriptor."""


class InvalidModeTransitionError(CoreRegistryError):
    """A core mode transition lacked trainer authority or a valid assignment claim."""


@dataclass(frozen=True, slots=True)
class OperatorCoreGrant:
    """Explicit operator-issued authorization for one real core identity.

    Grants are issued by the operator into the registry before any
    non-operator registration path may use them.  A core may present a grant,
    but it must already be present in the registry and must match the
    descriptor exactly; a core can never introduce its own grant.
    """

    core_id: str
    d_model: int
    issued_by: str = "operator"
    grant_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.core_id, str) or not self.core_id:
            raise ValueError("OperatorCoreGrant.core_id must be a non-empty string")
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int):
            raise TypeError("OperatorCoreGrant.d_model must be an integer")
        if self.d_model <= 0:
            raise ValueError("OperatorCoreGrant.d_model must be positive")
        if not isinstance(self.issued_by, str) or not self.issued_by:
            raise ValueError("OperatorCoreGrant.issued_by must be a non-empty string")
        object.__setattr__(
            self,
            "grant_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": OPERATOR_CORE_GRANT_SCHEMA,
            "core_id": self.core_id,
            "d_model": self.d_model,
            "issued_by": self.issued_by,
        }

    def matches(self, descriptor: "CoreDescriptor") -> bool:
        return (
            self.core_id == descriptor.core_id and self.d_model == descriptor.d_model
        )


@dataclass(frozen=True, slots=True)
class CoreBinding:
    """Immutable binding of one registered real core to exact generations.

    The ``binding_id`` is the canonical sha256 of the exact bound identity:
    core, architecture, parameter generation, optimizer generation, Soul
    generation, and mode.  Any change to any bound generation is a new
    binding with a new identity.
    """

    core_id: str
    architecture_id: str
    parameter_generation: str
    optimizer_generation: str
    soul_id: str
    mode: CoreStatus = CoreStatus.ACTIVE
    binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.core_id, str) or not self.core_id:
            raise ValueError("CoreBinding.core_id must be a non-empty string")
        for attr in (
            "architecture_id",
            "parameter_generation",
            "optimizer_generation",
            "soul_id",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value:
                raise ValueError(f"CoreBinding.{attr} must be a non-empty string")
        mode = self.mode if isinstance(self.mode, CoreStatus) else CoreStatus(self.mode)
        if mode is CoreStatus.DISABLED:
            raise InvalidCoreBindingError(
                "disabled is a lifecycle status, not a binding mode; bindings "
                "are exactly reasoning (active) or offline_training"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "binding_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": CORE_BINDING_SCHEMA,
            "core_id": self.core_id,
            "architecture_id": self.architecture_id,
            "parameter_generation": self.parameter_generation,
            "optimizer_generation": self.optimizer_generation,
            "soul_id": self.soul_id,
            "mode": self.mode.value,
        }


@dataclass(frozen=True, slots=True)
class CoreDescriptor:
    """Immutable registry record for one reasoning core."""

    core_id: str
    d_model: int
    status: CoreStatus = CoreStatus.ACTIVE
    writable_regions: frozenset[LogicalRegion] | None = None
    architecture_id: str = "untrained-reasoning-core-v1"
    parameter_generation: str = "untrained"

    def __post_init__(self) -> None:
        if not isinstance(self.core_id, str) or not self.core_id:
            raise ValueError("CoreDescriptor.core_id must be a non-empty string")
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int):
            raise TypeError("CoreDescriptor.d_model must be an integer")
        if self.d_model <= 0:
            raise ValueError("CoreDescriptor.d_model must be positive")
        if not isinstance(self.architecture_id, str) or not self.architecture_id:
            raise ValueError("CoreDescriptor.architecture_id must be non-empty")
        if not isinstance(self.parameter_generation, str) or not self.parameter_generation:
            raise ValueError("CoreDescriptor.parameter_generation must be non-empty")
        status = (
            self.status
            if isinstance(self.status, CoreStatus)
            else CoreStatus(self.status)
        )
        object.__setattr__(self, "status", status)
        if self.writable_regions is not None:
            regions = frozenset(
                region if isinstance(region, LogicalRegion) else LogicalRegion(region)
                for region in self.writable_regions
            )
            if not regions:
                raise ValueError(
                    "CoreDescriptor.writable_regions must be None or non-empty"
                )
            object.__setattr__(self, "writable_regions", regions)

    def authority_grant(self) -> AuthorityGrant:
        """The core-class authority scope this descriptor is permitted."""

        return AuthorityGrant.core(self.writable_regions)


def _assignment_claim_record_types() -> tuple[type, ...]:
    """The A4 assignment-store record types, when that store is importable.

    A4's durable assignment store lives in ``runtime.trainer.assignments``;
    import it lazily and defensively so its absence or refactoring never
    breaks this module.  When it is importable, claim records are
    cross-checked against its real types; otherwise claims are validated
    structurally.
    """

    try:
        from runtime.trainer.assignments import (  # type: ignore[import-not-found]
            TrainingAssignment,
        )
    except Exception:
        return ()
    return (TrainingAssignment,)


def _resolve_assignment_claim_reference(claim: object, core_id: str) -> str:
    """Fail-closed extraction of a durable assignment claim reference.

    Accepts a non-empty claim id string, a live A4 ``TrainingAssignment``
    record held by this core, or a duck-typed claim record exposing one of
    the expected identity attributes.  Raises ``InvalidModeTransitionError``
    on anything else.
    """

    claim_types = _assignment_claim_record_types()
    if claim_types and isinstance(claim, claim_types):
        lease = getattr(claim, "lease", None)
        holder = getattr(lease, "holder_core_id", None)
        if holder != core_id:
            raise InvalidModeTransitionError(
                "assignment claim is not held by the requesting core"
            )
        status = getattr(claim, "status", None)
        live = frozenset({"claimed", "active"})
        if getattr(status, "value", status) not in live:
            raise InvalidModeTransitionError(
                "assignment claim is not in a live (claimed/active) status"
            )
        reference = getattr(claim, "assignment_id", None)
        if not isinstance(reference, str) or not reference:
            raise InvalidModeTransitionError(
                "assignment claim record carries no durable assignment id"
            )
        return reference

    if isinstance(claim, str):
        reference = claim
    elif claim is not None:
        reference = None
        for attr in _CLAIM_ID_ATTRIBUTES:
            value = getattr(claim, attr, None)
            if isinstance(value, str) and value:
                reference = value
                break
        if reference is not None:
            claimed_core = getattr(claim, "core_id", core_id)
            if claimed_core != core_id:
                raise InvalidModeTransitionError(
                    "assignment claim belongs to a different core"
                )
    else:
        reference = None

    if not isinstance(reference, str) or not reference.strip():
        raise InvalidModeTransitionError(
            "entering offline training requires a durable assignment claim reference"
        )
    return reference


def _require_hex_identity(value: object, label: str) -> str:
    """Fail-closed validation of one 64-character lowercase sha256 identity."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise InvalidModeTransitionError(
            f"accepted bundle {label} must be a 64-character sha256 hex digest"
        )
    return value


def _resolve_accepted_bundle_reference(bundle: object, core_id: str) -> str:
    """Fail-closed extraction of the terminal Soul of one accepted bundle.

    Accepts a real ``AcceptedTrainingStepBundle`` (A-scope ``step_bundle``)
    or a duck-typed record exposing the same identity attributes.  Returns
    the bundle's terminal (``after_soul_id``) Soul generation.  Raises
    ``InvalidModeTransitionError`` on anything missing, malformed, foreign,
    or misshapen: an exit from training must name exactly one genuine
    accepted bundle for exactly this core.
    """

    if bundle is None:
        raise InvalidModeTransitionError(
            "exiting offline training requires an accepted bundle reference"
        )
    bundle_core = getattr(bundle, "core_id", None)
    if not isinstance(bundle_core, str) or not bundle_core:
        raise InvalidModeTransitionError(
            "accepted bundle carries no core identity"
        )
    if bundle_core != core_id:
        raise InvalidModeTransitionError(
            "accepted bundle belongs to a different core"
        )
    _require_hex_identity(getattr(bundle, "bundle_id", None), "bundle_id")
    after_soul_id = _require_hex_identity(
        getattr(bundle, "after_soul_id", None), "after_soul_id"
    )
    return after_soul_id


class CoreRegistry:
    """Fail-closed roster consulted to declare a tick's participant set."""

    def __init__(self, descriptors: Iterable[CoreDescriptor] = ()) -> None:
        self._cores: dict[str, CoreDescriptor] = {}
        self._grants: dict[str, OperatorCoreGrant] = {}
        self._bindings: dict[str, CoreBinding] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    # --- operator-issued grants -------------------------------------------

    def issue_operator_grant(self, grant: OperatorCoreGrant) -> None:
        """Operator path: authorize one real core identity for later use."""

        if not isinstance(grant, OperatorCoreGrant):
            raise TypeError("issue_operator_grant requires an OperatorCoreGrant")
        if grant.grant_id in self._grants:
            raise DuplicateCoreError(
                f"operator grant for core {grant.core_id!r} is already issued"
            )
        self._grants[grant.grant_id] = grant

    def _assert_grant_usable(
        self,
        grant: OperatorCoreGrant,
        descriptor: CoreDescriptor,
    ) -> None:
        """Fail closed unless the grant is already present and exactly matches.

        A core may present a grant, but it must already live in the registry;
        a core can never introduce its own authorization.
        """

        if not isinstance(grant, OperatorCoreGrant):
            raise TypeError("grant must be an OperatorCoreGrant")
        issued = self._grants.get(grant.grant_id)
        if issued is None or issued != grant:
            raise UngrantedCoreRegistrationError(
                f"no operator-issued grant for core {descriptor.core_id!r} is "
                "present in the registry; refusing auto-registration"
            )
        if not grant.matches(descriptor):
            raise UngrantedCoreRegistrationError(
                f"operator grant for core {grant.core_id!r} does not match "
                f"descriptor {descriptor.core_id!r}"
            )

    # --- registration -----------------------------------------------------

    def register(
        self,
        descriptor: CoreDescriptor,
        *,
        grant: OperatorCoreGrant | None = None,
    ) -> None:
        """Register one core descriptor.

        Without a grant this is the operator path (the caller holding the
        registry is the operator).  With a grant, the grant must already be
        present in the registry and match the descriptor exactly, so phantom
        core ids can never auto-register.
        """

        if not isinstance(descriptor, CoreDescriptor):
            raise TypeError("CoreRegistry.register requires a CoreDescriptor")
        if grant is not None:
            self._assert_grant_usable(grant, descriptor)
        if descriptor.core_id in self._cores:
            raise DuplicateCoreError(
                f"core {descriptor.core_id!r} is already registered"
            )
        self._cores[descriptor.core_id] = descriptor

    def update_descriptor(
        self,
        descriptor: CoreDescriptor,
        *,
        grant: OperatorCoreGrant | None = None,
    ) -> None:
        """Replace one registered descriptor, validating bound generations.

        The core must already be registered (updating a phantom is an
        auto-registration attempt and fails closed).  When a binding exists,
        the updated descriptor must not contradict the bound architecture or
        parameter generation; rebind instead.
        """

        if not isinstance(descriptor, CoreDescriptor):
            raise TypeError("update_descriptor requires a CoreDescriptor")
        if descriptor.core_id not in self._cores:
            raise UnknownCoreError(
                f"cannot update unknown core {descriptor.core_id!r}"
            )
        if grant is not None:
            self._assert_grant_usable(grant, descriptor)
        binding = self._bindings.get(descriptor.core_id)
        if binding is not None and (
            binding.architecture_id != descriptor.architecture_id
            or binding.parameter_generation != descriptor.parameter_generation
        ):
            raise InvalidCoreBindingError(
                f"updated descriptor for {descriptor.core_id!r} contradicts "
                "the bound architecture/parameter generation; rebind instead"
            )
        if (
            binding is not None
            and binding.mode is not descriptor.status
            and descriptor.status is not CoreStatus.DISABLED
        ):
            # DISABLED is a lifecycle status outside the bound mode: a bound
            # core may be disabled, but it may not quietly occupy a mode
            # other than its binding.
            raise InvalidCoreBindingError(
                f"updated descriptor status {descriptor.status.value!r} does not "
                f"match the bound mode {binding.mode.value!r}"
            )
        self._cores[descriptor.core_id] = descriptor

    # --- exact generation binding ------------------------------------------

    def bind_core(
        self,
        binding: CoreBinding,
        *,
        grant: OperatorCoreGrant,
    ) -> CoreBinding:
        """Bind a registered real core to exact generations.

        The core must already be registered (phantom ids fail closed), the
        binding must match the registered descriptor's architecture and
        parameter generation, and the operator grant must already be present
        in the registry.
        """

        if not isinstance(binding, CoreBinding):
            raise TypeError("bind_core requires a CoreBinding")
        descriptor = self.get(binding.core_id)
        if not grant.matches(descriptor):
            raise UngrantedCoreRegistrationError(
                f"operator grant for core {grant.core_id!r} does not match "
                f"core {binding.core_id!r}"
            )
        self._assert_grant_usable(grant, descriptor)
        if (
            binding.architecture_id != descriptor.architecture_id
            or binding.parameter_generation != descriptor.parameter_generation
        ):
            raise InvalidCoreBindingError(
                f"binding for {binding.core_id!r} contradicts the registered "
                "descriptor architecture/parameter generation"
            )
        if binding.mode is not descriptor.status:
            raise InvalidCoreBindingError(
                f"binding mode {binding.mode.value!r} does not match the "
                f"registered status {descriptor.status.value!r}"
            )
        self._bindings[binding.core_id] = binding
        return binding

    def binding(self, core_id: str) -> CoreBinding:
        try:
            return self._bindings[core_id]
        except KeyError:
            raise UnknownCoreError(
                f"core {core_id!r} has no exact generation binding"
            ) from None

    def bindings(self) -> tuple[CoreBinding, ...]:
        return tuple(self._bindings[core_id] for core_id in sorted(self._bindings))

    # --- mode transitions ---------------------------------------------------

    def enter_offline_training(
        self,
        core_id: str,
        *,
        grant: AuthorityGrant,
        claim: object,
    ) -> CoreBinding:
        """Switch one bound core into OFFLINE_TRAINING under trainer authority.

        Requires a valid trainer-class authority grant and a durable
        assignment claim reference from the assignment store (A4 scope).
        Updates the registry status and the binding mode together; the core
        leaves the active participant set immediately.
        """

        descriptor = self.get(core_id)
        binding = self.binding(core_id)
        if not isinstance(grant, AuthorityGrant):
            raise TypeError("enter_offline_training requires an AuthorityGrant")
        if grant.authority_class is not AuthorityClass.TRAINER:
            raise InvalidModeTransitionError(
                f"entering offline training requires trainer authority, not "
                f"{grant.authority_class.value!r}"
            )
        _resolve_assignment_claim_reference(claim, core_id)
        updated_descriptor = replace(descriptor, status=CoreStatus.OFFLINE_TRAINING)
        self._cores[core_id] = updated_descriptor
        updated_binding = replace(binding, mode=CoreStatus.OFFLINE_TRAINING)
        self._bindings[core_id] = updated_binding
        return updated_binding

    def rebind_core(
        self,
        binding: CoreBinding,
        *,
        grant: OperatorCoreGrant,
    ) -> CoreBinding:
        """Advance one bound core to new exact generations along accepted lineage.

        ``bind_core`` and ``update_descriptor`` each refuse to move a
        generation already bound (they cross-check against the other side);
        this is the single narrow path that moves the descriptor and the
        binding TOGETHER, which is the only generation advance that cannot
        contradict either record.  The architecture is immutable, the mode is
        immutable here (mode changes belong to ``enter_offline_training`` /
        ``exit_offline_training``), and the operator grant must already be
        present, so a core still can never advance its own binding.  The
        caller (the Heart-side training session) must only ever present
        generations that a Heart-accepted bundle published; generation
        continuity is enforced durably by the assignment journal and the
        accepted-bundle lineage, not by this control-plane record.
        """

        if not isinstance(binding, CoreBinding):
            raise TypeError("rebind_core requires a CoreBinding")
        descriptor = self.get(binding.core_id)
        current = self.binding(binding.core_id)
        if not isinstance(grant, OperatorCoreGrant):
            raise TypeError("rebind_core requires an OperatorCoreGrant")
        self._assert_grant_usable(grant, descriptor)
        if (
            binding.architecture_id != current.architecture_id
            or binding.architecture_id != descriptor.architecture_id
        ):
            raise InvalidCoreBindingError(
                f"rebind for {binding.core_id!r} changes the bound architecture; "
                "architecture is immutable across a rebind"
            )
        if binding.mode is not current.mode:
            raise InvalidModeTransitionError(
                "a generation rebind cannot change mode; use the offline-training "
                "mode transitions"
            )
        if binding == current:
            return current
        updated_descriptor = replace(
            descriptor, parameter_generation=binding.parameter_generation
        )
        self._cores[binding.core_id] = updated_descriptor
        self._bindings[binding.core_id] = binding
        return binding

    def exit_offline_training(
        self,
        core_id: str,
        *,
        grant: AuthorityGrant,
        accepted_bundle: object,
    ) -> CoreBinding:
        """Restore one bound core from OFFLINE_TRAINING to reasoning mode.

        Symmetric to ``enter_offline_training``: a trainer-class grant is
        required, the core must be registered and bound (phantom ids fail
        closed), and the accepted bundle reference must be one genuine
        accepted step bundle for exactly this core whose terminal Soul IS
        the binding's current bound Soul generation.  A bundle from before
        the latest accepted step, from another core, missing, malformed, or
        carrying generations that disagree with the binding fails closed:
        training ends only at an exact accepted boundary, never on stale or
        unverified lineage.
        """

        descriptor = self.get(core_id)
        binding = self.binding(core_id)
        if not isinstance(grant, AuthorityGrant):
            raise TypeError("exit_offline_training requires an AuthorityGrant")
        if grant.authority_class is not AuthorityClass.TRAINER:
            raise InvalidModeTransitionError(
                f"exiting offline training requires trainer authority, not "
                f"{grant.authority_class.value!r}"
            )
        if binding.mode is not CoreStatus.OFFLINE_TRAINING:
            raise InvalidModeTransitionError(
                f"core {core_id!r} is not bound in offline training "
                f"(mode={binding.mode.value!r})"
            )
        after_soul_id = _resolve_accepted_bundle_reference(accepted_bundle, core_id)
        if after_soul_id != binding.soul_id:
            raise InvalidModeTransitionError(
                "stale or mismatched accepted bundle: its terminal Soul generation "
                "does not match the binding's current bound Soul"
            )
        for attr in ("parameter_generation", "optimizer_generation"):
            value = getattr(accepted_bundle, attr, None)
            if isinstance(value, str) and value and value != getattr(binding, attr):
                raise InvalidModeTransitionError(
                    f"stale or mismatched accepted bundle: its {attr} disagrees "
                    "with the binding's current bound generation"
                )
        updated_descriptor = replace(descriptor, status=CoreStatus.ACTIVE)
        self._cores[core_id] = updated_descriptor
        updated_binding = replace(binding, mode=CoreStatus.ACTIVE)
        self._bindings[core_id] = updated_binding
        return updated_binding

    def get(self, core_id: str) -> CoreDescriptor:
        try:
            return self._cores[core_id]
        except KeyError:
            raise UnknownCoreError(f"unknown core {core_id!r}") from None

    def __contains__(self, core_id: object) -> bool:
        return core_id in self._cores

    def __len__(self) -> int:
        return len(self._cores)

    def descriptors(self) -> tuple[CoreDescriptor, ...]:
        return tuple(self._cores[core_id] for core_id in sorted(self._cores))

    def active(self, d_model: int | None = None) -> tuple[CoreDescriptor, ...]:
        """Active cores, optionally restricted to one d_model rail."""

        return tuple(
            descriptor
            for descriptor in self.descriptors()
            if descriptor.status is CoreStatus.ACTIVE
            and (d_model is None or descriptor.d_model == d_model)
        )

    def declare_participants(self, d_model: int) -> tuple[CoreDescriptor, ...]:
        """The tick participant set for one rail; fails closed when empty."""

        participants = self.active(d_model)
        if not participants:
            raise NoActiveParticipantsError(
                f"no active cores registered on d_model rail {d_model}"
            )
        return participants


__all__ = [
    "CORE_BINDING_SCHEMA",
    "OPERATOR_CORE_GRANT_SCHEMA",
    "CoreBinding",
    "CoreDescriptor",
    "CoreRegistry",
    "CoreStatus",
    "InvalidCoreBindingError",
    "InvalidModeTransitionError",
    "OperatorCoreGrant",
    "UngrantedCoreRegistrationError",
]

"""Core registry: the heart's roster of reasoning cores per d_model rail.

The registry is control-plane metadata only.  It records which cores exist,
which d_model rail each belongs to, and each core's lifecycle status
(active / offline-training / disabled).  The heart consults it to declare a
tick's participant set; only ACTIVE cores may be declared.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from runtime.field import LogicalRegion

from .authority import AuthorityGrant
from .errors import (
    DuplicateCoreError,
    NoActiveParticipantsError,
    UnknownCoreError,
)


class CoreStatus(str, Enum):
    """Lifecycle status of a registered core."""

    ACTIVE = "active"
    OFFLINE_TRAINING = "offline_training"
    DISABLED = "disabled"


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


class CoreRegistry:
    """Fail-closed roster consulted to declare a tick's participant set."""

    def __init__(self, descriptors: Iterable[CoreDescriptor] = ()) -> None:
        self._cores: dict[str, CoreDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: CoreDescriptor) -> None:
        if not isinstance(descriptor, CoreDescriptor):
            raise TypeError("CoreRegistry.register requires a CoreDescriptor")
        if descriptor.core_id in self._cores:
            raise DuplicateCoreError(
                f"core {descriptor.core_id!r} is already registered"
            )
        self._cores[descriptor.core_id] = descriptor

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
    "CoreDescriptor",
    "CoreRegistry",
    "CoreStatus",
]

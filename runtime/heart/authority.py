"""Ratified authority classes for heart-governed canonical mutation.

Doctrine (docs/SOURCE_OF_TRUTH.md, "Authority Classes"):

- external ingress (user/tool/advisor) may submit heart-governed mutations
  targeting only its runtime-owned regions, and only between ticks;
- the dormant valve may submit heart-governed materialization of governed
  ``cortex``; it never independently writes truth;
- core proposals may target only the scopes their authority class permits;
- the consolidator's proposal may address every ordinary canonical region;
- canonical identity amendments require the separate identity-steward class;
- only the heart's transaction layer converts any proposal into canonical
  state.

This module is the checkable authority matrix.  It decides *who may address
which region*; the canonical typed-delta machinery in ``runtime.field.delta``
remains the binding validator of every committed byte (including the current
bootstrap ``CORE_WRITABLE_REGIONS`` restriction, which this model does not
widen).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from runtime.field import (
    CANONICAL_REGION_ORDER,
    CORE_WRITABLE_REGIONS,
    FieldDelta,
    LogicalRegion,
)

from .errors import AuthorityViolationError, InvalidAuthorityGrantError


class AuthorityClass(str, Enum):
    """The ratified authority classes that may submit to the heart."""

    EXTERNAL_INGRESS = "external_ingress"
    DORMANT_VALVE = "dormant_valve"
    CORE = "core"
    CONSOLIDATOR = "consolidator"
    IDENTITY_STEWARD = "identity_steward"


class IngressChannel(str, Enum):
    """External ingress channels and their runtime-owned regions."""

    USER = "user"
    TOOL = "tool"
    ADVISOR = "advisor"


INGRESS_OWNED_REGIONS: Mapping[IngressChannel, frozenset[LogicalRegion]] = (
    MappingProxyType(
        {
            IngressChannel.USER: frozenset({LogicalRegion.USER_INPUT}),
            IngressChannel.TOOL: frozenset({LogicalRegion.TOOL_RESULTS}),
            IngressChannel.ADVISOR: frozenset({LogicalRegion.ADVISOR_INPUT}),
        }
    )
)

DORMANT_VALVE_GOVERNED_REGIONS: frozenset[LogicalRegion] = frozenset(
    {LogicalRegion.CORTEX}
)

CONSOLIDATOR_GOVERNED_REGIONS: frozenset[LogicalRegion] = frozenset(
    region for region in CANONICAL_REGION_ORDER if region is not LogicalRegion.IDENTITY
)

IDENTITY_STEWARD_GOVERNED_REGIONS: frozenset[LogicalRegion] = frozenset(
    {LogicalRegion.IDENTITY}
)

DEFAULT_CORE_GOVERNED_REGIONS: frozenset[LogicalRegion] = CORE_WRITABLE_REGIONS


def _coerce_regions(value: Iterable[LogicalRegion | str]) -> frozenset[LogicalRegion]:
    regions: set[LogicalRegion] = set()
    for item in value:
        try:
            regions.add(item if isinstance(item, LogicalRegion) else LogicalRegion(item))
        except (TypeError, ValueError) as exc:
            raise InvalidAuthorityGrantError(
                f"unknown governed region {item!r}"
            ) from exc
    return frozenset(regions)


@dataclass(frozen=True, slots=True)
class AuthorityGrant:
    """One checkable grant of heart-governed write scope.

    A grant names an authority class plus the class-specific scope evidence:
    ingress grants carry their channel, core grants carry their permitted
    regions.  Any other combination fails closed at construction.
    """

    authority_class: AuthorityClass
    channel: IngressChannel | None = None
    permitted_regions: frozenset[LogicalRegion] | None = None

    def __post_init__(self) -> None:
        try:
            authority_class = (
                self.authority_class
                if isinstance(self.authority_class, AuthorityClass)
                else AuthorityClass(self.authority_class)
            )
        except (TypeError, ValueError) as exc:
            raise InvalidAuthorityGrantError(
                f"unknown authority class {self.authority_class!r}"
            ) from exc
        object.__setattr__(self, "authority_class", authority_class)

        channel = self.channel
        if channel is not None and not isinstance(channel, IngressChannel):
            try:
                channel = IngressChannel(channel)
            except (TypeError, ValueError) as exc:
                raise InvalidAuthorityGrantError(
                    f"unknown ingress channel {self.channel!r}"
                ) from exc
        object.__setattr__(self, "channel", channel)

        permitted = self.permitted_regions
        if permitted is not None:
            permitted = _coerce_regions(permitted)
            if not permitted:
                raise InvalidAuthorityGrantError(
                    "permitted_regions must be None or non-empty"
                )
            object.__setattr__(self, "permitted_regions", permitted)

        if authority_class is AuthorityClass.EXTERNAL_INGRESS:
            if channel is None:
                raise InvalidAuthorityGrantError(
                    "external ingress grants require an ingress channel"
                )
            if permitted is not None:
                raise InvalidAuthorityGrantError(
                    "external ingress scope is fixed by its channel"
                )
        elif authority_class is AuthorityClass.CORE:
            if channel is not None:
                raise InvalidAuthorityGrantError(
                    "core grants may not carry an ingress channel"
                )
            if permitted is None:
                object.__setattr__(
                    self, "permitted_regions", DEFAULT_CORE_GOVERNED_REGIONS
                )
        else:
            if channel is not None or permitted is not None:
                raise InvalidAuthorityGrantError(
                    f"{authority_class.value} scope is fixed by doctrine"
                )

    @classmethod
    def ingress(cls, channel: IngressChannel | str) -> "AuthorityGrant":
        return cls(authority_class=AuthorityClass.EXTERNAL_INGRESS, channel=channel)

    @classmethod
    def dormant_valve(cls) -> "AuthorityGrant":
        return cls(authority_class=AuthorityClass.DORMANT_VALVE)

    @classmethod
    def core(
        cls,
        permitted_regions: Iterable[LogicalRegion | str] | None = None,
    ) -> "AuthorityGrant":
        return cls(
            authority_class=AuthorityClass.CORE,
            permitted_regions=(
                None if permitted_regions is None else frozenset(permitted_regions)
            ),
        )

    @classmethod
    def consolidator(cls) -> "AuthorityGrant":
        return cls(authority_class=AuthorityClass.CONSOLIDATOR)

    @classmethod
    def identity_steward(cls) -> "AuthorityGrant":
        """Exceptional, explicit grant for versioned canonical identity amendments."""

        return cls(authority_class=AuthorityClass.IDENTITY_STEWARD)

    @property
    def governed_regions(self) -> frozenset[LogicalRegion]:
        authority_class = self.authority_class
        if authority_class is AuthorityClass.EXTERNAL_INGRESS:
            # __post_init__ guarantees a channel for ingress grants.
            return INGRESS_OWNED_REGIONS[self.channel]  # type: ignore[index]
        if authority_class is AuthorityClass.DORMANT_VALVE:
            return DORMANT_VALVE_GOVERNED_REGIONS
        if authority_class is AuthorityClass.CORE:
            # __post_init__ guarantees permitted regions for core grants.
            return self.permitted_regions  # type: ignore[return-value]
        if authority_class is AuthorityClass.IDENTITY_STEWARD:
            return IDENTITY_STEWARD_GOVERNED_REGIONS
        return CONSOLIDATOR_GOVERNED_REGIONS

    def governs(self, region: LogicalRegion | str) -> bool:
        logical = region if isinstance(region, LogicalRegion) else LogicalRegion(region)
        return logical in self.governed_regions

    def assert_governs(self, region: LogicalRegion | str) -> None:
        logical = region if isinstance(region, LogicalRegion) else LogicalRegion(region)
        if logical not in self.governed_regions:
            raise AuthorityViolationError(
                f"{self.authority_class.value} authority does not govern "
                f"region {logical.value!r}"
            )

    def assert_delta_permitted(self, delta: FieldDelta) -> None:
        """Fail closed if any operation addresses an ungoverned region."""

        if not isinstance(delta, FieldDelta):
            raise TypeError("assert_delta_permitted requires a FieldDelta")
        for operation in delta.operations:
            self.assert_governs(operation.region)


__all__ = [
    "CONSOLIDATOR_GOVERNED_REGIONS",
    "DEFAULT_CORE_GOVERNED_REGIONS",
    "DORMANT_VALVE_GOVERNED_REGIONS",
    "IDENTITY_STEWARD_GOVERNED_REGIONS",
    "INGRESS_OWNED_REGIONS",
    "AuthorityClass",
    "AuthorityGrant",
    "IngressChannel",
]

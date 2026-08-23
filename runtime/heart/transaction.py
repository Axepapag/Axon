"""The heart transaction boundary: the only path to canonical state.

Only this layer converts a proposal into canonical shared-field state.  It
validates base freshness, authority class, and conflicts, then applies the
validated decision through the existing canonical typed-delta machinery in
``runtime.field.delta`` (``validate_delta`` / ``apply_delta``), which remains
the binding gate for typing, region sealing, bounds, and overlap.

External ingress commits only between ticks; while a tick is in flight, only
the consolidator's validated decision may commit, and the tick ends at that
commit — and only there.  Cores only propose; a core grant is never
commit-capable, regardless of tick state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from runtime.field import (
    FieldDelta,
    SharedFieldSnapshot,
    apply_delta,
    canonical_sha256,
    validate_delta,
)

from .authority import AuthorityClass, AuthorityGrant
from .errors import (
    CoreCommitError,
    FinalCommitAlreadyMadeError,
    HeartTransactionError,
    IngressDuringTickError,
    StaleBaseProposalError,
    AuthorityViolationError,
    TickBindingError,
    ValveDuringTickError,
)
from .tick import FrozenTickImage, TickIdentity

COMMIT_SCHEMA = "axon-heart-commit-v1"


@dataclass(frozen=True, slots=True)
class HeartCommit:
    """Immutable receipt of one atomic canonical commit by the heart."""

    base_field_id: str
    base_tick_id: int
    successor: SharedFieldSnapshot
    delta: FieldDelta
    grant: AuthorityGrant
    tick: TickIdentity | None = None
    valve_provenance: Mapping[str, Any] = field(default_factory=dict)
    commit_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.successor, SharedFieldSnapshot):
            raise TypeError("HeartCommit.successor must be a SharedFieldSnapshot")
        if not isinstance(self.delta, FieldDelta):
            raise TypeError("HeartCommit.delta must be a FieldDelta")
        if not isinstance(self.grant, AuthorityGrant):
            raise TypeError("HeartCommit.grant must be an AuthorityGrant")
        if self.tick is not None and not isinstance(self.tick, TickIdentity):
            raise TypeError("HeartCommit.tick must be a TickIdentity or None")
        if not isinstance(self.valve_provenance, Mapping):
            raise TypeError("HeartCommit.valve_provenance must be a mapping")
        object.__setattr__(self, "valve_provenance", dict(self.valve_provenance))
        if self.successor.parent_field_id != self.base_field_id:
            raise HeartTransactionError(
                "commit successor is not parented to the committed base field"
            )
        if self.delta.base_field_id != self.base_field_id:
            raise HeartTransactionError(
                "commit delta is not bound to the committed base field"
            )
        object.__setattr__(self, "commit_id", canonical_sha256(self.to_canonical_dict()))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": COMMIT_SCHEMA,
            "base_field_id": self.base_field_id,
            "base_tick_id": self.base_tick_id,
            "successor_field_id": self.successor.field_id,
            "successor_tick_id": self.successor.tick_id,
            "delta_id": self.delta.delta_id,
            "authority_class": self.grant.authority_class.value,
            "governed_regions": sorted(region.value for region in self.grant.governed_regions),
            "tick_uid": None if self.tick is None else self.tick.tick_uid,
            "valve_provenance": dict(self.valve_provenance),
        }


class HeartTransactionBoundary:
    """Validates and atomically commits proposals into canonical state."""

    def __init__(self) -> None:
        self._open_tick: TickIdentity | None = None
        self._last_consumed_tick: TickIdentity | None = None

    @property
    def tick_in_flight(self) -> bool:
        return self._open_tick is not None

    def note_tick_opened(self, image: FrozenTickImage) -> None:
        if not isinstance(image, FrozenTickImage):
            raise TypeError("note_tick_opened requires a FrozenTickImage")
        if self._open_tick is not None:
            raise HeartTransactionError(
                f"tick {self._open_tick.tick_sequence} is already in flight"
            )
        self._open_tick = image.identity

    def note_tick_closed(self, identity: TickIdentity) -> None:
        if self._open_tick is None:
            raise HeartTransactionError("no tick is in flight")
        if self._open_tick != identity:
            raise HeartTransactionError(
                "the closing tick identity does not match the in-flight tick"
            )
        self._open_tick = None

    def validate_proposal(
        self,
        base: SharedFieldSnapshot,
        delta: FieldDelta,
        grant: AuthorityGrant,
    ) -> None:
        """Full fail-closed validation; raises on any violation."""

        if not isinstance(base, SharedFieldSnapshot):
            raise TypeError("validate_proposal requires a SharedFieldSnapshot base")
        if not isinstance(delta, FieldDelta):
            raise TypeError("validate_proposal requires a FieldDelta")
        if not isinstance(grant, AuthorityGrant):
            raise TypeError("validate_proposal requires an AuthorityGrant")

        if (
            delta.base_field_id != base.field_id
            or delta.base_tick_id != base.tick_id
        ):
            raise StaleBaseProposalError(
                "proposal is not bound to the frozen base field "
                f"{base.field_id!r} tick {base.tick_id}"
            )
        grant.assert_delta_permitted(delta)
        # Canonical machinery remains the binding gate: typing, region
        # sealing, bounds, and overlapping/conflicting sparse edits.  The
        # authority model may permit regions beyond the bootstrap core-writable
        # set (ingress-owned regions, cortex for the valve).
        validate_delta(base, delta, permitted_regions=grant.governed_regions)

    def commit(
        self,
        base: SharedFieldSnapshot,
        delta: FieldDelta,
        grant: AuthorityGrant,
        *,
        tick: TickIdentity | None = None,
        valve_provenance: Mapping[str, Any] | None = None,
    ) -> HeartCommit:
        """Validate and atomically apply one proposal as the successor field.

        A successful consolidator commit atomically consumes the in-flight
        tick: the tick ends at this commit — and only there.
        """

        if not isinstance(grant, AuthorityGrant):
            raise TypeError("commit requires an AuthorityGrant")
        # Cores only propose; a core grant is never commit-capable,
        # regardless of tick state.
        if grant.authority_class is AuthorityClass.CORE:
            raise CoreCommitError(
                "cores only propose; a core grant may never commit canonical state"
            )
        if grant.authority_class is AuthorityClass.CONSOLIDATOR:
            # The consolidator's decision commits only against the tick that
            # produced it: the tick token is mandatory, never optional, and a
            # tick must be in flight.
            if tick is None:
                raise TickBindingError(
                    "a consolidator commit requires the in-flight tick token"
                )
            if self._open_tick is None:
                if (
                    self._last_consumed_tick is not None
                    and tick == self._last_consumed_tick
                ):
                    raise FinalCommitAlreadyMadeError(
                        "the final consolidator commit for this tick has already been made"
                    )
                raise TickBindingError(
                    "a consolidator decision may commit only while its tick "
                    "is in flight"
                )
        elif self._open_tick is not None:
            if grant.authority_class is AuthorityClass.EXTERNAL_INGRESS:
                raise IngressDuringTickError(
                    "external ingress commits only between ticks; a tick is in flight"
                )
            if grant.authority_class is AuthorityClass.DORMANT_VALVE:
                raise ValveDuringTickError(
                    "the dormant valve may materialize structured knowledge "
                    "only between ticks; a tick is in flight"
                )
            raise AuthorityViolationError(
                "only the consolidator's decision may commit while a tick "
                "is in flight"
            )
        if tick is not None:
            if self._open_tick is None:
                raise TickBindingError(
                    "commit claims a tick but no tick is in flight"
                )
            if tick != self._open_tick:
                raise TickBindingError(
                    "commit tick identity does not match the in-flight tick"
                )
            if (
                tick.base_field_id != base.field_id
                or tick.base_tick_id != base.tick_id
            ):
                raise TickBindingError(
                    "commit base does not match the in-flight tick's frozen base"
                )
        self.validate_proposal(base, delta, grant)
        successor = apply_delta(base, delta, permitted_regions=grant.governed_regions)
        if grant.authority_class is AuthorityClass.CONSOLIDATOR:
            # The tick ends at the heart commit — and only there.  Consuming
            # it here makes any second commit from this tick fail closed.
            self._last_consumed_tick = self._open_tick
            self._open_tick = None
        return HeartCommit(
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
            successor=successor,
            delta=delta,
            grant=grant,
            tick=tick,
            valve_provenance={} if valve_provenance is None else dict(valve_provenance),
        )


__all__ = [
    "COMMIT_SCHEMA",
    "HeartCommit",
    "HeartTransactionBoundary",
]

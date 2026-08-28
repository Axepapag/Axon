"""Heartbeat cadence, tick identity, and the frozen per-tick image.

The heartbeat is the heart's own cadence, distinct from a cognitive tick.
Identities issued here are strictly monotonic and every tick is bound to
exactly one frozen canonical base ``field_id``.  A ``FrozenTickImage`` is a
derived, immutable projection of that base: it carries rail references, never
second canonical state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from runtime.field import (
    CompiledD64Field,
    D64SemanticSurface,
    LogicalRegion,
    RegionMaskPolicy,
    SharedFieldSnapshot,
    canonical_sha256,
)

from .errors import HeartbeatError, RailWidthMismatchError, StaleRailBindingError

TICK_IDENTITY_SCHEMA = "axon-heart-tick-identity-v1"
TICK_IMAGE_SCHEMA = "axon-heart-frozen-tick-image-v2"
DERIVED_VIEW_SCHEMA = "axon-heart-derived-view-v1"


def derive_view_id(
    region_masks: Mapping[LogicalRegion, RegionMaskPolicy] | None = None,
) -> str:
    """Return an explicit identity for one noncanonical attention view."""

    masks = region_masks or {}
    payload = {
        "schema": DERIVED_VIEW_SCHEMA,
        "region_masks": [
            {
                "region": (region if isinstance(region, LogicalRegion) else LogicalRegion(region)).value,
                "policy": (
                    policy
                    if isinstance(policy, RegionMaskPolicy)
                    else RegionMaskPolicy(policy["kind"], policy.get("limit", 0))
                ).to_canonical_dict(),
            }
            for region, policy in sorted(
                masks.items(),
                key=lambda item: (
                    item[0].value if isinstance(item[0], LogicalRegion) else str(item[0])
                ),
            )
        ],
    }
    return canonical_sha256(payload)


@dataclass(frozen=True, slots=True)
class TickIdentity:
    """One tick bound to exactly one frozen canonical base field."""

    tick_sequence: int
    heartbeat_id: int
    base_field_id: str
    base_tick_id: int

    def __post_init__(self) -> None:
        for name in ("tick_sequence", "heartbeat_id", "base_tick_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"TickIdentity.{name} must be an integer")
        if self.tick_sequence < 1:
            raise ValueError("TickIdentity.tick_sequence must be >= 1")
        if self.heartbeat_id < 1:
            raise ValueError("TickIdentity.heartbeat_id must be >= 1")
        if self.base_tick_id < 0:
            raise ValueError("TickIdentity.base_tick_id must be non-negative")
        if not isinstance(self.base_field_id, str) or not self.base_field_id:
            raise ValueError("TickIdentity.base_field_id must be non-empty")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": TICK_IDENTITY_SCHEMA,
            "tick_sequence": self.tick_sequence,
            "heartbeat_id": self.heartbeat_id,
            "base_field_id": self.base_field_id,
            "base_tick_id": self.base_tick_id,
        }

    @property
    def tick_uid(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


class HeartbeatClock:
    """Issues strictly monotonic heartbeat ids and tick identities."""

    def __init__(self) -> None:
        self._heartbeat_id = 0
        self._tick_sequence = 0

    @property
    def heartbeat_id(self) -> int:
        return self._heartbeat_id

    @property
    def tick_sequence(self) -> int:
        return self._tick_sequence

    def beat(self) -> int:
        """Advance the heartbeat cadence and return the new heartbeat id."""

        self._heartbeat_id += 1
        return self._heartbeat_id

    def open_tick(self, base: SharedFieldSnapshot) -> TickIdentity:
        """Freeze ``base`` as the canonical base of the next tick."""

        if not isinstance(base, SharedFieldSnapshot):
            raise TypeError("HeartbeatClock.open_tick requires SharedFieldSnapshot")
        self._heartbeat_id += 1
        self._tick_sequence += 1
        return TickIdentity(
            tick_sequence=self._tick_sequence,
            heartbeat_id=self._heartbeat_id,
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
        )


@dataclass(frozen=True, slots=True)
class RailBinding:
    """A reference to one compiled rail bound to the frozen base field."""

    d_model: int
    rail_id: str
    source_field_id: str
    source_tick_id: int
    view_id: str = field(default_factory=derive_view_id)
    semantic_surface_id: str | None = None
    semantic_generation: str | None = None
    semantic_slot_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int):
            raise TypeError("RailBinding.d_model must be an integer")
        if self.d_model <= 0:
            raise ValueError("RailBinding.d_model must be positive")
        if isinstance(self.source_tick_id, bool) or not isinstance(
            self.source_tick_id, int
        ):
            raise TypeError("RailBinding.source_tick_id must be an integer")
        if self.source_tick_id < 0:
            raise ValueError("RailBinding.source_tick_id must be non-negative")
        if not isinstance(self.rail_id, str) or not self.rail_id:
            raise ValueError("RailBinding.rail_id must be non-empty")
        if not isinstance(self.source_field_id, str) or not self.source_field_id:
            raise ValueError("RailBinding.source_field_id must be non-empty")
        if not isinstance(self.view_id, str) or not self.view_id:
            raise ValueError("RailBinding.view_id must be non-empty")
        if isinstance(self.semantic_slot_count, bool) or not isinstance(self.semantic_slot_count, int):
            raise TypeError("RailBinding.semantic_slot_count must be an integer")
        if self.semantic_slot_count < 0:
            raise ValueError("RailBinding.semantic_slot_count must be non-negative")
        if self.semantic_surface_id is None:
            if self.semantic_generation is not None or self.semantic_slot_count != 0:
                raise ValueError(
                    "RailBinding semantic generation/count require semantic_surface_id"
                )
        else:
            if not isinstance(self.semantic_surface_id, str) or not self.semantic_surface_id:
                raise ValueError("RailBinding.semantic_surface_id must be None or non-empty")
            if not isinstance(self.semantic_generation, str) or not self.semantic_generation:
                raise ValueError(
                    "RailBinding.semantic_generation must be non-empty when semantic surface is bound"
                )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "rail_id": self.rail_id,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "view_id": self.view_id,
            "semantic_surface_id": self.semantic_surface_id,
            "semantic_generation": self.semantic_generation,
            "semantic_slot_count": self.semantic_slot_count,
        }


@dataclass(frozen=True, slots=True)
class FrozenTickImage:
    """Derived, immutable per-tick projection bound to one frozen base.

    The image carries rail *references* only.  It is never canonical state
    and holds no commit authority.
    """

    identity: TickIdentity
    rails: tuple[RailBinding, ...]
    view_id: str = field(default_factory=derive_view_id)
    image_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TickIdentity):
            raise TypeError("FrozenTickImage.identity must be a TickIdentity")
        rails = tuple(self.rails)
        if not rails:
            raise HeartbeatError("FrozenTickImage requires at least one rail")
        if not all(isinstance(rail, RailBinding) for rail in rails):
            raise TypeError("FrozenTickImage.rails must contain RailBinding values")
        if not isinstance(self.view_id, str) or not self.view_id:
            raise ValueError("FrozenTickImage.view_id must be non-empty")
        if any(rail.view_id != self.view_id for rail in rails):
            raise HeartbeatError("FrozenTickImage rail view ids must match the image view id")
        d_models = [rail.d_model for rail in rails]
        if len(d_models) != len(set(d_models)):
            raise HeartbeatError("FrozenTickImage rails must have unique d_model values")
        rails = tuple(sorted(rails, key=lambda rail: rail.d_model))
        for rail in rails:
            if (
                rail.source_field_id != self.identity.base_field_id
                or rail.source_tick_id != self.identity.base_tick_id
            ):
                raise StaleRailBindingError(
                    f"rail d_model={rail.d_model} is not bound to the frozen "
                    f"base field {self.identity.base_field_id!r}"
                )
        object.__setattr__(self, "rails", rails)
        object.__setattr__(
            self,
            "image_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": TICK_IMAGE_SCHEMA,
            "identity": self.identity.to_canonical_dict(),
            "view_id": self.view_id,
            "rails": [rail.to_canonical_dict() for rail in self.rails],
        }

    @property
    def base_field_id(self) -> str:
        return self.identity.base_field_id

    def rail_for(self, d_model: int) -> RailBinding | None:
        for rail in self.rails:
            if rail.d_model == d_model:
                return rail
        return None

    def require_rail(self, d_model: int) -> RailBinding:
        rail = self.rail_for(d_model)
        if rail is None:
            raise HeartbeatError(
                f"tick image has no rail binding for d_model {d_model}"
            )
        return rail

    @classmethod
    def from_compiled(
        cls,
        identity: TickIdentity,
        rails: Mapping[int, CompiledD64Field] | Iterable[tuple[int, CompiledD64Field]],
        *,
        view_id: str | None = None,
        semantic_surfaces: Mapping[int, D64SemanticSurface] | None = None,
    ) -> "FrozenTickImage":
        """Bind compiled D64 rail references to one frozen tick identity."""

        if not isinstance(identity, TickIdentity):
            raise TypeError("FrozenTickImage.from_compiled requires a TickIdentity")
        resolved_view_id = derive_view_id() if view_id is None else view_id
        if not isinstance(resolved_view_id, str) or not resolved_view_id:
            raise ValueError("FrozenTickImage.from_compiled view_id must be non-empty")
        items = (
            tuple(rails.items())
            if isinstance(rails, Mapping)
            else tuple(rails)
        )
        semantic_map = {} if semantic_surfaces is None else dict(semantic_surfaces)
        if any(isinstance(key, bool) or not isinstance(key, int) for key in semantic_map):
            raise TypeError("FrozenTickImage semantic surface keys must be integer d_model values")
        bindings: list[RailBinding] = []
        bound_d_models: set[int] = set()
        for d_model, compiled in items:
            if not isinstance(compiled, CompiledD64Field):
                raise TypeError(
                    "FrozenTickImage rails must be CompiledD64Field values"
                )
            # Build A is 64D-first: only the canonical D64 rail may be bound
            # until a real wider compiler is proven and ratified.
            if d_model != 64:
                raise RailWidthMismatchError(
                    f"Build A supports only the 64D rail; received d_model={d_model}"
                )
            # The compiled rail self-describes its physical width; a label
            # that disagrees with it is a fake rail and fails closed.  Today
            # CompiledD64Field rows are physically [N, 64], so any d_model
            # other than 64 is rejected until a real wider compiler exists.
            rail_width = int(compiled.rows.shape[1])
            if d_model != rail_width:
                raise RailWidthMismatchError(
                    f"rail label d_model={d_model} does not match the compiled "
                    f"rail's physical width {rail_width}"
                )
            if (
                compiled.source_field_id != identity.base_field_id
                or compiled.source_tick_id != identity.base_tick_id
            ):
                raise StaleRailBindingError(
                    f"compiled rail d_model={d_model} is stale for tick "
                    f"{identity.tick_sequence}: not bound to base field "
                    f"{identity.base_field_id!r}"
                )
            semantic = semantic_map.get(d_model)
            if semantic is not None:
                if not isinstance(semantic, D64SemanticSurface):
                    raise TypeError(
                        "FrozenTickImage semantic_surfaces must contain D64SemanticSurface values"
                    )
                if (
                    semantic.source_field_id != compiled.source_field_id
                    or semantic.source_tick_id != compiled.source_tick_id
                    or semantic.source_rail_id != compiled.rail_id
                ):
                    raise StaleRailBindingError(
                        f"semantic surface for d_model={d_model} is not bound to the supplied exact rail"
                    )
            bound_d_models.add(d_model)
            bindings.append(
                RailBinding(
                    d_model=d_model,
                    rail_id=compiled.rail_id,
                    source_field_id=compiled.source_field_id,
                    source_tick_id=compiled.source_tick_id,
                    view_id=resolved_view_id,
                    semantic_surface_id=(None if semantic is None else semantic.surface_id),
                    semantic_generation=(None if semantic is None else semantic.feature_generation),
                    semantic_slot_count=(0 if semantic is None else semantic.slot_count),
                )
            )
        unused_semantic = set(semantic_map) - bound_d_models
        if unused_semantic:
            raise HeartbeatError(
                "semantic surfaces were supplied for rails that are not present: "
                + ", ".join(str(item) for item in sorted(unused_semantic))
            )
        return cls(identity=identity, rails=tuple(bindings), view_id=resolved_view_id)


__all__ = [
    "DERIVED_VIEW_SCHEMA",
    "TICK_IDENTITY_SCHEMA",
    "TICK_IMAGE_SCHEMA",
    "FrozenTickImage",
    "HeartbeatClock",
    "RailBinding",
    "TickIdentity",
    "derive_view_id",
]

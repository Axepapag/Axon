"""Opaque, private, temperature-layered soul contracts.

The organism validates lineage and atomicity without interpreting a core's
private payload.  Payloads are exact bytes with no configured size ceiling;
their internal tensor/layout dialect belongs exclusively to the owning core.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from runtime.field import canonical_sha256

SOUL_LAYER_SCHEMA = "axon-private-soul-layer-v1"
SOUL_SNAPSHOT_SCHEMA = "axon-private-soul-snapshot-v1"
SOUL_PROMOTION_SCHEMA = "axon-private-soul-promotion-v1"
SOUL_TRANSITION_SCHEMA = "axon-private-soul-transition-v1"
SOUL_COMMIT_RECEIPT_SCHEMA = "axon-private-soul-commit-receipt-v1"


class SoulIntegrityError(RuntimeError):
    """Opaque soul state or lineage failed deterministic validation."""


class SoulTemperature(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DEEP_COLD = "deep_cold"


SOUL_TEMPERATURE_ORDER: tuple[SoulTemperature, ...] = (
    SoulTemperature.HOT,
    SoulTemperature.WARM,
    SoulTemperature.COLD,
    SoulTemperature.DEEP_COLD,
)

RUNTIME_SOUL_PHASES = frozenset({"first", "refined", "consolidated"})
SOUL_PHASES = frozenset({*RUNTIME_SOUL_PHASES, "outcome", "training", "merge"})
VETTED_OUTCOME_QUALITIES = frozenset({"success", "corrected", "endorsed"})


def _content_id(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA256 content identity")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty")
    return value


@dataclass(frozen=True, slots=True)
class SoulLayer:
    """One architecture-local opaque payload at one soul temperature."""

    temperature: SoulTemperature | str
    payload: bytes
    media_type: str = "application/x-axon-opaque-soul"
    tensor_layout: str = "core-private"
    payload_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        temperature = (
            self.temperature
            if isinstance(self.temperature, SoulTemperature)
            else SoulTemperature(self.temperature)
        )
        if not isinstance(self.payload, bytes):
            raise TypeError("SoulLayer.payload must be exact bytes")
        _nonempty(self.media_type, "SoulLayer.media_type")
        _nonempty(self.tensor_layout, "SoulLayer.tensor_layout")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "payload_sha256", hashlib.sha256(self.payload).hexdigest())

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": SOUL_LAYER_SCHEMA,
            "temperature": self.temperature.value,
            "media_type": self.media_type,
            "tensor_layout": self.tensor_layout,
            "payload_bytes": len(self.payload),
            "payload_sha256": self.payload_sha256,
            "payload_base64": base64.b64encode(self.payload).decode("ascii"),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SoulLayer":
        item = dict(value)
        required = {
            "schema",
            "temperature",
            "media_type",
            "tensor_layout",
            "payload_bytes",
            "payload_sha256",
            "payload_base64",
        }
        if set(item) != required or item["schema"] != SOUL_LAYER_SCHEMA:
            raise SoulIntegrityError("serialized soul layer fields are invalid")
        try:
            payload = base64.b64decode(item["payload_base64"], validate=True)
            layer = cls(
                temperature=item["temperature"],
                payload=payload,
                media_type=item["media_type"],
                tensor_layout=item["tensor_layout"],
            )
        except (TypeError, ValueError) as exc:
            raise SoulIntegrityError("serialized soul layer is invalid") from exc
        if item["payload_bytes"] != len(payload) or item["payload_sha256"] != layer.payload_sha256:
            raise SoulIntegrityError("soul layer payload identity mismatch")
        return layer


def empty_soul_layers() -> tuple[SoulLayer, ...]:
    return tuple(SoulLayer(temperature, b"") for temperature in SOUL_TEMPERATURE_ORDER)


@dataclass(frozen=True, slots=True)
class SoulSnapshot:
    """One immutable private soul generation for one core."""

    core_id: str
    architecture_id: str
    parameter_generation: str
    generation: int
    parent_soul_id: str | None
    layers: tuple[SoulLayer, ...] = field(default_factory=empty_soul_layers)
    soul_id: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.core_id, "SoulSnapshot.core_id")
        _nonempty(self.architecture_id, "SoulSnapshot.architecture_id")
        _nonempty(self.parameter_generation, "SoulSnapshot.parameter_generation")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("SoulSnapshot.generation must be non-negative")
        if self.parent_soul_id is not None:
            _content_id(self.parent_soul_id, "SoulSnapshot.parent_soul_id")
        layers = tuple(self.layers)
        if not all(isinstance(layer, SoulLayer) for layer in layers):
            raise TypeError("SoulSnapshot.layers must contain SoulLayer values")
        if tuple(layer.temperature for layer in layers) != SOUL_TEMPERATURE_ORDER:
            raise ValueError("SoulSnapshot must contain hot, warm, cold, and deep_cold exactly once in order")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "soul_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def layer(self, temperature: SoulTemperature | str) -> SoulLayer:
        target = temperature if isinstance(temperature, SoulTemperature) else SoulTemperature(temperature)
        return self.layers[SOUL_TEMPERATURE_ORDER.index(target)]

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SOUL_SNAPSHOT_SCHEMA,
            "core_id": self.core_id,
            "architecture_id": self.architecture_id,
            "parameter_generation": self.parameter_generation,
            "generation": self.generation,
            "parent_soul_id": self.parent_soul_id,
            "layers": [layer.to_canonical_dict() for layer in self.layers],
        }
        if include_id:
            value["soul_id"] = self.soul_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SoulSnapshot":
        item = dict(value)
        required = {
            "schema",
            "core_id",
            "architecture_id",
            "parameter_generation",
            "generation",
            "parent_soul_id",
            "layers",
            "soul_id",
        }
        if set(item) != required or item["schema"] != SOUL_SNAPSHOT_SCHEMA:
            raise SoulIntegrityError("serialized soul snapshot fields are invalid")
        try:
            snapshot = cls(
                core_id=item["core_id"],
                architecture_id=item["architecture_id"],
                parameter_generation=item["parameter_generation"],
                generation=item["generation"],
                parent_soul_id=item["parent_soul_id"],
                layers=tuple(SoulLayer.from_mapping(layer) for layer in item["layers"]),
            )
        except (TypeError, ValueError) as exc:
            raise SoulIntegrityError("serialized soul snapshot is invalid") from exc
        if snapshot.soul_id != item["soul_id"]:
            raise SoulIntegrityError("soul snapshot identity mismatch")
        return snapshot


@dataclass(frozen=True, slots=True)
class SoulPromotion:
    """Evidence that selected experience may move one temperature colder."""

    source: SoulTemperature | str
    target: SoulTemperature | str
    source_payload_sha256: str
    evidence_ids: tuple[str, ...] = ()
    outcome_quality: str = "observed"
    repeated_observations: int = 1
    validation_ids: tuple[str, ...] = ()
    promotion_id: str = field(init=False)

    def __post_init__(self) -> None:
        source = self.source if isinstance(self.source, SoulTemperature) else SoulTemperature(self.source)
        target = self.target if isinstance(self.target, SoulTemperature) else SoulTemperature(self.target)
        if SOUL_TEMPERATURE_ORDER.index(target) != SOUL_TEMPERATURE_ORDER.index(source) + 1:
            raise ValueError("soul promotion must move exactly one temperature colder")
        _content_id(self.source_payload_sha256, "SoulPromotion.source_payload_sha256")
        if isinstance(self.repeated_observations, bool) or not isinstance(self.repeated_observations, int):
            raise TypeError("SoulPromotion.repeated_observations must be an integer")
        if self.repeated_observations < 1:
            raise ValueError("SoulPromotion.repeated_observations must be positive")
        evidence = tuple(sorted(set(map(str, self.evidence_ids))))
        validations = tuple(sorted(set(map(str, self.validation_ids))))
        if target in {SoulTemperature.COLD, SoulTemperature.DEEP_COLD}:
            if self.outcome_quality not in VETTED_OUTCOME_QUALITIES or not evidence:
                raise ValueError("cold promotion requires explicit successful outcome evidence")
            if self.repeated_observations < 2:
                raise ValueError("cold promotion requires repeated observations")
        if target is SoulTemperature.DEEP_COLD and not validations:
            raise ValueError("deep-cold promotion requires held-out/replay validation ids")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "validation_ids", validations)
        object.__setattr__(self, "promotion_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SOUL_PROMOTION_SCHEMA,
            "source": self.source.value,
            "target": self.target.value,
            "source_payload_sha256": self.source_payload_sha256,
            "evidence_ids": list(self.evidence_ids),
            "outcome_quality": self.outcome_quality,
            "repeated_observations": self.repeated_observations,
            "validation_ids": list(self.validation_ids),
        }
        if include_id:
            value["promotion_id"] = self.promotion_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SoulPromotion":
        item = dict(value)
        if item.pop("schema", None) != SOUL_PROMOTION_SCHEMA:
            raise SoulIntegrityError("unsupported soul promotion schema")
        promotion_id = item.pop("promotion_id", None)
        try:
            promotion = cls(
                source=item["source"],
                target=item["target"],
                source_payload_sha256=item["source_payload_sha256"],
                evidence_ids=tuple(item["evidence_ids"]),
                outcome_quality=item["outcome_quality"],
                repeated_observations=item["repeated_observations"],
                validation_ids=tuple(item["validation_ids"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SoulIntegrityError("serialized soul promotion is invalid") from exc
        if promotion.promotion_id != promotion_id:
            raise SoulIntegrityError("soul promotion identity mismatch")
        return promotion


@dataclass(frozen=True, slots=True)
class SoulTransition:
    """One core-authored successor proposal bound to an exact inhale."""

    core_id: str
    architecture_id: str
    parameter_generation: str
    before_soul_id: str
    before_generation: int
    tick_uid: str
    request_id: str
    phase: str
    updates: tuple[SoulLayer, ...]
    promotions: tuple[SoulPromotion, ...] = ()
    transition_id: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.core_id, "SoulTransition.core_id")
        _nonempty(self.architecture_id, "SoulTransition.architecture_id")
        _nonempty(self.parameter_generation, "SoulTransition.parameter_generation")
        _content_id(self.before_soul_id, "SoulTransition.before_soul_id")
        if isinstance(self.before_generation, bool) or not isinstance(self.before_generation, int):
            raise TypeError("SoulTransition.before_generation must be an integer")
        if self.before_generation < 0:
            raise ValueError("SoulTransition.before_generation must be non-negative")
        _nonempty(self.tick_uid, "SoulTransition.tick_uid")
        _nonempty(self.request_id, "SoulTransition.request_id")
        if self.phase not in SOUL_PHASES:
            raise ValueError(f"unsupported soul transition phase {self.phase!r}")
        updates = tuple(self.updates)
        if not updates or not all(isinstance(layer, SoulLayer) for layer in updates):
            raise ValueError("SoulTransition requires one or more SoulLayer updates")
        temperatures = tuple(layer.temperature for layer in updates)
        if len(set(temperatures)) != len(temperatures):
            raise ValueError("SoulTransition updates a temperature more than once")
        if self.phase in RUNTIME_SOUL_PHASES and SoulTemperature.HOT not in temperatures:
            raise ValueError("every runtime inhale/exhale must return a hot-layer successor")
        promotions = tuple(self.promotions)
        if not all(isinstance(item, SoulPromotion) for item in promotions):
            raise TypeError("SoulTransition.promotions must contain SoulPromotion values")
        promoted_targets = {item.target for item in promotions}
        non_hot_updates = set(temperatures) - {SoulTemperature.HOT}
        if not non_hot_updates.issubset(promoted_targets):
            raise ValueError("warm/cold/deep-cold updates require matching promotion receipts")
        object.__setattr__(self, "updates", tuple(sorted(updates, key=lambda layer: SOUL_TEMPERATURE_ORDER.index(layer.temperature))))
        object.__setattr__(self, "promotions", tuple(sorted(promotions, key=lambda item: item.promotion_id)))
        object.__setattr__(self, "transition_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SOUL_TRANSITION_SCHEMA,
            "core_id": self.core_id,
            "architecture_id": self.architecture_id,
            "parameter_generation": self.parameter_generation,
            "before_soul_id": self.before_soul_id,
            "before_generation": self.before_generation,
            "tick_uid": self.tick_uid,
            "request_id": self.request_id,
            "phase": self.phase,
            "updates": [layer.to_canonical_dict() for layer in self.updates],
            "promotions": [item.to_canonical_dict() for item in self.promotions],
        }
        if include_id:
            value["transition_id"] = self.transition_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SoulTransition":
        item = dict(value)
        if item.pop("schema", None) != SOUL_TRANSITION_SCHEMA:
            raise SoulIntegrityError("unsupported soul transition schema")
        transition_id = item.pop("transition_id", None)
        try:
            transition = cls(
                core_id=item["core_id"],
                architecture_id=item["architecture_id"],
                parameter_generation=item["parameter_generation"],
                before_soul_id=item["before_soul_id"],
                before_generation=item["before_generation"],
                tick_uid=item["tick_uid"],
                request_id=item["request_id"],
                phase=item["phase"],
                updates=tuple(SoulLayer.from_mapping(layer) for layer in item["updates"]),
                promotions=tuple(
                    SoulPromotion.from_mapping(promotion) for promotion in item["promotions"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SoulIntegrityError("serialized soul transition is invalid") from exc
        if transition.transition_id != transition_id:
            raise SoulIntegrityError("soul transition identity mismatch")
        return transition


def apply_soul_transition(before: SoulSnapshot, transition: SoulTransition) -> SoulSnapshot:
    """Deterministically materialize a proposed private-soul successor."""

    if not isinstance(before, SoulSnapshot) or not isinstance(transition, SoulTransition):
        raise TypeError("apply_soul_transition requires SoulSnapshot and SoulTransition")
    if transition.before_soul_id != before.soul_id or transition.before_generation != before.generation:
        raise SoulIntegrityError("soul transition is stale for its inhale snapshot")
    if (
        transition.core_id != before.core_id
        or transition.architecture_id != before.architecture_id
        or transition.parameter_generation != before.parameter_generation
    ):
        raise SoulIntegrityError("soul transition core/architecture/parameter binding mismatch")
    layers = {layer.temperature: layer for layer in before.layers}
    for promotion in transition.promotions:
        if layers[promotion.source].payload_sha256 != promotion.source_payload_sha256:
            raise SoulIntegrityError("soul promotion source payload is stale")
    for update in transition.updates:
        layers[update.temperature] = update
    return SoulSnapshot(
        core_id=before.core_id,
        architecture_id=before.architecture_id,
        parameter_generation=before.parameter_generation,
        generation=before.generation + 1,
        parent_soul_id=before.soul_id,
        layers=tuple(layers[temperature] for temperature in SOUL_TEMPERATURE_ORDER),
    )


@dataclass(frozen=True, slots=True)
class SoulCommitReceipt:
    core_id: str
    branch_id: str
    transition_id: str
    before_soul_id: str
    after_soul_id: str
    generation: int
    phase: str
    tick_uid: str
    request_id: str
    commit_binding: str
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("core_id", "branch_id", "phase", "tick_uid", "request_id", "commit_binding"):
            _nonempty(getattr(self, name), f"SoulCommitReceipt.{name}")
        for name in ("transition_id", "before_soul_id", "after_soul_id"):
            _content_id(getattr(self, name), f"SoulCommitReceipt.{name}")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 1:
            raise ValueError("SoulCommitReceipt.generation must be positive")
        object.__setattr__(self, "receipt_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SOUL_COMMIT_RECEIPT_SCHEMA,
            "core_id": self.core_id,
            "branch_id": self.branch_id,
            "transition_id": self.transition_id,
            "before_soul_id": self.before_soul_id,
            "after_soul_id": self.after_soul_id,
            "generation": self.generation,
            "phase": self.phase,
            "tick_uid": self.tick_uid,
            "request_id": self.request_id,
            "commit_binding": self.commit_binding,
        }
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SoulCommitReceipt":
        item = dict(value)
        if item.pop("schema", None) != SOUL_COMMIT_RECEIPT_SCHEMA:
            raise SoulIntegrityError("unsupported soul commit receipt schema")
        receipt_id = item.pop("receipt_id", None)
        try:
            receipt = cls(**item)
        except (TypeError, ValueError) as exc:
            raise SoulIntegrityError("serialized soul commit receipt is invalid") from exc
        if receipt.receipt_id != receipt_id:
            raise SoulIntegrityError("soul commit receipt identity mismatch")
        return receipt


__all__ = [
    "RUNTIME_SOUL_PHASES",
    "SOUL_COMMIT_RECEIPT_SCHEMA",
    "SOUL_LAYER_SCHEMA",
    "SOUL_PHASES",
    "SOUL_PROMOTION_SCHEMA",
    "SOUL_SNAPSHOT_SCHEMA",
    "SOUL_TEMPERATURE_ORDER",
    "SOUL_TRANSITION_SCHEMA",
    "SoulCommitReceipt",
    "SoulIntegrityError",
    "SoulLayer",
    "SoulPromotion",
    "SoulSnapshot",
    "SoulTemperature",
    "SoulTransition",
    "apply_soul_transition",
    "empty_soul_layers",
]

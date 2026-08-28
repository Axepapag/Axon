"""Sovereign valve plane for the Heart runtime (H6).

A valve is a typed, fail-closed admission gate between an organ/ingress path and
the heart's canonical circulation.  CLOSED is the default.  CAPPED valves admit
only bounded, well-shaped traffic whose source class, envelope type, and payload
size match the valve definition.  The heart constructs authority internally;
valve envelopes identify source and provenance, they never carry an
``AuthorityGrant``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from runtime.field import LogicalRegion, canonical_json_bytes

from .authority import AuthorityClass, AuthorityGrant, IngressChannel
from .errors import (
    UnknownValveError,
    ValveAuthorityError,
    ValveSourceMismatchError,
)


class ValveState(str, Enum):
    """Admission posture of one heart valve."""

    CLOSED = "closed"
    CAPPED = "capped"
    OPEN = "open"


@dataclass(frozen=True, slots=True)
class ValveBudget:
    """Per-valve and per-beat intake budget."""

    pending_cap: int = 0
    items_per_beat: int = 0
    chars_per_beat: int = 0
    max_item_chars: int = 0
    max_item_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "pending_cap",
            "items_per_beat",
            "chars_per_beat",
            "max_item_chars",
            "max_item_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ValveBudget.{name} must be an integer")
            if value < 0:
                raise ValueError(f"ValveBudget.{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class HeartValveDefinition:
    """Immutable definition of one heart valve slot."""

    valve_id: str
    version: int
    state: ValveState
    source_class: str
    authority_class: AuthorityClass
    governed_regions: frozenset[LogicalRegion]
    envelope_type: str
    budget: ValveBudget
    rejection_policy: str

    def __post_init__(self) -> None:
        if not isinstance(self.valve_id, str) or not self.valve_id:
            raise ValueError("HeartValveDefinition.valve_id must be a non-empty string")

        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("HeartValveDefinition.version must be an integer")
        if self.version < 1:
            raise ValueError("HeartValveDefinition.version must be >= 1")

        state = self.state
        if not isinstance(state, ValveState):
            try:
                state = ValveState(state)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"HeartValveDefinition.state must be a ValveState, got {self.state!r}"
                ) from exc
        object.__setattr__(self, "state", state)

        if not isinstance(self.source_class, str) or not self.source_class:
            raise ValueError(
                "HeartValveDefinition.source_class must be a non-empty string"
            )

        authority_class = self.authority_class
        if not isinstance(authority_class, AuthorityClass):
            try:
                authority_class = AuthorityClass(authority_class)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"HeartValveDefinition.authority_class must be an AuthorityClass, "
                    f"got {self.authority_class!r}"
                ) from exc
        object.__setattr__(self, "authority_class", authority_class)

        regions: set[LogicalRegion] = set()
        for item in self.governed_regions:
            try:
                regions.add(item if isinstance(item, LogicalRegion) else LogicalRegion(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"HeartValveDefinition.governed_regions contains unknown region {item!r}"
                ) from exc
        regions = frozenset(regions)
        if not regions:
            raise ValueError("HeartValveDefinition.governed_regions must be non-empty")
        object.__setattr__(self, "governed_regions", regions)

        if not isinstance(self.envelope_type, str) or not self.envelope_type:
            raise ValueError(
                "HeartValveDefinition.envelope_type must be a non-empty string"
            )

        if not isinstance(self.budget, ValveBudget):
            raise TypeError("HeartValveDefinition.budget must be a ValveBudget")

        policy = self.rejection_policy
        if policy not in ("reject", "quarantine"):
            raise ValueError(
                f"HeartValveDefinition.rejection_policy must be 'reject' or 'quarantine', "
                f"got {policy!r}"
            )

    def to_authority_grant(self) -> AuthorityGrant:
        """Construct the Heart-derived authority grant for this valve."""

        if self.authority_class is AuthorityClass.EXTERNAL_INGRESS:
            channel = {
                "external_user": IngressChannel.USER,
                "external_tool": IngressChannel.TOOL,
                "external_advisor": IngressChannel.ADVISOR,
            }.get(self.source_class)
            if channel is None:
                raise ValveAuthorityError(
                    f"valve {self.valve_id!r} claims external ingress but source class "
                    f"{self.source_class!r} has no ingress channel"
                )
            return AuthorityGrant.ingress(channel)
        if self.authority_class is AuthorityClass.DORMANT_VALVE:
            return AuthorityGrant.dormant_valve()
        if self.authority_class is AuthorityClass.CONSOLIDATOR:
            return AuthorityGrant.consolidator()
        if self.authority_class is AuthorityClass.CORE:
            return AuthorityGrant.core(self.governed_regions)
        raise ValveAuthorityError(
            f"cannot construct authority grant for valve {self.valve_id!r}"
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "valve_id": self.valve_id,
            "version": self.version,
            "state": self.state.value,
            "source_class": self.source_class,
            "authority_class": self.authority_class.value,
            "governed_regions": sorted(region.value for region in self.governed_regions),
            "envelope_type": self.envelope_type,
            "budget": {
                "pending_cap": self.budget.pending_cap,
                "items_per_beat": self.budget.items_per_beat,
                "chars_per_beat": self.budget.chars_per_beat,
                "max_item_chars": self.budget.max_item_chars,
                "max_item_bytes": self.budget.max_item_bytes,
            },
            "rejection_policy": self.rejection_policy,
        }


@dataclass(frozen=True, slots=True)
class ValveEnvelope:
    """One payload arriving at a heart valve.

    Envelopes carry source identity and provenance only.  They deliberately do
    not carry an ``AuthorityGrant``; the registry resolves valve -> authority.
    """

    valve_id: str
    source_id: str
    payload: str
    provenance: str
    envelope_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.valve_id, str) or not self.valve_id:
            raise ValueError("ValveEnvelope.valve_id must be a non-empty string")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("ValveEnvelope.source_id must be a non-empty string")
        if not isinstance(self.payload, str):
            raise TypeError("ValveEnvelope.payload must be a string")
        if not isinstance(self.provenance, str):
            raise TypeError("ValveEnvelope.provenance must be a string")
        if not isinstance(self.envelope_type, str) or not self.envelope_type:
            raise ValueError(
                "ValveEnvelope.envelope_type must be a non-empty string"
            )
        if hasattr(self, "grant") or hasattr(self, "authority_grant"):
            raise ValueError(
                "ValveEnvelope must not carry an AuthorityGrant attribute"
            )

    def to_spool_dict(self) -> dict[str, Any]:
        return {
            "valve_id": self.valve_id,
            "source_id": self.source_id,
            "payload": self.payload,
            "provenance": self.provenance,
            "envelope_type": self.envelope_type,
        }


@dataclass(frozen=True, slots=True)
class ValveReceipt:
    """Immutable receipt issued when a valve admits one item."""

    valve_id: str
    valve_version: int
    source_id: str
    item_id: str
    authority_class: AuthorityClass
    governed_regions: frozenset[LogicalRegion]
    enqueued_at: datetime


@dataclass(frozen=True, slots=True)
class ValveDecision:
    """Admission outcome from the valve plane."""

    admitted: bool
    reason: str
    quarantine: bool
    receipt: ValveReceipt | None


class ValveBudgetTracker:
    """Mutable per-valve counters for pending and per-beat budgets."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}

    def ensure_valve(self, valve_id: str) -> None:
        if valve_id not in self._counts:
            self._counts[valve_id] = {
                "pending": 0,
                "items_this_beat": 0,
                "chars_this_beat": 0,
                "counter": 0,
            }

    def pending(self, valve_id: str) -> int:
        self.ensure_valve(valve_id)
        return self._counts[valve_id]["pending"]

    def items_this_beat(self, valve_id: str) -> int:
        self.ensure_valve(valve_id)
        return self._counts[valve_id]["items_this_beat"]

    def chars_this_beat(self, valve_id: str) -> int:
        self.ensure_valve(valve_id)
        return self._counts[valve_id]["chars_this_beat"]

    def next_counter(self, valve_id: str) -> int:
        self.ensure_valve(valve_id)
        self._counts[valve_id]["counter"] += 1
        return self._counts[valve_id]["counter"]

    def admit(self, valve_id: str, char_count: int) -> None:
        self.ensure_valve(valve_id)
        self._counts[valve_id]["pending"] += 1
        self._counts[valve_id]["items_this_beat"] += 1
        self._counts[valve_id]["chars_this_beat"] += char_count

    def complete(self, valve_id: str, count: int = 1) -> None:
        """Release pending budget after canonical persistence/quarantine."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        self.ensure_valve(valve_id)
        self._counts[valve_id]["pending"] = max(
            0, self._counts[valve_id]["pending"] - count
        )

    def reset_beat(self, valve_id: str | None = None) -> None:
        if valve_id is None:
            for counts in self._counts.values():
                counts["items_this_beat"] = 0
                counts["chars_this_beat"] = 0
        else:
            self.ensure_valve(valve_id)
            self._counts[valve_id]["items_this_beat"] = 0
            self._counts[valve_id]["chars_this_beat"] = 0


# The four primitive real valves that may currently mutate canonical state.
_PRIMITIVE_REAL_VALVE_IDS: frozenset[str] = frozenset(
    {
        "user_ingress",
        "tool_ingress",
        "advisor_ingress",
        "dormant_recall",
    }
)


# Default primitive production budget.  Caps are intentionally modest for v1.
_DEFAULT_PRIMITIVE_BUDGET = ValveBudget(
    pending_cap=1000,
    items_per_beat=100,
    chars_per_beat=10_000,
    max_item_chars=4096,
    max_item_bytes=16_384,
)


def _coerce_regions(
    value: Iterable[LogicalRegion | str],
) -> frozenset[LogicalRegion]:
    regions: set[LogicalRegion] = set()
    for item in value:
        regions.add(item if isinstance(item, LogicalRegion) else LogicalRegion(item))
    return frozenset(regions)


def _make_item_id(envelope: ValveEnvelope, version: int, counter: int) -> str:
    """Deterministic UUID-like item id from envelope contents and monotonic counter."""

    payload = canonical_json_bytes(
        {
            "valve_id": envelope.valve_id,
            "source_id": envelope.source_id,
            "payload": envelope.payload,
            "provenance": envelope.provenance,
            "envelope_type": envelope.envelope_type,
            "version": version,
            "counter": counter,
        }
    )
    return hashlib.sha256(payload).hexdigest()[:32]


class HeartValveRegistry:
    """Pre-loaded registry of heart valve slots."""

    def __init__(self, definitions: Iterable[HeartValveDefinition] | None = None) -> None:
        self._valves: dict[str, HeartValveDefinition] = {}
        self._tracker = ValveBudgetTracker()
        for definition in definitions or ():
            self._register(definition)

    def _register(self, definition: HeartValveDefinition) -> None:
        if not isinstance(definition, HeartValveDefinition):
            raise TypeError("HeartValveRegistry requires HeartValveDefinition values")
        if definition.valve_id in self._valves:
            raise ValueError(f"duplicate valve_id {definition.valve_id!r}")
        self._valves[definition.valve_id] = definition
        self._tracker.ensure_valve(definition.valve_id)

    def __contains__(self, valve_id: str) -> bool:
        return valve_id in self._valves

    def __iter__(self):
        return iter(self._valves[valve_id] for valve_id in sorted(self._valves))

    def get(self, valve_id: str) -> HeartValveDefinition:
        if valve_id not in self._valves:
            raise UnknownValveError(f"unknown valve_id {valve_id!r}")
        return self._valves[valve_id]

    def resolve_grant(self, valve_id: str, source_id: str) -> AuthorityGrant:
        """Heart-constructed authority from valve identity and source evidence."""

        definition = self.get(valve_id)
        if source_id != definition.source_class:
            raise ValveSourceMismatchError(
                f"valve {valve_id!r} expects source class {definition.source_class!r}; "
                f"received {source_id!r}"
            )
        return definition.to_authority_grant()

    def is_open_for_mutation(self, valve_id: str) -> bool:
        """Return True only for CAPPED/OPEN primitive real valves."""

        if valve_id not in self._valves:
            return False
        definition = self._valves[valve_id]
        return (
            definition.state in (ValveState.CAPPED, ValveState.OPEN)
            and valve_id in _PRIMITIVE_REAL_VALVE_IDS
        )

    def local_validate(
        self,
        envelope: ValveEnvelope,
        *,
        pending_count: int = 0,
    ) -> ValveDecision:
        """Valve-local admission check without consuming per-beat budget."""

        if not isinstance(envelope, ValveEnvelope):
            raise TypeError("HeartValveRegistry.local_validate requires a ValveEnvelope")
        if isinstance(pending_count, bool) or not isinstance(pending_count, int) or pending_count < 0:
            raise ValueError("pending_count must be a non-negative integer")
        if envelope.valve_id not in self._valves:
            return ValveDecision(False, f"unknown valve_id {envelope.valve_id!r}", False, None)
        definition = self._valves[envelope.valve_id]
        if definition.state is ValveState.CLOSED:
            return ValveDecision(False, "valve is closed", False, None)
        if envelope.envelope_type != definition.envelope_type:
            return ValveDecision(False, "envelope type mismatch", False, None)
        if envelope.source_id != definition.source_class:
            return ValveDecision(False, "source class mismatch", False, None)
        char_count = len(envelope.payload)
        byte_count = len(envelope.payload.encode("utf-8"))
        if char_count > definition.budget.max_item_chars:
            return ValveDecision(
                False,
                f"payload exceeds max_item_chars {definition.budget.max_item_chars}",
                definition.rejection_policy == "quarantine",
                None,
            )
        if byte_count > definition.budget.max_item_bytes:
            return ValveDecision(
                False,
                f"payload exceeds max_item_bytes {definition.budget.max_item_bytes}",
                definition.rejection_policy == "quarantine",
                None,
            )
        if pending_count >= definition.budget.pending_cap:
            return ValveDecision(False, "valve pending cap exceeded", False, None)
        return ValveDecision(True, "admitted_local", False, None)

    @property
    def tracker(self) -> ValveBudgetTracker:
        return self._tracker

    def decide(
        self,
        envelope: ValveEnvelope,
        *,
        beats_remaining_budget: dict[str, Any] | None = None,
        global_budget: ValveBudget | None = None,
    ) -> ValveDecision:
        """Admit or reject one envelope against its valve definition and budgets."""

        if not isinstance(envelope, ValveEnvelope):
            raise TypeError("HeartValveRegistry.decide requires a ValveEnvelope")

        if envelope.valve_id not in self._valves:
            return ValveDecision(
                admitted=False,
                reason=f"unknown valve_id {envelope.valve_id!r}",
                quarantine=False,
                receipt=None,
            )

        definition = self._valves[envelope.valve_id]

        if definition.state is ValveState.CLOSED:
            return ValveDecision(
                admitted=False,
                reason="valve is closed",
                quarantine=False,
                receipt=None,
            )

        if envelope.envelope_type != definition.envelope_type:
            return ValveDecision(
                admitted=False,
                reason=(
                    f"envelope type {envelope.envelope_type!r} does not match "
                    f"valve envelope type {definition.envelope_type!r}"
                ),
                quarantine=False,
                receipt=None,
            )

        if envelope.source_id != definition.source_class:
            return ValveDecision(
                admitted=False,
                reason=(
                    f"source class {envelope.source_id!r} does not match "
                    f"valve source class {definition.source_class!r}"
                ),
                quarantine=False,
                receipt=None,
            )

        char_count = len(envelope.payload)
        byte_count = len(envelope.payload.encode("utf-8"))

        if char_count > definition.budget.max_item_chars:
            reason = (
                f"payload size {char_count} chars exceeds valve max_item_chars "
                f"{definition.budget.max_item_chars}"
            )
            return ValveDecision(
                admitted=False,
                reason=reason,
                quarantine=definition.rejection_policy == "quarantine",
                receipt=None,
            )

        if byte_count > definition.budget.max_item_bytes:
            reason = (
                f"payload size {byte_count} bytes exceeds valve max_item_bytes "
                f"{definition.budget.max_item_bytes}"
            )
            return ValveDecision(
                admitted=False,
                reason=reason,
                quarantine=definition.rejection_policy == "quarantine",
                receipt=None,
            )

        # Per-valve budget checks against the internal tracker.
        budget = definition.budget
        if self._tracker.pending(definition.valve_id) >= budget.pending_cap:
            return ValveDecision(
                admitted=False,
                reason="valve pending cap exceeded",
                quarantine=False,
                receipt=None,
            )
        if budget.items_per_beat == 0:
            return ValveDecision(
                admitted=False,
                reason="valve items_per_beat is zero",
                quarantine=False,
                receipt=None,
            )
        if self._tracker.items_this_beat(definition.valve_id) >= budget.items_per_beat:
            return ValveDecision(
                admitted=False,
                reason="valve items-per-beat budget exceeded",
                quarantine=False,
                receipt=None,
            )
        if budget.chars_per_beat == 0:
            return ValveDecision(
                admitted=False,
                reason="valve chars_per_beat is zero",
                quarantine=False,
                receipt=None,
            )
        if (
            self._tracker.chars_this_beat(definition.valve_id) + char_count
            > budget.chars_per_beat
        ):
            return ValveDecision(
                admitted=False,
                reason="valve chars-per-beat budget exceeded",
                quarantine=False,
                receipt=None,
            )

        # Optional caller-supplied per-valve remaining-beat budget override.
        if beats_remaining_budget:
            remaining = beats_remaining_budget.get(definition.valve_id, {})
            items_remaining = remaining.get("items")
            chars_remaining = remaining.get("chars")
            if items_remaining is not None and items_remaining <= 0:
                return ValveDecision(
                    admitted=False,
                    reason="caller-supplied beat item budget exhausted",
                    quarantine=False,
                    receipt=None,
                )
            if (
                chars_remaining is not None
                and char_count > chars_remaining
            ):
                return ValveDecision(
                    admitted=False,
                    reason="caller-supplied beat char budget exhausted",
                    quarantine=False,
                    receipt=None,
                )

        # Optional global cardiac intake budget across all valves.
        if global_budget is not None:
            global_items = sum(
                self._tracker.items_this_beat(vid) for vid in self._valves
            )
            global_chars = sum(
                self._tracker.chars_this_beat(vid) for vid in self._valves
            )
            if global_budget.items_per_beat > 0 and global_items >= global_budget.items_per_beat:
                return ValveDecision(
                    admitted=False,
                    reason="global items-per-beat budget exceeded",
                    quarantine=False,
                    receipt=None,
                )
            if (
                global_budget.chars_per_beat > 0
                and global_chars + char_count > global_budget.chars_per_beat
            ):
                return ValveDecision(
                    admitted=False,
                    reason="global chars-per-beat budget exceeded",
                    quarantine=False,
                    receipt=None,
                )

        # Admit.
        counter = self._tracker.next_counter(definition.valve_id)
        item_id = _make_item_id(envelope, definition.version, counter)
        self._tracker.admit(definition.valve_id, char_count)
        receipt = ValveReceipt(
            valve_id=definition.valve_id,
            valve_version=definition.version,
            source_id=envelope.source_id,
            item_id=item_id,
            authority_class=definition.authority_class,
            governed_regions=definition.governed_regions,
            enqueued_at=datetime.now(timezone.utc),
        )
        return ValveDecision(
            admitted=True,
            reason="admitted",
            quarantine=False,
            receipt=receipt,
        )


def primitive_valve_registry() -> HeartValveRegistry:
    """Return the standard 20-slot sovereign valve plane.

    Only the four primitive real valves are CAPPED; every other slot is CLOSED
    until its organ actually exists.
    """

    definitions: list[HeartValveDefinition] = []

    # 01-04: primitive real valves, CAPPED.
    primitive_specs: tuple[tuple[str, str, AuthorityClass, LogicalRegion], ...] = (
        ("user_ingress", "external_user", AuthorityClass.EXTERNAL_INGRESS, LogicalRegion.USER_INPUT),
        ("tool_ingress", "external_tool", AuthorityClass.EXTERNAL_INGRESS, LogicalRegion.TOOL_RESULTS),
        ("advisor_ingress", "external_advisor", AuthorityClass.EXTERNAL_INGRESS, LogicalRegion.ADVISOR_INPUT),
        ("dormant_recall", "dormant_valve", AuthorityClass.DORMANT_VALVE, LogicalRegion.CORTEX),
    )
    for valve_id, source_class, authority_class, region in primitive_specs:
        definitions.append(
            HeartValveDefinition(
                valve_id=valve_id,
                version=1,
                state=ValveState.CAPPED,
                source_class=source_class,
                authority_class=authority_class,
                governed_regions=frozenset({region}),
                envelope_type="text/plain",
                budget=_DEFAULT_PRIMITIVE_BUDGET,
                rejection_policy="quarantine",
            )
        )

    # 05-20: reserved future slots, CLOSED.
    closed_slot_ids: tuple[str, ...] = (
        "semantic_cortex",
        "core_initial_proposal",
        "core_refinement",
        "consolidator",
        "vision",
        "hearing",
        "speech_feedback",
        "episodic_memory",
        "engineer_memory_service",
        "self_model",
        "planner",
        "actuation_feedback",
        "training_promotion",
        "external_sensor",
        "future_organ_a",
        "future_organ_b",
    )
    closed_budget = ValveBudget(
        pending_cap=0,
        items_per_beat=0,
        chars_per_beat=0,
        max_item_chars=0,
        max_item_bytes=0,
    )
    for valve_id in closed_slot_ids:
        definitions.append(
            HeartValveDefinition(
                valve_id=valve_id,
                version=1,
                state=ValveState.CLOSED,
                source_class="reserved",
                authority_class=AuthorityClass.CORE,
                governed_regions=frozenset({LogicalRegion.SCRATCH}),
                envelope_type="none",
                budget=closed_budget,
                rejection_policy="reject",
            )
        )

    return HeartValveRegistry(definitions)


__all__ = [
    "HeartValveDefinition",
    "HeartValveRegistry",
    "ValveBudget",
    "ValveBudgetTracker",
    "ValveDecision",
    "ValveEnvelope",
    "ValveReceipt",
    "ValveState",
    "primitive_valve_registry",
]

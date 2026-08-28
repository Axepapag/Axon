"""Exact categorical output contract for reasoning cores.

Learned tissue emits categories and addresses; it never writes text or canonical
state directly.  Text payloads use the permanent 351-token Unicode transport
plus explicit EMPTY and EOS categories.  The Heart alone decodes, validates
authority and converts an accepted emission into a typed ``FieldDelta``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from runtime.field import (
    DeleteText,
    FieldDelta,
    InsertText,
    LogicalRegion,
    ReplaceText,
    SharedFieldSnapshot,
    canonical_sha256,
    validate_delta,
)
from substrate import (
    TRANSPORT_VOCAB_SIZE,
    decode_unicode_tokens,
    encode_unicode_text,
)

from .authority import AuthorityGrant

REASONING_TEXT_FRAME_SCHEMA = "axon-reasoning-text-frame-v1"
REASONING_EMISSION_SCHEMA = "axon-reasoning-emission-v1"
EMPTY_CATEGORY_ID = TRANSPORT_VOCAB_SIZE
EOS_CATEGORY_ID = TRANSPORT_VOCAB_SIZE + 1
REASONING_CATEGORY_COUNT = TRANSPORT_VOCAB_SIZE + 2


class ReasoningOutputError(ValueError):
    """A learned output is malformed, stale, or not exactly decodable."""


class ReasoningDecision(str, Enum):
    DELTA = "delta"
    NO_OP = "no_op"
    ABSTAIN = "abstain"


class ReasoningOperationKind(str, Enum):
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class CategoricalTextFrame:
    """Unlimited row-major categorical text with explicit EMPTY/EOS lanes."""

    d_model: int
    rows: tuple[tuple[int, ...], ...]
    frame_id: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int):
            raise TypeError("CategoricalTextFrame.d_model must be an integer")
        if self.d_model < 16 or self.d_model % 16:
            raise ValueError("CategoricalTextFrame.d_model must be a positive multiple of 16")
        lanes = self.d_model // 16
        rows = tuple(tuple(row) for row in self.rows)
        if not rows:
            raise ReasoningOutputError("categorical text requires at least one row containing EOS")
        if any(len(row) != lanes for row in rows):
            raise ReasoningOutputError(
                f"every categorical row must contain exactly {lanes} lane decisions"
            )
        flat: list[int] = []
        for row in rows:
            for category in row:
                if isinstance(category, bool) or not isinstance(category, int):
                    raise TypeError("reasoning output categories must be integers")
                if not 0 <= category < REASONING_CATEGORY_COUNT:
                    raise ReasoningOutputError(f"reasoning category out of range: {category}")
                flat.append(category)
        eos_positions = [index for index, value in enumerate(flat) if value == EOS_CATEGORY_ID]
        if len(eos_positions) != 1:
            raise ReasoningOutputError("categorical text requires exactly one EOS category")
        eos = eos_positions[0]
        if EMPTY_CATEGORY_ID in flat[:eos]:
            raise ReasoningOutputError("EMPTY categories may appear only after EOS")
        if any(value != EMPTY_CATEGORY_ID for value in flat[eos + 1 :]):
            raise ReasoningOutputError("only EMPTY categories may follow EOS")
        transport = tuple(flat[:eos])
        try:
            decoded = decode_unicode_tokens(transport)
        except (TypeError, ValueError) as exc:
            raise ReasoningOutputError(f"categorical text is not canonical Unicode transport: {exc}") from exc
        if encode_unicode_text(decoded) != transport:
            raise ReasoningOutputError("categorical text failed exact canonical re-encode")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "frame_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def lanes_per_row(self) -> int:
        return self.d_model // 16

    @property
    def transport_token_ids(self) -> tuple[int, ...]:
        flat = tuple(value for row in self.rows for value in row)
        eos = flat.index(EOS_CATEGORY_ID)
        return flat[:eos]

    @property
    def text(self) -> str:
        return decode_unicode_tokens(self.transport_token_ids)

    @classmethod
    def from_text(cls, text: str, *, d_model: int) -> "CategoricalTextFrame":
        if not isinstance(text, str):
            raise TypeError("CategoricalTextFrame.from_text requires text")
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model < 16 or d_model % 16:
            raise ValueError("d_model must be a positive multiple of 16")
        lanes = d_model // 16
        categories = [*encode_unicode_text(text), EOS_CATEGORY_ID]
        while len(categories) % lanes:
            categories.append(EMPTY_CATEGORY_ID)
        rows = tuple(tuple(categories[index : index + lanes]) for index in range(0, len(categories), lanes))
        return cls(d_model=d_model, rows=rows)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CategoricalTextFrame":
        """Strictly reconstruct and verify one serialized categorical frame."""

        if not isinstance(value, Mapping):
            raise TypeError("categorical text frame must be a mapping")
        required = {
            "schema",
            "d_model",
            "lanes_per_row",
            "category_count",
            "transport_vocab_size",
            "empty_category_id",
            "eos_category_id",
            "rows",
            "frame_id",
        }
        if set(value) != required or value.get("schema") != REASONING_TEXT_FRAME_SCHEMA:
            raise ReasoningOutputError("serialized categorical text frame fields/schema are invalid")
        frame = cls(
            d_model=value["d_model"],
            rows=tuple(tuple(row) for row in value["rows"]),
        )
        if (
            value["lanes_per_row"] != frame.lanes_per_row
            or value["category_count"] != REASONING_CATEGORY_COUNT
            or value["transport_vocab_size"] != TRANSPORT_VOCAB_SIZE
            or value["empty_category_id"] != EMPTY_CATEGORY_ID
            or value["eos_category_id"] != EOS_CATEGORY_ID
            or value["frame_id"] != frame.frame_id
        ):
            raise ReasoningOutputError("serialized categorical text frame identity/geometry mismatch")
        return frame

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": REASONING_TEXT_FRAME_SCHEMA,
            "d_model": self.d_model,
            "lanes_per_row": self.d_model // 16,
            "category_count": REASONING_CATEGORY_COUNT,
            "transport_vocab_size": TRANSPORT_VOCAB_SIZE,
            "empty_category_id": EMPTY_CATEGORY_ID,
            "eos_category_id": EOS_CATEGORY_ID,
            "rows": [list(row) for row in self.rows],
        }
        if include_id:
            value["frame_id"] = self.frame_id
        return value


@dataclass(frozen=True, slots=True)
class ReasoningOperationEmission:
    kind: ReasoningOperationKind
    region: LogicalRegion
    start: int
    end: int
    payload: CategoricalTextFrame
    provenance: str = ""
    container_refs: tuple[str, ...] = ()
    edge_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, ReasoningOperationKind) else ReasoningOperationKind(self.kind)
        region = self.region if isinstance(self.region, LogicalRegion) else LogicalRegion(self.region)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "region", region)
        for name in ("start", "end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReasoningOutputError(f"operation {name} must be a non-negative integer")
        if self.end < self.start:
            raise ReasoningOutputError("operation end precedes start")
        if not isinstance(self.payload, CategoricalTextFrame):
            raise TypeError("ReasoningOperationEmission.payload must be CategoricalTextFrame")
        text = self.payload.text
        if kind is ReasoningOperationKind.INSERT:
            if self.start != self.end or not text:
                raise ReasoningOutputError("insert requires start == end and non-empty payload")
        elif kind is ReasoningOperationKind.DELETE:
            if self.end <= self.start or text:
                raise ReasoningOutputError("delete requires a non-empty address range and empty payload")
        elif self.end < self.start or not text:
            raise ReasoningOutputError("replace requires an ordered range and non-empty payload")
        object.__setattr__(self, "provenance", str(self.provenance))
        object.__setattr__(self, "container_refs", tuple(sorted(set(map(str, self.container_refs)))))
        object.__setattr__(self, "edge_refs", tuple(sorted(set(map(str, self.edge_refs)))))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "region": self.region.value,
            "start": self.start,
            "end": self.end,
            "payload": self.payload.to_canonical_dict(),
            "provenance": self.provenance,
            "container_refs": list(self.container_refs),
            "edge_refs": list(self.edge_refs),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReasoningOperationEmission":
        if not isinstance(value, Mapping):
            raise TypeError("reasoning operation emission must be a mapping")
        required = {
            "kind",
            "region",
            "start",
            "end",
            "payload",
            "provenance",
            "container_refs",
            "edge_refs",
        }
        if set(value) != required:
            raise ReasoningOutputError("serialized reasoning operation fields are invalid")
        return cls(
            kind=value["kind"],
            region=value["region"],
            start=value["start"],
            end=value["end"],
            payload=CategoricalTextFrame.from_mapping(value["payload"]),
            provenance=value["provenance"],
            container_refs=tuple(value["container_refs"]),
            edge_refs=tuple(value["edge_refs"]),
        )


@dataclass(frozen=True, slots=True)
class ReasoningEmission:
    """One core decision bound to a frozen base, pass, author and home rail."""

    base_field_id: str
    base_tick_id: int
    author_core_id: str
    pass_id: str
    rail_d_model: int
    decision: ReasoningDecision
    operations: tuple[ReasoningOperationEmission, ...] = ()
    detail: CategoricalTextFrame | None = None
    evidence: tuple[str, ...] = ()
    emission_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("base_field_id", "author_core_id", "pass_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ReasoningOutputError(f"{name} must be non-empty")
        if isinstance(self.base_tick_id, bool) or not isinstance(self.base_tick_id, int) or self.base_tick_id < 0:
            raise ReasoningOutputError("base_tick_id must be non-negative")
        if isinstance(self.rail_d_model, bool) or not isinstance(self.rail_d_model, int):
            raise TypeError("rail_d_model must be an integer")
        if self.rail_d_model < 16 or self.rail_d_model % 16:
            raise ReasoningOutputError("rail_d_model must be a positive multiple of 16")
        decision = self.decision if isinstance(self.decision, ReasoningDecision) else ReasoningDecision(self.decision)
        operations = tuple(self.operations)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "operations", operations)
        if not all(isinstance(item, ReasoningOperationEmission) for item in operations):
            raise TypeError("operations must contain ReasoningOperationEmission values")
        if any(item.payload.d_model != self.rail_d_model for item in operations):
            raise ReasoningOutputError("operation payload width differs from the home rail")
        if self.detail is not None:
            if not isinstance(self.detail, CategoricalTextFrame):
                raise TypeError("detail must be CategoricalTextFrame or None")
            if self.detail.d_model != self.rail_d_model:
                raise ReasoningOutputError("detail width differs from the home rail")
        if decision is ReasoningDecision.DELTA and not operations:
            raise ReasoningOutputError("delta decision requires at least one operation")
        if decision is not ReasoningDecision.DELTA and operations:
            raise ReasoningOutputError("no-op and abstain decisions cannot carry operations")
        object.__setattr__(self, "evidence", tuple(sorted(set(map(str, self.evidence)))))
        object.__setattr__(self, "emission_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @classmethod
    def no_op(
        cls,
        *,
        base: SharedFieldSnapshot,
        author_core_id: str,
        pass_id: str,
        rail_d_model: int,
        detail: str = "",
    ) -> "ReasoningEmission":
        return cls(
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
            author_core_id=author_core_id,
            pass_id=pass_id,
            rail_d_model=rail_d_model,
            decision=ReasoningDecision.NO_OP,
            detail=(None if not detail else CategoricalTextFrame.from_text(detail, d_model=rail_d_model)),
        )

    @classmethod
    def abstain(
        cls,
        *,
        base: SharedFieldSnapshot,
        author_core_id: str,
        pass_id: str,
        rail_d_model: int,
        detail: str,
    ) -> "ReasoningEmission":
        if not detail:
            raise ReasoningOutputError("abstention requires an exact categorical detail")
        return cls(
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
            author_core_id=author_core_id,
            pass_id=pass_id,
            rail_d_model=rail_d_model,
            decision=ReasoningDecision.ABSTAIN,
            detail=CategoricalTextFrame.from_text(detail, d_model=rail_d_model),
        )

    def decode_delta(
        self,
        base: SharedFieldSnapshot,
        grant: AuthorityGrant,
    ) -> FieldDelta | None:
        if self.base_field_id != base.field_id or self.base_tick_id != base.tick_id:
            raise ReasoningOutputError("reasoning emission is stale for the supplied frozen base")
        if self.decision is not ReasoningDecision.DELTA:
            return None
        operations = []
        for item in self.operations:
            common = {
                "region": item.region,
                "provenance": item.provenance or f"reasoning-emission:{self.emission_id}",
                "container_refs": item.container_refs,
                "edge_refs": item.edge_refs,
            }
            if item.kind is ReasoningOperationKind.INSERT:
                operation = InsertText(offset=item.start, text=item.payload.text, **common)
            elif item.kind is ReasoningOperationKind.DELETE:
                operation = DeleteText(start=item.start, end=item.end, **common)
            else:
                operation = ReplaceText(start=item.start, end=item.end, text=item.payload.text, **common)
            operations.append(operation)
        delta = FieldDelta(
            base_field_id=self.base_field_id,
            base_tick_id=self.base_tick_id,
            author_core_id=self.author_core_id,
            pass_id=self.pass_id,
            operations=tuple(operations),
            evidence=self.evidence,
        )
        validate_delta(base, delta, permitted_regions=grant.governed_regions)
        return delta

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": REASONING_EMISSION_SCHEMA,
            "base_field_id": self.base_field_id,
            "base_tick_id": self.base_tick_id,
            "author_core_id": self.author_core_id,
            "pass_id": self.pass_id,
            "rail_d_model": self.rail_d_model,
            "decision": self.decision.value,
            "operations": [item.to_canonical_dict() for item in self.operations],
            "detail": None if self.detail is None else self.detail.to_canonical_dict(),
            "evidence": list(self.evidence),
        }
        if include_id:
            value["emission_id"] = self.emission_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReasoningEmission":
        """Strictly reconstruct and verify one serialized core decision."""

        if not isinstance(value, Mapping):
            raise TypeError("reasoning emission must be a mapping")
        required = {
            "schema",
            "base_field_id",
            "base_tick_id",
            "author_core_id",
            "pass_id",
            "rail_d_model",
            "decision",
            "operations",
            "detail",
            "evidence",
            "emission_id",
        }
        if set(value) != required or value.get("schema") != REASONING_EMISSION_SCHEMA:
            raise ReasoningOutputError("serialized reasoning emission fields/schema are invalid")
        detail = value["detail"]
        emission = cls(
            base_field_id=value["base_field_id"],
            base_tick_id=value["base_tick_id"],
            author_core_id=value["author_core_id"],
            pass_id=value["pass_id"],
            rail_d_model=value["rail_d_model"],
            decision=value["decision"],
            operations=tuple(ReasoningOperationEmission.from_mapping(item) for item in value["operations"]),
            detail=None if detail is None else CategoricalTextFrame.from_mapping(detail),
            evidence=tuple(value["evidence"]),
        )
        if value["emission_id"] != emission.emission_id:
            raise ReasoningOutputError("serialized reasoning emission identity mismatch")
        return emission


def delta_emission(
    delta: FieldDelta,
    *,
    rail_d_model: int,
) -> ReasoningEmission:
    """Encode an existing exact typed delta through the categorical contract."""

    if not isinstance(delta, FieldDelta):
        raise TypeError("delta_emission requires FieldDelta")
    emitted: list[ReasoningOperationEmission] = []
    for operation in delta.operations:
        if isinstance(operation, InsertText):
            kind = ReasoningOperationKind.INSERT
        elif isinstance(operation, DeleteText):
            kind = ReasoningOperationKind.DELETE
        else:
            kind = ReasoningOperationKind.REPLACE
        emitted.append(
            ReasoningOperationEmission(
                kind=kind,
                region=operation.region,
                start=operation.start,
                end=operation.end,
                payload=CategoricalTextFrame.from_text(operation.replacement_text, d_model=rail_d_model),
                provenance=operation.provenance,
                container_refs=operation.container_refs,
                edge_refs=operation.edge_refs,
            )
        )
    return ReasoningEmission(
        base_field_id=delta.base_field_id,
        base_tick_id=delta.base_tick_id,
        author_core_id=delta.author_core_id,
        pass_id=str(delta.pass_id),
        rail_d_model=rail_d_model,
        decision=ReasoningDecision.DELTA,
        operations=tuple(emitted),
        evidence=delta.evidence,
    )


__all__ = [
    "EMPTY_CATEGORY_ID",
    "EOS_CATEGORY_ID",
    "REASONING_CATEGORY_COUNT",
    "REASONING_EMISSION_SCHEMA",
    "REASONING_TEXT_FRAME_SCHEMA",
    "CategoricalTextFrame",
    "ReasoningDecision",
    "ReasoningEmission",
    "ReasoningOperationEmission",
    "ReasoningOperationKind",
    "ReasoningOutputError",
    "delta_emission",
]

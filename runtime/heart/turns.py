"""Deterministic Heart materialization of a completed conversational turn."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from runtime.field import (
    DeleteText,
    FieldDelta,
    InsertText,
    LogicalRegion,
    SharedFieldSnapshot,
    apply_delta,
    canonical_sha256,
)

from .authority import AuthorityGrant

TURN_FRAME_SCHEMA = "axon-readable-turn-frame-v1"
TURN_FINALIZATION_SCHEMA = "axon-heart-turn-finalization-v1"


class TurnFinalizationError(ValueError):
    pass


def frame_completed_turn(user_text: str, response_text: str) -> str:
    """Readable, length-delimited framing that preserves both strings exactly."""

    if not isinstance(user_text, str) or not isinstance(response_text, str):
        raise TypeError("completed turn text must be strings")
    return (
        f"[{TURN_FRAME_SCHEMA} user_chars={len(user_text)} response_chars={len(response_text)}]\n"
        "USER\n"
        f"{user_text}\n"
        "AXON\n"
        f"{response_text}\n"
        f"[/{TURN_FRAME_SCHEMA}]\n"
    )


@dataclass(frozen=True, slots=True)
class TurnFinalizationReceipt:
    base_field_id: str
    source_delta_id: str
    user_text_sha256: str
    response_text_sha256: str
    framed_turn_sha256: str
    framed_turn_chars: int
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "base_field_id",
            "source_delta_id",
            "user_text_sha256",
            "response_text_sha256",
            "framed_turn_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TurnFinalizationError(f"{name} must be non-empty")
        object.__setattr__(self, "receipt_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TURN_FINALIZATION_SCHEMA,
            "base_field_id": self.base_field_id,
            "source_delta_id": self.source_delta_id,
            "user_text_sha256": self.user_text_sha256,
            "response_text_sha256": self.response_text_sha256,
            "framed_turn_sha256": self.framed_turn_sha256,
            "framed_turn_chars": self.framed_turn_chars,
        }
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value


def materialize_completed_turn(
    base: SharedFieldSnapshot,
    proposed: FieldDelta,
) -> tuple[FieldDelta, TurnFinalizationReceipt | None]:
    """Add exact history append/input-clear operations to a consolidator delta.

    The consolidator remains the semantic author.  Heart adds only deterministic
    bookkeeping after proving that the proposal itself produced a fresh response.
    During this first form, a proposal may not also edit ``user_input`` or
    ``conversation_history`` because overlapping authorship would be ambiguous.
    """

    if not isinstance(base, SharedFieldSnapshot) or not isinstance(proposed, FieldDelta):
        raise TypeError("turn finalization requires a base snapshot and FieldDelta")
    user_text = base.region(LogicalRegion.USER_INPUT).text
    if not user_text:
        return proposed, None
    reserved = {LogicalRegion.USER_INPUT, LogicalRegion.CONVERSATION_HISTORY}
    if any(operation.region in reserved for operation in proposed.operations):
        raise TurnFinalizationError(
            "automatic turn finalization reserves user_input and conversation_history in this first form"
        )
    if not any(operation.region is LogicalRegion.RESPONSE_DRAFT for operation in proposed.operations):
        raise TurnFinalizationError(
            "a user turn requires the consolidator proposal to author response_draft"
        )
    staged = apply_delta(base, proposed, permitted_regions=AuthorityGrant.consolidator().governed_regions)
    response_text = staged.region(LogicalRegion.RESPONSE_DRAFT).text
    if not response_text:
        raise TurnFinalizationError("a completed user turn requires a non-empty response_draft")
    framed = frame_completed_turn(user_text, response_text)
    provenance = f"heart-turn-finalizer:{proposed.delta_id}"
    operations = list(proposed.operations)
    operations.extend(
        (
            InsertText(
                region=LogicalRegion.CONVERSATION_HISTORY,
                offset=len(base.region(LogicalRegion.CONVERSATION_HISTORY).text),
                text=framed,
                provenance=provenance,
            ),
            DeleteText(
                region=LogicalRegion.USER_INPUT,
                start=0,
                end=len(user_text),
                provenance=provenance,
            ),
        )
    )
    materialized = FieldDelta(
        base_field_id=proposed.base_field_id,
        base_tick_id=proposed.base_tick_id,
        author_core_id=proposed.author_core_id,
        pass_id=proposed.pass_id,
        operations=tuple(operations),
        evidence=(*proposed.evidence, proposed.delta_id),
    )
    receipt = TurnFinalizationReceipt(
        base_field_id=base.field_id,
        source_delta_id=proposed.delta_id,
        user_text_sha256=hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
        response_text_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        framed_turn_sha256=hashlib.sha256(framed.encode("utf-8")).hexdigest(),
        framed_turn_chars=len(framed),
    )
    return materialized, receipt


__all__ = [
    "TURN_FINALIZATION_SCHEMA",
    "TURN_FRAME_SCHEMA",
    "TurnFinalizationError",
    "TurnFinalizationReceipt",
    "frame_completed_turn",
    "materialize_completed_turn",
]

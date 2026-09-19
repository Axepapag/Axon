"""Exact English reasoning contracts for inter-core proposals and final verdicts.

FIRST and REFINED communication is ordinary variable-length English text with one
exact Unicode identity independent of rail width.  CONSOLIDATED output is also
English, but uses a deliberately tiny tagged-region surface such as::

    #responseDraft# Hello Jeff.
    #scratch# Remember to inspect the trainer lineage next.

A tag names a canonical Shared Field region; the following text is the complete
desired text of that region.  Unmentioned regions remain unchanged.  The Heart
already owns the frozen base/tick/author context, so the learned consolidator does
not waste output reproducing transaction metadata or numeric character addresses.
Heart parses the tags mechanically, compares them with the frozen base, and
materializes the internal typed ``FieldDelta`` transaction.  It performs no
semantic interpretation.

This module intentionally contains no learned decision/no-op/abstain gate.
Runtime failure, timeout, and incompletion remain control-plane states.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from runtime.field import (
    CANONICAL_REGION_ORDER,
    FieldDelta,
    InsertText,
    LogicalRegion,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
    canonical_sha256,
    validate_delta,
)

from .authority import AuthorityGrant
from .reasoning_output import CategoricalTextFrame

ENGLISH_PROPOSAL_SCHEMA = "axon-english-proposal-v1"
TECHNICAL_FINAL_VERDICT_SCHEMA = "axon-tagged-final-verdict-v2"

# Human-facing canonical tag names.  ``journal`` is the public architectural
# name for the historical ``diary`` field region.
REGION_TAGS: Mapping[LogicalRegion, str] = {
    LogicalRegion.CONVERSATION_HISTORY: "conversationHistory",
    LogicalRegion.USER_INPUT: "userInput",
    LogicalRegion.CORTEX: "cortex",
    LogicalRegion.SITUATION_AWARENESS: "situationAwareness",
    LogicalRegion.TOOL_RESULTS: "toolResults",
    LogicalRegion.ADVISOR_INPUT: "advisorInput",
    LogicalRegion.TASK_STATE: "taskState",
    LogicalRegion.SCRATCH: "scratch",
    LogicalRegion.RESPONSE_DRAFT: "responseDraft",
    LogicalRegion.DIARY: "journal",
    LogicalRegion.IDENTITY: "identity",
    LogicalRegion.TRAINER_INSTRUCTIONS: "trainerInstructions",
    LogicalRegion.TRAINING_RESPONSES: "trainingResponses",
}
_TAG_TO_REGION: dict[str, LogicalRegion] = {tag: region for region, tag in REGION_TAGS.items()}
_TAG_TO_REGION["diary"] = LogicalRegion.DIARY  # accepted compatibility alias
_TAG_LINE_RE = re.compile(r"^#([A-Za-z][A-Za-z0-9]*)#(?: (.*))?$")


class EnglishReasoningContractError(ValueError):
    """English proposal or final-verdict text violates the exact contract."""


def _require_exact_unicode_text(value: str, *, name: str, nonempty: bool) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if nonempty and not value.strip():
        raise EnglishReasoningContractError(f"{name} must be nonempty English text")
    try:
        encoded = value.encode("utf-8", errors="strict")
        decoded = encoded.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise EnglishReasoningContractError(f"{name} is not valid exact Unicode") from exc
    if decoded != value:
        raise EnglishReasoningContractError(f"{name} failed exact Unicode roundtrip")
    return value


def _coerce_region(value: LogicalRegion | str) -> LogicalRegion:
    if isinstance(value, LogicalRegion):
        return value
    if value in _TAG_TO_REGION:
        return _TAG_TO_REGION[value]
    try:
        return LogicalRegion(value)
    except (TypeError, ValueError) as exc:
        raise EnglishReasoningContractError(f"unknown final-verdict region {value!r}") from exc


def _looks_like_tag_line(line: str) -> bool:
    return _TAG_LINE_RE.fullmatch(line) is not None


def _escape_body_line(line: str) -> str:
    # A leading backslash escapes itself.  A line that looks like a region tag is
    # escaped so arbitrary English/code can still contain literal tag-shaped text.
    if line.startswith("\\") or _looks_like_tag_line(line):
        return "\\" + line
    return line


def _render_section(region: LogicalRegion, text: str) -> list[str]:
    tag = REGION_TAGS[region]
    _require_exact_unicode_text(text, name=f"{tag} region text", nonempty=False)
    body_lines = text.split("\n")
    first = body_lines[0] if body_lines else ""
    lines = [f"#{tag}#" + (f" {first}" if first else "")]
    lines.extend(_escape_body_line(line) for line in body_lines[1:])
    return lines


def _parse_sections(text: str) -> tuple[tuple[LogicalRegion, str], ...]:
    _require_exact_unicode_text(text, name="final verdict", nonempty=True)
    lines = text.split("\n")
    sections: list[tuple[LogicalRegion, str]] = []
    seen: set[LogicalRegion] = set()
    current_region: LogicalRegion | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_region, current_lines
        if current_region is None:
            return
        sections.append((current_region, "\n".join(current_lines)))
        current_region = None
        current_lines = []

    for line in lines:
        if current_region is not None and line.startswith("\\"):
            current_lines.append(line[1:])
            continue

        match = _TAG_LINE_RE.fullmatch(line)
        if match is not None:
            tag = match.group(1)
            region = _TAG_TO_REGION.get(tag)
            if region is None:
                raise EnglishReasoningContractError(f"unknown final-verdict tag #{tag}#")
            if region in seen:
                raise EnglishReasoningContractError(f"duplicate final-verdict region tag #{tag}#")
            flush()
            seen.add(region)
            current_region = region
            first = match.group(2)
            current_lines = [] if first is None else [first]
            continue

        if current_region is None:
            raise EnglishReasoningContractError("final verdict text must begin with a known #region# tag")
        current_lines.append(line)

    flush()
    if not sections:
        raise EnglishReasoningContractError("final verdict must contain at least one #region# section")
    return tuple(sections)


@dataclass(frozen=True, slots=True)
class EnglishProposal:
    """One exact nonempty FIRST or REFINED proposal authored by a reasoning core."""

    base_field_id: str
    base_tick_id: int
    author_core_id: str
    pass_id: str
    rail_d_model: int
    text: str
    evidence: tuple[str, ...] = ()
    proposal_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("base_field_id", "author_core_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise EnglishReasoningContractError(f"{name} must be nonempty")
        if isinstance(self.base_tick_id, bool) or not isinstance(self.base_tick_id, int) or self.base_tick_id < 0:
            raise EnglishReasoningContractError("base_tick_id must be a non-negative integer")
        if self.pass_id not in {"first", "refined"}:
            raise EnglishReasoningContractError("EnglishProposal.pass_id must be 'first' or 'refined'")
        if (
            isinstance(self.rail_d_model, bool)
            or not isinstance(self.rail_d_model, int)
            or self.rail_d_model < 16
            or self.rail_d_model % 16
        ):
            raise EnglishReasoningContractError("rail_d_model must be a positive multiple of 16")
        _require_exact_unicode_text(self.text, name="proposal text", nonempty=True)
        evidence = tuple(sorted(set(str(item) for item in self.evidence)))
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "proposal_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": ENGLISH_PROPOSAL_SCHEMA,
            "base_field_id": self.base_field_id,
            "base_tick_id": self.base_tick_id,
            "author_core_id": self.author_core_id,
            "pass_id": self.pass_id,
            "rail_d_model": self.rail_d_model,
            "text": self.text,
            "evidence": list(self.evidence),
        }
        if include_id:
            value["proposal_id"] = self.proposal_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EnglishProposal":
        if not isinstance(value, Mapping) or value.get("schema") != ENGLISH_PROPOSAL_SCHEMA:
            raise EnglishReasoningContractError("serialized English proposal schema is invalid")
        required = {
            "schema", "base_field_id", "base_tick_id", "author_core_id", "pass_id",
            "rail_d_model", "text", "evidence", "proposal_id",
        }
        if set(value) != required:
            raise EnglishReasoningContractError("serialized English proposal fields are invalid")
        proposal = cls(
            base_field_id=value["base_field_id"],
            base_tick_id=value["base_tick_id"],
            author_core_id=value["author_core_id"],
            pass_id=value["pass_id"],
            rail_d_model=value["rail_d_model"],
            text=value["text"],
            evidence=tuple(value["evidence"]),
        )
        if value["proposal_id"] != proposal.proposal_id:
            raise EnglishReasoningContractError("serialized English proposal identity mismatch")
        return proposal

    def frame_for(self, d_model: int) -> CategoricalTextFrame:
        """Mechanically repack the exact proposal text for any registered rail width."""

        frame = CategoricalTextFrame.from_text(self.text, d_model=d_model)
        if frame.text != self.text:
            raise EnglishReasoningContractError("proposal text changed during rail repacking")
        return frame


def render_final_verdict(sections: Mapping[LogicalRegion | str, str]) -> str:
    """Render a compact tagged-region verdict.

    Each named section is the complete desired text of one region.  The mapping's
    insertion order is preserved so the consolidator is not trained on a needless
    region-order decision.  Heart semantics are carried entirely by the tags.
    """

    if not isinstance(sections, Mapping) or not sections:
        raise EnglishReasoningContractError("final verdict requires at least one region section")
    rendered: list[str] = []
    seen: set[LogicalRegion] = set()
    for raw_region, raw_text in sections.items():
        region = _coerce_region(raw_region)
        if region in seen:
            raise EnglishReasoningContractError(f"duplicate final-verdict region {region.value!r}")
        seen.add(region)
        rendered.extend(_render_section(region, raw_text))
    return "\n".join(rendered)


def parse_final_verdict(
    text: str,
    *,
    base: SharedFieldSnapshot,
    author_core_id: str,
    evidence: tuple[str, ...] = (),
) -> FieldDelta:
    """Mechanically turn tagged desired-region text into an internal typed delta."""

    if not isinstance(base, SharedFieldSnapshot):
        raise TypeError("parse_final_verdict requires the frozen SharedFieldSnapshot base")
    if not isinstance(author_core_id, str) or not author_core_id:
        raise EnglishReasoningContractError("author_core_id must be nonempty")

    operations: list[InsertText | ReplaceText] = []
    for region, desired in _parse_sections(text):
        current = base.region(region).text
        if desired == current:
            continue
        if not current:
            if desired:
                operations.append(InsertText(region=region, offset=0, text=desired))
        else:
            # Whole-region desired-state semantics keep the learned grammar tiny.
            # Heart may later optimize this internal materialization without changing
            # what the consolidator has to learn to say.
            operations.append(ReplaceText(region=region, start=0, end=len(current), text=desired))

    if not operations:
        raise EnglishReasoningContractError("final verdict must change at least one canonical region")

    delta = FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id=author_core_id,
        pass_id="consolidated",
        operations=tuple(operations),
        evidence=tuple(sorted(set(str(item) for item in evidence))),
    )
    grant = AuthorityGrant.consolidator()
    grant.assert_delta_permitted(delta)
    validate_delta(base, delta, permitted_regions=grant.governed_regions)
    return delta


@dataclass(frozen=True, slots=True)
class TechnicalFinalVerdict:
    """One tagged-region FINAL utterance authored by the rotating consolidator.

    The learned core emits only ``text``.  Binding metadata is supplied by the
    runtime envelope, not spoken by the model.  Heart materializes ``text``
    against the frozen base into an internal ``FieldDelta`` only after receipt.
    """

    base_field_id: str
    base_tick_id: int
    author_core_id: str
    rail_d_model: int
    text: str
    evidence: tuple[str, ...] = ()
    verdict_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("base_field_id", "author_core_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise EnglishReasoningContractError(f"{name} must be nonempty")
        if isinstance(self.base_tick_id, bool) or not isinstance(self.base_tick_id, int) or self.base_tick_id < 0:
            raise EnglishReasoningContractError("base_tick_id must be a non-negative integer")
        if (
            isinstance(self.rail_d_model, bool)
            or not isinstance(self.rail_d_model, int)
            or self.rail_d_model < 16
            or self.rail_d_model % 16
        ):
            raise EnglishReasoningContractError("rail_d_model must be a positive multiple of 16")
        _require_exact_unicode_text(self.text, name="final verdict", nonempty=True)
        # Syntax is checked without needing semantic state.  Actual change and
        # authority are checked later when Heart binds the frozen base.
        _parse_sections(self.text)
        evidence = tuple(sorted(set(str(item) for item in self.evidence)))
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "verdict_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TechnicalFinalVerdict":
        if not isinstance(value, Mapping) or value.get("schema") != TECHNICAL_FINAL_VERDICT_SCHEMA:
            raise EnglishReasoningContractError("serialized FINAL verdict schema is invalid")
        required = {
            "schema", "base_field_id", "base_tick_id", "author_core_id",
            "rail_d_model", "text", "evidence", "verdict_id",
        }
        if set(value) != required:
            raise EnglishReasoningContractError("serialized FINAL verdict fields are invalid")
        verdict = cls(
            base_field_id=value["base_field_id"],
            base_tick_id=value["base_tick_id"],
            author_core_id=value["author_core_id"],
            rail_d_model=value["rail_d_model"],
            text=value["text"],
            evidence=tuple(value["evidence"]),
        )
        if value["verdict_id"] != verdict.verdict_id:
            raise EnglishReasoningContractError("serialized FINAL verdict identity mismatch")
        return verdict

    def materialize(self, base: SharedFieldSnapshot) -> FieldDelta:
        if base.field_id != self.base_field_id or base.tick_id != self.base_tick_id:
            raise EnglishReasoningContractError("final verdict is stale for the supplied frozen base")
        return parse_final_verdict(
            self.text,
            base=base,
            author_core_id=self.author_core_id,
            evidence=self.evidence,
        )

    @classmethod
    def from_sections(
        cls,
        *,
        base: SharedFieldSnapshot,
        author_core_id: str,
        rail_d_model: int,
        sections: Mapping[LogicalRegion | str, str],
        evidence: tuple[str, ...] = (),
    ) -> "TechnicalFinalVerdict":
        return cls(
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
            author_core_id=author_core_id,
            rail_d_model=rail_d_model,
            text=render_final_verdict(sections),
            evidence=evidence,
        )

    @classmethod
    def from_delta(
        cls,
        base: SharedFieldSnapshot,
        delta: FieldDelta,
        *,
        rail_d_model: int = 64,
    ) -> "TechnicalFinalVerdict":
        """Convert a known typed delta into equivalent tagged-region training text."""

        if not isinstance(delta, FieldDelta):
            raise TypeError("TechnicalFinalVerdict.from_delta requires FieldDelta")
        if delta.base_field_id != base.field_id or delta.base_tick_id != base.tick_id:
            raise EnglishReasoningContractError("delta is stale for the supplied base")
        if str(delta.pass_id) != "consolidated":
            raise EnglishReasoningContractError("final verdict delta must use pass_id 'consolidated'")
        grant = AuthorityGrant.consolidator()
        grant.assert_delta_permitted(delta)
        successor = apply_delta(base, delta, permitted_regions=grant.governed_regions)
        touched = {operation.region for operation in delta.operations}
        sections = {
            region: successor.region(region).text
            for region in CANONICAL_REGION_ORDER
            if region in touched
        }
        verdict = cls.from_sections(
            base=base,
            author_core_id=delta.author_core_id,
            rail_d_model=rail_d_model,
            sections=sections,
            evidence=delta.evidence,
        )
        reconstructed = apply_delta(base, verdict.materialize(base), permitted_regions=grant.governed_regions)
        for region in touched:
            if reconstructed.region(region).text != successor.region(region).text:
                raise EnglishReasoningContractError("tagged verdict failed successor-state equivalence")
        return verdict

    def frame_for(self, d_model: int) -> CategoricalTextFrame:
        frame = CategoricalTextFrame.from_text(self.text, d_model=d_model)
        if frame.text != self.text:
            raise EnglishReasoningContractError("final verdict changed during rail repacking")
        return frame

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": TECHNICAL_FINAL_VERDICT_SCHEMA,
            "base_field_id": self.base_field_id,
            "base_tick_id": self.base_tick_id,
            "author_core_id": self.author_core_id,
            "rail_d_model": self.rail_d_model,
            "text": self.text,
            "evidence": list(self.evidence),
        }
        if include_id:
            value["verdict_id"] = self.verdict_id
        return value


__all__ = [
    "ENGLISH_PROPOSAL_SCHEMA",
    "REGION_TAGS",
    "TECHNICAL_FINAL_VERDICT_SCHEMA",
    "EnglishProposal",
    "EnglishReasoningContractError",
    "TechnicalFinalVerdict",
    "parse_final_verdict",
    "render_final_verdict",
]

"""Exact English reasoning contracts for inter-core proposals and final verdicts.

FIRST and REFINED communication is ordinary variable-length English text with one
exact Unicode identity independent of rail width.  CONSOLIDATED output uses a
strict technical-English envelope whose mutation payloads are canonical JSON;
Heart parses that grammar mechanically into the existing typed ``FieldDelta``
transaction boundary.

This module intentionally contains no learned decision/no-op/abstain gate.
Runtime failure, timeout, and incompletion remain control-plane states.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from runtime.field import (
    DeleteText,
    FieldDelta,
    InsertText,
    LogicalRegion,
    ReplaceText,
    canonical_sha256,
)

from .reasoning_output import CategoricalTextFrame

ENGLISH_PROPOSAL_SCHEMA = "axon-english-proposal-v1"
TECHNICAL_FINAL_VERDICT_SCHEMA = "axon-technical-final-verdict-v1"
FINAL_VERDICT_HEADER = "AXON FINAL VERDICT V1"
FINAL_VERDICT_FOOTER = "END AXON FINAL VERDICT V1"


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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_canonical_json(text: str, *, name: str) -> Any:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EnglishReasoningContractError(f"{name} is not valid JSON") from exc
    if _canonical_json(value) != text:
        raise EnglishReasoningContractError(f"{name} is not canonical JSON")
    return value


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

    def frame_for(self, d_model: int) -> CategoricalTextFrame:
        """Mechanically repack the exact proposal text for any registered rail width."""

        frame = CategoricalTextFrame.from_text(self.text, d_model=d_model)
        if frame.text != self.text:
            raise EnglishReasoningContractError("proposal text changed during rail repacking")
        return frame


@dataclass(frozen=True, slots=True)
class TechnicalFinalVerdict:
    """Exact constrained-English consolidator verdict and its parsed transaction."""

    text: str
    delta: FieldDelta
    verdict_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_exact_unicode_text(self.text, name="final verdict", nonempty=True)
        if not isinstance(self.delta, FieldDelta):
            raise TypeError("TechnicalFinalVerdict.delta must be FieldDelta")
        if str(self.delta.pass_id) != "consolidated":
            raise EnglishReasoningContractError("final verdict delta must use pass_id 'consolidated'")
        canonical = render_final_verdict(self.delta)
        if canonical != self.text:
            raise EnglishReasoningContractError("final verdict text is not the canonical rendering of its delta")
        object.__setattr__(
            self,
            "verdict_id",
            canonical_sha256(
                {
                    "schema": TECHNICAL_FINAL_VERDICT_SCHEMA,
                    "text": self.text,
                    "delta_id": self.delta.delta_id,
                }
            ),
        )

    @classmethod
    def from_delta(cls, delta: FieldDelta) -> "TechnicalFinalVerdict":
        return cls(text=render_final_verdict(delta), delta=delta)

    @classmethod
    def parse(cls, text: str) -> "TechnicalFinalVerdict":
        delta = parse_final_verdict(text)
        return cls(text=text, delta=delta)

    def frame_for(self, d_model: int) -> CategoricalTextFrame:
        frame = CategoricalTextFrame.from_text(self.text, d_model=d_model)
        if frame.text != self.text:
            raise EnglishReasoningContractError("final verdict changed during rail repacking")
        return frame

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": TECHNICAL_FINAL_VERDICT_SCHEMA,
            "verdict_id": self.verdict_id,
            "text": self.text,
            "delta": self.delta.to_canonical_dict(),
        }


def _operation_mapping(operation: InsertText | DeleteText | ReplaceText) -> dict[str, Any]:
    return operation.to_canonical_dict()


def render_final_verdict(delta: FieldDelta) -> str:
    """Render one canonical technical-English verdict from a typed consolidator delta."""

    if not isinstance(delta, FieldDelta):
        raise TypeError("render_final_verdict requires FieldDelta")
    if str(delta.pass_id) != "consolidated":
        raise EnglishReasoningContractError("technical final verdict requires pass_id 'consolidated'")
    lines = [
        FINAL_VERDICT_HEADER,
        f"The base field is {_canonical_json(delta.base_field_id)}.",
        f"The base tick is {delta.base_tick_id}.",
        f"The author core is {_canonical_json(delta.author_core_id)}.",
        f"The evidence references are {_canonical_json(list(delta.evidence))}.",
        f"There are {len(delta.operations)} canonical mutations.",
    ]
    for index, operation in enumerate(delta.operations, start=1):
        lines.append(f"Mutation {index} is {_canonical_json(_operation_mapping(operation))}.")
    lines.append(FINAL_VERDICT_FOOTER)
    return "\n".join(lines)


_BASE_FIELD_RE = re.compile(r"^The base field is (.+)\.$")
_BASE_TICK_RE = re.compile(r"^The base tick is ([0-9]+)\.$")
_AUTHOR_RE = re.compile(r"^The author core is (.+)\.$")
_EVIDENCE_RE = re.compile(r"^The evidence references are (.+)\.$")
_COUNT_RE = re.compile(r"^There are ([0-9]+) canonical mutations\.$")
_MUTATION_RE = re.compile(r"^Mutation ([1-9][0-9]*) is (.+)\.$")


def _match(regex: re.Pattern[str], line: str, *, name: str) -> re.Match[str]:
    match = regex.fullmatch(line)
    if match is None:
        raise EnglishReasoningContractError(f"malformed {name} line")
    return match


def _string_json(text: str, *, name: str) -> str:
    value = _parse_canonical_json(text, name=name)
    if not isinstance(value, str) or not value:
        raise EnglishReasoningContractError(f"{name} must encode a nonempty string")
    _require_exact_unicode_text(value, name=name, nonempty=True)
    return value


def _string_list_json(text: str, *, name: str) -> tuple[str, ...]:
    value = _parse_canonical_json(text, name=name)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EnglishReasoningContractError(f"{name} must encode a JSON string list")
    canonical = tuple(sorted(set(value)))
    if list(canonical) != value:
        raise EnglishReasoningContractError(f"{name} must be sorted and duplicate-free")
    return canonical


def _operation_from_mapping(value: Mapping[str, Any]) -> InsertText | DeleteText | ReplaceText:
    if not isinstance(value, Mapping):
        raise EnglishReasoningContractError("mutation must be a JSON object")
    op = value.get("op")
    if op == "insert":
        required = {"op", "region", "offset", "text", "provenance", "container_refs", "edge_refs"}
    elif op in {"delete", "replace"}:
        required = {"op", "region", "start", "end", "provenance", "container_refs", "edge_refs"}
        if op == "replace":
            required.add("text")
    else:
        raise EnglishReasoningContractError(f"unsupported mutation operation {op!r}")
    if set(value) != required:
        raise EnglishReasoningContractError("mutation object fields do not match its operation")
    try:
        region = LogicalRegion(value["region"])
    except (TypeError, ValueError) as exc:
        raise EnglishReasoningContractError(f"unknown mutation region {value.get('region')!r}") from exc
    provenance = value["provenance"]
    container_refs = value["container_refs"]
    edge_refs = value["edge_refs"]
    if not isinstance(provenance, str):
        raise EnglishReasoningContractError("mutation provenance must be text")
    if not isinstance(container_refs, list) or any(not isinstance(item, str) for item in container_refs):
        raise EnglishReasoningContractError("container_refs must be a JSON string list")
    if not isinstance(edge_refs, list) or any(not isinstance(item, str) for item in edge_refs):
        raise EnglishReasoningContractError("edge_refs must be a JSON string list")
    if container_refs != sorted(set(container_refs)) or edge_refs != sorted(set(edge_refs)):
        raise EnglishReasoningContractError("mutation references must be sorted and duplicate-free")
    common = {
        "region": region,
        "provenance": provenance,
        "container_refs": tuple(container_refs),
        "edge_refs": tuple(edge_refs),
    }
    try:
        if op == "insert":
            return InsertText(offset=value["offset"], text=value["text"], **common)
        if op == "delete":
            return DeleteText(start=value["start"], end=value["end"], **common)
        return ReplaceText(start=value["start"], end=value["end"], text=value["text"], **common)
    except (KeyError, TypeError, ValueError) as exc:
        raise EnglishReasoningContractError(f"invalid {op} mutation: {exc}") from exc


def parse_final_verdict(text: str) -> FieldDelta:
    """Parse canonical technical English into a typed delta with zero semantic inference."""

    _require_exact_unicode_text(text, name="final verdict", nonempty=True)
    lines = text.split("\n")
    if len(lines) < 7 or lines[0] != FINAL_VERDICT_HEADER or lines[-1] != FINAL_VERDICT_FOOTER:
        raise EnglishReasoningContractError("final verdict header/footer is invalid")
    if any(not line for line in lines):
        raise EnglishReasoningContractError("final verdict may not contain blank lines")

    base_field = _string_json(_match(_BASE_FIELD_RE, lines[1], name="base field").group(1), name="base field")
    base_tick = int(_match(_BASE_TICK_RE, lines[2], name="base tick").group(1))
    author = _string_json(_match(_AUTHOR_RE, lines[3], name="author core").group(1), name="author core")
    evidence = _string_list_json(
        _match(_EVIDENCE_RE, lines[4], name="evidence").group(1),
        name="evidence references",
    )
    count = int(_match(_COUNT_RE, lines[5], name="mutation count").group(1))
    if count < 1:
        raise EnglishReasoningContractError("final verdict must contain at least one canonical mutation")
    mutation_lines = lines[6:-1]
    if len(mutation_lines) != count:
        raise EnglishReasoningContractError("mutation count does not match the verdict body")

    operations: list[InsertText | DeleteText | ReplaceText] = []
    for expected_index, line in enumerate(mutation_lines, start=1):
        match = _match(_MUTATION_RE, line, name=f"mutation {expected_index}")
        actual_index = int(match.group(1))
        if actual_index != expected_index:
            raise EnglishReasoningContractError("mutations must be consecutively numbered from 1")
        value = _parse_canonical_json(match.group(2), name=f"mutation {expected_index}")
        operations.append(_operation_from_mapping(value))

    delta = FieldDelta(
        base_field_id=base_field,
        base_tick_id=base_tick,
        author_core_id=author,
        pass_id="consolidated",
        operations=tuple(operations),
        evidence=evidence,
    )
    if render_final_verdict(delta) != text:
        raise EnglishReasoningContractError("final verdict is not in the unique canonical rendering")
    return delta


__all__ = [
    "ENGLISH_PROPOSAL_SCHEMA",
    "FINAL_VERDICT_FOOTER",
    "FINAL_VERDICT_HEADER",
    "TECHNICAL_FINAL_VERDICT_SCHEMA",
    "EnglishProposal",
    "EnglishReasoningContractError",
    "TechnicalFinalVerdict",
    "parse_final_verdict",
    "render_final_verdict",
]

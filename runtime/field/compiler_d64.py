"""Deterministic D64 Field Compiler over canonical SharedFieldSnapshot state.

This module is the permanent exact compiler boundary shared by runtime and
training.  It is deliberately authority-free: it reads canonical state,
produces a reversible D64 rail with exact provenance, verifies freshness and
coverage, and constructs ordinary typed FieldDelta objects.  It does not decide
what Axon should believe or commit.

The rail is physical D64 storage, not a claim that one 64D Transformer token is
four independently attended characters.  Every row contains up to four literal
frozen 16D substrate cells and explicit lane addresses so consumers may unpack
or address the exact characters without inference.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator

import numpy as np

from substrate import SLOT_DIM, assert_supported_text, char_to_slot, get_letter_bank

from .delta import FieldDelta, ReplaceText, apply_delta
from .schema import (
    CANONICAL_REGION_ORDER,
    LOGICAL_REGION_IDS,
    LogicalRegion,
    RegionVisibility,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)


D64_COMPILER_SCHEMA = "axon-field-compiler-d64-v1"
D64_WIDTH = 64
SUBSTRATE_WIDTH = 16
D64_LANES_PER_ROW = D64_WIDTH // SUBSTRATE_WIDTH


class FieldCompilerError(RuntimeError):
    """Base class for deterministic compiler failures."""


class UnsupportedActiveCharacterError(FieldCompilerError):
    """An attended canonical character cannot be represented by the substrate."""


class IncompleteRailError(FieldCompilerError):
    """The compiler could not prove exact complete active-field coverage."""


class StaleCompiledFieldError(FieldCompilerError):
    """A compiled rail is being used against a different canonical snapshot."""


@dataclass(frozen=True, slots=True)
class CanonicalCharAddress:
    """Exact source identity for one valid 16D lane in a D64 row."""

    region: LogicalRegion
    region_position: int
    global_position: int
    span_id: str
    span_position: int
    source: str
    provenance: str
    character: str
    row_index: int
    lane_index: int

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "region": self.region.value,
            "region_position": self.region_position,
            "global_position": self.global_position,
            "span_id": self.span_id,
            "span_position": self.span_position,
            "source": self.source,
            "provenance": self.provenance,
            "character": self.character,
            "row_index": self.row_index,
            "lane_index": self.lane_index,
        }


@dataclass(frozen=True, slots=True)
class CartographicSpan:
    """Deterministic structural span over exact canonical region characters."""

    kind: str
    region: LogicalRegion
    start: int
    end: int
    text_sha256: str

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("cartographic span kind must be non-empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("cartographic span requires 0 <= start < end")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "region": self.region.value,
            "start": self.start,
            "end": self.end,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True, slots=True)
class D64CoverageManifest:
    schema: str
    source_field_id: str
    source_tick_id: int
    expected_regions: tuple[str, ...]
    visited_regions: tuple[str, ...]
    expected_active_characters: int
    compiled_active_characters: int
    row_count: int
    valid_lanes: int
    padding_lanes: int
    region_row_ranges: tuple[tuple[str, int, int], ...]
    rows_sha256: str
    address_sha256: str
    roundtrip_sha256: str
    complete: bool

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "expected_regions": list(self.expected_regions),
            "visited_regions": list(self.visited_regions),
            "expected_active_characters": self.expected_active_characters,
            "compiled_active_characters": self.compiled_active_characters,
            "row_count": self.row_count,
            "valid_lanes": self.valid_lanes,
            "padding_lanes": self.padding_lanes,
            "region_row_ranges": [list(item) for item in self.region_row_ranges],
            "rows_sha256": self.rows_sha256,
            "address_sha256": self.address_sha256,
            "roundtrip_sha256": self.roundtrip_sha256,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class D64CharacterPage:
    """One region-preserving physical page reconstructed from a D64 rail."""

    source_field_id: str
    region: LogicalRegion
    region_id: int
    region_page_index: int
    logical_page_index: int
    region_start: int
    region_end: int
    global_start: int
    global_end: int
    text: str
    cells16: np.ndarray
    addresses: tuple[CanonicalCharAddress, ...]
    empty_region_marker: bool = False

    def __post_init__(self) -> None:
        cells = np.asarray(self.cells16, dtype=np.float32)
        if cells.ndim != 2 or cells.shape[1] != SUBSTRATE_WIDTH:
            raise ValueError("D64CharacterPage.cells16 must have shape [N, 16]")
        if self.empty_region_marker:
            if self.text or self.addresses or cells.shape[0] != 0:
                raise ValueError("empty region marker may not contain character data")
        elif len(self.text) != cells.shape[0] or len(self.addresses) != cells.shape[0]:
            raise ValueError("page text/cells/addresses must have equal length")
        cells.setflags(write=False)
        object.__setattr__(self, "cells16", cells)


@dataclass(frozen=True, slots=True)
class CompiledD64Field:
    """A complete exact D64 rail bound to one immutable source snapshot."""

    source_field_id: str
    source_tick_id: int
    rows: np.ndarray
    lane_valid: np.ndarray
    addresses: tuple[CanonicalCharAddress | None, ...]
    cartography: tuple[CartographicSpan, ...]
    coverage: D64CoverageManifest
    rail_id: str

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.float32)
        lane_valid = np.asarray(self.lane_valid, dtype=np.bool_)
        if rows.ndim != 2 or rows.shape[1] != D64_WIDTH:
            raise ValueError("CompiledD64Field.rows must have shape [rows, 64]")
        if lane_valid.shape != (rows.shape[0], D64_LANES_PER_ROW):
            raise ValueError("lane_valid shape does not match D64 rows")
        if len(self.addresses) != rows.shape[0] * D64_LANES_PER_ROW:
            raise ValueError("address count does not match D64 lane count")
        if not self.coverage.complete:
            raise IncompleteRailError("cannot materialize an incomplete D64 rail")
        rows.setflags(write=False)
        lane_valid.setflags(write=False)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "lane_valid", lane_valid)

    @property
    def row_count(self) -> int:
        return int(self.rows.shape[0])

    def assert_fresh(self, snapshot: SharedFieldSnapshot) -> None:
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("snapshot must be SharedFieldSnapshot")
        if (
            snapshot.field_id != self.source_field_id
            or snapshot.tick_id != self.source_tick_id
        ):
            raise StaleCompiledFieldError(
                "compiled D64 rail is stale for the supplied canonical snapshot"
            )

    def address(self, row: int, lane: int) -> CanonicalCharAddress | None:
        if row < 0 or row >= self.row_count:
            raise IndexError("D64 row index out of range")
        if lane < 0 or lane >= D64_LANES_PER_ROW:
            raise IndexError("D64 lane index out of range")
        return self.addresses[row * D64_LANES_PER_ROW + lane]

    def lane_cell16(self, row: int, lane: int) -> np.ndarray:
        start = lane * SUBSTRATE_WIDTH
        cell = self.rows[row, start : start + SUBSTRATE_WIDTH]
        cell.setflags(write=False)
        return cell

    def region_addresses(self, region: LogicalRegion | str) -> tuple[CanonicalCharAddress, ...]:
        logical = region if isinstance(region, LogicalRegion) else LogicalRegion(region)
        return tuple(
            address
            for address in self.addresses
            if address is not None and address.region is logical
        )

    def region_text(self, region: LogicalRegion | str) -> str:
        addresses = self.region_addresses(region)
        return "".join(address.character for address in addresses)

    def active_texts(self) -> dict[str, str]:
        return {region.value: self.region_text(region) for region in CANONICAL_REGION_ORDER}

    def verify_roundtrip(self, snapshot: SharedFieldSnapshot) -> None:
        self.assert_fresh(snapshot)
        for region in CANONICAL_REGION_ORDER:
            expected = (
                snapshot.region(region).text
                if snapshot.region(region).visibility is RegionVisibility.ATTENDED
                else ""
            )
            observed = self.region_text(region)
            if observed != expected:
                raise IncompleteRailError(
                    f"D64 roundtrip mismatch in {region.value!r}: "
                    f"expected {len(expected)} chars, observed {len(observed)}"
                )

    def iter_character_pages(self, page_size: int) -> Iterator[D64CharacterPage]:
        """Yield exact region pages, using the rail as the character source."""

        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
            raise ValueError("page_size must be a positive integer")
        logical_page_index = 0
        global_cursor = 0
        for region in CANONICAL_REGION_ORDER:
            addresses = self.region_addresses(region)
            if not addresses:
                yield D64CharacterPage(
                    source_field_id=self.source_field_id,
                    region=region,
                    region_id=LOGICAL_REGION_IDS[region],
                    region_page_index=0,
                    logical_page_index=logical_page_index,
                    region_start=0,
                    region_end=0,
                    global_start=global_cursor,
                    global_end=global_cursor,
                    text="",
                    cells16=np.zeros((0, SUBSTRATE_WIDTH), dtype=np.float32),
                    addresses=(),
                    empty_region_marker=True,
                )
                logical_page_index += 1
                continue
            for page_index, start in enumerate(range(0, len(addresses), page_size)):
                chunk = addresses[start : start + page_size]
                cells = np.stack(
                    [
                        self.lane_cell16(address.row_index, address.lane_index)
                        for address in chunk
                    ],
                    axis=0,
                ).astype(np.float32, copy=False)
                yield D64CharacterPage(
                    source_field_id=self.source_field_id,
                    region=region,
                    region_id=LOGICAL_REGION_IDS[region],
                    region_page_index=page_index,
                    logical_page_index=logical_page_index,
                    region_start=start,
                    region_end=start + len(chunk),
                    global_start=chunk[0].global_position,
                    global_end=chunk[-1].global_position + 1,
                    text="".join(address.character for address in chunk),
                    cells16=cells,
                    addresses=chunk,
                )
                logical_page_index += 1
            global_cursor += len(addresses)


class D64FieldCompiler:
    """Compile exact attended SharedFieldSnapshot characters into D64 rows."""

    schema = D64_COMPILER_SCHEMA

    def compile(self, snapshot: SharedFieldSnapshot) -> CompiledD64Field:
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("D64FieldCompiler.compile requires SharedFieldSnapshot")
        if SLOT_DIM != SUBSTRATE_WIDTH:
            raise FieldCompilerError(
                f"frozen substrate width changed: expected 16, got {SLOT_DIM}"
            )

        rows: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        addresses: list[CanonicalCharAddress | None] = []
        cartography: list[CartographicSpan] = []
        region_row_ranges: list[tuple[str, int, int]] = []
        visited_regions: list[str] = []
        expected_active = 0
        global_position = 0

        for region in CANONICAL_REGION_ORDER:
            state = snapshot.region(region)
            visited_regions.append(region.value)
            region_row_start = len(rows)
            if state.visibility is not RegionVisibility.ATTENDED:
                region_row_ranges.append((region.value, region_row_start, region_row_start))
                continue

            text = state.text
            expected_active += len(text)
            cartography.extend(_cartography_for_region(region, text))
            region_cells: list[np.ndarray] = []
            region_addresses: list[CanonicalCharAddress] = []
            region_position = 0
            for span in state.spans:
                span_position = 0
                for character in span.text:
                    try:
                        assert_supported_text(character)
                        cell = np.asarray(char_to_slot(character), dtype=np.float32)
                    except Exception as exc:
                        raise UnsupportedActiveCharacterError(
                            "attended canonical character is not representable by the frozen "
                            f"16D substrate: region={region.value!r}, span={span.span_id!r}, "
                            f"region_position={region_position}, span_position={span_position}, "
                            f"character={character!r}"
                        ) from exc
                    if cell.shape != (SUBSTRATE_WIDTH,):
                        raise FieldCompilerError(
                            f"substrate returned invalid cell shape {cell.shape!r}"
                        )
                    region_cells.append(cell)
                    region_addresses.append(
                        CanonicalCharAddress(
                            region=region,
                            region_position=region_position,
                            global_position=global_position,
                            span_id=span.span_id,
                            span_position=span_position,
                            source=span.source,
                            provenance=span.provenance,
                            character=character,
                            row_index=-1,
                            lane_index=-1,
                        )
                    )
                    region_position += 1
                    global_position += 1
                    span_position += 1

            for start in range(0, len(region_cells), D64_LANES_PER_ROW):
                chunk_cells = region_cells[start : start + D64_LANES_PER_ROW]
                chunk_addresses = region_addresses[start : start + D64_LANES_PER_ROW]
                row = np.zeros((D64_WIDTH,), dtype=np.float32)
                valid = np.zeros((D64_LANES_PER_ROW,), dtype=np.bool_)
                row_index = len(rows)
                for lane, (cell, address) in enumerate(zip(chunk_cells, chunk_addresses)):
                    cell_start = lane * SUBSTRATE_WIDTH
                    row[cell_start : cell_start + SUBSTRATE_WIDTH] = cell
                    valid[lane] = True
                    addresses.append(
                        replace(address, row_index=row_index, lane_index=lane)
                    )
                addresses.extend(
                    [None] * (D64_LANES_PER_ROW - len(chunk_addresses))
                )
                rows.append(row)
                valid_masks.append(valid)

            region_row_ranges.append((region.value, region_row_start, len(rows)))

        rows_array = (
            np.stack(rows, axis=0).astype(np.float32, copy=False)
            if rows
            else np.zeros((0, D64_WIDTH), dtype=np.float32)
        )
        valid_array = (
            np.stack(valid_masks, axis=0).astype(np.bool_, copy=False)
            if valid_masks
            else np.zeros((0, D64_LANES_PER_ROW), dtype=np.bool_)
        )
        rows_sha = hashlib.sha256(rows_array.astype("<f4", copy=False).tobytes(order="C")).hexdigest()
        address_payload = [
            None if address is None else address.to_canonical_dict()
            for address in addresses
        ]
        address_sha = hashlib.sha256(canonical_json_bytes(address_payload)).hexdigest()
        valid_addresses = tuple(address for address in addresses if address is not None)
        compiled_count = len(valid_addresses)
        roundtrip_text = "".join(address.character for address in valid_addresses)
        roundtrip_sha = hashlib.sha256(roundtrip_text.encode("utf-8")).hexdigest()
        padding_lanes = len(addresses) - compiled_count

        coverage = D64CoverageManifest(
            schema=D64_COMPILER_SCHEMA,
            source_field_id=snapshot.field_id,
            source_tick_id=snapshot.tick_id,
            expected_regions=tuple(region.value for region in CANONICAL_REGION_ORDER),
            visited_regions=tuple(visited_regions),
            expected_active_characters=expected_active,
            compiled_active_characters=compiled_count,
            row_count=int(rows_array.shape[0]),
            valid_lanes=compiled_count,
            padding_lanes=padding_lanes,
            region_row_ranges=tuple(region_row_ranges),
            rows_sha256=rows_sha,
            address_sha256=address_sha,
            roundtrip_sha256=roundtrip_sha,
            complete=(
                tuple(visited_regions)
                == tuple(region.value for region in CANONICAL_REGION_ORDER)
                and compiled_count == expected_active
                and int(valid_array.sum()) == compiled_count
                and len(addresses) == rows_array.shape[0] * D64_LANES_PER_ROW
            ),
        )
        if not coverage.complete:
            raise IncompleteRailError("D64 compiler coverage proof is incomplete")

        rail_id = canonical_sha256(
            {
                "schema": D64_COMPILER_SCHEMA,
                "source_field_id": snapshot.field_id,
                "source_tick_id": snapshot.tick_id,
                "coverage": coverage.to_canonical_dict(),
                "cartography": [item.to_canonical_dict() for item in cartography],
            }
        )
        compiled = CompiledD64Field(
            source_field_id=snapshot.field_id,
            source_tick_id=snapshot.tick_id,
            rows=rows_array,
            lane_valid=valid_array,
            addresses=tuple(addresses),
            cartography=tuple(cartography),
            coverage=coverage,
            rail_id=rail_id,
        )
        compiled.verify_roundtrip(snapshot)
        _verify_vector_roundtrip(compiled)
        return compiled


def replacement_delta(
    snapshot: SharedFieldSnapshot,
    compiled: CompiledD64Field,
    *,
    region: LogicalRegion | str,
    text: str,
    author_core_id: str,
    pass_id: str | int,
    evidence: Iterable[str] = (),
    provenance: str = "d64_field_compiler_exact_replace",
) -> FieldDelta:
    """Construct an ordinary exact whole-region delta against a fresh rail."""

    compiled.assert_fresh(snapshot)
    logical = region if isinstance(region, LogicalRegion) else LogicalRegion(region)
    assert_supported_text(text)
    prior = snapshot.region(logical).text
    return FieldDelta(
        base_field_id=snapshot.field_id,
        base_tick_id=snapshot.tick_id,
        author_core_id=author_core_id,
        pass_id=pass_id,
        operations=(
            ReplaceText(
                region=logical,
                start=0,
                end=len(prior),
                text=text,
                provenance=provenance,
            ),
        ),
        evidence=tuple(evidence),
    )


def apply_compiled_delta(
    snapshot: SharedFieldSnapshot,
    compiled: CompiledD64Field,
    delta: FieldDelta,
) -> SharedFieldSnapshot:
    """Freshness-gate then delegate the commit mechanics to canonical delta code."""

    compiled.assert_fresh(snapshot)
    return apply_delta(snapshot, delta)


def _cartography_for_region(
    region: LogicalRegion,
    text: str,
) -> list[CartographicSpan]:
    result: list[CartographicSpan] = []

    # Source text itself is deterministic structure, not learned semantics.
    if text:
        result.append(_cartographic_span("region_text", region, 0, len(text), text))

    for match in re.finditer(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text):
        result.append(
            _cartographic_span("word", region, match.start(), match.end(), match.group(0))
        )

    sentence_start = 0
    for match in re.finditer(r"[.!?]+(?=\s|$)", text):
        end = match.end()
        start = sentence_start
        while start < end and text[start].isspace():
            start += 1
        if start < end:
            result.append(
                _cartographic_span("sentence", region, start, end, text[start:end])
            )
        sentence_start = end
    if sentence_start < len(text):
        start = sentence_start
        while start < len(text) and text[start].isspace():
            start += 1
        if start < len(text):
            result.append(
                _cartographic_span("sentence", region, start, len(text), text[start:])
            )

    paragraph_start = 0
    for match in re.finditer(r"\n\s*\n", text):
        end = match.start()
        if paragraph_start < end:
            result.append(
                _cartographic_span(
                    "paragraph", region, paragraph_start, end, text[paragraph_start:end]
                )
            )
        paragraph_start = match.end()
    if paragraph_start < len(text):
        result.append(
            _cartographic_span(
                "paragraph", region, paragraph_start, len(text), text[paragraph_start:]
            )
        )
    return result


def _cartographic_span(
    kind: str,
    region: LogicalRegion,
    start: int,
    end: int,
    text: str,
) -> CartographicSpan:
    return CartographicSpan(
        kind=kind,
        region=region,
        start=start,
        end=end,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _verify_vector_roundtrip(compiled: CompiledD64Field) -> None:
    bank = get_letter_bank()
    for region in CANONICAL_REGION_ORDER:
        addresses = compiled.region_addresses(region)
        if not addresses:
            continue
        cells: list[np.ndarray] = []
        for address in addresses:
            cells.append(
                compiled.lane_cell16(address.row_index, address.lane_index)
            )
        decoded = bank.decode_sequence(np.stack(cells, axis=0), blanks_as="")
        expected = "".join(address.character for address in addresses)
        if decoded != expected:
            raise IncompleteRailError(
                f"16D vector roundtrip mismatch in D64 rail region {region.value!r}"
            )


__all__ = [
    "D64_COMPILER_SCHEMA",
    "D64_WIDTH",
    "SUBSTRATE_WIDTH",
    "D64_LANES_PER_ROW",
    "FieldCompilerError",
    "UnsupportedActiveCharacterError",
    "IncompleteRailError",
    "StaleCompiledFieldError",
    "CanonicalCharAddress",
    "CartographicSpan",
    "D64CoverageManifest",
    "D64CharacterPage",
    "CompiledD64Field",
    "D64FieldCompiler",
    "replacement_delta",
    "apply_compiled_delta",
]

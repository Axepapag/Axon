"""Grounded first-form semantic slots over Axon's exact D64 rail.

Build D keeps two deliberately different surfaces bound to the same frozen
field image:

* ``CompiledD64Field`` remains the exact, lossless character scaffold and owns
  coverage/roundtrip guarantees.
* ``D64SemanticSurface`` is a derived, rebuildable set of word/sentence/
  paragraph/source-span slots.  Every slot points back to exact D64 lanes and
  canonical source spans.

The feature vectors in this module are deterministic structural/lexical
features, not a trained English embedding model.  They are intentionally
primitive first-form semantic tissue permitted by the Source of Truth.  A
future learned specialist may replace or augment this feature generation only
behind an explicit generation identity while exact grounding remains intact.

This module has no reasoning vote and no canonical commit authority.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .compiler_d64 import (
    D64_LANES_PER_ROW,
    D64_WIDTH,
    CartographicSpan,
    CompiledD64Field,
    StaleCompiledFieldError,
)
from .schema import (
    LOGICAL_REGION_IDS,
    FieldSpan,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)

D64_SEMANTIC_SURFACE_SCHEMA = "axon-d64-semantic-surface-v1"
D64_SEMANTIC_SLOT_SCHEMA = "axon-d64-semantic-slot-v1"
D64_STRUCTURAL_FEATURE_SCHEMA = "axon-d64-structural-features-v1"
D64_STRUCTURAL_FEATURE_GENERATION = "structural-lexical-v1"

_TOKEN_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", flags=re.UNICODE)
_SUPPORTED_CARTOGRAPHY_KINDS = frozenset({"word", "sentence", "paragraph"})


class D64SemanticSurfaceError(RuntimeError):
    """Base class for grounded D64 semantic-surface failures."""


class D64SemanticBindingError(D64SemanticSurfaceError):
    """A semantic slot/surface is not grounded in the exact rail it claims."""


class StaleD64SemanticSurfaceError(D64SemanticSurfaceError):
    """A semantic surface is being used against another field/rail generation."""


def _features_sha256(features: np.ndarray) -> str:
    values = np.asarray(features, dtype="<f4")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _feature_bucket(label: str) -> tuple[int, float]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little", signed=False) % D64_WIDTH
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def _add_feature(vector: np.ndarray, label: str, weight: float) -> None:
    index, sign = _feature_bucket(label)
    vector[index] += np.float32(sign * float(weight))


def structural_features64(
    text: str,
    *,
    region: LogicalRegion,
    slot_kind: str,
    source_kind: str = "",
) -> np.ndarray:
    """Return deterministic 64D structural/lexical features for one exact slot.

    This is deliberately not called an embedding.  It hashes auditable exact
    structure into a small feature space so first-form semantic slots have a
    stable D64 representation before any learned semantic specialist exists.
    """

    if not isinstance(text, str) or not text:
        raise ValueError("structural_features64 requires non-empty exact text")
    if not isinstance(region, LogicalRegion):
        raise TypeError("region must be LogicalRegion")
    if not isinstance(slot_kind, str) or not slot_kind:
        raise ValueError("slot_kind must be non-empty")
    if not isinstance(source_kind, str):
        raise TypeError("source_kind must be a string")

    vector = np.zeros((D64_WIDTH,), dtype=np.float32)
    normalized = _normalize_text(text)
    tokens = tuple(match.group(0) for match in _TOKEN_RE.finditer(normalized))

    _add_feature(vector, f"region:{region.value}", 0.50)
    _add_feature(vector, f"slot-kind:{slot_kind}", 0.75)
    if source_kind:
        _add_feature(vector, f"source-kind:{source_kind.casefold()}", 0.50)

    # Exact lexical content.  Repetition is bounded by logarithmic weighting so
    # long paragraphs do not dominate merely by repeating one word many times.
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    for token, count in sorted(counts.items()):
        _add_feature(vector, f"token:{token}", 1.0 + math.log1p(count - 1))
        padded = f"^{token}$"
        for size, weight in ((3, 0.20), (4, 0.12)):
            if len(padded) < size:
                _add_feature(vector, f"chargram:{size}:{padded}", weight)
                continue
            for start in range(0, len(padded) - size + 1):
                _add_feature(
                    vector,
                    f"chargram:{size}:{padded[start:start + size]}",
                    weight,
                )

    for left, right in zip(tokens, tokens[1:]):
        _add_feature(vector, f"token-bigram:{left}|{right}", 0.35)

    # Coarse structural scale is exact metadata, not learned semantics.
    char_bucket = min(15, int(math.log2(max(1, len(text)))))
    token_bucket = min(15, int(math.log2(max(1, len(tokens)))))
    _add_feature(vector, f"char-length-bucket:{char_bucket}", 0.20)
    _add_feature(vector, f"token-count-bucket:{token_bucket}", 0.20)

    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= np.float32(norm)
    if not np.isfinite(vector).all():
        raise D64SemanticSurfaceError("structural feature generation produced non-finite values")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, slots=True)
class D64SemanticSlot:
    """One derived semantic-scale slot grounded in exact canonical characters."""

    source_field_id: str
    source_tick_id: int
    source_rail_id: str
    feature_generation: str
    region: LogicalRegion
    slot_kind: str
    source_kind: str
    region_start: int
    region_end: int
    text_sha256: str
    exact_lane_refs: tuple[int, ...]
    source_span_ids: tuple[str, ...]
    container_refs: tuple[str, ...]
    edge_refs: tuple[str, ...]
    features64: np.ndarray
    slot_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("source_field_id", "source_rail_id", "feature_generation", "slot_kind", "text_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.source_kind, str):
            raise TypeError("source_kind must be a string")
        if isinstance(self.source_tick_id, bool) or not isinstance(self.source_tick_id, int) or self.source_tick_id < 0:
            raise ValueError("source_tick_id must be a non-negative integer")
        if not isinstance(self.region, LogicalRegion):
            raise TypeError("region must be LogicalRegion")
        if isinstance(self.region_start, bool) or not isinstance(self.region_start, int) or self.region_start < 0:
            raise ValueError("region_start must be a non-negative integer")
        if isinstance(self.region_end, bool) or not isinstance(self.region_end, int) or self.region_end <= self.region_start:
            raise ValueError("region_end must be an integer greater than region_start")
        if len(self.text_sha256) != 64:
            raise ValueError("text_sha256 must be a 64-character digest")
        try:
            int(self.text_sha256, 16)
        except ValueError as exc:
            raise ValueError("text_sha256 must be hexadecimal") from exc

        lane_refs = tuple(int(item) for item in self.exact_lane_refs)
        if len(lane_refs) != self.region_end - self.region_start:
            raise D64SemanticBindingError("semantic slot lane count must equal its exact character span length")
        if any(item < 0 for item in lane_refs) or any(right <= left for left, right in zip(lane_refs, lane_refs[1:])):
            raise D64SemanticBindingError("semantic slot exact_lane_refs must be strictly increasing non-negative indices")
        object.__setattr__(self, "exact_lane_refs", lane_refs)

        for name in ("source_span_ids", "container_refs", "edge_refs"):
            values = tuple(sorted(set(str(item) for item in getattr(self, name))))
            if any(not item for item in values):
                raise ValueError(f"{name} may not contain empty values")
            object.__setattr__(self, name, values)
        if not self.source_span_ids:
            raise D64SemanticBindingError("semantic slot requires at least one source_span_id")

        features = np.asarray(self.features64, dtype=np.float32)
        if features.shape != (D64_WIDTH,):
            raise ValueError("D64SemanticSlot.features64 must have shape [64]")
        if not np.isfinite(features).all():
            raise ValueError("D64SemanticSlot.features64 must contain only finite values")
        features.setflags(write=False)
        object.__setattr__(self, "features64", features)
        object.__setattr__(self, "slot_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def features_sha256(self) -> str:
        return _features_sha256(self.features64)

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": D64_SEMANTIC_SLOT_SCHEMA,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "source_rail_id": self.source_rail_id,
            "feature_schema": D64_STRUCTURAL_FEATURE_SCHEMA,
            "feature_generation": self.feature_generation,
            "region": self.region.value,
            "slot_kind": self.slot_kind,
            "source_kind": self.source_kind,
            "region_start": self.region_start,
            "region_end": self.region_end,
            "text_sha256": self.text_sha256,
            "exact_lane_refs": list(self.exact_lane_refs),
            "source_span_ids": list(self.source_span_ids),
            "container_refs": list(self.container_refs),
            "edge_refs": list(self.edge_refs),
            "features_sha256": self.features_sha256,
        }
        if include_id:
            payload["slot_id"] = self.slot_id
        return payload


@dataclass(frozen=True, slots=True)
class D64SemanticSurface:
    """Immutable derived semantic-slot surface bound to one exact D64 rail."""

    source_field_id: str
    source_tick_id: int
    source_rail_id: str
    feature_generation: str
    slots: tuple[D64SemanticSlot, ...]
    surface_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("source_field_id", "source_rail_id", "feature_generation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.source_tick_id, bool) or not isinstance(self.source_tick_id, int) or self.source_tick_id < 0:
            raise ValueError("source_tick_id must be a non-negative integer")
        slots = tuple(self.slots)
        if not all(isinstance(item, D64SemanticSlot) for item in slots):
            raise TypeError("slots must contain D64SemanticSlot values")
        expected_order = tuple(
            sorted(
                slots,
                key=lambda item: (
                    LOGICAL_REGION_IDS[item.region],
                    item.region_start,
                    item.region_end,
                    item.slot_kind,
                    item.source_kind,
                    item.slot_id,
                ),
            )
        )
        if [item.slot_id for item in slots] != [item.slot_id for item in expected_order]:
            raise D64SemanticBindingError("semantic slots must be in deterministic canonical order")
        slot_ids = [item.slot_id for item in slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise D64SemanticBindingError("semantic surface contains duplicate slot identities")
        for item in slots:
            if (
                item.source_field_id != self.source_field_id
                or item.source_tick_id != self.source_tick_id
                or item.source_rail_id != self.source_rail_id
                or item.feature_generation != self.feature_generation
            ):
                raise D64SemanticBindingError("semantic slot does not match its containing surface binding")
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "surface_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": D64_SEMANTIC_SURFACE_SCHEMA,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "source_rail_id": self.source_rail_id,
            "feature_schema": D64_STRUCTURAL_FEATURE_SCHEMA,
            "feature_generation": self.feature_generation,
            "slots": [item.to_canonical_dict() for item in self.slots],
        }
        if include_id:
            payload["surface_id"] = self.surface_id
        return payload

    def assert_fresh(
        self,
        snapshot: SharedFieldSnapshot,
        exact: CompiledD64Field,
    ) -> None:
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("snapshot must be SharedFieldSnapshot")
        if not isinstance(exact, CompiledD64Field):
            raise TypeError("exact must be CompiledD64Field")
        exact.assert_fresh(snapshot)
        if (
            self.source_field_id != snapshot.field_id
            or self.source_tick_id != snapshot.tick_id
            or self.source_rail_id != exact.rail_id
        ):
            raise StaleD64SemanticSurfaceError("semantic surface is stale for the supplied exact field rail")

    def slot_text(self, slot: D64SemanticSlot, exact: CompiledD64Field) -> str:
        if not isinstance(slot, D64SemanticSlot):
            raise TypeError("slot must be D64SemanticSlot")
        if all(item.slot_id != slot.slot_id for item in self.slots):
            raise KeyError(slot.slot_id)
        if exact.rail_id != self.source_rail_id:
            raise StaleD64SemanticSurfaceError("exact rail does not match semantic surface")
        characters: list[str] = []
        for flat_ref in slot.exact_lane_refs:
            row, lane = divmod(flat_ref, D64_LANES_PER_ROW)
            address = exact.address(row, lane)
            if address is None:
                raise D64SemanticBindingError("semantic slot points at D64 padding")
            characters.append(address.character)
        return "".join(characters)

    def verify_grounding(self, snapshot: SharedFieldSnapshot, exact: CompiledD64Field) -> None:
        self.assert_fresh(snapshot, exact)
        span_maps: dict[LogicalRegion, dict[str, FieldSpan]] = {
            region.name: {span.span_id: span for span in region.spans}
            for region in snapshot.regions
        }
        for slot in self.slots:
            text = self.slot_text(slot, exact)
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != slot.text_sha256:
                raise D64SemanticBindingError(f"semantic slot {slot.slot_id} exact text hash mismatch")
            expected_positions = list(range(slot.region_start, slot.region_end))
            observed_positions: list[int] = []
            observed_span_ids: set[str] = set()
            for flat_ref in slot.exact_lane_refs:
                row, lane = divmod(flat_ref, D64_LANES_PER_ROW)
                address = exact.address(row, lane)
                if address is None or address.region is not slot.region:
                    raise D64SemanticBindingError(f"semantic slot {slot.slot_id} crosses an exact rail region/padding boundary")
                observed_positions.append(address.region_position)
                observed_span_ids.add(address.span_id)
            if observed_positions != expected_positions:
                raise D64SemanticBindingError(f"semantic slot {slot.slot_id} does not cover its declared contiguous exact span")
            if observed_span_ids != set(slot.source_span_ids):
                raise D64SemanticBindingError(f"semantic slot {slot.slot_id} source span references do not match exact addresses")
            known = span_maps[slot.region]
            if any(span_id not in known for span_id in slot.source_span_ids):
                raise D64SemanticBindingError(f"semantic slot {slot.slot_id} references a missing canonical source span")
            expected_container_refs: set[str] = set()
            expected_edge_refs: set[str] = set()
            for span_id in slot.source_span_ids:
                source_span = known[span_id]
                expected_container_refs.update(source_span.container_refs)
                expected_edge_refs.update(source_span.edge_refs)
            if set(slot.container_refs) != expected_container_refs:
                raise D64SemanticBindingError(
                    f"semantic slot {slot.slot_id} container refs do not match grounded source spans"
                )
            if set(slot.edge_refs) != expected_edge_refs:
                raise D64SemanticBindingError(
                    f"semantic slot {slot.slot_id} edge refs do not match grounded source spans"
                )
            if slot.slot_kind == "field_span":
                if len(slot.source_span_ids) != 1:
                    raise D64SemanticBindingError("field_span semantic slot must bind exactly one canonical FieldSpan")
                if slot.source_kind != known[slot.source_span_ids[0]].kind:
                    raise D64SemanticBindingError("field_span semantic slot source_kind mismatch")
            elif slot.source_kind:
                raise D64SemanticBindingError("cartographic semantic slots may not invent source_kind")
            expected_features = structural_features64(
                text,
                region=slot.region,
                slot_kind=slot.slot_kind,
                source_kind=slot.source_kind,
            )
            if not np.array_equal(expected_features, slot.features64):
                raise D64SemanticBindingError(
                    f"semantic slot {slot.slot_id} features are not the deterministic grounded generation"
                )


@dataclass(frozen=True, slots=True)
class CompiledD64DualSurface:
    """One exact D64 rail plus its grounded derived semantic-slot surface."""

    exact: CompiledD64Field
    semantic: D64SemanticSurface
    dual_surface_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.exact, CompiledD64Field):
            raise TypeError("exact must be CompiledD64Field")
        if not isinstance(self.semantic, D64SemanticSurface):
            raise TypeError("semantic must be D64SemanticSurface")
        if (
            self.semantic.source_field_id != self.exact.source_field_id
            or self.semantic.source_tick_id != self.exact.source_tick_id
            or self.semantic.source_rail_id != self.exact.rail_id
        ):
            raise D64SemanticBindingError("exact and semantic surfaces are not bound to the same D64 rail")
        object.__setattr__(
            self,
            "dual_surface_id",
            canonical_sha256(
                {
                    "schema": "axon-d64-dual-surface-v1",
                    "source_field_id": self.exact.source_field_id,
                    "source_tick_id": self.exact.source_tick_id,
                    "exact_rail_id": self.exact.rail_id,
                    "semantic_surface_id": self.semantic.surface_id,
                    "feature_generation": self.semantic.feature_generation,
                }
            ),
        )

    def assert_fresh(self, snapshot: SharedFieldSnapshot) -> None:
        self.exact.assert_fresh(snapshot)
        self.semantic.verify_grounding(snapshot, self.exact)


class D64SemanticSurfaceCompiler:
    """Derive deterministic grounded semantic slots from one exact D64 rail."""

    feature_generation = D64_STRUCTURAL_FEATURE_GENERATION

    def compile(
        self,
        snapshot: SharedFieldSnapshot,
        exact: CompiledD64Field,
    ) -> D64SemanticSurface:
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("snapshot must be SharedFieldSnapshot")
        if not isinstance(exact, CompiledD64Field):
            raise TypeError("exact must be CompiledD64Field")
        exact.assert_fresh(snapshot)

        address_by_region_position: dict[tuple[LogicalRegion, int], int] = {}
        span_by_region_id: dict[tuple[LogicalRegion, str], FieldSpan] = {}
        for flat_ref, address in enumerate(exact.addresses):
            if address is None:
                continue
            key = (address.region, address.region_position)
            if key in address_by_region_position:
                raise D64SemanticBindingError("exact rail contains duplicate region-position addresses")
            address_by_region_position[key] = flat_ref
        for region in snapshot.regions:
            for span in region.spans:
                span_by_region_id[(region.name, span.span_id)] = span

        slots: list[D64SemanticSlot] = []
        seen_keys: set[tuple[LogicalRegion, int, int, str, str]] = set()

        for structural in exact.cartography:
            if structural.kind not in _SUPPORTED_CARTOGRAPHY_KINDS:
                continue
            slot = self._slot_for_range(
                snapshot,
                exact,
                address_by_region_position,
                span_by_region_id,
                region=structural.region,
                start=structural.start,
                end=structural.end,
                slot_kind=structural.kind,
                source_kind="",
                expected_text_sha256=structural.text_sha256,
            )
            if slot is not None:
                key = (slot.region, slot.region_start, slot.region_end, slot.slot_kind, slot.source_kind)
                if key not in seen_keys:
                    slots.append(slot)
                    seen_keys.add(key)

        # Canonical FieldSpan boundaries are also exact structural objects.  They
        # provide phrase/fact/procedure/container-scale slots and carry dormant
        # container/edge references forward without inventing semantic labels.
        for region in snapshot.regions:
            offset = 0
            for span in region.spans:
                start = offset
                end = offset + len(span.text)
                offset = end
                if not span.text.strip() or span.kind == "evidence_separator":
                    continue
                slot = self._slot_for_range(
                    snapshot,
                    exact,
                    address_by_region_position,
                    span_by_region_id,
                    region=region.name,
                    start=start,
                    end=end,
                    slot_kind="field_span",
                    source_kind=span.kind,
                    expected_text_sha256=hashlib.sha256(span.text.encode("utf-8")).hexdigest(),
                )
                if slot is not None:
                    key = (slot.region, slot.region_start, slot.region_end, slot.slot_kind, slot.source_kind)
                    if key not in seen_keys:
                        slots.append(slot)
                        seen_keys.add(key)

        slots.sort(
            key=lambda item: (
                LOGICAL_REGION_IDS[item.region],
                item.region_start,
                item.region_end,
                item.slot_kind,
                item.source_kind,
                item.slot_id,
            )
        )
        surface = D64SemanticSurface(
            source_field_id=snapshot.field_id,
            source_tick_id=snapshot.tick_id,
            source_rail_id=exact.rail_id,
            feature_generation=self.feature_generation,
            slots=tuple(slots),
        )
        surface.verify_grounding(snapshot, exact)
        return surface

    def compile_dual(
        self,
        snapshot: SharedFieldSnapshot,
        exact: CompiledD64Field,
    ) -> CompiledD64DualSurface:
        semantic = self.compile(snapshot, exact)
        return CompiledD64DualSurface(exact=exact, semantic=semantic)

    def _slot_for_range(
        self,
        snapshot: SharedFieldSnapshot,
        exact: CompiledD64Field,
        address_by_region_position: dict[tuple[LogicalRegion, int], int],
        span_by_region_id: dict[tuple[LogicalRegion, str], FieldSpan],
        *,
        region: LogicalRegion,
        start: int,
        end: int,
        slot_kind: str,
        source_kind: str,
        expected_text_sha256: str,
    ) -> D64SemanticSlot | None:
        if start < 0 or end <= start:
            raise D64SemanticBindingError("semantic source range must satisfy 0 <= start < end")
        refs: list[int] = []
        source_span_ids: set[str] = set()
        characters: list[str] = []
        for position in range(start, end):
            flat_ref = address_by_region_position.get((region, position))
            if flat_ref is None:
                # A semantic slot may not partially bridge masked text.  The
                # exact canonical characters still exist, but this derived view
                # does not attend the complete structural object.
                return None
            row, lane = divmod(flat_ref, D64_LANES_PER_ROW)
            address = exact.address(row, lane)
            if address is None:
                raise D64SemanticBindingError("exact address map resolved to padding")
            refs.append(flat_ref)
            source_span_ids.add(address.span_id)
            characters.append(address.character)
        text = "".join(characters)
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_sha != expected_text_sha256:
            raise D64SemanticBindingError("structural slot text does not match the exact D64 source range")

        container_refs: set[str] = set()
        edge_refs: set[str] = set()
        for span_id in source_span_ids:
            span = span_by_region_id.get((region, span_id))
            if span is None:
                raise D64SemanticBindingError("exact rail address references a canonical span that does not exist")
            container_refs.update(span.container_refs)
            edge_refs.update(span.edge_refs)

        return D64SemanticSlot(
            source_field_id=snapshot.field_id,
            source_tick_id=snapshot.tick_id,
            source_rail_id=exact.rail_id,
            feature_generation=self.feature_generation,
            region=region,
            slot_kind=slot_kind,
            source_kind=source_kind,
            region_start=start,
            region_end=end,
            text_sha256=text_sha,
            exact_lane_refs=tuple(refs),
            source_span_ids=tuple(source_span_ids),
            container_refs=tuple(container_refs),
            edge_refs=tuple(edge_refs),
            features64=structural_features64(
                text,
                region=region,
                slot_kind=slot_kind,
                source_kind=source_kind,
            ),
        )


__all__ = [
    "D64_SEMANTIC_SURFACE_SCHEMA",
    "D64_SEMANTIC_SLOT_SCHEMA",
    "D64_STRUCTURAL_FEATURE_SCHEMA",
    "D64_STRUCTURAL_FEATURE_GENERATION",
    "D64SemanticSurfaceError",
    "D64SemanticBindingError",
    "StaleD64SemanticSurfaceError",
    "structural_features64",
    "D64SemanticSlot",
    "D64SemanticSurface",
    "CompiledD64DualSurface",
    "D64SemanticSurfaceCompiler",
]

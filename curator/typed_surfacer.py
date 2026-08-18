"""Typed, provenance-preserving surfacing from dormant records.

Dormant containers are evidence records, not an attended tensor.  This module
copies selected readable semantic edges into an active-field span without
flattening away their identity or provenance.  Any record/edge budget is
explicit and every omitted item is represented by a deterministic audit
record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Protocol, Sequence


STRUCTURED_KNOWLEDGE_REGION = "structured_knowledge"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SurfacedSpan:
    """One readable, provenance-bearing active-field span."""

    span_id: str
    text: str
    query: str
    source: str
    provenance: str
    confidence: float
    container_refs: tuple[str, ...]
    edge_refs: tuple[str, ...]
    why_surfaced: str
    region: str = STRUCTURED_KNOWLEDGE_REGION

    @property
    def canonical_sha256(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True)
class SurfacingOmission:
    """An explicit active-view omission; the dormant record remains intact."""

    reason: str
    container_ref: str
    omitted_count: int
    omitted_sha256: str


@dataclass(frozen=True)
class SurfacingResult:
    query: str
    spans: tuple[SurfacedSpan, ...]
    omissions: tuple[SurfacingOmission, ...]

    @property
    def canonical_sha256(self) -> str:
        return _sha256_json(
            {
                "query": self.query,
                "spans": [asdict(span) for span in self.spans],
                "omissions": [asdict(item) for item in self.omissions],
            }
        )


class Searcher(Protocol):
    def search(self, query: str, k: int | None = None) -> Sequence[Mapping[str, Any]]:
        """Return dormant records in deterministic relevance order."""


def _container_ref(record: Mapping[str, Any], index: int) -> str:
    for key in ("container_id", "id", "record_id", "word"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return f"record-{index:08d}-{_sha256_json(record)[:16]}"


def _edge_parts(edge: Any) -> tuple[str, str, str]:
    """Return ``(edge_type, target, edge_ref)`` for supported record shapes."""

    if isinstance(edge, Mapping):
        edge_type = edge.get("edge_type", edge.get("type", edge.get("relation", "")))
        target = edge.get("target", edge.get("object", edge.get("value", "")))
        edge_ref = edge.get("edge_id", edge.get("id", ""))
    elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
        edge_type, target = edge[0], edge[1]
        edge_ref = edge[2] if len(edge) >= 3 else ""
    else:
        raise ValueError(f"unsupported semantic edge shape: {type(edge).__name__}")

    edge_type_text = "" if edge_type is None else str(edge_type)
    target_text = "" if target is None else str(target)
    if not edge_type_text.strip() or not target_text.strip():
        raise ValueError("semantic edges require non-empty edge type and target")
    return edge_type_text, target_text, "" if edge_ref is None else str(edge_ref)


def _record_edges(record: Mapping[str, Any]) -> list[Any]:
    value = record.get("edges", record.get("semantic_edges", ()))
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("record edges must be a sequence")
    return list(value)


def render_record(
    record: Mapping[str, Any],
    *,
    container_ref: str,
    edge_limit: int | None = None,
) -> tuple[str, tuple[str, ...], SurfacingOmission | None]:
    """Render every admitted semantic edge as readable exact text.

    ``edge_limit=None`` means no cap.  A supplied cap is an active-view budget,
    never a mutation of the dormant record.
    """

    if edge_limit is not None and edge_limit < 0:
        raise ValueError("edge_limit must be non-negative or None")

    subject_value = record.get(
        "word",
        record.get("text", record.get("letters", record.get("subject", ""))),
    )
    subject = "" if subject_value is None else str(subject_value)
    edges = _record_edges(record)
    admitted = edges if edge_limit is None else edges[:edge_limit]
    omitted = edges[len(admitted) :]

    rendered: list[str] = []
    edge_refs: list[str] = []
    for edge_index, edge in enumerate(admitted):
        edge_type, target, explicit_ref = _edge_parts(edge)
        rendered.append(" ".join(part for part in (subject, edge_type, target) if part))
        edge_refs.append(explicit_ref or f"{container_ref}:edge:{edge_index}")

    if not rendered and subject:
        rendered.append(subject)

    omission = None
    if omitted:
        omission = SurfacingOmission(
            reason="explicit_edge_budget",
            container_ref=container_ref,
            omitted_count=len(omitted),
            omitted_sha256=_sha256_json(omitted),
        )
    return "\n".join(rendered), tuple(edge_refs), omission


def surface_records(
    records: Iterable[Mapping[str, Any]],
    *,
    query: str,
    record_limit: int | None = None,
    edge_limit_per_record: int | None = None,
) -> SurfacingResult:
    """Copy ranked dormant records into typed active-field spans."""

    if record_limit is not None and record_limit < 0:
        raise ValueError("record_limit must be non-negative or None")

    materialized = list(records)
    admitted = materialized if record_limit is None else materialized[:record_limit]
    omissions: list[SurfacingOmission] = []
    if len(admitted) < len(materialized):
        hidden = materialized[len(admitted) :]
        omissions.append(
            SurfacingOmission(
                reason="explicit_record_budget",
                container_ref="*",
                omitted_count=len(hidden),
                omitted_sha256=_sha256_json(hidden),
            )
        )

    spans: list[SurfacedSpan] = []
    for index, record in enumerate(admitted):
        if not isinstance(record, Mapping):
            raise ValueError(f"dormant record {index} is not a mapping")
        container_ref = _container_ref(record, index)
        text, edge_refs, edge_omission = render_record(
            record,
            container_ref=container_ref,
            edge_limit=edge_limit_per_record,
        )
        if edge_omission is not None:
            omissions.append(edge_omission)
        if not text:
            continue

        source = str(record.get("source", ""))
        provenance_value = record.get("provenance", source)
        provenance = _canonical_json(provenance_value)
        try:
            confidence = float(record.get("confidence", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{container_ref}: confidence is not numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{container_ref}: confidence must be in [0, 1]")

        span_payload = {
            "container_ref": container_ref,
            "query": query,
            "text": text,
            "source": source,
            "provenance": provenance,
            "edge_refs": edge_refs,
        }
        spans.append(
            SurfacedSpan(
                span_id=f"surfaced-{_sha256_json(span_payload)[:24]}",
                text=text,
                query=query,
                source=source,
                provenance=provenance,
                confidence=confidence,
                container_refs=(container_ref,),
                edge_refs=edge_refs,
                why_surfaced=f"retrieved for query {_canonical_json(query)}",
            )
        )

    return SurfacingResult(query=query, spans=tuple(spans), omissions=tuple(omissions))


def surface_query(
    searcher: Searcher,
    query: str,
    *,
    record_limit: int | None = None,
    edge_limit_per_record: int | None = None,
) -> SurfacingResult:
    """Search dormant state, then surface the selected records with provenance."""

    hits = searcher.search(query, k=record_limit)
    return surface_records(
        hits,
        query=query,
        # The search itself already applied the explicit record budget.
        record_limit=None,
        edge_limit_per_record=edge_limit_per_record,
    )


def to_field_spans(result: SurfacingResult) -> tuple[Any, ...]:
    """Convert surfaced evidence to canonical shared-field spans.

    The import is intentionally local so dormant curation stays usable without
    importing the runtime package.  Canonical Unicode is preserved here; the
    bounded 16D view compiler records any unsupported characters as omissions.
    """

    from runtime.field import FieldSpan

    return tuple(
        FieldSpan(
            span_id=span.span_id,
            text=span.text,
            kind="surfaced_knowledge",
            source=span.source,
            provenance=_canonical_json(
                {
                    "source_provenance": span.provenance,
                    "query": span.query,
                    "why_surfaced": span.why_surfaced,
                    "surfacing_sha256": span.canonical_sha256,
                }
            ),
            confidence=span.confidence,
            container_refs=span.container_refs,
            edge_refs=span.edge_refs,
        )
        for span in result.spans
    )


def to_structured_knowledge_region(result: SurfacingResult) -> Any:
    """Build the immutable active ``structured_knowledge`` region."""

    from runtime.field import LogicalRegion, RegionState

    return RegionState(
        name=LogicalRegion.STRUCTURED_KNOWLEDGE,
        spans=to_field_spans(result),
    )

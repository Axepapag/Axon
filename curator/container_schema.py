#!/usr/bin/env python3
"""container_schema.py - the formal container contract (Layer 1 + Layer 2 edge).

This is the first formal implementation of the container schema contract from
SOURCE_OF_TRUTH.md Layer 1 (Containers) and Layer 2 (Semantic Edges And Symbols).

Doctrine:
    - Containers are structured, auditable records -- not token embeddings.
    - The 16D substrate remains the letter/symbol storage layer. Semantic
      meaning is carried by semantic edges and registered symbols attached to
      containers, NOT by the 16D letter geometry.
    - Containers must not become loose bags of whatever a trainer happened to
      emit. normalise_container() coerces existing loose dataset records into
      this typed schema.

This module is deliberately dependency-free (no torch, no numpy). It works with
plain Python dicts, enums, and dataclasses so it can be used by kg_search,
cold_read, the semantic core, and the collaboration bus equally.

Canonical container fields per SOURCE_OF_TRUTH Layer 1:
    id, kind, letters, text, normalized_text, spans, edges, symbols, source,
    source_tick, created_tick, updated_tick, confidence, provenance, status

The exact schema can evolve, but containers must stay structured and auditable.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContainerStatus(str, Enum):
    """Lifecycle status of a container."""
    DRAFT = "draft"             # just created, not yet validated
    ACTIVE = "active"            # validated, in active use
    DORMANT = "dormant"          # stored in dormant structured knowledge, masked
    SUPERSEDED = "superseded"    # replaced by a newer revision
    RETRACTED = "retracted"      # withdrawn / flagged as wrong


class EdgeType(str, Enum):
    """Common semantic edge relation types.

    This is not an exhaustive enumeration -- the semantic core may introduce new
    edge types. The enum provides the canonical names for the most common ones
    so that downstream consumers can pattern-match reliably.
    """
    IS_A = "is a"
    PART_OF = "part of"
    CONTAINS = "contains"
    USES = "uses"
    LOCATED_IN = "located in"
    CONFIGURES = "configures"
    DESCRIBED_BY = "described by"     # legacy noise relation; kg_search skips it
    CUSTOM = "custom"                 # catch-all for unrecognised edge types


# ---------------------------------------------------------------------------
# SemanticEdge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticEdge:
    """A single semantic edge from a container to a target entity.

    Attributes:
        edge_type: the relation type (e.g. "is a", "uses").
        target: the target entity text (the hub/neighbor).
        symbol: optional registered semantic symbol (e.g. "AA") if assigned.
        confidence: float in [0, 1]. Defaults to 1.0 for hand-authored edges.
        provenance: short string describing where the edge came from.
        status: ContainerStatus-like lifecycle marker.
    """
    edge_type: str
    target: str
    symbol: str | None = None
    confidence: float = 1.0
    provenance: str = ""
    status: str = "active"

    def __post_init__(self):
        if not isinstance(self.edge_type, str) or not self.edge_type.strip():
            raise ValueError(f"SemanticEdge.edge_type must be a non-empty str, got {self.edge_type!r}")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError(f"SemanticEdge.target must be a non-empty str, got {self.target!r}")
        if self.symbol is not None and not isinstance(self.symbol, str):
            raise TypeError(f"SemanticEdge.symbol must be str or None, got {type(self.symbol).__name__}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"SemanticEdge.confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe plain dict."""
        return {
            "edge_type": self.edge_type,
            "target": self.target,
            "symbol": self.symbol,
            "confidence": float(self.confidence),
            "provenance": self.provenance,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SemanticEdge:
        """Deserialize from a plain dict (tolerant: extra keys ignored)."""
        return cls(
            edge_type=str(d.get("edge_type", d.get("type", ""))).strip(),
            target=str(d.get("target", "")).strip(),
            symbol=d.get("symbol"),
            confidence=float(d.get("confidence", 1.0)),
            provenance=str(d.get("provenance", "")),
            status=str(d.get("status", "active")),
        )

    def render(self) -> str:
        """Render in the kg_search / structured_knowledge style:
        'edge_type target' or 'edge_type target [symbol]' if a symbol is attached.
        """
        sym = f" [{self.symbol}]" if self.symbol else ""
        return f"{self.edge_type} {self.target}{sym}"


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class Container:
    """A structured, auditable container record.

    This is the typed replacement for the loose {word, chars, edges,
    category, definition} dicts currently in containers.jsonl. It adds
    container_id, provenance, status, normalised text, symbols, and lifecycle
    metadata.

    Conceptual shape:
        (<[d][o][g]>{AA}{K9}{...})

    The container/envelope is the PyTorch dictionary-style record. The
    word/letter payload remains separate from attached semantic symbols/edges.
    The materializer maps individual characters, not whole words or whole
    containers, into 16D substrate slots.

    Substrate discipline:
        - The ensemble must see every materialized letter.
        - Symbols are additive overlays for registered semantic edges; they do
          not replace the letters of the word, edge type, or edge target.
        - Layout/index metadata must not be treated as ensemble-visible
          container symbols unless it is promoted to a real semantic edge.

    Attributes:
        container_id: stable unique identifier (auto-generated if not given).
        kind: container kind / category (e.g. "entity", "concept", "fact").
        text: the primary text/label (e.g. "Axon", ".env").
        normalized_text: lowercased / stripped text used for matching.
        letters: the 16D-substrate letter sequence, stored as a raw string.
            The materializer must encode this character-by-character through
            substrate.py; this field is not a whole-word or whole-container
            embedding.
        edges: list of SemanticEdge records. Edges stay separate from the
            word/letter payload and receive their own registered symbols.
        symbols: list of registered semantic symbol codes attached through
            actual SemanticEdge records. This field must not contain layout
            buckets or other non-edge additions.
        source: provenance -- where the container came from.
        source_tick: the tick at which the container was captured/created.
        created_tick: tick when this container was first created.
        updated_tick: tick when this container was last updated.
        confidence: float in [0, 1]. Defaults to 1.0.
        provenance: free-text provenance note (distinct from source).
        status: ContainerStatus value.
        metadata: open-ended dict for extra structured fields (e.g. definition).
    """
    container_id: str = ""
    kind: str = "entity"
    text: str = ""
    normalized_text: str = ""
    letters: str = ""
    edges: list[SemanticEdge] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    source: str = ""
    source_tick: int = -1
    created_tick: int = -1
    updated_tick: int = -1
    confidence: float = 1.0
    provenance: str = ""
    status: ContainerStatus = ContainerStatus.DRAFT
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.container_id:
            self.container_id = f"c-{uuid.uuid4().hex[:12]}"
        if not self.normalized_text:
            self.normalized_text = self.text.lower().strip()
        if not self.letters and self.text:
            self.letters = self.text
        if isinstance(self.status, str):
            self.status = ContainerStatus(self.status)
        # Coerce loose edge dicts to SemanticEdge
        coerced: list[SemanticEdge] = []
        for e in self.edges:
            if isinstance(e, SemanticEdge):
                coerced.append(e)
            elif isinstance(e, dict):
                coerced.append(SemanticEdge.from_dict(e))
            elif isinstance(e, (list, tuple)) and len(e) == 2:
                coerced.append(SemanticEdge(edge_type=str(e[0]), target=str(e[1])))
            else:
                raise TypeError(f"Cannot coerce edge {e!r} to SemanticEdge")
        self.edges = coerced
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"Container.confidence must be in [0, 1], got {self.confidence}")

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe plain dict."""
        return {
            "container_id": self.container_id,
            "kind": self.kind,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "letters": self.letters,
            "edges": [e.to_dict() for e in self.edges],
            "symbols": list(self.symbols),
            "source": self.source,
            "source_tick": self.source_tick,
            "created_tick": self.created_tick,
            "updated_tick": self.updated_tick,
            "confidence": float(self.confidence),
            "provenance": self.provenance,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Container:
        """Deserialize from a plain dict (tolerant: missing keys use defaults).

        Accepts both the formal schema keys and the legacy {word, chars,
        edges, category, definition} keys from containers.jsonl.
        """
        edges_raw = d.get("edges", [])
        edges: list[SemanticEdge] = []
        for e in edges_raw:
            if isinstance(e, SemanticEdge):
                edges.append(e)
            elif isinstance(e, dict):
                edges.append(SemanticEdge.from_dict(e))
            elif isinstance(e, (list, tuple)) and len(e) == 2:
                edges.append(SemanticEdge(edge_type=str(e[0]), target=str(e[1])))
        # Collect metadata from non-canonical keys
        canonical_keys = {
            "container_id", "kind", "text", "normalized_text", "letters",
            "edges", "symbols", "source", "source_tick", "created_tick",
            "updated_tick", "confidence", "provenance", "status", "metadata",
            # legacy keys handled below
            "word", "chars", "category", "definition",
        }
        metadata: dict[str, Any] = dict(d.get("metadata", {}))
        for k, v in d.items():
            if k not in canonical_keys:
                metadata[k] = v
        if "definition" in d:
            metadata.setdefault("definition", d["definition"])
        if "category" in d:
            metadata.setdefault("category", d["category"])
        return cls(
            container_id=d.get("container_id", ""),
            kind=d.get("kind", d.get("category", "entity")),
            text=d.get("text", d.get("word", "")),
            normalized_text=d.get("normalized_text", d.get("chars", "")),
            letters=d.get("letters", d.get("chars", d.get("word", ""))),
            edges=edges,
            symbols=list(d.get("symbols", [])),
            source=d.get("source", ""),
            source_tick=int(d.get("source_tick", -1)),
            created_tick=int(d.get("created_tick", -1)),
            updated_tick=int(d.get("updated_tick", -1)),
            confidence=float(d.get("confidence", 1.0)),
            provenance=d.get("provenance", ""),
            status=d.get("status", ContainerStatus.DRAFT.value),
            metadata=metadata,
        )

    # --- rendering --------------------------------------------------------

    def render(self, max_edges: int | None = None) -> str:
        """Render in the kg_search / structured_knowledge style:
        'text: edge_type target; edge_type target ...'

        This is compatible with the existing kg_search.KGSearch.render format
        so that containers serialized via this schema can drop into the same
        structured_knowledge region rendering. By default, all edges render.
        Pass max_edges only for an explicit display or curriculum budget.
        """
        rendered_edges = [
            e.render() for e in self.edges
            if e.edge_type != EdgeType.DESCRIBED_BY.value
        ]
        if max_edges is not None:
            rendered_edges = rendered_edges[:max_edges]
        body = "; ".join(rendered_edges)
        return f"{self.text}: {body}" if body else self.text

    # --- lifecycle helpers ------------------------------------------------

    def update_status(self, new_status: ContainerStatus, tick: int = -1) -> None:
        """Transition the container to a new status and record the tick."""
        self.status = new_status
        if tick >= 0:
            self.updated_tick = tick

    def add_edge(self, edge: SemanticEdge) -> None:
        """Add a semantic edge to the container."""
        self.edges.append(edge)

    def add_symbol(self, symbol: str) -> None:
        """Attach a registered semantic symbol code to this container."""
        if symbol and symbol not in self.symbols:
            self.symbols.append(symbol)


# ---------------------------------------------------------------------------
# Normalization helpers (accept loose dataset records)
# ---------------------------------------------------------------------------

def normalize_container(raw: dict[str, Any],
                        source: str = "",
                        tick: int = -1) -> Container:
    """Normalize a loose dataset record into a typed Container.

    Accepts records shaped like:
        {"word": "Axon", "chars": "axon", "edges": [["is a", "agent"], ...],
         "category": "entity", "definition": "..."}

    Also accepts records already in the formal schema shape (with "text"
    instead of "word", etc.).
    """
    c = Container.from_dict(raw)
    if source and not c.source:
        c.source = source
    if tick >= 0 and c.source_tick < 0:
        c.source_tick = tick
    return c


def normalize_container_list(records: Sequence[dict[str, Any]],
                              source: str = "",
                              tick: int = -1) -> list[Container]:
    """Normalize a list of loose dataset records into typed Containers."""
    return [normalize_container(r, source=source, tick=tick) for r in records]


def containers_to_dicts(containers: Sequence[Container]) -> list[dict[str, Any]]:
    """Serialize a list of Containers back to plain dicts (JSON-safe)."""
    return [c.to_dict() for c in containers]


def container_to_jsonl(containers: Sequence[Container]) -> str:
    """Serialize a list of Containers as JSONL (one JSON object per line)."""
    return "\n".join(c.to_json() for c in containers)


# ---------------------------------------------------------------------------
# Convenience: load a JSONL file of containers (loose or formal)
# ---------------------------------------------------------------------------

def load_containers_jsonl(path: str,
                           source: str = "",
                           tick: int = -1) -> list[Container]:
    """Load containers from a JSONL file. Each line is a JSON object.

    Works with both the legacy loose format (word/chars/edges/category/
    definition) and the formal schema format.
    """
    containers: list[Container] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            containers.append(normalize_container(obj, source=source, tick=tick))
    return containers


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test: round-trip a legacy record through the schema
    raw = {
        "word": "Axon",
        "chars": "axon",
        "edges": [["is a", "agent"], ["uses", "16D substrate"], ["described by", "noise"]],
        "category": "entity",
        "definition": "Axon is a forever-ticking agent.",
    }
    c = normalize_container(raw, source="smoke-test", tick=42)
    print(f"container_id  = {c.container_id}")
    print(f"kind          = {c.kind}")
    print(f"text          = {c.text}")
    print(f"normalized    = {c.normalized_text}")
    print(f"status        = {c.status.value}")
    print(f"edges         = {len(c.edges)}")
    for e in c.edges:
        print(f"  - {e.render()}")
    print(f"render()      = {c.render()}")
    d = c.to_dict()
    print(f"to_dict keys  = {sorted(d.keys())}")
    c2 = Container.from_dict(d)
    assert c2.text == c.text
    assert c2.normalized_text == c.normalized_text
    assert len(c2.edges) == len(c.edges)
    assert c2.render() == c.render()
    print("\nPASS: round-trip OK")

"""Tests for container_schema.py - the first formal container contract.

These tests verify:
  - ContainerStatus enum values
  - SemanticEdge creation, validation, serialization, rendering
  - Container creation, normalization from loose records, serialization
  - Rendering compatible with kg_search / cortex style
  - Round-trip: loose dict -> Container -> dict -> Container
  - Edge coercion from legacy [etype, target] list form
  - Lifecycle helpers (update_status, add_edge, add_symbol)
"""
import json

import pytest

from container_schema import (  # noqa: E402
    Container,
    ContainerStatus,
    EdgeType,
    SemanticEdge,
    normalize_container,
    normalize_container_list,
    containers_to_dicts,
    container_to_jsonl,
    load_containers_jsonl,
)


# ---------------------------------------------------------------------------
# ContainerStatus
# ---------------------------------------------------------------------------

class TestContainerStatus:
    def test_enum_values(self):
        assert ContainerStatus.DRAFT.value == "draft"
        assert ContainerStatus.ACTIVE.value == "active"
        assert ContainerStatus.DORMANT.value == "dormant"
        assert ContainerStatus.SUPERSEDED.value == "superseded"
        assert ContainerStatus.RETRACTED.value == "retracted"

    def test_is_str_enum(self):
        assert isinstance(ContainerStatus.ACTIVE, str)
        assert ContainerStatus.ACTIVE == "active"


# ---------------------------------------------------------------------------
# SemanticEdge
# ---------------------------------------------------------------------------

class TestSemanticEdge:
    def test_basic_creation(self):
        e = SemanticEdge(edge_type="is a", target="animal")
        assert e.edge_type == "is a"
        assert e.target == "animal"
        assert e.symbol is None
        assert e.confidence == 1.0
        assert e.status == "active"

    def test_with_symbol(self):
        e = SemanticEdge(edge_type="is a", target="animal", symbol="AA")
        assert e.symbol == "AA"
        assert e.render() == "is a animal"

    def test_render_no_symbol(self):
        e = SemanticEdge(edge_type="uses", target="16D substrate")
        assert e.render() == "uses 16D substrate"

    def test_empty_edge_type_raises(self):
        with pytest.raises(ValueError):
            SemanticEdge(edge_type="", target="x")

    def test_empty_target_raises(self):
        with pytest.raises(ValueError):
            SemanticEdge(edge_type="is a", target="")

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SemanticEdge(edge_type="is a", target="x", confidence=2.0)
        with pytest.raises(ValueError):
            SemanticEdge(edge_type="is a", target="x", confidence=-0.1)

    def test_to_dict_from_dict_roundtrip(self):
        e = SemanticEdge(edge_type="part of", target="backend", symbol="BB",
                         confidence=0.8, provenance="manual")
        d = e.to_dict()
        assert d["edge_type"] == "part of"
        assert d["target"] == "backend"
        assert d["symbol"] == "BB"
        assert d["confidence"] == 0.8
        assert d["provenance"] == "manual"

        e2 = SemanticEdge.from_dict(d)
        assert e2.edge_type == e.edge_type
        assert e2.target == e.target
        assert e2.symbol == e.symbol
        assert e2.confidence == e.confidence
        assert e2.provenance == e.provenance

    def test_from_dict_tolerant(self):
        """from_dict should handle missing keys and legacy 'type' key."""
        e = SemanticEdge.from_dict({"type": "uses", "target": "tool"})
        assert e.edge_type == "uses"
        assert e.target == "tool"
        assert e.confidence == 1.0

    def test_from_dict_extra_keys_ignored(self):
        e = SemanticEdge.from_dict({"edge_type": "is a", "target": "x",
                                    "extra": "ignored"})
        assert e.edge_type == "is a"


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

class TestContainer:
    def test_basic_creation(self):
        c = Container(text="Axon", kind="entity")
        assert c.text == "Axon"
        assert c.kind == "entity"
        assert c.normalized_text == "axon"
        assert c.letters == "Axon"
        assert c.status == ContainerStatus.DRAFT
        assert c.container_id.startswith("c-")
        assert c.edges == []
        assert c.symbols == []

    def test_auto_id_generation(self):
        c1 = Container(text="a")
        c2 = Container(text="b")
        assert c1.container_id != c2.container_id

    def test_explicit_id_preserved(self):
        c = Container(container_id="my-id-123", text="x")
        assert c.container_id == "my-id-123"

    def test_explicit_normalized_text(self):
        c = Container(text="Axon", normalized_text="axon_v7")
        assert c.normalized_text == "axon_v7"

    def test_status_string_coerced_to_enum(self):
        c = Container(text="x", status="active")
        assert c.status == ContainerStatus.ACTIVE

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Container(text="x", confidence=1.5)
        with pytest.raises(ValueError):
            Container(text="x", confidence=-0.1)

    def test_edge_dict_coercion(self):
        c = Container(text="x", edges=[{"edge_type": "is a", "target": "y"}])
        assert len(c.edges) == 1
        assert isinstance(c.edges[0], SemanticEdge)
        assert c.edges[0].edge_type == "is a"
        assert c.edges[0].target == "y"

    def test_edge_list_tuple_coercion(self):
        c = Container(text="x", edges=[["is a", "animal"], ["uses", "tool"]])
        assert len(c.edges) == 2
        assert c.edges[0].edge_type == "is a"
        assert c.edges[0].target == "animal"
        assert c.edges[1].edge_type == "uses"

    def test_edge_semantic_edge_passthrough(self):
        e = SemanticEdge(edge_type="is a", target="animal")
        c = Container(text="x", edges=[e])
        assert c.edges[0] is e or c.edges[0] == e

    def test_to_dict_from_dict_roundtrip(self):
        c = Container(
            text="Axon",
            kind="entity",
            edges=[SemanticEdge("is a", "agent"), SemanticEdge("uses", "16D")],
            symbols=["AA"],
            source="test",
            source_tick=42,
            confidence=0.9,
            status=ContainerStatus.ACTIVE,
            metadata={"definition": "Axon is an agent."},
        )
        d = c.to_dict()
        # Verify all canonical fields are present
        expected_keys = {
            "container_id", "kind", "text", "normalized_text", "letters",
            "edges", "symbols", "source", "source_tick", "created_tick",
            "updated_tick", "confidence", "provenance", "status", "metadata",
        }
        assert set(d.keys()) == expected_keys
        assert d["text"] == "Axon"
        assert d["status"] == "active"
        assert d["confidence"] == 0.9
        assert len(d["edges"]) == 2
        assert d["metadata"]["definition"] == "Axon is an agent."

        c2 = Container.from_dict(d)
        assert c2.text == c.text
        assert c2.kind == c.kind
        assert c2.normalized_text == c.normalized_text
        assert len(c2.edges) == len(c.edges)
        assert c2.edges[0].edge_type == "is a"
        assert c2.status == ContainerStatus.ACTIVE
        assert c2.confidence == c.confidence

    def test_to_json(self):
        c = Container(text="x", kind="entity")
        s = c.to_json()
        d = json.loads(s)
        assert d["text"] == "x"

    def test_render_with_edges(self):
        c = Container(text="Axon", edges=[
            SemanticEdge("is a", "agent"),
            SemanticEdge("uses", "16D substrate"),
        ])
        r = c.render()
        assert "Axon" in r
        assert "is a agent" in r
        assert "uses 16D substrate" in r

    def test_render_all_edges_by_default(self):
        c = Container(text="Axon", edges=[
            SemanticEdge("edge one", "target one"),
            SemanticEdge("edge two", "target two"),
            SemanticEdge("edge three", "target three"),
            SemanticEdge("edge four", "target four"),
            SemanticEdge("edge five", "target five"),
        ])
        r = c.render()
        assert "edge one target one" in r
        assert "edge five target five" in r
        limited = c.render(max_edges=2)
        assert "edge two target two" in limited
        assert "edge three target three" not in limited

    def test_render_skips_described_by(self):
        """render() should skip the 'described by' noise relation."""
        c = Container(text="x", edges=[
            SemanticEdge("is a", "entity"),
            SemanticEdge("described by", "noise"),
        ])
        r = c.render()
        assert "described by" not in r
        assert "is a entity" in r

    def test_render_empty_edges(self):
        c = Container(text="lonely")
        assert c.render() == "lonely"

    def test_render_max_edges(self):
        edges = [SemanticEdge(f"rel{i}", f"target{i}") for i in range(10)]
        c = Container(text="x", edges=edges)
        r = c.render(max_edges=3)
        # Should only include 3 edges
        assert r.count("; ") == 2  # 3 edges -> 2 semicolons


# ---------------------------------------------------------------------------
# Normalization from loose dataset records
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalize_legacy_record(self):
        raw = {
            "word": ".env",
            "chars": ".env",
            "edges": [["contains", "D:\\Axon"], ["configures", "Axon"]],
            "category": "entity",
            "definition": ".env is a file. Attributes purpose configuration.",
        }
        c = normalize_container(raw, source="containers.jsonl", tick=10)
        assert c.text == ".env"
        assert c.normalized_text == ".env"
        assert c.letters == ".env"
        assert c.kind == "entity"
        assert c.source == "containers.jsonl"
        assert c.source_tick == 10
        assert len(c.edges) == 2
        assert c.edges[0].edge_type == "contains"
        assert c.edges[0].target == "D:\\Axon"
        assert c.metadata["definition"] == ".env is a file. Attributes purpose configuration."
        assert c.metadata["category"] == "entity"

    def test_normalize_record_with_empty_edges(self):
        raw = {"word": ".factory", "chars": ".factory", "edges": [],
               "category": "entity", "definition": ".factory is a location."}
        c = normalize_container(raw)
        assert c.text == ".factory"
        assert c.edges == []
        assert c.metadata["definition"] == ".factory is a location."

    def test_normalize_record_with_described_by_edges(self):
        raw = {
            "word": "0chaos",
            "chars": "0chaos",
            "edges": [["contains", "backend"], ["described by", "noise"]],
            "category": "entity",
            "definition": "0chaos is a project.",
        }
        c = normalize_container(raw)
        # All edges are preserved (described by is not stripped during
        # normalization, only during render)
        assert len(c.edges) == 2
        r = c.render()
        assert "described by" not in r

    def test_normalize_list(self):
        records = [
            {"word": "a", "edges": [], "category": "entity"},
            {"word": "b", "edges": [["is a", "thing"]], "category": "concept"},
        ]
        containers = normalize_container_list(records)
        assert len(containers) == 2
        assert containers[0].text == "a"
        assert containers[1].text == "b"
        assert len(containers[1].edges) == 1

    def test_containers_to_dicts(self):
        cs = [Container(text="a"), Container(text="b")]
        ds = containers_to_dicts(cs)
        assert len(ds) == 2
        assert ds[0]["text"] == "a"
        assert ds[1]["text"] == "b"

    def test_container_to_jsonl(self):
        cs = [Container(text="a"), Container(text="b")]
        j = container_to_jsonl(cs)
        lines = j.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["text"] == "a"
        assert json.loads(lines[1])["text"] == "b"


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_update_status(self):
        c = Container(text="x")
        assert c.status == ContainerStatus.DRAFT
        c.update_status(ContainerStatus.ACTIVE, tick=5)
        assert c.status == ContainerStatus.ACTIVE
        assert c.updated_tick == 5

    def test_add_edge(self):
        c = Container(text="x")
        assert len(c.edges) == 0
        c.add_edge(SemanticEdge("is a", "thing"))
        assert len(c.edges) == 1
        assert c.edges[0].edge_type == "is a"

    def test_add_symbol(self):
        c = Container(text="dog")
        assert c.symbols == []
        c.add_symbol("AA")
        assert c.symbols == []
        # Deprecated no-op while old callers migrate.
        c.add_symbol("AA")
        assert c.symbols == []
        c.add_symbol("BB")
        assert c.symbols == []

    def test_add_empty_symbol_ignored(self):
        c = Container(text="x")
        c.add_symbol("")
        assert c.symbols == []


# ---------------------------------------------------------------------------
# EdgeType enum
# ---------------------------------------------------------------------------

class TestEdgeType:
    def test_enum_values(self):
        assert EdgeType.IS_A.value == "is a"
        assert EdgeType.CONTAINS.value == "contains"
        assert EdgeType.DESCRIBED_BY.value == "described by"

    def test_is_str_enum(self):
        assert isinstance(EdgeType.IS_A, str)
        assert EdgeType.IS_A == "is a"


# ---------------------------------------------------------------------------
# Integration: loose record -> Container -> render (kg_search style)
# ---------------------------------------------------------------------------

class TestIntegrationRender:
    def test_render_matches_kg_search_style(self):
        """The render output should match the format kg_search.KGSearch.render
        produces: 'word: etype target; etype target'."""
        raw = {
            "word": "Axon",
            "chars": "axon",
            "edges": [["is a", "agent"], ["uses", "16D substrate"]],
            "category": "entity",
            "definition": "Axon is an agent.",
        }
        c = normalize_container(raw)
        rendered = c.render(max_edges=4)
        # kg_search format: "word: etype target; etype target"
        assert rendered == "Axon: is a agent; uses 16D substrate"

    def test_render_ignores_legacy_symbol(self):
        raw = {
            "word": "dog",
            "chars": "dog",
            "edges": [["is a", "animal"]],
            "category": "entity",
        }
        c = normalize_container(raw)
        c.add_symbol("AA")
        # Render with symbol on edge
        c.edges[0] = SemanticEdge("is a", "animal", symbol="AA")
        rendered = c.render()
        assert rendered == "dog: is a animal"

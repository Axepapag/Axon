from __future__ import annotations

from curator.typed_surfacer import (
    STRUCTURED_KNOWLEDGE_REGION,
    surface_query,
    surface_records,
    to_structured_knowledge_region,
)
from runtime.field import SharedFieldSnapshot, compile_field_view


def _records() -> list[dict]:
    return [
        {
            "container_id": "c-dog",
            "word": "dog",
            "edges": [
                ["is a", "animal", "e-1"],
                ["lives near", "human", "e-2"],
                ["spelled", "dög", "e-3"],
                ["likes", "snow", "e-4"],
            ],
            "source": r"D:\00\axon_semantic_memory.db",
            "provenance": {"table": "entities", "row": 7},
            "confidence": 0.8,
        },
        {
            "word": "FastAPI",
            "edges": [["exposes", "endpoint"]],
            "source": "fixture",
        },
    ]


def test_surface_all_edges_without_a_hidden_cap() -> None:
    result = surface_records(_records(), query="dog")

    assert len(result.spans) == 2
    assert result.omissions == ()
    first = result.spans[0]
    assert first.region == STRUCTURED_KNOWLEDGE_REGION
    assert first.text.splitlines() == [
        "dog is a animal",
        "dog lives near human",
        "dog spelled dög",
        "dog likes snow",
    ]
    assert first.edge_refs == ("e-1", "e-2", "e-3", "e-4")
    assert first.container_refs == ("c-dog",)
    assert r"D:\00\axon_semantic_memory.db" == first.source
    assert '"row":7' in first.provenance


def test_explicit_budgets_create_auditable_omissions() -> None:
    result = surface_records(
        _records(),
        query="dog",
        record_limit=1,
        edge_limit_per_record=2,
    )

    assert len(result.spans) == 1
    assert [item.reason for item in result.omissions] == [
        "explicit_record_budget",
        "explicit_edge_budget",
    ]
    assert result.omissions[0].omitted_count == 1
    assert result.omissions[1].omitted_count == 2
    assert all(len(item.omitted_sha256) == 64 for item in result.omissions)


def test_surfacing_is_deterministic_and_preserves_unicode_canonically() -> None:
    first = surface_records(_records(), query="spelling")
    second = surface_records(_records(), query="spelling")

    assert first == second
    assert first.canonical_sha256 == second.canonical_sha256
    assert "dög" in first.spans[0].text
    assert len(first.spans[0].canonical_sha256) == 64


class _FakeSearch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, k: int | None = None) -> list[dict]:
        self.calls.append((query, k))
        rows = _records()
        return rows if k is None else rows[:k]


def test_query_budget_is_explicit_at_the_search_boundary() -> None:
    search = _FakeSearch()
    result = surface_query(search, "dog", record_limit=1, edge_limit_per_record=1)

    assert search.calls == [("dog", 1)]
    assert len(result.spans) == 1
    assert result.omissions[0].reason == "explicit_edge_budget"


def test_invalid_edges_and_confidence_fail_closed() -> None:
    bad_edge = [{"word": "x", "edges": [["", "target"]]}]
    try:
        surface_records(bad_edge, query="x")
    except ValueError as exc:
        assert "non-empty edge type" in str(exc)
    else:
        raise AssertionError("empty edge type should fail")

    bad_confidence = [{"word": "x", "confidence": 2.0}]
    try:
        surface_records(bad_confidence, query="x")
    except ValueError as exc:
        assert "confidence must be in [0, 1]" in str(exc)
    else:
        raise AssertionError("out-of-range confidence should fail")


def test_surfaced_evidence_enters_the_real_structured_region() -> None:
    surfaced = surface_records(_records(), query="dog")
    structured = to_structured_knowledge_region(surfaced)
    snapshot = SharedFieldSnapshot(regions=(structured,), tick_id=0)
    view = compile_field_view(snapshot)

    assert structured.name.value == STRUCTURED_KNOWLEDGE_REGION
    assert structured.spans[0].container_refs == ("c-dog",)
    assert "dog is a animal" in structured.text
    assert "[structured_knowledge]" in view.decode(0, 256)
    # The canonical non-ASCII character survives even though its bounded 16D
    # projection must audit it rather than silently substituting another char.
    assert "dög" in structured.text
    assert any(
        omission.reason == "unsupported_substrate"
        and omission.logical_region.value == STRUCTURED_KNOWLEDGE_REGION
        for omission in view.omissions
    )

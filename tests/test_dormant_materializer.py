from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from container_schema import Container, ContainerStatus, SemanticEdge
from dormant_materializer import (
    assert_charwise_materialization,
    build_surfacing_example,
    iter_surfacing_examples,
    materialize_visible_text,
    native_text,
    render_container_visible,
    validate_container,
)
from substrate import char_to_slot


def test_native_text_keeps_letters_digits_spaces_only():
    result = native_text("dog->animal_42\n")
    assert result.text == "dog animal 42"
    assert result.changed_chars >= 3


def test_render_materializes_every_visible_character():
    container = Container(
        container_id="c-dog",
        text="dog",
        letters="dog",
        status=ContainerStatus.DORMANT,
        edges=[SemanticEdge(edge_type="is a", target="animal")],
        symbols=[],
        metadata={"layout_symbols": ["LAY"]},
    )
    visible = render_container_visible(container)
    assert "word dog" in visible
    assert "edge is a target animal" in visible
    assert "symbol AA" not in visible
    assert "LAY" not in visible

    slots = materialize_visible_text(visible)
    assert slots.shape == (len(visible), 16)
    assert_charwise_materialization(visible, slots)
    for i, char in enumerate(visible):
        assert np.array_equal(slots[i], char_to_slot(char))


def test_validate_container_rejects_visible_symbols():
    container = Container(
        text="dog",
        edges=[SemanticEdge(edge_type="is a", target="animal")],
        symbols=["AA"],
    )
    errors = validate_container(container)
    assert "container.symbols must be empty; semantic edges are spelled out" in errors


def test_validate_container_rejects_edge_symbol():
    container = Container(
        text="dog",
        edges=[SemanticEdge(edge_type="is a", target="animal", symbol="AA")],
        symbols=[],
        metadata={"layout_symbols": ["LAY"]},
    )
    errors = validate_container(container)
    assert "edge.symbol must be empty; edge meaning is English text" in errors


def test_render_container_visible_does_not_truncate_edges_by_default():
    container = Container(
        text="dog",
        letters="dog",
        edges=[
            SemanticEdge(edge_type="is a", target="animal"),
            SemanticEdge(edge_type="friend of", target="human"),
            SemanticEdge(edge_type="uses", target="leash"),
            SemanticEdge(edge_type="part of", target="pack"),
            SemanticEdge(edge_type="located in", target="home"),
        ],
        symbols=[],
    )
    visible = render_container_visible(container)
    for expected in ("animal", "human", "leash", "pack", "home"):
        assert expected in visible

    limited = render_container_visible(container, max_edges=2)
    assert "animal" in limited
    assert "human" in limited
    assert "leash" not in limited


def test_build_surfacing_example_uses_source_edge_answer():
    container = Container(
        container_id="c-rel",
        text="dog is a animal",
        letters="dog is a animal",
        edges=[SemanticEdge(edge_type="is a", target="animal")],
        symbols=[],
        metadata={"source_entity": "dog", "layout_symbols": ["LAY"]},
    )
    example = build_surfacing_example(container, container.edges[0])
    assert example is not None
    assert example["query"] == "dog is a"
    assert example["answer"] == "animal"
    assert "edge_symbol" not in example
    assert "LAY" not in example["cortex"]


def test_iter_surfacing_examples_skips_invalid_symbol_leak(tmp_path: Path):
    valid = Container(
        container_id="c-valid",
        text="cat is a animal",
        letters="cat is a animal",
        edges=[SemanticEdge(edge_type="is a", target="animal")],
        symbols=[],
        metadata={"source_entity": "cat"},
    ).to_dict()
    invalid = Container(
        container_id="c-invalid",
        text="dog",
        letters="dog",
        edges=[SemanticEdge(edge_type="is a", target="animal", symbol="AA")],
        symbols=[],
    ).to_dict()
    path = tmp_path / "containers.jsonl"
    path.write_text(
        json.dumps(valid) + "\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )

    examples = list(iter_surfacing_examples(path))
    assert len(examples) == 1
    assert examples[0]["container_id"] == "c-valid"

from __future__ import annotations

import json
from pathlib import Path

from kg_search import KGSearch


def _write_legacy_kg(path: Path) -> None:
    rows = [
        {
            "word": "dog",
            "edges": [
                ["is a", "animal"],
                ["friend of", "human"],
                ["uses", "leash"],
                ["part of", "pack"],
                ["located in", "home"],
            ],
        },
        {"word": "cat", "edges": [["is a", "animal"]]},
        {"word": "wolf", "edges": [["is a", "animal"], ["part of", "pack"]]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_render_all_edges_by_default(tmp_path: Path):
    path = tmp_path / "containers.jsonl"
    _write_legacy_kg(path)
    kg = KGSearch(str(path))
    rendered = kg.render(kg.by_word["dog"])
    assert "is a animal" in rendered
    assert "located in home" in rendered

    limited = kg.render(kg.by_word["dog"], max_edges=2)
    assert "friend of human" in limited
    assert "uses leash" not in limited


def test_search_returns_full_ranked_set_by_default(tmp_path: Path):
    path = tmp_path / "containers.jsonl"
    _write_legacy_kg(path)
    kg = KGSearch(str(path))
    full = kg.search("dog")
    limited = kg.search("dog", k=1)
    assert len(full) == 2
    assert len(limited) == 1

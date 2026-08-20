#!/usr/bin/env python3
"""Tests for training/build_recovered_curriculum.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATOR = REPO_ROOT / "curator"
TRAINING = REPO_ROOT / "training"
for path in (REPO_ROOT, CURATOR, TRAINING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from container_schema import Container, ContainerStatus, SemanticEdge
from build_recovered_curriculum import RecoveredCurriculumBuilder, clean_text


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@pytest.fixture
def dormant_artifacts(tmp_path: Path) -> dict[str, Path]:
    containers = [
        Container(
            kind="fact",
            text="dog is animal",
            source="test:fact:1",
            provenance="test",
            status=ContainerStatus.DORMANT,
            edges=[SemanticEdge(edge_type="is a", target="animal")],
            symbols=[],
            metadata={"key": "dog", "value": "animal"},
        ).to_dict(),
        Container(
            kind="relation",
            text="dog likes mud",
            source="test:relation:2",
            provenance="test",
            status=ContainerStatus.DORMANT,
            edges=[SemanticEdge(edge_type="likes", target="mud")],
            symbols=[],
            metadata={"subject": "dog", "predicate": "likes", "object": "mud"},
        ).to_dict(),
        Container(
            kind="procedure",
            text="Clean dog",
            source="test:procedure:3",
            provenance="test",
            status=ContainerStatus.DORMANT,
            metadata={"steps": ["find towel", "wash dog"], "outcome": "clean dog"},
        ).to_dict(),
        Container(
            kind="episode",
            text="dog escaped",
            source="test:episode:4",
            provenance="test",
            status=ContainerStatus.DORMANT,
            edges=[SemanticEdge(edge_type="event", target="escape")],
            symbols=[],
            metadata={"summary": "dog escaped and returned"},
        ).to_dict(),
        Container(
            kind="diary",
            text="private diary note",
            source="test:diary:5",
            provenance="test",
            status=ContainerStatus.DORMANT,
            metadata={"redaction": "diary_only"},
        ).to_dict(),
    ]
    registry = [
        {"symbol": "AA", "name": "layout_fact", "level": 1},
    ]
    edges = [
        {"source_text": "dog is animal", "edge_type": "is a", "target": "animal"},
        {"source_text": "dog likes mud", "edge_type": "likes", "target": "mud"},
    ]
    containers_path = tmp_path / "containers.jsonl"
    registry_path = tmp_path / "symbol_registry.jsonl"
    edges_path = tmp_path / "semantic_edges.jsonl"
    _write_jsonl(containers_path, containers)
    _write_jsonl(registry_path, registry)
    _write_jsonl(edges_path, edges)
    return {
        "containers": containers_path,
        "registry": registry_path,
        "edges": edges_path,
        "out": tmp_path / "curriculum",
    }


def test_clean_text_preserves_word_boundaries():
    assert clean_text("dog->animal_42\n") == "dog animal 42"


def test_curriculum_builder_writes_all_families(dormant_artifacts):
    builder = RecoveredCurriculumBuilder(
        containers_path=str(dormant_artifacts["containers"]),
        registry_path=str(dormant_artifacts["registry"]),
        edges_path=str(dormant_artifacts["edges"]),
        out_dir=str(dormant_artifacts["out"]),
        max_examples=10,
        negative_ratio=1.0,
        no_personal_log=True,
    )
    stats = builder.run()

    manifest_path = dormant_artifacts["out"] / "curriculum_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "axon_recovered_curriculum_manifest"
    assert "diary" not in "\n".join(
        (dormant_artifacts["out"] / "field_surfacing_v1.jsonl").read_text(encoding="utf-8").splitlines()
    )

    assert stats.family_counts["field_surfacing_v1"] >= 3
    assert stats.family_counts["edge_prediction_v1"] >= 3
    assert not (dormant_artifacts["out"] / "symbol_assignment_v1.jsonl").exists()
    assert stats.family_counts["retrieval_qa_v1"] >= 2
    assert stats.family_counts["procedure_next_step_v1"] >= 2
    assert stats.family_counts["episodic_exhale_filter_v1"] >= 2

    surfacing = [
        json.loads(line)
        for line in (dormant_artifacts["out"] / "field_surfacing_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert all("edge_symbol" not in row for row in surfacing)
    prediction = [
        json.loads(line)
        for line in (dormant_artifacts["out"] / "edge_prediction_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert all("symbol" not in row for row in prediction)


def test_curriculum_builder_can_isolate_diary(dormant_artifacts):
    builder = RecoveredCurriculumBuilder(
        containers_path=str(dormant_artifacts["containers"]),
        registry_path=str(dormant_artifacts["registry"]),
        edges_path=str(dormant_artifacts["edges"]),
        out_dir=str(dormant_artifacts["out"]),
        families=["field_surfacing_v1"],
        no_personal_log=True,
    )
    builder.run()
    data = (dormant_artifacts["out"] / "field_surfacing_v1.jsonl").read_text(encoding="utf-8")
    assert "private diary note" not in data


def test_curriculum_builder_includes_diary_only_family_when_allowed(dormant_artifacts):
    builder = RecoveredCurriculumBuilder(
        containers_path=str(dormant_artifacts["containers"]),
        registry_path=str(dormant_artifacts["registry"]),
        edges_path=str(dormant_artifacts["edges"]),
        out_dir=str(dormant_artifacts["out"]),
        families=["diary_region_self_reflection_v1"],
        no_personal_log=False,
    )
    stats = builder.run()
    assert stats.family_counts["diary_region_self_reflection_v1"] == 1
    rows = [
        json.loads(line)
        for line in (dormant_artifacts["out"] / "diary_region_self_reflection_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert rows[0]["region"] == "diary"
    assert rows[0]["redaction"] == "diary_only"
    assert "retrieval_qa_v1" in rows[0]["blocked_families"]

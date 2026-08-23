from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.dormant import (
    DormantEvidenceGenerationStore,
    DormantEvidenceIndex,
    DormantGenerationError,
    DormantIncrementalError,
    DormantIncrementalFallbackRequired,
    DormantIncrementalMaintainer,
    DormantIncrementalRecoveryRequired,
)
from tests.test_dormant_evidence_bridge import _jsonl_line, _write_fixture_state


def _prepared_store(tmp_path: Path) -> tuple[Path, DormantEvidenceGenerationStore]:
    state_root = _write_fixture_state(tmp_path)
    with DormantEvidenceIndex.build(state_root):
        pass
    store = DormantEvidenceGenerationStore(state_root)
    store.bootstrap_legacy()
    return state_root, store


def _update_counts(state_root: Path, *, containers: int, edges: int) -> None:
    path = state_root / "dormant" / "corpus_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["output_counts"] = {"containers": containers, "semantic_edges": edges}
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _append_memory_container(state_root: Path) -> None:
    record = {
        "container_id": "c-memory",
        "kind": "concept",
        "text": "Memory",
        "normalized_text": "memory",
        "letters": "Memory",
        "edges": [],
        "source": "fixture.db:concepts:3",
        "provenance": "fixture:container:memory",
        "confidence": 0.91,
        "status": "dormant",
        "metadata": {"definition": "persistent evidence store"},
    }
    with (state_root / "dormant" / "containers.jsonl").open("ab") as handle:
        handle.write(_jsonl_line(record))


def _append_memory_edge(state_root: Path) -> None:
    edge = {
        "source_container_id": "c-axon",
        "source_text": "Axon stores Memory",
        "edge_type": "stores",
        "target": "Memory",
        "provenance": "fixture:edge:memory",
        "confidence": 0.93,
        "status": "dormant",
    }
    with (state_root / "dormant" / "semantic_edges.jsonl").open("ab") as handle:
        handle.write(_jsonl_line(edge))


def test_incremental_append_updates_same_index_without_copying_base(tmp_path: Path) -> None:
    state_root, store = _prepared_store(tmp_path)
    before = store.active_descriptor()
    before_token = store.active_token()
    before_path = store._safe_index_path(before.relative_index_path)

    _append_memory_container(state_root)
    _append_memory_edge(state_root)
    _update_counts(state_root, containers=3, edges=2)

    result = DormantIncrementalMaintainer(store).maintain()
    assert result.descriptor.mode == "incremental_update"
    assert result.descriptor.predecessor_generation_id == before.generation_id
    assert result.container_appends == 1
    assert result.edge_appends == 1
    assert result.container_updates == 0
    assert result.edge_updates == 0
    assert result.full_rebuild_fallback is False
    assert store._safe_index_path(result.descriptor.relative_index_path) == before_path
    assert store.active_token() != before_token

    generation_dir = store.generations_root / result.descriptor.generation_id
    assert (generation_dir / "generation.json").is_file()
    assert not (generation_dir / "index.sqlite3").exists()

    with store.open_active() as index:
        assert index.dereference_container("c-memory").text == "Memory"
        candidates = index.query_candidates("Axon stores", limit=8)
        memory = next(item for item in candidates if item.container_id == "c-memory")
        assert memory.relation_hits > 0
        evidence = index.retrieve("Axon stores", limit=8)
        assert any(item.container.container_id == "c-memory" for item in evidence)


def test_incremental_equal_length_container_update_repairs_terms_and_exact_pointer(tmp_path: Path) -> None:
    state_root, store = _prepared_store(tmp_path)
    path = state_root / "dormant" / "containers.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    original = json.loads(lines[0])
    original["text"] = "Axim"
    original["normalized_text"] = "axim"
    original["letters"] = "Axim"
    original["metadata"]["definition"] = "dynamicx architecture"
    replacement = _jsonl_line(original)
    assert len(replacement) == len(lines[0])
    lines[0] = replacement
    path.write_bytes(b"".join(lines))

    result = DormantIncrementalMaintainer(store).maintain()
    assert result.container_updates == 1
    assert result.container_appends == 0
    with store.open_active() as index:
        exact = index.dereference_container("c-axon")
        assert exact.text == "Axim"
        assert exact.byte_offset == 0
        assert any(
            item.container_id == "c-axon"
            for item in index.query_candidates("dynamicx", limit=4, include_graph=False)
        )
        assert not index.query_candidates("stateful", limit=4, include_graph=False)


def test_incremental_equal_length_edge_update_replaces_edge_identity_and_relation(tmp_path: Path) -> None:
    state_root, store = _prepared_store(tmp_path)
    with store.open_active() as index:
        old_edge_id = next(iter(index.iter_edge_sources()))[0]

    path = state_root / "dormant" / "semantic_edges.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    edge = json.loads(lines[0])
    edge["edge_type"] = "owns"
    edge["source_text"] = "Axon owns System"
    replacement = _jsonl_line(edge)
    assert len(replacement) == len(lines[0])
    lines[0] = replacement
    path.write_bytes(b"".join(lines))

    result = DormantIncrementalMaintainer(store).maintain()
    assert result.edge_updates == 1
    with store.open_active() as index:
        candidates = index.query_candidates("Axon owns", limit=8)
        system = next(item for item in candidates if item.container_id == "c-system")
        assert system.relation_hits > 0
        assert old_edge_id not in system.edge_ids
        new_edge_id = next(edge_id for edge_id in system.edge_ids if edge_id != old_edge_id)
        exact = index.dereference_edge(new_edge_id)
        assert exact.edge_type == "owns"
        assert exact.source_text == "Axon owns System"


def test_incremental_manifest_only_change_advances_binding_without_record_reindex(tmp_path: Path) -> None:
    state_root, store = _prepared_store(tmp_path)
    before_token = store.active_token()
    manifest_path = state_root / "dormant" / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["maintenance_epoch"] = 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    result = DormantIncrementalMaintainer(store).maintain()
    assert result.metadata_only is True
    assert result.container_appends == result.container_updates == 0
    assert result.edge_appends == result.edge_updates == 0
    assert store.active_token() != before_token
    with store.open_active() as index:
        assert index.dereference_container("c-axon").text == "Axon"


def test_incremental_noop_reuses_current_generation(tmp_path: Path) -> None:
    _, store = _prepared_store(tmp_path)
    before = store.active_descriptor()
    before_token = store.active_token()
    result = DormantIncrementalMaintainer(store).maintain()
    assert result.descriptor == before
    assert result.plan_id
    assert store.active_token() == before_token


def test_layout_shift_fails_closed_then_full_generation_fallback_can_promote(tmp_path: Path) -> None:
    state_root, store = _prepared_store(tmp_path)
    before = store.active_descriptor()
    path = state_root / "dormant" / "containers.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[0])
    record["text"] = "Axon expanded"
    record["normalized_text"] = "axon expanded"
    record["letters"] = "Axon expanded"
    lines[0] = _jsonl_line(record)
    path.write_bytes(b"".join(lines))

    maintainer = DormantIncrementalMaintainer(store)
    with pytest.raises(DormantIncrementalFallbackRequired, match="layout shifted"):
        maintainer.plan()

    result = maintainer.maintain(fallback_full=True)
    assert result.full_rebuild_fallback is True
    assert result.descriptor.mode == "full_generation"
    assert result.descriptor.predecessor_generation_id == before.generation_id
    assert result.descriptor.relative_index_path != before.relative_index_path
    with store.open_active() as index:
        assert index.dereference_container("c-axon").text == "Axon expanded"
        assert any(item.container_id == "c-axon" for item in index.query_candidates("expanded", limit=4))


def test_crash_after_sqlite_commit_before_pointer_is_recoverable_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root, store = _prepared_store(tmp_path)
    _append_memory_container(state_root)
    _update_counts(state_root, containers=3, edges=1)
    maintainer = DormantIncrementalMaintainer(store)
    plan = maintainer.plan()
    original_write_pointer = DormantEvidenceGenerationStore._write_pointer

    def _fail_pointer(self: DormantEvidenceGenerationStore, descriptor: object) -> None:
        raise OSError("injected pointer failure")

    monkeypatch.setattr(DormantEvidenceGenerationStore, "_write_pointer", _fail_pointer)
    with pytest.raises(DormantIncrementalRecoveryRequired, match="pointer publication"):
        maintainer.apply(plan)
    with pytest.raises(DormantGenerationError, match="does not match index metadata"):
        store.active_descriptor()

    monkeypatch.setattr(DormantEvidenceGenerationStore, "_write_pointer", original_write_pointer)
    recovered = maintainer.recover_pointer()
    assert recovered.mode == "incremental_update"
    assert store.active_descriptor() == recovered
    with store.open_active() as index:
        assert index.dereference_container("c-memory").text == "Memory"


def test_stale_incremental_plan_cannot_replay_after_generation_advances(tmp_path: Path) -> None:
    state_root, store = _prepared_store(tmp_path)
    _append_memory_container(state_root)
    _update_counts(state_root, containers=3, edges=1)
    maintainer = DormantIncrementalMaintainer(store)
    plan = maintainer.plan()
    first = maintainer.apply(plan)
    assert first.container_appends == 1

    with pytest.raises(DormantIncrementalError, match="predecessor index changed"):
        maintainer.apply(plan)
    with store.open_active() as index:
        assert index.dereference_container("c-memory").text == "Memory"


def test_prepared_manifest_recovers_if_final_manifest_write_fails_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root, store = _prepared_store(tmp_path)
    _append_memory_container(state_root)
    _update_counts(state_root, containers=3, edges=1)
    maintainer = DormantIncrementalMaintainer(store)
    plan = maintainer.plan()
    original_final = DormantIncrementalMaintainer._write_generation_manifest

    def _fail_final(self: DormantIncrementalMaintainer, *args: object, **kwargs: object) -> Path:
        raise OSError("injected final manifest failure")

    monkeypatch.setattr(DormantIncrementalMaintainer, "_write_generation_manifest", _fail_final)
    with pytest.raises(DormantIncrementalRecoveryRequired, match="pointer publication"):
        maintainer.apply(plan)
    monkeypatch.setattr(DormantIncrementalMaintainer, "_write_generation_manifest", original_final)

    recovered = maintainer.recover_pointer()
    assert recovered.mode == "incremental_update"
    manifest_path = store.generations_root / recovered.generation_id / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["publication_state"] == "recovered"
    assert manifest["recovered_after_pointer_gap"] is True
    with store.open_active() as index:
        assert index.dereference_container("c-memory").text == "Memory"


def test_heart_coordinator_reopens_reader_after_incremental_generation_swap(tmp_path: Path) -> None:
    from runtime.field import CanonicalStateBranch
    from runtime.heart import BeatCoordinator, CoreRegistry

    state_root, store = _prepared_store(tmp_path)
    branch = CanonicalStateBranch(
        state_root / "active" / "branches" / "incremental-reader-swap",
        branch_id="incremental-reader-swap",
        authority_root=state_root,
    )
    coordinator = BeatCoordinator(branch, CoreRegistry(), state_root=state_root)
    first_bridge = coordinator._ensure_bridge()
    first_index_id = first_bridge.index.index_id
    first_token = coordinator._bridge_generation_token

    _append_memory_container(state_root)
    _append_memory_edge(state_root)
    _update_counts(state_root, containers=3, edges=2)
    result = DormantIncrementalMaintainer(store).maintain()
    assert result.descriptor.index_id != first_index_id

    second_bridge = coordinator._ensure_bridge()
    assert second_bridge is not first_bridge
    assert second_bridge.index.index_id == result.descriptor.index_id
    assert coordinator._bridge_generation_token == store.active_token()
    assert coordinator._bridge_generation_token != first_token
    assert second_bridge.index.dereference_container("c-memory").text == "Memory"
    second_bridge.index.close()


def test_container_key_update_removes_old_graph_link_then_edge_retarget_restores_it(tmp_path: Path) -> None:
    state_root, store = _prepared_store(tmp_path)
    maintainer = DormantIncrementalMaintainer(store)

    containers_path = state_root / "dormant" / "containers.jsonl"
    container_lines = containers_path.read_bytes().splitlines(keepends=True)
    system = json.loads(container_lines[1])
    system["text"] = "Systam"
    system["normalized_text"] = "systam"
    system["letters"] = "Systam"
    replacement = _jsonl_line(system)
    assert len(replacement) == len(container_lines[1])
    container_lines[1] = replacement
    containers_path.write_bytes(b"".join(container_lines))

    first = maintainer.maintain()
    assert first.container_updates == 1
    with store.open_active() as index:
        candidates = index.query_candidates("Axon uses", limit=8)
        assert all(item.container_id != "c-system" for item in candidates)

    edges_path = state_root / "dormant" / "semantic_edges.jsonl"
    edge_lines = edges_path.read_bytes().splitlines(keepends=True)
    edge = json.loads(edge_lines[0])
    edge["target"] = "Systam"
    edge["source_text"] = "Axon uses Systam"
    edge_replacement = _jsonl_line(edge)
    assert len(edge_replacement) == len(edge_lines[0])
    edge_lines[0] = edge_replacement
    edges_path.write_bytes(b"".join(edge_lines))

    second = maintainer.maintain()
    assert second.edge_updates == 1
    with store.open_active() as index:
        candidates = index.query_candidates("Axon uses", limit=8)
        restored = next(item for item in candidates if item.container_id == "c-system")
        assert restored.relation_hits > 0
        assert index.dereference_container("c-system").text == "Systam"

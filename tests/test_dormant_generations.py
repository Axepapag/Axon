from __future__ import annotations

from pathlib import Path

import pytest

from runtime.dormant import (
    DormantEvidenceGenerationStore,
    DormantEvidenceIndex,
    DormantGenerationError,
    DormantIndexGeneration,
)
from tests.test_dormant_evidence_bridge import _write_fixture_state


def test_generation_store_adopts_legacy_index_without_copying_it(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    legacy = DormantEvidenceIndex.build(state_root)
    legacy_path = legacy.path
    legacy_id = legacy.index_id
    legacy_size = legacy_path.stat().st_size
    legacy.close()

    store = DormantEvidenceGenerationStore(state_root)
    before = store.active_descriptor()
    assert before.mode == "legacy_adopted"
    assert not store.pointer_path.exists()

    adopted = store.bootstrap_legacy()
    assert store.pointer_path.is_file()
    assert adopted.index_id == legacy_id
    assert store._safe_index_path(adopted.relative_index_path) == legacy_path
    assert legacy_path.stat().st_size == legacy_size
    assert not store.generations_root.exists()

    with store.open_active() as opened:
        assert opened.index_id == legacy_id


def test_candidate_generation_does_not_change_active_until_promotion(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    index.close()
    store = DormantEvidenceGenerationStore(state_root)
    active = store.bootstrap_legacy()
    active_token = store.active_token()

    candidate = store.build_generation(promote=False)
    assert candidate.mode == "full_generation"
    assert candidate.generation_id != active.generation_id
    assert store.active_token() == active_token
    candidate_path = store._safe_index_path(candidate.relative_index_path)
    assert candidate_path.is_file()

    promoted = store.promote(candidate)
    assert promoted.generation_id == candidate.generation_id
    assert promoted.predecessor_generation_id == active.generation_id
    assert store.active_descriptor().generation_id == candidate.generation_id
    assert store.active_token() != active_token
    assert store._safe_index_path(active.relative_index_path).is_file()


def test_failed_promotion_leaves_previous_active_generation_untouched(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    index.close()
    store = DormantEvidenceGenerationStore(state_root)
    active = store.bootstrap_legacy()
    token = store.active_token()

    invalid = DormantIndexGeneration(
        generation_id="bad-candidate",
        index_id="0" * 64,
        binding_id=active.binding_id,
        relative_index_path=active.relative_index_path,
        promoted_at=active.promoted_at,
        mode="full_generation",
    )
    with pytest.raises(DormantGenerationError):
        store.promote(invalid)
    assert store.active_token() == token
    assert store.active_descriptor().generation_id == active.generation_id


def test_generation_pointer_rejects_path_escape(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    index.close()
    store = DormantEvidenceGenerationStore(state_root)
    active = store.bootstrap_legacy()

    escaped = DormantIndexGeneration(
        generation_id="escape",
        index_id=active.index_id,
        binding_id=active.binding_id,
        relative_index_path="../outside.sqlite3",
        promoted_at=active.promoted_at,
        mode="full_generation",
    )
    with pytest.raises(DormantGenerationError, match="escapes"):
        store.promote(escaped)

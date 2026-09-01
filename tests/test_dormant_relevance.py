from __future__ import annotations

from pathlib import Path

from runtime.dormant import (
    DormantEvidenceBridge,
    DormantEvidenceIndex,
    DormantRelevanceAuditor,
    DormantRelevancePolicy,
)
from runtime.field import SharedFieldSnapshot
from tests.test_dormant_evidence_bridge import _write_fixture_state


def test_relevance_auditor_prefers_exact_query_match(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    with DormantEvidenceIndex.build(state_root) as index:
        evidence = index.retrieve("stateful architecture", limit=4)
        auditor = DormantRelevanceAuditor(
            DormantRelevancePolicy(items_per_materialization=1, target_chars=10_000)
        )
        decision = auditor.select("stateful architecture", evidence, SharedFieldSnapshot.empty(tick_id=0))
        assert decision.selected_container_ids == ("c-axon",)
        assert decision.scores[0].lexical_support == 1.0
        assert decision.total_chars == len("Axon")
        assert not decision.fallback_used


def test_relevance_auditor_never_truncates_oversize_exact_evidence(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    with DormantEvidenceIndex.build(state_root) as index:
        evidence = index.retrieve("stateful architecture", limit=4)
        auditor = DormantRelevanceAuditor(
            DormantRelevancePolicy(items_per_materialization=4, target_chars=3)
        )
        decision = auditor.select("stateful architecture", evidence, SharedFieldSnapshot.empty(tick_id=0))
        assert decision.selected_container_ids == ("c-axon",)
        assert decision.selected[0].container.text == "Axon"
        assert decision.work_target_overrun
        assert "c-system" in decision.skipped_budget


def test_relevance_auditor_falls_back_only_to_exact_lexical_candidate(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    with DormantEvidenceIndex.build(state_root) as index:
        evidence = index.retrieve("stateful architecture", limit=4)
        auditor = DormantRelevanceAuditor(
            DormantRelevancePolicy(
                items_per_materialization=1,
                target_chars=10_000,
                min_score=1.0,
            )
        )
        decision = auditor.select("stateful architecture", evidence, SharedFieldSnapshot.empty(tick_id=0))
        assert decision.fallback_used
        assert decision.selected_container_ids == ("c-axon",)
        assert decision.selected[0].candidate.lexical_hits > 0


def test_active_exact_evidence_gets_retention_floor_not_fake_novelty(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    with DormantEvidenceIndex.build(state_root) as index:
        evidence = index.retrieve("stateful architecture", limit=2)
        bridge = DormantEvidenceBridge(index)
        active = bridge.surface(SharedFieldSnapshot.empty(tick_id=0), evidence[:1])
        auditor = DormantRelevanceAuditor(
            DormantRelevancePolicy(items_per_materialization=2, target_chars=10_000)
        )
        decision = auditor.select("stateful architecture", evidence, active)
        score = next(item for item in decision.scores if item.container_id == "c-axon")
        assert score.already_active
        assert score.novelty_support >= 0.35
        assert "c-axon" in decision.selected_container_ids


def test_total_character_budget_counts_separators_without_truncation(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    with DormantEvidenceIndex.build(state_root) as index:
        evidence = index.retrieve("stateful architecture", limit=4)
        auditor = DormantRelevanceAuditor(
            DormantRelevancePolicy(
                items_per_materialization=4,
                target_chars=len("Axon") + 1,
            )
        )
        decision = auditor.select("stateful architecture", evidence, SharedFieldSnapshot.empty(tick_id=0))
        assert decision.selected_container_ids == ("c-axon",)
        assert decision.total_chars == len("Axon")
        assert "c-system" in decision.skipped_budget

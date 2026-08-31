from __future__ import annotations

import json

import pytest

from runtime.field import LogicalRegion, SharedFieldSnapshot
from runtime.heart import ReasoningDecision, ReasoningOperationKind
from training.communication_first_c1 import (
    C1_FAMILY,
    DEFAULT_C1_SPLIT_COUNTS,
    C1CurriculumCompiler,
    find_c1_target_leaks,
    publish_c1_curriculum,
    verify_c1_no_target_leakage,
)
from training.first_form_curriculum import (
    FFCS_MANIFEST_SCHEMA,
    FirstFormCase,
    FirstFormCurriculum,
    TeachingEligibility,
    load_first_form_curriculum,
)
from training.living_reasoning_curriculum import (
    LivingReasoningEpisode,
    LivingReasoningTarget,
)

IDENTITY = "Axon is Axon.\nEvery brother attends the complete canonical field."


def _compile(**kwargs) -> FirstFormCurriculum:
    return C1CurriculumCompiler().compile(identity_text=IDENTITY, **kwargs)


def test_c1_compile_publish_load_schema_roundtrip(tmp_path) -> None:
    curriculum = _compile()
    assert curriculum.actual_family_split_counts == curriculum.requested_family_split_counts
    assert len(curriculum.cases) == 36

    path = publish_c1_curriculum(curriculum, state_root=tmp_path / "State")
    restored = load_first_form_curriculum(path)
    assert restored.manifest_id == curriculum.manifest_id
    assert restored.to_canonical_dict() == curriculum.to_canonical_dict()
    assert [case.case_id for case in restored.cases] == [
        case.case_id for case in curriculum.cases
    ]
    assert publish_c1_curriculum(restored, state_root=tmp_path / "State") == path


def test_c1_leakage_verifier_passes_compiled_curriculum() -> None:
    curriculum = _compile()
    assert find_c1_target_leaks(curriculum.cases) == ()
    verify_c1_no_target_leakage(curriculum.cases)


def test_c1_leakage_verifier_fails_on_planted_leak() -> None:
    answer = "The valve stays shut until sunrise."
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: IDENTITY,
            LogicalRegion.USER_INPUT: f"Tell me about the valve. Hint: {answer}",
            LogicalRegion.SCRATCH: "",
            LogicalRegion.RESPONSE_DRAFT: "",
        }
    )
    episode = LivingReasoningEpisode(
        label="c1-planted-leak",
        split="train",
        snapshot=snapshot,
        first_workspace_text="fixture workspace",
        refined_workspace_text="fixture workspace",
        targets=(
            LivingReasoningTarget(
                phase="first", decision=ReasoningDecision.NO_OP, supervision_weight=0.0
            ),
            LivingReasoningTarget(
                phase="refined", decision=ReasoningDecision.NO_OP, supervision_weight=0.0
            ),
            LivingReasoningTarget(
                phase="consolidated",
                decision=ReasoningDecision.DELTA,
                operation=ReasoningOperationKind.REPLACE,
                region=LogicalRegion.RESPONSE_DRAFT,
                start=0,
                end=0,
                payload=answer,
            ),
        ),
        mechanism_tags=("c1", "planted_leak"),
        target_basis="planted leak for verifier testing",
    )
    case = FirstFormCase(
        family=C1_FAMILY,
        competency="grounded_short_answer",
        eligibility=TeachingEligibility.VERIFIED_TARGET,
        lineage_id="c1-planted-leak:train:000",
        source_record_ids=(),
        episode=episode,
        transport_pages=1,
        procedural_depth=1,
        derived=True,
    )

    leaks = find_c1_target_leaks([case])
    assert any(leak.surface == "field:user_input" for leak in leaks)
    with pytest.raises(ValueError, match="hidden-target leakage"):
        verify_c1_no_target_leakage([case])


def test_c1_split_lineage_disjointness() -> None:
    curriculum = _compile()
    by_lineage: dict[str, set[str]] = {}
    for case in curriculum.cases:
        by_lineage.setdefault(case.lineage_id, set()).add(case.episode.split)
    assert all(len(splits) == 1 for splits in by_lineage.values())
    per_split = {split: set() for split in ("train", "heldout", "regression")}
    for case in curriculum.cases:
        per_split[case.episode.split].add(case.lineage_id)
    assert per_split["train"].isdisjoint(per_split["heldout"])
    assert per_split["train"].isdisjoint(per_split["regression"])
    assert per_split["heldout"].isdisjoint(per_split["regression"])


def test_c1_deterministic_rebuild_identical_manifest_id() -> None:
    first = _compile()
    second = _compile()
    assert first.manifest_id == second.manifest_id
    assert first.to_canonical_dict() == second.to_canonical_dict()


def test_c1_budget_shortfall_and_selection() -> None:
    available = dict(zip(("train", "heldout", "regression"), (24, 6, 6), strict=True))
    with pytest.raises(ValueError, match="cannot satisfy declared budgets"):
        _compile(
            requested_counts=(
                (C1_FAMILY, (available["train"] + 1, available["heldout"], available["regression"])),
            )
        )
    reduced = _compile(requested_counts=((C1_FAMILY, (21, 5, 5)),))
    assert reduced.actual_family_split_counts == ((C1_FAMILY, (21, 5, 5)),)
    assert dict(reduced.excluded_counts)["c1_fixtures_not_selected"] == 5


def test_c1_default_counts_match_compiled_manifest() -> None:
    curriculum = _compile()
    assert curriculum.requested_family_split_counts == DEFAULT_C1_SPLIT_COUNTS
    counts = dict(curriculum.actual_family_split_counts)[C1_FAMILY]
    assert counts == (24, 6, 6)
    competencies = {case.competency for case in curriculum.cases}
    assert competencies == {
        "greeting",
        "acknowledgement",
        "grounded_short_answer",
        "clarification_request",
        "correction_acceptance",
        "uncertainty_admission",
        "turn_taking",
    }
    for case in curriculum.cases:
        consolidated = case.episode.targets[2]
        assert consolidated.region is LogicalRegion.RESPONSE_DRAFT
        assert consolidated.supervision_weight == 1.0
        identity_region = case.episode.snapshot.region(LogicalRegion.IDENTITY).text
        assert consolidated.payload not in identity_region


def test_c1_tournament_loader_compatibility(tmp_path) -> None:
    curriculum = _compile()
    path = publish_c1_curriculum(curriculum, state_root=tmp_path / "State")

    # Mirrors scripts/run_d64_tournament.py lines 147-157 exactly.
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body.get("schema") == FFCS_MANIFEST_SCHEMA == "axon-first-form-curriculum-v1"
    loaded = load_first_form_curriculum(path)
    assert loaded.manifest_id == curriculum.manifest_id
    assert len(loaded.living_curriculum.episodes) == len(loaded.cases)

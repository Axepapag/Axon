from __future__ import annotations

import json

import pytest

from runtime.field import LogicalRegion
from runtime.heart import ReasoningDecision, ReasoningOperationKind
from training.first_form_curriculum import (
    FFCS_MANIFEST_SCHEMA,
    FirstFormCurriculum,
    TeachingEligibility,
    load_first_form_curriculum,
)
from training.language_foundations import (
    COMPETENCY_PREREQUISITES,
    DEFAULT_LANGUAGE_SPLIT_COUNTS,
    LANGUAGE_FOUNDATIONS_FAMILIES,
    LanguageFoundationsCompiler,
    LanguageLessonSpec,
    TargetVisibility,
    find_language_target_leaks,
    publish_language_foundations_curriculum,
    verify_language_no_target_leakage,
    verify_language_transfer_disjointness,
    verify_language_visibility_consistency,
)
from training.language_foundations.compiler import _case
from training.language_foundations.contracts import VALIDATOR_EXACT_TRANSPORT
from training.language_foundations.validators import _tokens

IDENTITY = "Axon is Axon.\nEvery brother attends the complete canonical field."


def _compile(**kwargs) -> FirstFormCurriculum:
    return LanguageFoundationsCompiler().compile(identity_text=IDENTITY, **kwargs)


def _spec(**overrides) -> LanguageLessonSpec:
    base = {
        "family": "L1",
        "competency": "spelling_repair",
        "split": "train",
        "prompt": "Fix the spelling: helo",
        "answer": "hello",
        "visibility": TargetVisibility.HIDDEN,
        "validator": VALIDATOR_EXACT_TRANSPORT,
        "target_basis": "test fixture",
    }
    base.update(overrides)
    return LanguageLessonSpec(**base)


def test_language_compile_publish_load_schema_roundtrip(tmp_path) -> None:
    curriculum = _compile()
    assert curriculum.actual_family_split_counts == curriculum.requested_family_split_counts
    assert len(curriculum.cases) == 349

    path = publish_language_foundations_curriculum(curriculum, state_root=tmp_path / "State")
    restored = load_first_form_curriculum(path)
    assert restored.manifest_id == curriculum.manifest_id
    assert restored.to_canonical_dict() == curriculum.to_canonical_dict()
    assert [case.case_id for case in restored.cases] == [
        case.case_id for case in curriculum.cases
    ]
    assert publish_language_foundations_curriculum(restored, state_root=tmp_path / "State") == path


def test_language_immutable_manifest_rejects_mutation(tmp_path) -> None:
    curriculum = _compile()
    path = publish_language_foundations_curriculum(curriculum, state_root=tmp_path / "State")
    body = json.loads(path.read_text(encoding="utf-8"))
    body["cases"][0]["competency"] = "tampered"
    path.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable language manifest disagrees"):
        publish_language_foundations_curriculum(curriculum, state_root=tmp_path / "State")


def test_language_verifiers_pass_compiled_curriculum() -> None:
    curriculum = _compile()
    assert find_language_target_leaks(curriculum.cases) == ()
    verify_language_no_target_leakage(curriculum.cases)
    verify_language_visibility_consistency(curriculum.cases)
    verify_language_transfer_disjointness(curriculum.cases)


def test_language_leakage_verifier_fails_on_planted_leak() -> None:
    case = _case(
        spec=_spec(prompt="Fix the spelling: helo. Hint: hello", answer="hello"),
        identity_text=IDENTITY,
        sequence=0,
    )
    leaks = find_language_target_leaks([case])
    assert any(leak.surface == "field:user_input" for leak in leaks)
    with pytest.raises(ValueError, match="hidden-target leakage"):
        verify_language_no_target_leakage([case])


def test_language_leakage_verifier_checks_identity_for_multiword_payloads() -> None:
    case = _case(
        spec=_spec(prompt="Repair the phrase.", answer="Axon is Axon"),
        identity_text=IDENTITY,
        sequence=0,
    )
    leaks = find_language_target_leaks([case])
    assert any(leak.surface == "field:identity" for leak in leaks)


def test_language_visibility_consistency_fails_on_false_visible_claim() -> None:
    case = _case(
        spec=_spec(
            prompt="Fix the spelling: helo",
            answer="hello",
            visibility=TargetVisibility.VISIBLE_BY_DESIGN,
        ),
        identity_text=IDENTITY,
        sequence=0,
    )
    with pytest.raises(ValueError, match="visible-by-design target is absent"):
        verify_language_visibility_consistency([case])


def test_language_transfer_disjointness_fails_on_cross_split_item() -> None:
    train_case = _case(spec=_spec(split="train", transfer_item="stem:x"), identity_text=IDENTITY, sequence=0)
    heldout_case = _case(
        spec=_spec(split="heldout", transfer_item="stem:x", answer="hallo"),
        identity_text=IDENTITY,
        sequence=0,
    )
    with pytest.raises(ValueError, match="transfer items cross splits"):
        verify_language_transfer_disjointness([train_case, heldout_case])


def test_language_split_lineage_disjointness() -> None:
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


def test_language_deterministic_rebuild_identical_manifest_id() -> None:
    first = _compile()
    second = _compile()
    assert first.manifest_id == second.manifest_id
    assert first.to_canonical_dict() == second.to_canonical_dict()


def test_language_budget_shortfall_and_selection() -> None:
    requested = tuple(
        (family, (counts[0] + 1, counts[1], counts[2]))
        for family, counts in DEFAULT_LANGUAGE_SPLIT_COUNTS
    )
    with pytest.raises(ValueError, match="cannot satisfy declared budgets"):
        _compile(requested_counts=requested)
    reduced = _compile(
        requested_counts=(
            ("L0", (30, 8, 8)),
            ("L1", (40, 10, 10)),
            ("L2", (52, 11, 10)),
            ("L3", (50, 12, 11)),
            ("L4", (60, 13, 12)),
        )
    )
    assert dict(reduced.actual_family_split_counts)["L0"] == (30, 8, 8)
    assert dict(reduced.excluded_counts)["language_fixtures_not_selected"] == 12


def test_language_default_counts_and_contract() -> None:
    curriculum = _compile()
    assert curriculum.requested_family_split_counts == DEFAULT_LANGUAGE_SPLIT_COUNTS
    assert {family for family, _ in curriculum.actual_family_split_counts} == set(
        LANGUAGE_FOUNDATIONS_FAMILIES
    )
    competencies = {case.competency for case in curriculum.cases}
    assert competencies == set(COMPETENCY_PREREQUISITES)
    for case in curriculum.cases:
        assert case.eligibility is TeachingEligibility.VERIFIED_TARGET
        assert case.family in LANGUAGE_FOUNDATIONS_FAMILIES
        tags = case.episode.mechanism_tags
        assert len([tag for tag in tags if tag.startswith("target_validator:")]) == 1
        assert len([tag for tag in tags if tag.startswith("visibility:")]) == 1


def test_language_rule_transfer_holdouts_use_unseen_items() -> None:
    curriculum = _compile()
    pools: dict[str, dict[str, set[str]]] = {}
    for case in curriculum.cases:
        items = tuple(
            tag[len("transfer_item:") :] for tag in case.episode.mechanism_tags
            if tag.startswith("transfer_item:")
        )
        if items:
            pools.setdefault(case.competency, {"train": set(), "heldout": set(), "regression": set()})
            pools[case.competency][case.episode.split].update(items)
    assert pools, "expected transfer-tagged competencies"
    for competency, splits in pools.items():
        assert splits["heldout"].isdisjoint(splits["train"]), competency
        assert splits["regression"].isdisjoint(splits["train"]), competency
        assert splits["heldout"].isdisjoint(splits["regression"]), competency


def test_language_no_op_and_addressed_edit_targets() -> None:
    curriculum = _compile()
    no_ops = [case for case in curriculum.cases if case.competency == "no_op_distinction"]
    assert len(no_ops) == 6
    for case in no_ops:
        consolidated = case.episode.targets[2]
        assert consolidated.decision is ReasoningDecision.NO_OP
        assert consolidated.supervision_weight == 1.0
        assert consolidated.operation is None
    edits = [case for case in curriculum.cases if case.competency == "addressed_edit"]
    assert len(edits) == 6
    for case in edits:
        consolidated = case.episode.targets[2]
        assert consolidated.decision is ReasoningDecision.DELTA
        assert consolidated.operation in {
            ReasoningOperationKind.INSERT,
            ReasoningOperationKind.DELETE,
            ReasoningOperationKind.REPLACE,
        }
        starter = case.episode.snapshot.region(LogicalRegion.RESPONSE_DRAFT).text
        assert 0 <= consolidated.start <= consolidated.end <= len(starter)


def test_language_unicode_transport_cases_present() -> None:
    curriculum = _compile()
    copy_payloads = {
        case.episode.targets[2].payload
        for case in curriculum.cases
        if case.competency == "exact_copy"
    }
    for expected in ("λ", "🧠", "東京", "naïve", "café", "φίλος", "Москва"):
        assert expected in copy_payloads
        assert any(unit > 127 for unit in expected.encode("utf-8"))


def test_language_tournament_loader_compatibility(tmp_path) -> None:
    curriculum = _compile()
    path = publish_language_foundations_curriculum(curriculum, state_root=tmp_path / "State")

    # Mirrors scripts/run_d64_tournament.py lines 147-157 exactly.
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body.get("schema") == FFCS_MANIFEST_SCHEMA == "axon-first-form-curriculum-v1"
    loaded = load_first_form_curriculum(path)
    assert loaded.manifest_id == curriculum.manifest_id
    assert len(loaded.living_curriculum.episodes) == len(loaded.cases)


def test_language_token_matcher_semantics() -> None:
    assert _tokens('Fix: "The dogs run outside."  now') == (
        "fix",
        "the",
        "dogs",
        "run",
        "outside",
        "now",
    )

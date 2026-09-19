"""Tests for the ABC sequence curriculum."""
from __future__ import annotations

from training.abc_sequence_curriculum import compile_abc_sequence
from training.first_form_curriculum import (
    TeachingEligibility,
    load_first_form_curriculum,
    publish_first_form_curriculum,
)


def _compile(**kwargs):
    defaults = dict(
        identity_text="Axon is Axon. Every brother attends the complete exact field.",
        requested_counts=(("ABC", (16, 4, 4)),),
    )
    defaults.update(kwargs)
    return compile_abc_sequence(**defaults)


def test_abc_compiles_with_correct_budgets() -> None:
    curriculum = _compile()
    assert len(curriculum.cases) == 24
    assert curriculum.actual_family_split_counts == (("ABC", (16, 4, 4)),)


def test_all_cases_are_verified_target() -> None:
    curriculum = _compile()
    for case in curriculum.cases:
        assert case.eligibility is TeachingEligibility.VERIFIED_TARGET


def test_forward_backward_and_roundtrip_variants_present() -> None:
    curriculum = _compile()
    competencies = {case.competency for case in curriculum.cases}
    assert "abc_forward_full" in competencies
    assert "abc_backward_full" in competencies
    assert "abc_roundtrip_full" in competencies
    assert "abc_uppercase_forward" in competencies
    assert "abc_digits_forward" in competencies or "digits_forward" in competencies
    assert "abc_skip_pattern" in competencies


def test_forward_answer_is_exact_alphabet() -> None:
    curriculum = _compile()
    for case in curriculum.cases:
        if case.competency == "abc_forward_full":
            payload = case.episode.targets[-1].payload
            assert payload == "abcdefghijklmnopqrstuvwxyz"


def test_backward_answer_is_reversed_alphabet() -> None:
    curriculum = _compile()
    for case in curriculum.cases:
        if case.competency == "abc_backward_full":
            payload = case.episode.targets[-1].payload
            assert payload == "zyxwvutsrqponmlkjihgfedcba"


def test_roundtrip_answer_goes_forward_then_back() -> None:
    curriculum = _compile()
    for case in curriculum.cases:
        if case.competency == "abc_roundtrip_full":
            payload = case.episode.targets[-1].payload
            # Should be alphabet forward + reversed (z...a, with duplicate z)
            assert payload.startswith("abcdefghijklmnopqrstuvwxyz")
            assert payload.endswith("ba")
            assert len(payload) == 52  # 26 + 26 (z appears twice)


def test_cortex_contains_visible_alphabet() -> None:
    curriculum = _compile()
    for case in curriculum.cases:
        cortex = case.episode.snapshot.region("cortex").text
        assert "alphabet" in cortex.lower() or "uppercase" in cortex.lower() or "digits" in cortex.lower()


def test_publish_and_reload_roundtrip(tmp_path) -> None:
    curriculum = _compile()
    path = publish_first_form_curriculum(curriculum, state_root=tmp_path / "State")
    restored = load_first_form_curriculum(path)
    assert restored.manifest_id == curriculum.manifest_id
    assert len(restored.cases) == len(curriculum.cases)


def test_deterministic_compilation() -> None:
    c1 = _compile()
    c2 = _compile()
    assert c1.manifest_id == c2.manifest_id
    assert [c.case_id for c in c1.cases] == [c.case_id for c in c2.cases]


def test_d64_compilation_roundtrip() -> None:
    from runtime.field import D64FieldCompiler
    curriculum = _compile()
    for case in curriculum.cases:
        compiled = D64FieldCompiler().compile(case.episode.snapshot)
        compiled.verify_roundtrip(case.episode.snapshot)
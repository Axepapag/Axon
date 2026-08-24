from __future__ import annotations

import pytest

from runtime.heart import (
    CRITICAL_SEMANTIC_CLASSES,
    HeartEnsemblePolicy,
    HeartSemanticFidelityEvidence,
    HeartTranslatorDescriptor,
    HeartTranslatorRole,
    HeartTranslatorState,
    evaluate_heart_translator_promotion,
)
from runtime.trainer import OrganKind, ParameterModuleDescriptor


def _descriptor(*, generation_id: str = "heart-g1") -> HeartTranslatorDescriptor:
    return HeartTranslatorDescriptor(
        translator_id="heart-translator-a",
        generation_id=generation_id,
        role=HeartTranslatorRole.TRANSLATOR,
        architecture="heterogeneous-heart-transformer-v1",
        d_model=128,
        source_dialects=("rail-64d", "english"),
        destination_dialects=("heart-interlingua", "rail-4096d"),
        state=HeartTranslatorState.CANDIDATE,
    )


def _perfect_evidence(descriptor: HeartTranslatorDescriptor) -> HeartSemanticFidelityEvidence:
    return HeartSemanticFidelityEvidence(
        descriptor_id=descriptor.descriptor_id,
        evaluation_id="heldout-heart-translation-v1",
        heldout_case_count=10000,
        grounded_roundtrip_rate=1.0,
        aggregate_semantic_fidelity=1.0,
        critical_class_rates=tuple((name, 1.0) for name in CRITICAL_SEMANTIC_CLASSES),
        counterfactual_use_proven=True,
        regression_failures=0,
    )


def test_heart_translator_descriptor_is_content_addressed_and_requires_d64_floor() -> None:
    descriptor = _descriptor()
    assert descriptor.descriptor_id == _descriptor().descriptor_id
    assert descriptor.d_model == 128
    assert descriptor.source_dialects == ("english", "rail-64d")
    with pytest.raises(ValueError, match=">= 64"):
        HeartTranslatorDescriptor(
            translator_id="too-small",
            generation_id="g1",
            role="translator",
            architecture="test",
            d_model=32,
            source_dialects=("english",),
            destination_dialects=("heart-interlingua",),
        )


def test_heart_ensemble_policy_requires_three_serving_plus_candidate_lane() -> None:
    policy = HeartEnsemblePolicy()
    assert policy.minimum_serving_translators == 3
    assert policy.minimum_candidate_lanes == 1
    assert policy.minimum_grounded_roundtrip_rate == pytest.approx(0.9999)
    assert policy.minimum_aggregate_semantic_fidelity == pytest.approx(0.999)
    with pytest.raises(ValueError, match="at least three"):
        HeartEnsemblePolicy(minimum_serving_translators=2)
    with pytest.raises(ValueError, match="candidate lane"):
        HeartEnsemblePolicy(minimum_candidate_lanes=0)


def test_perfect_heldout_fidelity_passes_deterministic_promotion_floor() -> None:
    descriptor = _descriptor()
    policy = HeartEnsemblePolicy()
    evidence = _perfect_evidence(descriptor)
    decision = evaluate_heart_translator_promotion(descriptor, evidence, policy)
    assert decision.passed is True
    assert decision.reasons == ()


def test_single_critical_semantic_error_blocks_promotion_despite_high_average() -> None:
    descriptor = _descriptor()
    rates = [(name, 1.0) for name in CRITICAL_SEMANTIC_CLASSES]
    rates[[name for name, _ in rates].index("modality")] = ("modality", 0.9999)
    evidence = HeartSemanticFidelityEvidence(
        descriptor_id=descriptor.descriptor_id,
        evaluation_id="heldout-heart-translation-v1",
        heldout_case_count=10000,
        grounded_roundtrip_rate=1.0,
        aggregate_semantic_fidelity=0.99999,
        critical_class_rates=tuple(rates),
        counterfactual_use_proven=True,
    )
    decision = evaluate_heart_translator_promotion(descriptor, evidence, HeartEnsemblePolicy())
    assert decision.passed is False
    assert decision.reasons == ("critical_semantic_class_not_perfect:modality",)


def test_roundtrip_counterfactual_and_regression_failures_each_block_promotion() -> None:
    descriptor = _descriptor()
    evidence = HeartSemanticFidelityEvidence(
        descriptor_id=descriptor.descriptor_id,
        evaluation_id="heldout-heart-translation-v1",
        heldout_case_count=50000,
        grounded_roundtrip_rate=0.9998,
        aggregate_semantic_fidelity=0.998,
        critical_class_rates=tuple((name, 1.0) for name in CRITICAL_SEMANTIC_CLASSES),
        counterfactual_use_proven=False,
        regression_failures=1,
    )
    decision = evaluate_heart_translator_promotion(descriptor, evidence, HeartEnsemblePolicy())
    assert decision.passed is False
    assert "grounded_roundtrip_below_floor" in decision.reasons
    assert "aggregate_semantic_fidelity_below_floor" in decision.reasons
    assert "counterfactual_use_not_proven" in decision.reasons
    assert "regression_failures:1" in decision.reasons


def test_fidelity_evidence_is_generation_bound_and_trainer_recognizes_heart_tissue() -> None:
    descriptor = _descriptor()
    other = _descriptor(generation_id="heart-g2")
    with pytest.raises(ValueError, match="different Heart translator"):
        evaluate_heart_translator_promotion(other, _perfect_evidence(descriptor), HeartEnsemblePolicy())

    trainer_descriptor = ParameterModuleDescriptor(
        module_id="heart-translator-a",
        organ_kind=OrganKind.HEART_TRANSLATION_CORE,
        generation_id="heart-g1",
        architecture="heterogeneous-heart-transformer-v1",
        d_model=128,
        tags=("heart", "translation"),
    )
    assert trainer_descriptor.organ_kind is OrganKind.HEART_TRANSLATION_CORE

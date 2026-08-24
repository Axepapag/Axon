from __future__ import annotations

from pathlib import Path

import pytest
import torch

from runtime.heart import HeartEnsemblePolicy, evaluate_heart_translator_promotion
from runtime.heart.translation_core import (
    HEART_SEMANTIC_LABELS,
    HEART_TRANSLATION_ARCHITECTURE,
    HeartTranslationCore,
    HeartTranslationCoreConfig,
)
from runtime.trainer import OrganKind, ParameterModuleDescriptor, TrainerControlPlane, inspect_trainer_state
from scripts.train_heart_translation_smoke import run_heart_translation_smoke
from training.heart_translation import (
    HeartTranslationTrainingObjective,
    build_heart_translation_curriculum,
    collate_heart_translation_cases,
    evaluate_heart_translation_model,
    heart_translation_loss,
)


def _small_model() -> HeartTranslationCore:
    torch.manual_seed(17)
    return HeartTranslationCore(
        HeartTranslationCoreConfig(
            d_model=64,
            n_heads=4,
            n_layers=1,
            ffn_dim=128,
            dropout=0.0,
            source_page_chars=32,
        )
    )


def test_first_heart_translation_core_is_shallow_ffn_heavy_and_16d_grounded() -> None:
    model = HeartTranslationCore()
    assert model.cfg.d_model == 64
    assert model.cfg.n_layers == 2
    assert model.cfg.n_heads == 4
    assert model.cfg.ffn_dim == 4096
    assert model.bank16.shape[1] == 16
    assert model.char_lift.shape == (16, 64)
    assert "bank16" in dict(model.named_buffers())
    assert "char_lift" in dict(model.named_buffers())
    assert not model.bank16.requires_grad
    assert not model.char_lift.requires_grad
    assert sum(parameter.numel() for parameter in model.parameters()) > 1_000_000


def test_heart_curriculum_is_content_addressed_disjoint_and_counterfactual_complete(tmp_path: Path) -> None:
    first = build_heart_translation_curriculum()
    second = build_heart_translation_curriculum()
    assert first.curriculum_id == second.curriculum_id
    assert first.train_manifest_id == second.train_manifest_id
    assert first.heldout_manifest_id == second.heldout_manifest_id
    assert len(first.train_cases) == 590
    assert len(first.heldout_cases) == 77
    assert len(first.regression_cases) == 21
    assert len(first.counterfactual_pairs) == 8
    complete_field = [case for case in first.train_cases if ":complete-field:" in case.spec_id]
    assert len(complete_field) == 8
    assert {semantic for case in complete_field for semantic in case.critical_classes} >= set(
        HEART_SEMANTIC_LABELS
    ) | {"referent_identity", "grounding_provenance"}
    assert all(case.referent_start > 256 and len(case.source_text) > 256 for case in complete_field)
    assert all("heart-translation-v3" in case.provenance for case in complete_field)
    assert all("heart-translation-v2" not in case.provenance for case in complete_field)
    assert {case.spec_id for case in first.train_cases}.isdisjoint(
        {case.spec_id for case in first.heldout_cases + first.regression_cases}
    )
    artifact = first.write(tmp_path)
    assert artifact.exists()
    assert first.write(tmp_path) == artifact


def test_heart_training_objective_is_content_addressed_and_changes_loss_recipe(tmp_path: Path) -> None:
    default = HeartTranslationTrainingObjective()
    same = HeartTranslationTrainingObjective()
    semantic_heavy = HeartTranslationTrainingObjective(translation_weight=1.0, semantic_weight=2.0, pointer_weight=1.0)
    assert default.objective_id == same.objective_id
    assert default.objective_id != semantic_heavy.objective_id
    artifact = default.write(tmp_path)
    assert artifact.exists()
    assert default.write(tmp_path) == artifact

    model = _small_model()
    curriculum = build_heart_translation_curriculum()
    batch = collate_heart_translation_cases(model, curriculum.train_cases[:4])
    default_loss = heart_translation_loss(model, batch, default)
    semantic_heavy_loss = heart_translation_loss(model, batch, semantic_heavy)
    assert float(default_loss.total.detach()) != float(semantic_heavy_loss.total.detach())


def test_heart_translation_loss_uses_text_semantics_and_grounding() -> None:
    model = _small_model()
    curriculum = build_heart_translation_curriculum()
    batch = collate_heart_translation_cases(model, curriculum.train_cases[:6])
    loss = heart_translation_loss(model, batch)
    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.translation)
    assert torch.isfinite(loss.semantic)
    assert torch.isfinite(loss.pointers)
    assert float(loss.total.detach()) > 0.0
    loss.total.backward()
    assert model.semantic_heads["modality"].weight.grad is not None
    assert model.referent_start_query.weight.grad is not None
    assert model.decoder_output.weight.grad is not None


def test_untrained_heart_model_produces_real_fidelity_evidence_and_fails_promotion() -> None:
    model = _small_model()
    curriculum = build_heart_translation_curriculum()
    report = evaluate_heart_translation_model(model, curriculum, descriptor_id="candidate-heart")
    assert report.heldout_case_count == len(curriculum.heldout_cases)
    assert report.evidence.descriptor_id == "candidate-heart"
    assert report.evidence.grounded_roundtrip_rate < 1.0
    assert not report.evidence.counterfactual_use_proven
    descriptor = type("Descriptor", (), {"descriptor_id": "candidate-heart"})()
    decision = evaluate_heart_translator_promotion(descriptor, report.evidence, HeartEnsemblePolicy())
    assert not decision.passed
    assert decision.reasons


def test_heart_translation_core_registers_as_explicit_trainer_organ(tmp_path: Path) -> None:
    model = _small_model()
    descriptor = ParameterModuleDescriptor(
        module_id="heart-translator-a",
        organ_kind=OrganKind.HEART_TRANSLATION_CORE,
        generation_id="heart-init-v1",
        architecture=HEART_TRANSLATION_ARCHITECTURE,
        d_model=64,
        tags=("heart", "translator", "non-serving-base"),
    )
    control = TrainerControlPlane.active(state_root=tmp_path)
    control.declare_expected((descriptor,))
    control.register(descriptor, model)
    inventory = control.snapshot_inventory(exact_value_hashes=True)
    manifest = inventory.module(descriptor.module_id)
    assert manifest.descriptor.organ_kind is OrganKind.HEART_TRANSLATION_CORE
    assert manifest.parameter_count == sum(parameter.numel() for parameter in model.parameters())
    assert manifest.buffer_numel == model.bank16.numel() + model.char_lift.numel()
    control.close()


def test_heart_translation_config_rejects_sub_64d_or_bad_head_partition() -> None:
    with pytest.raises(ValueError, match=">= 64"):
        HeartTranslationCoreConfig(d_model=32)
    with pytest.raises(ValueError, match="divisible"):
        HeartTranslationCoreConfig(d_model=65, n_heads=4)


def test_smoke_generation_identity_binds_training_objective(tmp_path: Path) -> None:
    cfg = HeartTranslationCoreConfig(
        d_model=64,
        n_heads=4,
        n_layers=1,
        ffn_dim=128,
        dropout=0.0,
        source_page_chars=32,
    )
    first = run_heart_translation_smoke(
        state_root=tmp_path / "first",
        steps=2,
        batch_size=2,
        seed=31,
        device="cpu",
        model_config=cfg,
        training_objective=HeartTranslationTrainingObjective(),
    )
    second = run_heart_translation_smoke(
        state_root=tmp_path / "second",
        steps=2,
        batch_size=2,
        seed=31,
        device="cpu",
        model_config=cfg,
        training_objective=HeartTranslationTrainingObjective(
            translation_weight=1.0,
            semantic_weight=2.0,
            pointer_weight=1.0,
        ),
    )
    assert first.base_generation_id == second.base_generation_id
    assert first.architecture_config_id == second.architecture_config_id
    assert first.training_objective_id != second.training_objective_id
    assert first.candidate_generation_id != second.candidate_generation_id
    assert first.run_id != second.run_id


def test_trainer_governed_heart_smoke_rejects_without_activation(tmp_path: Path) -> None:
    result = run_heart_translation_smoke(
        state_root=tmp_path,
        steps=2,
        batch_size=2,
        seed=29,
        device="cpu",
        model_config=HeartTranslationCoreConfig(
            d_model=64,
            n_heads=4,
            n_layers=1,
            ffn_dim=128,
            dropout=0.0,
            source_page_chars=32,
        ),
    )
    assert result.candidate_status == "rejected_not_activated"
    assert not result.heart_promotion_passed
    assert result.heart_promotion_passed == result.generic_trainer_gate_passed
    assert result.preflight_receipt_id
    assert len(result.checkpoint_ids) == 2
    assert Path(result.summary_path).exists()
    inspection = inspect_trainer_state(state_root=tmp_path)
    assert inspection.active_generation_pointers == ()
    counts = dict(inspection.artifact_counts)
    assert counts["activation_receipts"] == 0
    assert counts["promotion_proposals"] == 0
    assert counts["learning_policies"] == 1
    assert counts["capacity_contracts"] == 1
    assert counts["preflight_receipts"] == 1


def test_heart_translation_reads_and_addresses_every_character_beyond_old_192_cap() -> None:
    model = _small_model()
    source_length = 521
    source_indices = torch.arange(source_length, dtype=torch.long).remainder(model.vocab_size).unsqueeze(0)
    source_mask = torch.ones_like(source_indices, dtype=torch.bool)
    output = model(
        source_indices,
        source_mask,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([[model.bos_index]]),
    )

    assert output.source_coverage.complete
    assert output.source_coverage.source_characters == (source_length,)
    assert output.source_coverage.visited_characters_per_sweep == (
        (source_length,),
        (source_length,),
    )
    assert output.source_coverage.page_spans[0][0] == (0, 32)
    assert output.source_coverage.page_spans[0][-1] == (512, 521)
    assert output.referent_start_logits.shape == (1, source_length)
    assert output.grounding_end_logits.shape == (1, source_length)


def test_heart_translation_target_training_has_no_learned_character_ceiling() -> None:
    model = _small_model()
    source_indices = torch.arange(257, dtype=torch.long).remainder(model.vocab_size).unsqueeze(0)
    source_mask = torch.ones_like(source_indices, dtype=torch.bool)
    decoder_input = torch.full((1, 257), model.bos_index, dtype=torch.long)
    output = model(
        source_indices,
        source_mask,
        torch.tensor([0]),
        torch.tensor([1]),
        decoder_input,
    )
    assert output.target_log_probs.shape[:2] == (1, 257)


def test_heart_greedy_compute_budget_reports_nontermination_instead_of_partial_success() -> None:
    model = _small_model()
    with torch.no_grad():
        model.decoder_output.weight.zero_()
        model.decoder_output.bias.zero_()
        model.copy_gate.weight.zero_()
        model.copy_gate.bias.fill_(100.0)
    source = torch.tensor([[model.char_to_index["a"]]], dtype=torch.long)
    result = model.greedy_translate(
        source,
        torch.ones_like(source, dtype=torch.bool),
        torch.tensor([0]),
        torch.tensor([1]),
        max_chars=1,
    )[0]

    assert result.text
    assert result.generated_characters == 1
    assert result.terminated is False

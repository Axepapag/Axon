from __future__ import annotations

import math
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
from substrate import default_alphabet
from training.heart_translation import (
    HeartDecoderGeneralizationObjective,
    HeartTranslationTrainingObjective,
    build_heart_decoder_generalization_curriculum,
    build_heart_decoder_long_position_curriculum,
    build_heart_decoder_mechanism_curriculum,
    build_heart_translation_curriculum,
    collate_heart_translation_cases,
    deterministic_length_bucketed_batches,
    deterministic_training_batches,
    evaluate_heart_decoder_diagnostics,
    evaluate_heart_translation_model,
    heart_decoder_generalization_loss,
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


def test_decoder_mechanism_curriculum_is_separate_exact_and_complete_field(tmp_path: Path) -> None:
    first = build_heart_decoder_mechanism_curriculum()
    second = build_heart_decoder_mechanism_curriculum()
    assert first.curriculum_id == second.curriculum_id
    assert first.train_manifest_id == second.train_manifest_id
    assert first.heldout_manifest_id == second.heldout_manifest_id
    assert all(
        case.source_text == case.target_text
        for case in first.train_cases + first.heldout_cases
    )
    complete_field = [
        case
        for case in first.train_cases + first.heldout_cases
        if case.spec_id.endswith(":complete-field")
    ]
    assert len(complete_field) == 2
    assert all(len(case.source_text) > 256 and case.referent_start > 256 for case in complete_field)
    artifact = first.write(tmp_path)
    assert artifact.exists()
    assert first.write(tmp_path) == artifact


def test_decoder_generalization_curriculum_is_unseen_diverse_and_counterfactual(tmp_path: Path) -> None:
    first = build_heart_decoder_generalization_curriculum()
    second = build_heart_decoder_generalization_curriculum()
    assert first.curriculum_id == second.curriculum_id
    assert first.curriculum_id == "00d000ce720eb8be5f43bbbb30fd2a672dee11f807a369a6c71e43ab15113f88"
    assert len(first.train_cases) == 37
    assert len(first.heldout_cases) == 21
    assert len(first.replay_case_ids) == 6
    assert {pair.position_kind for pair in first.counterfactual_pairs} == {"head", "middle", "tail"}
    assert {case.source_text for case in first.train_cases}.isdisjoint(
        {case.source_text for case in first.heldout_cases}
    )
    assert {character for case in first.train_cases for character in case.source_text} == set(
        default_alphabet()
    )
    assert max(map(lambda case: len(case.source_text), first.heldout_cases)) > max(
        map(lambda case: len(case.source_text), first.train_cases)
    )
    heldout = {case.case_id: case for case in first.heldout_cases}
    for pair in first.counterfactual_pairs:
        left = heldout[pair.left_case_id].source_text
        right = heldout[pair.right_case_id].source_text
        assert [
            index
            for index, chars in enumerate(zip(left, right, strict=True))
            if chars[0] != chars[1]
        ] == [
            pair.changed_position
        ]
    artifact = first.write(tmp_path)
    assert artifact.exists()
    assert first.write(tmp_path) == artifact


def test_decoder_long_position_curriculum_adds_boundaries_tails_and_full_replay(tmp_path: Path) -> None:
    base = build_heart_decoder_generalization_curriculum()
    first = build_heart_decoder_long_position_curriculum()
    second = build_heart_decoder_long_position_curriculum()

    assert first.curriculum_id == second.curriculum_id
    assert first.curriculum_id != base.curriculum_id
    assert len(first.train_cases) == 64
    assert len(first.heldout_cases) == 46
    assert len(first.replay_case_ids) == len(base.train_cases) == 37
    assert set(first.replay_case_ids) == {case.case_id for case in base.train_cases}
    assert max(len(case.source_text) for case in first.train_cases) == 640
    assert max(len(case.source_text) for case in first.heldout_cases) == 769
    labels = {pair.probe_label for pair in first.counterfactual_pairs if pair.probe_label}
    assert {
        "heldout-first-page-last",
        "heldout-second-page-first",
        "heldout-second-page-last",
        "heldout-third-page-first",
        "heldout-late-tail",
        "heldout-extrapolated-tail",
    } <= labels
    changed_positions = {pair.changed_position for pair in first.counterfactual_pairs}
    assert {0, 254, 255, 256, 257, 510, 511, 512, 513, 699, 768} <= changed_positions
    assert {case.source_text for case in first.train_cases}.isdisjoint(
        {case.source_text for case in first.heldout_cases}
    )
    artifact = first.write(tmp_path)
    assert artifact.exists()
    assert first.write(tmp_path) == artifact


def test_decoder_generalization_batches_cover_one_bucketed_epoch_without_omission() -> None:
    curriculum = build_heart_decoder_generalization_curriculum()
    batch_size = 4
    def bucket_for(length: int) -> str:
        if length <= 32:
            return "a"
        if length <= 64:
            return "b"
        if length <= 128:
            return "c"
        if length < 256:
            return "d"
        if length == 256:
            return "e"
        if length <= 512:
            return "f"
        return "g"

    bucket_counts: dict[str, int] = {}
    for case in curriculum.train_cases:
        bucket = bucket_for(len(case.source_text))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    epoch_steps = sum(math.ceil(count / batch_size) for count in bucket_counts.values())
    batches = deterministic_length_bucketed_batches(
        curriculum.train_cases,
        batch_size=batch_size,
        steps=epoch_steps,
        seed=91,
    )
    ids = [case.case_id for batch in batches for case in batch]
    assert len(ids) == len(curriculum.train_cases)
    assert len(ids) == len(set(ids))
    assert set(ids) == {case.case_id for case in curriculum.train_cases}
    assert all(len({bucket_for(len(case.source_text)) for case in batch}) == 1 for batch in batches)


def test_decoder_generalization_loss_trains_discrete_copy_and_diagonal_alignment() -> None:
    model = _small_model()
    curriculum = build_heart_decoder_generalization_curriculum()
    batch = collate_heart_translation_cases(model, curriculum.train_cases[:3])
    loss = heart_decoder_generalization_loss(
        model,
        batch,
        HeartDecoderGeneralizationObjective(),
    )
    assert all(
        torch.isfinite(value)
        for value in (
            loss.total,
            loss.translation,
            loss.diagonal_alignment,
            loss.copy_route,
            loss.eos_route,
            loss.semantic,
            loss.pointers,
        )
    )
    loss.total.backward()
    assert model.decoder_memory_attention.in_proj_weight.grad is not None
    assert model.copy_gate.weight.grad is not None


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


def test_heart_training_batches_cover_each_shuffled_epoch_without_omissions() -> None:
    curriculum = build_heart_translation_curriculum()
    batch_size = 8
    steps = (len(curriculum.train_cases) + batch_size - 1) // batch_size
    first = deterministic_training_batches(
        curriculum,
        batch_size=batch_size,
        steps=steps,
        seed=20260824,
    )
    second = deterministic_training_batches(
        curriculum,
        batch_size=batch_size,
        steps=steps,
        seed=20260824,
    )
    first_epoch_ids = [
        case.case_id for batch in first for case in batch
    ][: len(curriculum.train_cases)]
    assert first == second
    assert len(first_epoch_ids) == len(set(first_epoch_ids))
    assert set(first_epoch_ids) == {case.case_id for case in curriculum.train_cases}


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


def test_heart_decoder_trace_exposes_normalized_alignment_and_copy_gate() -> None:
    model = _small_model()
    curriculum = build_heart_translation_curriculum()
    batch = collate_heart_translation_cases(model, curriculum.train_cases[:3])
    output = model(
        batch.source_indices,
        batch.source_mask,
        batch.source_dialect_ids,
        batch.destination_dialect_ids,
        batch.decoder_input_ids,
        source_cells16=batch.source_cells16,
        source_positions=batch.source_positions,
    )

    trace = output.decoder_trace
    assert trace.target_log_probs is output.target_log_probs
    assert trace.memory_attention.shape == (
        batch.source_indices.shape[0],
        batch.target_ids.shape[1],
        batch.source_indices.shape[1],
    )
    assert trace.generation_gate.shape == (*batch.target_ids.shape, 1)
    assert torch.all((trace.generation_gate >= 0.0) & (trace.generation_gate <= 1.0))
    assert torch.allclose(
        trace.memory_attention.sum(dim=-1),
        torch.ones_like(trace.memory_attention[..., 0]),
        atol=1e-6,
    )


def test_decoder_diagnostic_is_content_addressed_and_localizes_failure() -> None:
    model = _small_model()
    curriculum = build_heart_translation_curriculum()
    cases = curriculum.heldout_cases[:4]
    first = evaluate_heart_decoder_diagnostics(
        model,
        cases,
        descriptor_id="untrained-heart",
        checkpoint_id="untrained-heart-checkpoint",
        curriculum_id=curriculum.curriculum_id,
        split="heldout:first-4",
    )
    second = evaluate_heart_decoder_diagnostics(
        model,
        cases,
        descriptor_id="untrained-heart",
        checkpoint_id="untrained-heart-checkpoint",
        curriculum_id=curriculum.curriculum_id,
        split="heldout:first-4",
    )

    assert first.diagnostic_id == second.diagnostic_id
    assert first.case_count == 4
    assert len(first.case_diagnostics) == 4
    assert first.position_accuracy
    assert 0.0 <= first.teacher_forced_character_accuracy <= 1.0
    assert 0.0 <= first.teacher_forced_eos_accuracy <= 1.0
    assert 0.0 <= first.greedy_termination_rate <= 1.0
    assert all(item.diagnostic_id for item in first.case_diagnostics)
    assert all(
        item.greedy_first_divergence is None
        or 0 <= item.greedy_first_divergence <= item.target_characters
        for item in first.case_diagnostics
    )


def test_untrained_heart_model_produces_real_fidelity_evidence_and_fails_promotion() -> None:
    model = _small_model()
    curriculum = build_heart_translation_curriculum()
    report = evaluate_heart_translation_model(model, curriculum, descriptor_id="candidate-heart")
    assert report.heldout_case_count == len(curriculum.heldout_cases)
    assert len(report.case_results) == report.heldout_case_count
    assert all(result.result_id for result in report.case_results)
    assert all(result.generated_characters == len(result.generated_text) for result in report.case_results)
    assert len(report.to_canonical_dict()["case_results"]) == report.heldout_case_count
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

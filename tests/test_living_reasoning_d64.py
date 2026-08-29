from __future__ import annotations

from pathlib import Path

import pytest
import torch

from runtime.field import D64FieldCompiler, LogicalRegion, SharedFieldSnapshot
from runtime.soul import (
    SoulSnapshot,
    SoulTemperature,
    apply_soul_transition,
    empty_soul_layers,
)
from runtime.trainer import (
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationPlan,
    ParameterRegistry,
    PreflightEvidenceKind,
)
from substrate import TRANSPORT_VOCAB_SIZE, encode_unicode_text
from training.living_reasoning_curriculum import (
    build_living_reasoning_smoke_curriculum,
    evaluate_living_episode,
    living_episode_objective,
)
from training.living_reasoning_d64 import (
    LivingReasoningCoreConfig,
    LivingReasoningCoreD64,
    candidate_a_config,
)
from training.living_reasoning_preflight import build_living_reasoning_preflight
from training.reasoning_tournament import (
    D64TournamentResult,
    assert_same_gate_surface,
    d64_head_geometry_tournament,
)

ROOT = Path(__file__).resolve().parent.parent


def _small_model(*, heads: int = 1) -> LivingReasoningCoreD64:
    torch.manual_seed(41)
    model = LivingReasoningCoreD64(
        LivingReasoningCoreConfig(
            n_heads=heads,
            n_layers=2,
            ffn_dim=128,
            state_tokens=2,
            page_size=2,
            dropout=0.0,
            inference_budget_transport_units=32,
        )
    )
    model.eval()
    return model


def _soul(model: LivingReasoningCoreD64, *, core_id: str = "core-a") -> SoulSnapshot:
    return SoulSnapshot(
        core_id=core_id,
        architecture_id=model.architecture_id,
        parameter_generation="g0",
        generation=0,
        parent_soul_id=None,
        layers=empty_soul_layers(),
    )


def _compiled():
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: "I am Axon.",
            LogicalRegion.USER_INPUT: "Read beginning λ, middle 🧠, and end.",
            LogicalRegion.SCRATCH: "",
        },
        tick_id=7,
    )
    return D64FieldCompiler().compile(snapshot)


def test_candidate_a_shape_is_exact_and_architecture_identity_changes_with_heads() -> None:
    candidate = candidate_a_config()
    assert candidate.d_model == 64
    assert candidate.n_heads == 1
    assert candidate.n_layers == 2
    assert candidate.ffn_dim == 131_072
    assert candidate.page_size == 32
    assert candidate.architecture_id != candidate_a_config(n_heads=2).architecture_id
    assert candidate.architecture_id != candidate_a_config(ffn_dim=4096).architecture_id


def test_candidate_a_parameter_estimate_preserves_deliberate_huge_ffn() -> None:
    model = LivingReasoningCoreD64(candidate_a_config(dropout=0.0))
    report = model.architecture_report()
    assert 33_000_000 < report["parameter_count"] < 35_000_000
    assert report["parameter_bytes_fp32"] == report["parameter_count"] * 4


def test_soul_is_inhaled_before_complete_unicode_field_and_proposal_sweeps() -> None:
    model = _small_model()
    compiled = _compiled()
    initial = _soul(model)
    first = model.forward_surfaces(
        soul=initial,
        expected_core_id="core-a",
        parameter_generation="g0",
        phase="first",
        canonical=compiled,
    )
    transition = model.exhale_transition(
        before=initial,
        exhaled_state=first.exhaled_state,
        tick_uid="tick-7",
        request_id="request-first",
        phase="first",
    )
    successor = apply_soul_transition(initial, transition)
    refined = model.forward_surfaces(
        soul=successor,
        expected_core_id="core-a",
        parameter_generation="g0",
        phase="refined",
        canonical=compiled,
        proposal_texts=('core-b proposes "preserve λ🧠"',),
    )

    assert first.canonical_coverage.complete
    assert first.canonical_coverage.page_count > len(tuple(LogicalRegion))
    assert refined.proposal_coverages[0].complete
    assert refined.soul_telemetry["decoded_soul_layers"] == 1.0
    assert refined.soul_telemetry["soul_contribution_l2"] > 0.0
    assert successor.layer(SoulTemperature.HOT).payload
    assert refined.decision_logits.shape == (1, 3)
    assert refined.operation_logits.shape == (1, 3)
    assert refined.region_logits.shape == (1, len(tuple(LogicalRegion)))


def test_unicode_decoder_trains_on_all_351_transport_categories_plus_eos() -> None:
    model = _small_model()
    compiled = _compiled()
    output = model.forward_surfaces(
        soul=_soul(model),
        expected_core_id="core-a",
        parameter_generation="g0",
        phase="first",
        canonical=compiled,
    )
    target = "λ🧠"
    logits, targets = model.decode_teacher(
        output.reader_state,
        target,
        head=1,
        memory=output.complete_memory,
    )
    assert logits.shape[-1] == TRANSPORT_VOCAB_SIZE + 2
    assert targets[0, :-1].tolist() == list(encode_unicode_text(target))
    assert int(targets[0, -1]) == TRANSPORT_VOCAB_SIZE + 1


def test_three_phase_training_unroll_uses_exact_causal_soul_successors() -> None:
    model = _small_model()
    initial = _soul(model)
    unroll = model.unroll_runtime_phases(
        initial_soul=initial,
        expected_core_id="core-a",
        parameter_generation="g0",
        tick_uid="tick-training-1",
        canonical=_compiled(),
        first_workspace_text='{"first":"proposal"}',
        refined_workspace_text='{"refined":"proposal"}',
    )
    assert [transition.phase for transition in unroll.transitions] == [
        "first",
        "refined",
        "consolidated",
    ]
    assert [soul.generation for soul in unroll.souls] == [0, 1, 2, 3]
    assert unroll.outputs[0].proposal_coverages == ()
    assert len(unroll.outputs[1].proposal_coverages) == 1
    assert len(unroll.outputs[2].proposal_coverages) == 2


def test_swapped_brother_and_wrong_parameter_generation_fail_closed() -> None:
    model = _small_model()
    swapped = _soul(model, core_id="core-b")
    with pytest.raises(ValueError, match="another core"):
        model.forward_surfaces(
            soul=swapped,
            expected_core_id="core-a",
            parameter_generation="g0",
            phase="first",
            canonical=_compiled(),
        )
    with pytest.raises(ValueError, match="parameter generation"):
        model.forward_surfaces(
            soul=swapped,
            expected_core_id="core-b",
            parameter_generation="g1",
            phase="first",
            canonical=_compiled(),
        )


def test_mechanism_curriculum_backpropagates_through_field_soul_and_typed_heads() -> None:
    model = _small_model()
    episode = build_living_reasoning_smoke_curriculum().split("train")[0]
    soul = _soul(model)
    loss, unroll, metrics = living_episode_objective(
        model,
        episode,
        soul,
        core_id="core-a",
        parameter_generation="g0",
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.page_encoder.layers[0].linear1.weight.grad is not None
    assert model.soul_projection["hot"].weight.grad is not None
    assert model.decision_head.weight.grad is not None
    assert model.decoder_output.weight.grad is not None
    assert unroll.souls[-1].generation == 3
    assert len(metrics) == 3


def test_teacher_forced_gate_uses_the_strongest_constant_category_floor() -> None:
    model = _small_model()
    episode = build_living_reasoning_smoke_curriculum().split("heldout")[0]
    result = evaluate_living_episode(
        model,
        episode,
        _soul(model),
        core_id="core-a",
        parameter_generation="g0",
    )
    counts = result["payload_teacher_forced_target_counts"]
    assert result["payload_teacher_forced_token_count"] == sum(counts)
    assert result["constant_payload_token_accuracy_floor"] == pytest.approx(
        max(counts) / sum(counts)
    )
    assert result["constant_payload_token_accuracy_floor"] > 1.0 / (
        model.eos_index + 1
    )


def test_living_reasoning_preflight_binds_all_launch_evidence(tmp_path: Path) -> None:
    model = _small_model()
    descriptor = ParameterModuleDescriptor(
        module_id="candidate-a-test",
        organ_kind=OrganKind.REASONING_CORE,
        generation_id="g0",
        architecture=model.architecture_id,
        d_model=64,
    )
    registry = ParameterRegistry()
    registry.declare_expected((descriptor,))
    registry.register(descriptor, model)
    inventory = registry.capture_inventory(exact_value_hashes=True)
    manifest = inventory.module(descriptor.module_id)
    curriculum = build_living_reasoning_smoke_curriculum()
    plan = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id=descriptor.module_id,
        base_generation_id=descriptor.generation_id,
        candidate_generation_id="g1-smoke",
        tensor_names=tuple(item.name for item in manifest.tensors if item.requires_grad),
        optimizer_name="AdamW",
        learning_rate=1e-4,
        max_steps=1,
        source_manifest_ids=(curriculum.train_manifest_id,),
        holdout_manifest_ids=(curriculum.heldout_manifest_id,),
    )
    receipt = build_living_reasoning_preflight(
        model=model,
        curriculum=curriculum,
        inventory=inventory,
        plan=plan,
        state_root=tmp_path,
        repo_root=ROOT,
    )
    assert receipt.passed
    assert {item.kind for item in receipt.evidence} == set(PreflightEvidenceKind)
    assert {
        item.name for item in receipt.contract.declared_bounds
    } == {"optimizer_steps", "reasoning_page_characters", "training_batch_size"}
    assert len(tuple((tmp_path / "training" / "reasoning" / "preflight_evidence").glob("*.json"))) == len(
        PreflightEvidenceKind
    )


def test_head_geometry_tournament_changes_only_heads_under_same_gate_surface() -> None:
    tournament = d64_head_geometry_tournament()
    assert [item.config.n_heads for item in tournament.candidates] == [1, 2, 4]
    assert {item.config.d_model for item in tournament.candidates} == {64}
    assert {item.config.n_layers for item in tournament.candidates} == {2}
    assert {item.config.ffn_dim for item in tournament.candidates} == {131_072}
    assert {item.config.page_size for item in tournament.candidates} == {32}
    metrics = {name: 0.0 for name in tournament.required_metrics}
    results = tuple(
        D64TournamentResult.from_mapping(
            tournament,
            candidate,
            metrics,
            gate_passed=False,
        )
        for candidate in tournament.candidates
    )
    assert_same_gate_surface(tournament, results)

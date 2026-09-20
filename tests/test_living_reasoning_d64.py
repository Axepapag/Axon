from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import training.living_reasoning_curriculum as living_curriculum
from runtime.field import D64FieldCompiler, LogicalRegion, SharedFieldSnapshot
from runtime.heart import EnglishProposal, ReasoningPassResult, TechnicalFinalVerdict
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
from training.complete_field_64d import CompleteField64D
from training.legacy_typed_reasoning_d64 import (
    LivingReasoningCoreConfig as LegacyTypedLivingReasoningCoreConfig,
)
from training.legacy_typed_reasoning_d64 import (
    LivingReasoningCoreD64 as LegacyTypedLivingReasoningCoreD64,
)
from training.living_reasoning_curriculum import (
    build_living_reasoning_smoke_curriculum,
    evaluate_living_episode,
    living_episode_objective,
)
from training.living_reasoning_d64 import (
    LivingReasoningCoreConfig,
    LivingReasoningCoreD64,
    candidate_a_config,
    migrate_typed_checkpoint_state_to_english_variant,
)
from training.living_reasoning_preflight import build_living_reasoning_preflight

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
            LogicalRegion.USER_INPUT: "Read beginning \u03bb, middle \U0001F9E0, and end.",
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


def test_generate_gate_bias_is_initialization_not_architecture_identity() -> None:
    default = candidate_a_config(dropout=0.0, ffn_dim=128, n_layers=1)
    fair = candidate_a_config(dropout=0.0, ffn_dim=128, n_layers=1, generate_gate_bias=0.0)
    copy_biased = candidate_a_config(
        dropout=0.0, ffn_dim=128, n_layers=1, generate_gate_bias=-1.5
    )
    assert default.architecture_id == fair.architecture_id == copy_biased.architecture_id
    assert "generate_gate_bias" not in default.to_canonical_dict()
    default_model = LivingReasoningCoreD64(default)
    fair_model = LivingReasoningCoreD64(fair)
    copy_model = LivingReasoningCoreD64(copy_biased)
    assert torch.allclose(default_model.copy_gate.bias, torch.full_like(default_model.copy_gate.bias, 1.5))
    assert torch.allclose(fair_model.copy_gate.bias, torch.zeros_like(fair_model.copy_gate.bias))
    assert torch.allclose(copy_model.copy_gate.bias, torch.full_like(copy_model.copy_gate.bias, -1.5))


def test_candidate_a_parameter_estimate_preserves_deliberate_huge_ffn() -> None:
    model = LivingReasoningCoreD64(candidate_a_config(dropout=0.0))
    report = model.architecture_report()
    assert 33_000_000 < report["parameter_count"] < 35_000_000
    assert report["parameter_bytes_fp32"] == report["parameter_count"] * 4


def test_architecture_report_declares_normal_text_termination_contract() -> None:
    config = candidate_a_config(dropout=0.0, ffn_dim=128, n_layers=1)
    report = LivingReasoningCoreD64(config).architecture_report()
    assert report["termination_contract"] == "generated-eos-independent-of-content-gate-v1"
    assert "eos_generate_head_route" not in report
    assert "termination_head_route" not in report
    assert report["architecture_id"] == config.architecture_id


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
        proposal_texts=('core-b proposes "preserve \u03bb\U0001F9E0"',),
    )

    assert first.canonical_coverage.complete
    assert first.canonical_coverage.page_count > len(tuple(LogicalRegion))
    assert refined.proposal_coverages[0].complete
    assert refined.soul_telemetry["decoded_soul_layers"] == 1.0
    assert refined.soul_telemetry["soul_contribution_l2"] > 0.0
    assert successor.layer(SoulTemperature.HOT).payload
    assert not hasattr(refined, "decision_logits")
    assert not hasattr(refined, "operation_logits")
    assert not hasattr(refined, "region_logits")
    assert not any(
        name.startswith(("decision_head.", "operation_head.", "region_head.", "start_query.", "end_query."))
        or name == "boundary_seed"
        for name in model.state_dict()
    )


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
    target = "\u03bb\U0001F9E0"
    logits, targets = model.decode_teacher(
        output.reader_state,
        target,
        head=1,
        memory=output.complete_memory,
    )
    assert logits.shape[-1] == TRANSPORT_VOCAB_SIZE + 2
    assert targets[0, :-1].tolist() == list(encode_unicode_text(target))
    assert int(targets[0, -1]) == TRANSPORT_VOCAB_SIZE + 1


def test_exact_alignment_supervises_every_multibyte_unicode_transport_cell() -> None:
    model = _small_model()
    compiled = _compiled()
    output = model.forward_surfaces(
        soul=_soul(model),
        expected_core_id="core-a",
        parameter_generation="g0",
        phase="first",
        canonical=compiled,
    )
    target = "\u03bb\U0001F9E0"
    _logits, _targets, decoder_alignment = model.decode_teacher(
        output.reader_state,
        target,
        head=1,
        memory=output.complete_memory,
        return_alignment=True,
    )
    source = "Read beginning \u03bb, middle \U0001F9E0, and end."
    segments = []
    for target_position, character in enumerate(target):
        source_position = source.index(character)
        segments.append(
            {
                "target_start": target_position,
                "target_end": target_position + 1,
                "source_region": LogicalRegion.USER_INPUT.value,
                "source_start": source_position,
                "source_end": source_position + 1,
                "text_sha256": hashlib.sha256(character.encode("utf-8")).hexdigest(),
                "authority": "exact_current_shared_field",
            }
        )
    supervision = model.alignment_supervision(
        target_text=target,
        memory=output.complete_memory,
        decoder_alignment=decoder_alignment,
        specification={
            "schema": "axon-r0-target-alignment-v1",
            "segments": segments,
            "supervise_eos_generate": True,
        },
    )

    assert supervision["copy_positions"] == len(encode_unicode_text(target)) == 6
    assert supervision["gate_supervised_positions"] == 7
    assert supervision["position_loss"].isfinite()
    assert supervision["gate_loss"].isfinite()
    summed = model.alignment_supervision(
        target_text=target,
        memory=output.complete_memory,
        decoder_alignment=decoder_alignment,
        specification={
            "schema": "axon-r0-target-alignment-v1",
            "segments": segments,
            "supervise_eos_generate": True,
        },
        position_reduction="sum",
    )
    assert torch.allclose(
        summed["position_loss"],
        supervision["position_loss"] * supervision["copy_positions"],
    )
    with pytest.raises(ValueError, match="position reduction"):
        model.alignment_supervision(
            target_text=target,
            memory=output.complete_memory,
            decoder_alignment=decoder_alignment,
            specification={
                "schema": "axon-r0-target-alignment-v1",
                "segments": segments,
                "supervise_eos_generate": True,
            },
            position_reduction="all_cells",
        )


def test_free_decoder_resumes_exact_state_across_renewable_work_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model()
    output = model.forward_surfaces(
        soul=_soul(model),
        expected_core_id="core-a",
        parameter_generation="g0",
        phase="first",
        canonical=_compiled(),
    )
    transport = list(encode_unicode_text("a"))
    categories = iter((*transport, model.eos_index))

    def deterministic_logits(
        decoder_states: torch.Tensor,
        _memory: object,
    ) -> torch.Tensor:
        logits = torch.full(
            (*decoder_states.shape[:2], model.eos_index + 1),
            -1_000.0,
            device=decoder_states.device,
        )
        logits[..., next(categories)] = 1_000.0
        return logits

    monkeypatch.setattr(model, "_decoder_logits", deterministic_logits)
    continuation = model.iter_decode_transport(output, work_units=1)

    assert next(continuation) == ("", False)
    assert next(continuation) == ("a", True)
    with pytest.raises(StopIteration):
        next(continuation)


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


def test_english_curriculum_backpropagates_through_field_soul_and_text_decoder() -> None:
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
    assert not hasattr(model, "decision_head")
    assert model.decoder_output.weight.grad is not None
    assert unroll.souls[-1].generation == 3
    assert len(metrics) == 3


def test_episode_objective_propagates_balanced_payload_eos_weight(monkeypatch) -> None:
    model = _small_model()
    episode = build_living_reasoning_smoke_curriculum().split("train")[0]
    observed: list[float] = []
    original = living_curriculum.sequence_cross_entropy

    def capture_weight(logits, targets, *, eos_weight=4.0, token_mask=None):
        observed.append(float(eos_weight))
        return original(
            logits,
            targets,
            eos_weight=eos_weight,
            token_mask=token_mask,
        )

    monkeypatch.setattr(living_curriculum, "sequence_cross_entropy", capture_weight)
    loss, _unroll, _metrics = living_episode_objective(
        model,
        episode,
        _soul(model),
        core_id="core-a",
        parameter_generation="g0",
        text_eos_weight=1.0,
    )
    assert torch.isfinite(loss)
    assert observed == [1.0, 1.0, 1.0]


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
    counts = result["text_teacher_forced_target_counts"]
    assert result["text_teacher_forced_token_count"] == sum(counts)
    assert result["constant_text_token_accuracy_floor"] == pytest.approx(max(counts) / sum(counts))
    assert result["constant_text_token_accuracy_floor"] > 1.0 / (model.eos_index + 1)
    histogram = result["constant_text_target_histogram"]
    assert result["constant_text_exact_floor"] == pytest.approx(
        max(histogram.values()) / result["supervised_phase_count"]
    )


def test_heldout_refinement_reads_the_production_first_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mastery evaluator must not substitute authored gold for a live pass."""

    model = _small_model()
    episode = build_living_reasoning_smoke_curriculum().split("heldout")[0]
    requests = []

    def scripted_emit(self: LivingReasoningCoreD64, request):
        requests.append(request)
        forward = self.forward_request(request)
        transition = self.exhale_transition(
            before=request.soul,
            exhaled_state=forward.exhaled_state,
            tick_uid=request.image.identity.tick_uid,
            request_id=request.request_id,
            phase=request.phase,
        )
        common = {
            "base_field_id": request.image.identity.base_field_id,
            "base_tick_id": request.image.identity.base_tick_id,
            "author_core_id": request.descriptor.core_id,
            "rail_d_model": request.descriptor.d_model,
        }
        if request.phase == "first":
            output = EnglishProposal(pass_id="first", text="actual first", **common)
        elif request.phase == "refined":
            output = EnglishProposal(pass_id="refined", text="actual refined", **common)
        else:
            output = TechnicalFinalVerdict(text="#responseDraft# done", **common)
        return ReasoningPassResult(output=output, soul_transition=transition)

    monkeypatch.setattr(LivingReasoningCoreD64, "emit", scripted_emit)
    result = evaluate_living_episode(
        model,
        episode,
        _soul(model),
        core_id="core-a",
        parameter_generation="g0",
    )

    assert [request.phase for request in requests] == ["first", "refined", "consolidated"]
    refined_request = requests[1]
    assert refined_request.soul.generation == 1
    assert len(refined_request.proposal_rails) == 1
    rendered_first = refined_request.proposal_rails[0].text
    assert rendered_first.startswith("FIRST PROPOSALS\n\n[core-a]\nactual first")
    assert episode.first_workspace_text not in rendered_first
    assert result["refinement_context"] == "production_first_workspace"
    assert result["authored_workspace_text_used"] is False
    assert result["phase_diagnostics"][1]["proposal_context"] == [rendered_first]


def test_failed_first_is_visible_to_refinement_and_does_not_advance_soul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model()
    episode = build_living_reasoning_smoke_curriculum().split("heldout")[0]
    requests = []

    def scripted_emit(self: LivingReasoningCoreD64, request):
        requests.append(request)
        if request.phase == "first":
            raise RuntimeError("planned first-pass failure")
        forward = self.forward_request(request)
        transition = self.exhale_transition(
            before=request.soul,
            exhaled_state=forward.exhaled_state,
            tick_uid=request.image.identity.tick_uid,
            request_id=request.request_id,
            phase=request.phase,
        )
        common = {
            "base_field_id": request.image.identity.base_field_id,
            "base_tick_id": request.image.identity.base_tick_id,
            "author_core_id": request.descriptor.core_id,
            "rail_d_model": request.descriptor.d_model,
        }
        if request.phase == "refined":
            output = EnglishProposal(pass_id="refined", text="recovery proposal", **common)
        else:
            output = TechnicalFinalVerdict(text="#responseDraft# done", **common)
        return ReasoningPassResult(output=output, soul_transition=transition)

    monkeypatch.setattr(LivingReasoningCoreD64, "emit", scripted_emit)
    result = evaluate_living_episode(
        model,
        episode,
        _soul(model),
        core_id="core-a",
        parameter_generation="g0",
    )

    refined_request = requests[1]
    assert refined_request.phase == "refined"
    assert refined_request.soul.generation == 0
    assert "FIRST PROPOSALS" in refined_request.proposal_rails[0].text
    assert "FAILED" in refined_request.proposal_rails[0].text
    first_row, refined_row = result["phase_diagnostics"][:2]
    assert first_row["participant_state"] == "failed"
    assert first_row["after_soul_generation"] == 0
    assert refined_row["before_soul_generation"] == 0


def test_the_transcript_sink_cap_is_display_only() -> None:
    """Capping the sink removes display rows and nothing else.

    The smoke script's QA panel was unreadable because the sink stopped after the
    third DELTA payload phase of each episode, so every sample came from the same
    few short cases.  Lifting the cap must be instrumentation only: the measured
    metrics have to be byte-identical whether the sink records none, some, or all
    of an episode's payload phases.
    """

    model = _small_model()
    episode = build_living_reasoning_smoke_curriculum().split("heldout")[0]
    text_phases = sum(1 for target in episode.targets if target.supervision_weight > 0)
    assert text_phases > 1, "the fixture must have more than one supervised text phase"

    def measure(cap: int | None) -> tuple[dict, list[dict]]:
        sink: list[dict] = []
        result = evaluate_living_episode(
            model,
            episode,
            _soul(model),
            core_id="core-a",
            parameter_generation="g0",
            transcript_sink=sink,
            transcript_sink_cap=cap,
        )
        return result, sink

    baseline, capped_default = measure(3)
    unlimited, everything = measure(None)
    nothing_metric, nothing = measure(0)

    assert unlimited == baseline == nothing_metric
    assert len(nothing) == 0
    assert len(capped_default) == min(3, text_phases)
    assert len(everything) == text_phases
    assert [row["expected_text"] for row in everything] == [
        target.text for target in episode.targets if target.supervision_weight > 0
    ]


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
    assert {item.name for item in receipt.contract.declared_bounds} == {
        "optimizer_steps",
        "reasoning_page_characters",
        "training_batch_size",
    }
    assert len(tuple((tmp_path / "training" / "reasoning" / "preflight_evidence").glob("*.json"))) == len(
        PreflightEvidenceKind
    )


class _StubMemory:
    """Minimal AddressableMemory stand-in for route-selection unit tests."""

    def __init__(self, slots: int, d_model: int, token_id: int = 0) -> None:
        self.states = torch.zeros(1, slots, d_model)
        self.char_indices = torch.full((1, slots), token_id, dtype=torch.long)
        self.receipts = ()
        self.segments = ()

    def receipt(self, index: int):
        from types import SimpleNamespace

        return SimpleNamespace(
            address=SimpleNamespace(transport_token_id=int(self.char_indices[0, index]))
        )


def test_active_core_mechanically_severs_legacy_typed_anatomy() -> None:
    assert LivingReasoningCoreD64.__bases__ == (CompleteField64D,)
    fields = LivingReasoningCoreConfig.__dataclass_fields__
    for retired in (
        "receipt_continuation",
        "eos_generate_head_route",
        "termination_head_route",
    ):
        assert retired not in fields
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            LivingReasoningCoreConfig(**{retired: True})

    assert LivingReasoningCoreConfig().architecture_id == "living-d64-english-33e7107432217c64a6b0f5fe"
    intended = LivingReasoningCoreConfig(
        n_heads=1,
        n_layers=2,
        ffn_dim=16384,
        state_tokens=4,
        page_size=32,
        generate_gate_bias=0.0,
    )
    assert intended.architecture_id == "living-d64-english-023f4b5e7d43d59968c9b133"


def test_generated_eos_is_independent_of_content_copy_gate() -> None:
    model = LivingReasoningCoreD64(
        LivingReasoningCoreConfig(
            n_heads=1,
            n_layers=1,
            ffn_dim=128,
            state_tokens=2,
            page_size=2,
            dropout=0.0,
        )
    )
    memory = _StubMemory(slots=4, d_model=model.living_config.d_model, token_id=7)
    decoder_state = torch.zeros(1, 1, model.living_config.d_model)
    with torch.no_grad():
        model.copy_gate.weight.zero_()
        model.copy_gate.bias.fill_(-8.0)
        model.decoder_output.weight.zero_()
        model.decoder_output.bias.zero_()
        model.decoder_output.bias[model.eos_index] = 12.0

    logits, alignment = model._decoder_logits(decoder_state, memory, return_alignment=True)
    probabilities = logits.exp()
    generated = torch.softmax(alignment["generated_logits"], dim=-1)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1, 1), atol=1e-6)
    assert torch.allclose(
        probabilities[..., model.eos_index], generated[..., model.eos_index], atol=1e-6
    )
    assert int(logits[0, 0].argmax().item()) == model.eos_index


def test_typed_checkpoint_migrates_by_exact_subset_without_obsolete_heads() -> None:
    torch.manual_seed(991)
    legacy_config = LegacyTypedLivingReasoningCoreConfig(
        n_heads=1,
        n_layers=1,
        ffn_dim=128,
        state_tokens=2,
        page_size=2,
        dropout=0.0,
    )
    source = LegacyTypedLivingReasoningCoreD64(legacy_config)
    target = LivingReasoningCoreD64(
        LivingReasoningCoreConfig(
            n_heads=1,
            n_layers=1,
            ffn_dim=128,
            state_tokens=2,
            page_size=2,
            dropout=0.0,
        )
    )
    receipt = migrate_typed_checkpoint_state_to_english_variant(
        source.state_dict(),
        target,
        source_architecture_id=source.architecture_id,
        source_parameter_generation="step-720-typed",
        target_parameter_generation="english-donor-g1",
    )

    target_state = target.state_dict()
    source_state = source.state_dict()
    assert target.architecture_id != source.architecture_id
    assert receipt.target_architecture_id == target.architecture_id
    assert receipt.source_architecture_id == source.architecture_id
    assert receipt.retired_tensors
    assert any(name.startswith("decision_head.") for name in receipt.retired_tensors)
    assert any(name.startswith("operation_head.") for name in receipt.retired_tensors)
    assert any(name.startswith("region_head.") for name in receipt.retired_tensors)
    assert all(name in source_state for name in target_state)
    assert all(torch.equal(target_state[name], source_state[name]) for name in target_state)
    assert not any(
        name.startswith(("decision_head.", "operation_head.", "region_head.", "start_query.", "end_query."))
        or name == "boundary_seed"
        for name in target_state
    )


def test_living_reasoning_gate_rejects_nonfinite_metrics_without_hash_failure() -> None:
    report = {
        "text_exact_rate": float("nan"),
        "text_teacher_forced_content_accuracy": 1.0,
        "text_teacher_forced_eos_accuracy": 1.0,
        "complete_field_coverage_rate": 1.0,
        "final_verdict_valid_rate": 1.0,
        "constant_text_exact_floor": 0.25,
    }
    decision = living_curriculum.decide_living_reasoning_mastery(report)
    assert decision["passed"] is False
    assert decision["failures"]
    assert decision["gate_id"]

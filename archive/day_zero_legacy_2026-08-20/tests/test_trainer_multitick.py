from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from cores.core import AxonCore, CoreConfig
from runtime.field import (
    PROPOSAL_END,
    PROPOSAL_START,
    LogicalRegion,
    SharedFieldSnapshot,
    SlotKind,
    compile_field_view,
    proposal_payload_capacity,
    proposal_tail_offset,
)
from training.build_multitick_curriculum import (
    FAMILY_SCRATCH,
    SEGMENT_COUNT,
    SourceRecord,
    build_episode,
    canonical_json_bytes,
    sha256_text,
    sha256_bytes,
)
from training.trainer_multitick import (
    MultiTickTrainer,
    MultiTickTrainingContractError,
    PROPOSAL_WIDTH,
    evaluate_free_running_episode,
    reconstruct_transition_metrics,
    require_exact_v4_launch_gate,
    transition_length_bin,
    validate_exact_v4_episode,
)


def _tiny_core() -> AxonCore:
    torch.manual_seed(1234)
    return AxonCore(
        CoreConfig(
            d_model=16,
            n_heads=1,
            n_layers=1,
            ffn_dim=32,
            dropout=0.0,
            soul_rows=0,
            soul_hot_rows=0,
            soul_mode="act_reflect_v2",
            char_slot_mode=True,
            char_slot_max_slots=384,
            char_n_regions=3,
        )
    )


def _episode() -> dict:
    row = {
        "family": FAMILY_SCRATCH,
        "lineage_id": "synthetic-scratch-response",
        "conversation_history": "Earlier evidence",
        "user_input": "Plan and answer",
        "initial_draft": "rough",
        "structured_knowledge": "A visible fact",
        "scratch_plan": "p" * 130,
        "response_after": "r" * 201,
        "evidence_refs": ["fixture:evidence:1"],
    }
    record = SourceRecord(
        row=row,
        pointer="jsonl:fixture:1",
        source_path="synthetic.jsonl",
        source_sha256="A" * 64,
        row_sha256=sha256_bytes(canonical_json_bytes(row)),
        adapter="test",
    )
    episode, reason = build_episode(
        record,
        unsupported_policy="reject",
        include_diary=False,
    )
    assert reason is None
    assert episode is not None
    assert [tick["target_region"] for tick in episode["ticks"]] == [
        "scratch",
        "scratch",
        "scratch",
        "scratch",
        "response_draft",
        "response_draft",
        "response_draft",
        "response_draft",
    ]
    return episode


def _single_transition_episode(target_length: int) -> dict:
    row = {
        "family": FAMILY_SCRATCH,
        "lineage_id": f"single-transition-{target_length}",
        "initial_draft": "prior",
        "targets": [
            {
                "region": "response_draft",
                "phase": "respond",
                "text": "x" * target_length,
            }
        ],
    }
    record = SourceRecord(
        row=row,
        pointer=f"jsonl:single-transition:{target_length}",
        source_path="synthetic.jsonl",
        source_sha256="B" * 64,
        row_sha256=sha256_bytes(canonical_json_bytes(row)),
        adapter="test",
    )
    episode, reason = build_episode(
        record,
        unsupported_policy="reject",
        include_diary=False,
    )
    assert reason is None
    assert episode is not None
    return episode


def _full_width_paging_episode() -> dict:
    row = {
        "family": FAMILY_SCRATCH,
        "lineage_id": "full-width-paging",
        "conversation_history": "history " * 60,
        "user_input": "question " * 45,
        "initial_draft": "d" * 256,
        "structured_knowledge": "knowledge " * 55,
        "situation_awareness": "situation " * 50,
        "tool_results": "tool " * 70,
        "advisor_input": "advisor " * 55,
        "task_state": "task " * 70,
        "diary": "diary " * 60,
        "scratch_plan": "s" * 256,
        "response_after": "r" * 256,
        "evidence_refs": ["fixture:evidence:full-width"],
    }
    record = SourceRecord(
        row=row,
        pointer="jsonl:full-width-paging:1",
        source_path="synthetic.jsonl",
        source_sha256="C" * 64,
        row_sha256=sha256_bytes(canonical_json_bytes(row)),
        adapter="test",
    )
    episode, reason = build_episode(
        record,
        unsupported_policy="reject",
        include_diary=True,
    )
    assert reason is None
    assert episode is not None
    return episode


def test_real_core_deep_supervision_commits_scratch_and_passes_masks() -> None:
    episode = _episode()
    core = _tiny_core()
    trainer = MultiTickTrainer(core)
    seen: list[dict[str, torch.Tensor]] = []
    original_forward = core.forward_charslot

    def captured_forward(**kwargs):
        seen.append(
            {
                "roles": kwargs["region_ids"].detach().clone(),
                "mask": kwargs["mask"].detach().clone(),
                "field16": kwargs["field16"].detach().clone(),
            }
        )
        return original_forward(**kwargs)

    core.forward_charslot = captured_forward  # type: ignore[method-assign]
    result = trainer.forward_episode(episode)

    assert len(result.tick_loss_tensors) == 2 * SEGMENT_COUNT
    assert len(result.ticks) == 2 * SEGMENT_COUNT
    assert torch.isfinite(result.total_loss).item()
    assert torch.allclose(
        result.total_loss,
        torch.stack(result.tick_loss_tensors).mean(),
    )
    assert all(tick.proposal_slots == PROPOSAL_WIDTH for tick in result.ticks)
    assert all(np.isfinite(tick.loss) for tick in result.ticks)
    assert all(0.0 <= tick.padded_accuracy <= 1.0 for tick in result.ticks)
    assert all(len(tick.input_view_hash) == 64 for tick in result.ticks)
    assert len(result.transitions) == 2
    assert [transition.target_length for transition in result.transitions] == [
        130,
        201,
    ]
    assert [transition.target_length_bin for transition in result.transitions] == [
        "129-192",
        "193-256",
    ]
    assert all(
        transition.evaluation_mode == "teacher_forced"
        for transition in result.transitions
    )
    assert dict(result.transition_bin_gates) == {
        "129-192": result.transitions[0].reconstruction_gate_passed,
        "193-256": result.transitions[1].reconstruction_gate_passed,
    }
    assert result.transition_reconstruction_gate_passed == all(
        transition.reconstruction_gate_passed
        for transition in result.transitions
    )
    empty_target_ticks = [
        tick for tick in result.ticks if tick.target_text == ""
    ]
    assert empty_target_ticks
    assert any(tick.predicted_text != "" for tick in empty_target_ticks)
    for tick in empty_target_ticks:
        assert tick.target_character_accuracy == tick.padded_accuracy
        assert tick.target_character_accuracy < 1.0

    expected_scratch = episode["ticks"][3]["complete_proposed_region_text"]
    response_tick_snapshot = result.ticks[4].input_snapshot
    assert response_tick_snapshot.region("scratch").text == expected_scratch
    assert response_tick_snapshot.field_id == result.ticks[3].output_snapshot_id
    assert result.ticks[0].target_region is LogicalRegion.SCRATCH
    assert result.ticks[3].target_region is LogicalRegion.SCRATCH
    assert result.ticks[4].target_region is LogicalRegion.RESPONSE_DRAFT
    assert result.ticks[7].target_region is LogicalRegion.RESPONSE_DRAFT

    assert len(seen) == 2 * SEGMENT_COUNT
    for tick, call in zip(result.ticks, seen):
        assert call["roles"].shape == (1, 384)
        assert set(call["roles"].unique().tolist()) == {0, 1, 2}
        assert call["mask"].shape == (1, 384)
        assert call["mask"].dtype == torch.bool
        assert call["mask"][:, 320:].all().item()
        assert call["field16"].shape == (1, 384, 16)
        prior = tick.input_snapshot.region(tick.target_region).text
        expected_view = compile_field_view(
            tick.input_snapshot,
            proposal_region=tick.target_region,
            proposal_offset=proposal_tail_offset(
                tick.target_region.value,
                prior,
            ),
        )
        assert tick.input_view_hash == expected_view.view_hash
    # All eight structural ticks keep the learned physical proposal role at 2.
    assert all((call["roles"][:, 320:] == 2).all().item() for call in seen)


def test_full_width_writable_regions_fail_readback_launch_gate_truthfully() -> None:
    episode = _full_width_paging_episode()
    validated = validate_exact_v4_episode(episode)
    contract = episode["read_coverage_contract"]
    ticks = episode["ticks"]

    assert validated.launch_gate_passed is False
    assert contract["whole_field_read_coverage_proven"] is False
    assert (
        contract["writable_region_post_commit_readback_guaranteed"]
        is False
    )
    assert contract["launch_gate_passed"] is False
    assert contract["transaction_target_coverage"] == {
        "transition_count": 2,
        "target_characters": 512,
        "supervised_segment_characters": 512,
        "each_target_character_supervised_exactly_once": True,
        "segment_count_per_transition": 4,
        "segment_width": 64,
    }
    assert "".join(
        tick["segment_target_text"] for tick in ticks[:4]
    ) == "s" * 256
    assert "".join(
        tick["segment_target_text"] for tick in ticks[4:]
    ) == "r" * 256
    assert [tick["read_page"]["page_index"] for tick in ticks] == list(
        range(8)
    )
    assert (
        ticks[4]["read_page"]["cursor_before"]
        == ticks[3]["read_page"]["cursor_after"]
    )

    # Scratch becomes read-only context only after the proposal switches to
    # response.  Four fair context pages expose some, but not all, of its 256
    # characters.  Response target characters have no post-commit read tick.
    scratch_read = sum(
        tick["read_page"]["coverage"]["region_character_counts"].get(
            "scratch",
            0,
        )
        for tick in ticks[4:]
    )
    assert 0 < scratch_read < 256
    assert ticks[-1]["target_region"] == "response_draft"
    assert all(
        tick["read_page"]["coverage"]["duplicate_character_count"] == 0
        for tick in ticks
    )
    for tick in ticks:
        projection = tick["active_view"]["proposal_projection"]
        base = tick["field_before"][tick["target_region"]]["text"]
        assert projection["proposal_view_offset"] == proposal_tail_offset(
            tick["target_region"],
            base,
        )
        assert projection["shows_current_tail"] is True

    with pytest.raises(
        MultiTickTrainingContractError,
        match="post-commit readback",
    ):
        require_exact_v4_launch_gate(episode)

    core = _tiny_core()
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.zero_()
    teacher = MultiTickTrainer(core).forward_episode(episode)
    free = evaluate_free_running_episode(
        core,
        episode,
        max_output_chars=8,
    )
    assert [tick.read_page_index for tick in teacher.ticks] == list(range(8))
    assert [page.cursor.page_index for page in free.read_pages] == list(range(8))
    assert [page.view.proposal_region for page in free.read_pages] == (
        [LogicalRegion.SCRATCH] * 4
        + [LogicalRegion.RESPONSE_DRAFT] * 4
    )
    assert all(
        right.cursor == left.next_cursor
        for left, right in zip(free.read_pages, free.read_pages[1:])
    )
    # Teacher and free-running share the structural cursor algorithm, but the
    # latter advances only over its own generated field after the switch.
    assert [tick.target_region for tick in teacher.ticks] == [
        page.view.proposal_region for page in free.read_pages
    ]


@pytest.mark.parametrize(
    ("target_length", "expected_bin"),
    [
        (65, "065-128"),
        (128, "065-128"),
        (192, "129-192"),
        (256, "193-256"),
    ],
)
def test_full_transition_reconstruction_bins_and_exact_gate(
    target_length: int,
    expected_bin: str,
) -> None:
    validated = validate_exact_v4_episode(
        _single_transition_episode(target_length)
    )
    transitions = reconstruct_transition_metrics(
        validated.ticks,
        tuple(tick.target_text for tick in validated.ticks),
        evaluation_mode="teacher_forced",
    )

    assert transition_length_bin(target_length) == expected_bin
    assert len(transitions) == 1
    transition = transitions[0]
    assert transition.target_length_bin == expected_bin
    assert transition.predicted_text == "x" * target_length
    assert transition.target_text == "x" * target_length
    assert transition.predicted_length == target_length
    assert transition.length_delta == 0
    assert transition.length_match is True
    assert transition.predicted_sha256 == sha256_text("x" * target_length)
    assert transition.hash_match is True
    assert transition.character_accuracy == 1.0
    assert transition.exact_text_match is True
    assert transition.reconstruction_gate_passed is True


def test_full_transition_gate_rejects_correct_prefix_with_empty_suffix() -> None:
    validated = validate_exact_v4_episode(_single_transition_episode(65))
    predicted_segments = (
        validated.ticks[0].target_text,
        "",
        "",
        "",
    )
    transition = reconstruct_transition_metrics(
        validated.ticks,
        predicted_segments,
        evaluation_mode="teacher_forced",
    )[0]

    assert transition.segment_predictions[0] == "x" * 64
    assert transition.predicted_text == "x" * 64
    assert transition.target_text == "x" * 65
    assert transition.predicted_length == 64
    assert transition.target_length == 65
    assert transition.length_delta == -1
    assert transition.length_match is False
    assert transition.hash_match is False
    assert transition.character_matches == 64
    assert transition.comparison_characters == 65
    assert transition.character_accuracy == pytest.approx(64 / 65)
    assert transition.exact_text_match is False
    assert transition.reconstruction_gate_passed is False


@pytest.mark.parametrize(
    "region",
    [LogicalRegion.SCRATCH, LogicalRegion.RESPONSE_DRAFT],
)
@pytest.mark.parametrize("length", [64, 65, 128, 192, 256])
def test_proposal_tail_offset_exposes_newest_region_character(
    region: LogicalRegion,
    length: int,
) -> None:
    text = "x" * length
    snapshot = SharedFieldSnapshot.from_texts(
        {region: text},
        source="tail-test",
        provenance="tail-test",
    )
    view = compile_field_view(
        snapshot,
        proposal_region=region,
        proposal_offset=proposal_tail_offset(region.value, text),
    )
    proposal_span_refs = [
        ref
        for ref in view.slot_refs[PROPOSAL_START:PROPOSAL_END]
        if ref.kind is SlotKind.SPAN
    ]
    assert proposal_span_refs
    assert proposal_span_refs[-1].region_char_index == length - 1
    assert proposal_payload_capacity(region) == (
        54 if region is LogicalRegion.SCRATCH else 47
    )


def test_train_step_updates_declared_params_and_excludes_soul_writers() -> None:
    core = _tiny_core()
    trainer = MultiTickTrainer(core)
    named = dict(core.named_parameters())
    writer_before = {
        name: parameter.detach().clone()
        for name, parameter in named.items()
        if name in trainer.frozen_writer_names
    }
    assert writer_before
    assert all(
        not named[name].requires_grad for name in trainer.frozen_writer_names
    )
    assert not (
        set(trainer.frozen_writer_names)
        & set(trainer.declared_trainable_names)
    )

    optimizer = trainer.build_optimizer(
        lr=2e-3,
        optimizer_cls=torch.optim.SGD,
    )
    step = trainer.train_episode(_episode(), optimizer)

    assert step.gradient_coverage.valid
    assert set(step.gradient_coverage.covered_names) == set(
        trainer.declared_trainable_names
    )
    assert set(step.optimizer_names) == set(trainer.declared_trainable_names)
    assert step.updated_names
    assert "char_slot_head.3.weight" in step.updated_names
    assert all(
        torch.equal(writer_before[name], named[name].detach())
        for name in writer_before
    )
    assert all(named[name].grad is None for name in writer_before)


def test_supplied_read_bearing_soul_is_immutable_and_not_written() -> None:
    core = _tiny_core()
    trainer = MultiTickTrainer(core, soul_read_bearing=True)
    soul = torch.randn(2, core.cfg.d_model)
    soul_mask = torch.tensor([True, True])
    soul_before = soul.clone()
    mask_before = soul_mask.clone()
    prior_readonly = core.soul_readonly

    result = trainer.forward_episode(
        _episode(),
        soul=soul,
        soul_mask=soul_mask,
    )

    assert result.soul_read_bearing is True
    assert torch.equal(soul, soul_before)
    assert torch.equal(soul_mask, mask_before)
    assert core.soul_readonly is prior_readonly
    assert all(
        name not in result.declared_trainable_names
        for name in trainer.frozen_writer_names
    )


def test_malformed_sealed_hash_and_diff_contracts_fail_closed() -> None:
    episode = _episode()

    legacy = copy.deepcopy(episode)
    legacy["builder_version"] = "2"
    with pytest.raises(MultiTickTrainingContractError, match="builder_version"):
        validate_exact_v4_episode(legacy)

    legacy_teacher = copy.deepcopy(episode)
    legacy_teacher["ticks"][0]["teacher_forced"][
        "commit_complete_proposed_region"
    ] = True
    with pytest.raises(MultiTickTrainingContractError, match="legacy fields"):
        validate_exact_v4_episode(legacy_teacher)

    missing_region = copy.deepcopy(episode)
    del missing_region["initial_field"]["diary"]
    with pytest.raises(
        MultiTickTrainingContractError,
        match="ten canonical regions",
    ):
        validate_exact_v4_episode(missing_region)

    sealed = copy.deepcopy(episode)
    sealed["ticks"][0]["target_region"] = "task_state"
    with pytest.raises(MultiTickTrainingContractError, match="sealed"):
        validate_exact_v4_episode(sealed)

    bad_hash = copy.deepcopy(episode)
    bad_hash["initial_field_sha256"] = "0" * 64
    with pytest.raises(MultiTickTrainingContractError, match="hash mismatch"):
        validate_exact_v4_episode(bad_hash)

    bad_diff = copy.deepcopy(episode)
    bad_diff["ticks"][0]["typed_delta"]["replacement"] = "tampered"
    with pytest.raises(
        MultiTickTrainingContractError,
        match="exact structural segment delta",
    ):
        validate_exact_v4_episode(bad_diff)

    bad_page = copy.deepcopy(episode)
    bad_page["ticks"][1]["read_page"]["cursor_before"]["user_offset"] += 1
    with pytest.raises(MultiTickTrainingContractError, match="read_page"):
        validate_exact_v4_episode(bad_page)

    false_promotion = copy.deepcopy(episode)
    false_promotion["read_coverage_contract"]["launch_gate_passed"] = True
    with pytest.raises(
        MultiTickTrainingContractError,
        match="fail closed",
    ):
        validate_exact_v4_episode(false_promotion)

    too_long = copy.deepcopy(episode)
    too_long["ticks"][0]["segment_target_text"] = "x" * 65
    with pytest.raises(MultiTickTrainingContractError, match="exceeds 64"):
        validate_exact_v4_episode(too_long)


def test_optimizer_with_writer_parameters_is_rejected_as_dishonest() -> None:
    core = _tiny_core()
    optimizer = torch.optim.SGD(core.parameters(), lr=1e-3)
    trainer = MultiTickTrainer(core)
    with pytest.raises(
        MultiTickTrainingContractError,
        match="optimizer coverage",
    ):
        trainer.train_episode(_episode(), optimizer)


def test_free_running_rollout_never_commits_or_reads_gold_targets() -> None:
    episode = _episode()
    gold_scratch = episode["ticks"][3]["complete_proposed_region_text"]
    gold_response = episode["ticks"][7]["complete_proposed_region_text"]
    core = _tiny_core()
    prior_readonly = core.soul_readonly
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.zero_()

    result = evaluate_free_running_episode(
        core,
        episode,
        max_output_chars=8,
        author_core_id="tiny-free-run",
    )

    assert result.free_running is True
    assert len(result.records) == 2 * SEGMENT_COUNT
    assert len(result.transitions) == 2
    assert all(
        transition.evaluation_mode == "free_running"
        for transition in result.transitions
    )
    assert result.transitions[0].target_text == gold_scratch
    assert result.transitions[1].target_text == gold_response
    assert result.transitions[0].predicted_text == "".join(
        record.proposed_text for record in result.records[:SEGMENT_COUNT]
    )
    assert result.transitions[1].predicted_text == "".join(
        record.proposed_text for record in result.records[SEGMENT_COUNT:]
    )
    assert result.transition_reconstruction_gate_passed is False
    assert all(not record.teacher_forced for record in result.records)
    assert result.records[0].proposed_text != gold_scratch
    assert result.records[0].committed_text == result.records[0].proposed_text
    assert result.records[1].prior_text == result.records[0].committed_text
    assert result.records[1].committed_text == (
        result.records[1].prior_text + result.records[1].proposed_text
    )
    assert result.records[4].committed_text == result.records[4].proposed_text
    assert result.records[4].prior_text != gold_response
    assert (
        result.records[1].input_field_id
        == result.records[0].output_field_id
    )
    assert result.final_snapshot.region("scratch").text != gold_scratch
    assert result.final_snapshot.region("response_draft").text != gold_response
    assert core.soul_readonly is prior_readonly

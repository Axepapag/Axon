from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from runtime.field import (
    AttendedInterval,
    D64FieldCompiler,
    FieldSpan,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
)
from runtime.soul import SoulSnapshot, empty_soul_layers
from substrate import TRANSPORT_VOCAB_SIZE, encode_unicode_text
from training.complete_field_64d import (
    AddressableMemory,
    CompleteField64D,
    MemoryCellReceipt,
    sequence_cross_entropy,
)
from training.living_reasoning_d64 import (
    D64_DECODER_EXECUTION_STATE_SCHEMA,
    DecoderEmissionRoute,
    DecoderExecutionState,
    LivingReasoningCoreConfig,
    LivingReasoningCoreD64,
    candidate_a_config,
    migrate_legacy_weights_to_receipt_variant,
)


def _model(*, receipt_continuation: bool = True) -> LivingReasoningCoreD64:
    torch.manual_seed(808)
    model = LivingReasoningCoreD64(
        LivingReasoningCoreConfig(
            n_heads=1,
            n_layers=1,
            ffn_dim=128,
            state_tokens=2,
            page_size=1,
            dropout=0.0,
            receipt_continuation=receipt_continuation,
        )
    )
    model.eval()
    return model


def _output(
    model: LivingReasoningCoreD64,
    text: str,
    *,
    region: LogicalRegion = LogicalRegion.USER_INPUT,
    proposals: tuple[str, ...] = (),
):
    snapshot = SharedFieldSnapshot.from_texts({region: text}, tick_id=19)
    return _output_from_snapshot(model, snapshot, proposals=proposals)


def _output_from_snapshot(
    model: LivingReasoningCoreD64,
    snapshot: SharedFieldSnapshot,
    *,
    proposals: tuple[str, ...] = (),
):
    compiled = D64FieldCompiler().compile(snapshot)
    soul = SoulSnapshot(
        core_id="pointer-test-core",
        architecture_id=model.architecture_id,
        parameter_generation="pointer-generation-v1",
        generation=0,
        parent_soul_id=None,
        layers=empty_soul_layers(),
    )
    return model.forward_surfaces(
        soul=soul,
        expected_core_id=soul.core_id,
        parameter_generation=soul.parameter_generation,
        phase="first",
        canonical=compiled,
        proposal_texts=proposals,
        view_id="pointer-test-view",
    )


def _receipt_index(output, character: str, unit_index: int = 0, occurrence: int = 0) -> int:
    matches = [
        index
        for index, receipt in enumerate(output.complete_memory.receipts)
        if receipt is not None
        and receipt.address.character == character
        and receipt.address.transport_unit_index == unit_index
    ]
    return matches[occurrence]


def _scripted_selector(model: LivingReasoningCoreD64, anchor_indices: tuple[int, ...]):
    actions = iter((*anchor_indices, "eos"))

    def select(step, memory):
        action = next(actions)
        if action == "eos":
            return DecoderEmissionRoute.LEARNED_GENERATE, model.eos_index, None, None
        receipt = memory.receipt(action)
        return (
            DecoderEmissionRoute.LEARNED_COPY_ANCHOR,
            receipt.address.transport_token_id,
            action,
            receipt,
        )

    return select


@pytest.mark.parametrize("character", ("A", "λ", "終", "🧠"))
def test_receipt_conduit_copies_native_and_two_three_four_cell_scalars(
    monkeypatch: pytest.MonkeyPatch,
    character: str,
) -> None:
    model = _model()
    output = _output(model, f"aaa{character}z")
    anchor_index = _receipt_index(output, character)
    anchor = output.complete_memory.receipt(anchor_index)
    unit_count = len(encode_unicode_text(character))
    assert anchor.address.transport_unit_count == unit_count
    if unit_count > 1:
        assert anchor.address.lane_index == 3
        final_index = _receipt_index(output, character, unit_count - 1)
        final = output.complete_memory.receipt(final_index)
        assert final.address.row_index > anchor.address.row_index

    monkeypatch.setattr(model, "_select_learned_emission", _scripted_selector(model, (anchor_index,)))
    initial = model.initial_decoder_execution_state(output)
    first = model.advance_decoder_execution(output, initial, work_units=1)
    assert not first.complete
    assert first.text == ""
    assert first.state.trace[0].route is DecoderEmissionRoute.LEARNED_COPY_ANCHOR
    assert first.state.pending_next_unit_index == (1 if unit_count > 1 else None)

    restored = DecoderExecutionState.from_json_bytes(first.state.to_json_bytes())
    assert restored.state_id == first.state.state_id
    finished = model.advance_decoder_execution(output, restored, work_units=unit_count)
    assert finished.complete
    assert finished.text == character
    assert finished.malformed_reason is None
    assert [item.route for item in finished.state.trace].count(
        DecoderEmissionRoute.DETERMINISTIC_RECEIPT_CONTINUATION
    ) == unit_count - 1

    monkeypatch.setattr(model, "_select_learned_emission", _scripted_selector(model, (anchor_index,)))
    uninterrupted = model.advance_decoder_execution(
        output,
        model.initial_decoder_execution_state(output),
        work_units=unit_count + 1,
    )
    assert uninterrupted.state.to_json_bytes() == finished.state.to_json_bytes()


def test_repeated_scalar_uses_exact_selected_receipt_and_never_crosses_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    output = _output(model, "λxλ")
    second_anchor = _receipt_index(output, "λ", occurrence=1)
    monkeypatch.setattr(
        model,
        "_select_learned_emission",
        _scripted_selector(model, (second_anchor,)),
    )
    result = model.advance_decoder_execution(
        output,
        model.initial_decoder_execution_state(output),
        work_units=3,
    )
    assert result.complete and result.text == "λ"
    copied = result.state.trace[:-1]
    assert {item.anchor_receipt_id for item in copied} == {
        output.complete_memory.receipt(second_anchor).receipt_id
    }
    assert all(
        output.complete_memory.receipt(item.memory_index).address.region_position == 2
        for item in copied
        if item.memory_index is not None
    )


def test_changed_source_minimal_pair_changes_anchor_identity_but_not_exact_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    first = _output(model, "xλ")
    second = _output(model, "xxλ")
    first_anchor = _receipt_index(first, "λ")
    second_anchor = _receipt_index(second, "λ")
    assert first.complete_memory.receipt(first_anchor).receipt_id != (
        second.complete_memory.receipt(second_anchor).receipt_id
    )
    assert first.complete_memory.receipt(first_anchor).address.region_position == 1
    assert second.complete_memory.receipt(second_anchor).address.region_position == 2

    for output, anchor in ((first, first_anchor), (second, second_anchor)):
        monkeypatch.setattr(model, "_select_learned_emission", _scripted_selector(model, (anchor,)))
        result = model.advance_decoder_execution(
            output,
            model.initial_decoder_execution_state(output),
            work_units=3,
        )
        assert result.complete and result.text == "λ"


def test_masked_cells_padding_invalid_categories_and_stale_bindings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    span = FieldSpan(
        span_id="mask-proof",
        text="A🧠B",
        source="user-turn-19",
        provenance="conversation:user",
    )
    snapshot = SharedFieldSnapshot(
        tick_id=19,
        regions=(
            RegionState(
                name=LogicalRegion.USER_INPUT,
                spans=(span,),
                attended_intervals=(AttendedInterval(0, 1), AttendedInterval(2, 3)),
            ),
        ),
    )
    output = _output_from_snapshot(model, snapshot)
    valid_receipts = tuple(item for item in output.complete_memory.receipts if item is not None)
    assert {item.address.character for item in valid_receipts} == {"A", "B"}
    assert all(item.address.provenance == "conversation:user" for item in valid_receipts)
    empty_index = next(
        index for index, item in enumerate(output.complete_memory.receipts) if item is None
    )

    def padding_copy(_step, _memory):
        return DecoderEmissionRoute.LEARNED_COPY_ANCHOR, 0, empty_index, None

    monkeypatch.setattr(model, "_select_learned_emission", padding_copy)
    rejected = model.advance_decoder_execution(
        output,
        model.initial_decoder_execution_state(output),
        work_units=1,
    )
    assert rejected.malformed_reason == "learned copy lacks its exact compiler receipt"
    assert rejected.text == ""

    for invalid_category in (model.empty_index, model.bos_index):
        monkeypatch.setattr(
            model,
            "_select_learned_emission",
            lambda _step, _memory, category=invalid_category: (
                DecoderEmissionRoute.LEARNED_GENERATE,
                category,
                None,
                None,
            ),
        )
        rejected = model.advance_decoder_execution(
            output,
            model.initial_decoder_execution_state(output),
            work_units=1,
        )
        assert "EMPTY or an invalid" in rejected.malformed_reason
        assert rejected.text == ""

    monkeypatch.setattr(
        model,
        "_select_learned_emission",
        lambda _step, _memory: (
            DecoderEmissionRoute.LEARNED_GENERATE,
            model.eos_index,
            None,
            None,
        ),
    )
    empty_success = model.advance_decoder_execution(
        output,
        model.initial_decoder_execution_state(output),
        work_units=1,
    )
    assert empty_success.complete and empty_success.text == ""

    stale = replace(model.initial_decoder_execution_state(output), view_id="substituted-view")
    with pytest.raises(ValueError, match=r"stale|another surface"):
        model.advance_decoder_execution(output, stale, work_units=1)


def test_mid_scalar_anchor_and_substituted_surface_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    output = _output(model, "🧠")
    mid_index = _receipt_index(output, "🧠", unit_index=1)

    def mid_scalar(_step, memory):
        receipt = memory.receipt(mid_index)
        return (
            DecoderEmissionRoute.LEARNED_COPY_ANCHOR,
            receipt.address.transport_token_id,
            mid_index,
            receipt,
        )

    monkeypatch.setattr(model, "_select_learned_emission", mid_scalar)
    initial = model.initial_decoder_execution_state(output)
    rejected = model.advance_decoder_execution(output, initial, work_units=1)
    assert not rejected.complete
    assert "mid-scalar" in rejected.malformed_reason
    assert rejected.text == ""

    changed = _output(model, "🧩")
    with pytest.raises(ValueError, match=r"stale|another surface"):
        model.advance_decoder_execution(changed, initial, work_units=1)


def test_receipts_reject_category_corruption_and_keep_joined_segments_distinct() -> None:
    model = _model()
    output = _output(
        model,
        "λ",
        region=LogicalRegion.ADVISOR_INPUT,
        proposals=("λ",),
    )
    anchors = [
        receipt
        for receipt in output.complete_memory.receipts
        if receipt is not None
        and receipt.address.character == "λ"
        and receipt.address.transport_unit_index == 0
    ]
    assert len(anchors) == 2
    assert anchors[0].address.region == anchors[1].address.region
    assert anchors[0].address.region_position == anchors[1].address.region_position == 0
    assert anchors[0].segment_id != anchors[1].segment_id

    index = output.complete_memory.receipts.index(anchors[0])
    corrupted_address = replace(
        anchors[0].address,
        transport_token_id=(anchors[0].address.transport_token_id + 1) % TRANSPORT_VOCAB_SIZE,
    )
    corrupted_receipt = MemoryCellReceipt(
        segment_id=anchors[0].segment_id,
        segment_kind=anchors[0].segment_kind,
        source_field_id=anchors[0].source_field_id,
        source_tick_id=anchors[0].source_tick_id,
        rail_id=anchors[0].rail_id,
        address=corrupted_address,
    )
    receipts = list(output.complete_memory.receipts)
    receipts[index] = corrupted_receipt
    with pytest.raises(ValueError, match="category disagrees"):
        AddressableMemory(
            states=output.complete_memory.states,
            char_indices=output.complete_memory.char_indices,
            region_ids=output.complete_memory.region_ids,
            region_positions=output.complete_memory.region_positions,
            receipts=tuple(receipts),
            segments=output.complete_memory.segments,
        )

    wrong_address = replace(anchors[0].address, row_index=anchors[0].address.row_index + 1)
    wrong_receipt = MemoryCellReceipt(
        segment_id=anchors[0].segment_id,
        segment_kind=anchors[0].segment_kind,
        source_field_id=anchors[0].source_field_id,
        source_tick_id=anchors[0].source_tick_id,
        rail_id=anchors[0].rail_id,
        address=wrong_address,
    )
    receipts[index] = wrong_receipt
    with pytest.raises(ValueError, match="segment address proof"):
        AddressableMemory(
            states=output.complete_memory.states,
            char_indices=output.complete_memory.char_indices,
            region_ids=output.complete_memory.region_ids,
            region_positions=output.complete_memory.region_positions,
            receipts=tuple(receipts),
            segments=output.complete_memory.segments,
        )


def test_right_categories_under_wrong_receipts_and_provenance_fail_closed() -> None:
    model = _model()
    output = _output(model, "λλ")
    anchors = [
        index
        for index, receipt in enumerate(output.complete_memory.receipts)
        if receipt is not None
        and receipt.address.character == "λ"
        and receipt.address.transport_unit_index == 0
    ]
    assert len(anchors) == 2

    wrong_receipts = list(output.complete_memory.receipts)
    wrong_receipts[anchors[0]], wrong_receipts[anchors[1]] = (
        wrong_receipts[anchors[1]],
        wrong_receipts[anchors[0]],
    )
    with pytest.raises(ValueError, match="position disagrees"):
        AddressableMemory(
            states=output.complete_memory.states,
            char_indices=output.complete_memory.char_indices,
            region_ids=output.complete_memory.region_ids,
            region_positions=output.complete_memory.region_positions,
            receipts=tuple(wrong_receipts),
            segments=output.complete_memory.segments,
        )

    source = output.complete_memory.receipt(anchors[0])
    changed_address = replace(source.address, provenance="substituted-provenance")
    wrong_receipts = list(output.complete_memory.receipts)
    wrong_receipts[anchors[0]] = MemoryCellReceipt(
        segment_id=source.segment_id,
        segment_kind=source.segment_kind,
        source_field_id=source.source_field_id,
        source_tick_id=source.source_tick_id,
        rail_id=source.rail_id,
        address=changed_address,
    )
    with pytest.raises(ValueError, match="segment address proof"):
        AddressableMemory(
            states=output.complete_memory.states,
            char_indices=output.complete_memory.char_indices,
            region_ids=output.complete_memory.region_ids,
            region_positions=output.complete_memory.region_positions,
            receipts=tuple(wrong_receipts),
            segments=output.complete_memory.segments,
        )

    corrupted_categories = output.complete_memory.char_indices.clone()
    corrupted_categories[0, anchors[0]] = (
        int(corrupted_categories[0, anchors[0]].item()) + 1
    ) % TRANSPORT_VOCAB_SIZE
    with pytest.raises(ValueError, match="category disagrees"):
        AddressableMemory(
            states=output.complete_memory.states,
            char_indices=corrupted_categories,
            region_ids=output.complete_memory.region_ids,
            region_positions=output.complete_memory.region_positions,
            receipts=output.complete_memory.receipts,
            segments=output.complete_memory.segments,
        )


def test_false_physical_adjacency_and_joined_memory_seams_never_continue() -> None:
    model = _model()
    output = _output(model, "🧠", proposals=("λ",))
    canonical_anchor = next(
        index
        for index, receipt in enumerate(output.complete_memory.receipts)
        if receipt is not None
        and receipt.segment_kind == "canonical"
        and receipt.address.character == "🧠"
        and receipt.address.transport_unit_index == 0
    )
    proposal_anchor = next(
        index
        for index, receipt in enumerate(output.complete_memory.receipts)
        if receipt is not None
        and receipt.segment_kind == "proposal"
        and receipt.address.character == "λ"
        and receipt.address.transport_unit_index == 0
    )
    anchor = output.complete_memory.receipt(canonical_anchor)
    continuation_indices = tuple(
        output.complete_memory.continuation_index(anchor, unit)
        for unit in range(1, anchor.address.transport_unit_count)
    )
    assert all(
        output.complete_memory.receipt(index).segment_id == anchor.segment_id
        for index in continuation_indices
    )
    assert proposal_anchor not in continuation_indices

    incomplete = list(output.complete_memory.receipts)
    incomplete[continuation_indices[0]] = None
    with pytest.raises(ValueError, match="missing its compiler receipt"):
        AddressableMemory(
            states=output.complete_memory.states,
            char_indices=output.complete_memory.char_indices,
            region_ids=output.complete_memory.region_ids,
            region_positions=output.complete_memory.region_positions,
            receipts=tuple(incomplete),
            segments=output.complete_memory.segments,
        )


def test_pending_continuation_rejects_padding_bounds_and_eos_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    output = _output(model, "🧠")
    anchor_index = _receipt_index(output, "🧠")
    monkeypatch.setattr(model, "_select_learned_emission", _scripted_selector(model, (anchor_index,)))
    paused = model.advance_decoder_execution(
        output,
        model.initial_decoder_execution_state(output),
        work_units=1,
    )
    padding_index = next(
        index for index, receipt in enumerate(output.complete_memory.receipts) if receipt is None
    )
    monkeypatch.setattr(
        AddressableMemory,
        "continuation_index",
        lambda _memory, _anchor, _unit: padding_index,
    )
    padding_failure = model.advance_decoder_execution(output, paused.state, work_units=1)
    assert padding_failure.malformed_reason == "addressable memory slot has no compiler receipt"
    assert padding_failure.text == ""

    out_of_bounds = replace(
        paused.state,
        pending_anchor_memory_index=len(output.complete_memory.receipts) + 1,
        pending_anchor_receipt_id="out-of-bounds-receipt",
    )
    with pytest.raises(IndexError, match="out of range"):
        model.advance_decoder_execution(output, out_of_bounds, work_units=1)

    receipt = output.complete_memory.receipt(anchor_index)
    eos_address = replace(receipt.address, transport_token_id=model.eos_index)
    eos_receipt = MemoryCellReceipt(
        segment_id=receipt.segment_id,
        segment_kind=receipt.segment_kind,
        source_field_id=receipt.source_field_id,
        source_tick_id=receipt.source_tick_id,
        rail_id=receipt.rail_id,
        address=eos_address,
    )
    eos_receipts = list(output.complete_memory.receipts)
    eos_receipts[anchor_index] = eos_receipt
    eos_categories = output.complete_memory.char_indices.clone()
    eos_categories[0, anchor_index] = model.eos_index
    with pytest.raises(ValueError, match=r"segment address proof|Unicode receipt proof"):
        AddressableMemory(
            states=output.complete_memory.states,
            char_indices=eos_categories,
            region_ids=output.complete_memory.region_ids,
            region_positions=output.complete_memory.region_positions,
            receipts=tuple(eos_receipts),
            segments=output.complete_memory.segments,
        )


def test_continuation_losses_are_masked_while_anchor_and_eos_remain_learned() -> None:
    model = _model()
    output = _output(model, "🧠")
    target = "🧠"
    target_tokens = tuple(encode_unicode_text(target))
    length = len(target_tokens) + 1
    memory_size = output.complete_memory.states.shape[1]
    position_logits = torch.randn(1, length, memory_size, requires_grad=True)
    gate_logits = torch.randn(1, length, requires_grad=True)
    payload_logits = torch.randn(1, length, model.eos_index + 1, requires_grad=True)
    targets = torch.tensor(
        [[*target_tokens, model.eos_index]],
        dtype=torch.long,
    )
    supervision = model.alignment_supervision(
        target_text=target,
        memory=output.complete_memory,
        decoder_alignment={
            "position_logits": position_logits,
            "generate_gate_logits": gate_logits,
        },
        specification={
            "schema": "axon-r0-target-alignment-v1",
            "segments": [
                {
                    "target_start": 0,
                    "target_end": 1,
                    "source_region": LogicalRegion.USER_INPUT.value,
                    "source_start": 0,
                    "source_end": 1,
                    "text_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                    "authority": "exact_current_shared_field",
                }
            ],
            "supervise_eos_generate": True,
        },
    )
    assert supervision["copy_positions"] == 1
    assert supervision["deterministic_continuation_positions"] == 3
    assert supervision["learned_decision_mask"].tolist() == [[True, False, False, False, True]]
    loss = (
        sequence_cross_entropy(
            payload_logits,
            targets,
            token_mask=supervision["learned_decision_mask"],
        )
        + supervision["position_loss"]
        + supervision["copy_gate_loss"]
        + supervision["eos_gate_loss"]
    )
    loss.backward()
    assert payload_logits.grad[:, 1:4].abs().sum().item() == 0.0
    assert position_logits.grad[:, 1:4].abs().sum().item() == 0.0
    assert gate_logits.grad[:, 1:4].abs().sum().item() == 0.0
    assert gate_logits.grad[:, 0].abs().sum().item() > 0.0
    assert gate_logits.grad[:, -1].abs().sum().item() > 0.0


def test_receipt_architecture_is_new_identity_with_legacy_tensor_compatibility() -> None:
    legacy_config = candidate_a_config(n_layers=1, ffn_dim=128, dropout=0.0)
    receipt_config = candidate_a_config(
        n_layers=1,
        ffn_dim=128,
        dropout=0.0,
        receipt_continuation=True,
    )
    assert legacy_config.to_canonical_dict()["schema"] == "axon-living-reasoning-architecture-v1"
    assert "receipt_continuation" not in legacy_config.to_canonical_dict()
    assert receipt_config.to_canonical_dict()["schema"] == "axon-living-reasoning-architecture-v2"
    assert receipt_config.to_canonical_dict()["decoder_execution_state_schema"] == (
        D64_DECODER_EXECUTION_STATE_SCHEMA
    )
    assert legacy_config.architecture_id != receipt_config.architecture_id

    torch.manual_seed(99)
    legacy = LivingReasoningCoreD64(legacy_config)
    receipt = LivingReasoningCoreD64(receipt_config)
    migration = migrate_legacy_weights_to_receipt_variant(
        legacy,
        receipt,
        source_parameter_generation="legacy-g0",
        target_parameter_generation="receipt-g0",
    )
    assert migration.source_architecture_id == legacy.architecture_id
    assert migration.target_architecture_id == receipt.architecture_id
    assert {item.name for item in migration.copied_tensors} == set(legacy.state_dict())
    assert all(item.source_sha256 == item.target_sha256 for item in migration.copied_tensors)
    assert "pending_anchor_receipt_id" in migration.new_state_fields
    assert set(legacy.state_dict()) == set(receipt.state_dict())
    assert all(
        torch.equal(value, receipt.state_dict()[name])
        for name, value in legacy.state_dict().items()
    )


def test_receipt_teacher_path_uses_the_same_causal_recurrent_step() -> None:
    model = _model()
    output = _output(model, "abcλ")
    target = "λ"
    teacher_logits, targets = model.decode_teacher(
        output.reader_state,
        target,
        head=1,
        memory=output.complete_memory,
    )
    hidden = model._initial_decoder_hidden(output.reader_state, 1)
    previous = model.bos_index
    manual = []
    for position in range(targets.shape[1]):
        step = model.causal_decoder_step(
            previous_category=previous,
            hidden=hidden,
            memory=output.complete_memory,
        )
        manual.append(step.mixed_logits)
        hidden = step.hidden
        previous = int(targets[0, position].item())
    assert torch.equal(teacher_logits, torch.cat(manual, dim=1))

    scheduled_logits, scheduled_targets = model.decode_scheduled(
        output.reader_state,
        target,
        head=1,
        teacher_forcing_ratio=1.0,
        memory=output.complete_memory,
    )
    assert torch.equal(scheduled_targets, targets)
    assert torch.equal(scheduled_logits, teacher_logits)


def test_teacher_scheduled_and_runtime_have_identical_post_scalar_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    output = _output(model, "x🧠y")
    target = "🧠"
    anchor_index = _receipt_index(output, target)
    teacher_logits, targets = model.decode_teacher(
        output.reader_state,
        target,
        head=1,
        memory=output.complete_memory,
    )
    scheduled_logits, scheduled_targets = model.decode_scheduled(
        output.reader_state,
        target,
        head=1,
        teacher_forcing_ratio=1.0,
        memory=output.complete_memory,
    )
    assert torch.equal(scheduled_targets, targets)
    assert torch.equal(scheduled_logits, teacher_logits)

    monkeypatch.setattr(model, "_select_learned_emission", _scripted_selector(model, (anchor_index,)))
    unit_count = len(encode_unicode_text(target))
    runtime = model.advance_decoder_execution(
        output,
        model.initial_decoder_execution_state(output),
        work_units=unit_count,
    )
    assert runtime.text == ""
    assert runtime.state.pending_anchor_memory_index is None
    assert tuple(event.route for event in runtime.state.trace) == (
        DecoderEmissionRoute.LEARNED_COPY_ANCHOR,
        *(
            DecoderEmissionRoute.DETERMINISTIC_RECEIPT_CONTINUATION
            for _ in range(unit_count - 1)
        ),
    )

    manual_hidden = model._initial_decoder_hidden(output.reader_state, 1)
    previous = model.bos_index
    for category in targets[0, :unit_count].tolist():
        step = model.causal_decoder_step(
            previous_category=previous,
            hidden=manual_hidden,
            memory=output.complete_memory,
        )
        manual_hidden = step.hidden
        previous = int(category)
    runtime_hidden = runtime.state.hidden_tensor(
        device=model.device,
        dtype=model.initial_state.dtype,
    )
    assert torch.equal(runtime_hidden, manual_hidden)
    next_runtime = model.causal_decoder_step(
        previous_category=runtime.state.previous_category,
        hidden=runtime_hidden,
        memory=output.complete_memory,
    )
    assert torch.equal(next_runtime.mixed_logits, teacher_logits[:, unit_count : unit_count + 1])
    assert torch.equal(next_runtime.mixed_logits, scheduled_logits[:, unit_count : unit_count + 1])


def test_state_identity_and_every_runtime_binding_detect_substitution() -> None:
    model = _model()
    output = _output(model, "same visible text")
    state = model.initial_decoder_execution_state(output)

    encoded = json.loads(state.to_json_bytes())
    encoded["state_id"] = "0" * 64
    with pytest.raises(ValueError, match="state identity mismatch"):
        DecoderExecutionState.from_json_bytes(
            json.dumps(encoded, sort_keys=True).encode("utf-8")
        )

    substitutions = {
        "architecture_id": "different-architecture",
        "parameter_generation": "different-generation",
        "core_id": "different-core",
        "field_id": "different-field",
        "tick_id": state.tick_id + 1,
        "view_id": "different-view",
        "rail_id": "different-rail",
        "surface_id": "different-surface",
        "memory_id": "different-memory",
        "memory_segment_ids": ("different-segment",),
        "pass_id": "different-pass",
    }
    for field_name, value in substitutions.items():
        with pytest.raises(ValueError, match=r"stale|another surface"):
            model.advance_decoder_execution(
                output,
                replace(state, **{field_name: value}),
                work_units=1,
            )
    with pytest.raises(ValueError, match="head is invalid"):
        model.advance_decoder_execution(output, replace(state, head=2), work_units=1)

    same_text_new_tick = SharedFieldSnapshot.from_texts(
        {LogicalRegion.USER_INPUT: "same visible text"},
        tick_id=20,
    )
    substituted = _output_from_snapshot(model, same_text_new_tick)
    with pytest.raises(ValueError, match=r"stale|another surface"):
        model.advance_decoder_execution(substituted, state, work_units=1)


def test_disabled_receipt_feature_preserves_legacy_teacher_decoder_path() -> None:
    model = _model(receipt_continuation=False)
    output = _output(model, "abc")
    inherited = CompleteField64D.decode_teacher(
        model,
        output.reader_state,
        "cab",
        head=1,
        memory=output.complete_memory,
    )
    delegated = model.decode_teacher(
        output.reader_state,
        "cab",
        head=1,
        memory=output.complete_memory,
    )
    assert torch.equal(delegated[0], inherited[0])
    assert torch.equal(delegated[1], inherited[1])

    clone = _model(receipt_continuation=False)
    loaded = clone.load_state_dict(model.state_dict(), strict=True)
    assert loaded.missing_keys == [] and loaded.unexpected_keys == []
    clone_output = _output(clone, "abc")
    restored = clone.decode_teacher(
        clone_output.reader_state,
        "cab",
        head=1,
        memory=clone_output.complete_memory,
    )
    assert torch.equal(restored[0], delegated[0])
    assert torch.equal(restored[1], delegated[1])

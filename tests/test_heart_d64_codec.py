from __future__ import annotations

import pytest
import torch

from runtime.field import LogicalRegion, RegionState, RegionVisibility, SharedFieldSnapshot
from runtime.heart import D64HeartCodec, D64HeartCodecError, D64HeartFrame
from runtime.heart.translation_core import HeartTranslationCore, HeartTranslationCoreConfig


def _model() -> HeartTranslationCore:
    return HeartTranslationCore(
        HeartTranslationCoreConfig(
            d_model=64,
            n_heads=4,
            n_layers=1,
            ffn_dim=128,
            dropout=0.0,
            source_page_chars=8,
        )
    )


def test_real_d64_codec_consumes_literal_cells_and_preserves_masked_global_positions() -> None:
    snapshot = SharedFieldSnapshot(
        tick_id=7,
        regions=(
            RegionState.from_text(
                LogicalRegion.CONVERSATION_HISTORY,
                "dormant older turn",
                visibility=RegionVisibility.MASKED,
            ),
            RegionState.from_text(LogicalRegion.USER_INPUT, "Attend me exactly!"),
        ),
    )
    codec = D64HeartCodec()
    frame = codec.compile(snapshot)
    assert frame.exact_text == "Attend me exactly!"
    assert frame.canonical_positions[0] == len("dormant older turn")
    assert len(frame.canonical_positions) == len(frame.exact_text)

    model = _model()
    batch = codec.batch_for_model(model, (frame,))
    assert torch.equal(
        batch.source_positions[0, : len(frame.characters)],
        torch.tensor(frame.canonical_positions),
    )
    output = model(
        batch.source_indices,
        batch.source_mask,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([[model.bos_index]]),
        source_cells16=batch.source_cells16,
        source_positions=batch.source_positions,
    )
    assert output.source_coverage.complete
    assert output.source_coverage.source_characters == (len(frame.exact_text),)


def test_real_d64_codec_rejects_substituted_lane_cells() -> None:
    snapshot = SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: "real rail"})
    codec = D64HeartCodec()
    frame = codec.compile(snapshot)
    corrupted = frame.cells16.copy()
    corrupted[0, 0] += 1.0
    substituted = D64HeartFrame(
        source_field_id=frame.source_field_id,
        source_tick_id=frame.source_tick_id,
        rail_id=frame.rail_id,
        semantic_surface_id=frame.semantic_surface_id,
        characters=frame.characters,
        transport_token_ids=frame.transport_token_ids,
        canonical_positions=frame.canonical_positions,
        cells16=corrupted,
        address_sha256=frame.address_sha256,
        coverage_sha256=frame.coverage_sha256,
    )
    with pytest.raises(D64HeartCodecError, match="lane cell"):
        codec.batch_for_model(_model(), (substituted,))


def test_real_d64_codec_binds_translation_proposal_to_frozen_frame() -> None:
    snapshot = SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: "hello"}, tick_id=4)
    codec = D64HeartCodec()
    dual = codec.compiler.compile_dual(snapshot, codec.exact_compiler.compile(snapshot))
    frame = codec.compile(snapshot, dual)
    delta = codec.proposed_replacement(
        snapshot,
        dual,
        region=LogicalRegion.RESPONSE_DRAFT,
        text="response",
        author_core_id="heart-translator-test",
        pass_id="cardiac-pass",
        frame=frame,
    )
    assert delta.base_field_id == snapshot.field_id
    assert frame.frame_id in delta.evidence
    assert delta.operations[0].text == "response"

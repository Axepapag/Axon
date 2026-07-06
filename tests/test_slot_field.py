"""Tests for slot field construction, masking, and train-as-you-live invariants."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from adapters.slot_adapter import get_adapter, get_char_prototype_table
from slots.slot_field_contract import FieldLayout, SlotField, default_layout
from slots.slot_spec import KIND_RESPONSE_DRAFT, KIND_TEXT, pack_slot, pack_text_chain, unpack_region
from training.trainer_slot import SlotFieldBuilder, render_field_snapshot


def test_default_layout_has_all_regions() -> None:
    layout = default_layout()
    from slots.slot_field_contract import REGION_NAMES
    for name in REGION_NAMES:
        assert name in layout.region_slots


def test_field_masking() -> None:
    field = SlotField.empty(default_layout())
    slots = [pack_slot(text="hello axon", kind=KIND_TEXT)]
    field.set_region("conversation_history", slots)
    mask = field.mask({"conversation_history", "response_draft"})
    conv_slice = field.layout.region_slice("conversation_history")
    draft_slice = field.layout.region_slice("response_draft")
    scratch_slice = field.layout.region_slice("scratch")
    assert mask[conv_slice].all()
    assert mask[draft_slice].all()
    assert not mask[scratch_slice].any()


def test_field_builder_shapes() -> None:
    layout = default_layout()
    adapter = get_adapter(64)
    table = get_char_prototype_table(64)
    builder = SlotFieldBuilder(layout, adapter, table, torch.device("cpu"), torch.float32, max_response_chars=32)
    built = builder.build(
        {"conversation_history": "hello axon"},
        answer_text="hi there",
        active_regions={"conversation_history", "response_draft"},
    )
    assert built["field_d"].shape == (1, layout.total_slots, 64)
    assert built["mask"].shape == (1, layout.total_slots)
    assert built["targets"].shape == (1, 32)
    assert built["field_delta_targets_d"].shape == (1, layout.total_slots, 64)
    assert built["field_delta_weights"].shape == (1, layout.total_slots)
    assert built["target_len"] == len("hi there")


def test_field_builder_does_not_leak_target_into_response_draft() -> None:
    layout = default_layout()
    adapter = get_adapter(64)
    table = get_char_prototype_table(64)
    builder = SlotFieldBuilder(layout, adapter, table, torch.device("cpu"), torch.float32, max_response_chars=32)
    built = builder.build(
        {"conversation_history": "question cue"},
        answer_text="secret target",
        active_regions={"conversation_history", "response_draft"},
    )
    draft_slots = unpack_region(built["field_np"][layout.region_slice("response_draft")])
    assert all("secret target" not in slot.text for slot in draft_slots)
    rendered = render_field_snapshot(
        built["field_np"],
        layout,
        {"conversation_history", "response_draft"},
        max_chars=64,
    )
    assert any("question cue" in line for line in rendered)
    assert not any("secret target" in line for line in rendered)

    target_slots = unpack_region(built["target_field_np"][layout.region_slice("response_draft")])
    assert any("secret target" in slot.text for slot in target_slots)


def test_field_builder_packs_user_input_region() -> None:
    layout = default_layout()
    adapter = get_adapter(64)
    table = get_char_prototype_table(64)
    builder = SlotFieldBuilder(layout, adapter, table, torch.device("cpu"), torch.float32, max_response_chars=32)
    built = builder.build(
        {"conversation_history": "prior turn", "user_input": "current request"},
        answer_text="answer",
        active_regions={"conversation_history", "user_input", "response_draft"},
    )
    user_slots = unpack_region(built["field_np"][layout.region_slice("user_input")])
    assert any("current request" in slot.text for slot in user_slots)


def test_field_builder_masks_inactive_regions() -> None:
    layout = default_layout()
    adapter = get_adapter(64)
    table = get_char_prototype_table(64)
    builder = SlotFieldBuilder(layout, adapter, table, torch.device("cpu"), torch.float32, max_response_chars=32)
    built = builder.build(
        {"conversation_history": "hello axon"},
        answer_text="hi",
        active_regions={"conversation_history", "response_draft"},
    )
    scratch_slice = layout.region_slice("scratch")
    assert not built["mask"][0, scratch_slice].any()


def test_no_input_into_soul() -> None:
    """Input is placed in the field; the soul starts empty (no curriculum preload)."""
    layout = default_layout()
    adapter = get_adapter(64)
    table = get_char_prototype_table(64)
    builder = SlotFieldBuilder(layout, adapter, table, torch.device("cpu"), torch.float32, max_response_chars=32)
    built = builder.build(
        {"conversation_history": "secret cue"},
        answer_text="answer",
        active_regions={"conversation_history", "response_draft"},
    )
    # The field contains the input
    field_np = built["field_np"]
    slots = unpack_region(field_np[layout.region_slice("conversation_history")])
    assert any("secret cue" in s.text for s in slots)
    # The soul is not part of the builder output; a fresh soul manager has zero active rows
    from cores.soul_v2 import SoulManagerV2, SoulV2Config, TierSpec
    soul_cfg = SoulV2Config(d_model=64, tiers=[TierSpec("hot", 16)])
    soul_mgr = SoulManagerV2(soul_cfg, torch.device("cpu"), torch.float32)
    assert soul_mgr.state.n_active() == 0

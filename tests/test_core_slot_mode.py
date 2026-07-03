"""Tests for cores/core.py slot-era mode and soul breathing."""
from __future__ import annotations

import pytest
import torch

from adapters.slot_adapter import get_adapter, get_char_prototype_table
from cores.core import AxonCore, CoreConfig
from cores.soul_v2 import SoulManagerV2, SoulV2Config, TierSpec
from slots.slot_field_contract import SlotField, default_layout
from slots.slot_spec import KIND_RESPONSE_DRAFT, KIND_TEXT, pack_text_chain


def _make_core(d_model: int = 64, slot_mode: bool = True, max_response_chars: int = 32) -> AxonCore:
    cfg = CoreConfig(
        d_model=d_model,
        n_layers=1,
        n_heads=1,
        ffn_dim=256,
        soul_mode="act_reflect_v2",
        soul_rows=16,
        soul_hot_rows=16,
        slot_mode=slot_mode,
        max_response_chars=max_response_chars,
    )
    return AxonCore(cfg)


def test_core_slot_mode_forward_shapes() -> None:
    core = _make_core()
    adapter = get_adapter(64)
    layout = default_layout()
    field = SlotField.empty(layout)
    field.set_region("conversation_history", pack_text_chain("hello", kind=KIND_TEXT))
    field.set_region("response_draft", pack_text_chain("world", kind=KIND_RESPONSE_DRAFT))
    mask = torch.from_numpy(field.mask({"conversation_history", "response_draft"})).unsqueeze(0)
    field_d = torch.from_numpy(adapter.project_down_region(field.data)).float()

    soul_cfg = SoulV2Config(d_model=64, tiers=[TierSpec("hot", 16)])
    soul_mgr = SoulManagerV2(soul_cfg, torch.device("cpu"), torch.float32)
    soul, soul_mask = soul_mgr.inhale()

    out = core.forward_slot(
        field_d.unsqueeze(0),
        soul,
        mask,
        soul_mask=soul_mask,
        response_draft_slice=layout.region_slice("response_draft"),
    )
    assert out["field"].shape == (1, layout.total_slots, 64)
    assert out["soul"].shape == soul.shape
    assert out["draft_chars"].shape == (1, 32, 64)


def test_core_char_logits_shape() -> None:
    core = _make_core(max_response_chars=16)
    draft_chars = torch.randn(1, 16, 64)
    logits = core.char_logits(draft_chars)
    assert logits.shape == (1, 16, len(get_char_prototype_table(64).chars))


def test_core_backward_compatibility_non_slot_mode() -> None:
    """Existing forward_with_soul interface is unchanged when slot_mode=False."""
    cfg = CoreConfig(d_model=32, n_layers=1, n_heads=1, ffn_dim=64, soul_mode="act_reflect_v2")
    core = AxonCore(cfg)
    field = torch.randn(1, 10, 32)
    soul = torch.randn(1, 4, 32)
    out = core.forward_with_soul(field, soul)
    assert out["field"].shape == (1, 10, 32)
    assert out["soul"].shape == (1, 4, 32)
    assert "draft_chars" not in out


def test_core_slot_mode_requires_act_reflect_v2() -> None:
    cfg = CoreConfig(d_model=32, slot_mode=True, soul_mode="concat")
    with pytest.raises(AssertionError):
        AxonCore(cfg)


def test_inhale_exhale_no_nan() -> None:
    """A full inhale -> attend -> exhale cycle with an empty soul must not NaN."""
    core = _make_core()
    adapter = get_adapter(64)
    layout = default_layout()
    field = SlotField.empty(layout)
    field.set_region("conversation_history", pack_text_chain("hello", kind=KIND_TEXT))
    mask = torch.from_numpy(field.mask({"conversation_history"})).unsqueeze(0)
    field_d = torch.from_numpy(adapter.project_down_region(field.data)).float().unsqueeze(0)

    soul_cfg = SoulV2Config(d_model=64, tiers=[TierSpec("hot", 16)])
    soul_mgr = SoulManagerV2(soul_cfg, torch.device("cpu"), torch.float32)
    soul, soul_mask = soul_mgr.inhale()
    assert soul_mgr.state.n_active() == 0

    out = core.forward_slot(
        field_d,
        soul,
        mask,
        soul_mask=soul_mask,
        response_draft_slice=layout.region_slice("response_draft"),
    )
    assert not out["field"].isnan().any()
    assert not out["soul"].isnan().any()

    soul_mgr.exhale_after_answer(out["field"].detach())
    assert soul_mgr.state.n_active() > 0

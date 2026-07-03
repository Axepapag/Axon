"""Tests for the counterfactual soul-conditioning probe harness."""
from __future__ import annotations

import torch

from adapters.slot_adapter import get_adapter, get_char_prototype_table
from cores.core import AxonCore, CoreConfig
from cores.soul_v2 import SoulManagerV2, SoulV2Config, TierSpec
from slots.slot_field_contract import default_layout
from training.trainer_slot import RecallCurriculum, SlotFieldBuilder, run_cf_probe


def _tiny_core_and_builder(max_response_chars: int = 32):
    cfg = CoreConfig(
        d_model=64,
        n_layers=1,
        n_heads=1,
        ffn_dim=256,
        soul_mode="act_reflect_v2",
        soul_rows=16,
        soul_hot_rows=16,
        slot_mode=True,
        max_response_chars=max_response_chars,
    )
    core = AxonCore(cfg)
    adapter = get_adapter(64)
    table = get_char_prototype_table(64)
    builder = SlotFieldBuilder(default_layout(), adapter, table, torch.device("cpu"), torch.float32, max_response_chars=max_response_chars)
    soul_cfg = SoulV2Config(d_model=64, tiers=[TierSpec("hot", 16)])
    soul_mgr = SoulManagerV2(soul_cfg, torch.device("cpu"), torch.float32)
    return core, builder, soul_mgr


def test_cf_probe_runs_and_counts() -> None:
    """The harness runs on recall lessons and produces well-formed counts."""
    core, builder, soul_mgr = _tiny_core_and_builder()
    recall = RecallCurriculum(n_synthetic=200)
    r = run_cf_probe(
        core,
        soul_mgr,
        builder,
        recall.eval_lessons,
        torch.device("cpu"),
        torch.float32,
        n=8,
    )
    assert r["tot"] == 8
    assert 0 <= r["orig_ok"] <= r["tot"]
    assert 0 <= r["swap_flip"] <= r["tot"]
    assert 0 <= r["zero_fail"] <= r["tot"]
    assert 0 <= r["irrelevant_diff"] <= r["tot"]
    assert 0 <= r["field_leak"] <= r["tot"]
    assert r["verdict"] in ("SOUL_IS_READ", "SOUL_NOT_READ")


def test_cf_probe_zero_soul_breaks_something() -> None:
    """With an untrained core, zero soul should usually change the answer,
    so zero_fail should be > 0 for a non-trivial set of lessons."""
    core, builder, soul_mgr = _tiny_core_and_builder()
    recall = RecallCurriculum(n_synthetic=200)
    r = run_cf_probe(
        core,
        soul_mgr,
        builder,
        recall.eval_lessons,
        torch.device("cpu"),
        torch.float32,
        n=8,
    )
    # We do not require this for every random seed, but on the fixed seed it
    # should hold because zeroing the soul removes the recalled information.
    assert r["zero_fail"] > 0


def test_recall_curriculum_orient() -> None:
    recall = RecallCurriculum(n_synthetic=10)
    lesson = recall.eval_lessons[0]
    bp, answers = RecallCurriculum.orient(lesson)
    assert isinstance(bp, list)
    assert isinstance(answers, list)
    assert len(bp) == len(answers)

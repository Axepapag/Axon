from __future__ import annotations

import torch

from runtime.soul import SoulSnapshot, empty_soul_layers
from training.living_reasoning_curriculum import evaluate_living_episode
from training.living_reasoning_d64 import LivingReasoningCoreConfig, LivingReasoningCoreD64
from training.pointer_bootstrap_curriculum import (
    POINTER_BOOTSTRAP_SCHEMA,
    build_pointer_bootstrap_curriculum,
    decide_pointer_bootstrap_mastery,
)


def _region_text(episode, name: str) -> str:
    for region in episode.snapshot.regions:
        if region.name.value == name:
            return "".join(span.text for span in region.spans)
    raise KeyError(name)


def _small_model() -> LivingReasoningCoreD64:
    torch.manual_seed(23)
    model = LivingReasoningCoreD64(
        LivingReasoningCoreConfig(n_heads=1, n_layers=1, ffn_dim=128, state_tokens=2, page_size=8, dropout=0.0)
    )
    model.eval()
    return model


def _soul(model: LivingReasoningCoreD64) -> SoulSnapshot:
    return SoulSnapshot(
        core_id="pointer-core", architecture_id=model.architecture_id,
        parameter_generation="pointer-generation", generation=0,
        parent_soul_id=None, layers=empty_soul_layers(),
    )


def test_pointer_bootstrap_targets_are_disjoint_native_scalars_at_exact_addresses() -> None:
    curriculum = build_pointer_bootstrap_curriculum()
    train = curriculum.split("train")
    heldout = curriculum.split("heldout")
    assert train and heldout
    assert all(item.target_basis == POINTER_BOOTSTRAP_SCHEMA for item in curriculum.episodes)
    assert {item.targets[0].text for item in train}.isdisjoint({item.targets[0].text for item in heldout})
    positions = set()
    for episode in curriculum.episodes:
        target = episode.targets[0]
        assert len(target.text) == 1
        assert target.text.strip()
        segment = target.text_alignment["segments"][0]
        cortex = _region_text(episode, "cortex")
        position = segment["source_start"]
        positions.add(position)
        assert cortex[position] == target.text
        assert segment["source_end"] == position + 1
    assert {31, 32, 33}.issubset(positions)


def test_pointer_bootstrap_gate_requires_all_heldout_mechanism_metrics() -> None:
    report = {
        "pointer_first_source_top1_rate": 1.0,
        "pointer_first_source_probability_mean": 1.0,
        "free_running_first_transport_accuracy": 1.0,
        "free_running_nonempty_valid_unicode_rate": 1.0,
        "complete_field_coverage_rate": 1.0,
    }
    assert decide_pointer_bootstrap_mastery(report)["passed"] is True
    report["pointer_first_source_top1_rate"] = 0.999
    assert decide_pointer_bootstrap_mastery(report)["passed"] is False


def test_evaluator_reports_first_pointer_and_free_running_measurements() -> None:
    model = _small_model()
    episode = build_pointer_bootstrap_curriculum().split("heldout")[0]
    report = evaluate_living_episode(
        model, episode, _soul(model), core_id="pointer-core", parameter_generation="pointer-generation"
    )
    for key in (
        "pointer_first_source_count", "pointer_first_source_top1_rate",
        "pointer_first_source_probability_mean", "free_running_first_transport_accuracy",
        "free_running_nonempty_valid_unicode_rate",
    ):
        assert key in report
    assert report["pointer_first_source_count"] == 2

from __future__ import annotations

import torch

from runtime.soul import SoulSnapshot, SoulTemperature, empty_soul_layers
from substrate import default_alphabet
from training.living_reasoning_curriculum import living_episode_objective
from training.living_reasoning_d64 import LivingReasoningCoreConfig, LivingReasoningCoreD64
from training.substrate_literacy_curriculum import (
    build_substrate_literacy_curriculum,
    decide_substrate_literacy_mastery,
)


def _small_model() -> LivingReasoningCoreD64:
    torch.manual_seed(17)
    model = LivingReasoningCoreD64(
        LivingReasoningCoreConfig(
            n_heads=1,
            n_layers=1,
            ffn_dim=128,
            state_tokens=2,
            page_size=8,
            dropout=0.0,
            generate_gate_bias=0.0,
        )
    )
    model.eval()
    return model


def _soul(model: LivingReasoningCoreD64) -> SoulSnapshot:
    return SoulSnapshot(
        core_id="core-substrate",
        architecture_id=model.architecture_id,
        parameter_generation="g-substrate",
        generation=0,
        parent_soul_id=None,
        layers=empty_soul_layers(),
    )


def test_substrate_curriculum_covers_frozen_native_bank_and_unseen_compositions() -> None:
    curriculum = build_substrate_literacy_curriculum()
    train = curriculum.split("train")
    heldout = curriculum.split("heldout")
    assert len(train) >= len(default_alphabet())
    assert len(heldout) >= 8
    assert all(item.target_basis == "axon-substrate-literacy-curriculum-v1" for item in curriculum.episodes)
    assert all(item.targets[0].supervision_weight == 1.0 for item in curriculum.episodes)
    assert all(item.targets[1].supervision_weight == 1.0 for item in curriculum.episodes)
    assert all(item.targets[2].supervision_weight == 0.0 for item in curriculum.episodes)
    assert all("substrate" in item.mechanism_tags for item in curriculum.episodes)


def test_substrate_life_threads_same_four_layer_soul_across_experiences() -> None:
    model = _small_model()
    curriculum = build_substrate_literacy_curriculum()
    soul = _soul(model)
    assert tuple(layer.temperature for layer in soul.layers) == tuple(SoulTemperature)
    cold_before = tuple(soul.layer(temp).payload for temp in SoulTemperature if temp is not SoulTemperature.HOT)

    for episode in curriculum.split("train")[:2]:
        loss, unroll, _metrics = living_episode_objective(
            model,
            episode,
            soul,
            core_id="core-substrate",
            parameter_generation="g-substrate",
            text_eos_weight=1.0,
        )
        assert torch.isfinite(loss)
        soul = unroll.souls[-1]

    assert soul.generation == 6
    assert soul.layer(SoulTemperature.HOT).payload
    assert tuple(soul.layer(temp).payload for temp in SoulTemperature if temp is not SoulTemperature.HOT) == cold_before


def test_substrate_gate_is_independent_of_unsupervised_final_phase() -> None:
    report = {
        "text_exact_rate": 1.0,
        "text_teacher_forced_content_accuracy": 1.0,
        "text_teacher_forced_eos_accuracy": 1.0,
        "complete_field_coverage_rate": 1.0,
        "constant_text_exact_floor": 0.25,
        "final_verdict_valid_rate": 0.0,
    }
    decision = decide_substrate_literacy_mastery(report)
    assert decision["passed"] is True
    assert "final_verdict_valid_rate" not in decision["requirements"]

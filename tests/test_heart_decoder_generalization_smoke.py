import torch

from runtime.heart.translation_core import (
    HeartTranslationCore,
    HeartTranslationCoreConfig,
    migrate_heart_translation_v3_to_v4,
)
from runtime.trainer import PromotionGate
from scripts.train_heart_decoder_generalization_smoke import (
    _generalization_gate,
    _migration_equivalence,
    _training_roles,
)
from training.heart_translation import build_heart_decoder_mechanism_curriculum


def test_generalization_gate_is_complete_and_bound_to_candidate() -> None:
    gate = _generalization_gate("heart64a", "candidate-test")

    assert isinstance(gate, PromotionGate)
    assert gate.module_id == "heart64a"
    assert gate.candidate_generation_id == "candidate-test"
    assert len(gate.requirements) == 14
    assert gate.required_suite_ids == ("heart-decoder-generalization-v1",)

    positional = _generalization_gate(
        "heart64a",
        "candidate-v4-test",
        positional_copy=True,
    )
    assert len(positional.requirements) == 15
    names = {requirement.metric_name for requirement in positional.requirements}
    assert "mean_character_positional_copy_gate" in names
    assert "mean_eos_positional_copy_gate" in names
    assert "diagonal_attention_mass" not in names


def test_protective_training_roles_are_deterministic_and_replay_heavy() -> None:
    assert _training_roles(steps=6, replay_every=2, replay_steps_per_novel=1) == (
        "novel",
        "replay",
        "novel",
        "replay",
        "novel",
        "replay",
    )
    assert _training_roles(steps=6, replay_every=2, replay_steps_per_novel=2) == (
        "novel",
        "replay",
        "replay",
        "novel",
        "replay",
        "replay",
    )


def test_migration_equivalence_compares_tensor_pairs() -> None:
    config = HeartTranslationCoreConfig(
        d_model=64,
        n_heads=4,
        n_layers=1,
        ffn_dim=128,
        dropout=0.0,
        source_page_chars=32,
    )
    source = HeartTranslationCore(config)
    destination_config, state = migrate_heart_translation_v3_to_v4(
        source.state_dict(),
        config,
    )
    destination = HeartTranslationCore(destination_config)
    destination.load_state_dict(state, strict=True)
    evidence = _migration_equivalence(
        source,
        destination,
        build_heart_decoder_mechanism_curriculum().train_cases[:1],
        torch.device("cpu"),
    )

    assert evidence["passed"]
    assert evidence["maximum_absolute_delta"] == 0.0

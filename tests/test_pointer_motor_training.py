from __future__ import annotations

import torch

from runtime.soul import SoulSnapshot, SoulTemperature, empty_soul_layers
from training import (
    LivingReasoningCoreConfig,
    LivingReasoningCoreD64,
    build_pointer_motor_curriculum,
    initialize_pointer_scaffold_from_donor,
    pointer_key_cross_episode_accuracy,
    pointer_target_address,
    rebind_d64_soul_layers,
)


def _config(*, scaffold: bool) -> LivingReasoningCoreConfig:
    return LivingReasoningCoreConfig(
        n_heads=1,
        n_layers=1,
        ffn_dim=64,
        state_tokens=2,
        page_size=16,
        pointer_address_scaffold_version="query-scaffold-v1" if scaffold else "",
    )


def test_pointer_scaffold_migration_is_exact_and_bounded() -> None:
    torch.manual_seed(11)
    donor = LivingReasoningCoreD64(_config(scaffold=False))
    target = LivingReasoningCoreD64(_config(scaffold=True))
    receipt = initialize_pointer_scaffold_from_donor(
        target,
        donor.state_dict(),
        source_architecture_id=donor.architecture_id,
        source_parameter_generation="donor-generation",
        target_parameter_generation="target-generation",
        source_checkpoint_id="donor-checkpoint",
        source_bundle_id="donor-bundle",
        source_soul_id="donor-soul",
    )
    assert len(receipt.copied_tensors) == len(donor.state_dict())
    assert {
        item.name for item in receipt.initialized_tensors
    } == {
        "pointer_address_query.0.weight",
        "pointer_address_query.0.bias",
        "pointer_address_query.2.weight",
        "pointer_address_query.2.bias",
    }
    assert all(item.source_sha256 == item.target_sha256 for item in receipt.copied_tensors)


def test_pointer_curriculum_covers_exact_canonical_addresses() -> None:
    curriculum = build_pointer_motor_curriculum()
    assert len(curriculum.split("train")) == 68
    assert len(curriculum.split("heldout")) == 68
    assert {
        pointer_target_address(episode)[1] for episode in curriculum.split("heldout")
    } == set(range(68))


def test_rebound_d64_soul_preserves_recurrent_bytes_and_changes_dialect() -> None:
    source_config = _config(scaffold=False)
    target_config = _config(scaffold=True)
    source_core = LivingReasoningCoreD64(source_config)
    state = source_core.initial_state.unsqueeze(0)
    source_layer = source_core.soul_codec.encode(state, SoulTemperature.HOT)
    source = SoulSnapshot(
        core_id="pointer-test-core",
        architecture_id=source_core.architecture_id,
        parameter_generation="source-generation",
        generation=4,
        parent_soul_id=None,
        layers=(source_layer, *empty_soul_layers()[1:]),
    )
    rebound = rebind_d64_soul_layers(source, target_config)
    assert rebound[0].payload == source_layer.payload
    assert rebound[0].tensor_layout != source_layer.tensor_layout
    target_core = LivingReasoningCoreD64(target_config)
    decoded = target_core.soul_codec.decode(
        rebound[0],
        device=target_core.device,
        dtype=target_core.initial_state.dtype,
    )
    assert decoded is not None
    assert tuple(decoded.shape) == (1, target_config.state_tokens, target_config.d_model)


def test_pointer_key_geometry_reports_cross_episode_address_accuracy() -> None:
    rows = []
    for episode_index in range(3):
        rows.append(
            (
                f"episode-{episode_index}",
                {
                    position: torch.nn.functional.one_hot(
                        torch.tensor(position), num_classes=3
                    ).float()
                    for position in range(3)
                },
            )
        )
    report = pointer_key_cross_episode_accuracy(rows)
    assert report["accuracy"] == 1.0
    assert report["uniform_chance"] == 1.0 / 3.0

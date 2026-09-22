from __future__ import annotations

import pytest
import torch

from training.pointer_oracle_diagnostic import (
    cross_episode_cortex_key_accuracy,
    leave_one_group_out_centroid_accuracy,
    mix_living_probabilities,
)


def test_oracle_mixture_preserves_generated_eos_and_normalizes() -> None:
    generated = torch.tensor([0.10, 0.20, 0.30, 0.40])
    copied = torch.tensor([0.0, 1.0, 0.0, 0.0])

    mixed = mix_living_probabilities(generated, copied, 0.25, eos_index=3)
    forced = mix_living_probabilities(generated, copied, 0.0, eos_index=3)

    assert mixed.sum().item() == pytest.approx(1.0)
    assert forced.sum().item() == pytest.approx(1.0)
    assert mixed[3].item() == pytest.approx(generated[3].item())
    assert forced[3].item() == pytest.approx(generated[3].item())
    assert int(forced.argmax().item()) == 1
    assert forced[1].item() == pytest.approx(0.60)


def test_leave_one_group_out_centroids_detect_address_geometry() -> None:
    vectors = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.9, 0.1]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.1, 0.9]),
    ]
    result = leave_one_group_out_centroid_accuracy(
        vectors,
        labels=[0, 0, 1, 1],
        groups=["a", "b", "c", "d"],
    )

    assert result["eligible_count"] == 4.0
    assert result["accuracy"] == 1.0
    assert result["uniform_chance"] == 0.5


def test_cross_episode_key_geometry_uses_other_episodes_only() -> None:
    episodes = [
        ("a", {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}),
        ("b", {0: torch.tensor([0.9, 0.1]), 1: torch.tensor([0.1, 0.9])}),
        ("c", {0: torch.tensor([0.8, 0.2]), 1: torch.tensor([0.2, 0.8])}),
    ]

    result = cross_episode_cortex_key_accuracy(episodes)

    assert result["eligible_count"] == 6.0
    assert result["accuracy"] == 1.0
    assert result["uniform_chance"] == 0.5

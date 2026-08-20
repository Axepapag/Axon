from __future__ import annotations

import math
import random

import numpy as np
import torch

from training.checkpoint_contract import validate_checkpoint_continuity


FAMILIES = ["runtime_v2", "structured_v2", "scratchpad_v2"]


def _payload() -> dict:
    return {
        "checkpoint_schema": "axon_charslot_checkpoint_v2",
        "step": 250000,
        "optimizer_state": {"state": {0: {"step": 1}}, "param_groups": [{"params": [0]}]},
        "training_state": {
            "schema": "axon_training_state_v1",
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": [torch.arange(4, dtype=torch.uint8)],
            "curriculum_state": {
                "schema": "axon_phase0b_sampler_v1",
                "position": 2,
                "order": [2, 0, 1],
                "rng_state": random.Random(17).getstate(),
                "example_count": 3,
                "families": FAMILIES,
            },
        },
    }


def test_checkpoint_continuity_contract_accepts_complete_payload() -> None:
    result = validate_checkpoint_continuity(
        _payload(), expected_step=250000, expected_families=FAMILIES
    )

    assert result["valid"] is True
    assert result["optimizer"] == {"valid": True, "state_entries": 1, "param_groups": 1}
    assert result["rng"]["cuda"] is True
    assert result["sampler"]["order_length"] == 3


def test_checkpoint_continuity_contract_rejects_empty_and_mismatched_state() -> None:
    payload = _payload()
    payload["optimizer_state"] = {"state": {}, "param_groups": []}
    payload["training_state"]["cuda_rng_state_all"] = []
    payload["training_state"]["curriculum_state"]["order"] = [0, 0, 1]
    payload["training_state"]["curriculum_state"]["families"] = ["legacy_v1"]

    result = validate_checkpoint_continuity(
        payload, expected_step=250000, expected_families=FAMILIES
    )

    assert result["valid"] is False
    assert "optimizer state entries are empty" in result["violations"]
    assert "optimizer parameter groups are empty" in result["violations"]
    assert "cuda_rng_state_all is missing or empty" in result["violations"]
    assert "sampler order is not a complete permutation" in result["violations"]
    assert "sampler families do not match the exact curriculum manifest" in result["violations"]


def _add_pilot_provenance(payload: dict) -> None:
    payload["optimizer_state"]["param_groups"][0]["lr"] = 1e-4
    payload["continuity_provenance"] = {
        "schema": "axon_continuity_provenance_v1",
        "source_step": 250000,
        "optimizer": {
            "policy": "restore_state_override_lr",
            "state_restored": True,
            "effective_lr": 1e-4,
        },
        "sampler": {
            "policy": "reset_exact_v3",
            "reset": True,
            "initial_position": 0,
            "families": FAMILIES,
        },
    }


def _weighted_payload() -> dict:
    payload = _payload()
    payload["step"] = 252000
    curriculum = payload["training_state"]["curriculum_state"]
    curriculum.update(
        {
            "position": 4,
            "order": [2, 0, 5, 1, 4, 3],
            "example_count": 6,
            "family_sampling_policy": "weighted",
            "family_weights": {
                FAMILIES[0]: 0.3,
                FAMILIES[1]: 0.4,
                FAMILIES[2]: 0.3,
            },
            "family_orders": {
                FAMILIES[0]: [0, 3],
                FAMILIES[1]: [1, 4],
                FAMILIES[2]: [2, 5],
            },
            "family_positions": {
                FAMILIES[0]: 1,
                FAMILIES[1]: 2,
                FAMILIES[2]: 1,
            },
            "family_credits": {
                FAMILIES[0]: 0.2,
                FAMILIES[1]: -0.4,
                FAMILIES[2]: 0.2,
            },
            "family_steps": 4,
            "family_exhaustion_policy": "error",
        }
    )
    _add_pilot_provenance(payload)
    payload["continuity_provenance"]["sampler"].update(
        {
            "family_sampling_policy": "weighted",
            "family_weights": dict(curriculum["family_weights"]),
            "family_exhaustion_policy": "error",
        }
    )
    return payload


def test_checkpoint_contract_validates_pilot_optimizer_and_sampler_provenance() -> None:
    payload = _payload()
    payload["step"] = 252000
    _add_pilot_provenance(payload)

    result = validate_checkpoint_continuity(
        payload,
        expected_step=252000,
        expected_families=FAMILIES,
        expected_source_step=250000,
        expected_optimizer_policy="restore_state_override_lr",
        expected_optimizer_state_restored=True,
        expected_effective_lr=1e-4,
        expected_sampler_policy="reset_exact_v3",
        expected_sampler_reset=True,
    )

    assert result["valid"] is True
    assert result["provenance"] == {
        "valid": True,
        "schema": "axon_continuity_provenance_v1",
        "source_step": 250000,
        "optimizer": {
            "policy": "restore_state_override_lr",
            "state_restored": True,
            "effective_lr": 1e-4,
        },
        "sampler": {
            "policy": "reset_exact_v3",
            "reset": True,
            "initial_position": 0,
            "families": FAMILIES,
        },
    }


def test_checkpoint_contract_fails_closed_on_lr_and_reset_provenance() -> None:
    payload = _payload()
    payload["step"] = 252000
    _add_pilot_provenance(payload)
    payload["optimizer_state"]["param_groups"][0]["lr"] = math.nan
    payload["continuity_provenance"]["sampler"]["initial_position"] = 3

    result = validate_checkpoint_continuity(
        payload,
        expected_step=252000,
        expected_families=FAMILIES,
        expected_source_step=250000,
        expected_optimizer_policy="restore_state_override_lr",
        expected_optimizer_state_restored=True,
        expected_effective_lr=1e-4,
        expected_sampler_policy="reset_exact_v3",
        expected_sampler_reset=True,
    )

    assert result["valid"] is False
    assert (
        "optimizer parameter group 0 learning rate is missing or non-finite"
        in result["violations"]
    )
    assert "reset sampler provenance initial_position is not zero" in result["violations"]


def test_checkpoint_contract_accepts_complete_weighted_sampler_state() -> None:
    result = validate_checkpoint_continuity(
        _weighted_payload(),
        expected_step=252000,
        expected_families=FAMILIES,
        expected_source_step=250000,
        expected_optimizer_policy="restore_state_override_lr",
        expected_optimizer_state_restored=True,
        expected_effective_lr=1e-4,
        expected_sampler_policy="reset_exact_v3",
        expected_sampler_reset=True,
    )

    assert result["valid"] is True
    assert result["sampler"]["family_sampling_policy"] == "weighted"
    assert result["sampler"]["weighted"]["valid"] is True
    assert result["sampler"]["weighted"]["provenance_matched"] is True


def test_checkpoint_contract_rejects_incomplete_weighted_sampler_state() -> None:
    payload = _weighted_payload()
    curriculum = payload["training_state"]["curriculum_state"]
    del curriculum["family_weights"][FAMILIES[2]]
    curriculum["family_orders"][FAMILIES[0]] = [0, 1]
    curriculum["family_positions"][FAMILIES[0]] = 3
    curriculum["family_credits"][FAMILIES[0]] = math.inf
    curriculum["family_steps"] = 4
    curriculum["family_exhaustion_policy"] = "wrap"
    payload["continuity_provenance"]["sampler"]["family_sampling_policy"] = "global"
    payload["continuity_provenance"]["sampler"]["family_weights"][FAMILIES[0]] = 0.5

    result = validate_checkpoint_continuity(
        payload,
        expected_step=252000,
        expected_families=FAMILIES,
    )

    assert result["valid"] is False
    assert (
        "weighted sampler family_weights keys do not match expected families"
        in result["violations"]
    )
    assert (
        "weighted sampler family_orders do not partition global sampler indices"
        in result["violations"]
    )
    assert (
        f"weighted sampler family position is out of bounds: {FAMILIES[0]}"
        in result["violations"]
    )
    assert (
        "weighted sampler family_steps does not equal consumed positions"
        in result["violations"]
    )
    assert "weighted sampler family_credits must be finite" in result["violations"]
    assert "weighted sampler family exhaustion policy is not error" in result["violations"]
    assert (
        "weighted sampler provenance family policy does not match training state"
        in result["violations"]
    )
    assert (
        "weighted sampler provenance family_weights do not match training state"
        in result["violations"]
    )


def test_checkpoint_contract_replays_weighted_positions_and_credits() -> None:
    payload = _weighted_payload()
    curriculum = payload["training_state"]["curriculum_state"]
    curriculum["family_positions"] = {
        FAMILIES[0]: 2,
        FAMILIES[1]: 1,
        FAMILIES[2]: 1,
    }
    curriculum["family_credits"] = {
        FAMILIES[0]: 0.1,
        FAMILIES[1]: -0.3,
        FAMILIES[2]: 0.2,
    }

    result = validate_checkpoint_continuity(
        payload,
        expected_step=252000,
        expected_families=FAMILIES,
    )

    assert result["valid"] is False
    assert (
        "weighted sampler family_positions do not match deterministic schedule"
        in result["violations"]
    )
    assert (
        "weighted sampler family_credits do not match deterministic schedule"
        in result["violations"]
    )


def test_checkpoint_contract_marks_weighted_provenance_mismatch_invalid() -> None:
    payload = _weighted_payload()
    payload["continuity_provenance"]["sampler"]["family_weights"] = {
        FAMILIES[0]: 0.5,
        FAMILIES[1]: 0.2,
        FAMILIES[2]: 0.3,
    }

    result = validate_checkpoint_continuity(
        payload,
        expected_step=252000,
        expected_families=FAMILIES,
        expected_source_step=250000,
        expected_optimizer_policy="restore_state_override_lr",
        expected_optimizer_state_restored=True,
        expected_effective_lr=1e-4,
        expected_sampler_policy="reset_exact_v3",
        expected_sampler_reset=True,
    )

    assert result["valid"] is False
    assert result["provenance"]["valid"] is False
    assert (
        "weighted sampler provenance family_weights do not match training state"
        in result["violations"]
    )

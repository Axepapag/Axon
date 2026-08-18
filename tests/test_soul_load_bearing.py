from __future__ import annotations

import math

import pytest
import torch

from cores.soul_v2 import SoulState, SoulV2Config, TierSpec
from training.soul_load_bearing import (
    SOUL_LOAD_BEARING_SCHEMA,
    SoulCausalCase,
    SoulLoadBearingThresholds,
    SoulResponderOutput,
    bootstrap_mean_ci,
    build_causal_soul_conditions,
    clone_soul_state,
    counterfactual_margin_loss,
    evaluate_soul_load_bearing,
    soul_state_sha256,
)


def _state(code: float, *, d_model: int = 4) -> SoulState:
    cfg = SoulV2Config(
        d_model=d_model,
        tiers=[
            TierSpec(name="hot", max_rows=6),
            TierSpec(name="warm", max_rows=3),
        ],
        categories=["episodic", "lessons"],
    )
    state = SoulState(cfg, torch.device("cpu"), torch.float32)
    state.active[:] = True
    state.category[:3] = 0
    state.category[3:6] = 1
    state.category[6:] = -1
    for row in range(state.total_rows):
        state.tensor[row] = torch.arange(d_model) + code + row * 10.0
        state.salience[row] = 0.05 * (row + 1)
        state.dormant_for[row] = row
        state.tick_born[row] = 100 + row
    state.current_tick = 999
    return state


def _assert_metadata_equal(left: SoulState, right: SoulState) -> None:
    assert left.cfg.to_dict() == right.cfg.to_dict()
    assert left.current_tick == right.current_tick
    for field in (
        "active",
        "tier",
        "category",
        "salience",
        "dormant_for",
        "tick_born",
    ):
        assert torch.equal(getattr(left, field), getattr(right, field))


def test_causal_conditions_are_pure_complete_clones() -> None:
    owner = _state(1.0)
    donor = _state(1_001.0)
    owner_before = soul_state_sha256(owner)
    donor_before = soul_state_sha256(donor)

    conditions = build_causal_soul_conditions(
        owner,
        donor,
        shuffle_seed=12345,
    )

    assert soul_state_sha256(owner) == owner_before
    assert soul_state_sha256(donor) == donor_before
    assert conditions.correct is not owner
    assert soul_state_sha256(conditions.correct) == owner_before
    assert soul_state_sha256(conditions.swapped) == donor_before

    assert torch.count_nonzero(conditions.zero.tensor).item() == 0
    _assert_metadata_equal(conditions.zero, owner)
    _assert_metadata_equal(conditions.shuffled, owner)
    _assert_metadata_equal(conditions.swapped, donor)

    assert not torch.equal(conditions.shuffled.tensor, owner.tensor)
    repeated = build_causal_soul_conditions(
        owner,
        donor,
        shuffle_seed=12345,
    )
    assert torch.equal(conditions.shuffled.tensor, repeated.shuffled.tensor)

    # Every shuffled row receives content from its own tier/category group.
    for row in range(owner.total_rows):
        group = (
            (owner.tier == owner.tier[row])
            & (owner.category == owner.category[row])
            & owner.active
        )
        possible = owner.tensor[group]
        assert any(
            torch.equal(conditions.shuffled.tensor[row], candidate)
            for candidate in possible
        )
        assert not torch.equal(conditions.shuffled.tensor[row], owner.tensor[row])


def test_clone_has_no_storage_aliases() -> None:
    original = _state(4.0)
    cloned = clone_soul_state(original)
    cloned.tensor[0, 0] = -1
    cloned.active[0] = False
    cloned.salience[0] = -1

    assert original.tensor[0, 0].item() == 4.0
    assert original.active[0].item() is True
    assert original.salience[0].item() > 0


def test_swap_rejects_a_different_core_layout() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        build_causal_soul_conditions(
            _state(1.0, d_model=4),
            _state(2.0, d_model=8),
            shuffle_seed=1,
        )


def _causal_cases(count: int = 4) -> list[SoulCausalCase]:
    cases = []
    for index in range(count):
        owner_target = f"owner-{index}"
        donor_target = f"donor-{index}"
        cases.append(
            SoulCausalCase(
                case_id=f"case-{index}",
                visible_input={
                    "owner_target": owner_target,
                    "donor_target": donor_target,
                    "owner_code": float(index + 1),
                },
                owner_state=_state(float(index + 1)),
                donor_state=_state(float(1_001 + index)),
                owner_target=owner_target,
                donor_target=donor_target,
                abstention_target="unknown",
                shuffle_seed=index * 17,
            )
        )
    return cases


def _toy_responder(visible_input: dict, soul: SoulState) -> SoulResponderOutput:
    owner = visible_input["owner_target"]
    donor = visible_input["donor_target"]
    abstain = "unknown"
    first = float(soul.tensor[0, 0])

    if torch.count_nonzero(soul.tensor).item() == 0:
        prediction = abstain
        nll = {owner: 0.50, donor: 0.90, abstain: 0.01}
    elif first >= 1_000:
        prediction = donor
        nll = {owner: 0.60, donor: 0.01, abstain: 0.90}
    elif math.isclose(first, visible_input["owner_code"]):
        prediction = owner
        nll = {owner: 0.05, donor: 0.90, abstain: 0.90}
    else:
        prediction = abstain
        nll = {owner: 0.50, donor: 0.90, abstain: 0.01}
    return SoulResponderOutput(prediction=prediction, target_nll=nll)


def _development_thresholds() -> SoulLoadBearingThresholds:
    return SoulLoadBearingThresholds(
        min_cases=4,
        correct_exact_min=1.0,
        correct_char_min=1.0,
        zero_nll_gap_mean_min=0.40,
        swapped_nll_gap_mean_min=0.40,
        shuffled_nll_gap_mean_min=0.40,
        nll_gap_ci_lower_exclusive_min=0.39,
        swapped_donor_follow_min=1.0,
        zero_abstention_exact_min=1.0,
        shuffled_abstention_exact_min=1.0,
    )


def test_per_core_causal_metrics_and_gate_pass() -> None:
    cases = _causal_cases()
    hashes_before = [
        (soul_state_sha256(case.owner_state), soul_state_sha256(case.donor_state))
        for case in cases
    ]

    report = evaluate_soul_load_bearing(
        core_id="coreB128",
        cases=cases,
        responder=_toy_responder,
        thresholds=_development_thresholds(),
        bootstrap_samples=250,
        seed=77,
    )

    assert report.passed is True
    assert report.violations == ()
    assert report.schema == SOUL_LOAD_BEARING_SCHEMA
    assert len(report.suite_sha256) == 64
    assert report.metrics["immutable"] is True
    assert report.metrics["conditions"]["correct"]["exact"] == 1.0
    assert report.metrics["conditions"]["correct"]["char_accuracy"] == 1.0
    assert report.metrics["conditions"]["swapped"]["exact"] == 1.0
    assert report.metrics["swapped_donor_follow"] == 1.0
    assert report.metrics["abstention_exact"] == {
        "zero": 1.0,
        "shuffled": 1.0,
        "combined": 1.0,
    }
    assert report.metrics["owner_nll_gaps"]["zero"]["mean"] == pytest.approx(0.45)
    assert report.metrics["owner_nll_gaps"]["swapped"]["mean"] == pytest.approx(0.55)
    assert report.metrics["owner_nll_gaps"]["shuffled"]["mean"] == pytest.approx(0.45)
    assert report.to_dict()["schema"] == SOUL_LOAD_BEARING_SCHEMA

    assert [
        (soul_state_sha256(case.owner_state), soul_state_sha256(case.donor_state))
        for case in cases
    ] == hashes_before


def test_gate_fails_closed_when_responder_mutates_a_clone() -> None:
    def mutating_responder(
        visible_input: dict,
        soul: SoulState,
    ) -> SoulResponderOutput:
        output = _toy_responder(visible_input, soul)
        soul.tensor[0, 0] += 1
        return output

    report = evaluate_soul_load_bearing(
        core_id="coreA64",
        cases=_causal_cases(),
        responder=mutating_responder,
        thresholds=_development_thresholds(),
        bootstrap_samples=25,
    )

    assert report.passed is False
    assert report.metrics["immutable"] is False
    assert any("responder mutated soul state" in item for item in report.violations)
    assert "causal evaluation was not immutable" in report.violations


def test_gate_fails_closed_on_missing_nll_and_too_few_cases() -> None:
    def incomplete_responder(
        visible_input: dict,
        soul: SoulState,
    ) -> SoulResponderOutput:
        return SoulResponderOutput(
            prediction=visible_input["owner_target"],
            target_nll={visible_input["owner_target"]: math.nan},
        )

    report = evaluate_soul_load_bearing(
        core_id="coreB128",
        cases=_causal_cases(1),
        responder=incomplete_responder,
        thresholds=_development_thresholds(),
        bootstrap_samples=10,
    )

    assert report.passed is False
    assert "n_cases=1 below minimum 4" in report.violations
    assert any("invalid NLL" in item for item in report.violations)
    assert any("missing NLL" in item for item in report.violations)
    assert any("non-finite" in item for item in report.violations)


def test_bootstrap_interval_is_deterministic() -> None:
    first = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], samples=500, seed=9)
    second = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], samples=500, seed=9)
    assert first == second
    assert first.lower <= first.mean <= first.upper


def test_locked_defaults_use_per_condition_gaps_and_positive_ci() -> None:
    thresholds = SoulLoadBearingThresholds()
    assert thresholds.zero_nll_gap_mean_min == 0.50
    assert thresholds.swapped_nll_gap_mean_min == 0.30
    assert thresholds.shuffled_nll_gap_mean_min == 0.30
    assert thresholds.nll_gap_ci_lower_exclusive_min == 0.0


def test_counterfactual_margin_loss_is_differentiable() -> None:
    correct = torch.tensor(
        [[[0.2, 0.0, -0.1], [0.0, 0.1, 0.2]]],
        requires_grad=True,
    )
    counterfactual = torch.tensor(
        [[[0.1, 0.2, -0.1], [0.1, 0.0, 0.2]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 1]])

    loss = counterfactual_margin_loss(
        correct,
        counterfactual,
        targets,
        margin=0.5,
    )
    loss.backward()

    assert loss.item() > 0
    assert correct.grad is not None
    assert counterfactual.grad is not None
    assert torch.isfinite(correct.grad).all()
    assert torch.isfinite(counterfactual.grad).all()
    assert correct.grad.abs().sum().item() > 0
    assert counterfactual.grad.abs().sum().item() > 0


def test_margin_loss_honors_ignore_index_and_stacked_counterfactuals() -> None:
    correct = torch.zeros(1, 2, 3, requires_grad=True)
    counterfactuals = torch.zeros(2, 1, 2, 3, requires_grad=True)
    targets = torch.tensor([[1, -100]])

    per_position = counterfactual_margin_loss(
        correct,
        counterfactuals,
        targets,
        margin=0.25,
        reduction="none",
    )

    assert per_position.shape == targets.shape
    assert per_position[0, 0].item() == pytest.approx(0.25)
    assert per_position[0, 1].item() == 0.0

from __future__ import annotations

import math

import pytest
import torch

from cores.core import AxonCore, CoreConfig
from cores.soul_v2 import SoulState, SoulV2Config, TierSpec
from training.soul_core_adapter import AxonCoreSoulAdapter, FieldView
from training.soul_load_bearing import (
    SoulCausalCase,
    SoulLoadBearingThresholds,
    clone_soul_state,
    evaluate_soul_load_bearing,
    soul_state_sha256,
)


def _tiny_core() -> AxonCore:
    torch.manual_seed(1234)
    cfg = CoreConfig(
        d_model=16,
        n_heads=1,
        n_layers=1,
        ffn_dim=32,
        dropout=0.0,
        n_ticks=1,
        grad_ticks=1,
        soul_rows=4,
        soul_mode="act_reflect_v2",
        soul_gate_init=1.0,
        soul_hot_rows=0,
        char_slot_mode=True,
        char_slot_max_slots=8,
        char_n_regions=3,
    )
    core = AxonCore(cfg)
    core.soul_readonly = True
    with torch.no_grad():
        for layer in core.layers:
            layer.soul_cross_gate.fill_(1.0)
    return core


def _soul(seed: int, *, active: bool = True, d_model: int = 16) -> SoulState:
    cfg = SoulV2Config(
        d_model=d_model,
        tiers=[TierSpec(name="hot", max_rows=4)],
        categories=["episodic"],
    )
    state = SoulState(cfg, torch.device("cpu"), torch.float32)
    generator = torch.Generator().manual_seed(seed)
    state.tensor = torch.randn(4, d_model, generator=generator)
    state.active.fill_(active)
    state.category.fill_(0)
    state.salience[:] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    state.dormant_for[:] = torch.tensor([1, 2, 3, 4])
    state.tick_born[:] = torch.tensor([10, 20, 30, 40])
    state.current_tick = 55
    return state


def _view(*, slots: int = 6, target: str = "aa") -> FieldView:
    generator = torch.Generator().manual_seed(91)
    field16 = torch.randn(1, slots, 16, generator=generator)
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]])[:, :slots]
    mask = torch.ones(1, slots, dtype=torch.bool)
    return FieldView(
        field16=field16,
        physical_role_ids=roles,
        attention_mask=mask,
        response_start=slots - 2,
        response_stop=slots,
        score_targets=(target, "bb", "cc"),
    )


def test_real_core_adapter_uses_identical_visible_tensors_and_soul_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _tiny_core()
    core.train()
    adapter = AxonCoreSoulAdapter(core)
    view = _view()
    owner = _soul(1)
    zero = clone_soul_state(owner)
    zero.tensor.zero_()
    owner_before = soul_state_sha256(owner)
    zero_before = soul_state_sha256(zero)

    calls: list[dict[str, torch.Tensor]] = []
    original_forward = core.forward_charslot

    def capture_forward(**kwargs):
        calls.append(
            {
                "field16": kwargs["field16"].detach().clone(),
                "roles": kwargs["region_ids"].detach().clone(),
                "mask": kwargs["mask"].detach().clone(),
                "soul_mask": kwargs["soul_mask"].detach().clone(),
            }
        )
        return original_forward(**kwargs)

    monkeypatch.setattr(core, "forward_charslot", capture_forward)
    owner_result = adapter.evaluate(view, owner)
    zero_result = adapter.evaluate(view, zero)

    assert len(calls) == 2
    for key in ("field16", "roles", "mask"):
        assert torch.equal(calls[0][key], calls[1][key])
    assert torch.equal(calls[0]["roles"], view.physical_role_ids)
    assert torch.equal(calls[0]["mask"], view.attention_mask)
    assert torch.equal(calls[0]["soul_mask"], owner.active.unsqueeze(0))
    assert not torch.equal(owner_result.logits, zero_result.logits)
    assert (owner_result.logits - zero_result.logits).abs().max().item() > 1e-6
    assert core.training is True
    assert soul_state_sha256(owner) == owner_before
    assert soul_state_sha256(zero) == zero_before
    assert torch.equal(view.field16, _view().field16)

    for result in (owner_result, zero_result):
        assert set(result.response.target_nll) == {"aa", "bb", "cc"}
        assert all(
            math.isfinite(value) and value >= 0.0
            for value in result.response.target_nll.values()
        )


def test_zero_content_and_all_masked_soul_are_finite_baselines() -> None:
    core = _tiny_core()
    adapter = AxonCoreSoulAdapter(core)
    view = _view()
    zero = _soul(2)
    zero.tensor.zero_()
    all_masked = _soul(3, active=False)
    zero_before = soul_state_sha256(zero)
    masked_before = soul_state_sha256(all_masked)

    zero_result = adapter.evaluate(view, zero)
    masked_result = adapter.evaluate(view, all_masked)

    assert torch.isfinite(zero_result.logits).all()
    assert torch.isfinite(masked_result.logits).all()
    assert torch.allclose(zero_result.logits, masked_result.logits)
    assert soul_state_sha256(zero) == zero_before
    assert soul_state_sha256(all_masked) == masked_before


def test_generic_gate_executes_a_tiny_real_axon_core() -> None:
    core = _tiny_core()
    adapter = AxonCoreSoulAdapter(core)
    owner = _soul(11)
    donor = _soul(22)
    view = _view()
    case = SoulCausalCase(
        case_id="tiny-real-core",
        visible_input=view,
        owner_state=owner,
        donor_state=donor,
        owner_target="aa",
        donor_target="bb",
        abstention_target="cc",
        shuffle_seed=5,
    )
    relaxed = SoulLoadBearingThresholds(
        min_cases=1,
        correct_exact_min=0.0,
        correct_char_min=0.0,
        zero_nll_gap_mean_min=-1_000_000.0,
        swapped_nll_gap_mean_min=-1_000_000.0,
        shuffled_nll_gap_mean_min=-1_000_000.0,
        nll_gap_ci_lower_exclusive_min=-1_000_000.0,
        swapped_donor_follow_min=0.0,
        zero_abstention_exact_min=0.0,
        shuffled_abstention_exact_min=0.0,
    )

    report = evaluate_soul_load_bearing(
        core_id="tiny16",
        cases=[case],
        responder=adapter,
        thresholds=relaxed,
        bootstrap_samples=25,
        seed=7,
    )

    assert report.passed is True
    assert report.violations == ()
    assert report.metrics["immutable"] is True
    assert report.metrics["conditions"]["correct"]["n"] == 1
    for condition in ("correct", "zero", "swapped", "shuffled"):
        assert math.isfinite(
            report.metrics["conditions"][condition]["owner_target_nll"]
        )


def test_adapter_accepts_exact_mapping_and_rejects_condition_labels() -> None:
    adapter = AxonCoreSoulAdapter(_tiny_core())
    view = _view()
    mapping = dict(view)
    result = adapter.evaluate(mapping, _soul(1))
    assert torch.isfinite(result.logits).all()

    mapping["condition"] = "correct"
    with pytest.raises(ValueError, match="unexpected keys: condition"):
        adapter.evaluate(mapping, _soul(1))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda view: FieldView(
                torch.zeros(1, 6, 15),
                view.physical_role_ids,
                view.attention_mask,
                view.response_start,
                view.response_stop,
                view.score_targets,
            ),
            "final dimension must be 16",
        ),
        (
            lambda view: FieldView(
                torch.zeros(1, 9, 16),
                torch.zeros(1, 9, dtype=torch.long),
                torch.ones(1, 9, dtype=torch.bool),
                7,
                9,
                view.score_targets,
            ),
            "exceeds core maximum",
        ),
        (
            lambda view: FieldView(
                view.field16,
                view.physical_role_ids[:, :-1],
                view.attention_mask,
                view.response_start,
                view.response_stop,
                view.score_targets,
            ),
            "physical_role_ids must have shape",
        ),
        (
            lambda view: FieldView(
                view.field16,
                view.physical_role_ids,
                view.attention_mask,
                view.response_start,
                view.response_stop,
                ("aaa",),
            ),
            "exceeds response slot width",
        ),
    ],
)
def test_adapter_fails_closed_on_slot_and_target_contracts(
    mutate,
    message: str,
) -> None:
    adapter = AxonCoreSoulAdapter(_tiny_core())
    with pytest.raises(ValueError, match=message):
        adapter.evaluate(mutate(_view()), _soul(1))


def test_adapter_rejects_wrong_soul_d_model() -> None:
    adapter = AxonCoreSoulAdapter(_tiny_core())
    with pytest.raises(ValueError, match="does not match core d_model"):
        adapter.evaluate(_view(), _soul(1, d_model=32))

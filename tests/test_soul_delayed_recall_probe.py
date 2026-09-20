from __future__ import annotations

import math

import pytest
import torch

from runtime.heart import EnglishProposal, ReasoningPassResult, TechnicalFinalVerdict
from runtime.soul import SoulSnapshot, empty_soul_layers
from training.living_reasoning_d64 import LivingReasoningCoreConfig, LivingReasoningCoreD64
from training.soul_delayed_recall_probe import (
    build_delayed_soul_recall_cases,
    decide_delayed_soul_recall_probe,
    run_delayed_soul_recall_probe,
)


def _small_model() -> LivingReasoningCoreD64:
    torch.manual_seed(29)
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
        core_id="core-soul-probe",
        architecture_id=model.architecture_id,
        parameter_generation="g-probe",
        generation=0,
        parent_soul_id=None,
        layers=empty_soul_layers(),
    )


def _all_text(snapshot) -> str:
    return "\n".join(region.text for region in snapshot.regions)


def test_delayed_recall_cases_are_disjoint_and_hide_replies_after_study() -> None:
    cases = build_delayed_soul_recall_cases()
    assert len(cases) == 8
    assert len({item.cue for item in cases}) == 8
    replies = [reply for item in cases for reply in (item.intact_reply, item.swapped_reply)]
    assert len(replies) == len(set(replies))

    for case in cases:
        assert case.intact_reply in _all_text(case.study_snapshot)
        assert case.swapped_reply in _all_text(case.swapped_study_snapshot)
        assert case.intact_reply not in _all_text(case.recall_snapshot)
        assert case.swapped_reply not in _all_text(case.recall_snapshot)
        for neutral in case.neutral_snapshots:
            assert case.intact_reply not in _all_text(neutral)
            assert case.swapped_reply not in _all_text(neutral)


def _passing_gate_report() -> dict[str, float | int]:
    return {
        "case_count": 8,
        "intact_exact_rate": 1.0,
        "swapped_exact_rate": 1.0,
        "reset_original_exact_rate": 0.0,
        "irrelevant_original_exact_rate": 0.0,
        "swapped_original_exact_rate": 0.0,
        "intact_termination_rate": 1.0,
        "swapped_termination_rate": 1.0,
        "intact_history_completion_rate": 1.0,
        "reset_history_completion_rate": 1.0,
        "swapped_history_completion_rate": 1.0,
        "irrelevant_history_completion_rate": 1.0,
        "input_target_absence_rate": 1.0,
        "constant_intact_exact_floor": 0.125,
    }


def test_delayed_recall_gate_is_counterfactual_and_fail_closed() -> None:
    report = _passing_gate_report()
    assert decide_delayed_soul_recall_probe(report)["passed"] is True

    contaminated = dict(report)
    contaminated["reset_original_exact_rate"] = 0.125
    assert decide_delayed_soul_recall_probe(contaminated)["passed"] is False

    nonfinite = dict(report)
    nonfinite["intact_exact_rate"] = math.nan
    decision = decide_delayed_soul_recall_probe(nonfinite)
    assert decision["passed"] is False
    assert decision["failures"]


def test_delayed_recall_probe_runs_through_runtime_without_target_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model()

    def scripted_emit(self: LivingReasoningCoreD64, request):
        forward = self.forward_request(request)
        transition = self.exhale_transition(
            before=request.soul,
            exhaled_state=forward.exhaled_state,
            tick_uid=request.image.identity.tick_uid,
            request_id=request.request_id,
            phase=request.phase,
        )
        common = {
            "base_field_id": request.image.identity.base_field_id,
            "base_tick_id": request.image.identity.base_tick_id,
            "author_core_id": request.descriptor.core_id,
            "rail_d_model": request.descriptor.d_model,
        }
        if request.phase == "first":
            output = EnglishProposal(pass_id="first", text="memory", **common)
        elif request.phase == "refined":
            output = EnglishProposal(pass_id="refined", text="memory refined", **common)
        else:
            output = TechnicalFinalVerdict(text="#scratch# memory", **common)
        return ReasoningPassResult(output=output, soul_transition=transition)

    monkeypatch.setattr(LivingReasoningCoreD64, "emit", scripted_emit)
    case = build_delayed_soul_recall_cases()[0]
    report = run_delayed_soul_recall_probe(
        model,
        _soul(model),
        core_id="core-soul-probe",
        parameter_generation="g-probe",
        cases=(case,),
    )

    assert report["case_count"] == 1
    assert len(report["rows"]) == 4
    assert {row["condition"] for row in report["rows"]} == {
        "intact",
        "reset",
        "swapped",
        "irrelevant",
    }
    assert all(row["history_completed"] for row in report["rows"])
    assert all(not row["recall_input_has_intact_reply"] for row in report["rows"])
    assert all(not row["recall_input_has_swapped_reply"] for row in report["rows"])
    assert all(not row["neutral_input_has_intact_reply"] for row in report["rows"])
    assert all(not row["neutral_input_has_swapped_reply"] for row in report["rows"])
    assert report["input_target_absence_rate"] == 1.0
    assert report["probe_id"]

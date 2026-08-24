from __future__ import annotations

import torch

from runtime.field import FieldDelta, SharedFieldSnapshot
from training.complete_field_64d import CompleteField64D, ReaderConfig
from training.train_complete_field_64d import (
    _forward_record,
    _read_record_with_scratch,
    _run_record,
)


def _record() -> dict:
    target_alignment = {
        "schema": "axon-r0-target-alignment-v1",
        "segments": [],
        "supervise_eos_generate": True,
    }
    return {
        "example_id": "canonical-training-adapter",
        "family": "test",
        "field": {
            "conversation_history": "",
            "user_input": "Where is the dog?",
            "response_draft": "",
            "cortex": "The dog is at the park.",
            "situation_awareness": "",
            "scratch": "",
            "tool_results": "",
            "advisor_input": "",
            "task_state": "",
            "diary": "",
        },
        "targets": {
            "scratch": "Use the evidence.",
            "response_draft": "The dog is at the park.",
        },
        "alignment": {
            "schema": "axon-r0-source-alignment-v1",
            "scratch": target_alignment,
            "response_draft": target_alignment,
            "response_counterfactuals": {},
        },
        "response_counterfactuals": [],
    }


def test_training_forward_uses_real_snapshots_and_field_deltas() -> None:
    model = CompleteField64D(
        ReaderConfig(page_size=8, inference_budget_chars=128, dropout=0.0)
    )
    model.eval()
    record = _record()
    with torch.no_grad():
        out = _forward_record(model, record)
    assert isinstance(out["base_snapshot"], SharedFieldSnapshot)
    assert isinstance(out["scratch_snapshot"], SharedFieldSnapshot)
    assert isinstance(out["final_snapshot"], SharedFieldSnapshot)
    assert isinstance(out["scratch_delta"], FieldDelta)
    assert isinstance(out["response_delta"], FieldDelta)
    assert out["scratch_snapshot"].region("scratch").text == record["targets"]["scratch"]
    assert out["final_snapshot"].region("response_draft").text == record["targets"]["response_draft"]
    assert out["compiled_tick1"].source_field_id == out["base_snapshot"].field_id
    assert out["compiled_tick2"].source_field_id == out["scratch_snapshot"].field_id


def test_training_scratch_intervention_is_a_canonical_snapshot() -> None:
    model = CompleteField64D(
        ReaderConfig(page_size=8, inference_budget_chars=128, dropout=0.0)
    )
    model.eval()
    state, memory, coverage = _read_record_with_scratch(
        model, _record(), "counterfactual scratch"
    )
    assert coverage.complete
    assert state.shape[-1] == 64
    assert memory.states.shape[-1] == 64


def test_canonical_greedy_path_survives_empty_noop_outputs() -> None:
    model = CompleteField64D(
        ReaderConfig(page_size=8, inference_budget_chars=65, dropout=0.0)
    )
    model.eval()
    model.decode_greedy = lambda *args, **kwargs: ("", True)  # type: ignore[method-assign]
    result = _run_record(model, _record())
    assert result["coverage_tick1"]["complete"] is True
    assert result["coverage_tick2"]["complete"] is True
    assert result["typed_delta"]["schema"] == "axon-canonical-d64-transaction-v1"

from __future__ import annotations

import torch

from runtime.field import FieldDelta, SharedFieldSnapshot
from training.complete_field_64d import (
    CompleteField64D,
    ReaderConfig,
    migrate_identity_region_embedding_state,
)
from training.train_complete_field_64d import (
    _forward_record,
    _read_record_with_scratch,
    _run_record,
    migrate_identity_region_optimizer_state,
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


def test_pre_identity_checkpoint_state_migrates_without_discarding_old_rows() -> None:
    source = CompleteField64D(ReaderConfig(ffn_dim=128, dropout=0.0))
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    source.region_embedding.weight.sum().backward()
    optimizer.step()

    legacy_model_state = dict(source.state_dict())
    legacy_model_state["region_embedding.weight"] = legacy_model_state[
        "region_embedding.weight"
    ][:-1].clone()
    legacy_optimizer_state = optimizer.state_dict()
    parameter_names = [name for name, _ in source.named_parameters()]
    region_index = parameter_names.index("region_embedding.weight")
    parameter_ids = [
        parameter_id
        for group in legacy_optimizer_state["param_groups"]
        for parameter_id in group["params"]
    ]
    region_parameter_id = parameter_ids[region_index]
    for name, value in tuple(legacy_optimizer_state["state"][region_parameter_id].items()):
        if isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(
            source.region_embedding.weight.shape
        ):
            legacy_optimizer_state["state"][region_parameter_id][name] = value[:-1].clone()

    destination = CompleteField64D(ReaderConfig(ffn_dim=128, dropout=0.0))
    destination_optimizer = torch.optim.AdamW(destination.parameters(), lr=1e-3)
    migrated_model, receipt = migrate_identity_region_embedding_state(
        destination,
        legacy_model_state,
    )
    assert receipt is not None
    assert receipt["old_rows_preserved_exactly"]
    assert torch.equal(
        migrated_model["region_embedding.weight"][:-1],
        legacy_model_state["region_embedding.weight"],
    )
    assert torch.count_nonzero(migrated_model["region_embedding.weight"][-1]) == 0
    migrated_optimizer, optimizer_receipt = migrate_identity_region_optimizer_state(
        destination,
        legacy_optimizer_state,
    )
    assert set(optimizer_receipt["extended_state_tensors"]) == {"exp_avg", "exp_avg_sq"}
    destination.load_state_dict(migrated_model, strict=True)
    destination_optimizer.load_state_dict(migrated_optimizer)
    destination_optimizer.zero_grad(set_to_none=True)
    destination.region_embedding.weight.sum().backward()
    destination_optimizer.step()

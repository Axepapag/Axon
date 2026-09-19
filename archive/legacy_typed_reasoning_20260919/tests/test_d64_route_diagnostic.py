from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.diagnose_d64_routes import (
    config_from_architecture,
    gate_bias_counterfactual,
    stop_at_exact_failure,
    summarize_positions,
    tensor_fingerprint,
)
from training.living_reasoning_d64 import LivingReasoningCoreConfig, LivingReasoningCoreD64


def test_config_from_report_architecture_round_trips_eos_route():
    config = LivingReasoningCoreConfig(
        n_layers=1, ffn_dim=64, receipt_continuation=True, dropout=0.0,
        eos_generate_head_route=True,
    )
    architecture = {
        field.name: getattr(config, field.name)
        for field in LivingReasoningCoreConfig.__dataclass_fields__.values()
        if field.init
    }
    # Report architecture dicts omit the opt-in flag and carry it only via
    # emission_routes, like the governed campaign reports do.
    architecture.pop("eos_generate_head_route")
    architecture.pop("generate_gate_bias")
    architecture.pop("field_schema_version")
    architecture["architecture_id"] = config.architecture_id
    architecture["emission_routes"] = [
        "learned_generate",
        "learned_copy_anchor",
        "deterministic_receipt_continuation",
        "generate_head_eos",
    ]
    rebuilt = config_from_architecture(architecture)
    assert rebuilt.architecture_id == config.architecture_id
    assert rebuilt.eos_generate_head_route is True


def test_bias_probe_restores_every_tensor_even_if_examination_raises():
    model = LivingReasoningCoreD64(LivingReasoningCoreConfig(
        n_layers=1, ffn_dim=64, receipt_continuation=True, dropout=0.0,
    ))
    with torch.no_grad():
        model.copy_gate.bias.fill_(-0.25)
    original = tensor_fingerprint(model)
    with pytest.raises(RuntimeError, match="probe interrupted"), gate_bias_counterfactual(model, True):
        assert float(model.copy_gate.bias.item()) == 0
        raise RuntimeError("probe interrupted")
    assert tensor_fingerprint(model) == original


def test_exact_failure_stops_at_irreversible_mismatch_and_keeps_state():
    class State:
        transport_categories = (0,)

        def to_canonical_dict(self):
            return {"transport_categories": list(self.transport_categories), "hidden": "preserved"}

    class Model:
        calls = 0

        def advance_decoder_execution(self, output, state, *, work_units):
            self.calls += 1
            return SimpleNamespace(state=State(), complete=False, malformed_reason=None)

    model = Model()
    episode = SimpleNamespace(targets=(SimpleNamespace(phase="first", payload="z"),))
    with stop_at_exact_failure(model, episode) as stops:
        result = model.advance_decoder_execution(SimpleNamespace(phase="first"), None, work_units=512)
    assert model.calls == 1
    assert not result.complete
    assert stops[0]["execution_state"]["hidden"] == "preserved"
    assert stops[0]["termination_after_mismatch"] == "not_measured"
    assert "advance_decoder_execution" not in model.__dict__


def test_diagnostic_never_counts_receipt_continuations_as_learned_routes():
    def row(role, logit, expected, generated, mixed, routed):
        return dict(position_role=role, gate_logit=logit, expected_category=expected,
                    generated_category=generated, mixed_category=mixed, routed_category=routed)

    summary = summarize_positions([
        row("copy_anchor", -0.2, 200, 352, 200, 200),
        row("continuation", 4.0, 220, 352, 352, 352),
        row("eos", -0.1, 352, 352, 352, 200),
    ])
    assert summary["copy_anchor"]["count"] == 1
    assert summary["eos"]["mixed_category_accuracy"] == 1.0
    assert summary["eos"]["routed_category_accuracy"] == 0.0
    assert summary["eos"]["generated_category_accuracy"] == 1.0
    assert summary["perfect_scalar_threshold_exists"] is True

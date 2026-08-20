from __future__ import annotations

import torch

from cores.core import AxonCore, CoreConfig
from runtime.field import SharedFieldSnapshot, compile_field_view
from training.differentiable_soul_writer import (
    DifferentiableSoulWriter,
    SoulWriterConfig,
    configure_writer_pilot_parameters,
    named_gradient_coverage,
    visible_view_text,
    writer_pilot_reader_parameter_names,
    write_delay_recall_forward,
    write_delay_recall_loss,
)


def _writer(d_model: int = 16) -> DifferentiableSoulWriter:
    torch.manual_seed(4)
    return DifferentiableSoulWriter(SoulWriterConfig(d_model=d_model))


def _tiny_core() -> AxonCore:
    torch.manual_seed(3)
    core = AxonCore(
        CoreConfig(
            d_model=16,
            n_heads=1,
            n_layers=1,
            ffn_dim=32,
            soul_rows=64,
            soul_hot_rows=64,
            soul_mode="act_reflect_v2",
            soul_gate_init=0.5,
            n_soul_compartments=4,
            char_slot_mode=True,
            char_slot_max_slots=384,
            char_n_regions=3,
        )
    )
    return core


def test_writer_only_changes_hot_rows_and_is_differentiable() -> None:
    writer = _writer()
    field = torch.randn(2, 12, 16)
    soul = torch.randn(2, 168, 16)
    active = torch.ones(2, 168, dtype=torch.bool)
    result = writer(field, soul, active)

    assert result.soul.shape == soul.shape
    assert result.strengths.shape == (2, 128)
    assert torch.allclose(result.strengths.sum(dim=1), torch.ones(2))
    assert torch.equal(result.soul[:, 128:], soul[:, 128:])
    assert result.hot_delta.abs().sum().item() > 0

    weights = torch.linspace(0.1, 1.0, 128).view(1, 128, 1)
    loss = (result.soul[:, :128] * weights).square().mean()
    loss.backward()
    coverage = named_gradient_coverage((("writer", writer),))
    assert coverage["coverage"] == 1.0
    assert coverage["missing_or_zero"] == ()
    assert coverage["nonfinite"] == ()


def test_field_mask_makes_padding_invariant() -> None:
    writer = _writer()
    soul = torch.zeros(1, 168, 16)
    active = torch.ones(1, 168, dtype=torch.bool)
    field = torch.randn(1, 8, 16)
    mask = torch.tensor([[True, True, True, False, False, False, False, False]])
    changed_padding = field.clone()
    changed_padding[:, 3:] = torch.randn_like(changed_padding[:, 3:]) * 1000

    first = writer(field, soul, active, field_mask=mask)
    second = writer(changed_padding, soul, active, field_mask=mask)
    assert torch.allclose(first.soul, second.soul)
    assert torch.allclose(first.strengths, second.strengths)


def test_all_masked_hot_tier_fails_closed() -> None:
    writer = _writer()
    field = torch.randn(1, 4, 16)
    soul = torch.zeros(1, 168, 16)
    active = torch.zeros(1, 168, dtype=torch.bool)
    try:
        writer(field, soul, active)
    except ValueError as exc:
        assert "no active hot" in str(exc)
    else:
        raise AssertionError("all-masked hot tier should fail")


def test_real_two_tick_loss_reaches_external_writer_and_read_path() -> None:
    core = _tiny_core()
    writer = _writer()
    policy = configure_writer_pilot_parameters(
        core,
        writer,
        train_read_path=True,
    )
    assert core.soul_readonly is True
    assert any(name.startswith("writer.") for name in policy.trainable_names)
    expected_reader_names = set(writer_pilot_reader_parameter_names(core))
    assert {
        name for name in policy.trainable_names if name.startswith("core.")
    } == expected_reader_names
    assert "core.soul_ingest.attn.qkv.weight" in expected_reader_names
    assert "core.layers.0.soul_cross_gate" in expected_reader_names
    assert all(
        not name.startswith("core.action_to_soul")
        for name in policy.trainable_names
    )

    tick_a = SharedFieldSnapshot.from_texts(
        {
            "conversation_history": "Remember the private code Z",
            "user_input": "Store it for later",
            "response_draft": "",
        }
    )
    tick_b = SharedFieldSnapshot.from_texts(
        {
            "conversation_history": "The earlier visible fact is now masked",
            "user_input": "What is the private code",
            "response_draft": "",
        },
        tick_id=1,
        parent_field_id=tick_a.field_id,
    )
    view_a = compile_field_view(tick_a)
    view_b = compile_field_view(tick_b)
    assert "Z" not in visible_view_text(view_b)

    soul = torch.zeros(1, 168, 16)
    soul_mask = torch.ones(1, 168, dtype=torch.bool)
    output = write_delay_recall_forward(
        core=core,
        writer=writer,
        tick_a_view=view_a,
        tick_b_view=view_b,
        soul_before=soul,
        soul_mask=soul_mask,
        fact_text="Z",
    )
    assert torch.equal(output.soul_after_write[:, 128:], soul[:, 128:])
    assert output.hot_delta.abs().sum().item() > 0
    loss = write_delay_recall_loss(output, "Z")
    assert torch.isfinite(loss)
    loss.backward()

    writer_coverage = named_gradient_coverage((("writer", writer),))
    assert writer_coverage["coverage"] == 1.0
    read_gates = [
        layer.soul_cross_gate.grad
        for layer in core.layers
        if hasattr(layer, "soul_cross_gate")
    ]
    assert all(
        gradient is not None and torch.isfinite(gradient)
        for gradient in read_gates
    )


def test_tick_b_fact_leak_and_wrong_layout_are_rejected() -> None:
    core = _tiny_core()
    writer = _writer()
    configure_writer_pilot_parameters(core, writer)
    tick_a = SharedFieldSnapshot.from_texts(
        {"user_input": "Remember Z", "response_draft": ""}
    )
    leaked_b = SharedFieldSnapshot.from_texts(
        {"user_input": "The answer is Z", "response_draft": ""}
    )

    try:
        write_delay_recall_forward(
            core=core,
            writer=writer,
            tick_a_view=compile_field_view(tick_a),
            tick_b_view=compile_field_view(leaked_b),
            soul_before=torch.zeros(1, 168, 16),
            soul_mask=torch.ones(1, 168, dtype=torch.bool),
            fact_text="Z",
        )
    except ValueError as exc:
        assert "leaks" in str(exc)
    else:
        raise AssertionError("fact leakage should fail")

    clean_b = SharedFieldSnapshot.from_texts(
        {"user_input": "What was the code", "response_draft": ""}
    )
    try:
        write_delay_recall_forward(
            core=core,
            writer=writer,
            tick_a_view=compile_field_view(tick_a),
            tick_b_view=compile_field_view(clean_b),
            soul_before=torch.zeros(1, 128, 16),
            soul_mask=torch.ones(1, 128, dtype=torch.bool),
            fact_text="Z",
        )
    except ValueError as exc:
        assert "authoritative layout" in str(exc)
    else:
        raise AssertionError("128-row state should fail the 168-row contract")

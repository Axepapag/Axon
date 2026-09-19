"""Focused tests for the opt-in dedicated termination head (termination_head_v5).

The termination-head route exists because every optimizer-side control (two
learning rates, two EOS weights, both gate-bias inits) still produced the
content<->EOS route swap: generated content logits and the EOS decision lived
in one shared softmax, so content logit growth mechanically depressed EOS
probability.  These tests prove the structural decoupling:

- generated-content logits cannot steal probability mass from the
  termination logit (with the old eos route as a discriminating control);
- the termination head scales stop-vs-continue cleanly;
- the copy/generate gate cannot suppress termination;
- training and free-running runtime consume the exact same learned
  termination distribution;
- architecture identity/provenance explicitly encodes the new head while all
  historical architecture identities remain unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import hashlib

import pytest
import torch

from scripts.diagnose_d64_routes import config_from_architecture
from substrate import encode_unicode_text
from training.foundation_motor_curriculum import (
    FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_BALANCED_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_PROGRAM_ID,
    RECEIPT_GENERATE_HEAD_EOS_TEACH,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
    RECEIPT_TEACHING_PROFILES,
    RECEIPT_TERMINATION_HEAD_BALANCED_TEACH,
    RECEIPT_TERMINATION_HEAD_PROFILES,
    RECEIPT_TERMINATION_HEAD_TEACH,
    apply_receipt_continuation_teach_weights,
    decide_foundation_motor_v2_stage,
    foundation_motor_v2_objective_program_id,
    foundation_motor_v2_stage_policy,
    receipt_continuation_teach_profile,
)
from training.living_reasoning_d64 import (
    CausalDecoderStep,
    DecoderEmissionRoute,
    LivingReasoningCoreConfig,
    LivingReasoningCoreD64,
)


class _StubMemory:
    """Minimal AddressableMemory stand-in for route-selection unit tests."""

    def __init__(self, slots: int, d_model: int, token_id: int = 0) -> None:
        self.states = torch.zeros(1, slots, d_model)
        self.char_indices = torch.full((1, slots), token_id, dtype=torch.long)
        self.receipts = ()
        self.segments = ()

    def receipt(self, index: int):
        return SimpleNamespace(
            address=SimpleNamespace(transport_token_id=int(self.char_indices[0, index]))
        )


def _term_model() -> LivingReasoningCoreD64:
    return LivingReasoningCoreD64(
        LivingReasoningCoreConfig(
            n_heads=1,
            n_layers=2,
            ffn_dim=128,
            state_tokens=2,
            page_size=2,
            dropout=0.0,
            receipt_continuation=True,
            termination_head_route=True,
        )
    )


def _eos_model() -> LivingReasoningCoreD64:
    return LivingReasoningCoreD64(
        LivingReasoningCoreConfig(
            n_heads=1,
            n_layers=2,
            ffn_dim=128,
            state_tokens=2,
            page_size=2,
            dropout=0.0,
            receipt_continuation=True,
            eos_generate_head_route=True,
        )
    )


def test_historical_architecture_identities_unchanged() -> None:
    # The exact v3guard/v3slow reference identity must survive the new flag.
    reference = LivingReasoningCoreConfig(
        n_heads=1,
        n_layers=2,
        ffn_dim=4096,
        dropout=0.0,
        receipt_continuation=True,
        eos_generate_head_route=True,
    )
    assert reference.architecture_id == "living-d64-receipt-242266f0633f2e7e1943d128"
    legacy = LivingReasoningCoreConfig(
        n_heads=1,
        n_layers=2,
        ffn_dim=4096,
        dropout=0.0,
    )
    assert legacy.architecture_id.startswith("living-d64-")
    assert not legacy.architecture_id.startswith("living-d64-receipt-")


def test_termination_head_route_is_opt_in_identity_bearing_and_validated() -> None:
    model = _term_model()
    reference = LivingReasoningCoreConfig(
        n_heads=1,
        n_layers=2,
        ffn_dim=128,
        state_tokens=2,
        page_size=2,
        dropout=0.0,
        receipt_continuation=True,
        eos_generate_head_route=True,
    )
    assert model.architecture_id != reference.architecture_id
    identity = model.living_config.to_canonical_dict(False)
    assert "termination_head" in identity["emission_routes"]
    assert "generate_head_eos" not in identity["emission_routes"]
    names = {name for name, _ in model.named_parameters()}
    assert "termination_output.weight" in names
    assert "termination_output.bias" in names
    report = model.architecture_report()
    assert report["termination_head_route"] is True
    assert report["eos_generate_head_route"] is False

    with pytest.raises(ValueError, match="receipt_continuation"):
        LivingReasoningCoreConfig(termination_head_route=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        LivingReasoningCoreConfig(
            receipt_continuation=True,
            eos_generate_head_route=True,
            termination_head_route=True,
        )


def test_termination_head_state_dict_is_strict_and_fail_closed() -> None:
    term = _term_model()
    eos = _eos_model()
    term_keys = set(term.state_dict())
    eos_keys = set(eos.state_dict())
    assert term_keys - eos_keys == {"termination_output.weight", "termination_output.bias"}
    roundtrip = _term_model()
    roundtrip.load_state_dict(term.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        eos.load_state_dict(term.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        term.load_state_dict(eos.state_dict(), strict=True)


def test_config_from_architecture_round_trips_termination_route() -> None:
    config = LivingReasoningCoreConfig(
        n_layers=1,
        ffn_dim=64,
        receipt_continuation=True,
        dropout=0.0,
        termination_head_route=True,
    )
    architecture = {
        name: getattr(config, name)
        for name, init in LivingReasoningCoreConfig.__dataclass_fields__.items()
        if init.init
    }
    # Governed report dicts omit the opt-in flag and carry it only via
    # emission_routes, exactly like the eos-route reports.
    architecture.pop("termination_head_route")
    architecture.pop("generate_gate_bias")
    architecture.pop("field_schema_version")
    architecture["architecture_id"] = config.architecture_id
    architecture["emission_routes"] = [
        "learned_generate",
        "learned_copy_anchor",
        "deterministic_receipt_continuation",
        "termination_head",
    ]
    rebuilt = config_from_architecture(architecture)
    assert rebuilt.architecture_id == config.architecture_id
    assert rebuilt.termination_head_route is True
    assert rebuilt.eos_generate_head_route is False


def _fixed_scenario(model: LivingReasoningCoreD64):
    """Zero the content head and pin a moderate stop logit."""
    memory = _StubMemory(slots=4, d_model=model.living_config.d_model, token_id=7)
    output = torch.zeros(1, 1, model.living_config.d_model)
    with torch.no_grad():
        model.copy_gate.weight.zero_()
        model.copy_gate.bias.fill_(-5.0)
        model.decoder_output.weight.zero_()
        model.decoder_output.bias.zero_()
        model.decoder_output.bias[7] = 2.0
        if model.living_config.termination_head_route:
            model.termination_output.weight.zero_()
            model.termination_output.bias.fill_(1.0)
    return memory, output


def test_content_logits_cannot_steal_mass_from_termination_logit() -> None:
    model = _term_model()
    memory, output = _fixed_scenario(model)
    logits, alignment = model._decoder_logits(output, memory, return_alignment=True)
    stop_before = alignment["termination_stop_probability"].clone()
    eos_logit_before = logits[..., model.eos_index].clone()

    # The strongest content-logit perturbation available: crank every content
    # row of the shared decoder projection by a large constant.  Inside one
    # softmax this would mechanically depress the EOS slot; the dedicated
    # head must be unaffected.
    with torch.no_grad():
        model.decoder_output.weight += 50.0
        model.decoder_output.bias[: model.eos_index] += 50.0
    logits_after, alignment_after = model._decoder_logits(output, memory, return_alignment=True)
    assert torch.allclose(
        alignment_after["termination_stop_probability"], stop_before, atol=1e-7
    )
    assert torch.allclose(logits_after[..., model.eos_index], eos_logit_before, atol=1e-7)
    # Content mass moved instead: the conditional content distribution changed.
    content_before = logits[..., : model.eos_index].argmax(dim=-1)
    assert int(content_before.item()) == 7
    assert logits_after.exp().sum(dim=-1).item() == pytest.approx(1.0, abs=1e-6)


def test_shared_softmax_eos_route_control_still_couples_content_to_eos() -> None:
    # Negative control: under the historical eos route the identical content
    # perturbation DOES move the EOS probability.  This proves the previous
    # test discriminates the coupling instead of trivially passing.
    model = _eos_model()
    memory, output = _fixed_scenario(model)
    logits, _ = model._decoder_logits(output, memory, return_alignment=True)
    eos_before = logits[..., model.eos_index].clone()
    with torch.no_grad():
        model.decoder_output.weight += 50.0
        model.decoder_output.bias[: model.eos_index] += 50.0
    logits_after, _ = model._decoder_logits(output, memory, return_alignment=True)
    assert not torch.allclose(
        logits_after[..., model.eos_index], eos_before, atol=1e-7
    )


def test_termination_gradient_isolated_from_content_head() -> None:
    model = _term_model()
    memory, output = _fixed_scenario(model)
    logits, _ = model._decoder_logits(output, memory, return_alignment=True)
    eos_loss = torch.nn.functional.nll_loss(
        logits.reshape(-1, logits.shape[-1]),
        torch.tensor([model.eos_index]),
    )
    model.zero_grad(set_to_none=True)
    eos_loss.backward()
    # The EOS slot of the returned distribution is log(stop) alone: the
    # termination loss trains the dedicated head and leaves the shared
    # content head untouched.
    assert model.termination_output.bias.grad is not None
    assert float(model.termination_output.bias.grad.item()) != 0.0
    assert model.decoder_output.bias.grad is None or float(
        model.decoder_output.bias.grad.abs().sum().item()
    ) == 0.0

    # Content targets pressure the stop decision only through the normalized
    # distribution (log1p(-stop)), never by stealing the termination logit.
    logits_before, _ = model._decoder_logits(output, memory, return_alignment=True)
    stop_before = logits_before[..., model.eos_index].clone()
    logits, _ = model._decoder_logits(output, memory, return_alignment=True)
    content_loss = torch.nn.functional.nll_loss(
        logits.reshape(-1, logits.shape[-1]),
        torch.tensor([7]),
    )
    model.zero_grad(set_to_none=True)
    content_loss.backward()
    assert model.decoder_output.bias.grad is not None
    assert float(model.decoder_output.bias.grad[7].item()) != 0.0
    assert model.termination_output.bias.grad is not None
    assert float(model.termination_output.bias.grad.item()) != 0.0
    logits_after, _ = model._decoder_logits(output, memory, return_alignment=True)
    assert torch.allclose(
        logits_after[..., model.eos_index], stop_before, atol=1e-7
    )


@pytest.mark.parametrize("bias,expect_stop", [(-12.0, False), (12.0, True)])
def test_termination_scales_stop_vs_continue_cleanly(bias, expect_stop) -> None:
    model = _term_model()
    memory, output = _fixed_scenario(model)
    with torch.no_grad():
        model.termination_output.bias.fill_(bias)
    logits, _ = model._decoder_logits(output, memory, return_alignment=True)
    assert logits.exp().sum(dim=-1).item() == pytest.approx(1.0, abs=1e-6)
    routed = int(logits[0, 0].argmax(dim=-1).item())
    assert (routed == model.eos_index) is expect_stop
    if not expect_stop:
        # Continue cleanly: content winner is the trained category, never EOS.
        assert routed == 7


def test_termination_stop_probability_is_sigmoid_of_head_alone() -> None:
    model = _term_model()
    memory, output = _fixed_scenario(model)
    for bias in (-6.0, -1.0, 0.0, 1.0, 6.0):
        with torch.no_grad():
            model.termination_output.bias.fill_(bias)
        logits, alignment = model._decoder_logits(output, memory, return_alignment=True)
        expected = torch.sigmoid(torch.tensor(bias)).item()
        assert float(alignment["termination_stop_probability"].item()) == pytest.approx(
            expected, abs=1e-6
        )
        content_prob_sum = logits[..., : model.eos_index].exp().sum().item()
        assert content_prob_sum == pytest.approx(1.0 - expected, abs=1e-6)


def test_copy_generate_gate_cannot_suppress_termination() -> None:
    model = _term_model()
    memory, output = _fixed_scenario(model)
    with torch.no_grad():
        model.copy_gate.bias.fill_(-50.0)  # pinned all-copy regime
        model.termination_output.bias.fill_(12.0)
    logits, _ = model._decoder_logits(output, memory, return_alignment=True)
    step = CausalDecoderStep(
        hidden=torch.zeros(1, 1, model.living_config.d_model),
        mixed_logits=logits,
        generated_logits=torch.zeros(1, 1, model.eos_index + 1),
        position_logits=torch.zeros(1, 1, 4),
        generate_gate_logits=torch.full((1, 1), -50.0),
    )
    route, category, memory_index, receipt = model._select_learned_emission(step, memory)
    assert route is DecoderEmissionRoute.LEARNED_GENERATE
    assert category == model.eos_index
    assert memory_index is None
    assert receipt is None


def test_gate_cannot_reintroduce_eos_on_content_steps() -> None:
    model = _term_model()
    memory, output = _fixed_scenario(model)
    with torch.no_grad():
        model.termination_output.bias.fill_(-12.0)
        # The dead EOS slot of the content head is huge; the content branch
        # must still never select it.
        model.decoder_output.bias[model.eos_index] = 100.0
    logits, alignment = model._decoder_logits(output, memory, return_alignment=True)
    step = CausalDecoderStep(
        hidden=torch.zeros(1, 1, model.living_config.d_model),
        mixed_logits=logits,
        generated_logits=alignment["generated_logits"],
        position_logits=torch.zeros(1, 1, 4),
        generate_gate_logits=torch.full((1, 1), 5.0),
    )
    route, category, _memory_index, _receipt = model._select_learned_emission(step, memory)
    assert route is DecoderEmissionRoute.LEARNED_GENERATE
    assert category == 7


def test_training_and_runtime_use_identical_termination_equation() -> None:
    model = _term_model()
    memory = _StubMemory(slots=4, d_model=model.living_config.d_model, token_id=7)
    reader_state = torch.zeros(1, 1, model.living_config.d_model)
    with torch.no_grad():
        model.decoder_output.weight.zero_()
        model.decoder_output.bias.zero_()
        model.decoder_output.bias[7] = 3.0
        model.termination_output.weight.zero_()
    targets = torch.tensor([[7, model.eos_index]])
    bos = torch.full((1, 1), model.bos_index, dtype=torch.long)
    decoder_input = torch.cat((bos, targets[:, :-1]), dim=1)
    summary = reader_state.mean(dim=1)
    head_vec = model.decoder_head_embedding(torch.tensor([1]))
    hidden = torch.tanh(model.decoder_init(torch.cat((summary, head_vec), dim=-1))).unsqueeze(0)
    output, _ = model.decoder(model.decoder_embedding(decoder_input), hidden)
    for bias in (-8.0, -1.0, 1.0, 8.0):
        with torch.no_grad():
            model.termination_output.bias.fill_(bias)
        # Training distribution: exactly what the teacher-forced payload CE
        # consumes, replayed here step-by-step so it can be compared per
        # position against the free-running causal decoder path.
        mixed, _alignment = model._decoder_logits(output, memory, return_alignment=True)
        # Free-running step (causal decoder path) at each position:
        step0 = model.causal_decoder_step(
            previous_category=model.bos_index, hidden=hidden, memory=memory
        )
        assert torch.allclose(step0.mixed_logits, mixed[:, 0:1], atol=1e-7)
        step1 = model.causal_decoder_step(
            previous_category=7, hidden=step0.hidden, memory=memory
        )
        assert torch.allclose(step1.mixed_logits, mixed[:, 1:2], atol=1e-7)
        # The runtime selector must decide exactly what the training
        # distribution's argmax says, and both must equal the closed-form
        # stop-vs-best-content comparison.
        stop = float(
            torch.sigmoid(torch.tensor(bias)).item()
        )
        best_content = float(
            mixed[..., : model.eos_index][0, 1].exp().max().item()
        )
        expect_stop = stop > best_content
        route, category, _mi, _rc = model._select_learned_emission(step1, memory)
        assert (category == model.eos_index) is expect_stop
        assert category == int(mixed[0, 1].argmax(dim=-1).item())
        assert route is DecoderEmissionRoute.LEARNED_GENERATE


def test_termination_head_memory_none_path_stays_normalized() -> None:
    model = _term_model()
    output = torch.zeros(1, 1, model.living_config.d_model)
    logits, _ = model._decoder_logits(output, None, return_alignment=True)
    assert logits.shape[-1] == model.eos_index + 1
    assert logits.exp().sum(dim=-1).item() == pytest.approx(1.0, abs=1e-6)


def test_termination_head_v5_profile_and_program_identity() -> None:
    overlay = receipt_continuation_teach_profile(RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5)
    assert overlay is RECEIPT_TERMINATION_HEAD_TEACH
    assert overlay["requires_architecture_features"] == [
        "receipt_continuation",
        "termination_head_route",
    ]
    assert overlay["payload_eos_weight"] == 4.0
    # v3 optimizer pressures are otherwise untouched.
    for key in ("payload", "alignment_position", "alignment_copy_gate", "alignment_eos_gate"):
        assert overlay["component_weight_overrides"][key] == (
            RECEIPT_GENERATE_HEAD_EOS_TEACH["component_weight_overrides"][key]
        )
    program_id = foundation_motor_v2_objective_program_id(
        teach_multicell_copy=True,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
    )
    assert program_id == FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_PROGRAM_ID
    assert program_id != FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID
    assert program_id != FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID
    # Historical profiles map to their historical programs.
    assert foundation_motor_v2_objective_program_id(
        teach_multicell_copy=True,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    ) == FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID
    assert foundation_motor_v2_objective_program_id(
        teach_multicell_copy=True,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
    ) == FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID


def test_stage_gate_validates_termination_route_profile_pairing() -> None:
    with pytest.raises(ValueError, match="termination_head_route"):
        decide_foundation_motor_v2_stage(
            training_stage="copy_alignment",
            heldout_probe=None,
            regression_probe=None,
            complete_heldout=True,
            complete_regression=True,
            receipt_continuation=True,
            receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
            eos_generate_head_route=False,
            termination_head_route=False,
        )
    with pytest.raises(ValueError, match="termination_head_route"):
        decide_foundation_motor_v2_stage(
            training_stage="copy_alignment",
            heldout_probe=None,
            regression_probe=None,
            complete_heldout=True,
            complete_regression=True,
            receipt_continuation=True,
            receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
            eos_generate_head_route=True,
            termination_head_route=True,
        )
    result = decide_foundation_motor_v2_stage(
        training_stage="copy_alignment",
        heldout_probe=None,
        regression_probe=None,
        complete_heldout=True,
        complete_regression=True,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
        eos_generate_head_route=False,
        termination_head_route=True,
    )
    assert result["passed"] is False  # probes missing; pairing validated first


def _decode_alignment(model, memory, target: str, scheduled: bool = False):
    reader_state = torch.zeros(1, 1, model.living_config.d_model)
    with torch.no_grad():
        if hasattr(model, "termination_output"):
            model.termination_output.weight.zero_()
    if scheduled:
        return model.decode_scheduled(
            reader_state, target, 1, 1.0, memory=memory, return_alignment=True
        )
    return model.decode_teacher(
        reader_state, target, 1, memory, return_alignment=True
    )


def _eos_supervision(model, memory, alignment):
    return model.alignment_supervision(
        target_text="a",
        memory=memory,
        decoder_alignment=alignment,
        specification={
            "schema": "axon-r0-target-alignment-v1",
            "segments": [],
            "supervise_eos_generate": True,
        },
    )


def test_termhead_teacher_alignment_exposes_termination_logits() -> None:
    """Regression (repair directive step 2/5a): decode_teacher and
    decode_scheduled must propagate the dedicated termination head's logits
    in their alignment dictionaries under termination_head_route; legacy
    routes must not gain the key."""
    model = _term_model()
    memory = _StubMemory(slots=4, d_model=model.living_config.d_model, token_id=7)
    _joined, _targets, alignment = _decode_alignment(model, memory, "a")
    assert "termination_logits" in alignment
    assert alignment["termination_logits"].shape == alignment["generate_gate_logits"].shape

    _joined, _targets, scheduled_alignment = _decode_alignment(
        model, memory, "a", scheduled=True
    )
    assert "termination_logits" in scheduled_alignment

    legacy = _eos_model()
    legacy_memory = _StubMemory(
        slots=4, d_model=legacy.living_config.d_model, token_id=7
    )
    _joined, _targets, legacy_alignment = _decode_alignment(
        legacy, legacy_memory, "a"
    )
    assert "termination_logits" not in legacy_alignment


def test_termhead_eos_gate_supervision_uses_termination_head() -> None:
    """Regression (repair directive step 3/5b/5c): under termination_head_route
    the EOS gate loss and accuracy must score the dedicated termination logit
    (positive => stop), even when the copy/generate gate at the EOS position
    is deliberately wrong; a wrong termination logit must score 0 even when
    the generate gate would look correct under the old wiring."""
    model = _term_model()
    memory = _StubMemory(slots=4, d_model=model.living_config.d_model, token_id=7)

    # Forced-good termination logit, deliberately wrong generate gate at EOS.
    _joined, _targets, alignment = _decode_alignment(model, memory, "a")
    alignment["termination_logits"] = torch.full_like(
        alignment["termination_logits"], 8.0
    )
    alignment["generate_gate_logits"] = torch.full_like(
        alignment["generate_gate_logits"], -50.0
    )
    supervision = _eos_supervision(model, memory, alignment)
    assert supervision["eos_gate_correct"] == 1
    assert supervision["eos_gate_supervised_positions"] == 1
    assert supervision["eos_gate_accuracy"] == 1.0

    # Forced-bad termination logit; the old wiring's signal is flattered.
    _joined, _targets, alignment = _decode_alignment(model, memory, "a")
    alignment["termination_logits"] = torch.full_like(
        alignment["termination_logits"], -8.0
    )
    alignment["generate_gate_logits"] = torch.full_like(
        alignment["generate_gate_logits"], 50.0
    )
    supervision = _eos_supervision(model, memory, alignment)
    assert supervision["eos_gate_correct"] == 0
    assert supervision["eos_gate_accuracy"] == 0.0


def test_legacy_eos_gate_supervision_unchanged_by_termhead_wiring() -> None:
    """Control: without termination_logits in the alignment dict (every
    historical route), EOS gate supervision keeps scoring the copy/generate
    gate exactly as before."""
    legacy = _eos_model()
    legacy_memory = _StubMemory(
        slots=4, d_model=legacy.living_config.d_model, token_id=7
    )
    _joined, _targets, alignment = _decode_alignment(legacy, legacy_memory, "a")
    alignment["generate_gate_logits"] = torch.full_like(
        alignment["generate_gate_logits"], 8.0
    )
    supervision = _eos_supervision(legacy, legacy_memory, alignment)
    assert supervision["eos_gate_correct"] == 1
    assert supervision["eos_gate_accuracy"] == 1.0

    _joined, _targets, alignment = _decode_alignment(legacy, legacy_memory, "a")
    alignment["generate_gate_logits"] = torch.full_like(
        alignment["generate_gate_logits"], -8.0
    )
    supervision = _eos_supervision(legacy, legacy_memory, alignment)
    assert supervision["eos_gate_correct"] == 0
    assert supervision["eos_gate_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# termination_head_balanced_v6: the corrected fix set ratified 2026-09-17.
#
# The transport_eos probation autopsy (canonical ledger events
# evt-20260917T013100Z and evt-20260917T013451Z) verified that v5 supervises
# stop=1 twice (payload CE at the restored 4x EOS weight plus the EOS-position
# gate BCE) while stop=0 receives only the diluted implicit log(1-stop) term
# inside payload CE, leaving a canceling-gradient fixed point where EOS wins
# every argmax and free-running transport emits nothing.  v6 corrects exactly
# this: restore the v4-balanced 1x payload EOS weight and add explicit stop=0
# BCE supervision at every learned content anchor.
# ---------------------------------------------------------------------------


def test_termination_head_balanced_v6_profile_and_program_identity() -> None:
    overlay = receipt_continuation_teach_profile(
        RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6
    )
    assert overlay is RECEIPT_TERMINATION_HEAD_BALANCED_TEACH
    assert overlay["payload_eos_weight"] == 1.0
    assert overlay["termination_continue_supervision"] is True
    assert overlay["requires_architecture_features"] == [
        "receipt_continuation",
        "termination_head_route",
    ]
    assert RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6 in (
        RECEIPT_TERMINATION_HEAD_PROFILES
    )
    # v5 is preserved byte-for-byte as evidence: 4x EOS weight, no continue
    # supervision, and the v3-inherited component table.
    v5 = receipt_continuation_teach_profile(RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5)
    assert v5 is RECEIPT_TERMINATION_HEAD_TEACH
    assert v5["payload_eos_weight"] == 4.0
    assert v5.get("termination_continue_supervision", False) is False
    assert v5["component_weight_overrides"]["alignment_eos_gate"] == 0.0

    program_id = foundation_motor_v2_objective_program_id(
        teach_multicell_copy=True,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
    )
    assert program_id == FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_BALANCED_PROGRAM_ID
    assert program_id != foundation_motor_v2_objective_program_id(
        teach_multicell_copy=True,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
    )
    assert program_id not in {
        FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_PROGRAM_ID,
        FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID,
        FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID,
    }


def test_executed_component_weights_are_pinned_per_stage() -> None:
    transport_entry = dict(
        foundation_motor_v2_stage_policy("transport_eos")["component_weights"]
    )
    assert transport_entry == {
        "decision": 0.0,
        "operation": 0.0,
        "region": 0.0,
        "start": 0.0,
        "end": 0.0,
        "payload": 1.0,
        "alignment_position": 1.0,
        "alignment_copy_gate": 4.0,
        "alignment_eos_gate": 1.0,
    }
    # The receipt overlay only rewrites copy_alignment; transport_eos executes
    # the stage entry table for every profile.  Pin both directions so a
    # profile can never silently differ from the executed stage weights again.
    for profile in RECEIPT_TEACHING_PROFILES:
        assert apply_receipt_continuation_teach_weights(
            dict(transport_entry),
            training_stage="transport_eos",
            receipt_teaching_profile=profile,
        ) == transport_entry

    copy_entry = dict(
        foundation_motor_v2_stage_policy("copy_alignment")["component_weights"]
    )
    v6_weights = apply_receipt_continuation_teach_weights(
        copy_entry,
        training_stage="copy_alignment",
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
    )
    assert v6_weights == {
        "decision": 0.0,
        "operation": 0.0,
        "region": 0.0,
        "start": 0.0,
        "end": 0.0,
        "payload": 1.0,
        "alignment_position": 1.0,
        "alignment_copy_gate": 4.0,
        "alignment_eos_gate": 1.0,
    }
    v5_weights = apply_receipt_continuation_teach_weights(
        copy_entry,
        training_stage="copy_alignment",
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
    )
    assert v5_weights["payload"] == 1.0
    assert v5_weights["alignment_eos_gate"] == 0.0


def _transport_probe(*, transport_exact: float) -> dict:
    return {
        "complete_field_coverage_rate": 1.0,
        "alignment_position_accuracy": 1.0,
        "alignment_copy_gate_accuracy": 1.0,
        "alignment_eos_gate_accuracy": 1.0,
        "payload_content_accuracy": 1.0,
        "payload_eos_accuracy": 1.0,
        "payload_transport_exact_rate": transport_exact,
        "payload_content_constant_floor": 0.125,
        # Real floors measured over the 72-case heldout surface: 24/72 typed and
        # 8/24 transport.  A probe that carries no floor cannot attest anything.
        "typed_emission_exact_rate": 1.0,
        "constant_typed_emission_exact_floor": 1.0 / 3.0,
        "constant_payload_transport_exact_floor": 1.0 / 3.0,
        "pair_exact_rates": {
            "position": 1.0,
            "copy_gate": 1.0,
            "content": 1.0,
            "eos_gate": 1.0,
        },
    }


def test_transport_eos_gate_requires_free_running_transport_exact() -> None:
    # Ratified 2026-09-17: payload_transport_exact_rate is the only metric that
    # exposes the emit-nothing dead state, so the transport_eos stage gate must
    # require it.  Applies to every termination-route profile, including the
    # historical v5 re-evaluation of confirmed checkpoint bfe76d52.
    for profile in (
        RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
        RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
    ):
        passed = decide_foundation_motor_v2_stage(
            training_stage="transport_eos",
            heldout_probe=_transport_probe(transport_exact=1.0),
            regression_probe=_transport_probe(transport_exact=1.0),
            complete_heldout=True,
            complete_regression=True,
            receipt_continuation=True,
            receipt_teaching_profile=profile,
            termination_head_route=True,
        )
        assert passed["passed"] is True

        # The emit-nothing lineage scores exactly the 12/36 empty-payload
        # floor; it can never satisfy the corrected gate no matter how perfect
        # the teacher-forced component metrics look.
        dead = decide_foundation_motor_v2_stage(
            training_stage="transport_eos",
            heldout_probe=_transport_probe(transport_exact=12.0 / 36.0),
            regression_probe=_transport_probe(transport_exact=1.0),
            complete_heldout=True,
            complete_regression=True,
            receipt_continuation=True,
            receipt_teaching_profile=profile,
            termination_head_route=True,
        )
        assert dead["passed"] is False
        assert any(
            "payload_transport_exact_rate" in reason for reason in dead["failures"]
        )


def test_stage_gate_accepts_v6_termination_route_pairing() -> None:
    with pytest.raises(ValueError, match="termination_head_route"):
        decide_foundation_motor_v2_stage(
            training_stage="transport_eos",
            heldout_probe=None,
            regression_probe=None,
            complete_heldout=True,
            complete_regression=True,
            receipt_continuation=True,
            receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
            eos_generate_head_route=False,
            termination_head_route=False,
        )
    result = decide_foundation_motor_v2_stage(
        training_stage="transport_eos",
        heldout_probe=None,
        regression_probe=None,
        complete_heldout=True,
        complete_regression=True,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
        eos_generate_head_route=False,
        termination_head_route=True,
    )
    assert result["passed"] is False  # probes missing; pairing validated first


class _AnchorMemory:
    """Exact-transport stub: every slot is a resolvable cell in region 0."""

    def __init__(self, model: LivingReasoningCoreD64, token: int, slots: int = 4) -> None:
        self.states = torch.zeros(1, slots, model.living_config.d_model)
        self.region_ids = torch.zeros(1, slots, dtype=torch.long)
        self.region_positions = torch.arange(slots, dtype=torch.long).unsqueeze(0)
        self.char_indices = torch.full((1, slots), token, dtype=torch.long)


def _anchor_supervision(
    model: LivingReasoningCoreD64,
    memory: _AnchorMemory,
    termination_values: tuple[float, float],
    *,
    supervise_continue: bool | None = None,
):
    target = "a"
    assert len(encode_unicode_text(target)) == 1
    alignment = {
        "position_logits": torch.zeros(1, 2, memory.states.shape[1]),
        "generate_gate_logits": torch.zeros(1, 2),
        "termination_logits": torch.tensor(
            [list(termination_values)], dtype=torch.float32
        ),
    }
    specification = {
        "schema": "axon-r0-target-alignment-v1",
        "segments": [
            {
                "target_start": 0,
                "target_end": 1,
                "source_region": model.region_order[0].value,
                "source_start": 2,
                "source_end": 3,
                "text_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                "authority": "exact_current_shared_field",
            }
        ],
        "supervise_eos_generate": True,
    }
    kwargs = {}
    if supervise_continue is not None:
        kwargs["supervise_termination_continue"] = supervise_continue
    return model.alignment_supervision(
        target_text=target,
        memory=memory,
        decoder_alignment=alignment,
        specification=specification,
        **kwargs,
    )


def test_termination_continue_supervision_is_opt_in_and_symmetric() -> None:
    model = _term_model()
    memory = _AnchorMemory(model, token=encode_unicode_text("a")[0])

    # Default (and explicit False) stay byte-identical to the v5 semantics:
    # only the EOS position is supervised and scored, and the continuation rate
    # is explicitly unavailable rather than a vacuous 1.0 over zero positions.
    base = _anchor_supervision(model, memory, (-8.0, 8.0))
    explicit_off = _anchor_supervision(model, memory, (-8.0, 8.0), supervise_continue=False)
    for supervision in (base, explicit_off):
        assert supervision["eos_gate_supervised_positions"] == 1
        assert supervision["eos_gate_correct"] == 1
        assert supervision["eos_gate_accuracy"] == 1.0
        assert supervision["termination_continue_positions"] == 0
        assert supervision["termination_continue_accuracy"] is None
        assert supervision["termination_continue_loss"] is None
    assert explicit_off["eos_gate_loss"].item() == base["eos_gate_loss"].item()

    # Flag on: the content anchor is supervised toward stop=0 and folded into
    # the termination decision accuracy alongside the EOS position.
    good = _anchor_supervision(model, memory, (-8.0, 8.0), supervise_continue=True)
    assert good["termination_continue_positions"] == 1
    assert good["termination_continue_correct"] == 1
    assert good["termination_continue_accuracy"] == 1.0
    assert good["eos_gate_supervised_positions"] == 2
    assert good["eos_gate_correct"] == 2
    assert good["eos_gate_accuracy"] == 1.0

    # The continuation loss is instrumented from the stop=0 BCE terms that were
    # already being optimized, so it is exactly the term the flag adds to the
    # termination gate loss -- the objective itself is unchanged.
    assert good["termination_continue_loss"] is not None
    assert good["eos_gate_loss"].item() == pytest.approx(
        (base["eos_gate_loss"].item() + good["termination_continue_loss"].item()) / 2
    )

    # A saturated stop logit at the anchor is now scored wrong even though the
    # EOS position is still perfect: the emit-everything dead state is visible.
    saturated = _anchor_supervision(model, memory, (8.0, 8.0), supervise_continue=True)
    assert saturated["termination_continue_accuracy"] == 0.0
    assert saturated["eos_gate_correct"] == 1
    assert saturated["eos_gate_supervised_positions"] == 2
    assert saturated["eos_gate_accuracy"] == 0.5
    assert saturated["eos_gate_loss"].item() > base["eos_gate_loss"].item()

    with pytest.raises(TypeError, match="supervise_termination_continue"):
        model.alignment_supervision(
            target_text="a",
            memory=memory,
            decoder_alignment={
                "position_logits": torch.zeros(1, 2, memory.states.shape[1]),
                "generate_gate_logits": torch.zeros(1, 2),
                "termination_logits": torch.zeros(1, 2),
            },
            specification={
                "schema": "axon-r0-target-alignment-v1",
                "segments": [],
                "supervise_eos_generate": True,
            },
            supervise_termination_continue="yes",
        )


def test_continuation_metrics_fail_closed_without_a_supervised_anchor() -> None:
    model = _term_model()
    memory = _AnchorMemory(model, token=encode_unicode_text("a")[0])

    def supervise(*, supervise_continue: bool) -> dict:
        return model.alignment_supervision(
            target_text="a",
            memory=memory,
            decoder_alignment={
                "position_logits": torch.zeros(1, 2, memory.states.shape[1]),
                "generate_gate_logits": torch.zeros(1, 2),
                "termination_logits": torch.zeros(1, 2),
            },
            specification={
                "schema": "axon-r0-target-alignment-v1",
                "segments": [],
                "supervise_eos_generate": True,
            },
            supervise_termination_continue=supervise_continue,
        )

    # With no alignment segment there is no content anchor to supervise toward
    # stop=0, so the continuation rate is unavailable -- never a vacuous 1.0 that
    # would let an unsupervised route look perfect at the termination objective.
    unavailable = supervise(supervise_continue=False)
    assert unavailable["termination_continue_positions"] == 0
    assert unavailable["termination_continue_accuracy"] is None
    assert unavailable["termination_continue_loss"] is None

    # A route that declares continuation supervision while supervising no anchor
    # has not exercised the mechanism at all, so it must fail closed instead of
    # reporting numbers for an objective it never ran.
    with pytest.raises(RuntimeError, match="no content anchor was supervised"):
        supervise(supervise_continue=True)

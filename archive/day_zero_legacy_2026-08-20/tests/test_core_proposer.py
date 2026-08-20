from __future__ import annotations

import torch

from cores.core import AxonCore, CoreConfig
from runtime.core_proposer import AxonCoreRegionProposer
from runtime.field import LogicalRegion, SharedFieldSnapshot, compile_field_view
from runtime.multi_tick_refiner import run_multi_tick_refinement
from substrate import default_alphabet


def _tiny_core() -> AxonCore:
    core = AxonCore(
        CoreConfig(
            d_model=16,
            n_heads=1,
            n_layers=1,
            ffn_dim=32,
            soul_rows=0,
            soul_hot_rows=0,
            soul_mode="act_reflect_v2",
            char_slot_mode=True,
            char_slot_max_slots=384,
            char_n_regions=3,
        )
    )
    # A zeroed real core has tied logits and deterministically emits alphabet
    # index zero at every proposal position.
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.zero_()
    return core


def test_real_core_proposes_over_the_canonical_384_view() -> None:
    core = _tiny_core()
    core.train()
    snapshot = SharedFieldSnapshot.from_texts(
        {
            "conversation_history": "Earlier context",
            "user_input": "Please answer",
            "structured_knowledge": "dog is a animal",
            "situation_awareness": "No tool is running",
            "response_draft": "",
        }
    )
    view = compile_field_view(snapshot)
    proposer = AxonCoreRegionProposer(core, max_output_chars=8)
    proposal = proposer(
        snapshot=snapshot,
        view=view,
        tick_index=0,
        target_region=LogicalRegion.RESPONSE_DRAFT,
    )

    assert proposal.text == default_alphabet()[0] * 8
    assert proposal.target_region is LogicalRegion.RESPONSE_DRAFT
    assert f"axon_core:{core.cfg.d_model}d" in proposal.provenance
    assert core.training is True


def test_real_core_output_is_committed_and_seen_on_the_next_tick() -> None:
    core = _tiny_core()
    initial = SharedFieldSnapshot.from_texts(
        {
            "user_input": "Answer",
            "scratch": "Use the visible evidence",
            "response_draft": "",
        }
    )
    proposer = AxonCoreRegionProposer(core, max_output_chars=6)
    result = run_multi_tick_refinement(
        initial_snapshot=initial,
        proposer=proposer,
        target_schedule=[
            LogicalRegion.RESPONSE_DRAFT,
            LogicalRegion.RESPONSE_DRAFT,
        ],
        author_core_id="tiny-real-core",
        max_ticks=2,
        min_ticks=2,
    )

    expected = default_alphabet()[0] * 6
    assert result.final_snapshot.region("response_draft").text == expected
    assert result.records[0].prior_text == ""
    assert result.records[1].prior_text == expected
    assert result.records[1].no_op is True
    assert len(result.deltas) == 1
    assert result.replay().field_id == result.final_snapshot.field_id

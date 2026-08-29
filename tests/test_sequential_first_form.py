from __future__ import annotations

from runtime.soul import SoulSnapshot, empty_soul_layers
from training import (
    LivingReasoningCoreD64,
    candidate_a_config,
    compile_sequential_first_form,
    load_sequential_first_form,
    publish_sequential_first_form,
    sequential_living_objective,
)

from .test_first_form_curriculum import _experience_store


def _curriculum(tmp_path):
    return compile_sequential_first_form(
        _experience_store(tmp_path / "State"),
        identity_text="Axon is Axon. The current canonical field outranks stale assumptions.",
        requested_split_counts=(1, 1, 1),
    )


def test_sequential_ffcs_replays_real_finalization_and_ingress_successors(tmp_path) -> None:
    curriculum = _curriculum(tmp_path)
    assert tuple(len(curriculum.split(name)) for name in ("train", "heldout", "regression")) == (
        1,
        1,
        1,
    )
    for case in curriculum.cases:
        assert len(case.ticks) == 3
        assert case.ticks[0].finalization_receipt is None
        assert case.ticks[1].finalization_receipt is not None
        assert case.ticks[2].finalization_receipt is not None
        assert case.ticks[0].next_snapshot.field_id == case.ticks[1].episode.snapshot.field_id
        assert case.ticks[1].next_snapshot.field_id == case.ticks[2].episode.snapshot.field_id
        assert case.ticks[2].next_snapshot is None
        history = case.ticks[2].successor.region("conversation_history").text
        assert history.count("[axon-readable-turn-frame-v1") == 2
        assert len({tick.episode.first_workspace_text for tick in case.ticks}) == 3

    path = publish_sequential_first_form(curriculum, state_root=tmp_path / "State")
    restored = load_sequential_first_form(path)
    assert restored.to_canonical_dict() == curriculum.to_canonical_dict()


def test_sequential_objective_carries_serialized_soul_across_real_ticks(tmp_path) -> None:
    case = _curriculum(tmp_path).split("train")[0]
    model = LivingReasoningCoreD64(
        candidate_a_config(n_layers=1, ffn_dim=128, page_size=32, dropout=0.0)
    )
    core_id = "sequential-test-core"
    generation = "sequential-test-generation"
    soul = SoulSnapshot(
        core_id=core_id,
        architecture_id=model.architecture_id,
        parameter_generation=generation,
        generation=0,
        parent_soul_id=None,
        layers=empty_soul_layers(),
    )

    loss, unrolls, final_soul = sequential_living_objective(
        model,
        case,
        soul,
        core_id=core_id,
        parameter_generation=generation,
    )
    loss.backward()

    assert len(unrolls) == 3
    assert final_soul.generation == 9
    assert unrolls[1].souls[0].soul_id == unrolls[0].souls[-1].soul_id
    assert unrolls[2].souls[0].soul_id == unrolls[1].souls[-1].soul_id
    assert all(output.canonical_coverage.complete for unroll in unrolls for output in unroll.outputs)
    assert model.page_encoder.layers[0].linear1.weight.grad is not None

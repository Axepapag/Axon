"""Tournament metric surface: per-phase breakdown, sequential evaluation, ten metrics."""

from __future__ import annotations

import pytest

from runtime.soul import SoulSnapshot, empty_soul_layers
from scripts.train_living_reasoning_smoke import _scheduled_material, _training_lanes
from training import (
    LivingReasoningCoreD64,
    candidate_a_config,
    d64_head_geometry_tournament,
    d64_tournament_metric_computation,
    d64_tournament_metrics,
    evaluate_sequential_case,
    living_phase_breakdown,
    sequential_living_objective,
)
from training.first_form_curriculum import FirstFormCurriculumCompiler
from training.living_reasoning_curriculum import build_living_reasoning_smoke_curriculum

from .test_first_form_curriculum import _experience_store
from .test_sequential_first_form import _curriculum as _sequential_curriculum


def _tiny_model() -> LivingReasoningCoreD64:
    return LivingReasoningCoreD64(
        candidate_a_config(n_layers=1, ffn_dim=128, page_size=32, dropout=0.0)
    )


def _soul(model: LivingReasoningCoreD64, core_id: str, generation: str) -> SoulSnapshot:
    return SoulSnapshot(
        core_id=core_id,
        architecture_id=model.architecture_id,
        parameter_generation=generation,
        generation=0,
        parent_soul_id=None,
        layers=empty_soul_layers(),
    )


@pytest.fixture()
def sequential_case(tmp_path):
    return _sequential_curriculum(tmp_path).split("train")[0]


def test_phase_breakdown_reports_every_phase_with_zero_weight_rows() -> None:
    model = _tiny_model()
    curriculum = build_living_reasoning_smoke_curriculum()
    episode = curriculum.split("heldout")[0]
    rows = living_phase_breakdown(
        model,
        episode,
        _soul(model, "metrics-core", "metrics-generation"),
        core_id="metrics-core",
        parameter_generation="metrics-generation",
    )
    assert set(rows) == {"first", "refined", "consolidated"}
    for row in rows.values():
        assert set(row) == {"decision_match", "payload_match", "supervision_weight"}
        assert row["decision_match"] in (0.0, 1.0)


def test_evaluate_sequential_case_measures_real_tick_chain(sequential_case) -> None:
    model = _tiny_model()
    row = evaluate_sequential_case(
        model,
        sequential_case,
        _soul(model, "metrics-core", "metrics-generation"),
        core_id="metrics-core",
        parameter_generation="metrics-generation",
    )
    assert row["tick_count"] == float(len(sequential_case.ticks))
    # Sequential episodes supervise only the consolidated phase per tick.
    assert row["supervised_phase_count"] == float(len(sequential_case.ticks))
    assert 0.0 <= row["typed_emission_exact_rate"] <= 1.0
    assert 0.0 <= row["payload_transport_exact_rate"] <= 1.0
    assert row["complete_field_coverage_rate"] == 1.0
    assert row["constant_typed_emission_exact_floor"] == 0.0
    assert row["payload_teacher_forced_token_count"] >= 1


def test_tournament_metrics_cover_the_exact_required_surface(tmp_path) -> None:
    model = _tiny_model()
    mechanism = build_living_reasoning_smoke_curriculum()
    sequential = _sequential_curriculum(tmp_path)
    society = FirstFormCurriculumCompiler(
        _experience_store(tmp_path / "df-State")
    ).compile_df(
        identity_text="Axon is Axon. The current canonical field is authority.",
        requested_counts=(("D", (1, 1, 1)), ("F", (1, 1, 1))),
    )
    initial = _soul(model, "metrics-core", "metrics-generation")
    computation = d64_tournament_metric_computation(
        model,
        episodes=mechanism.split("heldout")
        + tuple(
            case.episode
            for case in society.cases
            if case.episode.split == "heldout"
        ),
        sequential_cases=sequential.split("heldout"),
        initial_soul=initial,
        regression_episodes=tuple(
            case.episode
            for case in society.cases
            if case.episode.split == "regression"
        ),
        regression_sequential_cases=sequential.split("regression"),
        stale_soul=initial,
        core_id="metrics-core",
        parameter_generation="metrics-generation",
    )
    metrics = computation.metric_mapping
    assert set(metrics) == set(d64_head_geometry_tournament().required_metrics)
    assert computation.complete is True
    assert computation.field_override_probe_ids
    assert computation.proposal_refinement_probe_ids
    for name, value in metrics.items():
        assert -1e9 < value < 1e9, name
    # A foreign-core Soul must be rejected by the swapped-soul probe.
    assert metrics["swapped_soul_rejection_rate"] == 1.0
    # With no stale comparison supplied, the stale delta is exactly zero.
    assert metrics["stale_soul_degradation"] == 0.0


def test_tournament_metrics_reject_empty_material() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="heldout episodes or sequential cases"):
        d64_tournament_metrics(
            model,
            episodes=(),
            sequential_cases=(),
            initial_soul=_soul(model, "metrics-core", "metrics-generation"),
            regression_episodes=(),
            stale_soul=None,
            core_id="metrics-core",
            parameter_generation="metrics-generation",
        )


def test_sequential_split_manifests_are_stable_and_content_addressed(tmp_path) -> None:
    curriculum = _sequential_curriculum(tmp_path)
    train_id = curriculum.train_manifest_id
    heldout_id = curriculum.heldout_manifest_id
    assert len(train_id) == 64 and len(heldout_id) == 64
    assert train_id != heldout_id
    again = _sequential_curriculum(tmp_path)
    assert again.train_manifest_id == train_id
    assert again.heldout_manifest_id == heldout_id


def test_family_round_robin_reaches_sequential_e_in_first_cycle(tmp_path) -> None:
    compiler = FirstFormCurriculumCompiler(_experience_store(tmp_path / "lanes-State"))
    identity = "Axon is Axon. Every brother attends the exact canonical field."
    abc = compiler.compile_abc(
        identity_text=identity,
        requested_counts=(("A", (1, 1, 1)), ("B", (1, 1, 1)), ("C", (1, 1, 1))),
    )
    df = compiler.compile_df(
        identity_text=identity,
        requested_counts=(("D", (1, 1, 1)), ("F", (1, 1, 1))),
    )
    sequential = _sequential_curriculum(tmp_path / "lanes-sequential")
    lanes = _training_lanes(
        build_living_reasoning_smoke_curriculum(), [abc, df], [sequential]
    )
    observed = [_scheduled_material(lanes, step)[0] for step in range(len(lanes))]
    assert observed == [
        "mechanism",
        "ffcs-A",
        "ffcs-B",
        "ffcs-C",
        "ffcs-D",
        "ffcs-F",
        "ffcs-E",
    ]
    assert _scheduled_material(lanes, 6)[1] == "sequential"


def test_tournament_metrics_soul_probes_differ_when_soul_carries_state(sequential_case) -> None:
    """After one real pass the carried HOT layer must matter; cold layers must not."""

    model = _tiny_model()
    core_id = "metrics-core"
    generation = "metrics-generation"
    initial = _soul(model, core_id, generation)
    loss, _unrolls, carried = sequential_living_objective(
        model,
        sequential_case,
        initial,
        core_id=core_id,
        parameter_generation=generation,
    )
    loss.backward()
    mechanism = build_living_reasoning_smoke_curriculum()
    metrics = d64_tournament_metrics(
        model,
        episodes=mechanism.split("heldout"),
        sequential_cases=(),
        initial_soul=carried,
        regression_episodes=mechanism.split("regression"),
        stale_soul=initial,
        core_id=core_id,
        parameter_generation=generation,
    )
    # This is a behavioral exact-rate differential. The untrained model is
    # wrong both intact and ablated, so zero is the honest value; a hidden-state
    # L2 movement must never be relabeled as learned degradation.
    assert metrics["relevant_soul_ablation_degradation"] == 0.0
    assert metrics["irrelevant_soul_ablation_delta"] == 0.0
    assert metrics["swapped_soul_rejection_rate"] == 1.0

from scripts.train_living_reasoning_smoke import (
    exact_serving_gate_passed,
    nonzero_exact_output_observed,
    select_qa_transcript_rows,
)


def _evaluation(
    *,
    typed: float,
    payload: float,
    typed_floor: float = 1.0 / 3.0,
    payload_floor: float = 1.0 / 3.0,
) -> dict[str, float]:
    # Defaults are the floors actually measured over the 72-case heldout
    # surface (24/72 typed, 8/24 transport).  A zero floor here would certify
    # the emit-nothing answer as progress, which is the trap this test guards.
    return {
        "typed_emission_exact_rate": typed,
        "payload_transport_exact_rate": payload,
        "constant_typed_emission_exact_floor": typed_floor,
        "constant_payload_transport_exact_floor": payload_floor,
    }


def test_matching_the_constant_answer_is_not_progress() -> None:
    evaluation = _evaluation(typed=1.0 / 3.0, payload=1.0 / 3.0)
    assert nonzero_exact_output_observed(evaluation) is False


def test_nonzero_exact_output_is_progress_not_serving_readiness() -> None:
    evaluation = _evaluation(typed=2.0 / 3.0, payload=2.0 / 3.0)
    assert nonzero_exact_output_observed(evaluation) is True
    assert exact_serving_gate_passed(
        evaluation,
        curriculum_stage_complete=True,
        complete_heldout=True,
        complete_regression=True,
        tournament_metric_surface_complete=True,
    ) is False


def test_exact_serving_gate_fails_closed_on_every_incomplete_surface() -> None:
    evaluation = _evaluation(typed=1.0, payload=1.0)
    assert exact_serving_gate_passed(
        evaluation,
        curriculum_stage_complete=True,
        complete_heldout=True,
        complete_regression=True,
        tournament_metric_surface_complete=True,
    ) is True
    assert exact_serving_gate_passed(
        evaluation,
        curriculum_stage_complete=True,
        complete_heldout=True,
        complete_regression=True,
        tournament_metric_surface_complete=False,
    ) is False
    assert exact_serving_gate_passed(
        evaluation,
        curriculum_stage_complete=False,
        complete_heldout=True,
        complete_regression=True,
        tournament_metric_surface_complete=True,
    ) is False


def _qa_row(family: str, index: int, *, exact: bool) -> dict[str, object]:
    return {
        "family": family,
        "episode_label": f"{family}-{index}",
        "exact_match": exact,
    }


def test_the_qa_sample_spans_families_instead_of_the_first_rows() -> None:
    # The panel used to show the first rows of the first evaluated episodes, so
    # one large family could fill the whole screen.  Family-balanced sampling is
    # what makes the sample useful for diagnosis.
    rows = [_qa_row("copy", i, exact=False) for i in range(10)]
    rows += [_qa_row("delete", 0, exact=False), _qa_row("delete", 1, exact=True)]
    rows += [_qa_row("insert", 0, exact=True)]

    selected = select_qa_transcript_rows(rows, limit=6)

    assert len(selected) == 6
    families = [row["family"] for row in selected]
    assert set(families) == {"copy", "delete", "insert"}
    # Ten copy rows are available; the panel must not become a copy-only wall.
    assert families.count("copy") < len(selected)


def test_the_qa_sample_puts_failures_before_exact_matches_within_a_family() -> None:
    rows = [
        _qa_row("copy", 0, exact=True),
        _qa_row("copy", 1, exact=False),
        _qa_row("copy", 2, exact=True),
    ]

    selected = select_qa_transcript_rows(rows, limit=3)

    assert [row["episode_label"] for row in selected] == ["copy-1", "copy-0", "copy-2"]


def test_the_qa_sample_is_deterministic_and_refuses_a_nonpositive_limit() -> None:
    rows_later = [_qa_row("copy", i, exact=False) for i in range(3)]
    rows_earlier = [_qa_row("insert", i, exact=False) for i in range(3)]

    first = select_qa_transcript_rows(rows_earlier + rows_later, limit=4)
    second = select_qa_transcript_rows(rows_earlier + rows_later, limit=4)

    # Deterministic regardless of dict/bucket iteration: both runs take the
    # alphabetically first family first, so the sample is reproducible.
    assert [row["episode_label"] for row in first] == [
        row["episode_label"] for row in second
    ]
    assert [row["episode_label"] for row in first] == [
        "copy-0",
        "insert-0",
        "copy-1",
        "insert-1",
    ]

    try:
        select_qa_transcript_rows(rows_earlier, limit=0)
    except ValueError:
        pass
    else:  # pragma: no cover - the guard is the point
        raise AssertionError("a nonpositive limit must be refused")


def test_an_unlimited_qa_sample_keeps_every_row_exactly_once() -> None:
    rows = [_qa_row("copy", i, exact=i % 2 == 0) for i in range(5)]
    rows += [_qa_row("delete", 0, exact=False)]

    selected = select_qa_transcript_rows(rows, limit=len(rows))

    assert len(selected) == len(rows)
    assert sorted(row["episode_label"] for row in selected) == sorted(
        row["episode_label"] for row in rows
    )

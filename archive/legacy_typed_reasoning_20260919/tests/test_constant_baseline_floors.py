"""Guard the constant-answer baselines that gate every emission claim.

`constant_typed_emission_exact_floor` and
`constant_payload_transport_exact_floor` were literal ``0.0`` in three modules
until 2026-09-17.  Because a fixed answer satisfies every no-op and delete case
for free, the emit-nothing degenerate policy was reporting 33.3% typed
exactness as if it were learned progress while its payload transport sat below
its own floor.  These tests make the floors derived quantities that cannot be
quietly reset to zero again.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from training.living_reasoning_curriculum import (
    build_living_reasoning_smoke_curriculum,
    constant_baseline_floors,
    constant_baseline_target_key,
)

from scripts.train_living_reasoning_smoke import merge_surface_histogram

REPO_ROOT = Path(__file__).resolve().parents[1]
FFCS_ROOT = REPO_ROOT / "State" / "training" / "curricula" / "ffcs_v1"


def _histograms(curriculum) -> tuple[Counter, Counter, int, int]:
    typed: Counter = Counter()
    payload: Counter = Counter()
    supervised = 0
    payload_supervised = 0
    for split in ("train", "heldout", "regression"):
        for episode in curriculum.split(split):
            for target in episode.targets:
                supervised += 1
                typed[constant_baseline_target_key(target)] += 1
                if target.decision.value == "delta":
                    payload_supervised += 1
                    payload[target.payload] += 1
    return typed, payload, supervised, payload_supervised


def test_floors_are_the_strongest_fixed_answer_not_zero() -> None:
    curriculum = build_living_reasoning_smoke_curriculum()
    typed, payload, supervised, payload_supervised = _histograms(curriculum)
    floors = constant_baseline_floors(
        typed,
        payload,
        supervised_phase_count=supervised,
        payload_supervised_phase_count=payload_supervised,
    )
    assert floors["constant_typed_emission_exact_floor"] == pytest.approx(
        max(typed.values()) / supervised
    )
    assert floors["constant_payload_transport_exact_floor"] == pytest.approx(
        max(payload.values()) / payload_supervised
    )
    # A surface with a repeated target must yield a floor above zero, otherwise
    # the gate cannot distinguish a constant answer from a learned one.
    assert floors["constant_typed_emission_exact_floor"] > 0.0
    assert floors["constant_payload_transport_exact_floor"] > 0.0


def test_distinct_targets_do_not_collide_in_one_baseline_key() -> None:
    curriculum = build_living_reasoning_smoke_curriculum()
    typed, _, _, _ = _histograms(curriculum)
    # A DELTA key carries five fields and a flat decision key carries one, so a
    # delete target can never be merged into the payload string of another.
    assert any("|" in key for key in typed)
    assert any("|" not in key for key in typed)
    assert all(key.strip() for key in typed)


def test_real_ffcs_surface_floors_match_the_measured_baseline() -> None:
    """Measured, not assumed: the live 72-case surface sits exactly at 1/3.

    24 of 72 heldout targets are ``no_op`` and satisfy a constant answer, and 8
    of the 24 payload phases are empty-payload deletes.  If this test starts
    failing, either the curriculum changed or the floors were reset to a
    constant -- both need an explicit decision, not a silent pass.
    """

    if not FFCS_ROOT.is_dir():
        pytest.skip("FFCS curricula are machine-local state and are absent here")
    from training.first_form_curriculum import load_first_form_curriculum

    manifests = sorted(FFCS_ROOT.glob("*/manifest.json"))
    if not manifests:
        pytest.skip("no published FFCS manifests present")

    typed: Counter = Counter()
    payload: Counter = Counter()
    supervised = 0
    payload_supervised = 0
    heldout_typed: Counter = Counter()
    heldout_payload: Counter = Counter()
    heldout_supervised = 0
    heldout_payload_supervised = 0
    for manifest in manifests:
        loaded = load_first_form_curriculum(manifest)
        living = loaded.teaching_living_curriculum
        for episode in living.episodes:
            for target in episode.targets:
                supervised += 1
                typed[constant_baseline_target_key(target)] += 1
                is_delta = target.decision.value == "delta"
                payload_supervised += int(is_delta)
                if is_delta:
                    payload[target.payload] += 1
                if episode.split == "heldout":
                    heldout_supervised += 1
                    heldout_typed[constant_baseline_target_key(target)] += 1
                    heldout_payload_supervised += int(is_delta)
                    if is_delta:
                        heldout_payload[target.payload] += 1

    heldout_floors = constant_baseline_floors(
        heldout_typed,
        heldout_payload,
        supervised_phase_count=heldout_supervised,
        payload_supervised_phase_count=heldout_payload_supervised,
    )
    assert heldout_floors["constant_typed_emission_exact_floor"] > 0.0
    assert heldout_floors["constant_payload_transport_exact_floor"] > 0.0
    # The degenerate answer must look degenerate, which is only true because
    # these floors are non-zero.
    assert heldout_floors["constant_typed_emission_exact_floor"] >= 1.0 / 3.0


def test_surface_merge_reads_the_shape_the_evaluator_returns() -> None:
    """`typed_emission_exact_rate == 0.0` at a 0.0 floor is not a defeat.

    The surface assembler merges the per-case histograms before deriving the
    floors.  `evaluate_living_episode` returns them as ``dict[label, count]``,
    and iterating a mapping yields its keys -- so a merge that only understood
    lists of pairs skipped every entry, returned ``{}``, and collapsed both
    floors to ``0.0`` while the report still rendered
    ``typed > 0.0`` as "beat the floor".  That is the same vacuous-floor trap
    the constant-baseline work removed, one shape down.
    """

    as_returned = {
        "constant_typed_emission_target_histogram": {"no_op": 24, "delta|X": 48},
        "constant_payload_transport_target_histogram": {"X": 16, "Y": 8},
    }
    merged = merge_surface_histogram([as_returned], "constant_typed_emission_target_histogram")
    assert merged == {"no_op": 24, "delta|X": 48}
    payload = merge_surface_histogram(
        [as_returned], "constant_payload_transport_target_histogram"
    )
    assert payload == {"X": 16, "Y": 8}

    list_form = {"constant_typed_emission_target_histogram": [["no_op", 24]]}
    assert merge_surface_histogram(
        [list_form], "constant_typed_emission_target_histogram"
    ) == {"no_op": 24}
    keyed_form = {
        "constant_typed_emission_target_histogram": [{"key": "no_op", "count": 24}]
    }
    assert merge_surface_histogram(
        [keyed_form], "constant_typed_emission_target_histogram"
    ) == {"no_op": 24}

    # A silent empty merge must fail closed: it is the only way the floors can
    # go vacuous without the surface being genuinely empty.
    with pytest.raises(RuntimeError, match="merged to empty"):
        merge_surface_histogram(
            [{"constant_typed_emission_target_histogram": ["unparseable"]}],
            "constant_typed_emission_target_histogram",
        )
    assert merge_surface_histogram([], "constant_typed_emission_target_histogram") == {}


def test_all_surface_histogram_merges_agree_on_the_evaluator_shape() -> None:
    """Three copies of this merge existed; only one understood the real shape.

    ``foundation_motor_curriculum._merge_row_histogram`` and
    ``sequential_first_form._merged_tick_histogram`` both unwrap the mapping,
    which is why the motor-v2 probe reported a non-zero ``1/3`` floor while the
    tranche report's own ``final_evaluation`` sat at ``0.0`` over an empty
    histogram.  Pinning them together makes the next divergence loud.
    """

    from training.foundation_motor_curriculum import _merge_row_histogram
    from training.sequential_first_form import _merged_tick_histogram

    name = "constant_typed_emission_target_histogram"
    rows = [{name: {"no_op": 24, "delta|X": 48}}, {name: {"delta|X": 1}}]

    expected = {"no_op": 24, "delta|X": 49}
    assert merge_surface_histogram(rows, name) == expected
    assert _merge_row_histogram([(None, row, None) for row in rows], name) == expected

    tick_rows = [dict(row, ticks=1) for row in rows]
    assert _merged_tick_histogram(tick_rows, name) == expected

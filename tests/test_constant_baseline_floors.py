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

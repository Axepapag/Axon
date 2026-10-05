from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from runtime.field import (
    CANONICAL_REGION_ORDER,
    PRE_IDENTITY_REGION_ORDER,
    TRAINING_REGIONS,
    AttendedInterval,
    FieldSpan,
    LogicalRegion,
    RegionMaskPolicy,
    RegionState,
    SharedFieldSnapshot,
    materialize_d16_view,
    resolve_mask_policy,
)
from runtime.heart import (
    LEGACY_HEART_REGION_MASK_SCHEMA,
    HeartRegionMaskController,
    MaskStateCorruptionError,
)
from runtime.heart.masks import IDENTITY_HEART_REGION_MASK_SCHEMA


@pytest.mark.parametrize(
    ("percent", "expected"),
    ((0, ""), (1, "j"), (50, "fghij"), (100, "abcdefghij")),
)
def test_tail_percent_resolves_to_exact_newest_suffix(
    percent: int,
    expected: str,
) -> None:
    spans = (FieldSpan(span_id="history", text="abcdefghij"),)
    intervals = resolve_mask_policy(spans, RegionMaskPolicy("tail_percent", percent))
    text = spans[0].text
    assert "".join(text[item.start : item.end] for item in intervals) == expected


def test_tail_percent_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        RegionMaskPolicy("tail_percent", 101)


def test_durable_controller_owns_every_region_independently(tmp_path: Path) -> None:
    path = tmp_path / "region_masks.json"
    controller = HeartRegionMaskController(path)
    assert tuple(controller.state.policies) == CANONICAL_REGION_ORDER
    assert all(
        controller.state.policy_for(region) == RegionMaskPolicy("all")
        for region in CANONICAL_REGION_ORDER
        if region not in TRAINING_REGIONS
    )
    # Training regions default to no attendance for non-cohort readers; the
    # Trainer authority opts a training core into them explicitly.
    assert all(
        controller.state.policy_for(region) == RegionMaskPolicy("none")
        for region in TRAINING_REGIONS
    )

    first_id = controller.state.state_id
    updated = controller.set_unmasked_percent(LogicalRegion.CONVERSATION_HISTORY, 25)
    assert updated.revision == 1
    assert updated.state_id != first_id
    assert updated.policy_for(LogicalRegion.CONVERSATION_HISTORY) == RegionMaskPolicy("tail_percent", 25)
    assert updated.policy_for(LogicalRegion.DIARY) == RegionMaskPolicy("all")

    cohort = controller.set_policy(LogicalRegion.TRAINER_INSTRUCTIONS, RegionMaskPolicy("all"))
    assert cohort.policy_for(LogicalRegion.TRAINER_INSTRUCTIONS) == RegionMaskPolicy("all")
    assert cohort.policy_for(LogicalRegion.TRAINING_RESPONSES) == RegionMaskPolicy("none")

    reopened = HeartRegionMaskController(path)
    assert reopened.state.state_id == cohort.state_id
    assert reopened.state.revision == 2
    assert reopened.state.policy_for(LogicalRegion.CONVERSATION_HISTORY).limit == 25
    assert reopened.state.policy_for(LogicalRegion.TRAINER_INSTRUCTIONS) == RegionMaskPolicy("all")


def test_identity_is_always_attended_and_cannot_be_masked(tmp_path: Path) -> None:
    controller = HeartRegionMaskController(tmp_path / "region_masks.json")
    assert controller.state.policy_for(LogicalRegion.IDENTITY) == RegionMaskPolicy("all")
    with pytest.raises(ValueError, match="identity"):
        controller.set_unmasked_percent(LogicalRegion.IDENTITY, 99)


def test_legacy_mask_state_migrates_by_adding_always_attended_identity(tmp_path: Path) -> None:
    path = tmp_path / "region_masks.json"
    controller = HeartRegionMaskController(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema"] = LEGACY_HEART_REGION_MASK_SCHEMA
    keep = {region.value for region in PRE_IDENTITY_REGION_ORDER}
    value["policies"] = [
        item for item in value["policies"] if item["region"] in keep
    ]
    value_without_id = {key: item for key, item in value.items() if key != "state_id"}
    from runtime.field import canonical_sha256

    value["state_id"] = canonical_sha256(value_without_id)
    path.write_text(json.dumps(value), encoding="utf-8")

    migrated = HeartRegionMaskController(path).state
    assert migrated.revision == controller.state.revision + 1
    assert migrated.policy_for(LogicalRegion.IDENTITY) == RegionMaskPolicy("all")
    assert all(
        migrated.policy_for(region) == RegionMaskPolicy("none") for region in TRAINING_REGIONS
    )
    assert tuple(migrated.policies) == CANONICAL_REGION_ORDER


def test_identity_mask_schema_state_migrates_by_adding_unattended_training_regions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "region_masks.json"
    controller = HeartRegionMaskController(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema"] = IDENTITY_HEART_REGION_MASK_SCHEMA
    value["policies"] = [
        item
        for item in value["policies"]
        if item["region"] not in {region.value for region in TRAINING_REGIONS}
    ]
    value_without_id = {key: item for key, item in value.items() if key != "state_id"}
    from runtime.field import canonical_sha256

    value["state_id"] = canonical_sha256(value_without_id)
    path.write_text(json.dumps(value), encoding="utf-8")

    migrated = HeartRegionMaskController(path).state
    assert migrated.revision == controller.state.revision + 1
    assert migrated.policy_for(LogicalRegion.IDENTITY) == RegionMaskPolicy("all")
    assert all(
        migrated.policy_for(region) == RegionMaskPolicy("none") for region in TRAINING_REGIONS
    )
    assert tuple(migrated.policies) == CANONICAL_REGION_ORDER
    # The migrated state persists under the current mask schema and reloads
    # without a second migration.
    reopened = HeartRegionMaskController(path)
    assert reopened.state.state_id == migrated.state_id
    assert reopened.state.revision == migrated.revision


def test_durable_controller_fails_closed_on_tampering(tmp_path: Path) -> None:
    path = tmp_path / "region_masks.json"
    HeartRegionMaskController(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["policies"][0]["policy"]["limit"] = 99
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(MaskStateCorruptionError, match="identity mismatch"):
        HeartRegionMaskController(path)


def test_d16_view_serves_only_attended_cells_with_exact_addresses() -> None:
    state = RegionState(
        name=LogicalRegion.CONVERSATION_HISTORY,
        spans=(
            FieldSpan(span_id="turn-1", text="abcde", source="user", provenance="event:1"),
            FieldSpan(span_id="turn-2", text="fghij", source="axon", provenance="event:2"),
        ),
        attended_intervals=(AttendedInterval(1, 3), AttendedInterval(4, 7)),
    )
    view = materialize_d16_view(SharedFieldSnapshot(tick_id=1, regions=(state,)))
    region = view.region(LogicalRegion.CONVERSATION_HISTORY)
    assert region.text == "bcefg"
    assert region.cells16.shape == (5, 16)
    assert [
        (a.attended_interval_index, a.span_id, a.provenance, a.character) for a in region.addresses
    ] == [
        (0, "turn-1", "event:1", "b"),
        (0, "turn-1", "event:1", "c"),
        (1, "turn-1", "event:1", "e"),
        (1, "turn-2", "event:2", "f"),
        (1, "turn-2", "event:2", "g"),
    ]


def test_mask_movement_preserves_field_identity_addresses_and_cells() -> None:
    history = RegionState.from_text(
        LogicalRegion.CONVERSATION_HISTORY,
        "abcdefghij",
        span_id="first-to-latest",
        provenance="lived-history",
    )
    user = RegionState.from_text(LogicalRegion.USER_INPUT, "question")
    field = SharedFieldSnapshot(tick_id=9, regions=(history, user))

    def view(percent: int):
        return materialize_d16_view(
            field,
            region_masks={LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("tail_percent", percent)},
        )

    none, half, full = view(0), view(50), view(100)
    assert none.source_field_id == half.source_field_id == full.source_field_id == field.field_id
    assert len({none.view_id, half.view_id, full.view_id}) == 3
    assert none.region(LogicalRegion.CONVERSATION_HISTORY).text == ""
    assert half.region(LogicalRegion.CONVERSATION_HISTORY).text == "fghij"
    assert full.region(LogicalRegion.CONVERSATION_HISTORY).text == "abcdefghij"
    assert all(v.region(LogicalRegion.USER_INPUT).text == "question" for v in (none, half, full))

    full_region = full.region(LogicalRegion.CONVERSATION_HISTORY)
    full_by_position = {a.region_position: (a, full_region.cells16[i]) for i, a in enumerate(full_region.addresses)}
    half_region = half.region(LogicalRegion.CONVERSATION_HISTORY)
    for i, address in enumerate(half_region.addresses):
        original, original_cell = full_by_position[address.region_position]
        assert address.global_position == original.global_position
        assert address.span_id == original.span_id
        assert address.span_position == original.span_position
        assert np.array_equal(half_region.cells16[i], original_cell)

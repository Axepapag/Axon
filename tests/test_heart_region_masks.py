from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from runtime.field import (
    CANONICAL_REGION_ORDER,
    AttendedInterval,
    D64FieldCompiler,
    FieldSpan,
    LogicalRegion,
    RegionMaskPolicy,
    RegionState,
    SharedFieldSnapshot,
    resolve_mask_policy,
)
from runtime.heart import (
    BeatConfig,
    CoreDescriptor,
    CoreRegistry,
    HeartHost,
    HeartHostConfig,
    HeartRegionMaskController,
    HostBeatState,
    MaskStateCorruptionError,
)


def _host(
    state_root: Path,
    *,
    cores: CoreRegistry | None = None,
) -> HeartHost:
    return HeartHost(
        state_root=state_root,
        host_config=HeartHostConfig(idle_interval_seconds=60.0),
        beat_config=BeatConfig(recall_limit=0),
        core_registry=cores,
    )


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
    assert all(controller.state.policy_for(region) == RegionMaskPolicy("all") for region in CANONICAL_REGION_ORDER)

    first_id = controller.state.state_id
    updated = controller.set_unmasked_percent(LogicalRegion.CONVERSATION_HISTORY, 25)
    assert updated.revision == 1
    assert updated.state_id != first_id
    assert updated.policy_for(LogicalRegion.CONVERSATION_HISTORY) == RegionMaskPolicy("tail_percent", 25)
    assert updated.policy_for(LogicalRegion.DIARY) == RegionMaskPolicy("all")

    reopened = HeartRegionMaskController(path)
    assert reopened.state.state_id == updated.state_id
    assert reopened.state.revision == 1
    assert reopened.state.policy_for(LogicalRegion.CONVERSATION_HISTORY).limit == 25


def test_durable_controller_fails_closed_on_tampering(tmp_path: Path) -> None:
    path = tmp_path / "region_masks.json"
    HeartRegionMaskController(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["policies"][0]["policy"]["limit"] = 99
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(MaskStateCorruptionError, match="identity mismatch"):
        HeartRegionMaskController(path)


def test_compiler_never_packs_across_mask_or_provenance_boundaries() -> None:
    state = RegionState(
        name=LogicalRegion.CONVERSATION_HISTORY,
        spans=(
            FieldSpan(
                span_id="turn-1",
                text="abcde",
                source="user",
                provenance="event:1",
            ),
            FieldSpan(
                span_id="turn-2",
                text="fghij",
                source="axon",
                provenance="event:2",
            ),
        ),
        attended_intervals=(AttendedInterval(1, 3), AttendedInterval(4, 7)),
    )
    compiled = D64FieldCompiler().compile(SharedFieldSnapshot(tick_id=1, regions=(state,)))
    assert compiled.region_text(LogicalRegion.CONVERSATION_HISTORY) == "bcefg"

    nonempty_rows = []
    for row_index in range(compiled.row_count):
        row_addresses = tuple(
            address for lane in range(4) if (address := compiled.address(row_index, lane)) is not None
        )
        if row_addresses:
            nonempty_rows.append(row_addresses)
            assert [address.lane_index for address in row_addresses] == list(range(len(row_addresses)))
            assert len({address.attended_interval_index for address in row_addresses}) == 1
            assert len({address.span_id for address in row_addresses}) == 1
            assert len({address.provenance for address in row_addresses}) == 1

    assert ["".join(item.character for item in row) for row in nonempty_rows] == [
        "bc",
        "e",
        "fg",
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
    compiler = D64FieldCompiler()

    none = compiler.compile(
        field,
        region_masks={LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("tail_percent", 0)},
    )
    half = compiler.compile(
        field,
        region_masks={LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("tail_percent", 50)},
    )
    full = compiler.compile(
        field,
        region_masks={LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("tail_percent", 100)},
    )

    assert none.source_field_id == half.source_field_id == full.source_field_id == field.field_id
    assert none.region_text(LogicalRegion.CONVERSATION_HISTORY) == ""
    assert half.region_text(LogicalRegion.CONVERSATION_HISTORY) == "fghij"
    assert full.region_text(LogicalRegion.CONVERSATION_HISTORY) == "abcdefghij"
    assert all(compiled.region_text(LogicalRegion.USER_INPUT) == "question" for compiled in (none, half, full))

    full_by_position = {
        address.region_position: (
            address,
            full.lane_cell16(address.row_index, address.lane_index),
        )
        for address in full.region_addresses(LogicalRegion.CONVERSATION_HISTORY)
    }
    for address in half.region_addresses(LogicalRegion.CONVERSATION_HISTORY):
        original, original_cell = full_by_position[address.region_position]
        assert address.global_position == original.global_position
        assert address.span_id == original.span_id
        assert address.span_position == original.span_position
        assert np.array_equal(
            half.lane_cell16(address.row_index, address.lane_index),
            original_cell,
        )


def test_host_circulates_mask_only_change_and_persists_slider(tmp_path: Path) -> None:
    state_root = tmp_path / "State"
    with _host(state_root) as host:
        host.submit_user("abcdefghij", provenance="mask-test")
        first = host.heartbeat()
        assert first.state is HostBeatState.CIRCULATED
        assert first.tick_image is not None
        field_id = first.field.field_id

        mask_state = host.set_region_unmasked_percent(LogicalRegion.USER_INPUT, 50)
        second = host.heartbeat()
        assert second.state is HostBeatState.CIRCULATED
        assert second.commits == ()
        assert second.field.field_id == field_id
        assert second.tick_image is not None
        assert second.tick_image.view_id != first.tick_image.view_id
        compiled = D64FieldCompiler().compile(
            second.field,
            region_masks=host.mask_controller.policies(),
        )
        assert compiled.region_text(LogicalRegion.USER_INPUT) == "fghij"
        assert mask_state.revision == 1
        assert host.health()["mask_state_id"] == mask_state.state_id

    with _host(state_root) as reopened:
        assert reopened.region_mask_state().policy_for(LogicalRegion.USER_INPUT) == RegionMaskPolicy("tail_percent", 50)


def test_mask_move_waits_for_in_flight_tick(tmp_path: Path) -> None:
    cores = CoreRegistry((CoreDescriptor(core_id="core64", d_model=64),))
    with _host(tmp_path / "State", cores=cores) as host:
        host.submit_user("abcdefghij")
        first = host.heartbeat()
        assert first.tick_image is not None
        assert host.coordinator.tick_in_flight

        host.set_region_unmasked_percent(LogicalRegion.USER_INPUT, 0)
        held = host.heartbeat()
        assert held.state is HostBeatState.TICK_IN_FLIGHT
        assert held.tick_image is first.tick_image

        host.coordinator.close_tick()
        moved = host.heartbeat()
        assert moved.state is HostBeatState.CIRCULATED
        assert moved.field.field_id == first.field.field_id
        assert moved.tick_image is not None
        assert moved.tick_image.view_id != first.tick_image.view_id

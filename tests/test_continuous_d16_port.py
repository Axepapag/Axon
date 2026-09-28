"""Prove the ContinuousCoreLab's phase-0 gate 3 inside Axon.

Lab gate 3 (D:/ContinuousCoreLab/README.txt):
    "Applying INPUT/HISTORY deltas changes D512 state without replaying the whole field."

This test asserts the same claim against Axon's real D16 field machinery:
  * a resident ContinuousCoreD512-backed port ingests the exact field once,
  * subsequent beats deliver ONLY sub-region deltas,
  * the Core's recurrent state genuinely changes,
  * and the work done scales with the number of CHANGED regions, not with the
    size of the field.
"""
from __future__ import annotations

import torch

from runtime.axon_runtime import ContinuousCoreD512
from runtime.field import (
    D16View,
    D16ViewDelta,
    LogicalRegion,
    SharedFieldSnapshot,
    materialize_d16_view,
)
from runtime.heart import FieldDeltaEvent, FieldSnapshotEvent
from runtime.heart.continuous_d16_port import ContinuousD16CorePort, region_preamble
from substrate import encode_unicode_text


def _view(regions: dict, *, tick_id: int) -> D16View:
    return materialize_d16_view(SharedFieldSnapshot.from_texts(regions, tick_id=tick_id))


def _delta_between(base: D16View, target: D16View) -> D16ViewDelta:
    return D16ViewDelta.between(base, target)


def test_port_satisfies_the_runtime_d16_port_protocol() -> None:
    from runtime.heart.circulation import D16ReasoningCorePort

    port = ContinuousD16CorePort("proto-core", ContinuousCoreD512())
    assert isinstance(port, D16ReasoningCorePort)


def test_snapshot_ingests_exact_field_and_initialises_state() -> None:
    port = ContinuousD16CorePort("boot-core", ContinuousCoreD512())
    regions = {LogicalRegion.USER_INPUT: "CAT", LogicalRegion.CORTEX: "DOG"}
    target = _view(regions, tick_id=1)

    ack = port.apply_field_event(FieldSnapshotEvent(sequence=0, view=target))

    assert ack.identity == target.identity
    state = port.recurrent_state
    assert state is not None
    assert float(state.norm().item()) > 0.0
    assert port.stats.snapshots_applied == 1
    # exact mirror holds the field: decode round-trips
    assert port.mirror.view.region(LogicalRegion.USER_INPUT).text == "CAT"


def test_gate_three_delta_changes_state_without_full_replay() -> None:
    """THE LAB GATE: deltas change Core state without replaying the whole field."""
    port = ContinuousD16CorePort("gate-core", ContinuousCoreD512())

    base_regions = {
        LogicalRegion.USER_INPUT: "CAT",
        LogicalRegion.CORTEX: "nothing here yet",
        LogicalRegion.CONVERSATION_HISTORY: "",
    }
    base = _view(base_regions, tick_id=1)
    port.apply_field_event(FieldSnapshotEvent(sequence=0, view=base))
    state_after_snapshot = port.recurrent_state.clone()
    cells_after_snapshot = port.stats.cells_ingested
    regions_after_snapshot = port.stats.regions_ingested

    # ONE region changes: user_input only.
    changed = dict(base_regions)
    changed[LogicalRegion.USER_INPUT] = "DOG"
    target = _view(changed, tick_id=2)
    ack = port.apply_field_event(FieldDeltaEvent(sequence=1, delta=_delta_between(base, target)))

    state_after_delta = port.recurrent_state
    assert ack.identity == target.identity

    # 1. state genuinely changed
    assert not torch.equal(state_after_snapshot, state_after_delta)

    # 2. the mirror reconstructed exactly
    assert port.mirror.view.region(LogicalRegion.USER_INPUT).text == "DOG"

    # 3. NO FULL REPLAY: only the changed region's cells were ingested this beat
    cells_this_beat = port.stats.cells_ingested - cells_after_snapshot
    regions_this_beat = port.stats.regions_ingested - regions_after_snapshot
    assert regions_this_beat == 1, f"expected exactly one region ingested, got {regions_this_beat}"
    field_cells = sum(len(item.token_ids) for item in target.regions)
    assert cells_this_beat < field_cells, "delta beat must read fewer cells than the whole field"
    assert cells_this_beat > 0
    assert port.stats.deltas_applied == 1


def test_resident_delta_path_reads_less_than_full_replay() -> None:
    """The lab's core claim: a delta beat reads only what changed, not the whole field.

    Measured per beat against a field at realistic scale. The delta saving is real
    but it is NOT free: each ingested region carries an exact ``[region]`` preamble,
    so the win grows with region size. The crossover is tested separately below —
    on a three-character field the preamble costs more than it saves, which is
    exactly why this is measured rather than assumed.
    """
    regions = {
        LogicalRegion.USER_INPUT: "the user asked about the dog again",
        LogicalRegion.CORTEX: "cortex content " * 40,
        LogicalRegion.CONVERSATION_HISTORY: "earlier turns " * 40,
    }

    port = ContinuousD16CorePort("resident-core", ContinuousCoreD512())
    base = _view(regions, tick_id=1)
    port.apply_field_event(FieldSnapshotEvent(sequence=0, view=base))
    assert port.stats.snapshots_applied == 1

    # Walk a few beats, changing exactly one region each time.
    walked = dict(regions)
    changes = [
        (LogicalRegion.USER_INPUT, "the user asked about the cat now"),
        (LogicalRegion.CORTEX, "cortex grew further " * 40),
        (LogicalRegion.USER_INPUT, "the user asked about the fox now"),
    ]
    per_beat_cells: list[int] = []
    for index, (region, text) in enumerate(changes, start=2):
        walked = dict(walked)
        walked[region] = text
        target = _view(walked, tick_id=index)
        before = port.stats.cells_ingested
        port.apply_field_event(
            FieldDeltaEvent(sequence=index - 1, delta=_delta_between(port.mirror.view, target))
        )
        per_beat_cells.append(port.stats.cells_ingested - before)

    # Every beat after the first was a delta, never a replay.
    assert port.stats.deltas_applied == 3
    assert port.stats.snapshots_applied == 1

    # THE GATE-3 CLAIM, measured per beat: a delta beat is strictly cheaper than
    # replaying the field it lands on.
    full_replay_cells = sum(len(item.token_ids) for item in _view(walked, tick_id=99).regions)
    for beat_index, cells in enumerate(per_beat_cells):
        assert cells > 0, f"beat {beat_index} read nothing"
        assert cells < full_replay_cells, (
            f"beat {beat_index} read {cells} cells; a full replay would read "
            f"{full_replay_cells} — delta must be cheaper than replay"
        )

    # And the recurrent state is genuinely live (not a degenerate zero vector).
    state = port.recurrent_state
    assert state is not None
    assert state.shape == (1, 512)
    assert float(state.norm().item()) > 0.0

    # The Core holds exactly the regions it actually ingested — all three had
    # content at snapshot time, and three changing beats re-sent one region each.
    assert len(port._last_burst_by_region) == 3
    assert port.stats.regions_ingested == 3 + len(changes), (
        "3 regions at snapshot + one region per changing beat"
    )


def test_delta_saving_has_a_measured_crossover() -> None:
    """Honest limit: the exact region preamble means tiny fields do NOT benefit.

    This is a measured property of the design, not a defect. At realistic region
    sizes the saving is large (the real canonical field is 13 regions at ~2,000
    chars each, and a mean 1.00 region changes per beat -> ~13x). On a field of
    single characters the preamble dominates and deltas can cost slightly MORE.
    """
    # tiny field: preamble dominates
    tiny = {LogicalRegion.USER_INPUT: "A", LogicalRegion.CORTEX: "B"}
    port = ContinuousD16CorePort("tiny-core", ContinuousCoreD512())
    base = _view(tiny, tick_id=1)
    port.apply_field_event(FieldSnapshotEvent(sequence=0, view=base))

    target = _view({LogicalRegion.USER_INPUT: "C", LogicalRegion.CORTEX: "B"}, tick_id=2)
    before = port.stats.cells_ingested
    port.apply_field_event(FieldDeltaEvent(sequence=1, delta=_delta_between(base, target)))
    tiny_delta_cells = port.stats.cells_ingested - before

    preamble_cost = len(encode_unicode_text("[user_input]"))
    assert tiny_delta_cells == preamble_cost + 1, "one preamble + one character"
    # the preamble is the dominant cost here: this is the honest crossover
    assert preamble_cost > 1

    # large field: the same one-region change is far cheaper than a full replay
    big_region = "content " * 200
    big = {
        LogicalRegion.USER_INPUT: "A",
        LogicalRegion.CORTEX: big_region,
        LogicalRegion.CONVERSATION_HISTORY: big_region,
        LogicalRegion.SCRATCH: big_region,
        LogicalRegion.TASK_STATE: big_region,
    }
    port_big = ContinuousD16CorePort("big-core", ContinuousCoreD512())
    base_big = _view(big, tick_id=1)
    port_big.apply_field_event(FieldSnapshotEvent(sequence=0, view=base_big))

    target_big = dict(big)
    target_big[LogicalRegion.USER_INPUT] = "Z"
    view_big = _view(target_big, tick_id=2)
    before = port_big.stats.cells_ingested
    port_big.apply_field_event(
        FieldDeltaEvent(sequence=1, delta=_delta_between(base_big, view_big))
    )
    big_delta_cells = port_big.stats.cells_ingested - before
    big_full_replay = sum(len(item.token_ids) for item in view_big.regions)

    # a one-character change on a large field still costs ~one region, not the field
    assert big_delta_cells < big_full_replay
    assert big_full_replay / big_delta_cells > 4, (
        f"expected a large saving on a realistic field, got "
        f"{big_delta_cells} vs {big_full_replay}"
    )


def test_empty_region_costs_nothing_to_attend() -> None:
    """Empty regions must not be re-sent every beat — that overhead ate the delta gain."""
    port = ContinuousD16CorePort("empty-core", ContinuousCoreD512())
    base = _view({LogicalRegion.USER_INPUT: "CAT", LogicalRegion.CORTEX: ""}, tick_id=1)
    port.apply_field_event(FieldSnapshotEvent(sequence=0, view=base))
    assert LogicalRegion.CORTEX not in port._last_burst_by_region

    # an unrelated region changes; the empty one must stay free
    target = _view({LogicalRegion.USER_INPUT: "DOG", LogicalRegion.CORTEX: ""}, tick_id=2)
    before = port.stats.cells_ingested
    port.apply_field_event(FieldDeltaEvent(sequence=1, delta=_delta_between(base, target)))
    ingested = port.stats.cells_ingested - before
    expected = len(encode_unicode_text("[user_input]")) + len(encode_unicode_text("DOG"))
    assert ingested == expected, f"empty region leaked cells: {ingested} != {expected}"


def test_unchanged_region_is_not_reingested() -> None:
    """Repeated identical beats must not re-read unchanged content."""
    port = ContinuousD16CorePort("idle-core", ContinuousCoreD512())
    regions = {LogicalRegion.CORTEX: "stable content", LogicalRegion.USER_INPUT: "hi"}
    base = _view(regions, tick_id=1)
    port.apply_field_event(FieldSnapshotEvent(sequence=0, view=base))

    target = _view(regions, tick_id=2)
    delta = _delta_between(base, target)
    assert len(getattr(delta, "patches", ())) == 0, "identical content must produce an empty delta"

    before = port.stats.cells_ingested
    port.apply_field_event(FieldDeltaEvent(sequence=1, delta=delta))
    assert port.stats.cells_ingested == before, "an empty delta must ingest no cells"


def test_region_preamble_marks_where_the_reading_came_from() -> None:
    """The Core must know WHERE a reading came from, not just what it says."""
    from substrate import decode_unicode_tokens

    ids = region_preamble(LogicalRegion.CORTEX)
    assert len(ids) > 0
    assert decode_unicode_tokens(ids) == "[cortex]"


def test_stats_are_auditable() -> None:
    port = ContinuousD16CorePort("audit-core", ContinuousCoreD512())
    base = _view({LogicalRegion.USER_INPUT: "A", LogicalRegion.CORTEX: "B"}, tick_id=1)
    port.apply_field_event(FieldSnapshotEvent(sequence=0, view=base))
    target = _view({LogicalRegion.USER_INPUT: "C", LogicalRegion.CORTEX: "B"}, tick_id=2)
    port.apply_field_event(FieldDeltaEvent(sequence=1, delta=_delta_between(base, target)))

    audit = port.audit()
    assert audit["schema"] == "axon-continuous-d16-core-port-v1"
    assert audit["has_view"] is True
    assert audit["state_initialized"] is True
    assert audit["stats"]["snapshots_applied"] == 1
    assert audit["stats"]["deltas_applied"] == 1
    # snapshot ingested both regions (A, B); the delta ingested only the changed one (C)
    assert audit["stats"]["regions_ingested"] == 3
    # and exactly one region changed, so the delta beat did one region of work
    assert audit["stats"]["cells_ingested"] == (
        len(encode_unicode_text("[user_input]")) + 1  # A
        + len(encode_unicode_text("[cortex]")) + 1  # B
        + len(encode_unicode_text("[user_input]")) + 1  # C
    )

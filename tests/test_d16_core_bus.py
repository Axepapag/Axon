from __future__ import annotations

import numpy as np
import pytest

from runtime.field import (
    DeleteText,
    D16RegionView,
    D16ResyncRequired,
    D16ViewDelta,
    FieldDelta,
    InsertText,
    LogicalRegion,
    RegionMaskPolicy,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
    materialize_d16_view,
)
from runtime.heart import (
    D16CoreMirror,
    D16TextFrame,
    FieldDeltaEvent,
    FieldSnapshotEvent,
    MirrorAck,
    MirrorCoherenceError,
    MirrorCoherenceRegistry,
    MirrorSyncState,
)
from substrate import decode_unicode_tokens, encode_unicode_text, transport_token_cell16


def _base_snapshot() -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.USER_INPUT: "Hello 🙂 世界",
            LogicalRegion.RESPONSE_DRAFT: "CAT",
            LogicalRegion.SCRATCH: "alpha",
        },
        tick_id=7,
    )


def _delta(snapshot: SharedFieldSnapshot, *operations) -> FieldDelta:
    return FieldDelta(
        base_field_id=snapshot.field_id,
        base_tick_id=snapshot.tick_id,
        author_core_id="d16-test-core",
        pass_id=f"p-{snapshot.tick_id}",
        operations=tuple(operations),
        evidence=("test",),
    )


def test_complete_view_is_exact_registered_d16_transport() -> None:
    snapshot = _base_snapshot()
    view = materialize_d16_view(snapshot)

    for region_view in view.regions:
        expected_ids = encode_unicode_text(region_view.text)
        assert region_view.token_ids == expected_ids
        assert decode_unicode_tokens(expected_ids) == region_view.text
        assert region_view.cells16.shape == (len(expected_ids), 16)
        for index, token_id in enumerate(expected_ids):
            assert np.array_equal(region_view.cells16[index], transport_token_cell16(token_id))

    assert view.source_field_id == snapshot.field_id
    assert view.source_tick_id == snapshot.tick_id
    assert len(view.view_id) == 64
    assert len(view.view_hash) == 64


def test_region_view_owns_cell_buffer_and_identity_survives_caller_mutation() -> None:
    source = materialize_d16_view(_base_snapshot()).region(LogicalRegion.USER_INPUT)
    backing = source.cells16.copy()
    owned = D16RegionView(
        region=source.region,
        text=source.text,
        cells16=backing,
        addresses=source.addresses,
    )
    identity_before = (owned.cells_sha256, owned.region_hash)
    cells_before = owned.cells16.copy()

    # Mutate the caller's original writable buffer after construction.  The
    # region must own a detached immutable copy or Heart could trust a stale
    # hash/identity over changed cells.
    backing[0, 0] = backing[0, 0] + np.float32(123.0)

    assert not np.shares_memory(backing, owned.cells16)
    assert np.array_equal(owned.cells16, cells_before)
    assert (owned.cells_sha256, owned.region_hash) == identity_before
    assert not owned.cells16.flags.writeable


def test_insert_delete_replace_deltas_match_clean_rebuild_exactly() -> None:
    snapshot = _base_snapshot()
    view = materialize_d16_view(snapshot)
    mirror = D16CoreMirror("core-a", 1)
    mirror.apply_snapshot(FieldSnapshotEvent(sequence=10, view=view))
    sequence = 10

    mutations = (
        lambda s: _delta(s, InsertText(LogicalRegion.RESPONSE_DRAFT, 3, "🙂")),
        lambda s: _delta(s, ReplaceText(LogicalRegion.SCRATCH, 0, len(s.region(LogicalRegion.SCRATCH).text), "βeta")),
        lambda s: _delta(s, DeleteText(LogicalRegion.RESPONSE_DRAFT, 1, 2)),
    )

    for make_delta in mutations:
        canonical_delta = make_delta(snapshot)
        successor = apply_delta(snapshot, canonical_delta)
        rebuilt = materialize_d16_view(successor)
        bus_delta = D16ViewDelta.between(view, rebuilt)
        sequence += 1
        ack = mirror.apply_delta(FieldDeltaEvent(sequence=sequence, delta=bus_delta))
        assert mirror.view is not None
        assert mirror.view.identity == rebuilt.identity
        assert ack.identity == rebuilt.identity
        assert mirror.view.view_hash == rebuilt.view_hash
        for region in rebuilt.regions:
            mirrored = mirror.view.region(region.region)
            assert mirrored.text == region.text
            assert mirrored.region_hash == region.region_hash
            assert np.array_equal(mirrored.cells16, region.cells16)
        snapshot, view = successor, rebuilt


def test_event_gap_fails_closed_and_full_snapshot_repairs() -> None:
    base = _base_snapshot()
    base_view = materialize_d16_view(base)
    successor = apply_delta(
        base,
        _delta(base, ReplaceText(LogicalRegion.RESPONSE_DRAFT, 0, 3, "DOG")),
    )
    target = materialize_d16_view(successor)
    delta = D16ViewDelta.between(base_view, target)

    mirror = D16CoreMirror("core-gap", 2)
    mirror.apply_snapshot(FieldSnapshotEvent(sequence=20, view=base_view))
    with pytest.raises(D16ResyncRequired, match="sequence gap"):
        mirror.apply_delta(FieldDeltaEvent(sequence=22, delta=delta))

    repair = mirror.apply_snapshot(FieldSnapshotEvent(sequence=22, view=target))
    assert repair.identity == target.identity
    assert mirror.identity == target.identity


def test_stale_delta_base_fails_closed() -> None:
    base = _base_snapshot()
    base_view = materialize_d16_view(base)
    first = apply_delta(base, _delta(base, ReplaceText(LogicalRegion.RESPONSE_DRAFT, 0, 3, "DOG")))
    first_view = materialize_d16_view(first)
    stale_delta = D16ViewDelta.between(base_view, first_view)

    second = apply_delta(first, _delta(first, ReplaceText(LogicalRegion.RESPONSE_DRAFT, 0, 3, "OWL")))
    second_view = materialize_d16_view(second)

    mirror = D16CoreMirror("core-stale", 1)
    mirror.apply_snapshot(FieldSnapshotEvent(sequence=30, view=second_view))
    with pytest.raises(D16ResyncRequired, match="base identity"):
        mirror.apply_delta(FieldDeltaEvent(sequence=31, delta=stale_delta))


def test_mask_policy_change_requires_resync_not_guessing() -> None:
    snapshot = _base_snapshot()
    all_view = materialize_d16_view(
        snapshot,
        region_masks={LogicalRegion.USER_INPUT: RegionMaskPolicy("tail_percent", 100)},
    )
    half_view = materialize_d16_view(
        snapshot,
        region_masks={LogicalRegion.USER_INPUT: RegionMaskPolicy("tail_percent", 50)},
    )
    assert all_view.mask_id != half_view.mask_id
    with pytest.raises(D16ResyncRequired, match="mask policy identity changed"):
        D16ViewDelta.between(all_view, half_view)


def test_same_mask_policy_survives_content_delta_without_full_replay() -> None:
    base = _base_snapshot()
    masks = {LogicalRegion.RESPONSE_DRAFT: RegionMaskPolicy("all")}
    base_view = materialize_d16_view(base, region_masks=masks)
    successor = apply_delta(base, _delta(base, InsertText(LogicalRegion.RESPONSE_DRAFT, 3, "DOG")))
    target_view = materialize_d16_view(successor, region_masks=masks)
    assert base_view.mask_id == target_view.mask_id
    delta = D16ViewDelta.between(base_view, target_view)
    assert delta.apply(base_view).identity == target_view.identity


def test_reconnect_full_snapshot_equals_incremental_mirror() -> None:
    base = _base_snapshot()
    base_view = materialize_d16_view(base)
    successor = apply_delta(base, _delta(base, ReplaceText(LogicalRegion.RESPONSE_DRAFT, 0, 3, "FOX")))
    target_view = materialize_d16_view(successor)
    bus_delta = D16ViewDelta.between(base_view, target_view)

    resident = D16CoreMirror("resident", 1)
    resident.apply_snapshot(FieldSnapshotEvent(sequence=1, view=base_view))
    resident.apply_delta(FieldDeltaEvent(sequence=2, delta=bus_delta))

    reconnect = D16CoreMirror("reconnect", 1)
    reconnect.apply_snapshot(FieldSnapshotEvent(sequence=2, view=target_view))
    assert resident.identity == reconnect.identity == target_view.identity
    assert resident.view is not None and reconnect.view is not None
    assert resident.view.view_hash == reconnect.view.view_hash


def test_proposal_text_roundtrips_exactly_over_d16_bus() -> None:
    text = "FIRST PROPOSALS\n\n[core-a]\nI remember café 🙂 中文 العربية."
    frame = D16TextFrame(text)
    assert frame.decode() == text
    assert frame.token_ids == encode_unicode_text(text)
    assert frame.cells16.shape == (len(frame.token_ids), 16)
    assert len(frame.frame_id) == 64


def test_heart_mirror_coherence_excludes_stale_core_and_allows_rejoin() -> None:
    snapshot = _base_snapshot()
    view = materialize_d16_view(snapshot)
    event = FieldSnapshotEvent(sequence=100, view=view)
    core = D16CoreMirror("core-coherent", 4)
    registry = MirrorCoherenceRegistry()
    registry.register(core.core_id, core.core_generation)

    assert not registry.barrier_eligible(core.core_id, view.identity)
    registry.expect(core.core_id, event)
    ack = core.apply_snapshot(event)
    record = registry.accept_ack(ack)
    assert record.state is MirrorSyncState.SYNCED
    assert registry.barrier_eligible(core.core_id, view.identity)
    registry.require_barrier_eligible(core.core_id, view.identity)

    canonical_field_id = snapshot.field_id
    registry.mark_stale(core.core_id, "simulated missed event")
    assert not registry.barrier_eligible(core.core_id, view.identity)
    with pytest.raises(MirrorCoherenceError, match="not mirror-coherent"):
        registry.require_barrier_eligible(core.core_id, view.identity)

    repair_event = FieldSnapshotEvent(sequence=101, view=view)
    registry.expect(core.core_id, repair_event)
    registry.accept_ack(core.apply_snapshot(repair_event))
    assert registry.barrier_eligible(core.core_id, view.identity)
    assert snapshot.field_id == canonical_field_id


def test_wrong_ack_forces_resync_required() -> None:
    base = _base_snapshot()
    base_view = materialize_d16_view(base)
    successor = apply_delta(base, _delta(base, ReplaceText(LogicalRegion.RESPONSE_DRAFT, 0, 3, "BIR")))
    target_view = materialize_d16_view(successor)

    registry = MirrorCoherenceRegistry()
    registry.register("core-bad", 1)
    event = FieldSnapshotEvent(sequence=5, view=base_view)
    registry.expect("core-bad", event)
    bad_ack = MirrorAck(
        core_id="core-bad",
        core_generation=1,
        event_sequence=5,
        event_id=event.event_id,
        identity=target_view.identity,
    )
    with pytest.raises(MirrorCoherenceError, match="does not match"):
        registry.accept_ack(bad_ack)
    assert registry.record("core-bad").state is MirrorSyncState.RESYNC_REQUIRED


def test_registry_rejects_delta_when_heart_acknowledged_base_is_different() -> None:
    base = _base_snapshot()
    base_view = materialize_d16_view(base)
    successor = apply_delta(base, _delta(base, ReplaceText(LogicalRegion.RESPONSE_DRAFT, 0, 3, "EMU")))
    target_view = materialize_d16_view(successor)
    delta_event = FieldDeltaEvent(sequence=2, delta=D16ViewDelta.between(base_view, target_view))

    other = apply_delta(successor, _delta(successor, ReplaceText(LogicalRegion.RESPONSE_DRAFT, 0, 3, "ANT")))
    other_view = materialize_d16_view(other)

    registry = MirrorCoherenceRegistry()
    registry.register("core-base", 1)
    snapshot_event = FieldSnapshotEvent(sequence=1, view=other_view)
    core = D16CoreMirror("core-base", 1)
    registry.expect("core-base", snapshot_event)
    registry.accept_ack(core.apply_snapshot(snapshot_event))

    with pytest.raises(MirrorCoherenceError, match="delta base"):
        registry.expect("core-base", delta_event)
    assert registry.record("core-base").state is MirrorSyncState.RESYNC_REQUIRED

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.field import LogicalRegion, RegionMaskPolicy
from runtime.heart import (
    BeatConfig,
    HeartHost,
    HeartHostConfig,
    HostBeatState,
    HostStateError,
    LeaseDeniedError,
    ValveBudget,
    ValveEnvelope,
)


def _host(
    root: Path,
    *,
    region_policies=None,
    global_items: int = 256,
) -> HeartHost:
    return HeartHost(
        state_root=root,
        host_config=HeartHostConfig(
            idle_interval_seconds=0.01,
            global_budget=ValveBudget(
                pending_cap=10_000,
                items_per_beat=global_items,
                chars_per_beat=32_768,
                max_item_chars=16_384,
                max_item_bytes=65_536,
            ),
        ),
        beat_config=BeatConfig(recall_limit=0, region_policies=region_policies),
    )


def _user(text: str, *, source: str = "external_user") -> ValveEnvelope:
    return ValveEnvelope(
        valve_id="user_ingress",
        source_id=source,
        payload=text,
        provenance="test",
        envelope_type="text/plain",
    )


def test_idle_heartbeat_advances_health_without_canonical_churn(tmp_path: Path) -> None:
    with _host(tmp_path) as host:
        before = host.coordinator.current_field
        first = host.heartbeat()
        second = host.heartbeat()
        assert first.state is HostBeatState.IDLE
        assert second.state is HostBeatState.IDLE
        assert second.field.field_id == before.field_id
        assert second.field.tick_id == before.tick_id
        assert first.tick_image is None and second.tick_image is None
        health = host.health()
        assert health["heartbeat_sequence"] == 2
        assert health["tick_sequence"] == 0
        assert health["lease_owner_pid"] is not None


def test_second_host_fails_closed_on_single_writer_lease(tmp_path: Path) -> None:
    first = _host(tmp_path)
    first.start()
    try:
        second = _host(tmp_path)
        with pytest.raises(LeaseDeniedError):
            second.start()
        with pytest.raises(HostStateError):
            second.heartbeat()
    finally:
        first.stop()


def test_restart_preserves_monotonic_heartbeat_and_tick_identity(tmp_path: Path) -> None:
    first = _host(tmp_path)
    first.start()
    try:
        first.submit_user("hello")
        beat = first.heartbeat()
        assert beat.state is HostBeatState.CIRCULATED
        h1 = first.health()
        assert h1["tick_sequence"] == 1
        start1 = h1["start_sequence"]
    finally:
        first.stop()

    second = _host(tmp_path)
    second.start()
    try:
        idle = second.heartbeat()
        assert idle.state is HostBeatState.IDLE
        h2 = second.health()
        assert h2["heartbeat_sequence"] > h1["heartbeat_sequence"]
        assert h2["tick_sequence"] == h1["tick_sequence"]
        assert h2["start_sequence"] == start1 + 1
        assert h2["heart_epoch_id"] == h1["heart_epoch_id"]
    finally:
        second.stop()


def test_pending_ingress_survives_restart(tmp_path: Path) -> None:
    first = _host(tmp_path)
    first.start()
    try:
        decision = first.submit_user("survive")
        assert decision.admitted
        assert first.spool.pending_count == 1
    finally:
        first.stop()

    second = _host(tmp_path)
    second.start()
    try:
        result = second.heartbeat()
        assert result.state is HostBeatState.CIRCULATED
        assert result.field.region(LogicalRegion.USER_INPUT).text == "survive"
        assert second.spool.pending_count == 0
    finally:
        second.stop()


def test_commit_before_ack_recovery_does_not_duplicate_text(tmp_path: Path, monkeypatch) -> None:
    first = _host(tmp_path)
    first.start()
    try:
        first.submit_user("once")
        original_ack = first.spool.acknowledge
        called = {"value": False}

        def fail_once(event_id: str) -> None:
            if not called["value"]:
                called["value"] = True
                raise OSError("injected crash-before-ack")
            original_ack(event_id)

        monkeypatch.setattr(first.spool, "acknowledge", fail_once)
        with pytest.raises(OSError, match="crash-before-ack"):
            first.heartbeat()
        assert first.coordinator.current_field.region(LogicalRegion.USER_INPUT).text == "once"
        assert first.spool.pending_count == 1
    finally:
        first.stop()

    second = _host(tmp_path)
    second.start()
    try:
        recovered = second.heartbeat()
        assert recovered.field.region(LogicalRegion.USER_INPUT).text == "once"
        assert second.spool.pending_count == 0
    finally:
        second.stop()


def test_oversize_poison_is_quarantined_and_valid_successor_circulates(tmp_path: Path) -> None:
    with _host(tmp_path) as host:
        rejected = host.submit_user("x" * 10_000)
        admitted = host.submit_user("ok")
        assert not rejected.admitted
        assert admitted.admitted
        result = host.heartbeat()
        assert result.field.region(LogicalRegion.USER_INPUT).text == "ok"
        health = host.health()
        assert health["quarantine_count"] >= 1
        assert health["rejection_count"] >= 1


def test_closed_and_wrong_source_valves_cannot_mutate(tmp_path: Path) -> None:
    with _host(tmp_path) as host:
        closed = host.submit(
            ValveEnvelope(
                valve_id="semantic_cortex",
                source_id="reserved",
                payload="no",
                provenance="test",
                envelope_type="none",
            )
        )
        wrong_source = host.submit(_user("no", source="external_tool"))
        assert not closed.admitted
        assert not wrong_source.admitted
        beat = host.heartbeat()
        assert beat.state is HostBeatState.IDLE
        assert beat.field.region(LogicalRegion.USER_INPUT).text == ""


def test_global_budget_defers_fifo_head_to_next_heartbeat(tmp_path: Path) -> None:
    with _host(tmp_path, global_items=1) as host:
        host.submit_user("first")
        host.submit_user("second")
        first = host.heartbeat()
        assert first.field.region(LogicalRegion.USER_INPUT).text == "first"
        assert first.deferred_event_id is not None
        assert host.spool.pending_count == 1
        second = host.heartbeat()
        assert second.field.region(LogicalRegion.USER_INPUT).text == "firstsecond"
        assert host.spool.pending_count == 0


def test_failure_after_durable_ingress_resynchronizes_to_branch_head(tmp_path: Path, monkeypatch) -> None:
    with _host(tmp_path) as host:
        host.submit_user("persisted")

        def broken_recall(field):
            raise RuntimeError("injected recall failure")

        monkeypatch.setattr(host.coordinator, "stabilize_recall", broken_recall)
        # Force recall execution even though test recall is otherwise disabled.
        with pytest.raises(RuntimeError, match="injected recall failure"):
            host.heartbeat()
        branch_head = host.coordinator.branch.load_head()
        current = host.coordinator.current_field
        assert current.field_id == branch_head.field_id
        assert current.region(LogicalRegion.USER_INPUT).text == "persisted"


def test_derived_view_identity_distinguishes_mask_policies(tmp_path: Path) -> None:
    policy_a = {LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("last_n_spans", 1)}
    policy_b = {LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("last_n_spans", 2)}
    with _host(tmp_path / "a", region_policies=policy_a) as host_a:
        host_a.submit_user("x")
        image_a = host_a.heartbeat().tick_image
    with _host(tmp_path / "b", region_policies=policy_b) as host_b:
        host_b.submit_user("x")
        image_b = host_b.heartbeat().tick_image
    assert image_a is not None and image_b is not None
    assert image_a.view_id != image_b.view_id
    assert image_a.require_rail(64).view_id == image_a.view_id
    assert image_b.require_rail(64).view_id == image_b.view_id


def test_tool_and_advisor_route_only_to_heart_owned_regions(tmp_path: Path) -> None:
    with _host(tmp_path) as host:
        host.submit_tool("tool output")
        host.submit_advisor("advisor note")
        result = host.heartbeat()
        assert result.field.region(LogicalRegion.TOOL_RESULTS).text == "tool output"
        assert result.field.region(LogicalRegion.ADVISOR_INPUT).text == "advisor note"
        assert result.field.region(LogicalRegion.USER_INPUT).text == ""


def test_branch_journal_contains_rich_heart_valve_provenance(tmp_path: Path) -> None:
    with _host(tmp_path) as host:
        decision = host.submit_user("audit me", provenance="unit-test")
        assert decision.receipt is not None
        event_id = decision.receipt.item_id
        result = host.heartbeat()
        assert result.state is HostBeatState.CIRCULATED
        lines = host.coordinator.branch.journal_path.read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines if line.strip()]
        commit_events = [item for item in events if item.get("event") == "commit"]
        heart = commit_events[-1]["metadata"]["heart_commit"]
        provenance = heart["valve_provenance"]
        assert provenance["valve_id"] == "user_ingress"
        assert provenance["valve_version"] == 1
        assert provenance["source_id"] == "external_user"
        assert provenance["item_id"] == event_id
        assert provenance["authority_class"] == "external_ingress"
        assert provenance["governed_regions"] == ["user_input"]
        assert heart["base_field_id"] != heart["successor_field_id"]


def test_valve_envelope_has_no_authority_grant_surface() -> None:
    envelope = _user("hello")
    assert not hasattr(envelope, "grant")
    assert not hasattr(envelope, "authority_grant")


def test_final_gate_budget_slot_releases_when_commit_throws(tmp_path: Path, monkeypatch) -> None:
    with _host(tmp_path) as host:
        host.submit_user("retry-me")

        def broken_commit(*args, **kwargs):
            raise RuntimeError("injected commit failure")

        monkeypatch.setattr(host.coordinator, "commit_heart_delta", broken_commit)
        with pytest.raises(RuntimeError, match="injected commit failure"):
            host.heartbeat()
        assert host.valves.tracker.pending("user_ingress") == 0
        assert host.spool.pending_count == 1

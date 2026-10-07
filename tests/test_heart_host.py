"""HeartHost coordinator tests: lease, beat, admission, ids, checkpoints.

Every test uses ``tmp_path`` for all durable state so nothing touches the
repository's real State folder.  The one deliberately large test (renewable
per-beat valve budgets) exercises the real production valve registry with its
100-items-per-beat primitive budget.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from runtime.field import (
    BRANCH_EVENT_SCHEMA,
    CanonicalStateBranch,
    LogicalRegion,
    SharedFieldSnapshot,
    StaleDeltaError,
    replacement_delta,
)
from runtime.heart import (
    AuthorityClass,
    AuthorityGrant,
    DuplicateCoreError,
    DurableIngressSpool,
    HeartValveDefinition,
    HeartValveRegistry,
    LeaseDeniedError,
    ReplayEventError,
    SingleWriterLease,
    StaleBaseProposalError,
    UnknownCoreError,
    ValveBudget,
    ValveDecision,
    ValveEnvelope,
    ValveState,
)
from runtime.heart.errors import AuthorityViolationError, HostStateError
from runtime.heart.host import (
    CHECKPOINT_FORMAT,
    EVENT_LOG_CAP,
    EVENT_TYPES,
    CheckpointIntegrityError,
    HeartHost,
    submission_id_for_envelope,
)
from substrate.native import UnsupportedCharacterError


def _envelope(
    payload: str,
    *,
    provenance: str = "test",
    valve_id: str = "user_ingress",
    source_id: str = "external_user",
) -> ValveEnvelope:
    return ValveEnvelope(
        valve_id=valve_id,
        source_id=source_id,
        payload=payload,
        provenance=provenance,
        envelope_type="text/plain",
    )


def _host(root: Path, **kwargs: object) -> HeartHost:
    kwargs.setdefault("core_ids", ("core-a", "core-b"))
    kwargs.setdefault("consolidator_ids", ("core-a", "core-b"))
    return HeartHost(root, **kwargs)


def _proposal(
    host: HeartHost,
    text: str,
    *,
    core_id: str = "core-a",
    region: str = "scratch",
):
    head = host.branch.load_head()
    return host.submit_proposal(
        core_id,
        delta=replacement_delta(
            head,
            region=region,
            text=text,
            author_core_id=core_id,
            pass_id="test-pass",
        ),
    )


def _tiny_registry(items_per_beat: int = 1) -> HeartValveRegistry:
    """One CAPPED user-ingress valve with a deliberately tiny per-beat budget."""

    return HeartValveRegistry(
        [
            HeartValveDefinition(
                valve_id="user_ingress",
                version=2,
                state=ValveState.CAPPED,
                source_class="external_user",
                authority_class=AuthorityClass.EXTERNAL_INGRESS,
                governed_regions=frozenset({LogicalRegion.USER_INPUT}),
                envelope_type="text/plain",
                budget=ValveBudget(
                    items_per_beat=items_per_beat, target_chars_per_beat=10000
                ),
                rejection_policy="quarantine",
            )
        ]
    )


def _events_of_type(events: list[dict], event_type: str) -> list[dict]:
    """The drained (single, seq-ordered) event list, filtered by type."""

    return [event for event in events if event["type"] == event_type]


@pytest.fixture()
def host(tmp_path: Path):
    instance = _host(tmp_path)
    instance.start()
    yield instance
    instance.stop()


# --------------------------------------------------------------------- lease


def test_start_refuses_when_another_host_holds_the_lease(tmp_path: Path) -> None:
    holder = SingleWriterLease(tmp_path, owner_token="other-host")
    holder.acquire()
    host = _host(tmp_path)
    with pytest.raises(LeaseDeniedError):
        host.start()
    with pytest.raises(HostStateError):
        host.beat()  # a refused start leaves no writable host behind

    holder.release()
    status = host.start()
    assert status["started"] is True
    assert host.lease.is_held_by_us()
    host.stop()


def test_commit_and_beat_require_the_lease(tmp_path: Path) -> None:
    host = _host(tmp_path)
    host.start()
    proposal_id = _proposal(host, "held")
    head_before = host.branch.load_head_record()
    host.stop()

    with pytest.raises(LeaseDeniedError):
        host.commit("core-a", proposal_id)
    with pytest.raises(LeaseDeniedError):
        host.beat()
    with pytest.raises(HostStateError):
        HeartHost(tmp_path).beat()

    # the refused canonical write changed nothing: the host is the choke point
    assert host.branch.load_head_record() == head_before
    assert host.branch.load_snapshot(head_before.field_id).region("scratch").text == ""


# ---------------------------------------------------------------------- beat


def test_beat_rotates_renewable_valve_budgets_across_beats(tmp_path: Path) -> None:
    """More than 100 items pass across two beats with zero permanent rejections.

    Without the first-phase ``reset_beat`` the four CAPPED primitive valves
    would admit 100 items and then reject forever; here the surplus is deferred
    (``cap_exhausted``) and admitted on the next beat.
    """

    host = _host(tmp_path, core_ids=(), consolidator_ids=("core-a",))
    host.start()
    payloads = [f"x{index}" for index in range(105)]
    for payload in payloads:
        receipt = host.submit_ingress(_envelope(payload, provenance="budget"))
        assert receipt.status == "staged"

    first = host.beat()
    second = host.beat()

    assert len(first.commits) == 100
    assert len(first.deferred_submissions) == 5
    assert first.rejected_submissions == ()
    assert len(second.commits) == 5
    assert second.deferred_submissions == ()
    assert host.spool.rejection_count == 0
    assert host.spool.pending_count == 0
    assert host.committed_text("user_input") == "".join(payloads)
    host.stop()


def test_committed_ack_carries_submission_commit_generation_and_text(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    host.start()
    receipt = host.submit_ingress(_envelope("ack me", provenance="ack"))
    result = host.beat()
    ack = result.commits[0]

    assert ack.submission_id == receipt.submission_id
    assert ack.commit_id == ack.delta_id
    assert ack.generation == 1
    assert ack.committed_text == "ack me"
    assert ack.regions == ("user_input",)
    assert host.generation == 1
    host.stop()


# ------------------------------------------------------------------ ingress


def test_non_native_payload_is_rejected_before_the_spool(tmp_path: Path, host) -> None:
    receipt = host.submit_ingress(_envelope("tab\tand curly \u2019 and emoji \U0001F600"))

    assert receipt.status == "rejected"
    assert receipt.submission_id is None
    assert "non_native_payload" in receipt.reason
    assert "U+2019" in receipt.reason or "U+1F600" in receipt.reason
    assert host.spool.rejection_count == 1
    assert host.spool.quarantine_count == 1
    assert host.spool.pending_count == 0
    journal = host.spool.journal_path
    assert not journal.exists() or journal.read_text(encoding="utf-8").strip() == ""
    rejection_lines = host.spool.rejection_path.read_text(encoding="utf-8").splitlines()
    assert len(rejection_lines) == 1
    assert "non_native_payload" in json.loads(rejection_lines[0])["reason"]

    result = host.beat()
    assert result.commits == ()
    assert host.generation == 0


def test_empty_and_misaddressed_ingress_is_rejected_before_the_spool(host) -> None:
    assert host.submit_ingress(_envelope("")).status == "rejected"
    assert host.submit_ingress(_envelope("", provenance="empty")).reason == "empty_payload"

    wrong_source = _envelope("hello", source_id="external_tool")
    receipt = host.submit_ingress(wrong_source)
    assert receipt.status == "rejected"
    assert "valve_local_reject" in receipt.reason

    closed = _envelope("hello", valve_id="semantic_cortex", source_id="reserved")
    assert host.submit_ingress(closed).status == "rejected"

    assert host.spool.rejection_count == 4
    assert host.spool.pending_count == 0


def test_recovered_non_native_spool_head_is_quarantined_without_blocking_fifo(
    tmp_path: Path,
) -> None:
    """A poison durable-FIFO head must not block the tail behind it."""

    spool = DurableIngressSpool(tmp_path / "active" / "heart" / "ingress_spool")
    admitted = ValveDecision(True, "admitted_local", False, None)
    spool.submit(
        _envelope("poison \u2019", provenance="tamper"), decision=admitted, valve_version=2
    )
    spool.submit(
        _envelope("good tail", provenance="tamper"), decision=admitted, valve_version=2
    )

    host = _host(tmp_path)
    host.start()
    result = host.beat()

    assert host.committed_text("user_input") == "good tail"
    assert len(result.commits) == 1
    assert len(result.rejected_submissions) == 1
    assert host.spool.pending_count == 0
    assert host.spool.quarantine_count >= 1
    reasons = [
        event["payload"]["reason"]
        for event in _events_of_type(host.drain_events(), "quarantine")
    ]
    assert any("final_gate_reject" in reason for reason in reasons)
    host.stop()


# ----------------------------------------------------------------------- ids


def test_submission_id_is_stable_across_a_crash_retry(tmp_path: Path) -> None:
    first = _host(tmp_path)
    first.start()
    envelope = _envelope("remember me", provenance="retry")
    staged = first.submit_ingress(envelope)
    assert staged.status == "staged"
    checkpoint = first.save_checkpoint()
    first.stop()

    retried = _host(tmp_path)
    retried.start()
    retried.load_checkpoint(checkpoint)
    duplicate = retried.submit_ingress(envelope)
    assert duplicate.status == "duplicate"
    assert duplicate.submission_id == staged.submission_id

    result = retried.beat()
    assert len(result.commits) == 1
    assert result.commits[0].submission_id == staged.submission_id
    assert retried.committed_text("user_input") == "remember me"

    again = retried.submit_ingress(envelope)
    assert again.status == "duplicate"
    assert again.reason == "already_committed"
    assert retried.beat().commits == ()
    assert retried.committed_text("user_input") == "remember me"
    retried.stop()


def test_byte_identical_envelopes_dedupe_until_acknowledged(host) -> None:
    """The documented convention: same provenance is one arrival, retried."""

    first = host.submit_ingress(_envelope("same text", provenance="turn-1"))
    duplicate = host.submit_ingress(_envelope("same text", provenance="turn-1"))
    assert duplicate.status == "duplicate"
    assert duplicate.reason == "already_staged"
    assert duplicate.submission_id == first.submission_id

    # a different provenance marks a genuinely distinct arrival of the same text
    distinct = host.submit_ingress(_envelope("same text", provenance="turn-2"))
    assert distinct.status == "staged"
    assert distinct.submission_id != first.submission_id

    result = host.beat()
    assert len(result.commits) == 2
    assert host.committed_text("user_input") == "same textsame text"

    # after the ack, the byte-identical retry is still refused as committed
    again = host.submit_ingress(_envelope("same text", provenance="turn-1"))
    assert again.status == "duplicate"
    assert again.reason == "already_committed"
    assert host.beat().commits == ()
    assert host.committed_text("user_input") == "same textsame text"


def test_unacknowledged_commit_is_not_duplicated_on_restart(tmp_path: Path) -> None:
    host = _host(tmp_path)
    host.start()
    host.submit_ingress(_envelope("hello", provenance="ack-window"))
    host.beat()
    assert host.committed_text("user_input") == "hello"

    # Simulate a crash after the commit was durable but before the ack cursor
    # advanced: the spool tail is pending again while the journal has the commit.
    host.spool.ack_cursor_path.write_text(
        json.dumps({"last_event_id": "", "count": 0}), encoding="utf-8"
    )
    assert host.spool.pending_count == 1
    host.stop()

    restarted = _host(tmp_path)
    restarted.start()
    assert restarted.spool.pending_count == 0  # replay skipped it and acked it
    result = restarted.beat()
    assert result.commits == ()
    assert result.skipped_submissions == ()
    assert restarted.committed_text("user_input") == "hello"
    restarted.stop()


def test_applied_but_unlogged_head_commit_is_not_duplicated(tmp_path: Path) -> None:
    host = _host(tmp_path)
    host.start()
    host.submit_ingress(_envelope("hello", provenance="unlogged"))
    host.beat()
    journal = host.branch.journal_path
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # initialize + commit

    # Simulate the crash window inside CanonicalStateBranch.commit: the atomic
    # HEAD is durable but the audit append never happened, and the ack is lost.
    journal.write_text(lines[0] + "\n", encoding="utf-8")
    host.spool.ack_cursor_path.write_text(
        json.dumps({"last_event_id": "", "count": 0}), encoding="utf-8"
    )
    host.stop()

    restarted = _host(tmp_path)
    restarted.start()
    assert restarted.spool.pending_count == 0  # matched by its applied delta
    assert restarted.beat().commits == ()
    assert restarted.committed_text("user_input") == "hello"
    restarted.stop()


def test_committed_record_behind_uncommitted_fifo_fails_closed(tmp_path: Path) -> None:
    """The FIFO invariant: a committed record never follows an uncommitted one."""

    spool = DurableIngressSpool(tmp_path / "active" / "heart" / "ingress_spool")
    admitted = ValveDecision(True, "admitted_local", False, None)
    spool.submit(_envelope("alpha", provenance="fifo"), decision=admitted, valve_version=2)
    second = spool.submit(
        _envelope("beta", provenance="fifo"), decision=admitted, valve_version=2
    )

    branch = CanonicalStateBranch.active_runtime("active", state_root=tmp_path)
    branch.initialize(SharedFieldSnapshot.empty())
    head = branch.load_head_record()
    forged = {
        "schema": BRANCH_EVENT_SCHEMA,
        "branch_id": "active",
        "event": "commit",
        "generation": 0,
        "base_field_id": head.field_id,
        "delta_id": "f" * 64,
        "field_id": head.field_id,
        "tick_id": 0,
        "parent_field_id": head.parent_field_id,
        "metadata": {
            "committer_id": "core-a",
            "submission_id": submission_id_for_envelope(second, 2),
        },
    }
    with branch.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(forged, sort_keys=True) + "\n")

    host = _host(tmp_path)
    with pytest.raises(ReplayEventError, match="behind an uncommitted"):
        host.start()
    with pytest.raises(HostStateError):
        host.beat()  # a failed start leaves no writable host behind


# ------------------------------------------------------------- consolidators


def test_consolidator_rotation_and_commit_authority(tmp_path: Path) -> None:
    host = _host(tmp_path)
    host.start()
    assert host.current_consolidator() == "core-a"

    first = host.beat()
    assert first.consolidator_id == "core-a"
    assert host.current_consolidator() == "core-b"

    proposal_id = _proposal(host, "rotated")
    with pytest.raises(AuthorityViolationError):
        host.commit("core-a", proposal_id)
    ack = host.commit("core-b", proposal_id)
    assert ack.committer_id == "core-b"

    second = host.beat()
    assert second.consolidator_id == "core-b"
    assert host.current_consolidator() == "core-a"

    third = host.beat()
    assert third.consolidator_id == "core-a"
    assert host.current_consolidator() == "core-b"
    host.stop()


def test_proposals_commit_through_the_current_consolidator_in_the_beat(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    host.start()
    proposal_id = _proposal(host, "proposed")
    result = host.beat()
    assert len(result.commits) == 1
    assert result.commits[0].proposal_id == proposal_id
    assert result.commits[0].committer_id == "core-a"
    assert host.committed_text("scratch") == "proposed"
    host.stop()


# ------------------------------------------------------- controls and drafts


def test_wait_control_performs_zero_operations_and_zero_commits(host) -> None:
    proposal_id = host.submit_control("core-a", "wait")
    assert isinstance(proposal_id, str) and proposal_id

    result = host.beat()
    assert result.commits == ()
    assert host.generation == 0
    assert host.spool.pending_count == 0
    assert host.committed_text("user_input") == ""
    assert host.committed_text("response_draft") == ""

    events = host.drain_events()
    received = _events_of_type(events, "proposal_received")
    assert received[0]["payload"]["control"] == "wait"
    assert _events_of_type(events, "commit") == []


def test_end_control_fails_closed_until_episode_reset(host) -> None:
    host.submit_control("core-a", "end")
    result = host.beat()
    assert result.commits == ()

    with pytest.raises(HostStateError):
        host.submit_control("core-a", "wait")
    with pytest.raises(HostStateError):
        _proposal(host, "blocked")

    host.reset_episode()
    assert host.submit_control("core-a", "wait")


def test_three_surfaces_private_draft_numerical_state_and_committed_text(
    host,
) -> None:
    host.set_draft("core-a", "the composed answer")
    assert host.private_draft()["text"] == "the composed answer"
    assert host.committed_text("response_draft") == ""  # readable, not canonical

    proposal_id = _proposal(host, "note")
    host.commit(host.current_consolidator(), proposal_id)
    snapshot = host.snapshot()
    assert snapshot["pending_proposals"] == []

    host.set_draft("core-a", "second draft")
    pending = host.submit_proposal(
        "core-a",
        delta=replacement_delta(
            host.branch.load_head(),
            region="scratch",
            text="numerical",
            author_core_id="core-a",
            pass_id="state",
        ),
        response_state={"logits": [0.1, 0.8, 0.1]},
    )
    assert pending
    snapshot = host.snapshot()
    assert snapshot["pending_proposals"][0]["response_state"] == {
        "logits": [0.1, 0.8, 0.1]
    }
    assert host.committed_text("response_draft") == ""  # never canonical until committed

    host.submit_control("core-a", "commit")
    result = host.beat()
    draft_acks = [ack for ack in result.commits if ack.proposal_id is not None]
    assert draft_acks[-1].committed_text == "second draft"
    assert host.committed_text("response_draft") == "second draft"
    # the private draft surface stays inspectable after the commit
    assert host.private_draft()["text"] == "second draft"


def test_reset_clears_draft_and_cursors_but_keeps_committed_text(host) -> None:
    host.submit_ingress(_envelope("keep me", provenance="reset"))
    host.beat()
    host.set_draft("core-a", "private answer")
    proposal_id = _proposal(host, "note")
    host.commit(host.current_consolidator(), proposal_id)

    before = host.snapshot()
    assert before["draft"]["text"] == "private answer"
    assert before["cursors"] != {}

    summary = host.reset_episode()
    after = host.snapshot()

    assert summary["draft_cleared"] is True
    assert after["draft"] is None
    assert after["cursors"] == {}
    assert after["pending_proposals"] == []
    assert host.committed_text("user_input") == "keep me"
    assert host.committed_text("scratch") == "note"


# ------------------------------------------------------- checkpoints & replay


def test_checkpoint_resume_matches_uninterrupted_execution(tmp_path: Path) -> None:
    def run(root: Path, *, crash_mid_episode: bool) -> tuple[str, str]:
        host = _host(root)
        host.start()
        host.submit_ingress(_envelope("ab", provenance="turn-1"))
        host.beat()
        host.submit_ingress(_envelope("cd", provenance="turn-2"))
        if crash_mid_episode:
            checkpoint = host.save_checkpoint()
            host.stop()
            host = _host(root)
            host.start()
            host.load_checkpoint(checkpoint)
        result = host.beat()
        text = host.committed_text("user_input")
        commit_id = result.commits[0].commit_id
        host.stop()
        return text, commit_id

    resumed_text, resumed_commit = run(tmp_path / "resumed", crash_mid_episode=True)
    plain_text, plain_commit = run(tmp_path / "plain", crash_mid_episode=False)

    assert resumed_text == plain_text == "abcd"
    # the resumed delta is byte-identical to the uninterrupted one: no duplicate
    # canonical text and the same content-addressed commit identity
    assert resumed_commit == plain_commit


def test_draft_commit_resume_matches_uninterrupted_execution(tmp_path: Path) -> None:
    """Mid-output crash after a draft commit must not write a second generation."""

    def run(root: Path, *, crash_after_commit: bool) -> dict:
        host = _host(root, core_ids=("core-a",), consolidator_ids=("core-a",))
        host.start()
        host.set_draft("core-a", "the answer")
        proposal_id = host.submit_control("core-a", "commit")
        pre_commit = (
            host.save_checkpoint(root / "pre_commit.json") if crash_after_commit else None
        )
        first = host.beat()
        assert len(first.commits) == 1
        outcome = {
            "proposal_id": proposal_id,
            "commit_id": first.commits[0].commit_id,
            "generation": host.generation,
            "field_id": host.head_field_id,
            "text": host.committed_text("response_draft"),
        }
        if crash_after_commit:
            # crash between the draft commit and any newer checkpoint: only the
            # pre-commit checkpoint survives
            host.stop()
            resumed = _host(root, core_ids=("core-a",), consolidator_ids=("core-a",))
            resumed.start()
            report = resumed.load_checkpoint(pre_commit)
            result = resumed.beat()
            events = resumed.drain_events()

            assert report["behind_head"] is True
            assert result.commits == ()
            assert result.skipped_proposals == (proposal_id,)
            assert resumed.generation == outcome["generation"]
            assert resumed.head_field_id == outcome["field_id"]
            assert resumed.committed_text("response_draft") == outcome["text"]
            snapshot = resumed.snapshot()
            assert snapshot["last_commit_id"] == outcome["commit_id"]
            assert snapshot["cursors"]["response_draft"] == outcome["commit_id"]
            journal = [
                json.loads(line)
                for line in resumed.branch.journal_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            assert sum(1 for event in journal if event["event"] == "commit") == 1
            skipped_events = [
                item
                for item in _events_of_type(events, "commit")
                if item["payload"].get("skipped")
            ]
            assert [item["payload"]["proposal_id"] for item in skipped_events] == [
                proposal_id
            ]
            resumed.stop()
        else:
            host.stop()
        return outcome

    interrupted = run(tmp_path / "interrupted", crash_after_commit=True)
    uninterrupted = run(tmp_path / "uninterrupted", crash_after_commit=False)

    assert interrupted["generation"] == uninterrupted["generation"] == 1
    assert interrupted["commit_id"] == uninterrupted["commit_id"]
    assert interrupted["field_id"] == uninterrupted["field_id"]
    assert interrupted["text"] == uninterrupted["text"] == "the answer"


def test_commit_acknowledges_an_already_committed_proposal_without_a_new_write(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path, core_ids=("core-a",), consolidator_ids=("core-a",))
    host.start()
    host.set_draft("core-a", "answer")
    proposal_id = host.submit_control("core-a", "commit")
    pre_commit = host.save_checkpoint(tmp_path / "pre_commit.json")
    first = host.commit("core-a", proposal_id)
    assert first.generation == 1
    head_before = host.branch.load_head_record()

    # a stale checkpoint brings the same proposal back as a pending entry
    host.load_checkpoint(pre_commit)
    assert host.snapshot()["pending_proposals"][0]["proposal_id"] == proposal_id

    second = host.commit(host.current_consolidator(), proposal_id)
    assert second.commit_id == first.commit_id  # the durable identity, not a rewrite
    assert second.generation == 1
    assert host.branch.load_head_record() == head_before
    assert host.snapshot()["pending_proposals"] == []
    events = _events_of_type(host.drain_events(), "commit")
    assert any(
        item["payload"].get("skipped")
        and item["payload"]["proposal_id"] == proposal_id
        for item in events
    )
    host.stop()


def test_checkpoint_behind_head_derives_consistent_commit_state(tmp_path: Path) -> None:
    host = _host(tmp_path)
    host.start()
    host.submit_ingress(_envelope("first", provenance="behind"))
    first = host.beat()
    stale = host.save_checkpoint(tmp_path / "stale.json")
    host.submit_ingress(_envelope("second", provenance="behind"))
    second = host.beat()
    live_commit = second.commits[0].commit_id
    assert second.commits[0].generation == 2

    stale_doc = json.loads(stale.read_text(encoding="utf-8"))
    assert stale_doc["generation"] == 1
    assert stale_doc["cursors"]["user_input"] == first.commits[0].commit_id

    report = host.load_checkpoint(stale)
    assert report == {
        "behind_head": True,
        "checkpoint_generation": 1,
        "head_generation": 2,
        "derived_last_commit_id": live_commit,
        "derived_cursors": 1,
    }
    snapshot = host.snapshot()
    assert snapshot["generation"] == 2
    assert snapshot["last_commit_id"] == live_commit
    assert snapshot["cursors"]["user_input"] == live_commit  # not the stale cursor
    assert snapshot["last_restore"]["behind_head"] is True
    # loading never writes canonical state
    assert host.branch.load_head_record().generation == 2
    assert host.committed_text("user_input") == "firstsecond"
    host.stop()


def test_checkpoint_sidecar_is_verified_before_parsing(tmp_path: Path, host) -> None:
    checkpoint = host.save_checkpoint()
    sidecar = checkpoint.with_name(checkpoint.name + ".sha256")
    assert sidecar.exists()

    checkpoint.write_text('{"schema": "tampered"}', encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError):
        host.load_checkpoint(checkpoint)

    # A matching sidecar for a malformed body still fails closed on the schema.
    sidecar.write_text(
        hashlib.sha256(checkpoint.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointIntegrityError):
        host.load_checkpoint(checkpoint)

    sidecar.unlink()
    with pytest.raises(CheckpointIntegrityError):
        host.load_checkpoint(checkpoint)


def test_snapshot_replay_round_trip(tmp_path: Path) -> None:
    host = _host(tmp_path)
    host.start()
    host.submit_ingress(_envelope("first", provenance="snap"))
    host.beat()
    host.set_draft("core-a", "draft state")
    _proposal(host, "pending")
    host.submit_ingress(_envelope("second", provenance="snap"))

    snapshot = host.snapshot()
    assert snapshot["schema"] == CHECKPOINT_FORMAT
    assert json.loads(json.dumps(snapshot)) == snapshot  # JSON-serializable
    host.stop()

    fresh = _host(tmp_path)
    fresh.start()
    fresh.replay_from(snapshot)
    restored = fresh.snapshot()

    for key in (
        "generation",
        "last_commit_id",
        "epoch",
        "event_seq",
        "cursors",
        "draft",
        "episode_ended",
        "beat_count",
        "roster_index",
        "consolidator_id",
        "pending_submissions",
        "pending_proposals",
        "budget",
    ):
        assert restored[key] == snapshot[key], key
    assert restored["events"]["counters"] == snapshot["events"]["counters"]
    assert restored["pending_submissions"][0]["ack_state"] == "staged"
    fresh.stop()


def test_replay_from_reconciles_against_the_durable_spool(tmp_path: Path) -> None:
    spool = DurableIngressSpool(tmp_path / "active" / "heart" / "ingress_spool")
    spool.submit(
        _envelope("held", provenance="replay"),
        decision=ValveDecision(True, "admitted_local", False, None),
        valve_version=2,
    )
    host = _host(tmp_path)
    host.start()
    snapshot = host.snapshot()
    assert snapshot["pending_submissions"][0]["ack_state"] == "admitted"

    # forge a snapshot that claims the still-pending submission was committed;
    # replay must trust durable truth and reconcile it back
    forged = json.loads(json.dumps(snapshot))
    forged["pending_submissions"][0]["ack_state"] = "committed"
    forged["pending_submissions"][0]["commit_id"] = "f" * 64
    report = host.replay_from(forged)

    assert report["behind_head"] is False
    assert host.snapshot()["pending_submissions"][0]["ack_state"] == "admitted"
    host.stop()


# ------------------------------------------------------------------- failures


def test_invalid_proposal_outside_native95_fails_closed(host) -> None:
    head = host.branch.load_head()
    delta = replacement_delta(
        head,
        region="response_draft",
        text="curly \u2019 quote",
        author_core_id="core-a",
        pass_id="bad",
    )
    with pytest.raises(UnsupportedCharacterError):
        host.submit_proposal("core-a", delta=delta)
    assert host.generation == 0
    assert host.snapshot()["pending_proposals"] == []


def test_proposal_outside_core_authority_fails_closed(host) -> None:
    head = host.branch.load_head()
    delta = replacement_delta(
        head,
        region="user_input",
        text="forced",
        author_core_id="core-a",
        pass_id="bad",
    )
    with pytest.raises(AuthorityViolationError):
        host.submit_proposal("core-a", delta=delta)

    widened = AuthorityGrant.core(("identity",))
    with pytest.raises(AuthorityViolationError):
        host.submit_proposal("core-a", grant=widened, delta=delta)

    with pytest.raises(UnknownCoreError):
        host.submit_control("not-registered", "wait")
    with pytest.raises(DuplicateCoreError):
        host.register_core("core-a")


def test_stale_proposal_fails_closed(host) -> None:
    stale_head = host.branch.load_head()
    host.submit_ingress(_envelope("progress", provenance="stale"))
    host.beat()
    delta = replacement_delta(
        stale_head,
        region="scratch",
        text="stale",
        author_core_id="core-a",
        pass_id="stale",
    )
    with pytest.raises(StaleBaseProposalError):
        host.submit_proposal("core-a", delta=delta)


def test_stale_pending_proposal_fails_the_beat_closed_and_reset_recovers(
    host,
) -> None:
    head = host.branch.load_head()
    first = host.submit_proposal(
        "core-a",
        delta=replacement_delta(
            head, region="scratch", text="one", author_core_id="core-a", pass_id="p1"
        ),
    )
    second = host.submit_proposal(
        "core-a",
        delta=replacement_delta(
            head, region="scratch", text="two", author_core_id="core-a", pass_id="p2"
        ),
    )
    assert first != second

    with pytest.raises(StaleDeltaError):
        host.beat()
    # the first proposal committed exactly once; the stale one never did
    assert host.committed_text("scratch") == "one"
    assert host.generation == 1
    assert host.health_latest()["last_failure_reason"].startswith("beat_failed")

    # fail closed until the operator clears the episode; then circulation resumes
    with pytest.raises(StaleDeltaError):
        host.beat()
    host.reset_episode()
    assert host.beat().commits == ()
    assert host.committed_text("scratch") == "one"


# --------------------------------------------------------------------- events


def test_event_log_is_bounded_per_type_and_counters_persist(host) -> None:
    for revision in range(EVENT_LOG_CAP + 5):
        host.set_draft("core-a", f"draft {revision}")
        host.submit_control("core-a", "commit")

    counters = host.event_counters()
    assert counters["proposal_received"] == EVENT_LOG_CAP + 5

    drained = host.drain_events()
    assert len(_events_of_type(drained, "proposal_received")) == EVENT_LOG_CAP
    assert host.drain_events() == []
    assert host.event_counters()["proposal_received"] == EVENT_LOG_CAP + 5
    # the global ordinal kept counting past the per-type retention cap: the last
    # retained seq equals the total number of events ever emitted in this epoch
    assert drained[-1]["seq"] == max(event["seq"] for event in drained)
    assert drained[-1]["seq"] == sum(host.event_counters().values())


def test_event_types_are_exactly_the_contract_set() -> None:
    assert EVENT_TYPES == {
        "beat_start",
        "beat_end",
        "valve_admission",
        "valve_reject",
        "cap_exhausted",
        "quarantine",
        "proposal_received",
        "commit",
        "ack",
        "budget_remaining",
        "lease_state",
    }


def test_every_lab_event_type_emits_in_its_own_scenario(tmp_path: Path) -> None:
    host = HeartHost(
        tmp_path,
        core_ids=("core-a",),
        consolidator_ids=("core-a",),
        valves=_tiny_registry(items_per_beat=1),
    )
    host.start()
    host.submit_ingress(_envelope("one", provenance="events"))
    host.submit_ingress(_envelope("two", provenance="events"))
    host.submit_ingress(_envelope("bad \u2019", provenance="events"))  # valve_reject
    host.submit_control("core-a", "wait")  # proposal_received
    first = host.beat()  # one admission, one cap_exhausted deferral
    second = host.beat()  # the deferred item
    host.stop()

    assert len(first.commits) == 1
    assert len(first.deferred_submissions) == 1
    assert len(second.commits) == 1

    events = host.drain_events()
    emitted = {event["type"] for event in events}
    assert emitted == EVENT_TYPES
    assert len(_events_of_type(events, "beat_start")) == 2
    assert len(_events_of_type(events, "beat_end")) == 2
    assert len(_events_of_type(events, "valve_admission")) == 2
    assert len(_events_of_type(events, "valve_reject")) == 1
    assert len(_events_of_type(events, "quarantine")) == 1
    cap_events = _events_of_type(events, "cap_exhausted")
    assert len(cap_events) == 1
    assert cap_events[0]["payload"]["valve_id"] == "user_ingress"
    assert len(_events_of_type(events, "proposal_received")) == 1
    assert len(_events_of_type(events, "commit")) == 2
    assert len(_events_of_type(events, "ack")) == 2
    assert _events_of_type(events, "budget_remaining")
    assert _events_of_type(events, "lease_state")[0]["payload"]["held"] is True
    counters = host.event_counters()
    assert all(counters[name] >= 1 for name in EVENT_TYPES)


def test_event_ordinal_is_globally_increasing_across_types(host) -> None:
    """One ordinal per epoch, shared by every event type, starting at 1."""

    host.submit_ingress(_envelope("ordinal", provenance="seq"))
    host.submit_ingress(_envelope("reject \u2019", provenance="seq"))
    host.submit_control("core-a", "wait")
    host.beat()

    events = host.drain_events()
    seqs = [event["seq"] for event in events]
    assert seqs == sorted(seqs)
    assert seqs == list(range(1, len(events) + 1))
    assert len({event["type"] for event in events}) > 1  # mixed types, one ordinal
    epoch = host.snapshot()["epoch"]
    assert all(event["epoch"] == epoch for event in events)
    assert host.snapshot()["event_seq"] == len(events)
    assert host.drain_events() == []


def test_epoch_is_new_on_fresh_start_and_continues_through_checkpoints(
    tmp_path: Path,
) -> None:
    first = _host(tmp_path)
    first.start()
    first.submit_ingress(_envelope("epoch", provenance="epoch"))
    first.beat()
    first_epoch = first.snapshot()["epoch"]
    checkpoint = first.save_checkpoint(tmp_path / "epoch.json")
    first_seq = first.snapshot()["event_seq"]
    assert first_epoch.startswith("hearthost-")
    first.stop()

    resumed = _host(tmp_path)
    resumed.start()
    assert resumed.snapshot()["epoch"] != first_epoch  # a fresh start is a new epoch
    report = resumed.load_checkpoint(checkpoint)
    assert report["behind_head"] is False
    snapshot = resumed.snapshot()
    assert snapshot["epoch"] == first_epoch  # the checkpoint's epoch is continued
    assert snapshot["event_seq"] == first_seq
    resumed.stop()

    # replay_from continues the epoch too, and the ordinal never regresses
    replayed = _host(tmp_path)
    replayed.start()
    pre_restore = replayed.drain_events()  # the fresh start's own epoch
    assert pre_restore[0]["epoch"] != first_epoch
    report = replayed.replay_from(snapshot)
    assert report["behind_head"] is False
    assert replayed.snapshot()["epoch"] == first_epoch
    assert replayed.snapshot()["event_seq"] == first_seq
    replayed.submit_ingress(_envelope("after", provenance="epoch"))
    replayed.beat()
    assert replayed.snapshot()["event_seq"] > first_seq
    restored_epoch_events = replayed.drain_events()
    assert restored_epoch_events
    assert all(event["epoch"] == first_epoch for event in restored_epoch_events)
    assert restored_epoch_events[0]["seq"] == first_seq + 1
    replayed.stop()


def test_drain_events_returns_one_seq_ordered_list(host) -> None:
    host.submit_ingress(_envelope("list", provenance="drain"))
    host.beat()
    drained = host.drain_events()

    assert isinstance(drained, list)
    assert all(isinstance(event, dict) for event in drained)
    assert {"seq", "epoch", "at", "type", "payload"} == set(drained[0])
    assert [event["seq"] for event in drained] == sorted(
        event["seq"] for event in drained
    )
    assert host.drain_events() == []


def test_session_api_propose_advance_inspect_matches_the_explicit_path(
    tmp_path: Path,
) -> None:
    session = _host(tmp_path / "session")
    session.start()
    session.submit_ingress(_envelope("same path", provenance="session"))
    session.set_draft("core-a", "session answer")
    session.propose("core-a", commit_draft=True)
    result = session.advance()
    snapshot = session.inspect()

    assert len(result.commits) == 2  # staged ingress plus the private-draft commit
    assert session.committed_text("user_input") == "same path"
    assert session.committed_text("response_draft") == "session answer"
    assert snapshot["generation"] == 2
    assert snapshot["schema"] == CHECKPOINT_FORMAT
    assert snapshot["lease"]["held"] is True
    session_user_input = session.committed_text("user_input")
    session_response = session.committed_text("response_draft")
    session.stop()

    explicit = _host(tmp_path / "explicit")
    explicit.start()
    explicit.submit_ingress(_envelope("same path", provenance="session"))
    explicit.set_draft("core-a", "session answer")
    explicit.submit_control("core-a", "commit")
    explicit.beat()

    assert explicit.committed_text("user_input") == session_user_input
    assert explicit.committed_text("response_draft") == session_response
    explicit.stop()


def test_health_and_identity_journals_are_written(host) -> None:
    host.submit_ingress(_envelope("beat", provenance="health"))
    host.beat()

    latest = host.health_latest()
    assert latest["schema"] == "axon-heart-health-v2"
    assert latest["canonical_head_tick_id"] == 1
    assert latest["last_failure_reason"] is None
    assert latest["lease_owner_token"] == host.lease.owner_token
    assert latest["queue_depth_by_valve"]["user_ingress"] == 0
    identity = json.loads(
        (host.heart_dir / "identity.json").read_text(encoding="utf-8")
    )
    assert identity["heartbeat_sequence"] == 1
    assert identity["tick_sequence"] == 1


def test_episode_initial_field_is_preserved(tmp_path: Path) -> None:
    initial = SharedFieldSnapshot.from_texts({"cortex": "seeded memory"}, tick_id=3)
    host = _host(tmp_path, initial_field=initial)
    host.start()
    assert host.branch.load_head().field_id == initial.field_id
    assert host.committed_text("cortex") == "seeded memory"
    host.submit_ingress(_envelope("after seed", provenance="seed"))
    host.beat()
    assert host.committed_text("cortex") == "seeded memory"
    assert host.committed_text("user_input") == "after seed"
    host.stop()

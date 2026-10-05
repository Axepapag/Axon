from __future__ import annotations

from pathlib import Path

import pytest

from runtime.heart import (
    DurableIngressSpool,
    ReplayEventError,
    ValveDecision,
    ValveEnvelope,
)


def _envelope(text: str) -> ValveEnvelope:
    return ValveEnvelope(
        valve_id="user_ingress",
        source_id="external_user",
        payload=text,
        provenance="test",
        envelope_type="text/plain",
    )


def _accept() -> ValveDecision:
    return ValveDecision(True, "admitted_local", False, None)


def test_submit_pending_ack_cycle(tmp_path: Path) -> None:
    spool = DurableIngressSpool(tmp_path)
    record = spool.submit(_envelope("hello"), decision=_accept(), valve_version=1)
    assert spool.pending() == (record,)
    assert spool.pending_count == 1
    spool.acknowledge(record.event_id)
    assert spool.pending() == ()
    assert spool.pending_count == 0
    # A second acknowledgement is an idempotent recovery operation.
    spool.acknowledge(record.event_id)


def test_pending_survives_restart(tmp_path: Path) -> None:
    first = DurableIngressSpool(tmp_path)
    record = first.submit(_envelope("survive"), decision=_accept(), valve_version=1)
    second = DurableIngressSpool(tmp_path)
    assert second.recover() == (record,)


def test_fifo_ack_cannot_skip_predecessor(tmp_path: Path) -> None:
    spool = DurableIngressSpool(tmp_path)
    first = spool.submit(_envelope("first"), decision=_accept(), valve_version=1)
    second = spool.submit(_envelope("second"), decision=_accept(), valve_version=1)
    with pytest.raises(ReplayEventError):
        spool.acknowledge(second.event_id)
    spool.acknowledge(first.event_id)
    spool.acknowledge(second.event_id)
    assert spool.pending() == ()


def test_duplicate_event_id_in_unacked_tail_fails_closed(tmp_path: Path) -> None:
    spool = DurableIngressSpool(tmp_path)
    spool.submit(_envelope("once"), decision=_accept(), valve_version=1)
    line = spool.journal_path.read_text(encoding="utf-8").strip()
    with spool.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
    with pytest.raises(ReplayEventError, match="duplicate event_id"):
        spool.recover()


def test_rejected_poison_is_durably_quarantined_without_blocking_fifo(tmp_path: Path) -> None:
    spool = DurableIngressSpool(tmp_path)
    bad = _envelope("bad")
    rejected = ValveDecision(False, "oversize", True, None)
    spool.reject_envelope(bad, reason=rejected.reason, valve_version=1, quarantine=True)
    good = spool.submit(_envelope("ok"), decision=_accept(), valve_version=1)
    assert spool.quarantine_count == 1
    assert spool.rejection_count == 1
    assert spool.pending() == (good,)

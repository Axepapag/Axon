from __future__ import annotations

from pathlib import Path

import pytest

from runtime.heart import LeaseDeniedError, SingleWriterLease


def test_lease_can_be_acquired(tmp_path: Path) -> None:
    lease = SingleWriterLease(tmp_path)
    record = lease.acquire()

    assert record["owner_token"] == lease.owner_token
    assert record["pid"] > 0
    assert lease.is_held_by_us()
    assert lease.lease_path.exists()


def test_second_acquire_by_different_owner_token_fails(tmp_path: Path) -> None:
    first = SingleWriterLease(tmp_path, owner_token="first-token")
    first.acquire()

    second = SingleWriterLease(tmp_path, owner_token="second-token")
    with pytest.raises(LeaseDeniedError):
        second.acquire()

    first.release()


def test_release_allows_reacquire(tmp_path: Path) -> None:
    first = SingleWriterLease(tmp_path, owner_token="token-a")
    first.acquire()
    assert first.is_held_by_us()

    first.release()
    assert not first.is_held_by_us()
    assert not first.lease_path.exists()

    second = SingleWriterLease(tmp_path, owner_token="token-b")
    second.acquire()
    assert second.is_held_by_us()


def test_corrupt_lease_file_fails_closed(tmp_path: Path) -> None:
    lease_path = SingleWriterLease(tmp_path).lease_path
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text("not-json", encoding="utf-8")

    lease = SingleWriterLease(tmp_path, owner_token="recovery-token")
    with pytest.raises(LeaseDeniedError):
        lease.acquire()

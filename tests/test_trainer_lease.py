from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from torch import nn

from runtime.trainer import (
    OrganKind,
    ParameterModuleDescriptor,
    TrainerControlPlane,
    TrainerLeaseDeniedError,
    TrainerWriterLease,
)


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_writer_lease_is_single_owner_and_recoverable(tmp_path: Path) -> None:
    first = TrainerWriterLease(tmp_path, owner_token="first")
    first.acquire()
    assert first.is_held_by_us()
    second = TrainerWriterLease(tmp_path, owner_token="second")
    with pytest.raises(TrainerLeaseDeniedError):
        second.acquire()
    first.release()
    second.acquire()
    assert second.is_held_by_us()
    second.release()


def test_corrupt_trainer_lease_metadata_fails_closed(tmp_path: Path) -> None:
    lease = TrainerWriterLease(tmp_path, owner_token="owner")
    lease.lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease.lease_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(TrainerLeaseDeniedError, match="corrupt or unreadable"):
        lease.acquire()


def test_active_control_plane_requires_its_writer_lease(tmp_path: Path) -> None:
    control = TrainerControlPlane.active(state_root=tmp_path, acquire_lease=False)
    descriptor = ParameterModuleDescriptor(
        module_id="core",
        organ_kind=OrganKind.REASONING_CORE,
        generation_id="g0",
        architecture="linear",
        d_model=4,
    )
    control.declare_expected((descriptor,))
    control.register(descriptor, nn.Linear(4, 4, bias=False))
    with pytest.raises(TrainerLeaseDeniedError, match="lease is not held"):
        control.snapshot_inventory()
    control.lease.acquire()
    assert control.snapshot_inventory().complete
    control.close()


def test_trainer_writer_lease_blocks_a_second_process(tmp_path: Path) -> None:
    script = r'''
import sys
from runtime.trainer import TrainerWriterLease
lease = TrainerWriterLease(sys.argv[1], owner_token="child")
lease.acquire()
print("READY", flush=True)
sys.stdin.readline()
lease.release()
'''
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        contender = TrainerWriterLease(tmp_path, owner_token="parent")
        with pytest.raises(TrainerLeaseDeniedError):
            contender.acquire()
    finally:
        if process.stdin is not None:
            process.stdin.write("release\n")
            process.stdin.flush()
        process.communicate(timeout=15)
    assert process.returncode == 0

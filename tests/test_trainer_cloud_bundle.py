"""Hash-manifested cloud bundles, mid-run sync cadence, and bundle fetch.

All Kaggle CLI/API surfaces are mocked; nothing here touches the live account.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path

import pytest

from runtime.trainer import cloud_bundle
from runtime.trainer.cloud_bundle import (
    SYNC_PAYLOAD_KEEP,
    CloudBundleError,
    KaggleDatasetUploader,
    MidRunSyncHook,
    SyncCredentialsMissing,
    parse_sync_step_range,
    prune_sync_payloads,
    verify_and_extract,
    write_bundle,
)
from runtime.trainer.cloud_jobs import (
    CLOUD_JOB_RECORD_SCHEMA,
    CloudJobConfig,
    CloudPacketError,
    write_job_record,
)
from runtime.trainer.kaggle_adapter import KaggleTrainerAdapter, _runner_source
from scripts.axon_training_watch import Watcher, apply_sync_pull

# 32 hex chars keep paths short enough for deep pytest temp dirs on Windows
# (real job ids are 64 chars; the logic is length-independent).
JOB_ID = "ab" * 16


def _members(root: Path, names: dict[str, str]) -> list[tuple[str, Path]]:
    members = []
    for arcname, text in names.items():
        path = root / arcname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        members.append((arcname, path))
    return members


def _write_output_bundle(
    folder: Path,
    job_id: str,
    tree: dict[str, str],
    *,
    result_job_id: str | None = None,
) -> dict[str, str]:
    """Build the two files a completed kernel publishes, as bytes."""
    source = folder / "tree"
    members = _members(source, tree)
    result = {"schema": "axon-cloud-training-result-v1", "job_id": result_job_id or job_id, "status": "completed"}
    result_path = source / "axon_job_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    members.append(("axon_job_result.json", result_path))
    bundle = folder / f"axon_outputs_{job_id}.tar.gz"
    manifest = folder / f"axon_outputs_{job_id}.sha256.json"
    write_bundle(bundle, manifest, members, kind="outputs", job_id=job_id)
    return {bundle.name: bundle, manifest.name: manifest}


class _FakeKaggle:
    """Serves canned files for kernels-output / datasets-download commands."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.pattern_files: dict[str, Path] = {}
        self.full_files: dict[str, str] = {}
        self.download_files: dict[str, Path] = {}

    def __call__(self, argv, *, cwd=None, capture_output=True):
        command = tuple(str(item) for item in argv)
        self.calls.append(command)
        if command[:2] == ("kaggle", "--version"):
            return subprocess.CompletedProcess(command, 0, stdout="Kaggle API 2.2.4\n", stderr="")
        if command[:2] == ("kaggle", "quota"):
            return subprocess.CompletedProcess(command, 0, stdout="[]\n", stderr="")
        if command[:3] == ("kaggle", "kernels", "output"):
            target = Path(command[command.index("-p") + 1])
            target.mkdir(parents=True, exist_ok=True)
            if "--file-pattern" in command:
                for name, source in self.pattern_files.items():
                    shutil.copy2(source, target / name)
            else:
                for relative, text in self.full_files.items():
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(text, encoding="utf-8")
        elif command[:3] == ("kaggle", "datasets", "download"):
            target = Path(command[command.index("-p") + 1])
            target.mkdir(parents=True, exist_ok=True)
            for name, source in self.download_files.items():
                shutil.copy2(source, target / name)
        elif command[:3] == ("kaggle", "config", "view"):
            return subprocess.CompletedProcess(command, 0, stdout="- username: axongliksbot\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")


def _job(state: Path, job_id: str = JOB_ID, **extra) -> Path:
    job_dir = state / "training" / "cloud" / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": CLOUD_JOB_RECORD_SCHEMA,
        "job_id": job_id,
        "provider": "kaggle",
        "accelerator": "gpu",
        "phase": "submitted",
        "kernel_ref": f"axongliksbot/axon-job-{job_id[:16]}",
        "public": False,
        **extra,
    }
    write_job_record(state, job_id, record)
    return job_dir


def _adapter(tmp_path: Path, state: Path, runner) -> KaggleTrainerAdapter:
    return KaggleTrainerAdapter(repo_root=tmp_path, state_root=state, owner="axongliksbot", runner=runner)


# ── bundle writer / verifier ────────────────────────────────────────────────


def test_bundle_roundtrip_verifies_extracts_and_is_deterministic(tmp_path) -> None:
    source = tmp_path / "source"
    members = _members(
        source,
        {
            "axon_observability/trainer/events.jsonl": "{\"a\":1}\n",
            "State/training/水/segment.json": "{\"unicode\":\"水🙂\"}\n",
        },
    )
    bundle = tmp_path / "out.tar.gz"
    manifest = tmp_path / "out.sha256.json"
    document = write_bundle(bundle, manifest, members, kind="outputs", job_id=JOB_ID)
    again = write_bundle(tmp_path / "again.tar.gz", tmp_path / "again.sha256.json", members, kind="outputs", job_id=JOB_ID)
    assert document["archive_sha256"] == again["archive_sha256"]
    assert document["archive_bytes"] == (tmp_path / "again.tar.gz").stat().st_size

    destination = tmp_path / "extracted"
    report = verify_and_extract(bundle, manifest, destination, quarantine_root=tmp_path / "quarantine")
    assert report["ok"] is True
    assert report["job_id"] == JOB_ID
    assert report["member_count"] == 2
    assert (destination / "State" / "training" / "水" / "segment.json").read_text(
        encoding="utf-8"
    ) == '{"unicode":"水🙂"}\n'
    assert not (tmp_path / "quarantine").exists()


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended-path regression")
def test_bundle_extracts_without_shortening_a_path_beyond_legacy_win32_limit(
    tmp_path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"ancestry":"exact"}\n', encoding="utf-8")
    arcname = "/".join(
        (
            "axon_job",
            "State",
            "training",
            "soul_candidates",
            "r64v2-" + "a" * 56,
            "reasoning-d64-" + "b" * 52,
            "branch",
            "prepared",
            "c" * 64 + ".json",
        )
    )
    bundle = tmp_path / "long.tar.gz"
    manifest = tmp_path / "long.sha256.json"
    write_bundle(bundle, manifest, [(arcname, source)], kind="outputs", job_id=JOB_ID)
    destination = tmp_path / "extracted"

    report = verify_and_extract(
        bundle,
        manifest,
        destination,
        quarantine_root=tmp_path / "quarantine",
    )

    logical_target = destination / Path(*arcname.split("/"))
    assert len(str(logical_target.resolve(strict=False))) > 260
    assert report["ok"] is True
    assert cloud_bundle._io_path(logical_target).read_text(encoding="utf-8") == (
        '{"ancestry":"exact"}\n'
    )


def test_planted_manifest_corruption_is_quarantined_not_skipped(tmp_path) -> None:
    source = tmp_path / "source"
    members = _members(source, {"a.txt": "alpha\n", "nested/b.txt": "bravo\n"})
    bundle = tmp_path / "out.tar.gz"
    manifest = tmp_path / "out.sha256.json"
    write_bundle(bundle, manifest, members, kind="outputs", job_id=JOB_ID)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["members"]["a.txt"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")

    destination = tmp_path / "extracted"
    report = verify_and_extract(bundle, manifest, destination, quarantine_root=tmp_path / "quarantine")
    assert report["ok"] is False
    assert any("a.txt" in item for item in report["mismatches"])
    stall = Path(report["quarantine_dir"])
    assert (stall / "members" / "a.txt").is_file()
    assert (stall / bundle.name).is_file()
    # Nothing half-verified remains in the destination.
    assert not (destination / "a.txt").exists()


def test_planted_archive_corruption_blocks_extraction(tmp_path) -> None:
    source = tmp_path / "source"
    members = _members(source, {"a.txt": "alpha\n"})
    bundle = tmp_path / "out.tar.gz"
    manifest = tmp_path / "out.sha256.json"
    write_bundle(bundle, manifest, members, kind="outputs", job_id=JOB_ID)
    body = bytearray(bundle.read_bytes())
    body[-10] ^= 0xFF
    bundle.write_bytes(bytes(body))

    destination = tmp_path / "extracted"
    report = verify_and_extract(bundle, manifest, destination, quarantine_root=tmp_path / "quarantine")
    assert report["ok"] is False
    assert report["extracted_members"] == []
    assert Path(report["quarantine_dir"]).is_dir()
    assert not destination.exists() or not list(destination.rglob("*"))


def test_archive_member_missing_from_manifest_is_rejected(tmp_path) -> None:
    source = tmp_path / "source"
    (members_src,) = [_members(source, {"a.txt": "alpha\n"})]
    bundle = tmp_path / "out.tar.gz"
    manifest = tmp_path / "out.sha256.json"
    write_bundle(bundle, manifest, members_src, kind="outputs", job_id=JOB_ID)
    # Graft an unmanifested member into the archive.
    raw = gzip.decompress(bundle.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as existing:
        present = {info.name: existing.extractfile(info).read() for info in existing if info.isfile()}
    present["evil.txt"] = b"surprise\n"
    with (
        bundle.open("wb") as handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, data in present.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    # Archive now differs from the manifest hash AND carries an extra member.
    report = verify_and_extract(bundle, manifest, tmp_path / "out", quarantine_root=tmp_path / "q")
    assert report["ok"] is False
    assert any("archive sha256" in item for item in report["mismatches"])


def test_unsafe_member_paths_are_refused() -> None:
    with pytest.raises(CloudBundleError, match="unsafe"):
        cloud_bundle._checked_arcname("../escape.txt")
    with pytest.raises(CloudBundleError, match="unsafe"):
        cloud_bundle._checked_arcname("/absolute.txt")


def test_empty_bundle_is_refused(tmp_path) -> None:
    with pytest.raises(CloudBundleError, match="empty"):
        write_bundle(tmp_path / "x.tar.gz", tmp_path / "x.sha256.json", [], kind="outputs", job_id=JOB_ID)


def test_sync_bundle_name_round_trip() -> None:
    name = cloud_bundle.sync_bundle_name(JOB_ID, 361, 390)
    assert parse_sync_step_range(name) == (361, 390)
    assert parse_sync_step_range(name.replace(".tar.gz", ".sha256.json")) == (361, 390)
    assert parse_sync_step_range("random_file.txt") is None


# ── mid-run sync hook ───────────────────────────────────────────────────────


class _RecordingUploader:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.uploads: list[dict] = []

    def upload(self, folder, *, version_notes: str) -> None:
        folder = Path(folder)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("provider 500")
        manifests = sorted(folder.glob("*.sha256.json"))
        self.uploads.append(
            {
                "notes": version_notes,
                "files": sorted(path.name for path in folder.iterdir()),
                "manifest": json.loads(manifests[0].read_text(encoding="utf-8")) if manifests else None,
            }
        )


def _receipts(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_sync_hook_uploads_each_checkpoint_boundary_range(tmp_path) -> None:
    state = tmp_path / "State"
    state.mkdir()
    (state / "baseline.json").write_text("{}\n", encoding="utf-8")
    receipt_log = tmp_path / "obs" / "sync_receipts.jsonl"
    uploader = _RecordingUploader()
    hook = MidRunSyncHook(
        job_id=JOB_ID,
        dataset_slug="axon-job-abababab-sync",
        state_root=state,
        staging_root=tmp_path / "staging",
        receipt_log=receipt_log,
        uploader=uploader,
    )
    hook.set_base_step(360)
    (state / "checkpoints").mkdir()
    (state / "checkpoints" / "c390.pt").write_bytes(b"checkpoint-390")
    hook.boundary(390)
    hook.flush()
    (state / "checkpoints" / "c420.pt").write_bytes(b"checkpoint-420")
    hook.boundary(420)
    hook.close()

    assert [upload["notes"] for upload in uploader.uploads] == ["steps 361-390", "steps 391-420"]
    first = uploader.uploads[0]
    assert f"sync_{JOB_ID}_steps_361_390.tar.gz" in first["files"]
    assert f"sync_{JOB_ID}_steps_361_390.sha256.json" in first["files"]
    assert sorted(first["manifest"]["members"]) == ["checkpoints/c390.pt"]
    assert first["manifest"]["job_id"] == JOB_ID
    assert first["manifest"]["details"]["step_range"] == [361, 390]
    # The pre-existing baseline file is never bundled.
    assert "baseline.json" not in first["manifest"]["members"]
    statuses = [receipt["status"] for receipt in _receipts(receipt_log)]
    assert statuses == ["uploaded", "uploaded"]
    assert sorted(uploader.uploads[1]["manifest"]["members"]) == ["checkpoints/c420.pt"]
    # Staging is cleaned after each successful upload.
    assert not (tmp_path / "staging").exists() or not list((tmp_path / "staging").iterdir())


def test_failed_upload_is_receipted_and_retried_with_extended_range(tmp_path) -> None:
    state = tmp_path / "State"
    state.mkdir()
    receipt_log = tmp_path / "obs" / "sync_receipts.jsonl"
    uploader = _RecordingUploader(failures=1)
    hook = MidRunSyncHook(
        job_id=JOB_ID,
        dataset_slug="slug",
        state_root=state,
        staging_root=tmp_path / "staging",
        receipt_log=receipt_log,
        uploader=uploader,
    )
    hook.set_base_step(0)
    (state / "a.pt").write_bytes(b"a")
    hook.boundary(30)
    hook.flush()
    (state / "b.pt").write_bytes(b"b")
    hook.boundary(60)
    hook.close()

    receipts = _receipts(receipt_log)
    assert [receipt["status"] for receipt in receipts] == ["failed", "uploaded"]
    assert receipts[0]["details"]["step_range"] == [1, 30]
    assert receipts[0]["details"]["error_type"] == "RuntimeError"
    # The retry at the next boundary covers the whole un-uploaded range.
    assert len(uploader.uploads) == 1
    assert uploader.uploads[0]["notes"] == "steps 1-60"
    assert sorted(uploader.uploads[0]["manifest"]["members"]) == ["a.pt", "b.pt"]


def test_slow_upload_cannot_mislabel_a_later_boundary_artifact(tmp_path) -> None:
    class DelayedUploader(_RecordingUploader):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def upload(self, folder, *, version_notes: str) -> None:
            if not self.uploads:
                self.started.set()
                assert self.release.wait(timeout=5.0)
            super().upload(folder, version_notes=version_notes)

    state = tmp_path / "State"
    state.mkdir()
    uploader = DelayedUploader()
    hook = MidRunSyncHook(
        job_id=JOB_ID,
        dataset_slug="slug",
        state_root=state,
        staging_root=tmp_path / "staging",
        uploader=uploader,
    )
    hook.set_base_step(0)
    (state / "c30.pt").write_bytes(b"checkpoint-30")
    hook.boundary(30)
    assert uploader.started.wait(timeout=5.0)
    (state / "c60.pt").write_bytes(b"checkpoint-60")
    hook.boundary(60)
    uploader.release.set()
    hook.close()

    assert [upload["notes"] for upload in uploader.uploads] == ["steps 1-30", "steps 31-60"]
    assert sorted(uploader.uploads[0]["manifest"]["members"]) == ["c30.pt"]
    assert sorted(uploader.uploads[1]["manifest"]["members"]) == ["c60.pt"]


def test_no_secret_noop_mode_runs_training_path_with_one_note(tmp_path, capsys) -> None:
    state = tmp_path / "State"
    state.mkdir()
    receipt_log = tmp_path / "obs" / "sync_receipts.jsonl"
    hook = MidRunSyncHook.from_environment(
        job_id=JOB_ID,
        state_root=state,
        staging_root=tmp_path / "staging",
        receipt_log=receipt_log,
        environ={},  # no AXON_SYNC_* injected: the no-secret packet path
    )
    assert hook.enabled is False
    hook.set_base_step(0)
    (state / "a.pt").write_bytes(b"a")
    hook.boundary(30)  # no-op, must not raise
    hook.close()
    receipts = _receipts(receipt_log)
    assert len(receipts) == 1
    assert receipts[0]["status"] == "disabled"
    # Local stdout stays pure JSON for harnesses that parse it.
    assert "AXON_SYNC" not in capsys.readouterr().out


def test_missing_credentials_disable_sync_without_failing_training(tmp_path) -> None:
    class CredentiallessUploader:
        def upload(self, folder, *, version_notes: str) -> None:
            raise SyncCredentialsMissing("no Kaggle credential env and no User Secrets client")

    state = tmp_path / "State"
    state.mkdir()
    receipt_log = tmp_path / "obs" / "sync_receipts.jsonl"
    hook = MidRunSyncHook.from_environment(
        job_id=JOB_ID,
        state_root=state,
        staging_root=tmp_path / "staging",
        receipt_log=receipt_log,
        environ={"AXON_SYNC_MID_RUN": "1", "AXON_SYNC_DATASET": "slug"},
        uploader=CredentiallessUploader(),
    )
    assert hook.enabled is True
    hook.set_base_step(0)
    (state / "a.pt").write_bytes(b"a")
    hook.boundary(30)
    hook.close()
    assert hook.enabled is False
    statuses = [receipt["status"] for receipt in _receipts(receipt_log)]
    assert statuses == ["disabled"]


def test_secret_env_is_never_printed_or_receipted(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "sync-account")
    monkeypatch.setenv("KAGGLE_KEY", "SECRET-TOKEN-VALUE")
    uploader = KaggleDatasetUploader("slug")
    assert uploader._resolve_credentials() == "sync-account"
    state = tmp_path / "State"
    state.mkdir()
    receipt_log = tmp_path / "obs" / "sync_receipts.jsonl"
    hook = MidRunSyncHook(
        job_id=JOB_ID,
        dataset_slug="slug",
        state_root=state,
        staging_root=tmp_path / "staging",
        receipt_log=receipt_log,
        uploader=_RecordingUploader(),
    )
    hook.note_disabled("test note")
    output = capsys.readouterr().out
    assert "SECRET-TOKEN-VALUE" not in output
    assert "SECRET-TOKEN-VALUE" not in receipt_log.read_text(encoding="utf-8")


# ── adapter: fetch bundle path, legacy fallback, sync pull/status ───────────


def test_fetch_bundle_path_verifies_and_places_outputs(tmp_path) -> None:
    state = tmp_path / "State"
    _job(state)
    remote = tmp_path / "remote"
    remote.mkdir()
    files = _write_output_bundle(
        remote,
        JOB_ID,
        {"axon_observability/trainer/events.jsonl": "{\"schema\":\"x\"}\n"},
    )
    runner = _FakeKaggle()
    runner.pattern_files = files
    adapter = _adapter(tmp_path, state, runner)
    record = adapter.fetch(JOB_ID)

    assert record["phase"] == "outputs_fetched"
    assert record["fetch_mode"] == "bundle"
    outputs = state / "training" / "cloud" / "jobs" / JOB_ID / "outputs"
    assert (outputs / "axon_observability" / "trainer" / "events.jsonl").is_file()
    assert record["result"]["job_id"] == JOB_ID
    pattern_call = next(call for call in runner.calls if "--file-pattern" in call)
    assert f"^axon_outputs_{JOB_ID}" in pattern_call
    # Only the bundle download happened; no full-tree fallback.
    assert sum(call[:3] == ("kaggle", "kernels", "output") for call in runner.calls) == 1


def test_fetch_falls_back_to_legacy_per_file_when_no_bundle(tmp_path) -> None:
    state = tmp_path / "State"
    _job(state)
    runner = _FakeKaggle()
    runner.full_files = {
        "axon_job_result.json": json.dumps({"job_id": JOB_ID, "status": "completed"}),
        "axon_observability/trainer/events.jsonl": "{}\n",
    }
    adapter = _adapter(tmp_path, state, runner)
    record = adapter.fetch(JOB_ID)

    assert record["fetch_mode"] == "legacy_per_file"
    outputs = state / "training" / "cloud" / "jobs" / JOB_ID / "outputs"
    assert (outputs / "axon_observability" / "trainer" / "events.jsonl").is_file()
    assert sum(call[:3] == ("kaggle", "kernels", "output") for call in runner.calls) == 2


def test_fetch_rejects_a_bundle_receipt_belonging_to_another_job(tmp_path) -> None:
    state = tmp_path / "State"
    _job(state)
    remote = tmp_path / "remote"
    remote.mkdir()
    files = _write_output_bundle(remote, JOB_ID, {}, result_job_id="cd" * 32)
    runner = _FakeKaggle()
    runner.pattern_files = files
    adapter = _adapter(tmp_path, state, runner)
    with pytest.raises(CloudPacketError, match="another Axon job"):
        adapter.fetch(JOB_ID)


def test_sync_pull_verifies_extracts_and_is_idempotent(tmp_path) -> None:
    state = tmp_path / "State"
    _job(state, sync_dataset_ref="axongliksbot/axon-job-abababab-sync")
    remote = tmp_path / "remote"
    members = _members(remote / "tree", {"checkpoints/c390.pt": "bytes-390\n"})
    bundle = remote / f"sync_{JOB_ID}_steps_361_390.tar.gz"
    manifest = remote / f"sync_{JOB_ID}_steps_361_390.sha256.json"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    write_bundle(bundle, manifest, members, kind="mid_run_sync", job_id=JOB_ID, extra={"step_range": [361, 390]})
    runner = _FakeKaggle()
    runner.download_files = {bundle.name: bundle, manifest.name: manifest}
    adapter = _adapter(tmp_path, state, runner)

    pulled = adapter.sync_pull(JOB_ID)
    assert len(pulled["pulled"]) == 1
    assert pulled["pulled"][0]["step_range"] == [361, 390]
    sync_root = state / "training" / "cloud" / "jobs" / JOB_ID / "sync"
    assert (sync_root / "members" / "checkpoints" / "c390.pt").read_text(encoding="utf-8") == "bytes-390\n"
    assert (sync_root / "bundles" / bundle.name).is_file()

    again = adapter.sync_pull(JOB_ID)
    assert again["pulled"] == []
    assert again["already_present"] == [bundle.name]
    assert again["payload_retention"]["keep"] == SYNC_PAYLOAD_KEEP
    assert again["payload_retention"]["kept_ranges"] == [[361, 390]]


def test_sync_status_reports_verified_ranges_and_quarantine(tmp_path) -> None:
    state = tmp_path / "State"
    _job(state)
    adapter = _adapter(tmp_path, state, _FakeKaggle())
    sync_root = state / "training" / "cloud" / "jobs" / JOB_ID / "sync"
    members_dir = sync_root / "members"
    bundles_dir = sync_root / "bundles"
    receipts_dir = sync_root / "receipts"
    for start, end in ((1, 30), (31, 60)):
        source = tmp_path / f"src{start}"
        members = _members(source, {f"checkpoints/c{end}.pt": f"bytes-{end}\n"})
        bundles_dir.mkdir(parents=True, exist_ok=True)
        bundle = bundles_dir / f"sync_{JOB_ID}_steps_{start}_{end}.tar.gz"
        manifest = bundles_dir / f"sync_{JOB_ID}_steps_{start}_{end}.sha256.json"
        document = write_bundle(bundle, manifest, members, kind="mid_run_sync", job_id=JOB_ID)
        for arcname, path in members:
            target = members_dir / arcname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        receipts_dir.mkdir(parents=True, exist_ok=True)
        (receipts_dir / f"sync_{JOB_ID}_steps_{start}_{end}.json").write_text(
            json.dumps({"archive_sha256": document["archive_sha256"]}),
            encoding="utf-8",
        )
    (sync_root / "quarantine" / f"sync_{JOB_ID}_steps_61_90.tar.gz.1").mkdir(parents=True)

    status = adapter.sync_status(JOB_ID)
    assert status["verified_ranges"] == [[1, 30], [31, 60]]
    assert status["verified_through_step"] == 60
    assert status["verified_member_count"] == 2
    assert status["mismatches"] == []
    assert status["quarantined"] == [f"sync_{JOB_ID}_steps_61_90.tar.gz.1"]
    assert status["observation_only"] is True
    assert status["released_ranges"] == []
    assert status["payload_keep"] == SYNC_PAYLOAD_KEEP

    # A tampered member is flagged by rehash, never silently trusted.
    (members_dir / "checkpoints" / "c60.pt").write_text("tampered\n", encoding="utf-8")
    degraded = adapter.sync_status(JOB_ID)
    assert degraded["verified_ranges"] == [[1, 30]]
    assert any("31-60" in item for item in degraded["mismatches"])


def test_sync_payload_retention_releases_older_than_three_windows(tmp_path) -> None:
    state = tmp_path / "State"
    _job(state)
    adapter = _adapter(tmp_path, state, _FakeKaggle())
    sync_root = state / "training" / "cloud" / "jobs" / JOB_ID / "sync"
    for start, end in ((1, 15), (16, 30), (31, 45), (46, 60)):
        source = tmp_path / f"src{end}"
        members = _members(source, {f"checkpoints/c{end}.pt": f"bytes-{end}\n"})
        bundles_dir = sync_root / "bundles"
        members_dir = sync_root / "members"
        receipts_dir = sync_root / "receipts"
        bundles_dir.mkdir(parents=True, exist_ok=True)
        bundle = bundles_dir / f"sync_{JOB_ID}_steps_{start}_{end}.tar.gz"
        manifest = bundles_dir / f"sync_{JOB_ID}_steps_{start}_{end}.sha256.json"
        document = write_bundle(bundle, manifest, members, kind="mid_run_sync", job_id=JOB_ID)
        for arcname, path in members:
            target = members_dir / Path(*arcname.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        receipts_dir.mkdir(parents=True, exist_ok=True)
        (receipts_dir / f"sync_{JOB_ID}_steps_{start}_{end}.json").write_text(
            json.dumps(
                {
                    "schema": "axon-mid-run-sync-local-receipt-v1",
                    "job_id": JOB_ID,
                    "bundle_name": bundle.name,
                    "step_range": [start, end],
                    "archive_sha256": document["archive_sha256"],
                    "observation_only": True,
                }
            ),
            encoding="utf-8",
        )

    retention = prune_sync_payloads(sync_root, keep=3)
    assert retention["released_ranges"] == [[1, 15]]
    assert retention["kept_ranges"] == [[16, 30], [31, 45], [46, 60]]
    members_dir = sync_root / "members" / "checkpoints"
    assert not (members_dir / "c15.pt").is_file()
    assert (members_dir / "c30.pt").read_text(encoding="utf-8") == "bytes-30\n"
    assert not (sync_root / "bundles" / f"sync_{JOB_ID}_steps_1_15.tar.gz").is_file()
    assert (sync_root / "bundles" / f"sync_{JOB_ID}_steps_1_15.sha256.json").is_file()
    assert (sync_root / "receipts" / f"sync_{JOB_ID}_steps_1_15.json").is_file()

    status = adapter.sync_status(JOB_ID)
    assert status["released_ranges"] == [[1, 15]]
    assert status["verified_ranges"] == [[16, 30], [31, 45], [46, 60]]
    assert status["verified_through_step"] == 60
    assert status["mismatches"] == []


def test_apply_sync_pull_skips_jobs_without_sync_flag(tmp_path) -> None:
    state = tmp_path / "State"
    _job(state)
    watcher = Watcher(window=8, show_qa=False, transcript_lines=1)
    result = apply_sync_pull(JOB_ID, watcher, state_root=state)
    assert result["skipped"] is True
    assert watcher.sync_enabled is False
    assert "did not enable mid-run checkpoint uploads" in watcher.render()


@pytest.mark.parametrize("provider_error", ("404 Not Found", "403 Client Error: Forbidden"))
def test_apply_sync_pull_treats_missing_dataset_as_waiting(
    tmp_path,
    provider_error: str,
) -> None:
    state = tmp_path / "State"
    _job(state, sync_mid_run=True, sync_dataset_ref="axongliksbot/axon-job-abababab-sync")

    class FailKaggle(_FakeKaggle):
        def __call__(self, argv, *, cwd=None, capture_output=True):
            command = tuple(str(item) for item in argv)
            if command[:3] == ("kaggle", "datasets", "download"):
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr=provider_error,
                )
            return super().__call__(argv, cwd=cwd, capture_output=capture_output)

    adapter = _adapter(tmp_path, state, FailKaggle())
    watcher = Watcher(window=8, show_qa=False, transcript_lines=1)
    result = apply_sync_pull(JOB_ID, watcher, adapter=adapter)
    assert result["waiting"] is True
    assert "waiting for first checkpoint upload" in watcher.render()


def test_apply_sync_pull_surfaces_integrity_failure_as_fatal(tmp_path) -> None:
    state = tmp_path / "State"
    _job(state, sync_mid_run=True, sync_dataset_ref="axongliksbot/axon-job-abababab-sync")

    class CorruptSyncAdapter:
        state_root = state

        def sync_pull(self, job_id):
            raise CloudPacketError("sync bundle failed hash verification")

    watcher = Watcher(window=8, show_qa=False, transcript_lines=1)
    result = apply_sync_pull(JOB_ID, watcher, adapter=CorruptSyncAdapter())
    assert result["fatal"] is True
    assert watcher.status == "failed"
    assert "sync integrity/provider failure" in watcher.render()


# ── config flag, launch metadata, generated runner ──────────────────────────


def test_sync_flag_is_identity_neutral_for_legacy_recipes() -> None:
    base = dict(
        name="Job",
        provider="kaggle",
        accelerator="gpu",
        entrypoint_argv=("python", "scripts/train.py"),
        include_paths=(),
    )
    legacy = CloudJobConfig(**base)
    assert "sync_mid_run" not in legacy.to_canonical_dict()
    synced = CloudJobConfig(**base, sync_mid_run=True)
    assert synced.to_canonical_dict()["sync_mid_run"] is True
    assert synced.config_id != legacy.config_id
    parsed = CloudJobConfig.from_mapping(synced.to_canonical_dict())
    assert parsed.sync_mid_run is True
    legacy_parsed = CloudJobConfig.from_mapping(legacy.to_canonical_dict())
    assert legacy_parsed.sync_mid_run is False


def test_launch_with_sync_enables_internet_and_records_dataset(tmp_path) -> None:
    state = tmp_path / "State"
    job_dir = _job(state, phase="prepared", sync_mid_run=True)
    (job_dir / "packet").mkdir(parents=True, exist_ok=True)
    (job_dir / "packet" / "axon_packet.zip").write_bytes(b"packet")
    (job_dir / "packet_manifest.json").write_text(
        json.dumps(
            {
                "schema": "axon-cloud-training-packet-v1",
                "job_id": JOB_ID,
                "config": {"accelerator": "gpu", "sync_mid_run": True},
            }
        ),
        encoding="utf-8",
    )

    class LaunchKaggle(_FakeKaggle):
        def __call__(self, argv, *, cwd=None, capture_output=True):
            command = tuple(str(item) for item in argv)
            if command[:3] == ("kaggle", "datasets", "status"):
                return subprocess.CompletedProcess(command, 0, stdout="ready\n", stderr="")
            return super().__call__(argv, cwd=cwd, capture_output=capture_output)

    adapter = _adapter(tmp_path, state, LaunchKaggle())
    launched = adapter.launch(JOB_ID, confirmed=True)
    assert launched["sync_dataset_ref"] == f"axongliksbot/axon-job-{JOB_ID[:8]}-sync"
    metadata = json.loads((job_dir / "kaggle" / "kernel" / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_internet"] is True
    assert metadata["is_private"] is True


def test_generated_runner_archives_outputs_and_injects_sync_env() -> None:
    source = _runner_source("test-input")
    compile(source, "axon_kaggle_runner.py", "exec")
    assert "try_write_output_archive(manifest[\"job_id\"])" in source
    assert 'env["AXON_SYNC_MID_RUN"] = "1"' in source
    assert 'env["AXON_SYNC_DATASET"] = "axon-job-" + manifest["job_id"][:8] + "-sync"' in source
    assert "outputs_archive_failed" in source

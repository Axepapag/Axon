"""Content-addressed cloud packet and private Kaggle adapter safety."""

from __future__ import annotations

import json
import subprocess
import zipfile

import pytest

from runtime.trainer import cloud_jobs
from runtime.trainer.cloud_jobs import (
    CLOUD_JOB_RECORD_SCHEMA,
    CloudJobConfig,
    CloudPacketError,
    prepare_cloud_job,
    write_job_record,
)
from runtime.trainer.kaggle_adapter import KaggleTrainerAdapter


def _config(*, include_paths=(), sensitive=False) -> CloudJobConfig:
    return CloudJobConfig(
        name="Test D64 job",
        provider="kaggle",
        accelerator="gpu",
        entrypoint_argv=("python", "scripts/train.py", "--tranche-steps", "8"),
        include_paths=tuple(include_paths),
        allow_sensitive_state_upload=sensitive,
    )


def test_packet_binds_committed_source_explicit_state_and_no_credentials(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "Axon"
    state = repo / "State"
    script = repo / "scripts" / "train.py"
    identity = state / "active" / "identity.json"
    script.parent.mkdir(parents=True)
    identity.parent.mkdir(parents=True)
    script.write_text("print('train')\n", encoding="utf-8")
    identity.write_text('{"identity":"Axon"}\n', encoding="utf-8")
    monkeypatch.setattr(
        cloud_jobs,
        "_committed_source",
        lambda _repo: ("a" * 40, (("scripts/train.py", script),)),
    )

    prepared = prepare_cloud_job(
        repo_root=repo,
        state_root=state,
        config=_config(include_paths=("State/active/identity.json",), sensitive=True),
    )
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["credential_files_included"] is False
    assert manifest["public_visibility_allowed"] is False
    assert {item["role"] for item in manifest["files"]} == {
        "committed_source",
        "operational_state",
    }
    with zipfile.ZipFile(prepared.packet_path) as archive:
        assert "axon_packet_manifest.json" in archive.namelist()
        assert "State/active/identity.json" in archive.namelist()


def test_state_upload_requires_explicit_acknowledgement(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "Axon"
    state = repo / "State"
    script = repo / "scripts" / "train.py"
    selected = state / "active" / "identity.json"
    script.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    script.write_text("pass\n", encoding="utf-8")
    selected.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        cloud_jobs,
        "_committed_source",
        lambda _repo: ("b" * 40, (("scripts/train.py", script),)),
    )
    with pytest.raises(CloudPacketError, match="acknowledgement"):
        prepare_cloud_job(
            repo_root=repo,
            state_root=state,
            config=_config(include_paths=("State/active/identity.json",), sensitive=False),
        )


def test_credential_named_input_is_rejected(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "Axon"
    state = repo / "State"
    script = repo / "scripts" / "train.py"
    secret = repo / "inputs" / "credentials.json"
    script.parent.mkdir(parents=True)
    secret.parent.mkdir(parents=True)
    script.write_text("pass\n", encoding="utf-8")
    secret.write_text("never\n", encoding="utf-8")
    monkeypatch.setattr(
        cloud_jobs,
        "_committed_source",
        lambda _repo: ("c" * 40, (("scripts/train.py", script),)),
    )
    with pytest.raises(CloudPacketError, match="credential-bearing"):
        prepare_cloud_job(
            repo_root=repo,
            state_root=state,
            config=_config(include_paths=("inputs/credentials.json",)),
        )


class _FakeKaggle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv, *, cwd=None, capture_output=True):
        command = tuple(str(item) for item in argv)
        self.calls.append(command)
        if command[:2] == ("kaggle", "--version"):
            output = "Kaggle API 2.2.4\n"
        elif command[:2] == ("kaggle", "quota"):
            output = '[{"resource":"GPU","remaining":"30.00h","total":"30.00h","refreshAt":"soon"}]'
        elif command[:3] == ("kaggle", "datasets", "status"):
            output = "ready\n"
        else:
            output = "ok\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")


def test_kaggle_launch_is_private_idempotent_and_uses_no_credentials_in_argv(tmp_path) -> None:
    repo = tmp_path / "Axon"
    state = repo / "State"
    repo.mkdir()
    job_id = "d" * 64
    job_dir = state / "training" / "cloud" / "jobs" / job_id
    (job_dir / "packet").mkdir(parents=True)
    (job_dir / "packet" / "axon_packet.zip").write_bytes(b"packet")
    (job_dir / "packet_manifest.json").write_text(
        json.dumps(
            {
                "schema": "axon-cloud-training-packet-v1",
                "job_id": job_id,
                "config": {"accelerator": "gpu"},
            }
        ),
        encoding="utf-8",
    )
    write_job_record(
        state,
        job_id,
        {
            "schema": CLOUD_JOB_RECORD_SCHEMA,
            "job_id": job_id,
            "provider": "kaggle",
            "accelerator": "gpu",
            "phase": "prepared",
            "git_revision": "e" * 40,
            "packet_path": str(job_dir / "packet" / "axon_packet.zip"),
            "packet_sha256": "f" * 64,
            "dataset_ref": None,
            "kernel_ref": None,
            "public": False,
        },
    )
    runner = _FakeKaggle()
    adapter = KaggleTrainerAdapter(
        repo_root=repo,
        state_root=state,
        owner="axepapgt",
        runner=runner,
    )
    launched = adapter.launch(job_id, confirmed=True)
    metadata = json.loads((job_dir / "kaggle" / "kernel" / "kernel-metadata.json").read_text(encoding="utf-8"))
    generated_runner = (job_dir / "kaggle" / "kernel" / "axon_kaggle_runner.py").read_text(encoding="utf-8")
    compile(generated_runner, "axon_kaggle_runner.py", "exec")
    assert launched["phase"] == "submitted"
    dataset_metadata = json.loads(
        (job_dir / "kaggle" / "dataset" / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    assert dataset_metadata["licenses"] == [{"name": "other"}]
    assert "All rights reserved" in dataset_metadata["description"]
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    create = next(call for call in runner.calls if call[:3] == ("kaggle", "datasets", "create"))
    push = next(call for call in runner.calls if call[:3] == ("kaggle", "kernels", "push"))
    assert "-u" not in create
    assert "--accelerator" not in push
    assert "unpacked_input_detected" in generated_runner
    assert 'rglob("axon_packet_manifest.json")' in generated_runner
    assert 'env["PYTHONPATH"]' in generated_runner
    status_index = next(index for index, call in enumerate(runner.calls) if call[:3] == ("kaggle", "datasets", "status"))
    push_index = next(index for index, call in enumerate(runner.calls) if call[:3] == ("kaggle", "kernels", "push"))
    assert status_index < push_index
    assert all("token" not in " ".join(call).lower() for call in runner.calls)


def test_kaggle_launch_requires_explicit_confirmation(tmp_path) -> None:
    repo = tmp_path / "Axon"
    state = repo / "State"
    repo.mkdir()
    adapter = KaggleTrainerAdapter(repo_root=repo, state_root=state, owner="axepapgt", runner=_FakeKaggle())
    with pytest.raises(CloudPacketError, match="confirmation"):
        adapter.launch("missing", confirmed=False)


def test_kaggle_dataset_semantic_error_fails_closed(tmp_path) -> None:
    repo = tmp_path / "Axon"
    state = repo / "State"
    repo.mkdir()
    job_id = "a" * 64
    job_dir = state / "training" / "cloud" / "jobs" / job_id
    (job_dir / "packet").mkdir(parents=True)
    (job_dir / "packet" / "axon_packet.zip").write_bytes(b"packet")
    (job_dir / "packet_manifest.json").write_text(
        json.dumps(
            {
                "schema": "axon-cloud-training-packet-v1",
                "job_id": job_id,
                "config": {"accelerator": "gpu"},
            }
        ),
        encoding="utf-8",
    )
    write_job_record(
        state,
        job_id,
        {
            "schema": CLOUD_JOB_RECORD_SCHEMA,
            "job_id": job_id,
            "provider": "kaggle",
            "accelerator": "gpu",
            "phase": "prepared",
            "git_revision": "b" * 40,
            "packet_path": str(job_dir / "packet" / "axon_packet.zip"),
            "packet_sha256": "c" * 64,
            "dataset_ref": None,
            "kernel_ref": None,
            "public": False,
        },
    )

    class SemanticFailure(_FakeKaggle):
        def __call__(self, argv, *, cwd=None, capture_output=True):
            result = super().__call__(argv, cwd=cwd, capture_output=capture_output)
            if tuple(str(item) for item in argv)[:3] == ("kaggle", "datasets", "create"):
                return subprocess.CompletedProcess(
                    result.args,
                    0,
                    stdout="Dataset creation error: Please select a valid license\n",
                    stderr="",
                )
            return result

    adapter = KaggleTrainerAdapter(
        repo_root=repo,
        state_root=state,
        owner="axepapgt",
        runner=SemanticFailure(),
    )
    with pytest.raises(CloudPacketError, match="exit code zero"):
        adapter.launch(job_id, confirmed=True)
    assert json.loads((job_dir / "job.json").read_text(encoding="utf-8"))["phase"] == "prepared"

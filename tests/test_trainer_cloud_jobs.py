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
from runtime.trainer.kaggle_adapter import (
    KaggleTrainerAdapter,
    _runner_source,
    classify_kaggle_provider_status,
)


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
        elif command[:3] == ("kaggle", "kernels", "status"):
            output = 'status "KernelWorkerStatus.ERROR"\n'
        else:
            output = "ok\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")


def test_kaggle_doctor_does_not_claim_accelerator_entitlement_from_quota(tmp_path) -> None:
    repo = tmp_path / "Axon"
    state = repo / "State"
    repo.mkdir()
    result = KaggleTrainerAdapter(
        repo_root=repo,
        state_root=state,
        owner="axepapgt",
        runner=_FakeKaggle(),
    ).doctor()
    assert result["healthy"] is True
    assert "not proven by quota" in result["accelerator_entitlement"]


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
    notebook = json.loads(
        (job_dir / "kaggle" / "kernel" / "axon_kaggle_runner.ipynb").read_text(encoding="utf-8")
    )
    generated_runner = "".join(notebook["cells"][0]["source"])
    compile(generated_runner, "axon_kaggle_runner.py", "exec")
    assert launched["phase"] == "submitted"
    dataset_metadata = json.loads(
        (job_dir / "kaggle" / "dataset" / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    assert dataset_metadata["licenses"] == [{"name": "other"}]
    assert "All rights reserved" in dataset_metadata["description"]
    assert metadata["is_private"] is True
    assert metadata["kernel_type"] == "notebook"
    assert metadata["code_file"] == "axon_kaggle_runner.ipynb"
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    create = next(call for call in runner.calls if call[:3] == ("kaggle", "datasets", "create"))
    push = next(call for call in runner.calls if call[:3] == ("kaggle", "kernels", "push"))
    assert "-u" not in create
    assert "--accelerator" not in push
    assert "unpacked_input_detected" in generated_runner
    assert 'rglob("axon_packet_manifest.json")' in generated_runner
    assert 'torch.ones(1, device=\'cuda\')' in generated_runner
    assert 'publish("python_selected"' in generated_runner
    assert 'raise RuntimeError("no Kaggle Python interpreter passed a real CUDA compute probe")' in generated_runner
    assert 'env["PYTHONPATH"]' in generated_runner
    assert notebook["nbformat"] == 4
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


@pytest.mark.parametrize("exit_code", [0, 2])
def test_generated_runner_preserves_success_and_child_failure_receipts(tmp_path, exit_code) -> None:
    namespace = {"__name__": "runner_test"}
    exec(compile(_runner_source("test-input"), "generated_runner.py", "exec"), namespace)
    writes = []
    namespace["main"] = lambda: exit_code
    namespace["OBS"] = tmp_path
    namespace["atomic_json"] = lambda path, body: writes.append(body)
    assert namespace["run_guarded"]() == exit_code
    # main owns the subprocess result receipt; an ordinary return must not be
    # turned into a synthetic SystemExit failure by the notebook wrapper.
    assert writes == []


def test_generated_runner_records_unexpected_exceptions(tmp_path) -> None:
    namespace = {"__name__": "runner_test"}
    exec(compile(_runner_source("test-input"), "generated_runner.py", "exec"), namespace)
    writes = []

    def fail():
        raise RuntimeError("broken packet")

    namespace["main"] = fail
    namespace["OBS"] = tmp_path
    namespace["atomic_json"] = lambda path, body: writes.append(body)
    namespace["publish"] = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="broken packet"):
        namespace["run_guarded"]()
    assert writes[0]["status"] == "failed"
    assert writes[0]["error_type"] == "RuntimeError"


def test_dataset_readiness_timeout_preserves_upload_for_retry(tmp_path, monkeypatch) -> None:
    from runtime.trainer import kaggle_adapter

    monkeypatch.setattr(kaggle_adapter, "KAGGLE_DATASET_READY_ATTEMPTS", 2)
    monkeypatch.setattr(kaggle_adapter, "KAGGLE_DATASET_READY_INTERVAL_SECONDS", 0)
    job_id = "7" * 64
    state = tmp_path / "State"
    write_job_record(state, job_id, {
        "schema": CLOUD_JOB_RECORD_SCHEMA,
        "job_id": job_id,
        "phase": "prepared",
        "public": False,
    })

    class EventuallyReady(_FakeKaggle):
        ready = False

        def __call__(self, argv, *, cwd=None, capture_output=True):
            result = super().__call__(argv, cwd=cwd, capture_output=capture_output)
            if tuple(argv[:3]) == ("kaggle", "datasets", "status") and not self.ready:
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="403 not indexed yet")
            return result

    runner = EventuallyReady()
    adapter = KaggleTrainerAdapter(repo_root=tmp_path, state_root=state, owner="axepapgt", runner=runner)
    monkeypatch.setattr(adapter, "_materialize_kaggle_files", lambda key: (
        cloud_jobs.read_job_record(state, key), tmp_path / "dataset", tmp_path / "kernel",
    ))
    with pytest.raises(CloudPacketError, match="403 not indexed yet"):
        adapter.launch(job_id, confirmed=True)
    record = cloud_jobs.read_job_record(state, job_id)
    assert record["phase"] == "dataset_uploaded"
    assert record["dataset_ref"] == f"axepapgt/axon-job-{job_id[:16]}-input"
    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in runner.calls)
    runner.ready = True
    assert adapter.launch(job_id, confirmed=True)["phase"] == "submitted"
    assert sum(call[:3] == ("kaggle", "datasets", "create") for call in runner.calls) == 1


def test_kaggle_launch_retries_a_finished_private_job_without_reuploading_dataset(tmp_path) -> None:
    repo = tmp_path / "Axon"
    state = repo / "State"
    repo.mkdir()
    job_id = "9" * 64
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
    slug = f"axon-job-{job_id[:16]}"
    write_job_record(
        state,
        job_id,
        {
            "schema": CLOUD_JOB_RECORD_SCHEMA,
            "job_id": job_id,
            "provider": "kaggle",
            "accelerator": "gpu",
            "phase": "outputs_fetched",
            "git_revision": "1" * 40,
            "packet_path": str(job_dir / "packet" / "axon_packet.zip"),
            "packet_sha256": "2" * 64,
            "dataset_ref": f"axepapgt/{slug}-input",
            "kernel_ref": f"axepapgt/{slug}",
            "public": False,
        },
    )
    runner = _FakeKaggle()
    adapter = KaggleTrainerAdapter(repo_root=repo, state_root=state, owner="axepapgt", runner=runner)
    retried = adapter.launch(job_id, confirmed=True)
    assert retried["phase"] == "submitted"
    assert retried["provider_status_before_resubmission"] == 'status "KernelWorkerStatus.ERROR"'
    assert not any(call[:3] == ("kaggle", "datasets", "create") for call in runner.calls)
    assert sum(call[:3] == ("kaggle", "kernels", "push") for call in runner.calls) == 1


@pytest.mark.parametrize("provider_status", ["RUNNING", "QUEUED", "UNKNOWN"])
def test_kaggle_launch_refuses_active_or_unknown_jobs(tmp_path, provider_status) -> None:
    class RunningKaggle(_FakeKaggle):
        def __call__(self, argv, *, cwd=None, capture_output=True):
            result = super().__call__(argv, cwd=cwd, capture_output=capture_output)
            if tuple(str(item) for item in argv)[:3] == ("kaggle", "kernels", "status"):
                return subprocess.CompletedProcess(
                    result.args,
                    0,
                    stdout=f'status "KernelWorkerStatus.{provider_status}"\n',
                    stderr="",
                )
            return result

    repo = tmp_path / "Axon"
    state = repo / "State"
    repo.mkdir()
    job_id = "8" * 64
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
    slug = f"axon-job-{job_id[:16]}"
    write_job_record(
        state,
        job_id,
        {
            "schema": CLOUD_JOB_RECORD_SCHEMA,
            "job_id": job_id,
            "provider": "kaggle",
            "accelerator": "gpu",
            "phase": "submitted",
            "git_revision": "3" * 40,
            "packet_path": str(job_dir / "packet" / "axon_packet.zip"),
            "packet_sha256": "4" * 64,
            "dataset_ref": f"axepapgt/{slug}-input",
            "kernel_ref": f"axepapgt/{slug}",
            "public": False,
        },
    )
    runner = RunningKaggle()
    adapter = KaggleTrainerAdapter(repo_root=repo, state_root=state, owner="axepapgt", runner=runner)
    with pytest.raises(CloudPacketError, match=r"already active|recognized terminal"):
        adapter.launch(job_id, confirmed=True)
    assert not any(call[:3] == ("kaggle", "kernels", "push") for call in runner.calls)


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


def test_kaggle_owner_defaults_to_authenticated_cli_identity(tmp_path) -> None:
    class IdentityKaggle(_FakeKaggle):
        def __call__(self, argv, *, cwd=None, capture_output=True):
            if tuple(str(item) for item in argv)[:2] == ("kaggle", "config"):
                return subprocess.CompletedProcess(
                    list(argv),
                    0,
                    stdout="Configuration values from /home/tester/.kaggle\n- username: axongliksbot\n- auth_method: OAUTH\n",
                    stderr="",
                )
            return super().__call__(argv, cwd=cwd, capture_output=capture_output)

    repo = tmp_path / "Axon"
    state = repo / "State"
    repo.mkdir()
    adapter = KaggleTrainerAdapter(repo_root=repo, state_root=state, runner=IdentityKaggle())
    assert adapter.resolved_owner == "axongliksbot"


def test_kaggle_launch_refuses_cross_account_ownership(tmp_path) -> None:
    class IdentityKaggle(_FakeKaggle):
        def __call__(self, argv, *, cwd=None, capture_output=True):
            if tuple(str(item) for item in argv)[:2] == ("kaggle", "config"):
                return subprocess.CompletedProcess(
                    list(argv),
                    0,
                    stdout="Configuration values from /home/tester/.kaggle\n- username: axepapgt\n- auth_method: OAUTH\n",
                    stderr="",
                )
            return super().__call__(argv, cwd=cwd, capture_output=capture_output)

    repo = tmp_path / "Axon"
    state = repo / "State"
    repo.mkdir()
    job_id = "b" * 64
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
            "git_revision": "d" * 40,
            "packet_path": str(job_dir / "packet" / "axon_packet.zip"),
            "packet_sha256": "e" * 64,
            "dataset_ref": None,
            "kernel_ref": None,
            "public": False,
        },
    )
    adapter = KaggleTrainerAdapter(
        repo_root=repo,
        state_root=state,
        owner="axongliksbot",
        runner=IdentityKaggle(),
    )
    with pytest.raises(CloudPacketError, match="different account") as caught:
        adapter.launch(job_id, confirmed=True)
    message = str(caught.value)
    assert "axepapgt" in message and "axongliksbot" in message
    # Nothing was created or uploaded: the gate fires before any side effect.
    assert not (job_dir / "kaggle" / "dataset" / "dataset-metadata.json").exists()
    assert not any(call[:3] == ("kaggle", "datasets", "create") for call in adapter.runner.calls)


def test_classify_kaggle_provider_status() -> None:
    assert classify_kaggle_provider_status('status "KernelWorkerStatus.RUNNING"') == "running"
    assert classify_kaggle_provider_status('status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert classify_kaggle_provider_status("not submitted") == "not_submitted"
    assert classify_kaggle_provider_status('status "KernelWorkerStatus.ERROR"') == "error"


def test_job_catalog_names_jobs_and_puts_running_first(tmp_path) -> None:
    repo = tmp_path / "Axon"
    state = repo / "State"
    repo.mkdir()
    running_id = "a" * 64
    fetched_id = "b" * 64

    def write_job(job_id: str, *, phase: str, name: str, teach: bool) -> None:
        job_dir = state / "training" / "cloud" / "jobs" / job_id
        job_dir.mkdir(parents=True)
        argv = [
            "python",
            "scripts/train_living_reasoning_smoke.py",
            "--candidate-label",
            "axon-d64-mixer-4l-ffn256-h1",
            "--heads",
            "1",
            "--layers",
            "4",
            "--ffn-dim",
            "256",
            "--tranche-steps",
            "60",
            "--resume",
        ]
        if teach:
            argv.append("--teach-multicell-copy")
        (job_dir / "packet_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "axon-cloud-training-packet-v1",
                    "job_id": job_id,
                    "config": {
                        "name": name,
                        "accelerator": "gpu",
                        "entrypoint_argv": argv,
                    },
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
                "phase": phase,
                "kernel_ref": f"axepapgt/axon-job-{job_id[:16]}",
                "public": False,
            },
        )

    write_job(fetched_id, phase="outputs_fetched", name="Old fetched job", teach=False)
    write_job(running_id, phase="submitted", name="Axon D64 mixer multi-cell copy teach", teach=True)

    class LiveKaggle(_FakeKaggle):
        def __call__(self, argv, *, cwd=None, capture_output=True):
            command = tuple(str(item) for item in argv)
            if command[:3] == ("kaggle", "kernels", "status"):
                ref = command[3]
                if running_id[:16] in ref:
                    output = 'status "KernelWorkerStatus.RUNNING"\n'
                else:
                    output = 'status "KernelWorkerStatus.COMPLETE"\n'
                return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
            return super().__call__(argv, cwd=cwd, capture_output=capture_output)

    adapter = KaggleTrainerAdapter(
        repo_root=repo,
        state_root=state,
        owner="axepapgt",
        runner=LiveKaggle(),
    )
    catalog = adapter.job_catalog(refresh_live=True)
    assert [row["job_id"] for row in catalog] == [running_id, fetched_id]
    assert catalog[0]["live_status"] == "running"
    assert catalog[0]["name"] == "Axon D64 mixer multi-cell copy teach"
    assert catalog[0]["shape"] == "1h/4L/FFN256"
    assert catalog[0]["teach_multicell_copy"] is True
    assert catalog[1]["live_status"] == "fetched"
    local_only = adapter.job_catalog(refresh_live=False)
    assert local_only[0]["job_id"] == running_id
    assert local_only[0]["live_status"] == "submitted"

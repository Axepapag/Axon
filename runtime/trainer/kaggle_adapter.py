"""Official Kaggle CLI adapter for durable Axon cloud-training jobs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .cloud_jobs import (
    CLOUD_JOB_RECORD_SCHEMA,
    CloudJobConfig,
    CloudPacketError,
    prepare_cloud_job,
    read_job_record,
    write_job_record,
)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
KAGGLE_GPU_MACHINE_SHAPE = "NvidiaTeslaT4"
KAGGLE_DATASET_READY_ATTEMPTS = 24
KAGGLE_DATASET_READY_INTERVAL_SECONDS = 2.0


def _default_runner(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
    )


def _runner_source(dataset_slug: str) -> str:
    return f'''"""Generated Axon Kaggle runner.  Do not edit: packet identity owns execution."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

DATASET_SLUG = {dataset_slug!r}
INPUT = Path("/kaggle/input") / DATASET_SLUG / "axon_packet.zip"
WORK = Path("/kaggle/working/axon_job")
OBS = Path("/kaggle/working/axon_observability")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
    os.replace(temporary, path)


def publish(status: str, **details) -> None:
    event = {{
        "schema": "axon-kaggle-runner-event-v1",
        "status": status,
        "unix_seconds": time.time(),
        "details": details,
    }}
    atomic_json(OBS / "runner_current.json", event)
    with (OBS / "runner_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\\n")
        handle.flush()
    print("AXON_KAGGLE " + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def select_python(accelerator: str) -> str:
    if accelerator != "gpu":
        return sys.executable
    requested = [
        "/opt/conda/bin/python",
        "/opt/conda/bin/python3",
        shutil.which("python"),
        shutil.which("python3"),
        "/usr/local/bin/python",
        "/usr/local/bin/python3",
        sys.executable,
    ]
    candidates = []
    for raw in requested:
        if raw and raw not in candidates and Path(raw).is_file():
            candidates.append(raw)
    probe_source = (
        "import json, torch; "
        "available=bool(torch.cuda.is_available()); "
        "device=None; compute=False; error=None; "
        "\\ntry:\\n"
        " device=torch.cuda.get_device_name(0) if available else None; "
        " compute=bool(torch.ones(1, device='cuda').item()==1.0) if available else False\\n"
        "except Exception as exc:\\n error=f'{{type(exc).__name__}}: {{exc}}'\\n"
        "print(json.dumps({{'torch':torch.__version__,'cuda_available':available,"
        "'cuda_compute':compute,'device':device,'error':error}}))"
    )
    results = []
    for executable in candidates:
        completed = subprocess.run(
            [executable, "-c", probe_source],
            check=False,
            capture_output=True,
            text=True,
        )
        record = {{
            "executable": executable,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }}
        results.append(record)
        if completed.returncode == 0:
            try:
                probe = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                probe = {{}}
            if probe.get("cuda_available") is True and probe.get("cuda_compute") is True:
                publish("python_selected", selected=executable, probe=probe, candidates=results)
                return executable
    publish("python_selection_failed", candidates=results)
    raise RuntimeError("no Kaggle Python interpreter passed a real CUDA compute probe")


def main() -> int:
    OBS.mkdir(parents=True, exist_ok=True)
    publish("starting", python=sys.version, input=str(INPUT))
    input_root = Path("/kaggle/input")
    input_dir = INPUT.parent
    WORK.mkdir(parents=True, exist_ok=True)
    if INPUT.is_file():
        with zipfile.ZipFile(INPUT) as archive:
            archive.extractall(WORK)
    elif input_dir.is_dir():
        # Kaggle may expose a dataset ZIP as already-unpacked files. Preserve
        # the packet tree exactly in either provider representation.
        publish("unpacked_input_detected", input_dir=str(input_dir))
        for item in input_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, WORK / item.name)
            elif item.is_dir():
                shutil.copytree(item, WORK / item.name, dirs_exist_ok=True)
    else:
        # Provider mount names are not a packet identity. Discover the one
        # verified manifest if Kaggle rewrites the private dataset mount path.
        manifests = sorted(input_root.rglob("axon_packet_manifest.json")) if input_root.is_dir() else []
        publish(
            "input_discovery",
            input_root_exists=input_root.is_dir(),
            manifest_candidates=[str(path) for path in manifests],
        )
        if len(manifests) != 1:
            raise RuntimeError(
                f"Axon packet is missing at {{INPUT}} and discovery found {{len(manifests)}} manifests"
            )
        discovered_root = manifests[0].parent
        for item in discovered_root.iterdir():
            if item.is_file():
                shutil.copy2(item, WORK / item.name)
            elif item.is_dir():
                shutil.copytree(item, WORK / item.name, dirs_exist_ok=True)
    manifest_path = WORK / "axon_packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "axon-cloud-training-packet-v1":
        raise RuntimeError("unsupported Axon cloud packet schema")
    for record in manifest["files"]:
        candidate = (WORK / record["path"]).resolve()
        candidate.relative_to(WORK.resolve())
        if not candidate.is_file() or candidate.stat().st_size != record["bytes"]:
            raise RuntimeError(f"packet file missing or wrong size: {{record['path']}}")
        if sha256_file(candidate) != record["sha256"]:
            raise RuntimeError(f"packet file hash mismatch: {{record['path']}}")
    accelerator = manifest["config"]["accelerator"]
    argv = list(manifest["config"]["entrypoint_argv"])
    argv[0] = select_python(accelerator)
    if "--progress-dir" not in argv:
        argv.extend(["--progress-dir", str(OBS / "trainer")])
    if "--external-job-id" not in argv:
        argv.extend(["--external-job-id", manifest["job_id"]])
    publish(
        "running",
        job_id=manifest["job_id"],
        git_revision=manifest["git_revision"],
        accelerator=accelerator,
        argv=argv,
    )
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(WORK) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(argv, cwd=WORK, env=env, check=False)
    result = {{
        "schema": "axon-cloud-training-result-v1",
        "job_id": manifest["job_id"],
        "git_revision": manifest["git_revision"],
        "returncode": completed.returncode,
        "status": "completed" if completed.returncode == 0 else "failed",
    }}
    atomic_json(Path("/kaggle/working/axon_job_result.json"), result)
    publish(result["status"], returncode=completed.returncode, job_id=manifest["job_id"])
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        OBS.mkdir(parents=True, exist_ok=True)
        failure = {{
            "schema": "axon-cloud-training-result-v1",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }}
        atomic_json(Path("/kaggle/working/axon_job_result.json"), failure)
        publish("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    '''


def _runner_notebook(source: str) -> dict[str, Any]:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class KaggleTrainerAdapter:
    """Prepare, launch, observe, and retrieve private Kaggle jobs.

    The official ``kaggle`` executable owns authentication.  This class never
    opens or copies the credential file and never places credentials in child
    command arguments.
    """

    def __init__(
        self,
        *,
        repo_root: Path | str,
        state_root: Path | str,
        owner: str | None = None,
        runner: CommandRunner = _default_runner,
    ) -> None:
        self.repo_root = Path(repo_root).resolve(strict=True)
        self.state_root = Path(state_root).resolve(strict=False)
        self.runner = runner
        self.owner: str | None = None
        if owner:
            self.owner = self._validated_owner(owner)

    _OWNER_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")

    @classmethod
    def _validated_owner(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized or any(character not in cls._OWNER_CHARACTERS for character in normalized):
            raise ValueError("Kaggle owner must be a username slug")
        return normalized

    def _authenticated_username(self) -> str | None:
        """Ask the CLI who it is authenticated as (never reads credential files)."""
        completed = self._run(("kaggle", "config", "view"))
        for line in completed.stdout.splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            if key.strip().lstrip("- ").strip().lower() == "username":
                username = value.strip().lower()
                if username and username != "none":
                    return username
        return None

    @property
    def resolved_owner(self) -> str:
        """Explicit owner, or the CLI's authenticated identity (lazy, cached)."""
        if self.owner is None:
            username = self._authenticated_username()
            if not username:
                raise CloudPacketError(
                    "Kaggle owner not specified and the CLI has no authenticated identity; "
                    "pass --owner explicitly or authenticate first"
                )
            self.owner = self._validated_owner(username)
        return self.owner

    def _run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = self.runner(argv, cwd=cwd, capture_output=capture_output)
        if completed.returncode:
            message = (completed.stderr or completed.stdout or "command failed").strip()
            raise CloudPacketError(f"{' '.join(argv[:3])} failed: {message}")
        return completed

    def doctor(self) -> dict[str, Any]:
        version = self._run(("kaggle", "--version")).stdout.strip()
        quota_raw = self._run(("kaggle", "quota", "--format", "json")).stdout
        quota = json.loads(quota_raw)
        return {
            "schema": "axon-kaggle-doctor-v1",
            "healthy": True,
            "owner": self.resolved_owner,
            "cli": version,
            "authentication": "verified by authenticated quota request",
            "quota": quota,
            "accelerator_entitlement": "not proven by quota; proved only by a real device compute probe",
            "credential_storage": "official Kaggle user credential store (not inspected)",
        }

    def prepare(self, config_path: Path | str) -> dict[str, Any]:
        config = CloudJobConfig.read(config_path)
        prepared = prepare_cloud_job(
            repo_root=self.repo_root,
            state_root=self.state_root,
            config=config,
        )
        return {
            "job_id": prepared.job_id,
            "job_dir": str(prepared.job_dir),
            "packet_path": str(prepared.packet_path),
            "git_revision": prepared.revision,
            "file_count": prepared.file_count,
            "uncompressed_bytes": prepared.total_bytes,
            "phase": "prepared",
            "uploaded": False,
            "launched": False,
        }

    def _wait_for_dataset_ready(self, dataset_ref: str) -> None:
        last_status = "unqueried"
        for attempt in range(KAGGLE_DATASET_READY_ATTEMPTS):
            completed = self._run(("kaggle", "datasets", "status", dataset_ref))
            last_status = completed.stdout.strip().lower()
            if last_status == "ready":
                return
            if attempt + 1 < KAGGLE_DATASET_READY_ATTEMPTS:
                time.sleep(KAGGLE_DATASET_READY_INTERVAL_SECONDS)
        raise CloudPacketError(
            f"Kaggle dataset did not become ready for kernel attachment: {dataset_ref} ({last_status})"
        )

    def _materialize_kaggle_files(self, job_id: str) -> tuple[dict[str, Any], Path, Path]:
        record = read_job_record(self.state_root, job_id)
        job_dir = self.state_root / "training" / "cloud" / "jobs" / job_id
        manifest = json.loads((job_dir / "packet_manifest.json").read_text(encoding="utf-8"))
        slug = f"axon-job-{job_id[:16]}"
        dataset_slug = f"{slug}-input"
        dataset_ref = f"{self.resolved_owner}/{dataset_slug}"
        kernel_ref = f"{self.resolved_owner}/{slug}"
        dataset_dir = job_dir / "kaggle" / "dataset"
        kernel_dir = job_dir / "kaggle" / "kernel"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        kernel_dir.mkdir(parents=True, exist_ok=True)
        packet = job_dir / "packet" / "axon_packet.zip"
        destination = dataset_dir / "axon_packet.zip"
        if not destination.exists():
            try:
                os.link(packet, destination)
            except OSError:
                import shutil

                shutil.copy2(packet, destination)
        dataset_metadata = {
            "title": f"Axon job {job_id[:16]} input",
            "id": dataset_ref,
            "licenses": [{"name": "other"}],
            "description": (
                "Private, content-addressed Axon training packet. All rights reserved; "
                "not licensed for public redistribution."
            ),
        }
        kernel_metadata = {
            "id": kernel_ref,
            "title": f"Axon job {job_id[:16]}",
            "code_file": "axon_kaggle_runner.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": manifest["config"]["accelerator"] == "gpu",
            "enable_tpu": manifest["config"]["accelerator"] == "tpu",
            "enable_internet": False,
            "dataset_sources": [dataset_ref],
            "competition_sources": [],
            "kernel_sources": [],
        }
        if manifest["config"]["accelerator"] == "gpu":
            # Kaggle's generic/default GPU may resolve to a CPU image or a P100.
            # The current default PyTorch cu128 build does not support P100
            # compute, so select the supported T4 shape explicitly and fail
            # closed in the training entrypoint if CUDA is still unavailable.
            kernel_metadata["machine_shape"] = KAGGLE_GPU_MACHINE_SHAPE
        (dataset_dir / "dataset-metadata.json").write_text(
            json.dumps(dataset_metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(kernel_metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        source = _runner_source(dataset_slug)
        (kernel_dir / "axon_kaggle_runner.ipynb").write_text(
            json.dumps(_runner_notebook(source), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return record, dataset_dir, kernel_dir

    def launch(self, job_id: str, *, confirmed: bool) -> dict[str, Any]:
        if not confirmed:
            raise CloudPacketError("Kaggle launch requires explicit operator confirmation")
        self.doctor()
        # Cross-account gate: Kaggle namespaces by the authenticated identity,
        # not by what the metadata declares. If the CLI is logged into a
        # different account than the owner these refs target, dataset creation
        # returns null slugs (the failure that cost this project a day of work
        # on 2026-09-03). Fail loudly BEFORE creating anything.
        authenticated_as = self._authenticated_username()
        if authenticated_as is not None and authenticated_as != self.resolved_owner:
            raise CloudPacketError(
                "Kaggle CLI is authenticated as a different account: "
                f"authenticated={authenticated_as!r} but job owner={self.resolved_owner!r}. "
                "Log in as the job owner (or pass the matching --owner) before launching."
            )
        record, dataset_dir, kernel_dir = self._materialize_kaggle_files(job_id)
        if record.get("public") is not False:
            raise CloudPacketError("Kaggle job record does not prove private visibility")
        slug = f"axon-job-{job_id[:16]}"
        dataset_ref = f"{self.resolved_owner}/{slug}-input"
        kernel_ref = f"{self.resolved_owner}/{slug}"
        resubmission_status: str | None = None
        if record.get("phase") in {"submitted", "outputs_fetched"}:
            if record.get("dataset_ref") != dataset_ref or record.get("kernel_ref") != kernel_ref:
                raise CloudPacketError("existing Kaggle references do not match this private job identity")
            completed = self._run(("kaggle", "kernels", "status", kernel_ref))
            resubmission_status = completed.stdout.strip()
            normalized_status = resubmission_status.lower()
            if any(marker in normalized_status for marker in ("running", "queued", "pending")):
                raise CloudPacketError(
                    "Kaggle job is already active; refusing to submit a duplicate version: "
                    f"{resubmission_status}"
                )
            self._wait_for_dataset_ready(dataset_ref)
        if record.get("phase") == "prepared":
            created = self._run(("kaggle", "datasets", "create", "-p", str(dataset_dir), "-r", "skip"))
            combined_output = f"{created.stdout}\n{created.stderr}".lower()
            if "dataset creation error" in combined_output:
                raise CloudPacketError(
                    "Kaggle reported dataset creation failure despite returning exit code zero: "
                    f"{(created.stdout or created.stderr).strip()}"
                )
            self._wait_for_dataset_ready(dataset_ref)
            record = {
                **record,
                "phase": "dataset_uploaded",
                "dataset_ref": dataset_ref,
                "kernel_ref": kernel_ref,
            }
            write_job_record(self.state_root, job_id, record)
        if record.get("phase") == "dataset_uploaded" or resubmission_status is not None:
            # The machine shape is already explicit in kernel metadata. Passing
            # the CLI --accelerator override currently drops dataset_sources
            # from the submitted kernel, so metadata is the single authority.
            self._run(("kaggle", "kernels", "push", "-p", str(kernel_dir)))
            record = {
                **record,
                "phase": "submitted",
                "kernel_ref": kernel_ref,
                **(
                    {"provider_status_before_resubmission": resubmission_status}
                    if resubmission_status is not None
                    else {}
                ),
            }
            write_job_record(self.state_root, job_id, record)
        return record

    def status(self, job_id: str) -> dict[str, Any]:
        record = read_job_record(self.state_root, job_id)
        kernel_ref = record.get("kernel_ref")
        if not kernel_ref:
            return {**record, "provider_status": "not submitted"}
        completed = self._run(("kaggle", "kernels", "status", str(kernel_ref)))
        return {**record, "provider_status": completed.stdout.strip()}

    def follow_logs(self, job_id: str) -> int:
        record = read_job_record(self.state_root, job_id)
        kernel_ref = record.get("kernel_ref")
        if not kernel_ref:
            raise CloudPacketError("job has not been submitted to Kaggle")
        completed = self.runner(
            ("kaggle", "kernels", "logs", "-f", str(kernel_ref)),
            cwd=self.repo_root,
            capture_output=False,
        )
        return completed.returncode

    def fetch(self, job_id: str) -> dict[str, Any]:
        record = read_job_record(self.state_root, job_id)
        kernel_ref = record.get("kernel_ref")
        if not kernel_ref:
            raise CloudPacketError("job has not been submitted to Kaggle")
        destination = self.state_root / "training" / "cloud" / "jobs" / job_id / "outputs"
        destination.mkdir(parents=True, exist_ok=True)
        self._run(("kaggle", "kernels", "output", str(kernel_ref), "-p", str(destination), "-o"))
        result_path = destination / "axon_job_result.json"
        result = None
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("job_id") not in {None, job_id}:
                raise CloudPacketError("downloaded result belongs to another Axon job")
        updated = {
            **record,
            "phase": "outputs_fetched",
            "output_dir": str(destination),
            "result": result,
        }
        write_job_record(self.state_root, job_id, updated)
        return updated

    def jobs(self) -> list[dict[str, Any]]:
        root = self.state_root / "training" / "cloud" / "jobs"
        if not root.exists():
            return []
        values = []
        for path in sorted(root.glob("*/job.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema") == CLOUD_JOB_RECORD_SCHEMA:
                values.append(value)
        return values

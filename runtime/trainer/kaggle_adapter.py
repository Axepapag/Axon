"""Official Kaggle CLI adapter for durable Axon cloud-training jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .cloud_bundle import (
    SYNC_PAYLOAD_KEEP,
    parse_sync_step_range,
    prune_sync_payloads,
    sha256_file,
    verify_and_extract,
)
from .cloud_jobs import (
    CLOUD_JOB_RECORD_SCHEMA,
    CloudJobConfig,
    CloudPacketError,
    _atomic_json,
    prepare_cloud_job,
    read_job_record,
    write_job_record,
)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
KAGGLE_GPU_MACHINE_SHAPE = "NvidiaTeslaT4"
KAGGLE_DATASET_READY_ATTEMPTS = 24
KAGGLE_DATASET_READY_INTERVAL_SECONDS = 2.0


class SyncDatasetUnavailable(CloudPacketError):
    """The optional mid-run sync dataset has not been published yet."""


_LIVE_STATUS_RANK = {
    "running": 0,
    "queued": 1,
    "complete": 2,
    "submitted": 3,
    "unknown": 4,
    "error": 5,
    "fetched": 6,
    "not_submitted": 7,
}
_ENTRYPOINT_VALUE_FLAGS = {
    "--candidate-label": "candidate_label",
    "--tranche-steps": "tranche_steps",
    "--heads": "heads",
    "--layers": "layers",
    "--ffn-dim": "ffn_dim",
}


def classify_kaggle_provider_status(text: str | None) -> str:
    """Map Kaggle's status string to a short operator label."""

    raw = str(text or "").strip()
    if not raw or raw.lower() == "not submitted":
        return "not_submitted"
    upper = raw.upper()
    if "RUNNING" in upper:
        return "running"
    if "QUEUED" in upper or "PENDING" in upper:
        return "queued"
    if "COMPLETE" in upper or "SUCCESS" in upper:
        return "complete"
    if "CANCEL" in upper:
        return "error"
    if "ERROR" in upper or "FAILED" in upper:
        return "error"
    return "unknown"


def _entrypoint_details(argv: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {
        "candidate_label": None,
        "tranche_steps": None,
        "heads": None,
        "layers": None,
        "ffn_dim": None,
        "resume": False,
        "teach_multicell_copy": False,
        "receipt_continuation": False,
    }
    items = [str(item) for item in argv]
    index = 0
    while index < len(items):
        item = items[index]
        mapped = _ENTRYPOINT_VALUE_FLAGS.get(item)
        if mapped is not None and index + 1 < len(items):
            values[mapped] = items[index + 1]
            index += 2
            continue
        if item == "--resume":
            values["resume"] = True
        elif item == "--teach-multicell-copy":
            values["teach_multicell_copy"] = True
        elif item == "--receipt-continuation":
            values["receipt_continuation"] = True
        index += 1
    return values


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
        errors="replace",
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


def write_output_archive(job_id: str) -> dict:
    """Tar the complete kernel output tree with a detached SHA256 manifest."""
    # Load the bundling module straight from the packet: the notebook
    # interpreter is not the CUDA-probed training interpreter and may lack the
    # training stack, so importing the runtime.trainer package is unsafe here.
    import importlib.util

    module_path = WORK / "runtime" / "trainer" / "cloud_bundle.py"
    spec = importlib.util.spec_from_file_location("axon_cloud_bundle", module_path)
    cloud_bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud_bundle)

    root = Path("/kaggle/working")
    bundle_path = root / f"axon_outputs_{{job_id}}.tar.gz"
    manifest_path = root / f"axon_outputs_{{job_id}}.sha256.json"
    members = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith("axon_outputs_") or "axon_sync_staging" in path.parts:
            continue
        members.append((relative, path))
    document = cloud_bundle.write_bundle(
        bundle_path,
        manifest_path,
        members,
        kind="outputs",
        job_id=job_id,
    )
    return {{
        "bundle": bundle_path.name,
        "archive_sha256": document["archive_sha256"],
        "member_count": len(document["members"]),
    }}


def try_write_output_archive(job_id: str) -> None:
    """Archive failures are loud receipts; they never mask the training result."""
    try:
        publish("outputs_archived", **write_output_archive(job_id))
    except Exception as exc:
        publish("outputs_archive_failed", error_type=type(exc).__name__, error=str(exc))


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
    if manifest["config"].get("sync_mid_run"):
        # Opt-in mid-run sync: the training process reads these two variables,
        # resolves credentials only from the kernel secret store, and runs as a
        # no-op (with one journal note) when the secret is absent.
        env["AXON_SYNC_MID_RUN"] = "1"
        env["AXON_SYNC_DATASET"] = "axon-job-" + manifest["job_id"][:8] + "-sync"
        publish("sync_enabled", dataset=env["AXON_SYNC_DATASET"])
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
    try_write_output_archive(manifest["job_id"])
    return completed.returncode


def run_guarded() -> int:
    try:
        return main()
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
        try:
            manifest_path = WORK / "axon_packet_manifest.json"
            if manifest_path.is_file():
                failed_job_id = json.loads(manifest_path.read_text(encoding="utf-8")).get("job_id")
                if failed_job_id:
                    try_write_output_archive(failed_job_id)
        except Exception:
            # The governed failure receipt above is the authoritative record;
            # a broken packet must not gain a second failure from archiving.
            pass
        raise


if __name__ == "__main__":
    exit_code = run_guarded()
    if exit_code:
        raise SystemExit(exit_code)
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
            try:
                completed = self._run(("kaggle", "datasets", "status", dataset_ref))
                last_status = completed.stdout.strip().lower()
            except CloudPacketError as exc:
                # Kaggle can 403 a fresh dataset's status for a short window
                # after creation (eventual consistency). Treat as not-ready
                # and retry within the existing attempt budget.
                last_status = f"status_unavailable: {exc}"
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
            # Mid-run sync is the only feature that needs kernel internet, and
            # only recipes that explicitly opt in get it.
            "enable_internet": bool(manifest["config"].get("sync_mid_run")),
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
        if record.get("sync_mid_run") and not record.get("sync_dataset_ref"):
            # The opt-in sync target is a private per-job dataset on the same
            # account; recording it binds the job record to its sync identity.
            record = {
                **record,
                "sync_dataset_ref": f"{self.resolved_owner}/axon-job-{job_id[:8]}-sync",
            }
            write_job_record(self.state_root, job_id, record)
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
            if not any(
                f"kernelworkerstatus.{status}" in normalized_status
                for status in ("complete", "error", "cancelled", "canceled")
            ):
                raise CloudPacketError(
                    "Kaggle job status is not a recognized terminal state; refusing resubmission: "
                    f"{resubmission_status}"
                )
        if record.get("phase") == "prepared":
            created = self._run(("kaggle", "datasets", "create", "-p", str(dataset_dir), "-r", "skip"))
            combined_output = f"{created.stdout}\n{created.stderr}".lower()
            if "dataset creation error" in combined_output:
                raise CloudPacketError(
                    "Kaggle reported dataset creation failure despite returning exit code zero: "
                    f"{(created.stdout or created.stderr).strip()}"
                )
            record = {
                **record,
                "phase": "dataset_uploaded",
                "dataset_ref": dataset_ref,
                "kernel_ref": kernel_ref,
            }
            # Persist upload success before waiting for provider indexing. A
            # transient readiness timeout must not cause duplicate creation.
            write_job_record(self.state_root, job_id, record)
        if record.get("phase") == "dataset_uploaded" or resubmission_status is not None:
            self._wait_for_dataset_ready(dataset_ref)
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

    def _move_into_place(self, source: Path, destination: Path) -> None:
        """Move a downloaded tree into canonical position (long-path aware)."""
        # Validate the exact generated transfer targets before any recursive
        # move/cleanup; never let an environment value or job reference widen it.
        source.resolve(strict=True).relative_to(
            (Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "axon_fetch").resolve(strict=True)
        )
        destination.resolve(strict=True).relative_to(self.state_root.resolve(strict=True))
        if os.name == "nt":
            completed = subprocess.run(
                [
                    "robocopy",
                    str(source),
                    str(destination),
                    "/E",
                    "/MOVE",
                    "/NFL",
                    "/NDL",
                    "/NJH",
                    "/NJS",
                    "/NP",
                ],
                capture_output=True,
                text=True,
            )
            # robocopy exit codes < 8 are success (1 = files copied)
            if completed.returncode >= 8:
                raise CloudPacketError(
                    f"robocopy failed moving outputs into canonical position: rc={completed.returncode}"
                )
        else:
            if source.exists():
                shutil.copytree(source, destination, dirs_exist_ok=True)

    def _fetch_bundle(self, job_id: str, kernel_ref: str, temp_root: Path, destination: Path) -> bool:
        """Bundle-first retrieval: two downloads, full rehash, then placement.

        Returns False when the kernel published no output bundle (legacy jobs),
        so the caller falls back to the per-file fetch.
        """
        bundle_name = f"axon_outputs_{job_id}.tar.gz"
        manifest_name = f"axon_outputs_{job_id}.sha256.json"
        temp_target = Path(tempfile.mkdtemp(prefix="axon_out_bundle_", dir=temp_root))
        self._run(
            (
                "kaggle",
                "kernels",
                "output",
                kernel_ref,
                "-p",
                str(temp_target),
                "-o",
                "--file-pattern",
                f"^axon_outputs_{job_id}",
            )
        )
        bundle_path = temp_target / bundle_name
        manifest_path = temp_target / manifest_name
        if not bundle_path.is_file() or not manifest_path.is_file():
            shutil.rmtree(temp_target, ignore_errors=True)
            return False
        job_dir = self.state_root / "training" / "cloud" / "jobs" / job_id
        staging = temp_target / "extracted"
        report = verify_and_extract(
            bundle_path,
            manifest_path,
            staging,
            quarantine_root=job_dir / "quarantine",
        )
        if not report["ok"]:
            shutil.rmtree(temp_target, ignore_errors=True)
            raise CloudPacketError(
                "output bundle failed hash verification and was quarantined: "
                f"{report['quarantine_dir']} ({'; '.join(report['mismatches'])})"
            )
        if report.get("job_id") != job_id:
            shutil.rmtree(temp_target, ignore_errors=True)
            raise CloudPacketError("output bundle manifest belongs to another Axon job")
        result_path = staging / "axon_job_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("job_id") not in {None, job_id}:
                raise CloudPacketError("downloaded result belongs to another Axon job")
        self._move_into_place(staging, destination)
        # Keep the verified transfer contract alongside the extracted outputs.
        shutil.copy2(manifest_path, destination / manifest_name)
        shutil.rmtree(temp_target, ignore_errors=True)
        return True

    def fetch(self, job_id: str, *, bundle: bool = True) -> dict[str, Any]:
        record = read_job_record(self.state_root, job_id)
        kernel_ref = record.get("kernel_ref")
        if not kernel_ref:
            raise CloudPacketError("job has not been submitted to Kaggle")
        destination = self.state_root / "training" / "cloud" / "jobs" / job_id / "outputs"
        destination.mkdir(parents=True, exist_ok=True)
        # Kaggle output trees contain Soul snapshot paths that exceed Windows
        # MAX_PATH when placed directly under the canonical job directory.
        # Download into a short temp directory first, then move the tree into
        # canonical position with robocopy (long-path aware on Windows).
        temp_root = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "axon_fetch"
        temp_root.mkdir(parents=True, exist_ok=True)
        fetch_mode = "legacy_per_file"
        if bundle and self._fetch_bundle(job_id, str(kernel_ref), temp_root, destination):
            fetch_mode = "bundle"
        if fetch_mode != "bundle":
            temp_target = Path(tempfile.mkdtemp(prefix="axon_out_", dir=temp_root))
            self._run(("kaggle", "kernels", "output", str(kernel_ref), "-p", str(temp_target), "-o"))
            self._move_into_place(temp_target, destination)
            shutil.rmtree(temp_target, ignore_errors=True)
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
            "fetch_mode": fetch_mode,
        }
        write_job_record(self.state_root, job_id, updated)
        return updated

    def _sync_root(self, job_id: str) -> Path:
        return self.state_root / "training" / "cloud" / "jobs" / job_id / "sync"

    def _sync_dataset_ref(self, job_id: str, record: dict[str, Any]) -> str:
        ref = record.get("sync_dataset_ref")
        if ref:
            return str(ref)
        return f"{self.resolved_owner}/axon-job-{job_id[:8]}-sync"

    def sync_pull(self, job_id: str) -> dict[str, Any]:
        """Download the latest sync dataset version, verify, and extract it.

        Extraction lands under ``jobs/<job-id>/sync/`` — observation-only
        evidence, never canonical training State, never continuation authority.
        """
        record = read_job_record(self.state_root, job_id)
        dataset_ref = self._sync_dataset_ref(job_id, record)
        sync_root = self._sync_root(job_id)
        bundles_dir = sync_root / "bundles"
        members_dir = sync_root / "members"
        receipts_dir = sync_root / "receipts"
        temp_root = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "axon_fetch"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_target = Path(tempfile.mkdtemp(prefix="axon_sync_", dir=temp_root))
        try:
            try:
                self._run(
                    (
                        "kaggle",
                        "datasets",
                        "download",
                        "-d",
                        dataset_ref,
                        "-p",
                        str(temp_target),
                        "--unzip",
                        "-o",
                    )
                )
            except CloudPacketError as exc:
                message = str(exc).lower()
                if any(
                    marker in message
                    for marker in ("404", "not found", "does not exist", "no such dataset")
                ):
                    raise SyncDatasetUnavailable(
                        f"mid-run sync dataset is not available yet: {dataset_ref}"
                    ) from exc
                raise
            pulled: list[dict[str, Any]] = []
            already_present: list[str] = []
            for bundle_path in sorted(temp_target.glob("sync_*_steps_*_*.tar.gz")):
                manifest_path = temp_target / bundle_path.name.replace(
                    ".tar.gz", ".sha256.json"
                )
                step_range = parse_sync_step_range(bundle_path.name)
                name = bundle_path.name.replace(".tar.gz", "")
                if step_range is None or not manifest_path.is_file():
                    raise CloudPacketError(
                        f"sync dataset carries an incomplete bundle: {bundle_path.name}"
                    )
                receipt_path = receipts_dir / f"{name}.json"
                if receipt_path.is_file():
                    existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if existing.get("archive_sha256") == sha256_file(bundle_path):
                        already_present.append(bundle_path.name)
                        continue
                    raise CloudPacketError(f"a different bundle already occupies {name}")
                report = verify_and_extract(
                    bundle_path,
                    manifest_path,
                    members_dir,
                    quarantine_root=sync_root / "quarantine",
                )
                if not report["ok"]:
                    raise CloudPacketError(
                        "sync bundle failed hash verification and was quarantined: "
                        f"{report['quarantine_dir']} ({'; '.join(report['mismatches'])})"
                    )
                if report.get("job_id") != job_id:
                    raise CloudPacketError("sync bundle belongs to another Axon job")
                bundles_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(bundle_path), str(bundles_dir / bundle_path.name))
                shutil.move(str(manifest_path), str(bundles_dir / manifest_path.name))
                manifest_doc = json.loads(
                    (bundles_dir / manifest_path.name).read_text(encoding="utf-8")
                )
                receipt = {
                    "schema": "axon-mid-run-sync-local-receipt-v1",
                    "job_id": job_id,
                    "dataset_ref": dataset_ref,
                    "bundle_name": bundle_path.name,
                    "step_range": list(step_range),
                    "archive_sha256": manifest_doc["archive_sha256"],
                    "member_count": report["member_count"],
                    "observation_only": True,
                }
                _atomic_json(receipt_path, receipt)
                pulled.append(receipt)
            retention = prune_sync_payloads(sync_root, keep=SYNC_PAYLOAD_KEEP)
            return {
                "schema": "axon-mid-run-sync-pull-v1",
                "job_id": job_id,
                "sync_dataset_ref": dataset_ref,
                "sync_root": str(sync_root),
                "pulled": pulled,
                "already_present": already_present,
                "payload_retention": retention,
                "observation_only": True,
            }
        finally:
            shutil.rmtree(temp_target, ignore_errors=True)

    def sync_status(self, job_id: str, *, rehash: bool = True) -> dict[str, Any]:
        """Report which synced step ranges are locally verified.

        With ``rehash`` (the default) every recorded member is rehashed against
        its manifest before a range is reported as verified; anything failing
        is listed, never silently skipped.
        """
        record = read_job_record(self.state_root, job_id)
        sync_root = self._sync_root(job_id)
        bundles_dir = sync_root / "bundles"
        members_dir = sync_root / "members"
        receipts_dir = sync_root / "receipts"
        quarantine_dir = sync_root / "quarantine"
        verified_ranges: list[list[int]] = []
        mismatches: list[str] = []
        member_total = 0
        released_ranges: list[list[int]] = []
        retention_keep = SYNC_PAYLOAD_KEEP
        retention_path = sync_root / "payload_retention.json"
        released_set: set[tuple[int, int]] = set()
        if retention_path.is_file():
            try:
                retention = json.loads(retention_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                retention = {}
            if isinstance(retention, dict):
                if isinstance(retention.get("keep"), int) and retention["keep"] >= 1:
                    retention_keep = int(retention["keep"])
                for item in retention.get("released_ranges") or []:
                    if isinstance(item, list) and len(item) == 2:
                        start, end = int(item[0]), int(item[1])
                        released_set.add((start, end))
                        released_ranges.append([start, end])
        released_ranges.sort()
        if receipts_dir.is_dir():
            for receipt_path in sorted(receipts_dir.glob("sync_*_steps_*_*.json")):
                # A receipt that does not parse flags the range, never skips it.
                try:
                    json.loads(receipt_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    mismatches.append(f"corrupt sync receipt: {receipt_path.name}")
                    continue
                step_range = parse_sync_step_range(receipt_path.name)
                if step_range is None:
                    mismatches.append(f"unparseable sync receipt: {receipt_path.name}")
                    continue
                start, end = step_range
                if (start, end) in released_set:
                    continue
                manifest_path = bundles_dir / f"sync_{job_id}_steps_{start}_{end}.sha256.json"
                if not manifest_path.is_file():
                    mismatches.append(f"sync manifest missing for steps {start}-{end}")
                    continue
                members = json.loads(manifest_path.read_text(encoding="utf-8"))["members"]
                failed = []
                if rehash:
                    for arcname, member in sorted(members.items()):
                        candidate = members_dir / Path(*arcname.split("/"))
                        if not candidate.is_file() or sha256_file(candidate) != member["sha256"]:
                            failed.append(arcname)
                if failed:
                    mismatches.append(
                        f"sync bundle steps {start}-{end} failed rehash: {', '.join(failed)}"
                    )
                    continue
                member_total += len(members)
                verified_ranges.append([start, end])
        verified_ranges.sort()
        verified_through = None
        if verified_ranges:
            verified_through = verified_ranges[0][1]
            for start, end in verified_ranges[1:]:
                if start <= verified_through + 1:
                    verified_through = max(verified_through, end)
        quarantined = (
            sorted(path.name for path in quarantine_dir.iterdir()) if quarantine_dir.is_dir() else []
        )
        return {
            "schema": "axon-mid-run-sync-status-v1",
            "job_id": job_id,
            "sync_dataset_ref": record.get("sync_dataset_ref"),
            "sync_root": str(sync_root),
            "verified_ranges": verified_ranges,
            "verified_through_step": verified_through,
            "verified_member_count": member_total,
            "released_ranges": released_ranges,
            "payload_keep": retention_keep,
            "rehashed": rehash,
            "mismatches": mismatches,
            "quarantined": quarantined,
            "observation_only": True,
        }

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

    def job_catalog(self, *, refresh_live: bool = False, live_limit: int = 12) -> list[dict[str, Any]]:
        """Local jobs with recipe names, newest first, running jobs on top when live."""

        rows: list[dict[str, Any]] = []
        root = self.state_root / "training" / "cloud" / "jobs"
        for record in self.jobs():
            job_id = str(record.get("job_id") or "")
            job_dir = root / job_id
            manifest_path = job_dir / "packet_manifest.json"
            config: dict[str, Any] = {}
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(manifest.get("config"), dict):
                        config = manifest["config"]
                except (OSError, json.JSONDecodeError):
                    config = {}
            details = _entrypoint_details(config.get("entrypoint_argv") or ())
            try:
                mtime = (job_dir / "job.json").stat().st_mtime
            except OSError:
                mtime = 0.0
            phase = str(record.get("phase") or "unknown")
            live_status = {
                "outputs_fetched": "fetched",
                "prepared": "not_submitted",
                "submitted": "submitted",
            }.get(phase, phase)
            heads = details["heads"]
            layers = details["layers"]
            ffn_dim = details["ffn_dim"]
            shape = None
            if heads and layers and ffn_dim:
                shape = f"{heads}h/{layers}L/FFN{ffn_dim}"
            rows.append(
                {
                    "schema": "axon-kaggle-job-catalog-row-v1",
                    "job_id": job_id,
                    "name": str(config.get("name") or f"Axon job {job_id[:16]}"),
                    "phase": phase,
                    "accelerator": record.get("accelerator") or config.get("accelerator"),
                    "kernel_ref": record.get("kernel_ref"),
                    "candidate_label": details["candidate_label"],
                    "tranche_steps": details["tranche_steps"],
                    "shape": shape,
                    "resume": bool(details["resume"]),
                    "teach_multicell_copy": bool(details["teach_multicell_copy"]),
                    "receipt_continuation": bool(details["receipt_continuation"]),
                    "mtime": mtime,
                    "provider_status": None,
                    "live_status": live_status,
                }
            )
        rows.sort(key=lambda row: float(row["mtime"]), reverse=True)
        if refresh_live:
            probed = 0
            for row in rows:
                if probed >= live_limit:
                    break
                if not row.get("kernel_ref"):
                    continue
                if row["phase"] not in {"submitted", "dataset_uploaded"}:
                    continue
                try:
                    status = self.status(str(row["job_id"]))
                except CloudPacketError:
                    probed += 1
                    continue
                provider_status = str(status.get("provider_status") or "").strip()
                if not provider_status:
                    probed += 1
                    continue
                row["provider_status"] = provider_status
                classified = classify_kaggle_provider_status(provider_status)
                if classified != "not_submitted":
                    row["live_status"] = classified
                probed += 1
        rows.sort(
            key=lambda row: (
                _LIVE_STATUS_RANK.get(str(row.get("live_status")), 9),
                -float(row["mtime"]),
            )
        )
        return rows

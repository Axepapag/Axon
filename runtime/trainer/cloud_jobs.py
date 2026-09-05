"""Provider-neutral, content-addressed cloud training packets.

Packets contain committed executable tissue plus explicitly named operational
State.  They never inherit a workstation environment or credentials.  A
provider adapter may transport a packet, but provider/resource choices do not
alter Axon's candidate, plan, or learning-policy identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from runtime.field import canonical_sha256

CLOUD_JOB_CONFIG_SCHEMA = "axon-cloud-training-job-config-v1"
CLOUD_PACKET_SCHEMA = "axon-cloud-training-packet-v1"
CLOUD_JOB_RECORD_SCHEMA = "axon-cloud-training-job-record-v1"

_SOURCE_ROOTS = (
    "adapters/",
    "configs/",
    "cores/",
    "curator/",
    "runtime/",
    "scripts/",
    "slots/",
    "substrate/",
    "training/",
)
_SOURCE_FILES = {"pyproject.toml"}
_PROHIBITED_NAMES = {
    ".env",
    "access_token",
    "credentials.json",
    "kaggle.json",
    "id_rsa",
    "id_ed25519",
}
_PROHIBITED_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


class CloudPacketError(RuntimeError):
    """A cloud packet could not be proven safe and reproducible."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    body = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("cloud job name must contain a letter or number")
    return slug[:40]


def _assert_safe_file(path: Path, repo_root: Path) -> str:
    if path.is_symlink():
        raise CloudPacketError(f"symbolic links are not allowed in cloud packets: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise CloudPacketError(f"cloud packet input escapes the Axon repository: {path}") from exc
    parts = {part.lower() for part in PurePosixPath(relative).parts}
    lower_name = resolved.name.lower()
    if (
        lower_name in _PROHIBITED_NAMES
        or lower_name.endswith(tuple(_PROHIBITED_SUFFIXES))
        or ".git" in parts
        or ".kaggle" in parts
    ):
        raise CloudPacketError(f"credential-bearing path is prohibited: {relative}")
    return relative


def _expand_inputs(repo_root: Path, requested: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    expanded: dict[str, Path] = {}
    for raw in requested:
        candidate = (repo_root / str(raw)).resolve(strict=False)
        try:
            candidate.relative_to(repo_root)
        except ValueError as exc:
            raise CloudPacketError(f"include path escapes Axon: {raw}") from exc
        if not candidate.exists():
            raise CloudPacketError(f"required cloud input does not exist: {raw}")
        paths: Iterable[Path] = (
            (item for item in candidate.rglob("*") if item.is_file()) if candidate.is_dir() else (candidate,)
        )
        for path in paths:
            relative = _assert_safe_file(path, repo_root)
            expanded[relative] = path
    return tuple(sorted(expanded.items()))


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode:
        message = (completed.stderr or completed.stdout).strip()
        raise CloudPacketError(f"git {' '.join(arguments)} failed: {message}")
    return completed.stdout.strip()


def _committed_source(repo_root: Path) -> tuple[str, tuple[tuple[str, Path], ...]]:
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise CloudPacketError("tracked Axon files are modified; commit them before exporting a cloud packet")
    revision = _git(repo_root, "rev-parse", "HEAD")
    tracked = _git(repo_root, "ls-files", "-z").split("\0")
    selected: list[tuple[str, Path]] = []
    for relative in tracked:
        normalized = relative.replace("\\", "/")
        if not normalized:
            continue
        if normalized not in _SOURCE_FILES and not normalized.startswith(_SOURCE_ROOTS):
            continue
        path = repo_root / normalized
        if not path.is_file():
            raise CloudPacketError(f"tracked source is missing: {normalized}")
        _assert_safe_file(path, repo_root)
        selected.append((normalized, path))
    if not selected:
        raise CloudPacketError("no executable Axon source was selected")
    return revision, tuple(sorted(selected))


@dataclass(frozen=True, slots=True)
class CloudJobConfig:
    name: str
    provider: str
    accelerator: str
    entrypoint_argv: tuple[str, ...]
    include_paths: tuple[str, ...]
    allow_sensitive_state_upload: bool = False
    notes: str = ""
    sync_mid_run: bool = False
    config_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("cloud job name must be non-empty")
        if self.provider != "kaggle":
            raise ValueError("only the Kaggle adapter is currently implemented")
        if self.accelerator not in {"cpu", "gpu", "tpu"}:
            raise ValueError("accelerator must be cpu, gpu, or tpu")
        argv = tuple(str(item) for item in self.entrypoint_argv)
        if len(argv) < 2 or argv[0] not in {"python", "python3"}:
            raise ValueError("entrypoint_argv must begin with python and a repository script")
        script = PurePosixPath(argv[1].replace("\\", "/"))
        if script.is_absolute() or ".." in script.parts or script.parts[0] != "scripts":
            raise ValueError("cloud entrypoint must be a repository script under scripts/")
        includes = tuple(str(item).replace("\\", "/").strip("/") for item in self.include_paths)
        if any(not item for item in includes):
            raise ValueError("include_paths cannot contain empty paths")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "entrypoint_argv", argv)
        object.__setattr__(self, "include_paths", includes)
        object.__setattr__(self, "notes", str(self.notes))
        object.__setattr__(self, "sync_mid_run", bool(self.sync_mid_run))
        object.__setattr__(self, "config_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": CLOUD_JOB_CONFIG_SCHEMA,
            "name": self.name,
            "provider": self.provider,
            "accelerator": self.accelerator,
            "entrypoint_argv": list(self.entrypoint_argv),
            "include_paths": list(self.include_paths),
            "allow_sensitive_state_upload": self.allow_sensitive_state_upload,
            "notes": self.notes,
        }
        if self.sync_mid_run:
            # Opt-in keys join the identity only when enabled, so existing
            # recipes keep their historical config and job identities.
            value["sync_mid_run"] = True
        if include_id:
            value["config_id"] = self.config_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CloudJobConfig":
        item = dict(value)
        if item.pop("schema", None) != CLOUD_JOB_CONFIG_SCHEMA:
            raise ValueError("unsupported cloud job config schema")
        expected_id = item.pop("config_id", None)
        item["entrypoint_argv"] = tuple(item["entrypoint_argv"])
        item["include_paths"] = tuple(item.get("include_paths", ()))
        item["sync_mid_run"] = bool(item.get("sync_mid_run", False))
        config = cls(**item)
        if expected_id is not None and expected_id != config.config_id:
            raise ValueError("cloud job config identity mismatch")
        return config

    @classmethod
    def read(cls, path: Path | str) -> "CloudJobConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class PreparedCloudJob:
    job_id: str
    job_dir: Path
    packet_path: Path
    manifest_path: Path
    revision: str
    file_count: int
    total_bytes: int
    config: CloudJobConfig


def prepare_cloud_job(
    *,
    repo_root: Path | str,
    state_root: Path | str,
    config: CloudJobConfig,
) -> PreparedCloudJob:
    """Build one immutable local packet.  This does not upload or spend quota."""

    repo = Path(repo_root).resolve(strict=True)
    state = Path(state_root).resolve(strict=False)
    try:
        state.relative_to(repo)
    except ValueError as exc:
        raise CloudPacketError("State root must live inside the Axon repository") from exc
    revision, source = _committed_source(repo)
    explicit = _expand_inputs(repo, config.include_paths)
    sensitive = tuple(relative for relative, _path in explicit if relative.startswith("State/"))
    if sensitive and not config.allow_sensitive_state_upload:
        raise CloudPacketError(
            "State was selected but allow_sensitive_state_upload is false; "
            "this explicit acknowledgement is required before any provider upload"
        )
    files: dict[str, Path] = {relative: path for relative, path in source}
    files.update({relative: path for relative, path in explicit})
    records = [
        {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "role": "operational_state" if relative.startswith("State/") else "committed_source",
        }
        for relative, path in sorted(files.items())
    ]
    identity = {
        "schema": CLOUD_PACKET_SCHEMA,
        "config": config.to_canonical_dict(),
        "git_revision": revision,
        "files": records,
        "credential_files_included": False,
        "public_visibility_allowed": False,
    }
    job_id = canonical_sha256(identity)
    manifest = {**identity, "job_id": job_id}
    job_dir = state / "training" / "cloud" / "jobs" / job_id
    packet_path = job_dir / "packet" / "axon_packet.zip"
    manifest_path = job_dir / "packet_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise CloudPacketError("existing cloud job identity disagrees with prepared manifest")
    else:
        _atomic_json(manifest_path, manifest)
    if not packet_path.exists():
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = packet_path.with_name(packet_path.name + f".{os.getpid()}.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
            manifest_info = zipfile.ZipInfo("axon_packet_manifest.json", (2026, 1, 1, 0, 0, 0))
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(manifest_info, manifest_bytes)
            for relative, path in sorted(files.items()):
                info = zipfile.ZipInfo(relative, (2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with path.open("rb") as handle:
                    archive.writestr(info, handle.read())
        os.replace(temporary, packet_path)
    record = {
        "schema": CLOUD_JOB_RECORD_SCHEMA,
        "job_id": job_id,
        "provider": config.provider,
        "accelerator": config.accelerator,
        "phase": "prepared",
        "git_revision": revision,
        "packet_path": str(packet_path),
        "packet_sha256": _sha256_file(packet_path),
        "dataset_ref": None,
        "kernel_ref": None,
        "public": False,
        "sync_mid_run": config.sync_mid_run,
    }
    _atomic_json(job_dir / "job.json", record)
    return PreparedCloudJob(
        job_id=job_id,
        job_dir=job_dir,
        packet_path=packet_path,
        manifest_path=manifest_path,
        revision=revision,
        file_count=len(records),
        total_bytes=sum(item["bytes"] for item in records),
        config=config,
    )


def read_job_record(state_root: Path | str, job_id: str) -> dict[str, Any]:
    path = Path(state_root) / "training" / "cloud" / "jobs" / job_id / "job.json"
    if not path.is_file():
        raise CloudPacketError(f"unknown cloud job {job_id}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != CLOUD_JOB_RECORD_SCHEMA or value.get("job_id") != job_id:
        raise CloudPacketError("cloud job record is invalid")
    return value


def write_job_record(state_root: Path | str, job_id: str, value: Mapping[str, Any]) -> None:
    path = Path(state_root) / "training" / "cloud" / "jobs" / job_id / "job.json"
    _atomic_json(path, value)

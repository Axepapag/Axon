"""Hash-manifested artifact bundles for Axon cloud training transport.

One implementation serves both directions of the mid-run sync design:

* the Kaggle-side packet writes bundles (the end-of-run outputs archive and
  the opt-in mid-run sync datasets), and
* the local side verifies and extracts them.

The detached manifest is the transfer contract: every member path maps to its
SHA256, and the archive itself carries a bundle-level hash.  Local import
rehashes every member before any record is written; any mismatch quarantines
the bundle and flags, never a silent skip.

Synced mid-run artifacts are observation-only: they never write into canonical
training State, never authorize a continuation, and never relax the
exact-parent tranche-renewal rule.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import queue
import shutil
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

CLOUD_BUNDLE_MANIFEST_SCHEMA = "axon-cloud-bundle-manifest-v1"
CLOUD_BUNDLE_REPORT_SCHEMA = "axon-cloud-bundle-verification-v1"
SYNC_RECEIPT_SCHEMA = "axon-mid-run-sync-receipt-v1"
SYNC_PAYLOAD_RETENTION_SCHEMA = "axon-mid-run-sync-payload-retention-v1"
SYNC_SECRET_LABEL = "AXON_KAGGLE_SYNC"
SYNC_PAYLOAD_KEEP = 3
"""Observation payload windows retained locally (receipts and manifests stay)."""
# Fixed archive epoch, matching the packet ZIP convention in cloud_jobs.py.
_ARCHIVE_EPOCH = 1767225600  # 2026-01-01T00:00:00Z


class CloudBundleError(RuntimeError):
    """A bundle or its manifest failed the transfer contract."""


class SyncCredentialsMissing(CloudBundleError):
    """The sync secret/env is absent; sync must disable, never fail training."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _io_path(path: Path | str) -> Path:
    """Return an absolute Windows extended path for filesystem I/O.

    Bundle member names remain ordinary relative POSIX paths for validation
    and provenance. Only the final local I/O path receives the Windows
    extended-path prefix, allowing immutable Trainer/Soul ancestry names to exceed the
    legacy Win32 260-character path limit without shortening or renaming any
    evidence.
    """

    resolved = Path(path).resolve(strict=False)
    if os.name != "nt":
        return resolved
    raw = str(resolved)
    if raw.startswith("\\\\?\\"):
        return Path(raw)
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    body = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _checked_arcname(arcname: str) -> PurePosixPath:
    """A bundle member name must be a relative POSIX path that cannot escape."""

    relative = PurePosixPath(arcname)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CloudBundleError(f"unsafe bundle member path: {arcname!r}")
    return relative


def write_bundle(
    bundle_path: Path | str,
    manifest_path: Path | str,
    members: Iterable[tuple[str, Path | str]],
    *,
    kind: str,
    job_id: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one deterministic tar.gz plus its detached SHA256 manifest.

    ``members`` are ``(archive_name, source_path)`` pairs.  Archive metadata
    (mtime, uid, names) is frozen so identical content yields an identical
    archive hash.  Returns the manifest document.
    """

    bundle = Path(bundle_path)
    manifest = Path(manifest_path)
    ordered = sorted(((str(name), Path(source)) for name, source in members), key=lambda item: item[0])
    if not ordered:
        raise CloudBundleError("refusing to write an empty bundle")
    names = [name for name, _source in ordered]
    if len(set(names)) != len(names):
        raise CloudBundleError("bundle member names must be unique")
    member_records: dict[str, Any] = {}
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_name(bundle.name + f".{os.getpid()}.tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=_ARCHIVE_EPOCH) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for arcname, source in ordered:
            _checked_arcname(arcname)
            if not source.is_file():
                raise CloudBundleError(f"bundle member is missing: {source}")
            body = source.read_bytes()
            record = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
            member_records[arcname] = record
            info = tarfile.TarInfo(arcname)
            info.size = len(body)
            info.mtime = _ARCHIVE_EPOCH
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(body))
    os.replace(temporary, bundle)
    document: dict[str, Any] = {
        "schema": CLOUD_BUNDLE_MANIFEST_SCHEMA,
        "kind": str(kind),
        "job_id": str(job_id),
        "bundle_name": bundle.name,
        "archive_sha256": sha256_file(bundle),
        "archive_bytes": bundle.stat().st_size,
        "members": member_records,
    }
    if extra:
        document["details"] = dict(extra)
    _atomic_json(manifest, document)
    return document


def _load_manifest(manifest_path: Path | str) -> dict[str, Any]:
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise CloudBundleError(f"bundle manifest is missing: {manifest}")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("schema") != CLOUD_BUNDLE_MANIFEST_SCHEMA:
        raise CloudBundleError(f"unsupported bundle manifest schema: {manifest}")
    if not isinstance(document.get("members"), dict) or not document["members"]:
        raise CloudBundleError(f"bundle manifest has no members: {manifest}")
    return document


def verify_and_extract(
    bundle_path: Path | str,
    manifest_path: Path | str,
    destination: Path | str,
    *,
    quarantine_root: Path | str,
) -> dict[str, Any]:
    """Verify the archive hash, extract, and rehash every member.

    On ANY mismatch the extracted members, the bundle, and the manifest are
    moved under ``quarantine_root`` and the returned report carries
    ``ok: False`` with the mismatches enumerated.  Nothing is silently
    skipped and nothing half-verified is left in ``destination``.
    """

    bundle = Path(bundle_path)
    manifest_file = Path(manifest_path)
    target = Path(destination)
    quarantine = Path(quarantine_root)
    mismatches: list[str] = []
    extracted: list[str] = []
    document: dict[str, Any] | None = None
    try:
        document = _load_manifest(manifest_file)
        if document.get("bundle_name") != bundle.name:
            mismatches.append(f"manifest names a different bundle: {document.get('bundle_name')!r}")
        if not bundle.is_file():
            mismatches.append(f"bundle archive is missing: {bundle}")
        else:
            if sha256_file(bundle) != document["archive_sha256"]:
                mismatches.append("archive sha256 mismatch")
            if bundle.stat().st_size != document["archive_bytes"]:
                mismatches.append("archive byte count mismatch")
        expected = {str(name) for name in document["members"]}
        for arcname in expected:
            _checked_arcname(arcname)
        observed: set[str] = set()
        if not mismatches:
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, mode="r:gz") as archive:
                for info in archive:
                    if not info.isfile():
                        mismatches.append(f"non-file archive member: {info.name}")
                        continue
                    relative = _checked_arcname(info.name)
                    observed.add(info.name)
                    member_target = (target / Path(*relative.parts)).resolve(strict=False)
                    member_target.relative_to(target.resolve(strict=False))
                    _io_path(member_target.parent).mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(info)
                    assert source is not None  # guaranteed by info.isfile()
                    with source, _io_path(member_target).open("wb") as handle:
                        shutil.copyfileobj(source, handle)
                    extracted.append(info.name)
            missing = expected - observed
            extra_members = observed - expected
            for name in sorted(missing):
                mismatches.append(f"manifest member absent from archive: {name}")
            for name in sorted(extra_members):
                mismatches.append(f"archive member absent from manifest: {name}")
        for arcname in sorted(expected & observed):
            record = document["members"][arcname]
            member_target = target / Path(*PurePosixPath(arcname).parts)
            member_io_target = _io_path(member_target)
            if sha256_file(member_io_target) != record["sha256"]:
                mismatches.append(f"member sha256 mismatch: {arcname}")
            elif member_io_target.stat().st_size != record["bytes"]:
                mismatches.append(f"member byte count mismatch: {arcname}")
    except (CloudBundleError, KeyError, OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        mismatches.append(f"{type(exc).__name__}: {exc}")
    report: dict[str, Any] = {
        "schema": CLOUD_BUNDLE_REPORT_SCHEMA,
        "bundle": str(bundle),
        "manifest": str(manifest_file),
        "destination": str(target),
        "job_id": None if document is None else document.get("job_id"),
        "kind": None if document is None else document.get("kind"),
        "member_count": 0 if document is None else len(document["members"]),
        "extracted_members": sorted(extracted),
        "mismatches": mismatches,
        "ok": not mismatches,
        "quarantine_dir": None,
    }
    if mismatches:
        # Quarantine-and-flag: preserve the offending evidence out-of-band,
        # never leave it where a later import could trust it.
        stall = quarantine / f"{bundle.name}.{int(time.time())}"
        members_stall = stall / "members"
        for arcname in extracted:
            relative = PurePosixPath(arcname)
            source = _io_path(target / Path(*relative.parts))
            if source.is_file():
                parked = members_stall / Path(*relative.parts)
                parked_io = _io_path(parked)
                _io_path(parked.parent).mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(parked_io))
        stall.mkdir(parents=True, exist_ok=True)
        for evidence in (bundle, manifest_file):
            if evidence.is_file():
                shutil.move(str(evidence), str(stall / evidence.name))
        report["quarantine_dir"] = str(stall)
    return report


class KaggleDatasetUploader:
    """Upload one staging folder as a private per-job sync dataset version.

    Credentials come from the kernel environment (Kaggle User Secrets inject
    ``KAGGLE_USERNAME``/``KAGGLE_KEY`` or the ``AXON_KAGGLE_SYNC`` secret as a
    JSON ``{"username", "key"}`` payload).  Token material is never printed,
    logged, or persisted by this class.
    """

    def __init__(
        self,
        dataset_slug: str,
        *,
        environ: Mapping[str, str] | None = None,
        api_factory: Any = None,
    ) -> None:
        self.dataset_slug = str(dataset_slug).strip()
        if not self.dataset_slug:
            raise ValueError("sync dataset slug must be non-empty")
        self._environ = os.environ if environ is None else environ
        self._api_factory = api_factory
        self._dataset_exists = False

    def _resolve_credentials(self) -> str:
        """Return the account username or raise SyncCredentialsMissing."""

        username = (self._environ.get("KAGGLE_USERNAME") or "").strip()
        key = (self._environ.get("KAGGLE_KEY") or "").strip()
        if username and key:
            return username
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
        except ImportError as exc:
            raise SyncCredentialsMissing("no Kaggle credential env and no User Secrets client") from exc
        try:
            payload = UserSecretsClient().get_secret(SYNC_SECRET_LABEL)
        except Exception as exc:
            raise SyncCredentialsMissing(
                f"Kaggle User Secret {SYNC_SECRET_LABEL} is not attached to this kernel"
            ) from exc
        try:
            parsed = json.loads(payload)
            username = str(parsed["username"]).strip()
            key = str(parsed["key"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SyncCredentialsMissing(
                f"Kaggle User Secret {SYNC_SECRET_LABEL} is not a username/key JSON payload"
            ) from exc
        if not username or not key:
            raise SyncCredentialsMissing(f"Kaggle User Secret {SYNC_SECRET_LABEL} is empty")
        # The official client authenticates from these exact env names.
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        return username

    def _api(self) -> Any:
        if self._api_factory is not None:
            return self._api_factory()
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

        api = KaggleApi()
        api.authenticate()
        return api

    def upload(self, folder: Path | str, *, version_notes: str) -> None:
        directory = Path(folder)
        username = self._resolve_credentials()
        metadata = {
            "title": f"Axon job mid-run sync ({self.dataset_slug})",
            "id": f"{username}/{self.dataset_slug}",
            "licenses": [{"name": "other"}],
            "description": (
                "Private, observation-only Axon mid-run training sync bundles. "
                "All rights reserved; not licensed for public redistribution. "
                "Synced artifacts never authorize continuation on their own."
            ),
        }
        (directory / "dataset-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        api = self._api()
        if not self._dataset_exists:
            try:
                api.dataset_create_new(str(directory))
                self._dataset_exists = True
                return
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" not in message and "409" not in message and "conflict" not in message:
                    raise
        api.dataset_create_version(str(directory), version_notes, delete_old_versions=False)
        self._dataset_exists = True


class MidRunSyncHook:
    """Bundle new artifacts at accepted checkpoint boundaries and upload them.

    Uploads run on a single daemon worker so GPU compute never blocks on the
    network.  A failed upload is recorded as a receipt and retried at the next
    boundary with an extended step range; it is never a training failure.  A
    missing secret disables the hook with exactly one journal note.
    """

    def __init__(
        self,
        *,
        job_id: str,
        dataset_slug: str,
        state_root: Path | str,
        staging_root: Path | str,
        receipt_log: Path | str | None = None,
        uploader: Any = None,
        enabled: bool = True,
        close_timeout: float = 300.0,
    ) -> None:
        self.job_id = str(job_id).strip()
        if not self.job_id:
            raise ValueError("job_id must be non-empty")
        self.dataset_slug = str(dataset_slug).strip()
        self.state_root = Path(state_root).resolve(strict=False)
        self.staging_root = Path(staging_root).resolve(strict=False)
        self.receipt_log = None if receipt_log is None else Path(receipt_log).resolve(strict=False)
        self.close_timeout = float(close_timeout)
        self._enabled = bool(enabled)
        self._mirror_stdout = self._enabled
        self._uploader = uploader
        if self._enabled and self._uploader is None:
            self._uploader = KaggleDatasetUploader(self.dataset_slug)
        self._baseline = self._snapshot() if self._enabled else frozenset()
        self._pending: dict[str, Path] = {}
        self._synced: set[str] = set()
        self._range_start: int | None = None
        self._disabled_note_written = False
        self._lock = threading.Lock()
        # Each queue item freezes the paths first observed at that exact
        # checkpoint boundary. The worker assigns the range start only after
        # the preceding upload outcome is known. This prevents a slow upload
        # from leaking later-step artifacts into an earlier-labelled bundle.
        self._last_enqueued_end: int | None = None
        self._queue: queue.Queue[tuple[int, dict[str, Path]] | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        job_id: str,
        state_root: Path | str,
        staging_root: Path | str,
        receipt_log: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        uploader: Any = None,
    ) -> "MidRunSyncHook":
        """Build the hook from the packet's injected sync environment.

        Without ``AXON_SYNC_MID_RUN=1`` and ``AXON_SYNC_DATASET`` the hook is a
        no-op with one journal note: the packet runs correctly with sync
        disabled, per the ratified no-secret requirement.
        """

        env = os.environ if environ is None else environ
        slug = (env.get("AXON_SYNC_DATASET") or "").strip()
        if env.get("AXON_SYNC_MID_RUN") != "1" or not slug:
            hook = cls(
                job_id=job_id,
                dataset_slug=slug or "disabled",
                state_root=state_root,
                staging_root=staging_root,
                receipt_log=receipt_log,
                enabled=False,
            )
            hook.note_disabled("mid-run sync environment is not set; running with sync disabled")
            return hook
        return cls(
            job_id=job_id,
            dataset_slug=slug,
            state_root=state_root,
            staging_root=staging_root,
            receipt_log=receipt_log,
            uploader=uploader,
            enabled=True,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _snapshot(self) -> frozenset[str]:
        if not self.state_root.is_dir():
            return frozenset()
        return frozenset(
            path.relative_to(self.state_root).as_posix()
            for path in self.state_root.rglob("*")
            if path.is_file()
        )

    def _receipt(self, status: str, **details: Any) -> dict[str, Any]:
        receipt = {
            "schema": SYNC_RECEIPT_SCHEMA,
            "job_id": self.job_id,
            "dataset_slug": self.dataset_slug,
            "status": status,
            "unix_seconds": time.time(),
            "details": details,
        }
        if self.receipt_log is not None:
            self.receipt_log.parent.mkdir(parents=True, exist_ok=True)
            with self.receipt_log.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        if self._mirror_stdout:
            # The console mirror is reserved for the sync-enabled packet path:
            # local harnesses parse the trainer's stdout as pure JSON, and a
            # hook that was never enabled must not pollute it.  The line never
            # carries credential material, only identities and types.
            print("AXON_SYNC " + json.dumps(receipt, ensure_ascii=True, sort_keys=True), flush=True)
        return receipt

    def note_disabled(self, reason: str) -> None:
        """Record the single no-secret journal note required by the proposal."""

        if self._disabled_note_written:
            return
        self._disabled_note_written = True
        self._receipt("disabled", reason=reason)

    def set_base_step(self, base_step: int) -> None:
        """Anchor the first sync range at the segment's exact base step."""

        if not isinstance(base_step, int) or base_step < 0:
            raise ValueError("base step must be a non-negative integer")
        with self._lock:
            self._range_start = base_step
            self._last_enqueued_end = base_step

    def _collect_new(self) -> dict[str, Path]:
        discovered: dict[str, Path] = {}
        for path in sorted(self.state_root.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            try:
                path.relative_to(self.staging_root)
                continue  # our own staging is never a sync artifact
            except ValueError:
                pass
            relative = path.relative_to(self.state_root).as_posix()
            if (
                relative in self._baseline
                or relative in self._synced
                or relative in self._pending
            ):
                continue
            if self.receipt_log is not None and path == self.receipt_log:
                continue
            discovered[relative] = path
        self._pending.update(discovered)
        return discovered

    def boundary(self, end_step: int) -> None:
        """Enqueue one sync range ending at an accepted checkpoint boundary."""

        if not self._enabled:
            return
        with self._lock:
            if self._range_start is None or self._last_enqueued_end is None:
                raise RuntimeError("set_base_step must be called before the first sync boundary")
            if end_step <= self._last_enqueued_end:
                return
            item = (end_step, self._collect_new())
            self._last_enqueued_end = end_step
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._work,
                    name=f"axon-sync-{self.job_id[:8]}",
                    daemon=True,
                )
                self._worker.start()
        self._queue.put(item)

    def _work(self) -> None:
        carried: dict[str, Path] = {}
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            end, newly_observed = item
            carried.update(newly_observed)
            if not self._enabled:
                self._queue.task_done()
                continue
            with self._lock:
                if self._range_start is None:
                    self._queue.task_done()
                    raise RuntimeError("sync worker has no base step")
                start = self._range_start + 1
            try:
                self._upload_range(start, end, sorted(carried.items()))
            except SyncCredentialsMissing as exc:
                # Permanent for this kernel: disable with one journal note and
                # let training continue exactly as if sync were never enabled.
                self._enabled = False
                self.note_disabled(f"sync credentials unavailable: {type(exc).__name__}")
            except Exception as exc:
                # Transient provider/network failure: receipt only, retried at
                # the next boundary with an extended range.  Never raises.
                self._receipt(
                    "failed",
                    step_range=[start, end],
                    error_type=type(exc).__name__,
                )
            else:
                carried.clear()
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Wait until every queued boundary has been processed."""

        self._queue.join()

    def _upload_range(self, start: int, end: int, members: list[tuple[str, Path]]) -> None:
        if not members:
            self._receipt("empty", step_range=[start, end])
            with self._lock:
                self._range_start = end
            return
        name = f"sync_{self.job_id}_steps_{start}_{end}"
        staging = self.staging_root / name
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            document = write_bundle(
                staging / f"{name}.tar.gz",
                staging / f"{name}.sha256.json",
                members,
                kind="mid_run_sync",
                job_id=self.job_id,
                extra={"step_range": [start, end]},
            )
            self._uploader.upload(staging, version_notes=f"steps {start}-{end}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        uploaded = {arcname for arcname, _source in members}
        with self._lock:
            for arcname in uploaded:
                self._pending.pop(arcname, None)
            self._synced.update(uploaded)
            self._range_start = end
        self._receipt(
            "uploaded",
            step_range=[start, end],
            bundle_name=f"{name}.tar.gz",
            archive_sha256=document["archive_sha256"],
            member_count=len(document["members"]),
        )

    def close(self, *, timeout: float | None = None) -> None:
        """Drain pending uploads; a daemon worker never blocks kernel exit."""

        if self._worker is None:
            return
        self._queue.put(None)
        self._worker.join(self.close_timeout if timeout is None else timeout)
        if self._worker.is_alive():
            self._receipt(
                "close_timeout",
                note="an upload was still in flight at close; the daemon thread does not block exit",
            )


def sync_bundle_name(job_id: str, start: int, end: int) -> str:
    return f"sync_{job_id}_steps_{start}_{end}.tar.gz"


def parse_sync_step_range(name: str) -> tuple[int, int] | None:
    """Extract ``(start, end)`` from a sync bundle or manifest filename."""

    stem = Path(name).name
    for suffix in (".tar.gz", ".sha256.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("_steps_")
    if len(parts) != 2 or not parts[0].startswith("sync_"):
        return None
    bounds = parts[1].split("_")
    if len(bounds) != 2:
        return None
    try:
        start, end = int(bounds[0]), int(bounds[1])
    except ValueError:
        return None
    return (start, end) if start <= end else None


def _sync_job_id_from_name(name: str) -> str:
    """Recover the job id from a ``sync_<job>_steps_<a>_<b>`` filename."""

    stem = Path(name).name
    for suffix in (".tar.gz", ".sha256.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if not stem.startswith("sync_") or "_steps_" not in stem:
        return ""
    return stem[len("sync_") :].rsplit("_steps_", 1)[0]


def prune_sync_payloads(
    sync_root: Path | str,
    *,
    keep: int = SYNC_PAYLOAD_KEEP,
) -> dict[str, Any]:
    """Release observation payload bytes older than the last ``keep`` windows.

    Receipts and manifests remain. This is disk paging of mid-run evidence,
    never a tissue ceiling and never continuation authority. Canonical
    ``checkpoint_records`` and accepted bundles are not in this tree and are
    never touched.
    """

    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("keep must be a positive integer")
    root = Path(sync_root)
    receipts_dir = root / "receipts"
    bundles_dir = root / "bundles"
    members_dir = root / "members"
    items: list[tuple[list[int], str, dict[str, Any], Path]] = []
    if receipts_dir.is_dir():
        for path in sorted(receipts_dir.glob("sync_*_steps_*_*.json")):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(receipt, dict):
                continue
            parsed = parse_sync_step_range(path.name)
            if parsed is None:
                continue
            start, end = parsed
            job_id = str(receipt.get("job_id") or _sync_job_id_from_name(path.name))
            if not job_id:
                continue
            items.append(([start, end], job_id, receipt, path))
    items.sort(key=lambda item: (item[0][1], item[0][0], item[3].name))
    kept_items = items[-keep:]
    released_items = items[:-keep] if len(items) > keep else []
    kept_arcnames: set[str] = set()
    for step_range, job_id, _receipt, _path in kept_items:
        manifest_path = bundles_dir / f"sync_{job_id}_steps_{step_range[0]}_{step_range[1]}.sha256.json"
        if not manifest_path.is_file():
            continue
        try:
            members = json.loads(manifest_path.read_text(encoding="utf-8")).get("members") or {}
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(members, dict):
            kept_arcnames.update(str(name) for name in members)
    document = {
        "schema": SYNC_PAYLOAD_RETENTION_SCHEMA,
        "keep": keep,
        "kept_ranges": [item[0] for item in kept_items],
        "released_ranges": [item[0] for item in released_items],
        "observation_only": True,
        "note": (
            "Released payloads are disk paging. Receipts and manifests remain. "
            "Not continuation authority."
        ),
    }
    _atomic_json(root / "payload_retention.json", document)
    deleted_members = 0
    deleted_bundles = 0
    for step_range, job_id, _receipt, _path in released_items:
        start, end = step_range
        manifest_path = bundles_dir / f"sync_{job_id}_steps_{start}_{end}.sha256.json"
        bundle_path = bundles_dir / f"sync_{job_id}_steps_{start}_{end}.tar.gz"
        members: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8")).get("members") or {}
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                members = loaded
        for arcname in members:
            if arcname in kept_arcnames:
                continue
            candidate = members_dir / Path(*str(arcname).split("/"))
            io_path = _io_path(candidate)
            if io_path.is_file():
                io_path.unlink()
                deleted_members += 1
        bundle_io = _io_path(bundle_path)
        if bundle_io.is_file():
            bundle_io.unlink()
            deleted_bundles += 1
    document["deleted_member_count"] = deleted_members
    document["deleted_bundle_count"] = deleted_bundles
    _atomic_json(root / "payload_retention.json", document)
    return document

"""Atomic generations for Axon's disposable dormant evidence index.

The exact JSONL corpus remains memory authority.  This module only controls
which *derived* evidence-index generation readers should use.  Promotion is an
atomic pointer swap after full index/binding verification.  Failed candidate
builds or failed promotion leave the previous active generation untouched.

Build C.1 provided generational promotion and legacy-index adoption without
copying the existing multi-gigabyte index. Build C.2 adds transactional
append/layout-preserving-update maintenance through ``incremental.py``. Logical
generations may reuse the same disposable SQLite file; exact JSONL remains the
authority, and unsupported destructive/layout-shifting mutations fail closed to
the isolated full-generation rebuild path.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from runtime.field import canonical_json_bytes, canonical_sha256

from .evidence_bridge import (
    DEFAULT_INDEX_RELATIVE,
    INDEX_SCHEMA,
    DormantCorpusBinding,
    DormantEvidenceError,
    DormantEvidenceIndex,
    default_index_path,
)

GENERATION_POINTER_SCHEMA = "axon-dormant-evidence-generation-pointer-v1"
GENERATION_MANIFEST_SCHEMA = "axon-dormant-evidence-generation-manifest-v1"
ACTIVE_POINTER_NAME = "ACTIVE.json"
GENERATIONS_DIR_NAME = "generations"


class DormantGenerationError(DormantEvidenceError):
    """A derived evidence-index generation or pointer is invalid."""


@dataclass(frozen=True, slots=True)
class DormantIndexGeneration:
    generation_id: str
    index_id: str
    binding_id: str
    relative_index_path: str
    promoted_at: str
    mode: str
    predecessor_generation_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("generation_id", "index_id", "binding_id", "relative_index_path", "promoted_at", "mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.mode not in {"legacy_adopted", "full_generation", "incremental_update"}:
            raise ValueError("mode must be legacy_adopted, full_generation, or incremental_update")
        if self.predecessor_generation_id is not None and (
            not isinstance(self.predecessor_generation_id, str) or not self.predecessor_generation_id
        ):
            raise ValueError("predecessor_generation_id must be None or a non-empty string")

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "index_id": self.index_id,
            "binding_id": self.binding_id,
            "relative_index_path": self.relative_index_path,
            "promoted_at": self.promoted_at,
            "mode": self.mode,
            "predecessor_generation_id": self.predecessor_generation_id,
        }

    @property
    def descriptor_id(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DormantIndexGeneration":
        return cls(
            generation_id=str(value["generation_id"]),
            index_id=str(value["index_id"]),
            binding_id=str(value["binding_id"]),
            relative_index_path=str(value["relative_index_path"]),
            promoted_at=str(value["promoted_at"]),
            mode=str(value["mode"]),
            predecessor_generation_id=(
                None
                if value.get("predecessor_generation_id") is None
                else str(value["predecessor_generation_id"])
            ),
        )


class DormantEvidenceGenerationStore:
    """Resolve, verify, build, and atomically promote derived index generations."""

    def __init__(self, state_root: str | os.PathLike[str]) -> None:
        self.state_root = Path(state_root).resolve(strict=False)
        self.derived_root = default_index_path(self.state_root).parent
        self.pointer_path = self.derived_root / ACTIVE_POINTER_NAME
        self.generations_root = self.derived_root / GENERATIONS_DIR_NAME

    def _atomic_write(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
        data = canonical_json_bytes(payload) + b"\n"
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _safe_index_path(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise DormantGenerationError("generation index path must be relative to the evidence derived root")
        candidate = (self.derived_root / relative).resolve(strict=False)
        try:
            candidate.relative_to(self.derived_root.resolve(strict=False))
        except ValueError as exc:
            raise DormantGenerationError("generation index path escapes evidence derived root") from exc
        return candidate

    @staticmethod
    def _read_index_identity(index_path: Path) -> tuple[str, str]:
        if not index_path.is_file():
            raise DormantGenerationError(f"generation index does not exist: {index_path}")
        connection = sqlite3.connect(index_path)
        try:
            meta = dict(connection.execute("SELECT key, value FROM meta"))
        except sqlite3.DatabaseError as exc:
            raise DormantGenerationError(f"cannot read dormant index metadata: {index_path}") from exc
        finally:
            connection.close()
        if meta.get("schema") != INDEX_SCHEMA:
            raise DormantGenerationError("generation points at an unsupported dormant evidence index schema")
        try:
            binding = DormantCorpusBinding.from_mapping(json.loads(meta["binding_json"]))
            index_id = str(meta["index_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DormantGenerationError("generation index metadata is malformed") from exc
        expected = canonical_sha256({"schema": INDEX_SCHEMA, "binding": binding.to_canonical_dict()})
        if index_id != expected:
            raise DormantGenerationError("generation index identity does not match its corpus binding")
        return index_id, binding.binding_id

    def _legacy_descriptor(self) -> DormantIndexGeneration:
        index_path = default_index_path(self.state_root)
        index_id, binding_id = self._read_index_identity(index_path)
        promoted_at = datetime.fromtimestamp(index_path.stat().st_mtime, timezone.utc).isoformat()
        return DormantIndexGeneration(
            generation_id=f"legacy-{index_id[:24]}",
            index_id=index_id,
            binding_id=binding_id,
            relative_index_path=index_path.relative_to(self.derived_root).as_posix(),
            promoted_at=promoted_at,
            mode="legacy_adopted",
        )

    def pointer_descriptor_unverified(self) -> DormantIndexGeneration | None:
        """Read and hash-verify the logical pointer without trusting index bytes.

        C.2 needs this narrow recovery primitive because an authoritative corpus
        change may make the predecessor index stale before a replacement full or
        incremental generation is published. A malformed/tampered pointer still
        fails closed; only index-to-pointer identity verification is deferred.
        """

        if not self.pointer_path.is_file():
            return None
        try:
            raw = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DormantGenerationError("active dormant generation pointer is unreadable or invalid") from exc
        if not isinstance(raw, Mapping) or raw.get("schema") != GENERATION_POINTER_SCHEMA:
            raise DormantGenerationError("unsupported active dormant generation pointer schema")
        try:
            descriptor = DormantIndexGeneration.from_mapping(raw["generation"])
            descriptor_id = str(raw["descriptor_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DormantGenerationError("active dormant generation pointer is malformed") from exc
        if descriptor.descriptor_id != descriptor_id:
            raise DormantGenerationError("active dormant generation descriptor hash mismatch")
        self._safe_index_path(descriptor.relative_index_path)
        return descriptor

    def active_descriptor(self) -> DormantIndexGeneration:
        """Return the active descriptor, falling back to the legacy v1 location."""

        descriptor = self.pointer_descriptor_unverified()
        if descriptor is None:
            return self._legacy_descriptor()
        index_path = self._safe_index_path(descriptor.relative_index_path)
        index_id, binding_id = self._read_index_identity(index_path)
        if index_id != descriptor.index_id or binding_id != descriptor.binding_id:
            raise DormantGenerationError("active dormant generation pointer does not match index metadata")
        return descriptor

    def active_token(self) -> str:
        return self.active_descriptor().descriptor_id

    def open_active(self, *, verify_binding: bool = True) -> DormantEvidenceIndex:
        descriptor = self.active_descriptor()
        index_path = self._safe_index_path(descriptor.relative_index_path)
        index = DormantEvidenceIndex.open(
            self.state_root,
            index_path=index_path,
            verify_binding=verify_binding,
        )
        if index.index_id != descriptor.index_id or index.binding.binding_id != descriptor.binding_id:
            index.close()
            raise DormantGenerationError("opened dormant index does not match active generation descriptor")
        return index

    def _write_pointer(self, descriptor: DormantIndexGeneration) -> None:
        payload = {
            "schema": GENERATION_POINTER_SCHEMA,
            "descriptor_id": descriptor.descriptor_id,
            "generation": descriptor.to_canonical_dict(),
            "authority_note": (
                "Derived index selection only. State/dormant exact JSONL remains dormant-memory authority."
            ),
        }
        self._atomic_write(self.pointer_path, payload)

    def bootstrap_legacy(self) -> DormantIndexGeneration:
        """Adopt the existing evidence_v1/index.sqlite3 by pointer, without copying it."""

        if self.pointer_path.exists():
            return self.active_descriptor()
        descriptor = self._legacy_descriptor()
        # Promotion is allowed only after a full authoritative binding check.
        with DormantEvidenceIndex.open(
            self.state_root,
            index_path=self._safe_index_path(descriptor.relative_index_path),
            verify_binding=True,
        ) as verified:
            if verified.index_id != descriptor.index_id or verified.binding.binding_id != descriptor.binding_id:
                raise DormantGenerationError("legacy index changed during generation bootstrap")
        descriptor = DormantIndexGeneration(
            generation_id=descriptor.generation_id,
            index_id=descriptor.index_id,
            binding_id=descriptor.binding_id,
            relative_index_path=descriptor.relative_index_path,
            promoted_at=datetime.now(timezone.utc).isoformat(),
            mode="legacy_adopted",
        )
        self._write_pointer(descriptor)
        return descriptor

    def promote(self, descriptor: DormantIndexGeneration) -> DormantIndexGeneration:
        """Verify a complete candidate generation, then atomically make it active."""

        if not isinstance(descriptor, DormantIndexGeneration):
            raise TypeError("descriptor must be DormantIndexGeneration")
        index_path = self._safe_index_path(descriptor.relative_index_path)
        with DormantEvidenceIndex.open(
            self.state_root,
            index_path=index_path,
            verify_binding=True,
        ) as verified:
            if verified.index_id != descriptor.index_id or verified.binding.binding_id != descriptor.binding_id:
                raise DormantGenerationError("candidate generation does not match its descriptor")
        predecessor = None
        pointer_predecessor = self.pointer_descriptor_unverified()
        if pointer_predecessor is not None:
            # The predecessor may already be stale against changed authoritative
            # JSONL. Its pointer must still be structurally/hash valid, but full
            # rebuild promotion must not require stale predecessor bytes to match
            # the old descriptor after the corpus itself has legitimately moved.
            predecessor = pointer_predecessor.generation_id
        else:
            try:
                predecessor = self._legacy_descriptor().generation_id
            except DormantGenerationError:
                predecessor = None
        promoted = DormantIndexGeneration(
            generation_id=descriptor.generation_id,
            index_id=descriptor.index_id,
            binding_id=descriptor.binding_id,
            relative_index_path=descriptor.relative_index_path,
            promoted_at=datetime.now(timezone.utc).isoformat(),
            mode=descriptor.mode,
            predecessor_generation_id=predecessor,
        )
        self._write_pointer(promoted)
        return promoted

    def build_generation(self, *, promote: bool = False) -> DormantIndexGeneration:
        """Build a complete candidate generation in isolation.

        This intentionally uses the existing full builder.  It is the atomic
        generation/promotion substrate required before later append/update
        incremental maintenance; it is not itself an incremental build.
        """

        self.generations_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        building_dir = self.generations_root / f".building.{stamp}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        building_dir.mkdir(parents=True, exist_ok=False)
        building_index_path = building_dir / "index.sqlite3"
        try:
            built = DormantEvidenceIndex.build(self.state_root, index_path=building_index_path)
            try:
                index_id = built.index_id
                binding_id = built.binding.binding_id
            finally:
                built.close()
            # Reopen with authoritative verification before preserving candidate.
            with DormantEvidenceIndex.open(
                self.state_root,
                index_path=building_index_path,
                verify_binding=True,
            ) as verified:
                if verified.index_id != index_id or verified.binding.binding_id != binding_id:
                    raise DormantGenerationError("candidate generation identity changed after build")
            final_name = f"gen-{stamp}-{index_id[:16]}"
            final_dir = self.generations_root / final_name
            if final_dir.exists():
                raise DormantGenerationError(f"generation directory already exists: {final_dir}")
            os.replace(building_dir, final_dir)
            descriptor = DormantIndexGeneration(
                generation_id=final_name,
                index_id=index_id,
                binding_id=binding_id,
                relative_index_path=(final_dir / "index.sqlite3").relative_to(self.derived_root).as_posix(),
                promoted_at=datetime.now(timezone.utc).isoformat(),
                mode="full_generation",
            )
            generation_manifest = {
                "schema": GENERATION_MANIFEST_SCHEMA,
                "descriptor_id": descriptor.descriptor_id,
                "generation": descriptor.to_canonical_dict(),
                "build_mode": "full",
                "incremental_append_update": False,
                "authority_note": (
                    "Derived metadata generation only. Exact dormant JSONL remains memory authority."
                ),
            }
            self._atomic_write(final_dir / "generation.json", generation_manifest)
            return self.promote(descriptor) if promote else descriptor
        except Exception:
            # Never delete forensic evidence.  Preserve any incomplete build by
            # moving it to a clearly failed path when possible.
            if building_dir.exists():
                failed = self.generations_root / f"failed-{stamp}-{uuid.uuid4().hex[:8]}"
                try:
                    os.replace(building_dir, failed)
                except OSError:
                    pass
            raise


__all__ = [
    "GENERATION_POINTER_SCHEMA",
    "GENERATION_MANIFEST_SCHEMA",
    "ACTIVE_POINTER_NAME",
    "GENERATIONS_DIR_NAME",
    "DormantGenerationError",
    "DormantIndexGeneration",
    "DormantEvidenceGenerationStore",
]

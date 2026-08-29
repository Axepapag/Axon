"""File-backed canonical state branches for runtime/training isolation.

A branch is not a second state schema.  It persists the same immutable
SharedFieldSnapshot and FieldDelta objects used by runtime, with an append-only
journal and an atomic HEAD pointer.  Training branches live beneath
D:\\Axon\\State\\training; tests may use temporary roots explicitly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .delta import (
    DeleteText,
    FieldDelta,
    InsertText,
    ReplaceText,
    apply_delta,
)
from .schema import (
    CORTEX_SCHEMA_VERSION,
    LEGACY_CORTEX_REGION_NAME,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AttendedInterval,
    FieldSpan,
    LogicalRegion,
    RegionMaskPolicy,
    RegionState,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)

BRANCH_SCHEMA = "axon-canonical-state-branch-v1"
BRANCH_HEAD_SCHEMA = "axon-canonical-state-branch-head-v1"
BRANCH_EVENT_SCHEMA = "axon-canonical-state-branch-event-v1"
DEFAULT_STATE_ROOT = Path(r"D:\Axon\State")


class CanonicalStateBranchError(RuntimeError):
    """Base class for branch integrity/authority failures."""


class BranchIntegrityError(CanonicalStateBranchError):
    """Persisted branch material failed canonical verification."""


class BranchAuthorityError(CanonicalStateBranchError):
    """A branch path is outside its declared authority root."""


@dataclass(frozen=True, slots=True)
class BranchHead:
    branch_id: str
    generation: int
    field_id: str
    tick_id: int
    parent_field_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BRANCH_HEAD_SCHEMA,
            "branch_id": self.branch_id,
            "generation": self.generation,
            "field_id": self.field_id,
            "tick_id": self.tick_id,
            "parent_field_id": self.parent_field_id,
        }


class CanonicalStateBranch:
    """Atomic append-only branch over canonical SharedFieldSnapshot objects."""

    def __init__(
        self,
        root: Path | str,
        *,
        branch_id: str,
        authority_root: Path | str | None = None,
    ) -> None:
        if not isinstance(branch_id, str) or not branch_id.strip():
            raise ValueError("branch_id must be a non-empty string")
        self.root = Path(root).resolve(strict=False)
        self.branch_id = branch_id.strip()
        self.authority_root = (
            None if authority_root is None else Path(authority_root).resolve(strict=False)
        )
        if self.authority_root is not None:
            _require_under(self.root, self.authority_root)
        self.snapshots_dir = self.root / "snapshots"
        self.deltas_dir = self.root / "deltas"
        self.head_path = self.root / "HEAD.json"
        self.branch_path = self.root / "branch.json"
        self.journal_path = self.root / "journal.jsonl"

    @classmethod
    def training(
        cls,
        branch_id: str,
        *,
        state_root: Path | str = DEFAULT_STATE_ROOT,
    ) -> "CanonicalStateBranch":
        state = Path(state_root).resolve(strict=False)
        training_root = (state / "training" / "branches").resolve(strict=False)
        return cls(
            training_root / _safe_component(branch_id),
            branch_id=branch_id,
            authority_root=training_root,
        )

    @classmethod
    def active_runtime(
        cls,
        branch_id: str = "active",
        *,
        state_root: Path | str = DEFAULT_STATE_ROOT,
    ) -> "CanonicalStateBranch":
        state = Path(state_root).resolve(strict=False)
        runtime_root = (state / "active" / "branches").resolve(strict=False)
        return cls(
            runtime_root / _safe_component(branch_id),
            branch_id=branch_id,
            authority_root=runtime_root,
        )

    @property
    def initialized(self) -> bool:
        return self.branch_path.exists() and self.head_path.exists()

    def initialize(self, snapshot: SharedFieldSnapshot) -> BranchHead:
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("snapshot must be SharedFieldSnapshot")
        if self.root.exists() and any(self.root.iterdir()):
            if self.initialized:
                existing = self.load_head()
                if existing.field_id != snapshot.field_id:
                    raise CanonicalStateBranchError(
                        "branch already initialized with a different canonical field"
                    )
                return existing
            raise CanonicalStateBranchError(
                f"branch root is non-empty but not initialized: {self.root}"
            )
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.deltas_dir.mkdir(parents=True, exist_ok=True)
        descriptor = {
            "schema": BRANCH_SCHEMA,
            "branch_id": self.branch_id,
            "root": str(self.root),
            "initial_field_id": snapshot.field_id,
            "initial_tick_id": snapshot.tick_id,
        }
        _atomic_json(self.branch_path, descriptor)
        self._persist_snapshot(snapshot)
        head = BranchHead(
            branch_id=self.branch_id,
            generation=0,
            field_id=snapshot.field_id,
            tick_id=snapshot.tick_id,
            parent_field_id=snapshot.parent_field_id,
        )
        _atomic_json(self.head_path, head.to_dict())
        self._append_event(
            {
                "event": "initialize",
                "generation": 0,
                "field_id": snapshot.field_id,
                "tick_id": snapshot.tick_id,
                "parent_field_id": snapshot.parent_field_id,
            }
        )
        return head

    def load_head_record(self) -> BranchHead:
        value = _read_json(self.head_path)
        if set(value) != {
            "schema",
            "branch_id",
            "generation",
            "field_id",
            "tick_id",
            "parent_field_id",
        }:
            raise BranchIntegrityError("branch HEAD fields are invalid")
        if value["schema"] != BRANCH_HEAD_SCHEMA:
            raise BranchIntegrityError("unsupported branch HEAD schema")
        if value["branch_id"] != self.branch_id:
            raise BranchIntegrityError("branch HEAD id mismatch")
        return BranchHead(
            branch_id=value["branch_id"],
            generation=_nonnegative_int(value["generation"], "generation"),
            field_id=_nonempty(value["field_id"], "field_id"),
            tick_id=_nonnegative_int(value["tick_id"], "tick_id"),
            parent_field_id=(
                None
                if value["parent_field_id"] is None
                else _nonempty(value["parent_field_id"], "parent_field_id")
            ),
        )

    def load_head(self) -> SharedFieldSnapshot:
        head = self.load_head_record()
        snapshot = self.load_snapshot(head.field_id)
        if snapshot.tick_id != head.tick_id or snapshot.parent_field_id != head.parent_field_id:
            raise BranchIntegrityError("branch HEAD metadata does not match its snapshot")
        return snapshot

    def load_snapshot(self, field_id: str) -> SharedFieldSnapshot:
        field_id = _nonempty(field_id, "field_id")
        payload = _read_json(self.snapshots_dir / f"{field_id}.json")
        snapshot = _snapshot_from_dict(payload)
        if snapshot.field_id != field_id:
            raise BranchIntegrityError(
                f"snapshot filename/id mismatch: {field_id} != {snapshot.field_id}"
            )
        return snapshot

    def commit(
        self,
        delta: FieldDelta,
        *,
        permitted_regions: frozenset[LogicalRegion] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SharedFieldSnapshot:
        if not isinstance(delta, FieldDelta):
            raise TypeError("delta must be FieldDelta")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        current_record = self.load_head_record()
        current = self.load_snapshot(current_record.field_id)
        successor = apply_delta(current, delta, permitted_regions=permitted_regions)
        self._persist_delta(delta)
        self._persist_snapshot(successor)
        next_head = BranchHead(
            branch_id=self.branch_id,
            generation=current_record.generation + 1,
            field_id=successor.field_id,
            tick_id=successor.tick_id,
            parent_field_id=successor.parent_field_id,
        )
        # HEAD is canonical authority.  Advance it atomically before writing the
        # audit event; if the later journal append fails, callers resynchronize
        # to this durable HEAD rather than mistaking an audit line for reality.
        _atomic_json(self.head_path, next_head.to_dict())
        event = {
            "event": "commit",
            "generation": next_head.generation,
            "base_field_id": current.field_id,
            "delta_id": delta.delta_id,
            "field_id": successor.field_id,
            "tick_id": successor.tick_id,
            "parent_field_id": successor.parent_field_id,
        }
        if metadata:
            event["metadata"] = dict(metadata)
        self._append_event(event)
        return successor

    def migrate_to_current_schema(self) -> SharedFieldSnapshot:
        """Advance HEAD to the current shared-field schema without rewriting history.

        Historical snapshots remain immutable and retain their original field IDs.
        The migration creates one new successor containing the exact same spans and
        manifests, increments tick/generation, and parents it to the legacy HEAD.
        """

        current_record = self.load_head_record()
        current = self.load_snapshot(current_record.field_id)
        if current.schema_version == SCHEMA_VERSION:
            return current
        if current.schema_version not in {
            LEGACY_SCHEMA_VERSION,
            CORTEX_SCHEMA_VERSION,
        }:
            raise CanonicalStateBranchError(
                f"cannot migrate unsupported shared-field schema {current.schema_version!r}"
            )
        successor = SharedFieldSnapshot(
            tick_id=current.tick_id + 1,
            regions=current.regions,
            parent_field_id=current.field_id,
            source_manifest_ids=current.source_manifest_ids,
            schema_version=SCHEMA_VERSION,
        )
        self._persist_snapshot(successor)
        next_head = BranchHead(
            branch_id=self.branch_id,
            generation=current_record.generation + 1,
            field_id=successor.field_id,
            tick_id=successor.tick_id,
            parent_field_id=successor.parent_field_id,
        )
        _atomic_json(self.head_path, next_head.to_dict())
        self._append_event(
            {
                "event": "schema_migration",
                "generation": next_head.generation,
                "from_schema": current.schema_version,
                "to_schema": successor.schema_version,
                "base_field_id": current.field_id,
                "field_id": successor.field_id,
                "tick_id": successor.tick_id,
                "parent_field_id": successor.parent_field_id,
                "region_rename": (
                    {LEGACY_CORTEX_REGION_NAME: LogicalRegion.CORTEX.value}
                    if current.schema_version == LEGACY_SCHEMA_VERSION
                    else {}
                ),
                "regions_added": [LogicalRegion.IDENTITY.value],
            }
        )
        return successor

    def _persist_snapshot(self, snapshot: SharedFieldSnapshot) -> None:
        path = self.snapshots_dir / f"{snapshot.field_id}.json"
        value = snapshot.to_dict()
        if path.exists():
            existing = _read_json(path)
            if canonical_json_bytes(existing) != canonical_json_bytes(value):
                raise BranchIntegrityError(
                    f"existing snapshot bytes disagree for field {snapshot.field_id}"
                )
            return
        _atomic_json(path, value)

    def _persist_delta(self, delta: FieldDelta) -> None:
        path = self.deltas_dir / f"{delta.delta_id}.json"
        value = delta.to_canonical_dict()
        value["delta_id"] = delta.delta_id
        value["canonical_hash"] = delta.canonical_hash
        if path.exists():
            existing = _read_json(path)
            if canonical_json_bytes(existing) != canonical_json_bytes(value):
                raise BranchIntegrityError(
                    f"existing delta bytes disagree for delta {delta.delta_id}"
                )
            return
        _atomic_json(path, value)

    def _append_event(self, body: Mapping[str, Any]) -> None:
        event = {
            "schema": BRANCH_EVENT_SCHEMA,
            "branch_id": self.branch_id,
            **dict(body),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _snapshot_from_dict(value: Mapping[str, Any]) -> SharedFieldSnapshot:
    item = dict(value)
    required = {
        "schema",
        "tick_id",
        "parent_field_id",
        "source_manifest_ids",
        "regions",
        "field_id",
        "canonical_hash",
    }
    if set(item) != required:
        raise BranchIntegrityError("serialized canonical snapshot fields are invalid")
    schema_version = str(item["schema"])
    if schema_version not in {
        LEGACY_SCHEMA_VERSION,
        CORTEX_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        raise BranchIntegrityError("unsupported canonical snapshot schema")
    if not isinstance(item["regions"], list):
        raise BranchIntegrityError("serialized regions must be a list")
    regions: list[RegionState] = []
    for region_item in item["regions"]:
        if not isinstance(region_item, Mapping):
            raise BranchIntegrityError("serialized region must be an object")
        required_region_fields = {"name", "visibility", "write_policy", "spans"}
        if not required_region_fields.issubset(set(region_item)):
            raise BranchIntegrityError("serialized region fields are invalid")
        if not isinstance(region_item["spans"], list):
            raise BranchIntegrityError("serialized spans must be a list")
        spans: list[FieldSpan] = []
        for span_item in region_item["spans"]:
            if not isinstance(span_item, Mapping):
                raise BranchIntegrityError("serialized span must be an object")
            required_span_fields = {
                "span_id",
                "text",
                "kind",
                "source",
                "provenance",
                "confidence",
                "container_refs",
                "edge_refs",
            }
            if not required_span_fields.issubset(set(span_item)):
                raise BranchIntegrityError("serialized span fields are invalid")
            spans.append(
                FieldSpan(
                    span_id=span_item["span_id"],
                    text=span_item["text"],
                    kind=span_item["kind"],
                    source=span_item["source"],
                    provenance=span_item["provenance"],
                    confidence=span_item["confidence"],
                    container_refs=tuple(span_item["container_refs"]),
                    edge_refs=tuple(span_item["edge_refs"]),
                )
            )

        attended_intervals = region_item.get("attended_intervals")
        if attended_intervals is not None:
            attended_intervals = tuple(
                AttendedInterval(interval["start"], interval["end"])
                for interval in attended_intervals
            )

        mask_policy = region_item.get("mask_policy")
        if mask_policy is not None:
            mask_policy = RegionMaskPolicy(mask_policy["kind"], mask_policy["limit"])

        raw_region_name = region_item["name"]
        if schema_version == LEGACY_SCHEMA_VERSION and raw_region_name == LEGACY_CORTEX_REGION_NAME:
            raw_region_name = LogicalRegion.CORTEX.value
        regions.append(
            RegionState(
                name=raw_region_name,
                visibility=region_item["visibility"],
                write_policy=region_item["write_policy"],
                spans=tuple(spans),
                attended_intervals=attended_intervals,
                mask_policy=mask_policy,
            )
        )
    snapshot = SharedFieldSnapshot(
        tick_id=item["tick_id"],
        parent_field_id=item["parent_field_id"],
        source_manifest_ids=tuple(item["source_manifest_ids"]),
        regions=tuple(regions),
        schema_version=schema_version,
    )
    if snapshot.field_id != item["field_id"] or snapshot.canonical_hash != item["canonical_hash"]:
        raise BranchIntegrityError("serialized canonical snapshot hash mismatch")
    return snapshot


def _field_delta_from_dict(value: Mapping[str, Any]) -> FieldDelta:
    """Strict parser retained for branch audit/replay tooling."""

    item = dict(value)
    required = {
        "schema",
        "base_field_id",
        "base_tick_id",
        "author_core_id",
        "pass_id",
        "operations",
        "evidence",
        "delta_id",
        "canonical_hash",
    }
    if set(item) != required or item["schema"] != "shared-field-delta-v1":
        raise BranchIntegrityError("serialized field delta fields/schema are invalid")
    operations = []
    for raw in item["operations"]:
        if not isinstance(raw, Mapping):
            raise BranchIntegrityError("serialized field operation must be an object")
        op = raw.get("op")
        raw_region = raw.get("region")
        if raw_region == LEGACY_CORTEX_REGION_NAME:
            raw_region = LogicalRegion.CORTEX.value
        common = {
            "region": raw_region,
            "provenance": raw.get("provenance", ""),
            "container_refs": tuple(raw.get("container_refs", ())),
            "edge_refs": tuple(raw.get("edge_refs", ())),
        }
        if op == "insert":
            operations.append(InsertText(offset=raw["offset"], text=raw["text"], **common))
        elif op == "delete":
            operations.append(DeleteText(start=raw["start"], end=raw["end"], **common))
        elif op == "replace":
            operations.append(
                ReplaceText(start=raw["start"], end=raw["end"], text=raw["text"], **common)
            )
        else:
            raise BranchIntegrityError(f"unknown serialized operation {op!r}")
    delta = FieldDelta(
        base_field_id=item["base_field_id"],
        base_tick_id=item["base_tick_id"],
        author_core_id=item["author_core_id"],
        pass_id=item["pass_id"],
        operations=tuple(operations),
        evidence=tuple(item["evidence"]),
    )
    if delta.delta_id != item["delta_id"] or delta.canonical_hash != item["canonical_hash"]:
        raise BranchIntegrityError("serialized field delta hash mismatch")
    return delta


def snapshot_from_canonical_dict(value: Mapping[str, Any]) -> SharedFieldSnapshot:
    """Strictly reconstruct and verify one fully identified canonical snapshot."""

    if not isinstance(value, Mapping):
        raise TypeError("snapshot value must be a mapping")
    return _snapshot_from_dict(value)


def field_delta_from_canonical_dict(value: Mapping[str, Any]) -> FieldDelta:
    """Strictly reconstruct a canonical or persisted typed delta mapping."""

    if not isinstance(value, Mapping):
        raise TypeError("delta value must be a mapping")
    item = dict(value)
    canonical_fields = {
        "schema",
        "base_field_id",
        "base_tick_id",
        "author_core_id",
        "pass_id",
        "operations",
        "evidence",
    }
    if set(item) == canonical_fields:
        digest = canonical_sha256(item)
        item["delta_id"] = digest
        item["canonical_hash"] = digest
    return _field_delta_from_dict(item)


def _require_under(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BranchAuthorityError(f"branch root {path} is outside authority root {root}") from exc
    if path == root:
        raise BranchAuthorityError("branch root must be below, not equal to, authority root")


def _safe_component(value: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or any(char in value for char in "\\/:*?\"<>|"):
        raise ValueError("branch_id is not a safe Windows path component")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BranchIntegrityError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BranchIntegrityError(f"JSON resource must be an object: {path}")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BranchIntegrityError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BranchIntegrityError(f"{label} must be a non-negative integer")
    return value


__all__ = [
    "BRANCH_EVENT_SCHEMA",
    "BRANCH_HEAD_SCHEMA",
    "BRANCH_SCHEMA",
    "DEFAULT_STATE_ROOT",
    "BranchAuthorityError",
    "BranchHead",
    "BranchIntegrityError",
    "CanonicalStateBranch",
    "CanonicalStateBranchError",
    "field_delta_from_canonical_dict",
    "snapshot_from_canonical_dict",
]

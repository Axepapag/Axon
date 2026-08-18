"""Content-addressed, non-pickle storage for per-core runtime souls."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import struct
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from runtime.field import FieldViewCursor, canonical_json_bytes, canonical_sha256


SOUL_BLOB_SCHEMA = "axon-runtime-soul-blob-v1"
SOUL_BLOB_MAGIC = b"AXONSOULBLOB1\n"
PRIVATE_RUNTIME_STATE_SCHEMA = "axon-private-runtime-state-v1"
PRIVATE_RUNTIME_STATE_MAGIC = b"AXONRUNTIMESTATE1\n"
_HEADER_LENGTH = struct.Struct("<Q")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_HEADER_KEYS = {
    "schema",
    "soul_id",
    "generation",
    "base_field_id",
    "rows",
    "d_model",
    "tensor_dtype",
    "mask_dtype",
    "tensor_nbytes",
    "mask_nbytes",
    "payload_sha256",
}
_PRIVATE_STATE_KEYS = {
    "schema",
    "core_id",
    "soul_id",
    "model_id",
    "core_generation",
    "soul_generation",
    "committed_field_id",
    "soul_blob_sha256",
    "candidate_manifest_id",
    "cursor",
    "cursor_sha256",
    "cursor_anchor_manifest",
    "cursor_anchor_sha256",
    "adapter_manifest",
    "rng_manifest",
}


class SoulStoreError(RuntimeError):
    """Base error for runtime soul encoding or storage."""


class SoulBlobContractError(SoulStoreError):
    """A tensor/header does not satisfy the runtime soul format."""


class SoulBlobCorruptionError(SoulStoreError):
    """Persisted bytes failed their content or payload hash."""


@dataclass(frozen=True, slots=True)
class SoulBlob:
    """Deterministic staged bytes; no filesystem effect has occurred."""

    sha256: str
    header: Mapping[str, Any]
    data: bytes


@dataclass(frozen=True, slots=True)
class DecodedSoulBlob:
    sha256: str
    header: Mapping[str, Any]
    soul: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class StoredSoulBlob:
    sha256: str
    path: Path
    header: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PrivateRuntimeStateBlob:
    """Staged canonical metadata for crash-safe logical-core recovery."""

    sha256: str
    payload: Mapping[str, Any]
    data: bytes


@dataclass(frozen=True, slots=True)
class DecodedPrivateRuntimeState:
    sha256: str
    core_id: str
    soul_id: str
    model_id: str
    core_generation: int
    soul_generation: int
    committed_field_id: str
    soul_blob_sha256: str
    candidate_manifest_id: str
    cursor: FieldViewCursor
    cursor_sha256: str
    cursor_anchor_manifest: Mapping[str, Any]
    cursor_anchor_sha256: str
    adapter_manifest: Mapping[str, Any]
    rng_manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StoredPrivateRuntimeState:
    sha256: str
    path: Path
    cursor_payload_sha256: str

    @property
    def cursor_state_sha256(self) -> str:
        """CoreStateManifest pointer to this complete continuity artifact."""

        return self.sha256


def _valid_hash(value: Any, label: str = "SHA-256") -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SoulBlobContractError(
            f"{label} must be a 64-character hexadecimal SHA-256"
        )
    return value.lower()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SoulBlobContractError(f"{label} must be a non-empty string")
    return value


def _generation(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SoulBlobContractError(
            "generation must be a non-negative integer"
        )
    return value


def _owned_runtime_tensors(
    soul: Any,
    mask: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(soul, torch.Tensor):
        raise SoulBlobContractError("soul must be a tensor")
    if not isinstance(mask, torch.Tensor):
        raise SoulBlobContractError("mask must be a tensor")
    if soul.layout is not torch.strided or mask.layout is not torch.strided:
        raise SoulBlobContractError("soul and mask must use strided layout")
    if soul.ndim != 2 or soul.shape[0] <= 0 or soul.shape[1] <= 0:
        raise SoulBlobContractError(
            "soul must have shape (positive rows, positive d_model)"
        )
    if tuple(mask.shape) != (soul.shape[0],):
        raise SoulBlobContractError("mask shape must match the soul row count")
    if soul.dtype is not torch.float32:
        raise SoulBlobContractError("soul dtype must be torch.float32")
    if mask.dtype is not torch.bool:
        raise SoulBlobContractError("mask dtype must be torch.bool")
    if not bool(torch.isfinite(soul).all()):
        raise SoulBlobContractError("soul contains non-finite values")
    return (
        soul.detach().cpu().contiguous().clone(),
        mask.detach().cpu().contiguous().clone(),
    )


def encode_soul_blob(
    soul: torch.Tensor,
    mask: torch.Tensor,
    *,
    soul_id: str,
    generation: int,
    base_field_id: str,
) -> SoulBlob:
    """Encode owned tensor copies into canonical header + raw payload bytes."""

    identity = _nonempty(soul_id, "soul_id")
    field_id = _nonempty(base_field_id, "base_field_id")
    generation_value = _generation(generation)
    soul_value, mask_value = _owned_runtime_tensors(soul, mask)
    soul_array = (
        soul_value.numpy().astype(np.dtype("<f4"), copy=True, order="C")
    )
    mask_array = (
        mask_value.numpy().astype(np.dtype("u1"), copy=True, order="C")
    )
    tensor_bytes = soul_array.tobytes(order="C")
    mask_bytes = mask_array.tobytes(order="C")
    payload = tensor_bytes + mask_bytes
    header = {
        "schema": SOUL_BLOB_SCHEMA,
        "soul_id": identity,
        "generation": generation_value,
        "base_field_id": field_id,
        "rows": int(soul_value.shape[0]),
        "d_model": int(soul_value.shape[1]),
        "tensor_dtype": "float32-le",
        "mask_dtype": "bool-u8",
        "tensor_nbytes": len(tensor_bytes),
        "mask_nbytes": len(mask_bytes),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    header_bytes = canonical_json_bytes(header)
    data = (
        SOUL_BLOB_MAGIC
        + _HEADER_LENGTH.pack(len(header_bytes))
        + header_bytes
        + payload
    )
    digest = hashlib.sha256(data).hexdigest()
    return SoulBlob(
        sha256=digest,
        header=MappingProxyType(dict(header)),
        data=data,
    )


def _parse_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SoulBlobCorruptionError(f"{label} must be a positive integer")
    return value


def decode_soul_blob(
    data: bytes | bytearray | memoryview,
    expected_sha256: str,
) -> DecodedSoulBlob:
    """Validate every byte and return tensors with independent owned storage."""

    expected = _valid_hash(expected_sha256, "expected_sha256")
    raw = bytes(data)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise SoulBlobCorruptionError(
            f"soul blob hash mismatch: expected {expected}, got {actual}"
        )
    prefix_size = len(SOUL_BLOB_MAGIC) + _HEADER_LENGTH.size
    if len(raw) < prefix_size or not raw.startswith(SOUL_BLOB_MAGIC):
        raise SoulBlobCorruptionError("soul blob magic is invalid")
    (header_length,) = _HEADER_LENGTH.unpack_from(
        raw,
        len(SOUL_BLOB_MAGIC),
    )
    if header_length <= 0 or header_length > 1024 * 1024:
        raise SoulBlobCorruptionError("soul blob header length is invalid")
    header_start = prefix_size
    header_end = header_start + header_length
    if header_end > len(raw):
        raise SoulBlobCorruptionError("soul blob header is truncated")
    header_bytes = raw[header_start:header_end]
    try:
        import json

        header = json.loads(
            header_bytes,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise SoulBlobCorruptionError("soul blob header is invalid JSON") from exc
    if not isinstance(header, dict) or not all(
        isinstance(key, str) for key in header
    ):
        raise SoulBlobCorruptionError("soul blob header must be an object")
    if set(header) != _HEADER_KEYS:
        raise SoulBlobCorruptionError("soul blob header keys mismatch")
    if canonical_json_bytes(header) != header_bytes:
        raise SoulBlobCorruptionError("soul blob header is not canonical")
    if header["schema"] != SOUL_BLOB_SCHEMA:
        raise SoulBlobCorruptionError("unknown soul blob schema")
    try:
        _nonempty(header["soul_id"], "soul_id")
        _nonempty(header["base_field_id"], "base_field_id")
        _generation(header["generation"])
    except SoulBlobContractError as exc:
        raise SoulBlobCorruptionError(str(exc)) from exc
    rows = _parse_positive_int(header["rows"], "rows")
    d_model = _parse_positive_int(header["d_model"], "d_model")
    tensor_nbytes = _parse_positive_int(
        header["tensor_nbytes"],
        "tensor_nbytes",
    )
    mask_nbytes = _parse_positive_int(
        header["mask_nbytes"],
        "mask_nbytes",
    )
    if header["tensor_dtype"] != "float32-le":
        raise SoulBlobCorruptionError("unsupported soul tensor dtype")
    if header["mask_dtype"] != "bool-u8":
        raise SoulBlobCorruptionError("unsupported soul mask dtype")
    expected_tensor_bytes = rows * d_model * 4
    if tensor_nbytes != expected_tensor_bytes or mask_nbytes != rows:
        raise SoulBlobCorruptionError("soul blob byte counts do not match shape")
    payload = raw[header_end:]
    if len(payload) != tensor_nbytes + mask_nbytes:
        raise SoulBlobCorruptionError("soul blob payload length mismatch")
    payload_hash = header["payload_sha256"]
    if not isinstance(payload_hash, str) or _SHA256.fullmatch(payload_hash) is None:
        raise SoulBlobCorruptionError("payload_sha256 is invalid")
    if hashlib.sha256(payload).hexdigest() != payload_hash.lower():
        raise SoulBlobCorruptionError("soul blob payload hash mismatch")

    tensor_raw = payload[:tensor_nbytes]
    mask_raw = payload[tensor_nbytes:]
    # np.copy breaks the bytes-backed view; torch then owns that array storage.
    soul_array = np.frombuffer(tensor_raw, dtype="<f4").reshape(
        rows,
        d_model,
    ).copy()
    mask_u8 = np.frombuffer(mask_raw, dtype="u1").copy()
    if np.any((mask_u8 != 0) & (mask_u8 != 1)):
        raise SoulBlobCorruptionError("soul mask contains values other than 0/1")
    if not np.isfinite(soul_array).all():
        raise SoulBlobCorruptionError("soul blob contains non-finite values")
    soul = torch.from_numpy(soul_array)
    mask = torch.from_numpy(mask_u8.astype(np.bool_, copy=True))
    return DecodedSoulBlob(
        sha256=actual,
        header=MappingProxyType(dict(header)),
        soul=soul,
        mask=mask,
    )


def _json_object_copy(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise SoulBlobContractError(f"{label} must be a string-keyed mapping")
    try:
        import json

        decoded = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as exc:
        raise SoulBlobContractError(
            f"{label} must be finite JSON-safe data"
        ) from exc
    if not isinstance(decoded, dict):
        raise SoulBlobContractError(f"{label} must encode an object")
    return decoded


def encode_private_runtime_state(
    *,
    core_id: str,
    soul_id: str,
    model_id: str,
    core_generation: int,
    soul_generation: int,
    committed_field_id: str,
    soul_blob_sha256: str,
    candidate_manifest_id: str,
    cursor: FieldViewCursor,
    cursor_anchor_manifest: Mapping[str, Any],
    adapter_manifest: Mapping[str, Any],
    rng_manifest: Mapping[str, Any],
) -> PrivateRuntimeStateBlob:
    """Stage all non-weight logical state as one canonical JSON artifact."""

    if not isinstance(cursor, FieldViewCursor):
        raise TypeError("cursor must be FieldViewCursor")
    cursor_value = cursor.to_canonical_dict()
    anchor_value = _json_object_copy(
        cursor_anchor_manifest,
        "cursor_anchor_manifest",
    )
    payload = {
        "schema": PRIVATE_RUNTIME_STATE_SCHEMA,
        "core_id": _nonempty(core_id, "core_id"),
        "soul_id": _nonempty(soul_id, "soul_id"),
        "model_id": _nonempty(model_id, "model_id"),
        "core_generation": _generation(core_generation),
        "soul_generation": _generation(soul_generation),
        "committed_field_id": _nonempty(
            committed_field_id,
            "committed_field_id",
        ),
        "soul_blob_sha256": _valid_hash(
            soul_blob_sha256,
            "soul_blob_sha256",
        ),
        "candidate_manifest_id": _valid_hash(
            candidate_manifest_id,
            "candidate_manifest_id",
        ),
        "cursor": cursor_value,
        "cursor_sha256": canonical_sha256(cursor_value),
        "cursor_anchor_manifest": anchor_value,
        "cursor_anchor_sha256": canonical_sha256(anchor_value),
        "adapter_manifest": _json_object_copy(
            adapter_manifest,
            "adapter_manifest",
        ),
        "rng_manifest": _json_object_copy(rng_manifest, "rng_manifest"),
    }
    canonical = canonical_json_bytes(payload)
    data = PRIVATE_RUNTIME_STATE_MAGIC + canonical
    return PrivateRuntimeStateBlob(
        sha256=hashlib.sha256(data).hexdigest(),
        payload=MappingProxyType(payload),
        data=data,
    )


def _decode_cursor(value: Any, expected_sha256: Any) -> FieldViewCursor:
    if not isinstance(value, Mapping) or set(value) != {
        "page_index",
        "context_offsets",
        "user_offset",
    }:
        raise SoulBlobCorruptionError("runtime-state cursor is malformed")
    offsets = value["context_offsets"]
    if not isinstance(offsets, Mapping) or not all(
        isinstance(key, str) for key in offsets
    ):
        raise SoulBlobCorruptionError(
            "runtime-state cursor offsets are malformed"
        )
    try:
        cursor = FieldViewCursor(
            page_index=value["page_index"],
            context_offsets=tuple(offsets.items()),
            user_offset=value["user_offset"],
        )
    except (TypeError, ValueError) as exc:
        raise SoulBlobCorruptionError(
            "runtime-state cursor violates its contract"
        ) from exc
    actual = canonical_sha256(cursor.to_canonical_dict())
    try:
        expected = _valid_hash(expected_sha256, "cursor_sha256")
    except SoulBlobContractError as exc:
        raise SoulBlobCorruptionError(str(exc)) from exc
    if actual != expected:
        raise SoulBlobCorruptionError("runtime-state cursor hash mismatch")
    return cursor


def decode_private_runtime_state(
    data: bytes | bytearray | memoryview,
    expected_sha256: str,
) -> DecodedPrivateRuntimeState:
    """Strictly decode a crash-continuity artifact without deserialization."""

    expected = _valid_hash(expected_sha256, "expected_sha256")
    raw = bytes(data)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise SoulBlobCorruptionError(
            f"private runtime-state hash mismatch: {actual} != {expected}"
        )
    if not raw.startswith(PRIVATE_RUNTIME_STATE_MAGIC):
        raise SoulBlobCorruptionError("private runtime-state magic is invalid")
    canonical = raw[len(PRIVATE_RUNTIME_STATE_MAGIC) :]
    try:
        import json

        payload = json.loads(
            canonical,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise SoulBlobCorruptionError(
            "private runtime-state payload is invalid JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _PRIVATE_STATE_KEYS:
        raise SoulBlobCorruptionError(
            "private runtime-state payload keys mismatch"
        )
    if canonical_json_bytes(payload) != canonical:
        raise SoulBlobCorruptionError(
            "private runtime-state payload is not canonical"
        )
    if payload["schema"] != PRIVATE_RUNTIME_STATE_SCHEMA:
        raise SoulBlobCorruptionError("unknown private runtime-state schema")
    try:
        core_id = _nonempty(payload["core_id"], "core_id")
        soul_id = _nonempty(payload["soul_id"], "soul_id")
        model_id = _nonempty(payload["model_id"], "model_id")
        core_generation = _generation(payload["core_generation"])
        soul_generation = _generation(payload["soul_generation"])
        committed_field_id = _nonempty(
            payload["committed_field_id"],
            "committed_field_id",
        )
        soul_hash = _valid_hash(
            payload["soul_blob_sha256"],
            "soul_blob_sha256",
        )
        candidate_id = _valid_hash(
            payload["candidate_manifest_id"],
            "candidate_manifest_id",
        )
        anchors = _json_object_copy(
            payload["cursor_anchor_manifest"],
            "cursor_anchor_manifest",
        )
        anchor_hash = _valid_hash(
            payload["cursor_anchor_sha256"],
            "cursor_anchor_sha256",
        )
        if canonical_sha256(anchors) != anchor_hash:
            raise SoulBlobContractError("cursor anchor hash mismatch")
        adapters = _json_object_copy(
            payload["adapter_manifest"],
            "adapter_manifest",
        )
        rng = _json_object_copy(payload["rng_manifest"], "rng_manifest")
    except SoulBlobContractError as exc:
        raise SoulBlobCorruptionError(str(exc)) from exc
    cursor = _decode_cursor(payload["cursor"], payload["cursor_sha256"])
    return DecodedPrivateRuntimeState(
        sha256=actual,
        core_id=core_id,
        soul_id=soul_id,
        model_id=model_id,
        core_generation=core_generation,
        soul_generation=soul_generation,
        committed_field_id=committed_field_id,
        soul_blob_sha256=soul_hash,
        candidate_manifest_id=candidate_id,
        cursor=cursor,
        cursor_sha256=payload["cursor_sha256"],
        cursor_anchor_manifest=MappingProxyType(anchors),
        cursor_anchor_sha256=anchor_hash,
        adapter_manifest=MappingProxyType(adapters),
        rng_manifest=MappingProxyType(rng),
    )


class SoulBlobStore:
    """Atomic content-addressed filesystem store for runtime soul blobs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise SoulStoreError("soul store root is not a directory")

    def path_for(self, sha256: str) -> Path:
        digest = _valid_hash(sha256)
        return self.root / digest[:2] / f"{digest}.soul"

    def persist(self, blob: SoulBlob) -> StoredSoulBlob:
        if not isinstance(blob, SoulBlob):
            raise TypeError("blob must be SoulBlob")
        decoded = decode_soul_blob(blob.data, blob.sha256)
        target = self.path_for(blob.sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = target.read_bytes()
            if hashlib.sha256(existing).hexdigest() != blob.sha256:
                raise SoulBlobCorruptionError(
                    "existing content-addressed soul blob is corrupt"
                )
            if existing != blob.data:
                raise SoulBlobCorruptionError(
                    "existing soul blob bytes do not match staged content"
                )
            return StoredSoulBlob(
                blob.sha256,
                target,
                decoded.header,
            )

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{blob.sha256}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(blob.data)
                handle.flush()
                os.fsync(handle.fileno())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != blob.sha256:
                raise SoulBlobCorruptionError(
                    "temporary soul blob failed hash validation"
                )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        if hashlib.sha256(target.read_bytes()).hexdigest() != blob.sha256:
            raise SoulBlobCorruptionError(
                "persisted soul blob failed hash validation"
            )
        return StoredSoulBlob(blob.sha256, target, decoded.header)

    def write(
        self,
        soul: torch.Tensor,
        mask: torch.Tensor,
        *,
        soul_id: str,
        generation: int,
        base_field_id: str,
    ) -> StoredSoulBlob:
        return self.persist(
            encode_soul_blob(
                soul,
                mask,
                soul_id=soul_id,
                generation=generation,
                base_field_id=base_field_id,
            )
        )

    def load(self, sha256: str) -> DecodedSoulBlob:
        digest = _valid_hash(sha256)
        path = self.path_for(digest)
        if not path.is_file():
            raise FileNotFoundError(path)
        return decode_soul_blob(path.read_bytes(), digest)


class PrivateRuntimeStateStore:
    """Atomic content-addressed store for crash-continuity metadata."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise SoulStoreError(
                "private runtime-state store root is not a directory"
            )

    def path_for(self, sha256: str) -> Path:
        digest = _valid_hash(sha256)
        return self.root / digest[:2] / f"{digest}.state"

    def persist(
        self,
        blob: PrivateRuntimeStateBlob,
    ) -> StoredPrivateRuntimeState:
        if not isinstance(blob, PrivateRuntimeStateBlob):
            raise TypeError("blob must be PrivateRuntimeStateBlob")
        decoded = decode_private_runtime_state(blob.data, blob.sha256)
        target = self.path_for(blob.sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = target.read_bytes()
            if (
                hashlib.sha256(existing).hexdigest() != blob.sha256
                or existing != blob.data
            ):
                raise SoulBlobCorruptionError(
                    "existing content-addressed runtime state is corrupt"
                )
            return StoredPrivateRuntimeState(
                blob.sha256,
                target,
                decoded.cursor_sha256,
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{blob.sha256}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(blob.data)
                handle.flush()
                os.fsync(handle.fileno())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != blob.sha256:
                raise SoulBlobCorruptionError(
                    "temporary private runtime state failed hash validation"
                )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        if hashlib.sha256(target.read_bytes()).hexdigest() != blob.sha256:
            raise SoulBlobCorruptionError(
                "persisted private runtime state failed hash validation"
            )
        return StoredPrivateRuntimeState(
            blob.sha256,
            target,
            decoded.cursor_sha256,
        )

    def write(self, **kwargs: Any) -> StoredPrivateRuntimeState:
        return self.persist(encode_private_runtime_state(**kwargs))

    def load(self, sha256: str) -> DecodedPrivateRuntimeState:
        digest = _valid_hash(sha256)
        path = self.path_for(digest)
        if not path.is_file():
            raise FileNotFoundError(path)
        return decode_private_runtime_state(path.read_bytes(), digest)


__all__ = [
    "SOUL_BLOB_SCHEMA",
    "SOUL_BLOB_MAGIC",
    "PRIVATE_RUNTIME_STATE_SCHEMA",
    "PRIVATE_RUNTIME_STATE_MAGIC",
    "SoulStoreError",
    "SoulBlobContractError",
    "SoulBlobCorruptionError",
    "SoulBlob",
    "DecodedSoulBlob",
    "StoredSoulBlob",
    "PrivateRuntimeStateBlob",
    "DecodedPrivateRuntimeState",
    "StoredPrivateRuntimeState",
    "encode_soul_blob",
    "decode_soul_blob",
    "encode_private_runtime_state",
    "decode_private_runtime_state",
    "SoulBlobStore",
    "PrivateRuntimeStateStore",
]

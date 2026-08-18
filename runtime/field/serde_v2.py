"""Strict shared-field-v2 serialization and deserialization.

This module deliberately never imports or calls the v1 field serializer.  It
operates only on the isolated v2 schema types defined in
:mod:`runtime.field.schema_v2`.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .schema_v2 import (
    CHARTER_SCHEMA_V2,
    FIELD_SCHEMA_V2,
    CANONICAL_REGION_ORDER_V2,
    LOGICAL_REGION_IDS_V2,
    FieldSpanV2,
    IdentityCharterV2,
    LogicalRegionV2,
    RegionStateV2,
    RegionVisibilityV2,
    SharedFieldSnapshotV2,
    WritePolicyV2,
    canonical_json_bytes,
)


class SerdeV2Error(ValueError):
    """A v2 field payload is malformed or violates the v2 contract."""


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _strict_json_loads(text: str | bytes) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SerdeV2Error(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise SerdeV2Error(f"non-finite JSON constant {token}")

    try:
        return json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except SerdeV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SerdeV2Error("invalid v2 field JSON") from exc


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise SerdeV2Error(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_identity_region(
    region: RegionStateV2,
    source_manifest_ids: tuple[str, ...],
) -> None:
    """Verify that the identity region honors the sealed charter contract."""

    if region.name is not LogicalRegionV2.IDENTITY:
        raise SerdeV2Error(
            "identity region validation called on non-identity region"
        )

    if region.visibility is not RegionVisibilityV2.ATTENDED:
        raise SerdeV2Error("identity region must be attended")

    if region.write_policy is not WritePolicyV2.SEALED:
        raise SerdeV2Error("identity region must be sealed")

    if len(region.spans) != 1:
        raise SerdeV2Error(
            f"identity region must contain exactly one charter span, "
            f"got {len(region.spans)}"
        )

    span = region.spans[0]
    if span.kind != "identity_charter":
        raise SerdeV2Error("identity span kind must be 'identity_charter'")

    source = span.source
    prefix = f"{CHARTER_SCHEMA_V2}:"
    if not source.startswith(prefix):
        raise SerdeV2Error(
            "identity span source must declare the charter schema and hash"
        )

    declared_hash = source[len(prefix) :]
    if len(declared_hash) != 64:
        raise SerdeV2Error(
            "identity span source declares an invalid charter hash length"
        )

    if _HEX64_RE.fullmatch(declared_hash) is None:
        raise SerdeV2Error(
            "identity span source declares a non-hex charter hash"
        )

    try:
        charter = IdentityCharterV2(text=span.text)
    except SerdeV2Error:
        raise
    except (TypeError, ValueError) as exc:
        raise SerdeV2Error(f"identity span text is not a valid charter: {exc}") from exc

    if charter.charter_hash != declared_hash:
        raise SerdeV2Error(
            "identity span text does not match declared charter hash"
        )

    if charter.source_manifest_id not in source_manifest_ids:
        raise SerdeV2Error(
            "snapshot source_manifest_ids is missing the charter source manifest"
        )


def _field_span_from_dict(value: dict[str, Any]) -> FieldSpanV2:
    _require_exact_keys(
        value,
        {
            "span_id",
            "text",
            "kind",
            "source",
            "provenance",
            "confidence",
            "container_refs",
            "edge_refs",
        },
        "span",
    )
    if not isinstance(value["container_refs"], list):
        raise SerdeV2Error("span container_refs must be a JSON array")
    if not isinstance(value["edge_refs"], list):
        raise SerdeV2Error("span edge_refs must be a JSON array")
    try:
        return FieldSpanV2(
            span_id=value["span_id"],
            text=value["text"],
            kind=value["kind"],
            source=value["source"],
            provenance=value["provenance"],
            confidence=value["confidence"],
            container_refs=tuple(value["container_refs"]),
            edge_refs=tuple(value["edge_refs"]),
        )
    except (TypeError, ValueError) as exc:
        raise SerdeV2Error(f"span construction failed: {exc}") from exc


def _region_state_from_dict(value: dict[str, Any]) -> RegionStateV2:
    _require_exact_keys(
        value, {"name", "visibility", "write_policy", "spans"}, "region"
    )
    spans_raw = value["spans"]
    if not isinstance(spans_raw, list):
        raise SerdeV2Error("region spans must be a JSON array")
    if not all(isinstance(span, dict) for span in spans_raw):
        raise SerdeV2Error("every region span must be a JSON object")
    try:
        spans = tuple(_field_span_from_dict(span) for span in spans_raw)
        return RegionStateV2(
            name=value["name"],
            spans=spans,
            visibility=value["visibility"],
            write_policy=value["write_policy"],
        )
    except (TypeError, ValueError) as exc:
        raise SerdeV2Error(f"region construction failed: {exc}") from exc


def _require_hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise SerdeV2Error(
            f"declared {label} must be a 64-character lowercase hex SHA-256"
        )
    return value


def deserialize_shared_field_v2(text: str | bytes) -> SharedFieldSnapshotV2:
    """Parse a strict v2 field payload and verify every contract gate.

    Rejects v1 payloads, unknown keys, duplicate JSON keys, non-finite values,
    duplicate or wrongly-ordered regions, a missing or malformed identity
    region, mutable identity policy, and any field/canonical hash mismatch.
    """

    value = _strict_json_loads(text)
    if not isinstance(value, dict):
        raise SerdeV2Error("v2 field payload must be a JSON object")

    schema = value.get("schema")
    if schema == "shared-field-v1":
        raise SerdeV2Error("v2 serde rejects v1 field payload")
    if schema != FIELD_SCHEMA_V2:
        raise SerdeV2Error(f"unknown field schema {schema!r}")

    _require_exact_keys(
        value,
        {
            "schema",
            "tick_id",
            "parent_field_id",
            "source_manifest_ids",
            "regions",
            "field_id",
            "canonical_hash",
        },
        "snapshot",
    )

    tick_id = value["tick_id"]
    if isinstance(tick_id, bool) or not isinstance(tick_id, int) or tick_id < 0:
        raise SerdeV2Error("tick_id must be a non-negative integer")

    parent_field_id = value["parent_field_id"]
    if tick_id == 0:
        if parent_field_id is not None:
            raise SerdeV2Error(
                "genesis snapshot (tick_id == 0) must have parent_field_id=None"
            )
    elif parent_field_id is None:
        raise SerdeV2Error(
            "non-genesis snapshot must declare a parent_field_id hash link"
        )
    elif (
        not isinstance(parent_field_id, str)
        or _HEX64_RE.fullmatch(parent_field_id) is None
    ):
        raise SerdeV2Error(
            "parent_field_id must be a 64-character lowercase hex SHA-256"
        )

    source_manifest_ids = value["source_manifest_ids"]
    if isinstance(source_manifest_ids, (str, bytes, bytearray)) or not isinstance(
        source_manifest_ids, list
    ) or not all(isinstance(item, str) and item for item in source_manifest_ids):
        raise SerdeV2Error(
            "source_manifest_ids must be a list of non-empty strings"
        )

    regions_raw = value["regions"]
    if not isinstance(regions_raw, list):
        raise SerdeV2Error("regions must be a list")

    if len(regions_raw) != len(CANONICAL_REGION_ORDER_V2):
        raise SerdeV2Error(
            f"v2 snapshot requires exactly {len(CANONICAL_REGION_ORDER_V2)} regions"
        )

    identity_index = LOGICAL_REGION_IDS_V2[LogicalRegionV2.IDENTITY]
    identity_region_dict = regions_raw[identity_index]
    if not isinstance(identity_region_dict, dict):
        raise SerdeV2Error("identity region must be an object")
    if identity_region_dict.get("name") != LogicalRegionV2.IDENTITY.value:
        raise SerdeV2Error(
            f"region at index {identity_index} must be {LogicalRegionV2.IDENTITY.value!r}"
        )
    if identity_region_dict.get("visibility") != RegionVisibilityV2.ATTENDED.value:
        raise SerdeV2Error("identity region must be attended")
    if identity_region_dict.get("write_policy") != WritePolicyV2.SEALED.value:
        raise SerdeV2Error("identity region must be sealed")
    identity_spans = identity_region_dict.get("spans")
    if not isinstance(identity_spans, list) or len(identity_spans) != 1:
        raise SerdeV2Error(
            f"identity region must contain exactly one charter span, "
            f"got {len(identity_spans) if isinstance(identity_spans, list) else type(identity_spans).__name__}"
        )

    try:
        regions: list[RegionStateV2] = []
        for index, region_dict in enumerate(regions_raw):
            expected_name = CANONICAL_REGION_ORDER_V2[index].value
            if not isinstance(region_dict, dict):
                raise SerdeV2Error(f"region at index {index} must be an object")
            if region_dict.get("name") != expected_name:
                raise SerdeV2Error(
                    f"region at index {index} must be {expected_name!r} "
                    f"in fixed v2 order"
                )
            region = _region_state_from_dict(region_dict)
            if region.name != CANONICAL_REGION_ORDER_V2[index]:
                raise SerdeV2Error(
                    f"region at index {index} has unexpected name "
                    f"{region.name.value!r}"
                )
            regions.append(region)
    except SerdeV2Error:
        raise
    except (TypeError, ValueError) as exc:
        raise SerdeV2Error(f"region/span parsing failed: {exc}") from exc

    identity_region = regions[identity_index]
    _validate_identity_region(identity_region, tuple(source_manifest_ids))

    try:
        snapshot = SharedFieldSnapshotV2(
            tick_id=tick_id,
            regions=tuple(regions),
            parent_field_id=parent_field_id,
            source_manifest_ids=tuple(source_manifest_ids),
        )
    except SerdeV2Error:
        raise
    except (TypeError, ValueError) as exc:
        raise SerdeV2Error(f"snapshot construction failed: {exc}") from exc

    declared_field_id = _require_hex64(value["field_id"], "field_id")
    declared_canonical_hash = _require_hex64(value["canonical_hash"], "canonical_hash")

    if declared_field_id != snapshot.field_id:
        raise SerdeV2Error(
            "declared field_id does not match recomputed canonical hash"
        )
    if declared_canonical_hash != snapshot.canonical_hash:
        raise SerdeV2Error(
            "declared canonical_hash does not match recomputed canonical hash"
        )

    return snapshot


def serialize_shared_field_v2(snapshot: SharedFieldSnapshotV2) -> bytes:
    """Return the canonical JSON byte representation of ``snapshot``.

    The persisted form includes both ``field_id`` and ``canonical_hash`` so that
    every byte payload carries and verifies its own content hash.  All primitive
    inputs are captured inside a guarded boundary; the supplied hashes must be
    exact built-in ``str`` instances and must match the fresh canonical result
    before any bytes are emitted.  Missing or malformed primitive attributes on
    an exact-base forged object normalize to ``SerdeV2Error``.
    """

    if type(snapshot) is not SharedFieldSnapshotV2:
        raise TypeError(
            "serialize_shared_field_v2 requires the exact base SharedFieldSnapshotV2"
        )

    try:
        tick_id = snapshot.tick_id
        regions = snapshot.regions
        parent_field_id = snapshot.parent_field_id
        source_manifest_ids = snapshot.source_manifest_ids
        supplied_field_id = snapshot.field_id
        supplied_canonical_hash = snapshot.canonical_hash
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise SerdeV2Error(
            "snapshot is missing required primitive fields"
        ) from exc

    if (
        type(supplied_field_id) is not str
        or _HEX64_RE.fullmatch(supplied_field_id) is None
    ):
        raise SerdeV2Error(
            "supplied field_id must be a 64-character lowercase hex SHA-256"
        )
    if (
        type(supplied_canonical_hash) is not str
        or _HEX64_RE.fullmatch(supplied_canonical_hash) is None
    ):
        raise SerdeV2Error(
            "supplied canonical_hash must be a 64-character lowercase hex SHA-256"
        )

    try:
        fresh = SharedFieldSnapshotV2(
            tick_id=tick_id,
            regions=regions,
            parent_field_id=parent_field_id,
            source_manifest_ids=source_manifest_ids,
        )
    except (TypeError, ValueError) as exc:
        raise SerdeV2Error(f"snapshot reconstruction failed: {exc}") from exc

    if supplied_field_id != fresh.field_id:
        raise SerdeV2Error(
            "supplied field_id does not match reconstructed canonical hash"
        )
    if supplied_canonical_hash != fresh.canonical_hash:
        raise SerdeV2Error(
            "supplied canonical_hash does not match reconstructed canonical hash"
        )

    return canonical_json_bytes(SharedFieldSnapshotV2.to_dict(fresh))


__all__ = [
    "SerdeV2Error",
    "deserialize_shared_field_v2",
    "serialize_shared_field_v2",
]

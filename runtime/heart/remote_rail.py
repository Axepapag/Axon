"""Exact transport of an attended D64 rail, never a canonical-state export.

The receiving worker reconstructs frozen 16D cells from categorical receipts.
The local Heart retains authority, the expected rail identity, and all dormant
state. Authentication/worker leases belong to the transport using this codec.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from runtime.field import LogicalRegion, canonical_json_bytes, canonical_sha256
from runtime.field.compiler_d64 import (
    D64_COMPILER_SCHEMA,
    CanonicalCharAddress,
    CartographicSpan,
    CompiledD64Field,
    D64CoverageManifest,
    _rows_respect_source_boundaries,
    _transport_addresses_complete,
    _verify_vector_roundtrip,
)
from substrate import decode_unicode_tokens, transport_token_cell16, transport_token_kind, transport_token_value

RAIL_WIRE_SCHEMA = "axon-remote-attended-d64-v1"


class RemoteRailError(ValueError):
    """A received rail is not the exact authorized Heart projection."""


def encode_attended_rail(rail: CompiledD64Field) -> dict[str, Any]:
    """Send only rail receipts; categorical IDs reproduce every FP32 lane."""
    value = {
        "schema": RAIL_WIRE_SCHEMA,
        "rail_id": rail.rail_id,
        "source_field_id": rail.source_field_id,
        "source_tick_id": rail.source_tick_id,
        "coverage": rail.coverage.to_canonical_dict(),
        "addresses": [None if item is None else item.to_canonical_dict() for item in rail.addresses],
        "cartography": [item.to_canonical_dict() for item in rail.cartography],
    }
    # Prove that this codec can reproduce the actual outgoing vectors, rather
    # than silently substituting a codebook for arbitrary neural vectors.
    restored = decode_attended_rail(value, expected_rail_id=rail.rail_id)
    if not np.array_equal(restored.rows, rail.rows) or not np.array_equal(restored.lane_valid, rail.lane_valid):
        raise RemoteRailError("outgoing rail is not exact categorical substrate")
    return value


def decode_attended_rail(value: Mapping[str, Any], *, expected_rail_id: str) -> CompiledD64Field:
    """Reconstruct exact packed vectors and verify the Heart's expected hash.

    ``expected_rail_id`` must come from the authenticated assignment envelope,
    not from an untrusted worker's self-declared result.
    """
    try:
        return _decode(value, expected_rail_id)
    except RemoteRailError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise RemoteRailError("malformed remote rail") from exc


def _decode(value: Mapping[str, Any], expected_rail_id: str) -> CompiledD64Field:
    if value["schema"] != RAIL_WIRE_SCHEMA or value["rail_id"] != expected_rail_id:
        raise RemoteRailError("remote rail schema or authorized identity mismatch")
    coverage_body = dict(value["coverage"])
    for name in ("expected_regions", "visited_regions", "region_row_ranges"):
        coverage_body[name] = tuple(
            tuple(item) if isinstance(item, list) else item for item in coverage_body[name]
        )
    coverage = D64CoverageManifest(**coverage_body)
    if coverage.schema != D64_COMPILER_SCHEMA or coverage.complete is not True:
        raise RemoteRailError("remote rail lacks complete compiler coverage")
    if coverage.source_field_id != value["source_field_id"] or coverage.source_tick_id != value["source_tick_id"]:
        raise RemoteRailError("remote rail field/tick binding mismatch")
    raw_addresses = value["addresses"]
    if len(raw_addresses) % 4 or coverage.row_count != len(raw_addresses) // 4:
        raise RemoteRailError("remote rail lane count mismatch")
    rows = np.zeros((len(raw_addresses) // 4, 64), dtype=np.float32)
    valid = np.zeros((len(raw_addresses) // 4, 4), dtype=np.bool_)
    addresses = []
    for index, raw in enumerate(raw_addresses):
        if raw is None:
            addresses.append(None)
            continue
        item = CanonicalCharAddress(**{**raw, "region": LogicalRegion(raw["region"])})
        row, lane = divmod(index, 4)
        if item.row_index != row or item.lane_index != lane:
            raise RemoteRailError("remote rail address order mismatch")
        if (
            item.transport_kind != transport_token_kind(item.transport_token_id)
            or item.transport_value != transport_token_value(item.transport_token_id)
        ):
            raise RemoteRailError("remote rail categorical receipt mismatch")
        valid[row, lane] = True
        rows[row, lane * 16:(lane + 1) * 16] = transport_token_cell16(item.transport_token_id)
        addresses.append(item)
    addresses = tuple(addresses)
    occupied = tuple(item for item in addresses if item is not None)
    if not _rows_respect_source_boundaries(addresses) or not _transport_addresses_complete(occupied):
        raise RemoteRailError("remote rail splits a scalar or crosses a source boundary")
    text = decode_unicode_tokens(item.transport_token_id for item in occupied)
    hashes = {
        "rows_sha256": hashlib.sha256(rows.astype("<f4", copy=False).tobytes(order="C")).hexdigest(),
        "address_sha256": canonical_sha256(raw_addresses),
        "roundtrip_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    if any(getattr(coverage, key) != observed for key, observed in hashes.items()):
        raise RemoteRailError("remote rail vector/address/text checksum mismatch")
    if (
        coverage.compiled_active_characters != len(text)
        or coverage.expected_active_characters != len(text)
        or coverage.compiled_transport_units != len(occupied)
        or coverage.valid_lanes != len(occupied)
        or coverage.padding_lanes != len(addresses) - len(occupied)
        or coverage.expected_regions != coverage.visited_regions
    ):
        raise RemoteRailError("remote rail coverage counts disagree")
    cartography = tuple(
        CartographicSpan(**{**item, "region": LogicalRegion(item["region"])}) for item in value["cartography"]
    )
    identity = canonical_sha256({
        "schema": D64_COMPILER_SCHEMA,
        "source_field_id": value["source_field_id"],
        "source_tick_id": value["source_tick_id"],
        "coverage": coverage.to_canonical_dict(),
        "cartography": [item.to_canonical_dict() for item in cartography],
    })
    if identity != expected_rail_id:
        raise RemoteRailError("remote rail differs from the authorized Heart projection")
    rail = CompiledD64Field(
        source_field_id=value["source_field_id"], source_tick_id=value["source_tick_id"],
        rows=rows, lane_valid=valid, addresses=addresses, cartography=cartography,
        coverage=coverage, rail_id=identity,
    )
    _verify_vector_roundtrip(rail)
    return rail


def rail_wire_bytes(rail: CompiledD64Field) -> bytes:
    """Deterministic bytes suitable for authenticated, optionally compressed HTTP."""
    return canonical_json_bytes(encode_attended_rail(rail))

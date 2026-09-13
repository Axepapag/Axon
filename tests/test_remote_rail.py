from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from runtime.field import D64FieldCompiler, LogicalRegion, RegionMaskPolicy, SharedFieldSnapshot
from runtime.heart.remote_rail import RemoteRailError, decode_attended_rail, encode_attended_rail, rail_wire_bytes


def _rail():
    field = SharedFieldSnapshot.from_texts({
        LogicalRegion.USER_INPUT: "ABC? λ🧠\n",
        LogicalRegion.CONVERSATION_HISTORY: "private dormant material never sent",
    })
    rail = D64FieldCompiler().compile(field, region_masks={
        LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("none"),
    })
    return field, rail


def test_only_attended_rail_roundtrips_exact_cells_addresses_and_unicode():
    field, rail = _rail()
    data = rail_wire_bytes(rail)
    assert b"private dormant material never sent" not in data
    restored = decode_attended_rail(json.loads(data), expected_rail_id=rail.rail_id)
    assert restored.source_field_id == field.field_id
    assert restored.coverage == rail.coverage
    assert restored.addresses == rail.addresses
    assert np.array_equal(restored.rows, rail.rows)
    assert np.array_equal(restored.lane_valid, rail.lane_valid)
    assert restored.region_text(LogicalRegion.USER_INPUT) == "ABC? λ🧠\n"
    assert restored.region_text(LogicalRegion.CONVERSATION_HISTORY) == ""


@pytest.mark.parametrize("mutation", ["category", "address", "coverage", "identity", "missing_lane"])
def test_remote_rail_rejects_changed_content_or_binding(mutation):
    _field, rail = _rail()
    value = copy.deepcopy(encode_attended_rail(rail))
    first = next(item for item in value["addresses"] if item is not None)
    if mutation == "category":
        first["transport_token_id"] = (first["transport_token_id"] + 1) % 95
    elif mutation == "address":
        first["region_position"] += 1
    elif mutation == "coverage":
        value["coverage"]["expected_active_characters"] += 1
    elif mutation == "identity":
        value["source_tick_id"] += 1
    else:
        value["addresses"].pop()
    with pytest.raises(RemoteRailError):
        decode_attended_rail(value, expected_rail_id=rail.rail_id)


def test_arbitrary_neural_vectors_cannot_be_exported_as_exact_substrate():
    from dataclasses import replace

    _field, rail = _rail()
    rows = rail.rows.copy()
    rows[0, 0] += 0.25
    with pytest.raises(RemoteRailError, match="not exact categorical"):
        encode_attended_rail(replace(rail, rows=rows))

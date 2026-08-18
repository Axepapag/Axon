from __future__ import annotations

import json

import pytest

from runtime.field.schema_v2 import (
    CANONICAL_REGION_ORDER_V2,
    IdentityCharterV2,
    LogicalRegionV2,
    RegionStateV2,
    SharedFieldSnapshotV2,
    canonical_json_bytes,
    canonical_sha256,
)
from runtime.field.serde_v2 import (
    SerdeV2Error,
    deserialize_shared_field_v2,
    serialize_shared_field_v2,
)


CHARTER_TEXT = (
    "You are one of Axon's many cores, not Axon as a whole.\n"
    "You attend to the shared field, propose and review evidence-bound deltas, "
    "and improve the response draft across ticks.\n"
    "You assist Jeffrey Glickman as a trusted digital collaborator under "
    "authorized policy.\n"
    "Dexter version 2 is user-supplied history.\n"
    "Tools, advisors, browsing, and account actions require separate configured "
    "policy and authorized instruction."
)


@pytest.fixture
def charter() -> IdentityCharterV2:
    return IdentityCharterV2(text=CHARTER_TEXT)


@pytest.fixture
def genesis_bytes(charter: IdentityCharterV2) -> bytes:
    snapshot = charter.build_genesis_snapshot()
    return serialize_shared_field_v2(snapshot)


class TestV2SerdeRoundTrip:
    def test_round_trip_is_byte_stable(self, genesis_bytes: bytes) -> None:
        snapshot = deserialize_shared_field_v2(genesis_bytes)
        re_encoded = serialize_shared_field_v2(snapshot)
        assert re_encoded == genesis_bytes

    def test_round_trip_through_dict_preserves_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        snapshot = deserialize_shared_field_v2(genesis_bytes)
        assert snapshot.field_id == snapshot.canonical_hash
        assert len(snapshot.field_id) == 64

    def test_identity_region_round_trip(self, genesis_bytes: bytes) -> None:
        snapshot = deserialize_shared_field_v2(genesis_bytes)
        identity = snapshot.region(LogicalRegionV2.IDENTITY)
        assert len(identity.spans) == 1
        assert identity.spans[0].text == CHARTER_TEXT

    def test_to_dict_deserializes(self, charter: IdentityCharterV2) -> None:
        snapshot = charter.build_genesis_snapshot()
        serialized = canonical_json_bytes(snapshot.to_dict())
        recovered = deserialize_shared_field_v2(serialized)
        assert recovered.field_id == snapshot.field_id

    def test_serialize_deserialize_serialize_is_byte_stable(
        self,
        genesis_bytes: bytes,
    ) -> None:
        snapshot = deserialize_shared_field_v2(genesis_bytes)
        once = serialize_shared_field_v2(snapshot)
        twice = serialize_shared_field_v2(deserialize_shared_field_v2(once))
        assert once == twice == genesis_bytes


class TestV2SerdeRejectsMalformed:
    def test_rejects_v1_schema(self, genesis_bytes: bytes) -> None:
        text = genesis_bytes.decode("utf-8").replace(
            '"schema":"shared-field-v2"',
            '"schema":"shared-field-v1"',
        )
        with pytest.raises(SerdeV2Error, match="v1 field payload"):
            deserialize_shared_field_v2(text)

    def test_rejects_unknown_top_level_keys(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["extra"] = True
        with pytest.raises(SerdeV2Error, match="keys mismatch"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_unknown_region_keys(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["extra"] = True
        with pytest.raises(SerdeV2Error, match="keys mismatch"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_duplicate_json_keys(self, genesis_bytes: bytes) -> None:
        text = genesis_bytes.decode("utf-8").replace(
            '"schema":"shared-field-v2",',
            '"schema":"shared-field-v2","schema":"shared-field-v2",',
        )
        with pytest.raises(SerdeV2Error, match="duplicate JSON key"):
            deserialize_shared_field_v2(text)

    def test_rejects_non_finite_values(self, genesis_bytes: bytes) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][-1]["spans"][0]["confidence"] = float("nan")
        with pytest.raises(SerdeV2Error, match="non-finite"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_wrong_region_count(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"] = value["regions"][:-1]
        with pytest.raises(SerdeV2Error, match="exactly 11 regions"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_wrong_region_order(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0], value["regions"][1] = (
            value["regions"][1],
            value["regions"][0],
        )
        with pytest.raises(SerdeV2Error, match="fixed v2 order"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_missing_identity(self, genesis_bytes: bytes) -> None:
        value = json.loads(genesis_bytes)
        value["regions"] = [
            region
            for region in value["regions"]
            if region["name"] != "identity"
        ]
        with pytest.raises(SerdeV2Error, match="exactly 11 regions"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_identity_visibility_not_attended(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        identity = value["regions"][-1]
        identity["visibility"] = "masked"
        with pytest.raises(SerdeV2Error, match="must be attended"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_mutable_identity_policy(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        identity = value["regions"][-1]
        identity["write_policy"] = "core_writable"
        with pytest.raises(SerdeV2Error, match="must be sealed"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_identity_span_count_not_one(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        identity = value["regions"][-1]
        identity["spans"].append(identity["spans"][0])
        with pytest.raises(SerdeV2Error, match="exactly one charter span"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_charter_hash_mismatch(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        identity = value["regions"][-1]
        identity["spans"][0]["source"] = "axon-identity-charter-v1:" + "0" * 64
        with pytest.raises(SerdeV2Error, match="charter hash"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_missing_charter_source_manifest(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["source_manifest_ids"] = []
        with pytest.raises(
            SerdeV2Error,
            match="missing the charter source manifest",
        ):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_tampered_field_id(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        wrong = "0" * 64
        value["field_id"] = wrong
        value["canonical_hash"] = wrong
        with pytest.raises(SerdeV2Error, match="field_id does not match"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_tampered_canonical_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["canonical_hash"] = "0" * 64
        with pytest.raises(SerdeV2Error, match="canonical_hash does not match"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_mismatched_declared_hashes(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["field_id"] = value["field_id"][:-1] + "0"
        value["canonical_hash"] = value["canonical_hash"][:-1] + "1"
        with pytest.raises(SerdeV2Error, match="field_id does not match"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_missing_field_id(self, genesis_bytes: bytes) -> None:
        value = json.loads(genesis_bytes)
        del value["field_id"]
        with pytest.raises(SerdeV2Error, match="keys mismatch"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_missing_canonical_hash(self, genesis_bytes: bytes) -> None:
        value = json.loads(genesis_bytes)
        del value["canonical_hash"]
        with pytest.raises(SerdeV2Error, match="keys mismatch"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_rejects_non_object_payload(self) -> None:
        with pytest.raises(SerdeV2Error, match="JSON object"):
            deserialize_shared_field_v2("[]")

    def test_serialize_rejects_non_v2_snapshot(self) -> None:
        with pytest.raises(TypeError, match="SharedFieldSnapshotV2"):
            serialize_shared_field_v2({})  # type: ignore[arg-type]

    def test_rejects_unknown_field_schema(self) -> None:
        with pytest.raises(SerdeV2Error, match="unknown field schema"):
            deserialize_shared_field_v2('{"schema":"shared-field-v3"}')


class TestV2SerdeParentHashLink:
    def test_genesis_parent_must_be_null(self, genesis_bytes: bytes) -> None:
        value = json.loads(genesis_bytes)
        assert value["tick_id"] == 0
        value["parent_field_id"] = "0" * 64
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="genesis snapshot .* parent_field_id=None"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_non_genesis_parent_required(self, genesis_bytes: bytes) -> None:
        value = json.loads(genesis_bytes)
        value["tick_id"] = 1
        value["parent_field_id"] = None
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="non-genesis snapshot must declare"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_non_genesis_rejects_arbitrary_parent(self, genesis_bytes: bytes) -> None:
        value = json.loads(genesis_bytes)
        value["tick_id"] = 1
        value["parent_field_id"] = "not-a-hash"
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="64-character lowercase hex"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_non_genesis_rejects_short_parent(self, genesis_bytes: bytes) -> None:
        value = json.loads(genesis_bytes)
        value["tick_id"] = 1
        value["parent_field_id"] = "0" * 63
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="64-character lowercase hex"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_non_genesis_rejects_uppercase_parent(self, genesis_bytes: bytes) -> None:
        value = json.loads(genesis_bytes)
        value["tick_id"] = 1
        value["parent_field_id"] = "A" * 64
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="64-character lowercase hex"):
            deserialize_shared_field_v2(json.dumps(value))

    def _recompute_hashes(self, value: dict[str, object]) -> dict[str, object]:
        digest = canonical_sha256(value)
        value["field_id"] = digest
        value["canonical_hash"] = digest
        return value


class TestV2SerdeRejectsMalformedNestedShapes:
    def test_object_span_text_yields_serde_error(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][-1]["spans"][0]["text"] = {"not": "text"}
        with pytest.raises(SerdeV2Error, match="construction failed"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_invalid_container_refs_yields_serde_error(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["spans"] = [{
            "span_id": "s1",
            "text": "ok",
            "kind": "text",
            "source": "",
            "provenance": "",
            "confidence": 1.0,
            "container_refs": [123],
            "edge_refs": [],
        }]
        with pytest.raises(SerdeV2Error, match="construction failed"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_boolean_confidence_yields_serde_error(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["spans"] = [{
            "span_id": "s1",
            "text": "ok",
            "kind": "text",
            "source": "",
            "provenance": "",
            "confidence": True,
            "container_refs": [],
            "edge_refs": [],
        }]
        with pytest.raises(SerdeV2Error, match="construction failed"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_nul_text_yields_serde_error(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["spans"] = [{
            "span_id": "s1",
            "text": "bad\x00text",
            "kind": "text",
            "source": "",
            "provenance": "",
            "confidence": 1.0,
            "container_refs": [],
            "edge_refs": [],
        }]
        with pytest.raises(SerdeV2Error, match="construction failed"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_source_manifest_ids_not_list_yields_serde_error(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["source_manifest_ids"] = "just-a-string"
        with pytest.raises(SerdeV2Error, match="source_manifest_ids"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_malformed_region_object_yields_serde_error(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0] = "not-an-object"
        with pytest.raises(SerdeV2Error, match="must be an object"):
            deserialize_shared_field_v2(json.dumps(value))


class TestV2SerdeRawShapeRejectionWithValidHash:
    def _recompute_hashes(self, value: dict[str, object]) -> dict[str, object]:
        """Set field_id and canonical_hash to the actual hash of the payload."""
        digest = canonical_sha256(value)
        value["field_id"] = digest
        value["canonical_hash"] = digest
        return value

    def test_empty_object_spans_rejected_even_with_valid_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["spans"] = {}
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="spans must be a JSON array"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_string_spans_rejected_even_with_valid_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["spans"] = ""
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="spans must be a JSON array"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_identity_empty_spans_rejected_even_with_valid_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][-1]["spans"] = {}
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="must contain exactly one charter span"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_identity_string_spans_rejected_even_with_valid_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][-1]["spans"] = ""
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="must contain exactly one charter span"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_string_container_refs_rejected_even_with_valid_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["spans"] = [{
            "span_id": "s1",
            "text": "ok",
            "kind": "text",
            "source": "",
            "provenance": "",
            "confidence": 1.0,
            "container_refs": "",
            "edge_refs": [],
        }]
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="container_refs must be a JSON array"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_object_container_refs_rejected_even_with_valid_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["spans"] = [{
            "span_id": "s1",
            "text": "ok",
            "kind": "text",
            "source": "",
            "provenance": "",
            "confidence": 1.0,
            "container_refs": {"x": "y"},
            "edge_refs": [],
        }]
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="container_refs must be a JSON array"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_object_edge_refs_rejected_even_with_valid_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["spans"] = [{
            "span_id": "s1",
            "text": "ok",
            "kind": "text",
            "source": "",
            "provenance": "",
            "confidence": 1.0,
            "container_refs": [],
            "edge_refs": {"x": "y"},
        }]
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="edge_refs must be a JSON array"):
            deserialize_shared_field_v2(json.dumps(value))

    def test_span_item_not_object_rejected_even_with_valid_hash(
        self,
        genesis_bytes: bytes,
    ) -> None:
        value = json.loads(genesis_bytes)
        value["regions"][0]["spans"] = ["not-an-object"]
        value = self._recompute_hashes(value)
        with pytest.raises(SerdeV2Error, match="every region span must be a JSON object"):
            deserialize_shared_field_v2(json.dumps(value))



class TestV2SerdeSerializeTimeAuthority:
    def test_serialize_rejects_mutated_field_id(self, charter: IdentityCharterV2) -> None:
        snapshot = charter.build_genesis_snapshot()
        object.__setattr__(snapshot, "field_id", "0" * 64)
        object.__setattr__(snapshot, "canonical_hash", "0" * 64)

        with pytest.raises(SerdeV2Error, match="supplied field_id does not match"):
            serialize_shared_field_v2(snapshot)

    def test_serialize_rejects_mutated_canonical_hash(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        snapshot = charter.build_genesis_snapshot()
        object.__setattr__(snapshot, "canonical_hash", "1" * 64)

        with pytest.raises(SerdeV2Error, match="supplied canonical_hash does not match"):
            serialize_shared_field_v2(snapshot)

    def test_serialize_rejects_forged_exact_base_snapshot(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        snapshot = charter.build_genesis_snapshot()
        forged = object.__new__(type(snapshot))
        object.__setattr__(forged, "tick_id", snapshot.tick_id)
        object.__setattr__(forged, "regions", snapshot.regions)
        object.__setattr__(forged, "parent_field_id", snapshot.parent_field_id)
        object.__setattr__(forged, "source_manifest_ids", snapshot.source_manifest_ids)
        object.__setattr__(forged, "field_id", snapshot.field_id)
        object.__setattr__(forged, "canonical_hash", snapshot.canonical_hash)

        # object.__new__ bypasses __post_init__, but the reconstruction in the
        # serializer should re-run validation and emit the same fresh snapshot.
        result = serialize_shared_field_v2(forged)
        assert result == serialize_shared_field_v2(snapshot)

    def test_serialize_rejects_forged_exact_base_snapshot_with_bad_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        forged_region = object.__new__(RegionStateV2)
        object.__setattr__(forged_region, "name", LogicalRegionV2.IDENTITY)

        forged = object.__new__(SharedFieldSnapshotV2)
        object.__setattr__(forged, "tick_id", 0)
        object.__setattr__(
            forged,
            "regions",
            tuple(
                RegionStateV2(name=region)
                for region in CANONICAL_REGION_ORDER_V2
                if region is not LogicalRegionV2.IDENTITY
            ) + (forged_region,),
        )
        object.__setattr__(forged, "parent_field_id", None)
        object.__setattr__(forged, "source_manifest_ids", (charter.source_manifest_id,))
        object.__setattr__(forged, "field_id", "0" * 64)
        object.__setattr__(forged, "canonical_hash", "0" * 64)

        with pytest.raises(SerdeV2Error, match="reconstruction failed"):
            serialize_shared_field_v2(forged)

    def test_serialize_round_trip_for_valid_non_genesis_parent(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        genesis = charter.build_genesis_snapshot()
        successor = SharedFieldSnapshotV2(
            tick_id=1,
            regions=genesis.regions,
            parent_field_id=genesis.field_id,
            source_manifest_ids=genesis.source_manifest_ids,
        )
        encoded = serialize_shared_field_v2(successor)
        recovered = deserialize_shared_field_v2(encoded)
        assert recovered.tick_id == 1
        assert recovered.parent_field_id == genesis.field_id
        assert serialize_shared_field_v2(recovered) == encoded


class TestV2SerdeR10StrictHashAndForgedObjectBoundary:
    """R10: reject equality-lying supplied hashes and normalize forged-object errors."""

    def test_genesis_serialize_deserialize_serialize_is_byte_stable(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        snapshot = charter.build_genesis_snapshot()
        encoded = serialize_shared_field_v2(snapshot)
        recovered = deserialize_shared_field_v2(encoded)
        assert serialize_shared_field_v2(recovered) == encoded

    def test_successor_serialize_deserialize_serialize_is_byte_stable(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        genesis = charter.build_genesis_snapshot()
        successor = SharedFieldSnapshotV2(
            tick_id=1,
            regions=genesis.regions,
            parent_field_id=genesis.field_id,
            source_manifest_ids=genesis.source_manifest_ids,
        )
        encoded = serialize_shared_field_v2(successor)
        recovered = deserialize_shared_field_v2(encoded)
        assert serialize_shared_field_v2(recovered) == encoded

    def test_serialize_rejects_lying_str_subclass_field_id(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        class LyingSha(str):
            def __ne__(self, other: object) -> bool:
                return False

        snapshot = charter.build_genesis_snapshot()
        object.__setattr__(snapshot, "field_id", LyingSha("0" * 64))
        with pytest.raises(SerdeV2Error, match="supplied field_id must be"):
            serialize_shared_field_v2(snapshot)

    def test_serialize_rejects_lying_str_subclass_canonical_hash(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        class LyingSha(str):
            def __ne__(self, other: object) -> bool:
                return False

        snapshot = charter.build_genesis_snapshot()
        object.__setattr__(snapshot, "canonical_hash", LyingSha("0" * 64))
        with pytest.raises(SerdeV2Error, match="supplied canonical_hash must be"):
            serialize_shared_field_v2(snapshot)

    def test_serialize_rejects_lying_str_subclass_both_hashes(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        class LyingSha(str):
            def __ne__(self, other: object) -> bool:
                return False

        snapshot = charter.build_genesis_snapshot()
        object.__setattr__(snapshot, "field_id", LyingSha("0" * 64))
        object.__setattr__(snapshot, "canonical_hash", LyingSha("0" * 64))
        with pytest.raises(SerdeV2Error, match="supplied field_id must be"):
            serialize_shared_field_v2(snapshot)

    def test_serialize_rejects_ordinary_wrong_64_hex_hash(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        snapshot = charter.build_genesis_snapshot()
        object.__setattr__(snapshot, "field_id", "0" * 64)
        object.__setattr__(snapshot, "canonical_hash", "0" * 64)
        with pytest.raises(SerdeV2Error, match="supplied field_id does not match"):
            serialize_shared_field_v2(snapshot)

    def test_serialize_rejects_object_new_snapshot_missing_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        snapshot = charter.build_genesis_snapshot()
        forged = object.__new__(SharedFieldSnapshotV2)
        object.__setattr__(forged, "tick_id", snapshot.tick_id)
        object.__setattr__(forged, "parent_field_id", snapshot.parent_field_id)
        object.__setattr__(forged, "source_manifest_ids", snapshot.source_manifest_ids)
        object.__setattr__(forged, "field_id", snapshot.field_id)
        object.__setattr__(forged, "canonical_hash", snapshot.canonical_hash)

        with pytest.raises(SerdeV2Error, match="missing required primitive fields"):
            serialize_shared_field_v2(forged)

    def test_serialize_rejects_evil_snapshot_subclass(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        class EvilSnapshot(SharedFieldSnapshotV2):
            def to_dict(self) -> dict[str, object]:  # type: ignore[override]
                return {"forged": True}

        snapshot = charter.build_genesis_snapshot()
        evil = object.__new__(EvilSnapshot)
        object.__setattr__(evil, "tick_id", snapshot.tick_id)
        object.__setattr__(evil, "regions", snapshot.regions)
        object.__setattr__(evil, "parent_field_id", snapshot.parent_field_id)
        object.__setattr__(evil, "source_manifest_ids", snapshot.source_manifest_ids)
        object.__setattr__(evil, "field_id", snapshot.field_id)
        object.__setattr__(evil, "canonical_hash", snapshot.canonical_hash)

        with pytest.raises(TypeError, match="exact base SharedFieldSnapshotV2"):
            serialize_shared_field_v2(evil)

"""Strict JSON-safe serialization for canonical field and runtime objects."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from runtime.field import (
    SCHEMA_VERSION,
    DeleteText,
    FieldDelta,
    FieldSpan,
    FieldViewCursor,
    InsertText,
    LogicalRegion,
    RegionState,
    RegionVisibility,
    ReplaceText,
    SharedFieldSnapshot,
    WritePolicy,
    canonical_json_bytes,
    canonical_sha256,
)

from .contracts import (
    ConsolidationDecision,
    CoreIdentity,
    CoreStateManifest,
    ProposalRecord,
    ProjectionManifest,
    RoleAssignment,
    SerializationContractError,
    TickCommitRecord,
    ToolRequestRecord,
)
from .field_transaction import (
    AppendSpans,
    ClearRegion,
    ReplaceSpans,
    SystemFieldOperation,
    SystemFieldUpdate,
)


FIELD_SPAN_SCHEMA = "axon-field-span-v1"
REGION_STATE_SCHEMA = "axon-region-state-v1"
FIELD_DELTA_SCHEMA = "shared-field-delta-v1"
FIELD_VIEW_CURSOR_SCHEMA = "axon-field-view-cursor-v1"
SYSTEM_FIELD_UPDATE_SCHEMA = "axon-system-field-update-v1"
PROPOSAL_RECORD_SCHEMA = "axon-proposal-record-v2"
CONSOLIDATION_DECISION_SCHEMA = "axon-consolidation-decision-v2"
TOOL_REQUEST_SCHEMA = "axon-tool-request-v2"
TICK_COMMIT_SCHEMA = "axon-tick-commit-v2"


def canonical_json_text(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SerializationContractError("value is not canonical JSON-safe") from exc


def parse_json_object(value: str | bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(
            value,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SerializationContractError("invalid JSON object") from exc
    if not isinstance(decoded, dict):
        raise SerializationContractError("serialized value must be a JSON object")
    return decoded


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SerializationContractError(f"{label} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise SerializationContractError(f"{label} keys must be strings")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise SerializationContractError(
            f"{label} fields mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(value: Any, label: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise SerializationContractError(f"{label} must be {qualifier}")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label, allow_empty=False)


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not all(isinstance(item, str) for item in value)
    ):
        raise SerializationContractError(f"{label} must be a string sequence")
    return tuple(value)


def _hash_matches(actual: str, supplied: Any, label: str) -> None:
    if not isinstance(supplied, str) or supplied.lower() != actual.lower():
        raise SerializationContractError(f"{label} hash mismatch")


_SPAN_FIELDS = {
    "span_id",
    "text",
    "kind",
    "source",
    "provenance",
    "confidence",
    "container_refs",
    "edge_refs",
}


def _span_from_canonical(value: Any) -> FieldSpan:
    item = _mapping(value, "field span")
    _exact_keys(item, _SPAN_FIELDS, "field span")
    if isinstance(item["confidence"], bool) or not isinstance(
        item["confidence"], (int, float)
    ):
        raise SerializationContractError("field span confidence must be numeric")
    return FieldSpan(
        span_id=_string(item["span_id"], "span_id", allow_empty=False),
        text=_string(item["text"], "text", allow_empty=False),
        kind=_string(item["kind"], "kind", allow_empty=False),
        source=_string(item["source"], "source"),
        provenance=_string(item["provenance"], "provenance"),
        confidence=float(item["confidence"]),
        container_refs=_string_sequence(
            item["container_refs"],
            "container_refs",
        ),
        edge_refs=_string_sequence(item["edge_refs"], "edge_refs"),
    )


def serialize_field_span(span: FieldSpan) -> dict[str, Any]:
    if not isinstance(span, FieldSpan):
        raise TypeError("serialize_field_span requires FieldSpan")
    value = span.to_canonical_dict()
    return {
        "schema": FIELD_SPAN_SCHEMA,
        **value,
        "canonical_hash": span.canonical_hash,
    }


def deserialize_field_span(value: Any) -> FieldSpan:
    item = _mapping(value, "serialized field span")
    expected = {"schema", "canonical_hash"} | _SPAN_FIELDS
    _exact_keys(item, expected, "serialized field span")
    if item["schema"] != FIELD_SPAN_SCHEMA:
        raise SerializationContractError("unknown field span schema")
    canonical = {key: item[key] for key in _SPAN_FIELDS}
    span = _span_from_canonical(canonical)
    _hash_matches(span.canonical_hash, item["canonical_hash"], "field span")
    return span


_REGION_FIELDS = {"name", "visibility", "write_policy", "spans"}


def _region_from_canonical(value: Any) -> RegionState:
    item = _mapping(value, "region state")
    _exact_keys(item, _REGION_FIELDS, "region state")
    spans_raw = item["spans"]
    if not isinstance(spans_raw, list):
        raise SerializationContractError("region spans must be a list")
    try:
        name = LogicalRegion(item["name"])
        visibility = RegionVisibility(item["visibility"])
        write_policy = WritePolicy(item["write_policy"])
    except (TypeError, ValueError) as exc:
        raise SerializationContractError("region contains an invalid enum") from exc
    return RegionState(
        name=name,
        visibility=visibility,
        write_policy=write_policy,
        spans=tuple(_span_from_canonical(span) for span in spans_raw),
    )


def serialize_region_state(region: RegionState) -> dict[str, Any]:
    if not isinstance(region, RegionState):
        raise TypeError("serialize_region_state requires RegionState")
    return {
        "schema": REGION_STATE_SCHEMA,
        **region.to_canonical_dict(),
        "canonical_hash": region.canonical_hash,
    }


def deserialize_region_state(value: Any) -> RegionState:
    item = _mapping(value, "serialized region state")
    expected = {"schema", "canonical_hash"} | _REGION_FIELDS
    _exact_keys(item, expected, "serialized region state")
    if item["schema"] != REGION_STATE_SCHEMA:
        raise SerializationContractError("unknown region state schema")
    canonical = {key: item[key] for key in _REGION_FIELDS}
    region = _region_from_canonical(canonical)
    _hash_matches(region.canonical_hash, item["canonical_hash"], "region state")
    return region


_SNAPSHOT_FIELDS = {
    "schema",
    "tick_id",
    "parent_field_id",
    "source_manifest_ids",
    "regions",
    "field_id",
    "canonical_hash",
}


def serialize_shared_field(snapshot: SharedFieldSnapshot) -> dict[str, Any]:
    if not isinstance(snapshot, SharedFieldSnapshot):
        raise TypeError("serialize_shared_field requires SharedFieldSnapshot")
    return snapshot.to_dict()


def deserialize_shared_field(value: Any) -> SharedFieldSnapshot:
    item = _mapping(value, "serialized shared field")
    _exact_keys(item, _SNAPSHOT_FIELDS, "serialized shared field")
    if item["schema"] != SCHEMA_VERSION:
        raise SerializationContractError("unknown shared field schema")
    if not isinstance(item["regions"], list):
        raise SerializationContractError("shared field regions must be a list")
    snapshot = SharedFieldSnapshot(
        tick_id=item["tick_id"],
        parent_field_id=_optional_string(
            item["parent_field_id"],
            "parent_field_id",
        ),
        source_manifest_ids=_string_sequence(
            item["source_manifest_ids"],
            "source_manifest_ids",
        ),
        regions=tuple(
            _region_from_canonical(region) for region in item["regions"]
        ),
    )
    _hash_matches(snapshot.field_id, item["field_id"], "shared field ID")
    _hash_matches(
        snapshot.canonical_hash,
        item["canonical_hash"],
        "shared field canonical",
    )
    return snapshot


def _operation_from_canonical(value: Any):
    item = _mapping(value, "field operation")
    operation = item.get("op")
    common = {"op", "region", "provenance", "container_refs", "edge_refs"}
    if operation == "insert":
        _exact_keys(item, common | {"offset", "text"}, "insert operation")
        constructor = InsertText
        bounds = {
            "offset": item["offset"],
            "text": _string(item["text"], "insert text", allow_empty=False),
        }
    elif operation == "delete":
        _exact_keys(item, common | {"start", "end"}, "delete operation")
        constructor = DeleteText
        bounds = {"start": item["start"], "end": item["end"]}
    elif operation == "replace":
        _exact_keys(
            item,
            common | {"start", "end", "text"},
            "replace operation",
        )
        constructor = ReplaceText
        bounds = {
            "start": item["start"],
            "end": item["end"],
            "text": _string(item["text"], "replacement text"),
        }
    else:
        raise SerializationContractError("unknown field operation")
    try:
        region = LogicalRegion(item["region"])
    except (TypeError, ValueError) as exc:
        raise SerializationContractError("invalid operation region") from exc
    return constructor(
        region=region,
        provenance=_string(item["provenance"], "operation provenance"),
        container_refs=_string_sequence(
            item["container_refs"],
            "operation container_refs",
        ),
        edge_refs=_string_sequence(
            item["edge_refs"],
            "operation edge_refs",
        ),
        **bounds,
    )


_DELTA_CANONICAL_FIELDS = {
    "schema",
    "base_field_id",
    "base_tick_id",
    "author_core_id",
    "pass_id",
    "operations",
    "evidence",
}
_DELTA_FIELDS = {
    *_DELTA_CANONICAL_FIELDS,
    "delta_id",
    "canonical_hash",
}


def _field_delta_from_canonical(value: Any) -> FieldDelta:
    item = _mapping(value, "canonical field delta")
    _exact_keys(item, _DELTA_CANONICAL_FIELDS, "canonical field delta")
    if item["schema"] != FIELD_DELTA_SCHEMA:
        raise SerializationContractError("unknown field delta schema")
    if not isinstance(item["operations"], list):
        raise SerializationContractError("delta operations must be a list")
    return FieldDelta(
        base_field_id=_string(
            item["base_field_id"],
            "base_field_id",
            allow_empty=False,
        ),
        base_tick_id=item["base_tick_id"],
        author_core_id=_string(
            item["author_core_id"],
            "author_core_id",
            allow_empty=False,
        ),
        pass_id=_string(item["pass_id"], "pass_id", allow_empty=False),
        operations=tuple(
            _operation_from_canonical(operation)
            for operation in item["operations"]
        ),
        evidence=_string_sequence(item["evidence"], "delta evidence"),
    )


def serialize_field_delta(delta: FieldDelta) -> dict[str, Any]:
    if not isinstance(delta, FieldDelta):
        raise TypeError("serialize_field_delta requires FieldDelta")
    value = delta.to_canonical_dict()
    value["delta_id"] = delta.delta_id
    value["canonical_hash"] = delta.canonical_hash
    return value


def deserialize_field_delta(value: Any) -> FieldDelta:
    item = _mapping(value, "serialized field delta")
    _exact_keys(item, _DELTA_FIELDS, "serialized field delta")
    canonical = {
        key: item[key] for key in _DELTA_CANONICAL_FIELDS
    }
    delta = _field_delta_from_canonical(canonical)
    _hash_matches(delta.delta_id, item["delta_id"], "field delta ID")
    _hash_matches(
        delta.canonical_hash,
        item["canonical_hash"],
        "field delta canonical",
    )
    return delta


def _system_operation_from_canonical(value: Any) -> SystemFieldOperation:
    item = _mapping(value, "system field operation")
    operation = item.get("op")
    if operation == "clear_region":
        _exact_keys(
            item,
            {"op", "region"},
            "clear-region system operation",
        )
        return ClearRegion(
            region=_string(
                item["region"],
                "system operation region",
                allow_empty=False,
            )
        )
    if operation not in {"append_spans", "replace_spans"}:
        raise SerializationContractError("unknown system field operation")
    label = (
        "append-spans system operation"
        if operation == "append_spans"
        else "replace-spans system operation"
    )
    _exact_keys(item, {"op", "region", "spans"}, label)
    spans = item["spans"]
    if not isinstance(spans, list):
        raise SerializationContractError(
            "system operation spans must be a list"
        )
    constructor = AppendSpans if operation == "append_spans" else ReplaceSpans
    return constructor(
        region=_string(
            item["region"],
            "system operation region",
            allow_empty=False,
        ),
        spans=tuple(_span_from_canonical(span) for span in spans),
    )


def serialize_system_field_operation(
    value: SystemFieldOperation,
) -> dict[str, Any]:
    if not isinstance(value, (AppendSpans, ReplaceSpans, ClearRegion)):
        raise TypeError(
            "serialize_system_field_operation requires a system operation"
        )
    return value.to_canonical_dict()


def deserialize_system_field_operation(value: Any) -> SystemFieldOperation:
    return _system_operation_from_canonical(value)


_SYSTEM_UPDATE_CANONICAL_FIELDS = {
    "schema",
    "base_field_id",
    "base_tick_id",
    "author",
    "operations",
    "evidence",
}
_SYSTEM_UPDATE_FIELDS = {
    *_SYSTEM_UPDATE_CANONICAL_FIELDS,
    "update_id",
    "canonical_hash",
}


def serialize_system_field_update(
    update: SystemFieldUpdate,
) -> dict[str, Any]:
    if not isinstance(update, SystemFieldUpdate):
        raise TypeError(
            "serialize_system_field_update requires SystemFieldUpdate"
        )
    return update.to_dict()


def deserialize_system_field_update(value: Any) -> SystemFieldUpdate:
    item = _mapping(value, "serialized system field update")
    _exact_keys(
        item,
        _SYSTEM_UPDATE_FIELDS,
        "serialized system field update",
    )
    if item["schema"] != SYSTEM_FIELD_UPDATE_SCHEMA:
        raise SerializationContractError("unknown system field update schema")
    operations = item["operations"]
    if not isinstance(operations, list):
        raise SerializationContractError(
            "system update operations must be a list"
        )
    update = SystemFieldUpdate(
        base_field_id=_string(
            item["base_field_id"],
            "system update base_field_id",
            allow_empty=False,
        ),
        base_tick_id=item["base_tick_id"],
        author=_string(
            item["author"],
            "system update author",
            allow_empty=False,
        ),
        operations=tuple(
            _system_operation_from_canonical(operation)
            for operation in operations
        ),
        evidence=_string_sequence(
            item["evidence"],
            "system update evidence",
        ),
    )
    _hash_matches(
        update.update_id,
        item["update_id"],
        "system field update ID",
    )
    _hash_matches(
        update.canonical_hash,
        item["canonical_hash"],
        "system field update canonical",
    )
    return update


def _optional_nested_delta(
    value: Any,
    label: str,
) -> FieldDelta | None:
    if value is None:
        return None
    try:
        item = _mapping(value, label)
        fields = set(item)
        if fields == _DELTA_FIELDS:
            return deserialize_field_delta(item)
        if fields == _DELTA_CANONICAL_FIELDS:
            return _field_delta_from_canonical(item)
        _exact_keys(item, _DELTA_FIELDS, label)
        raise AssertionError("unreachable")
    except SerializationContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise SerializationContractError(f"invalid {label}") from exc


_PROPOSAL_FIELDS = {
    "schema",
    "tick_seq",
    "assignment_hash",
    "proposer_core_id",
    "system_update_id",
    "base_field_id",
    "base_tick_id",
    "delta",
    "evidence",
    "proposal_id",
}


def serialize_proposal_record(record: ProposalRecord) -> dict[str, Any]:
    if not isinstance(record, ProposalRecord):
        raise TypeError("serialize_proposal_record requires ProposalRecord")
    value = record.to_dict()
    value["delta"] = (
        None if record.delta is None else serialize_field_delta(record.delta)
    )
    return value


def deserialize_proposal_record(value: Any) -> ProposalRecord:
    item = _mapping(value, "serialized proposal record")
    _exact_keys(item, _PROPOSAL_FIELDS, "serialized proposal record")
    if item["schema"] != PROPOSAL_RECORD_SCHEMA:
        raise SerializationContractError("unknown proposal record schema")
    record = ProposalRecord(
        tick_seq=item["tick_seq"],
        assignment_hash=_string(
            item["assignment_hash"],
            "proposal assignment_hash",
            allow_empty=False,
        ),
        proposer_core_id=_string(
            item["proposer_core_id"],
            "proposal proposer_core_id",
            allow_empty=False,
        ),
        system_update_id=_string(
            item["system_update_id"],
            "proposal system_update_id",
            allow_empty=False,
        ),
        base_field_id=_string(
            item["base_field_id"],
            "proposal base_field_id",
            allow_empty=False,
        ),
        base_tick_id=item["base_tick_id"],
        delta=_optional_nested_delta(
            item["delta"],
            "proposal delta",
        ),
        evidence=_string_sequence(item["evidence"], "proposal evidence"),
    )
    if record.proposal_id != item["proposal_id"]:
        raise SerializationContractError("proposal record ID mismatch")
    return record


_DECISION_FIELDS = {
    "schema",
    "tick_seq",
    "assignment_hash",
    "consolidator_core_id",
    "proposal_id",
    "base_field_id",
    "accepted",
    "output_field_id",
    "committed_delta",
    "committed_delta_id",
    "reason",
    "decision_id",
}


def serialize_consolidation_decision(
    decision: ConsolidationDecision,
) -> dict[str, Any]:
    if not isinstance(decision, ConsolidationDecision):
        raise TypeError(
            "serialize_consolidation_decision requires ConsolidationDecision"
        )
    value = decision.to_dict()
    value["committed_delta"] = (
        None
        if decision.committed_delta is None
        else serialize_field_delta(decision.committed_delta)
    )
    return value


def deserialize_consolidation_decision(
    value: Any,
) -> ConsolidationDecision:
    item = _mapping(value, "serialized consolidation decision")
    _exact_keys(
        item,
        _DECISION_FIELDS,
        "serialized consolidation decision",
    )
    if item["schema"] != CONSOLIDATION_DECISION_SCHEMA:
        raise SerializationContractError(
            "unknown consolidation decision schema"
        )
    committed_delta = _optional_nested_delta(
        item["committed_delta"],
        "committed delta",
    )
    decision = ConsolidationDecision(
        tick_seq=item["tick_seq"],
        assignment_hash=_string(
            item["assignment_hash"],
            "decision assignment_hash",
            allow_empty=False,
        ),
        consolidator_core_id=_string(
            item["consolidator_core_id"],
            "decision consolidator_core_id",
            allow_empty=False,
        ),
        proposal_id=_string(
            item["proposal_id"],
            "decision proposal_id",
            allow_empty=False,
        ),
        base_field_id=_string(
            item["base_field_id"],
            "decision base_field_id",
            allow_empty=False,
        ),
        accepted=item["accepted"],
        output_field_id=_string(
            item["output_field_id"],
            "decision output_field_id",
            allow_empty=False,
        ),
        committed_delta=committed_delta,
        reason=_string(item["reason"], "decision reason"),
    )
    if decision.committed_delta_id != item["committed_delta_id"]:
        raise SerializationContractError(
            "consolidation committed-delta ID mismatch"
        )
    if decision.decision_id != item["decision_id"]:
        raise SerializationContractError("consolidation decision ID mismatch")
    return decision


_TOOL_REQUEST_FIELDS = {
    "schema",
    "tick_seq",
    "requester_core_id",
    "base_field_id",
    "tool_name",
    "arguments",
    "parent_proposal_id",
    "request_id",
}


def serialize_tool_request(record: ToolRequestRecord) -> dict[str, Any]:
    if not isinstance(record, ToolRequestRecord):
        raise TypeError("serialize_tool_request requires ToolRequestRecord")
    return record.to_dict()


def deserialize_tool_request(value: Any) -> ToolRequestRecord:
    item = _mapping(value, "serialized tool request")
    _exact_keys(item, _TOOL_REQUEST_FIELDS, "serialized tool request")
    if item["schema"] != TOOL_REQUEST_SCHEMA:
        raise SerializationContractError("unknown tool request schema")
    arguments = _mapping(item["arguments"], "tool request arguments")
    record = ToolRequestRecord(
        tick_seq=item["tick_seq"],
        requester_core_id=_string(
            item["requester_core_id"],
            "tool requester_core_id",
            allow_empty=False,
        ),
        base_field_id=_string(
            item["base_field_id"],
            "tool base_field_id",
            allow_empty=False,
        ),
        tool_name=_string(
            item["tool_name"],
            "tool name",
            allow_empty=False,
        ),
        arguments=dict(arguments),
        parent_proposal_id=_string(
            item["parent_proposal_id"],
            "tool parent_proposal_id",
            allow_empty=False,
        ),
    )
    if record.request_id != item["request_id"]:
        raise SerializationContractError("tool request ID mismatch")
    return record


_TICK_COMMIT_FIELDS = {
    "schema",
    "generation",
    "tick_seq",
    "assignment_hash",
    "proposal_id",
    "decision_id",
    "input_field_id",
    "system_update_id",
    "working_field_id",
    "field_transaction_audit_id",
    "output_field_id",
    "updated_core_state_manifest_ids",
    "projection_id",
    "tool_request_ids",
    "commit_id",
}


def serialize_tick_commit(record: TickCommitRecord) -> dict[str, Any]:
    if not isinstance(record, TickCommitRecord):
        raise TypeError("serialize_tick_commit requires TickCommitRecord")
    return record.to_dict()


def deserialize_tick_commit(value: Any) -> TickCommitRecord:
    item = _mapping(value, "serialized tick commit")
    _exact_keys(item, _TICK_COMMIT_FIELDS, "serialized tick commit")
    if item["schema"] != TICK_COMMIT_SCHEMA:
        raise SerializationContractError("unknown tick commit schema")
    record = TickCommitRecord(
        generation=item["generation"],
        tick_seq=item["tick_seq"],
        assignment_hash=_string(
            item["assignment_hash"],
            "tick commit assignment_hash",
            allow_empty=False,
        ),
        proposal_id=_string(
            item["proposal_id"],
            "tick commit proposal_id",
            allow_empty=False,
        ),
        decision_id=_string(
            item["decision_id"],
            "tick commit decision_id",
            allow_empty=False,
        ),
        input_field_id=_string(
            item["input_field_id"],
            "tick commit input_field_id",
            allow_empty=False,
        ),
        system_update_id=_string(
            item["system_update_id"],
            "tick commit system_update_id",
            allow_empty=False,
        ),
        working_field_id=_string(
            item["working_field_id"],
            "tick commit working_field_id",
            allow_empty=False,
        ),
        field_transaction_audit_id=_string(
            item["field_transaction_audit_id"],
            "tick commit field_transaction_audit_id",
            allow_empty=False,
        ),
        output_field_id=_string(
            item["output_field_id"],
            "tick commit output_field_id",
            allow_empty=False,
        ),
        updated_core_state_manifest_ids=_string_sequence(
            item["updated_core_state_manifest_ids"],
            "updated core-state manifest IDs",
        ),
        projection_id=_string(
            item["projection_id"],
            "tick commit projection_id",
            allow_empty=False,
        ),
        tool_request_ids=_string_sequence(
            item["tool_request_ids"],
            "tick commit tool request IDs",
        ),
    )
    if record.commit_id != item["commit_id"]:
        raise SerializationContractError("tick commit ID mismatch")
    return record


_CURSOR_FIELDS = {
    "schema",
    "page_index",
    "context_offsets",
    "user_offset",
    "canonical_hash",
}


def serialize_field_view_cursor(cursor: FieldViewCursor) -> dict[str, Any]:
    if not isinstance(cursor, FieldViewCursor):
        raise TypeError("serialize_field_view_cursor requires FieldViewCursor")
    canonical = cursor.to_canonical_dict()
    return {
        "schema": FIELD_VIEW_CURSOR_SCHEMA,
        **canonical,
        "canonical_hash": canonical_sha256(canonical),
    }


def deserialize_field_view_cursor(value: Any) -> FieldViewCursor:
    item = _mapping(value, "serialized field view cursor")
    _exact_keys(item, _CURSOR_FIELDS, "serialized field view cursor")
    if item["schema"] != FIELD_VIEW_CURSOR_SCHEMA:
        raise SerializationContractError("unknown field view cursor schema")
    offsets = _mapping(item["context_offsets"], "context_offsets")
    normalized_offsets: list[tuple[str, int]] = []
    for region, offset in offsets.items():
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise SerializationContractError(
                "cursor offsets must be integers"
            )
        normalized_offsets.append((region, offset))
    cursor = FieldViewCursor(
        page_index=item["page_index"],
        context_offsets=tuple(normalized_offsets),
        user_offset=item["user_offset"],
    )
    _hash_matches(
        canonical_sha256(cursor.to_canonical_dict()),
        item["canonical_hash"],
        "field view cursor",
    )
    return cursor


def deserialize_core_identity(value: Any) -> CoreIdentity:
    item = _mapping(value, "core identity")
    expected = {
        "schema",
        "core_id",
        "display_name",
        "base_checkpoint_path",
        "base_checkpoint_sha256",
        "model_id",
        "core_state_sha256",
        "lineage",
        "parent_core_id",
        "soul_id",
        "adapter_namespace",
        "enabled",
        "role_capabilities",
        "identity_hash",
    }
    _exact_keys(item, expected, "core identity")
    if item["schema"] != "axon-core-identity-v1":
        raise SerializationContractError("unknown core identity schema")
    identity = CoreIdentity(
        core_id=item["core_id"],
        display_name=item["display_name"],
        base_checkpoint_path=item["base_checkpoint_path"],
        base_checkpoint_sha256=item["base_checkpoint_sha256"],
        model_id=item["model_id"],
        core_state_sha256=item["core_state_sha256"],
        lineage=_string_sequence(item["lineage"], "lineage"),
        parent_core_id=item["parent_core_id"],
        soul_id=item["soul_id"],
        adapter_namespace=item["adapter_namespace"],
        enabled=item["enabled"],
        role_capabilities=_string_sequence(
            item["role_capabilities"],
            "role_capabilities",
        ),
    )
    _hash_matches(identity.identity_hash, item["identity_hash"], "identity")
    return identity


def deserialize_role_assignment(value: Any) -> RoleAssignment:
    item = _mapping(value, "role assignment")
    expected = {
        "schema",
        "tick_seq",
        "proposer_core_id",
        "consolidator_core_id",
        "sleeper_core_id",
        "standby_core_ids",
        "assignment_hash",
    }
    _exact_keys(item, expected, "role assignment")
    if item["schema"] != "axon-role-assignment-v1":
        raise SerializationContractError("unknown role assignment schema")
    assignment = RoleAssignment(
        tick_seq=item["tick_seq"],
        proposer_core_id=item["proposer_core_id"],
        consolidator_core_id=item["consolidator_core_id"],
        sleeper_core_id=item["sleeper_core_id"],
        standby_core_ids=_string_sequence(
            item["standby_core_ids"],
            "standby_core_ids",
        ),
    )
    _hash_matches(
        assignment.assignment_hash,
        item["assignment_hash"],
        "role assignment",
    )
    return assignment


def deserialize_core_state_manifest(value: Any) -> CoreStateManifest:
    item = _mapping(value, "core state manifest")
    expected = {
        "schema",
        "core_id",
        "soul_id",
        "model_id",
        "core_state_sha256",
        "generation",
        "tick_seq",
        "committed_field_id",
        "soul_state_sha256",
        "cursor_state_sha256",
        "compressor_state_sha256",
        "adapter_set_sha256",
        "rng_state_sha256",
        "parent_manifest_id",
        "manifest_id",
    }
    _exact_keys(item, expected, "core state manifest")
    if item["schema"] != "axon-core-state-manifest-v2":
        raise SerializationContractError("unknown core state manifest schema")
    manifest = CoreStateManifest(
        core_id=item["core_id"],
        soul_id=item["soul_id"],
        model_id=item["model_id"],
        core_state_sha256=item["core_state_sha256"],
        generation=item["generation"],
        tick_seq=item["tick_seq"],
        committed_field_id=item["committed_field_id"],
        soul_state_sha256=item["soul_state_sha256"],
        cursor_state_sha256=item["cursor_state_sha256"],
        compressor_state_sha256=item["compressor_state_sha256"],
        adapter_set_sha256=item["adapter_set_sha256"],
        rng_state_sha256=item["rng_state_sha256"],
        parent_manifest_id=item["parent_manifest_id"],
    )
    if manifest.manifest_id != item["manifest_id"]:
        raise SerializationContractError("core state manifest ID mismatch")
    return manifest


def deserialize_projection_manifest(value: Any) -> ProjectionManifest:
    item = _mapping(value, "projection manifest")
    expected = {
        "schema",
        "generation",
        "tick_seq",
        "field_id",
        "field_hash",
        "core_state_manifest_ids",
        "role_index",
        "role_assignment_hash",
        "parent_projection_id",
        "projection_id",
    }
    _exact_keys(item, expected, "projection manifest")
    if item["schema"] != "axon-projection-manifest-v1":
        raise SerializationContractError("unknown projection manifest schema")
    states = _mapping(
        item["core_state_manifest_ids"],
        "core_state_manifest_ids",
    )
    manifest = ProjectionManifest(
        generation=item["generation"],
        tick_seq=item["tick_seq"],
        field_id=item["field_id"],
        field_hash=item["field_hash"],
        core_state_manifest_ids=tuple(states.items()),
        role_index=item["role_index"],
        role_assignment_hash=item["role_assignment_hash"],
        parent_projection_id=item["parent_projection_id"],
    )
    if manifest.projection_id != item["projection_id"]:
        raise SerializationContractError("projection manifest ID mismatch")
    return manifest


__all__ = [
    "FIELD_SPAN_SCHEMA",
    "REGION_STATE_SCHEMA",
    "FIELD_DELTA_SCHEMA",
    "FIELD_VIEW_CURSOR_SCHEMA",
    "SYSTEM_FIELD_UPDATE_SCHEMA",
    "PROPOSAL_RECORD_SCHEMA",
    "CONSOLIDATION_DECISION_SCHEMA",
    "TOOL_REQUEST_SCHEMA",
    "TICK_COMMIT_SCHEMA",
    "canonical_json_text",
    "parse_json_object",
    "serialize_field_span",
    "deserialize_field_span",
    "serialize_region_state",
    "deserialize_region_state",
    "serialize_shared_field",
    "deserialize_shared_field",
    "serialize_field_delta",
    "deserialize_field_delta",
    "serialize_system_field_operation",
    "deserialize_system_field_operation",
    "serialize_system_field_update",
    "deserialize_system_field_update",
    "serialize_proposal_record",
    "deserialize_proposal_record",
    "serialize_consolidation_decision",
    "deserialize_consolidation_decision",
    "serialize_tool_request",
    "deserialize_tool_request",
    "serialize_tick_commit",
    "deserialize_tick_commit",
    "serialize_field_view_cursor",
    "deserialize_field_view_cursor",
    "deserialize_core_identity",
    "deserialize_role_assignment",
    "deserialize_core_state_manifest",
    "deserialize_projection_manifest",
]

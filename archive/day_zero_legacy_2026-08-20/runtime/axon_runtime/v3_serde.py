"""Strict canonical-JSON serde for the isolated v3 protocol slice."""
from __future__ import annotations

import json
import math
from typing import Any

from runtime.field import canonical_json_bytes

from .v3_contracts import (
    MAX_COLLECTION_ITEMS,
    MAX_JSON_DEPTH,
    MAX_JSON_NODES,
    MAX_PROTOCOL_INT,
    PROTOCOL_VERSION,
    ArtifactRejectionRecord,
    ConsolidationRecord,
    CorePassRecord,
    ProposalBoardManifest,
    ReadCycleManifest,
    SoulTransitionDispositionRecord,
    SoulTransitionRecord,
    TickCommitRecordV3,
    TickPhasePlan,
    V3SerializationError,
)


TICK_PHASE_PLAN_SCHEMA = "axon-tick-phase-plan-v3"
READ_CYCLE_SCHEMA = "axon-read-cycle-manifest-v3"
SOUL_TRANSITION_SCHEMA = "axon-soul-transition-v3"
SOUL_DISPOSITION_SCHEMA = "axon-soul-transition-disposition-v3"
CORE_PASS_SCHEMA = "axon-core-pass-v3"
PROPOSAL_BOARD_SCHEMA = "axon-proposal-board-v3"
CONSOLIDATION_SCHEMA = "axon-consolidation-v3"
TICK_COMMIT_V3_SCHEMA = "axon-tick-commit-v3"
ARTIFACT_REJECTION_SCHEMA = "axon-artifact-rejection-v3"


def canonical_json_text(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise V3SerializationError("value is not canonical JSON-safe") from exc


def parse_json_object(value: str | bytes) -> dict[str, Any]:
    """Parse exactly one canonical UTF-8 JSON object."""

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, nested in pairs:
            if key in result:
                raise V3SerializationError(
                    f"serialized JSON contains duplicate key {key!r}"
                )
            result[key] = nested
        return result

    def reject_constant(token: str) -> None:
        raise V3SerializationError(
            f"serialized JSON contains non-finite constant {token}"
        )

    def parse_integer(token: str) -> int:
        digits = token[1:] if token.startswith("-") else token
        if len(digits) > 19:
            raise V3SerializationError(
                "serialized JSON integer exceeds the signed 64-bit limit"
            )
        parsed = int(token)
        if not (-MAX_PROTOCOL_INT - 1 <= parsed <= MAX_PROTOCOL_INT):
            raise V3SerializationError(
                "serialized JSON integer exceeds the signed 64-bit limit"
            )
        return parsed

    if type(value) not in (str, bytes):
        raise V3SerializationError("serialized JSON must be text or bytes")
    try:
        if type(value) is bytes:
            encoded = value
            text = encoded.decode("utf-8")
        else:
            text = value
            encoded = text.encode("utf-8")
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_int=parse_integer,
        )
        result = _mapping(decoded, "serialized JSON")
        if encoded != canonical_json_bytes(result):
            raise V3SerializationError(
                "serialized JSON is not in canonical UTF-8 form"
            )
        return result
    except V3SerializationError:
        raise
    except (
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise V3SerializationError("invalid serialized JSON object") from exc


def _validate_json_tree(value: Any, label: str, path: str = "$") -> None:
    stack: list[tuple[Any, str, int, bool]] = [(value, path, 0, False)]
    active_container_ids: set[int] = set()
    node_count = 0
    while stack:
        current, current_path, depth, exiting = stack.pop()
        if exiting:
            active_container_ids.remove(id(current))
            continue
        node_count += 1
        if node_count > MAX_JSON_NODES:
            raise V3SerializationError(f"{label} exceeds the JSON node limit")
        value_type = type(current)
        if value_type in (dict, list):
            if depth >= MAX_JSON_DEPTH:
                raise V3SerializationError(
                    f"{label} exceeds the JSON depth limit"
                )
            if len(current) > MAX_COLLECTION_ITEMS:
                raise V3SerializationError(
                    f"{label} exceeds the JSON collection-item limit"
                )
            container_id = id(current)
            if container_id in active_container_ids:
                raise V3SerializationError(
                    f"{label} contains a JSON cycle"
                )
            active_container_ids.add(container_id)
            stack.append((current, current_path, depth, True))
            if value_type is dict:
                children: list[tuple[Any, str]] = []
                for key, nested in current.items():
                    if type(key) is not str:
                        raise V3SerializationError(
                            f"{label} key at {current_path} "
                            "must be an exact string"
                        )
                    try:
                        key.encode("utf-8")
                    except UnicodeEncodeError as exc:
                        raise V3SerializationError(
                            f"{label} key at {current_path} "
                            "must be valid UTF-8"
                        ) from exc
                    children.append((nested, f"{current_path}.{key}"))
            else:
                children = [
                    (nested, f"{current_path}[{index}]")
                    for index, nested in enumerate(current)
                ]
            for nested, nested_path in reversed(children):
                stack.append((nested, nested_path, depth + 1, False))
            continue
        if value_type is str:
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise V3SerializationError(
                    f"{label} value at {current_path} must be valid UTF-8"
                ) from exc
        elif current is None or value_type is bool:
            continue
        elif value_type is int:
            if not (
                -MAX_PROTOCOL_INT - 1
                <= current
                <= MAX_PROTOCOL_INT
            ):
                raise V3SerializationError(
                    f"{label} integer at {current_path} is out of range"
                )
        elif value_type is float:
            if not math.isfinite(current):
                raise V3SerializationError(
                    f"{label} float at {current_path} must be finite"
                )
        else:
            raise V3SerializationError(
                f"{label} value at {current_path} "
                "is not an exact canonical JSON type"
            )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise V3SerializationError(f"{label} must be a JSON object")
    _validate_json_tree(value, label)
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise V3SerializationError(
            f"{label} fields mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise V3SerializationError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label, allow_empty=False)


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise V3SerializationError(f"{label} must be a boolean")
    return value


def _int(value: Any, label: str) -> int:
    if (
        type(value) is not int
        or not (-MAX_PROTOCOL_INT - 1 <= value <= MAX_PROTOCOL_INT)
    ):
        raise V3SerializationError(
            f"{label} must be a signed 64-bit integer"
        )
    return value


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise V3SerializationError(f"{label} must be a list")
    return tuple(_string(item, f"{label} item") for item in value)


def _int_sequence(value: Any, label: str) -> tuple[int, ...]:
    if type(value) is not list:
        raise V3SerializationError(f"{label} must be a list")
    return tuple(_int(item, f"{label} item") for item in value)


def _hash_matches(actual: str, supplied: Any, label: str) -> None:
    if type(supplied) is not str or supplied != actual:
        raise V3SerializationError(f"{label} ID/hash mismatch")


def _canonical_wire_matches(
    supplied: dict[str, Any],
    canonical: dict[str, Any],
    label: str,
) -> None:
    try:
        supplied_bytes = canonical_json_bytes(supplied)
        canonical_bytes = canonical_json_bytes(canonical)
    except (TypeError, ValueError) as exc:
        raise V3SerializationError(
            f"{label} is not canonical JSON-safe"
        ) from exc
    if supplied_bytes != canonical_bytes:
        raise V3SerializationError(
            f"{label} is not in canonical serialized form"
        )


_TICK_PHASE_PLAN_FIELDS = {
    "schema",
    "tick_seq",
    "protocol_version",
    "protocol_name",
    "input_field_id",
    "system_update_id",
    "working_field_id",
    "no_core_delta_output_field_id",
    "projection_id",
    "online_core_ids",
    "input_core_state_leaf_ids",
    "input_soul_sha256_by_core",
    "offline_core_id",
    "consolidator_core_id",
    "model_binding_epoch_id",
    "plan_id",
}


def serialize_tick_phase_plan(plan: TickPhasePlan) -> dict[str, Any]:
    if type(plan) is not TickPhasePlan:
        raise TypeError("serialize_tick_phase_plan requires TickPhasePlan")
    return plan.to_dict()


def deserialize_tick_phase_plan(value: Any) -> TickPhasePlan:
    item = _mapping(value, "serialized tick phase plan")
    _exact_keys(item, _TICK_PHASE_PLAN_FIELDS, "serialized tick phase plan")
    if item["schema"] != TICK_PHASE_PLAN_SCHEMA:
        raise V3SerializationError("unknown tick phase plan schema")
    if item["protocol_name"] != "axon-runtime-protocol-v3":
        raise V3SerializationError("unknown protocol name")
    input_leaf_mapping = _mapping(
        item["input_core_state_leaf_ids"],
        "input_core_state_leaf_ids",
    )
    input_soul_mapping = _mapping(
        item["input_soul_sha256_by_core"],
        "input_soul_sha256_by_core",
    )
    plan = TickPhasePlan(
        tick_seq=_int(item["tick_seq"], "tick_seq"),
        protocol_version=_int(item["protocol_version"], "protocol_version"),
        input_field_id=_string(item["input_field_id"], "input_field_id"),
        system_update_id=_string(item["system_update_id"], "system_update_id"),
        working_field_id=_string(item["working_field_id"], "working_field_id"),
        no_core_delta_output_field_id=_string(
            item["no_core_delta_output_field_id"],
            "no_core_delta_output_field_id",
        ),
        projection_id=_string(item["projection_id"], "projection_id"),
        online_core_ids=_string_sequence(
            item["online_core_ids"],
            "online_core_ids",
        ),
        input_core_state_leaf_ids=tuple(input_leaf_mapping.items()),
        input_soul_sha256_by_core=tuple(input_soul_mapping.items()),
        offline_core_id=_string(item["offline_core_id"], "offline_core_id"),
        consolidator_core_id=_string(
            item["consolidator_core_id"],
            "consolidator_core_id",
        ),
        model_binding_epoch_id=_string(
            item["model_binding_epoch_id"],
            "model_binding_epoch_id",
        ),
    )
    _hash_matches(plan.plan_id, item["plan_id"], "tick phase plan")
    _canonical_wire_matches(item, plan.to_dict(), "serialized tick phase plan")
    return plan


_READ_CYCLE_FIELDS = {
    "schema",
    "core_id",
    "tick_seq",
    "phase",
    "working_field_id",
    "projection_id",
    "board_id",
    "page_view_hashes",
    "page_character_counts",
    "cursor_chain",
    "expected_character_count",
    "coverage_complete",
    "selected_span_fingerprint",
    "cycle_id",
}


def serialize_read_cycle_manifest(manifest: ReadCycleManifest) -> dict[str, Any]:
    if type(manifest) is not ReadCycleManifest:
        raise TypeError("serialize_read_cycle_manifest requires ReadCycleManifest")
    return manifest.to_dict()


def deserialize_read_cycle_manifest(value: Any) -> ReadCycleManifest:
    item = _mapping(value, "serialized read cycle manifest")
    _exact_keys(item, _READ_CYCLE_FIELDS, "serialized read cycle manifest")
    if item["schema"] != READ_CYCLE_SCHEMA:
        raise V3SerializationError("unknown read cycle schema")
    manifest = ReadCycleManifest(
        core_id=_string(item["core_id"], "core_id"),
        tick_seq=_int(item["tick_seq"], "tick_seq"),
        phase=_string(item["phase"], "phase"),
        working_field_id=_string(
            item["working_field_id"],
            "working_field_id",
        ),
        projection_id=_string(item["projection_id"], "projection_id"),
        board_id=_optional_string(item["board_id"], "board_id"),
        page_view_hashes=_string_sequence(
            item["page_view_hashes"],
            "page_view_hashes",
        ),
        page_character_counts=_int_sequence(
            item["page_character_counts"],
            "page_character_counts",
        ),
        cursor_chain=_string_sequence(item["cursor_chain"], "cursor_chain"),
        expected_character_count=_int(
            item["expected_character_count"],
            "expected_character_count",
        ),
        coverage_complete=_boolean(
            item["coverage_complete"],
            "coverage_complete",
        ),
        selected_span_fingerprint=_string(
            item["selected_span_fingerprint"],
            "selected_span_fingerprint",
        ),
    )
    _hash_matches(manifest.cycle_id, item["cycle_id"], "read cycle")
    _canonical_wire_matches(
        item,
        manifest.to_dict(),
        "serialized read cycle manifest",
    )
    return manifest


_SOUL_TRANSITION_FIELDS = {
    "schema",
    "core_id",
    "tick_seq",
    "phase",
    "substep",
    "input_soul_sha256",
    "output_soul_sha256",
    "working_field_id",
    "projection_id",
    "read_cycle_id",
    "board_id",
    "delta_id",
    "candidate_private_state_id",
    "parent_transition_id",
    "cursor_state_sha256",
    "rng_state_sha256",
    "binding_manifest_id",
    "transition_id",
}


def serialize_soul_transition(transition: SoulTransitionRecord) -> dict[str, Any]:
    if type(transition) is not SoulTransitionRecord:
        raise TypeError("serialize_soul_transition requires SoulTransitionRecord")
    return transition.to_dict()


def deserialize_soul_transition(value: Any) -> SoulTransitionRecord:
    item = _mapping(value, "serialized soul transition")
    _exact_keys(item, _SOUL_TRANSITION_FIELDS, "serialized soul transition")
    if item["schema"] != SOUL_TRANSITION_SCHEMA:
        raise V3SerializationError("unknown soul transition schema")
    transition = SoulTransitionRecord(
        core_id=_string(item["core_id"], "core_id"),
        tick_seq=_int(item["tick_seq"], "tick_seq"),
        phase=_string(item["phase"], "phase"),
        substep=_int(item["substep"], "substep"),
        input_soul_sha256=_string(item["input_soul_sha256"], "input_soul_sha256"),
        output_soul_sha256=_string(
            item["output_soul_sha256"],
            "output_soul_sha256",
        ),
        working_field_id=_string(item["working_field_id"], "working_field_id"),
        projection_id=_string(item["projection_id"], "projection_id"),
        read_cycle_id=_string(item["read_cycle_id"], "read_cycle_id"),
        board_id=_optional_string(item["board_id"], "board_id"),
        delta_id=_string(item["delta_id"], "delta_id"),
        candidate_private_state_id=_string(
            item["candidate_private_state_id"],
            "candidate_private_state_id",
        ),
        parent_transition_id=_optional_string(
            item["parent_transition_id"],
            "parent_transition_id",
        ),
        cursor_state_sha256=_string(
            item["cursor_state_sha256"],
            "cursor_state_sha256",
        ),
        rng_state_sha256=_optional_string(
            item["rng_state_sha256"],
            "rng_state_sha256",
        ),
        binding_manifest_id=_string(
            item["binding_manifest_id"],
            "binding_manifest_id",
        ),
    )
    _hash_matches(transition.transition_id, item["transition_id"], "soul transition")
    _canonical_wire_matches(
        item,
        transition.to_dict(),
        "serialized soul transition",
    )
    return transition


_SOUL_DISPOSITION_FIELDS = {
    "schema",
    "transition_id",
    "disposition",
    "reason",
    "superseded_by_transition_id",
    "disposition_id",
}


def serialize_soul_transition_disposition(
    disposition: SoulTransitionDispositionRecord,
) -> dict[str, Any]:
    if type(disposition) is not SoulTransitionDispositionRecord:
        raise TypeError(
            "serialize_soul_transition_disposition requires "
            "SoulTransitionDispositionRecord"
        )
    return disposition.to_dict()


def deserialize_soul_transition_disposition(
    value: Any,
) -> SoulTransitionDispositionRecord:
    item = _mapping(value, "serialized soul disposition")
    _exact_keys(item, _SOUL_DISPOSITION_FIELDS, "serialized soul disposition")
    if item["schema"] != SOUL_DISPOSITION_SCHEMA:
        raise V3SerializationError("unknown soul disposition schema")
    disposition = SoulTransitionDispositionRecord(
        transition_id=_string(item["transition_id"], "transition_id"),
        disposition=_string(item["disposition"], "disposition"),
        reason=_string(item["reason"], "reason"),
        superseded_by_transition_id=_optional_string(
            item["superseded_by_transition_id"],
            "superseded_by_transition_id",
        ),
    )
    _hash_matches(
        disposition.disposition_id,
        item["disposition_id"],
        "soul disposition",
    )
    _canonical_wire_matches(
        item,
        disposition.to_dict(),
        "serialized soul disposition",
    )
    return disposition


_CORE_PASS_FIELDS = {
    "schema",
    "core_id",
    "tick_seq",
    "phase",
    "working_field_id",
    "board_id",
    "read_cycle_id",
    "delta_id",
    "soul_transition_id",
    "candidate_private_state_id",
    "pass_id",
}


def serialize_core_pass(pass_record: CorePassRecord) -> dict[str, Any]:
    if type(pass_record) is not CorePassRecord:
        raise TypeError("serialize_core_pass requires CorePassRecord")
    return pass_record.to_dict()


def deserialize_core_pass(value: Any) -> CorePassRecord:
    item = _mapping(value, "serialized core pass")
    _exact_keys(item, _CORE_PASS_FIELDS, "serialized core pass")
    if item["schema"] != CORE_PASS_SCHEMA:
        raise V3SerializationError("unknown core pass schema")
    pass_record = CorePassRecord(
        core_id=_string(item["core_id"], "core_id"),
        tick_seq=_int(item["tick_seq"], "tick_seq"),
        phase=_string(item["phase"], "phase"),
        working_field_id=_string(
            item["working_field_id"],
            "working_field_id",
        ),
        board_id=_optional_string(item["board_id"], "board_id"),
        read_cycle_id=_string(item["read_cycle_id"], "read_cycle_id"),
        delta_id=_string(item["delta_id"], "delta_id"),
        soul_transition_id=_string(
            item["soul_transition_id"],
            "soul_transition_id",
        ),
        candidate_private_state_id=_string(
            item["candidate_private_state_id"],
            "candidate_private_state_id",
        ),
    )
    _hash_matches(pass_record.pass_id, item["pass_id"], "core pass")
    _canonical_wire_matches(item, pass_record.to_dict(), "serialized core pass")
    return pass_record


_PROPOSAL_BOARD_FIELDS = {
    "schema",
    "working_field_id",
    "phase",
    "pass_ids_by_author",
    "required_authors",
    "explicit_conflicts",
    "board_id",
}


def serialize_proposal_board(board: ProposalBoardManifest) -> dict[str, Any]:
    if type(board) is not ProposalBoardManifest:
        raise TypeError("serialize_proposal_board requires ProposalBoardManifest")
    return board.to_dict()


def deserialize_proposal_board(value: Any) -> ProposalBoardManifest:
    item = _mapping(value, "serialized proposal board")
    _exact_keys(item, _PROPOSAL_BOARD_FIELDS, "serialized proposal board")
    if item["schema"] != PROPOSAL_BOARD_SCHEMA:
        raise V3SerializationError("unknown proposal board schema")
    mapping = _mapping(item["pass_ids_by_author"], "pass_ids_by_author")
    board = ProposalBoardManifest(
        working_field_id=_string(
            item["working_field_id"],
            "working_field_id",
        ),
        phase=_string(item["phase"], "phase"),
        pass_ids_by_author=tuple(mapping.items()),
        required_authors=_string_sequence(
            item["required_authors"],
            "required_authors",
        ),
        explicit_conflicts=_string_sequence(
            item["explicit_conflicts"],
            "explicit_conflicts",
        ),
    )
    _hash_matches(board.board_id, item["board_id"], "proposal board")
    _canonical_wire_matches(
        item,
        board.to_dict(),
        "serialized proposal board",
    )
    return board


_CONSOLIDATION_FIELDS = {
    "schema",
    "core_id",
    "tick_seq",
    "working_field_id",
    "refinement_board_id",
    "final_delta_id",
    "final_soul_transition_id",
    "accepted",
    "reason",
    "invocation_request_ids",
    "consolidation_id",
}


def serialize_consolidation(consolidation: ConsolidationRecord) -> dict[str, Any]:
    if type(consolidation) is not ConsolidationRecord:
        raise TypeError("serialize_consolidation requires ConsolidationRecord")
    return consolidation.to_dict()


def deserialize_consolidation(value: Any) -> ConsolidationRecord:
    item = _mapping(value, "serialized consolidation")
    _exact_keys(item, _CONSOLIDATION_FIELDS, "serialized consolidation")
    if item["schema"] != CONSOLIDATION_SCHEMA:
        raise V3SerializationError("unknown consolidation schema")
    consolidation = ConsolidationRecord(
        core_id=_string(item["core_id"], "core_id"),
        tick_seq=_int(item["tick_seq"], "tick_seq"),
        working_field_id=_string(item["working_field_id"], "working_field_id"),
        refinement_board_id=_string(
            item["refinement_board_id"],
            "refinement_board_id",
        ),
        final_delta_id=_string(item["final_delta_id"], "final_delta_id"),
        final_soul_transition_id=_string(
            item["final_soul_transition_id"],
            "final_soul_transition_id",
        ),
        accepted=_boolean(item["accepted"], "accepted"),
        reason=_string(item["reason"], "reason"),
        invocation_request_ids=_string_sequence(
            item["invocation_request_ids"],
            "invocation_request_ids",
        ),
    )
    _hash_matches(
        consolidation.consolidation_id,
        item["consolidation_id"],
        "consolidation",
    )
    _canonical_wire_matches(
        item,
        consolidation.to_dict(),
        "serialized consolidation",
    )
    return consolidation


_TICK_COMMIT_V3_FIELDS = {
    "schema",
    "tick_seq",
    "plan_id",
    "initial_board_id",
    "refine_board_id",
    "initial_pass_ids",
    "refine_pass_ids",
    "consolidation_id",
    "disposition_ids_by_transition",
    "final_core_state_leaf_ids",
    "field_transaction_audit_id",
    "output_field_id",
    "invocation_request_ids",
    "commit_id",
}


def serialize_tick_commit_v3(commit: TickCommitRecordV3) -> dict[str, Any]:
    if type(commit) is not TickCommitRecordV3:
        raise TypeError("serialize_tick_commit_v3 requires TickCommitRecordV3")
    return commit.to_dict()


def deserialize_tick_commit_v3(value: Any) -> TickCommitRecordV3:
    item = _mapping(value, "serialized tick commit v3")
    _exact_keys(item, _TICK_COMMIT_V3_FIELDS, "serialized tick commit v3")
    if item["schema"] != TICK_COMMIT_V3_SCHEMA:
        raise V3SerializationError("unknown tick commit v3 schema")
    initial_mapping = _mapping(item["initial_pass_ids"], "initial_pass_ids")
    refine_mapping = _mapping(item["refine_pass_ids"], "refine_pass_ids")
    leaf_mapping = _mapping(
        item["final_core_state_leaf_ids"],
        "final_core_state_leaf_ids",
    )
    disposition_mapping = _mapping(
        item["disposition_ids_by_transition"],
        "disposition_ids_by_transition",
    )
    commit = TickCommitRecordV3(
        tick_seq=_int(item["tick_seq"], "tick_seq"),
        plan_id=_string(item["plan_id"], "plan_id"),
        initial_board_id=_string(item["initial_board_id"], "initial_board_id"),
        refine_board_id=_string(item["refine_board_id"], "refine_board_id"),
        initial_pass_ids=tuple(initial_mapping.items()),
        refine_pass_ids=tuple(refine_mapping.items()),
        consolidation_id=_string(item["consolidation_id"], "consolidation_id"),
        disposition_ids_by_transition=tuple(disposition_mapping.items()),
        final_core_state_leaf_ids=tuple(leaf_mapping.items()),
        field_transaction_audit_id=_string(
            item["field_transaction_audit_id"],
            "field_transaction_audit_id",
        ),
        output_field_id=_string(item["output_field_id"], "output_field_id"),
        invocation_request_ids=_string_sequence(
            item["invocation_request_ids"],
            "invocation_request_ids",
        ),
    )
    _hash_matches(commit.commit_id, item["commit_id"], "tick commit v3")
    _canonical_wire_matches(
        item,
        commit.to_dict(),
        "serialized tick commit v3",
    )
    return commit


_ARTIFACT_REJECTION_FIELDS = {
    "schema",
    "artifact_kind",
    "reason",
    "source_context",
    "raw_fingerprint_sha256",
    "rejection_id",
}


def serialize_artifact_rejection(rejection: ArtifactRejectionRecord) -> dict[str, Any]:
    if type(rejection) is not ArtifactRejectionRecord:
        raise TypeError("serialize_artifact_rejection requires ArtifactRejectionRecord")
    return rejection.to_dict()


def deserialize_artifact_rejection(value: Any) -> ArtifactRejectionRecord:
    item = _mapping(value, "serialized artifact rejection")
    _exact_keys(item, _ARTIFACT_REJECTION_FIELDS, "serialized artifact rejection")
    if item["schema"] != ARTIFACT_REJECTION_SCHEMA:
        raise V3SerializationError("unknown artifact rejection schema")
    context = _mapping(item["source_context"], "source_context")
    rejection = ArtifactRejectionRecord(
        artifact_kind=_string(item["artifact_kind"], "artifact_kind"),
        reason=_string(item["reason"], "reason"),
        source_context=dict(context),
        raw_fingerprint_sha256=_string(
            item["raw_fingerprint_sha256"],
            "raw_fingerprint_sha256",
        ),
    )
    _hash_matches(rejection.rejection_id, item["rejection_id"], "artifact rejection")
    _canonical_wire_matches(
        item,
        rejection.to_dict(),
        "serialized artifact rejection",
    )
    return rejection


def assert_round_trip_identity(record: Any) -> None:
    """Helper for tests: serialize then deserialize must recover the record."""
    serializers = {
        TickPhasePlan: (serialize_tick_phase_plan, deserialize_tick_phase_plan),
        ReadCycleManifest: (
            serialize_read_cycle_manifest,
            deserialize_read_cycle_manifest,
        ),
        SoulTransitionRecord: (serialize_soul_transition, deserialize_soul_transition),
        SoulTransitionDispositionRecord: (
            serialize_soul_transition_disposition,
            deserialize_soul_transition_disposition,
        ),
        CorePassRecord: (serialize_core_pass, deserialize_core_pass),
        ProposalBoardManifest: (serialize_proposal_board, deserialize_proposal_board),
        ConsolidationRecord: (serialize_consolidation, deserialize_consolidation),
        TickCommitRecordV3: (serialize_tick_commit_v3, deserialize_tick_commit_v3),
        ArtifactRejectionRecord: (
            serialize_artifact_rejection,
            deserialize_artifact_rejection,
        ),
    }
    record_type = type(record)
    if record_type not in serializers:
        raise TypeError(f"no round-trip serializer for {record_type.__name__}")
    serialize, deserialize = serializers[record_type]
    serialized = serialize(record)
    restored = deserialize(serialized)
    if restored != record:
        raise V3SerializationError(
            f"round-trip mismatch for {record_type.__name__}"
        )
    reserialized = serialize(restored)
    if canonical_json_bytes(reserialized) != canonical_json_bytes(serialized):
        raise V3SerializationError(
            f"round-trip canonical wire mismatch for {record_type.__name__}"
        )


__all__ = [
    "TICK_PHASE_PLAN_SCHEMA",
    "READ_CYCLE_SCHEMA",
    "SOUL_TRANSITION_SCHEMA",
    "SOUL_DISPOSITION_SCHEMA",
    "CORE_PASS_SCHEMA",
    "PROPOSAL_BOARD_SCHEMA",
    "CONSOLIDATION_SCHEMA",
    "TICK_COMMIT_V3_SCHEMA",
    "ARTIFACT_REJECTION_SCHEMA",
    "canonical_json_text",
    "parse_json_object",
    "serialize_tick_phase_plan",
    "deserialize_tick_phase_plan",
    "serialize_read_cycle_manifest",
    "deserialize_read_cycle_manifest",
    "serialize_soul_transition",
    "deserialize_soul_transition",
    "serialize_soul_transition_disposition",
    "deserialize_soul_transition_disposition",
    "serialize_core_pass",
    "deserialize_core_pass",
    "serialize_proposal_board",
    "deserialize_proposal_board",
    "serialize_consolidation",
    "deserialize_consolidation",
    "serialize_tick_commit_v3",
    "deserialize_tick_commit_v3",
    "serialize_artifact_rejection",
    "deserialize_artifact_rejection",
    "assert_round_trip_identity",
]

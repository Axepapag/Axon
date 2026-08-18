"""Read-only, exact-grounded ``D:\00`` source adapter for the real field.

This module deliberately does not write curriculum artifacts.  It joins the
canonical semantic store to exact objects in the canonical episodic store and
returns dictionaries that can be passed directly to
``training.build_multitick_curriculum.SourceRecord``.

Only ``axon_semantic_memory.db`` and ``axon_episodic_memory.db`` are opened.
The old-memory residual databases are quarantined by source name, and
``axon_personal_log.json`` is detected as manual-only without reading it.
Every emitted row is private/local-only and carries both semantic and episodic
provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from runtime.field import (
    LogicalRegion,
    SharedFieldSnapshot,
    SlotKind,
    compile_field_view,
)
from substrate import default_alphabet


SCHEMA = "axon_d00_exact_grounded_field_source_v1"
ADAPTER = "d00_exact_grounded_field_v1"
DEFAULT_SOURCE_ROOT = Path(r"D:\00")
SEMANTIC_DATABASE = "axon_semantic_memory.db"
EPISODIC_DATABASE = "axon_episodic_memory.db"
PERSONAL_LOG = "axon_personal_log.json"
EXCLUDED_OLD_MEMORY_SOURCES = (
    "axon_memory.db",
    "axon_memory_backlog.db",
    "axon_runtime_state.db",
)

FAMILY_STRUCTURED = "structured_evidence_revision_v4"
FAMILY_SCRATCH = "scratch_plan_response_v4"
KIND_ORDER = ("fact", "relation", "procedure")
KIND_WEIGHTS = {"fact": 45, "relation": 45, "procedure": 10}
DEFAULT_TOTAL_LIMIT = 8192
MAX_TARGET_CHARS = 256
MAX_HISTORY_CHARS = 512
MAX_STRUCTURED_KNOWLEDGE_CHARS = 2048
SELECTION_SALT = "axon-d00-exact-grounded-field-v1"

_ALPHABET = frozenset(default_alphabet())


class D00FieldSourceError(RuntimeError):
    """Base class for the exact-grounded source adapter."""


class SourceMutationError(D00FieldSourceError):
    """Raised when one of the read-only source databases changes."""


@dataclass(frozen=True)
class EpisodeMeta:
    episode_id: int
    episode_type: str
    summary: str
    source: str
    created_at: str


@dataclass(frozen=True)
class GroundingRef:
    episode_id: int
    json_path: str
    item_sha256: str


@dataclass(frozen=True)
class Candidate:
    kind: str
    semantic_table: str
    semantic_id: int
    semantic_pointer: str
    semantic_raw_sha256: str
    selection_sha256: str
    row: dict[str, Any]


@dataclass(frozen=True)
class D00FieldExtraction:
    """Bounded extraction result without any persisted curriculum artifact."""

    source_record_kwargs: tuple[dict[str, Any], ...]
    audit: dict[str, Any]

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(item["row"] for item in self.source_record_kwargs)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sqlite_ro_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri() + "?mode=ro"


@contextmanager
def _connect_read_only(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(_sqlite_ro_uri(path), uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def deterministic_family_quotas(total_limit: int = DEFAULT_TOTAL_LIMIT) -> dict[str, int]:
    """Allocate an exact 45/45/10 quota using deterministic largest remainder."""

    if (
        isinstance(total_limit, bool)
        or not isinstance(total_limit, int)
        or total_limit <= 0
    ):
        raise ValueError("total_limit must be a positive integer")
    denominator = sum(KIND_WEIGHTS.values())
    quotas = {
        kind: (total_limit * KIND_WEIGHTS[kind]) // denominator
        for kind in KIND_ORDER
    }
    remainder = total_limit - sum(quotas.values())
    ranked = sorted(
        KIND_ORDER,
        key=lambda kind: (
            -((total_limit * KIND_WEIGHTS[kind]) % denominator),
            KIND_ORDER.index(kind),
        ),
    )
    for kind in ranked[:remainder]:
        quotas[kind] += 1
    return quotas


def normalize_for_query_leak_check(text: str) -> str:
    """Normalize only for leakage detection, never for source matching."""

    return " ".join(
        "".join(char.casefold() if char.isalnum() else " " for char in text).split()
    )


def query_contains_normalized_answer(query: str, answers: Iterable[str]) -> bool:
    query_norm = normalize_for_query_leak_check(query)
    for answer in answers:
        answer_norm = normalize_for_query_leak_check(answer)
        if answer_norm and answer_norm in query_norm:
            return True
    return False


@lru_cache(maxsize=256)
def preflight_user_input(text: str) -> dict[str, Any]:
    """Compile one user input through the real tagged 384-slot view.

    The dedicated physical window is 64 slots, but the canonical region tag
    consumes 13 of them.  This preflight compares the rendered span characters
    to the complete canonical input so integration cannot silently clip it.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    snapshot = SharedFieldSnapshot.from_texts(
        {LogicalRegion.USER_INPUT: text},
        tick_id=0,
        source="d00_field_source_preflight",
        provenance=ADAPTER,
    )
    view = compile_field_view(
        snapshot,
        proposal_region=LogicalRegion.RESPONSE_DRAFT,
    )
    observed = "".join(
        ref.rendered_char or ""
        for ref, active in zip(
            view.slot_refs,
            view.attention_mask.tolist(),
            strict=True,
        )
        if (
            active
            and ref.kind is SlotKind.SPAN
            and ref.logical_region is LogicalRegion.USER_INPUT
        )
    )
    user_omissions = [
        omission
        for omission in view.omissions
        if omission.logical_region is LogicalRegion.USER_INPUT
    ]
    return {
        "canonical_chars": len(text),
        "visible_chars": len(observed),
        "fully_visible": observed == text and not user_omissions,
        "canonical_sha256": _sha256_bytes(text.encode("utf-8")),
        "visible_sha256": _sha256_bytes(observed.encode("utf-8")),
        "omission_count": len(user_omissions),
    }


@lru_cache(maxsize=1)
def tagged_user_payload_capacity() -> int:
    """Derive the user payload capacity from the runtime renderer itself."""

    width_probe = "x" * 64
    result = preflight_user_input(width_probe)
    capacity = int(result["visible_chars"])
    if capacity <= 0 or capacity >= 64:
        raise D00FieldSourceError(
            f"unexpected tagged user payload capacity {capacity}"
        )
    return capacity


def _is_supported_text(value: str) -> bool:
    return all(char in _ALPHABET for char in value)


def _as_exact_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_exact_steps(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _load_json_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _procedure_key_from_object(item: Mapping[str, Any]) -> tuple[Any, ...] | None:
    name = _as_exact_text(item.get("name"))
    trigger = _as_exact_text(item.get("trigger"))
    steps = _as_exact_steps(item.get("steps"))
    outcome = _as_exact_text(item.get("outcome"))
    applicability = _as_exact_text(item.get("applicability"))
    if None in (name, trigger, steps, outcome, applicability):
        return None
    return (name, trigger, steps, outcome, applicability)


def _procedure_key_from_semantic_row(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    name = _as_exact_text(row.get("name"))
    trigger = _as_exact_text(row.get("trigger"))
    outcome = _as_exact_text(row.get("outcome"))
    applicability = _as_exact_text(row.get("applicability"))
    raw_steps = row.get("steps_json")
    if not isinstance(raw_steps, str):
        return None
    try:
        steps_value = json.loads(raw_steps)
    except (TypeError, ValueError):
        return None
    steps = _as_exact_steps(steps_value)
    if None in (name, trigger, steps, outcome, applicability):
        return None
    return (name, trigger, steps, outcome, applicability)


def _episode_summary(
    raw_summary: Any,
    extracted: Mapping[str, Any],
    *,
    episode_id: int,
    episode_type: str,
) -> str:
    for value in (raw_summary, extracted.get("episode_summary")):
        if (
            isinstance(value, str)
            and value
            and len(value) <= MAX_HISTORY_CHARS
            and _is_supported_text(value)
        ):
            return value
    safe_type = episode_type if _is_supported_text(episode_type) else "unknown"
    return f"Episodic record {episode_id} has source type {safe_type}."


def _build_grounding_index(
    episodic_path: Path,
) -> tuple[
    dict[tuple[str, str], GroundingRef],
    dict[tuple[str, str, str], GroundingRef],
    dict[tuple[Any, ...], GroundingRef],
    dict[int, EpisodeMeta],
    dict[str, int],
]:
    facts: dict[tuple[str, str], GroundingRef] = {}
    relations: dict[tuple[str, str, str], GroundingRef] = {}
    procedures: dict[tuple[Any, ...], GroundingRef] = {}
    episodes: dict[int, EpisodeMeta] = {}
    counts: Counter[str] = Counter()

    with _connect_read_only(episodic_path) as connection:
        cursor = connection.execute(
            """
            SELECT id, episode_type, summary, extracted_json, source, created_at
            FROM episodes
            ORDER BY id
            """
        )
        for raw in cursor:
            counts["episode_rows_scanned"] += 1
            episode_id = int(raw["id"])
            episode_type = (
                raw["episode_type"] if isinstance(raw["episode_type"], str) else ""
            )
            extracted = _load_json_object(raw["extracted_json"])
            if extracted is None:
                counts["episode_rows_without_valid_extracted_object"] += 1
                continue
            episodes[episode_id] = EpisodeMeta(
                episode_id=episode_id,
                episode_type=episode_type,
                summary=_episode_summary(
                    raw["summary"],
                    extracted,
                    episode_id=episode_id,
                    episode_type=episode_type,
                ),
                source=raw["source"] if isinstance(raw["source"], str) else "",
                created_at=(
                    raw["created_at"] if isinstance(raw["created_at"], str) else ""
                ),
            )

            for index, item in enumerate(extracted.get("facts", [])):
                if not isinstance(item, dict):
                    counts["malformed_grounding_fact"] += 1
                    continue
                key = _as_exact_text(item.get("key"))
                value = _as_exact_text(item.get("value"))
                if key is None or value is None:
                    counts["malformed_grounding_fact"] += 1
                    continue
                facts.setdefault(
                    (key, value),
                    GroundingRef(
                        episode_id=episode_id,
                        json_path=f"$.facts[{index}]",
                        item_sha256=_sha256_bytes(_canonical_json_bytes(item)),
                    ),
                )

            for index, item in enumerate(extracted.get("relations", [])):
                if not isinstance(item, dict):
                    counts["malformed_grounding_relation"] += 1
                    continue
                subject = _as_exact_text(item.get("subject"))
                predicate = _as_exact_text(item.get("predicate"))
                object_text = _as_exact_text(item.get("object"))
                if subject is None or predicate is None or object_text is None:
                    counts["malformed_grounding_relation"] += 1
                    continue
                relations.setdefault(
                    (subject, predicate, object_text),
                    GroundingRef(
                        episode_id=episode_id,
                        json_path=f"$.relations[{index}]",
                        item_sha256=_sha256_bytes(_canonical_json_bytes(item)),
                    ),
                )

            for index, item in enumerate(extracted.get("procedures", [])):
                if not isinstance(item, dict):
                    counts["malformed_grounding_procedure"] += 1
                    continue
                key = _procedure_key_from_object(item)
                if key is None:
                    counts["malformed_grounding_procedure"] += 1
                    continue
                procedures.setdefault(
                    key,
                    GroundingRef(
                        episode_id=episode_id,
                        json_path=f"$.procedures[{index}]",
                        item_sha256=_sha256_bytes(_canonical_json_bytes(item)),
                    ),
                )

    counts["unique_grounded_facts"] = len(facts)
    counts["unique_grounded_relations"] = len(relations)
    counts["unique_grounded_procedures"] = len(procedures)
    return facts, relations, procedures, episodes, dict(sorted(counts.items()))


def _semantic_pointer(path: Path, table: str, semantic_id: int) -> str:
    return f"sqlite:{path.resolve()}#{table}:{semantic_id}"


def _episodic_pointer(path: Path, grounding: GroundingRef) -> str:
    return (
        f"sqlite:{path.resolve()}#episodes:{grounding.episode_id}"
        f"/extracted_json{grounding.json_path}"
    )


def _selection_sha256(kind: str, semantic_id: int, raw_key: Sequence[Any]) -> str:
    payload = {
        "salt": SELECTION_SALT,
        "kind": kind,
        "semantic_id": semantic_id,
        "raw_key": raw_key,
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _component_paths(kind: str, grounding: GroundingRef) -> dict[str, str]:
    base = grounding.json_path
    if kind == "fact":
        return {"key": f"{base}.key", "value": f"{base}.value"}
    if kind == "relation":
        return {
            "subject": f"{base}.subject",
            "predicate": f"{base}.predicate",
            "object": f"{base}.object",
        }
    return {
        "name": f"{base}.name",
        "trigger": f"{base}.trigger",
        "steps": f"{base}.steps",
        "outcome": f"{base}.outcome",
        "applicability": f"{base}.applicability",
    }


def _source_metadata(value: str) -> dict[str, Any]:
    """Retain source provenance without copying a free-form identifier."""

    return {
        "present": bool(value),
        "sha256": _sha256_bytes(value.encode("utf-8")) if value else "",
    }


def _references_excluded_old_memory(*values: str) -> bool:
    normalized = " ".join(value.casefold() for value in values if value)
    return any(
        name.casefold() in normalized
        or Path(name).stem.casefold() in normalized
        for name in EXCLUDED_OLD_MEMORY_SOURCES
    )


def _base_row(
    *,
    kind: str,
    semantic_table: str,
    semantic_id: int,
    semantic_pointer: str,
    semantic_raw_sha256: str,
    semantic_source: str,
    semantic_created_at: str,
    grounding: GroundingRef,
    episode: EpisodeMeta,
    episodic_path: Path,
    semantic_path: Path,
    semantic_sha256: str,
    episodic_sha256: str,
    selection_sha256: str,
) -> dict[str, Any]:
    episodic_pointer = _episodic_pointer(episodic_path, grounding)
    safe_episode_type = (
        episode.episode_type
        if _is_supported_text(episode.episode_type)
        else "unknown"
    )
    return {
        "schema": SCHEMA,
        "adapter": ADAPTER,
        "source_id": f"d00-semantic:{semantic_table}:{semantic_id}",
        "lineage_id": f"d00-episodic:{grounding.episode_id}",
        "conversation_history": episode.summary,
        "situation_awareness": (
            f"Exact episodic grounding is available from {safe_episode_type}."
        ),
        "initial_draft": "",
        "source": semantic_pointer,
        "raw_pointer": episodic_pointer,
        "evidence_refs": [semantic_pointer, episodic_pointer],
        "local_only": True,
        "cloud_export_allowed": False,
        "privacy": {
            "classification": "private_local_only",
            "local_only": True,
            "cloud_export_allowed": False,
            "personal_log_included": False,
            "old_memory_residual_included": False,
        },
        "context_origins": {
            "conversation_history": (
                f"episodic:episodes:{grounding.episode_id}:summary_or_type"
            ),
            "structured_knowledge": (
                f"semantic:{semantic_table}:{semantic_id}"
            ),
            "situation_awareness": (
                f"episodic:episodes:{grounding.episode_id}:episode_type"
            ),
            "task_state": "adapter_contract",
            "user_input": "adapter_contract",
        },
        "provenance": {
            "semantic": {
                "path": str(semantic_path.resolve()),
                "sha256": semantic_sha256,
                "table": semantic_table,
                "id": semantic_id,
                "pointer": semantic_pointer,
                "row_material_sha256": semantic_raw_sha256,
                "source_metadata": _source_metadata(semantic_source),
                "created_at": semantic_created_at,
            },
            "episodic": {
                "path": str(episodic_path.resolve()),
                "sha256": episodic_sha256,
                "table": "episodes",
                "id": grounding.episode_id,
                "json_path": grounding.json_path,
                "pointer": episodic_pointer,
                "item_sha256": grounding.item_sha256,
                "source_metadata": _source_metadata(episode.source),
                "created_at": episode.created_at,
            },
            "exact_grounding": {
                "matched": True,
                "comparison": "raw_python_string_equality_no_normalization",
                "component_paths": _component_paths(kind, grounding),
            },
            "selection": {
                "salt": SELECTION_SALT,
                "sha256": selection_sha256,
                "kind": kind,
            },
        },
    }


def _fact_candidate(
    raw: Mapping[str, Any],
    grounding: GroundingRef,
    episode: EpisodeMeta,
    *,
    semantic_path: Path,
    episodic_path: Path,
    semantic_sha256: str,
    episodic_sha256: str,
) -> tuple[Candidate | None, str | None]:
    semantic_id = int(raw["id"])
    key = raw["key"]
    value = raw["value"]
    if not isinstance(key, str) or not isinstance(value, str) or not key or not value:
        return None, "empty_or_non_text_semantic_candidate"
    target = value
    if not 1 <= len(target) <= MAX_TARGET_CHARS:
        return None, "target_outside_1_256_chars"
    structured = f"Fact key: {key}\nFact value: {value}"
    if len(structured) > MAX_STRUCTURED_KNOWLEDGE_CHARS:
        return None, "structured_knowledge_over_2048_chars"
    if not _is_supported_text(structured) or not _is_supported_text(target):
        return None, "unsupported_substrate_text"
    query = "Use the grounded fact to revise the response draft."
    if query_contains_normalized_answer(query, (target,)):
        return None, "query_contains_normalized_answer"
    user_preflight = preflight_user_input(query)
    if not user_preflight["fully_visible"]:
        return None, "user_input_exceeds_tagged_capacity"

    raw_material = {"key": key, "value": value}
    raw_sha = _sha256_bytes(_canonical_json_bytes(raw_material))
    selection_sha = _selection_sha256("fact", semantic_id, (key, value))
    semantic_pointer = _semantic_pointer(
        semantic_path, "extracted_facts", semantic_id
    )
    row = _base_row(
        kind="fact",
        semantic_table="extracted_facts",
        semantic_id=semantic_id,
        semantic_pointer=semantic_pointer,
        semantic_raw_sha256=raw_sha,
        semantic_source=raw["source"] if isinstance(raw["source"], str) else "",
        semantic_created_at=(
            raw["created_at"] if isinstance(raw["created_at"], str) else ""
        ),
        grounding=grounding,
        episode=episode,
        episodic_path=episodic_path,
        semantic_path=semantic_path,
        semantic_sha256=semantic_sha256,
        episodic_sha256=episodic_sha256,
        selection_sha256=selection_sha,
    )
    row.update(
        {
            "family": FAMILY_STRUCTURED,
            "memory_kind": "grounded_fact",
            "user_input": query,
            "task_state": "Revise the response draft from grounded fact evidence.",
            "structured_knowledge": structured,
            "response_after": target,
            "target_grounding": {
                "response_after": (
                    f"{row['raw_pointer']}.value"
                ),
                "raw_exact": True,
            },
            "active_view_preflight": {
                "user_input": dict(user_preflight),
                "tagged_payload_capacity": tagged_user_payload_capacity(),
                "paging_required": False,
            },
        }
    )
    return (
        Candidate(
            kind="fact",
            semantic_table="extracted_facts",
            semantic_id=semantic_id,
            semantic_pointer=semantic_pointer,
            semantic_raw_sha256=raw_sha,
            selection_sha256=selection_sha,
            row=row,
        ),
        None,
    )


def _relation_candidate(
    raw: Mapping[str, Any],
    grounding: GroundingRef,
    episode: EpisodeMeta,
    *,
    semantic_path: Path,
    episodic_path: Path,
    semantic_sha256: str,
    episodic_sha256: str,
) -> tuple[Candidate | None, str | None]:
    semantic_id = int(raw["id"])
    subject = raw["subject"]
    predicate = raw["predicate"]
    object_text = raw["object"]
    if not all(
        isinstance(value, str) and value
        for value in (subject, predicate, object_text)
    ):
        return None, "empty_or_non_text_semantic_candidate"
    target = object_text
    if not 1 <= len(target) <= MAX_TARGET_CHARS:
        return None, "target_outside_1_256_chars"
    structured = (
        f"Relation subject: {subject}\n"
        f"Relation predicate: {predicate}\n"
        f"Relation object: {object_text}"
    )
    if len(structured) > MAX_STRUCTURED_KNOWLEDGE_CHARS:
        return None, "structured_knowledge_over_2048_chars"
    if not _is_supported_text(structured) or not _is_supported_text(target):
        return None, "unsupported_substrate_text"
    query = "Use grounded relation evidence to revise the draft."
    if query_contains_normalized_answer(query, (target,)):
        return None, "query_contains_normalized_answer"
    user_preflight = preflight_user_input(query)
    if not user_preflight["fully_visible"]:
        return None, "user_input_exceeds_tagged_capacity"

    raw_material = {
        "subject": subject,
        "predicate": predicate,
        "object": object_text,
    }
    raw_sha = _sha256_bytes(_canonical_json_bytes(raw_material))
    selection_sha = _selection_sha256(
        "relation", semantic_id, (subject, predicate, object_text)
    )
    semantic_pointer = _semantic_pointer(
        semantic_path, "extracted_relations", semantic_id
    )
    row = _base_row(
        kind="relation",
        semantic_table="extracted_relations",
        semantic_id=semantic_id,
        semantic_pointer=semantic_pointer,
        semantic_raw_sha256=raw_sha,
        semantic_source=raw["source"] if isinstance(raw["source"], str) else "",
        semantic_created_at=(
            raw["created_at"] if isinstance(raw["created_at"], str) else ""
        ),
        grounding=grounding,
        episode=episode,
        episodic_path=episodic_path,
        semantic_path=semantic_path,
        semantic_sha256=semantic_sha256,
        episodic_sha256=episodic_sha256,
        selection_sha256=selection_sha,
    )
    row.update(
        {
            "family": FAMILY_STRUCTURED,
            "memory_kind": "grounded_relation",
            "user_input": query,
            "task_state": "Revise the response draft from grounded relation evidence.",
            "structured_knowledge": structured,
            "response_after": target,
            "target_grounding": {
                "response_after": (
                    f"{row['raw_pointer']}.object"
                ),
                "raw_exact": True,
            },
            "active_view_preflight": {
                "user_input": dict(user_preflight),
                "tagged_payload_capacity": tagged_user_payload_capacity(),
                "paging_required": False,
            },
        }
    )
    return (
        Candidate(
            kind="relation",
            semantic_table="extracted_relations",
            semantic_id=semantic_id,
            semantic_pointer=semantic_pointer,
            semantic_raw_sha256=raw_sha,
            selection_sha256=selection_sha,
            row=row,
        ),
        None,
    )


def _procedure_candidate(
    raw: Mapping[str, Any],
    key: tuple[Any, ...],
    grounding: GroundingRef,
    episode: EpisodeMeta,
    *,
    semantic_path: Path,
    episodic_path: Path,
    semantic_sha256: str,
    episodic_sha256: str,
) -> tuple[Candidate | None, str | None]:
    semantic_id = int(raw["id"])
    name, trigger, steps, outcome, applicability = key
    step_index = next(
        (index for index, item in enumerate(steps) if item),
        -1,
    )
    step = steps[step_index] if step_index >= 0 else ""
    if not step or not outcome:
        return None, "empty_procedure_target"
    targets = (step, outcome)
    if any(not 1 <= len(target) <= MAX_TARGET_CHARS for target in targets):
        return None, "target_outside_1_256_chars"
    structured = (
        f"Procedure name: {name}\n"
        f"Trigger: {trigger}\n"
        f"Grounded step: {step}\n"
        f"Outcome: {outcome}\n"
        f"Applicability: {applicability}"
    )
    if len(structured) > MAX_STRUCTURED_KNOWLEDGE_CHARS:
        return None, "structured_knowledge_over_2048_chars"
    if not _is_supported_text(structured) or any(
        not _is_supported_text(target) for target in targets
    ):
        return None, "unsupported_substrate_text"
    query = "Use the grounded procedure to plan and respond."
    if query_contains_normalized_answer(query, targets):
        return None, "query_contains_normalized_answer"
    user_preflight = preflight_user_input(query)
    if not user_preflight["fully_visible"]:
        return None, "user_input_exceeds_tagged_capacity"

    raw_material = {
        "name": name,
        "trigger": trigger,
        "steps": list(steps),
        "outcome": outcome,
        "applicability": applicability,
    }
    raw_sha = _sha256_bytes(_canonical_json_bytes(raw_material))
    selection_sha = _selection_sha256(
        "procedure",
        semantic_id,
        (name, trigger, list(steps), outcome, applicability),
    )
    semantic_pointer = _semantic_pointer(
        semantic_path, "extracted_procedures", semantic_id
    )
    row = _base_row(
        kind="procedure",
        semantic_table="extracted_procedures",
        semantic_id=semantic_id,
        semantic_pointer=semantic_pointer,
        semantic_raw_sha256=raw_sha,
        semantic_source=raw["source"] if isinstance(raw["source"], str) else "",
        semantic_created_at=(
            raw["created_at"] if isinstance(raw["created_at"], str) else ""
        ),
        grounding=grounding,
        episode=episode,
        episodic_path=episodic_path,
        semantic_path=semantic_path,
        semantic_sha256=semantic_sha256,
        episodic_sha256=episodic_sha256,
        selection_sha256=selection_sha,
    )
    row.update(
        {
            "family": FAMILY_SCRATCH,
            "memory_kind": "grounded_procedure",
            "user_input": query,
            "task_state": "Build a scratch plan, then revise the response draft.",
            "structured_knowledge": structured,
            "scratch_plan": step,
            "response_after": outcome,
            "target_grounding": {
                "scratch_plan": (
                    f"{row['raw_pointer']}.steps[{step_index}]"
                ),
                "response_after": (
                    f"{row['raw_pointer']}.outcome"
                ),
                "raw_exact": True,
            },
            "active_view_preflight": {
                "user_input": dict(user_preflight),
                "tagged_payload_capacity": tagged_user_payload_capacity(),
                "paging_required": False,
            },
        }
    )
    return (
        Candidate(
            kind="procedure",
            semantic_table="extracted_procedures",
            semantic_id=semantic_id,
            semantic_pointer=semantic_pointer,
            semantic_raw_sha256=raw_sha,
            selection_sha256=selection_sha,
            row=row,
        ),
        None,
    )


def _retain_bounded_candidate(
    heap: list[tuple[int, int, Candidate]],
    candidate: Candidate,
    limit: int,
) -> None:
    priority = int(candidate.selection_sha256, 16)
    entry = (-priority, -candidate.semantic_id, candidate)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
        return
    current_largest_priority = -heap[0][0]
    current_largest_id = -heap[0][1]
    if (priority, candidate.semantic_id) < (
        current_largest_priority,
        current_largest_id,
    ):
        heapq.heapreplace(heap, entry)


def _semantic_candidates(
    semantic_path: Path,
    episodic_path: Path,
    *,
    semantic_sha256: str,
    episodic_sha256: str,
    facts: Mapping[tuple[str, str], GroundingRef],
    relations: Mapping[tuple[str, str, str], GroundingRef],
    procedures: Mapping[tuple[Any, ...], GroundingRef],
    episodes: Mapping[int, EpisodeMeta],
    quotas: Mapping[str, int],
) -> tuple[dict[str, list[Candidate]], dict[str, int]]:
    heaps: dict[str, list[tuple[int, int, Candidate]]] = {
        kind: [] for kind in KIND_ORDER
    }
    counts: Counter[str] = Counter()

    with _connect_read_only(semantic_path) as connection:
        fact_rows = connection.execute(
            """
            SELECT id, key, value, source, created_at
            FROM extracted_facts
            ORDER BY id
            """
        )
        for raw_row in fact_rows:
            raw = dict(raw_row)
            counts["semantic_fact_candidates"] += 1
            key = (
                raw["key"] if isinstance(raw["key"], str) else None,
                raw["value"] if isinstance(raw["value"], str) else None,
            )
            grounding = facts.get(key) if None not in key else None
            if grounding is None or grounding.episode_id not in episodes:
                counts["ungrounded_semantic_candidates"] += 1
                counts["ungrounded_fact_candidates"] += 1
                continue
            episode = episodes[grounding.episode_id]
            if _references_excluded_old_memory(
                raw["source"] if isinstance(raw["source"], str) else "",
                episode.source,
            ):
                counts["quarantined_old_memory_residual_candidate"] += 1
                continue
            candidate, reason = _fact_candidate(
                raw,
                grounding,
                episode,
                semantic_path=semantic_path,
                episodic_path=episodic_path,
                semantic_sha256=semantic_sha256,
                episodic_sha256=episodic_sha256,
            )
            if candidate is None:
                counts[f"quarantined_{reason}"] += 1
                continue
            counts["eligible_fact_candidates"] += 1
            _retain_bounded_candidate(
                heaps["fact"], candidate, quotas["fact"]
            )

        relation_rows = connection.execute(
            """
            SELECT id, subject, predicate, object, source, created_at
            FROM extracted_relations
            ORDER BY id
            """
        )
        for raw_row in relation_rows:
            raw = dict(raw_row)
            counts["semantic_relation_candidates"] += 1
            key = (
                raw["subject"] if isinstance(raw["subject"], str) else None,
                raw["predicate"] if isinstance(raw["predicate"], str) else None,
                raw["object"] if isinstance(raw["object"], str) else None,
            )
            grounding = relations.get(key) if None not in key else None
            if grounding is None or grounding.episode_id not in episodes:
                counts["ungrounded_semantic_candidates"] += 1
                counts["ungrounded_relation_candidates"] += 1
                continue
            episode = episodes[grounding.episode_id]
            if _references_excluded_old_memory(
                raw["source"] if isinstance(raw["source"], str) else "",
                episode.source,
            ):
                counts["quarantined_old_memory_residual_candidate"] += 1
                continue
            candidate, reason = _relation_candidate(
                raw,
                grounding,
                episode,
                semantic_path=semantic_path,
                episodic_path=episodic_path,
                semantic_sha256=semantic_sha256,
                episodic_sha256=episodic_sha256,
            )
            if candidate is None:
                counts[f"quarantined_{reason}"] += 1
                continue
            counts["eligible_relation_candidates"] += 1
            _retain_bounded_candidate(
                heaps["relation"], candidate, quotas["relation"]
            )

        procedure_rows = connection.execute(
            """
            SELECT
                id, name, trigger, steps_json, outcome, applicability,
                source, created_at
            FROM extracted_procedures
            ORDER BY id
            """
        )
        for raw_row in procedure_rows:
            raw = dict(raw_row)
            counts["semantic_procedure_candidates"] += 1
            key = _procedure_key_from_semantic_row(raw)
            grounding = procedures.get(key) if key is not None else None
            if grounding is None or grounding.episode_id not in episodes:
                counts["ungrounded_semantic_candidates"] += 1
                counts["ungrounded_procedure_candidates"] += 1
                continue
            episode = episodes[grounding.episode_id]
            if _references_excluded_old_memory(
                raw["source"] if isinstance(raw["source"], str) else "",
                episode.source,
            ):
                counts["quarantined_old_memory_residual_candidate"] += 1
                continue
            candidate, reason = _procedure_candidate(
                raw,
                key,
                grounding,
                episode,
                semantic_path=semantic_path,
                episodic_path=episodic_path,
                semantic_sha256=semantic_sha256,
                episodic_sha256=episodic_sha256,
            )
            if candidate is None:
                counts[f"quarantined_{reason}"] += 1
                continue
            counts["eligible_procedure_candidates"] += 1
            _retain_bounded_candidate(
                heaps["procedure"], candidate, quotas["procedure"]
            )

    selected = {
        kind: sorted(
            (entry[2] for entry in heaps[kind]),
            key=lambda candidate: (
                candidate.selection_sha256,
                candidate.semantic_id,
            ),
        )
        for kind in KIND_ORDER
    }
    return selected, dict(sorted(counts.items()))


def _record_kwargs(
    candidate: Candidate,
    *,
    semantic_path: Path,
    semantic_sha256: str,
) -> dict[str, Any]:
    row = candidate.row
    return {
        "row": row,
        "pointer": candidate.semantic_pointer,
        "source_path": str(semantic_path.resolve()),
        "source_sha256": semantic_sha256,
        "row_sha256": _sha256_bytes(_canonical_json_bytes(row)),
        "adapter": ADAPTER,
    }


def _row_targets(row: Mapping[str, Any]) -> tuple[str, ...]:
    targets: list[str] = []
    for key in ("scratch_plan", "response_after"):
        value = row.get(key)
        if isinstance(value, str):
            targets.append(value)
    return tuple(targets)


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int],
) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    selected_kinds: Counter[str] = Counter()
    target_lengths: list[int] = []
    user_input_lengths: list[int] = []
    user_input_preflights = 0
    for record in records:
        row = record.get("row")
        if not isinstance(row, dict):
            failures["missing_row"] += 1
            continue
        kind = str(row.get("memory_kind", "")).removeprefix("grounded_")
        selected_kinds[kind] += 1
        targets = _row_targets(row)
        if not targets:
            failures["missing_target"] += 1
        for target in targets:
            target_lengths.append(len(target))
            if not 1 <= len(target) <= MAX_TARGET_CHARS:
                failures["target_outside_1_256_chars"] += 1
            if not _is_supported_text(target):
                failures["unsupported_target_text"] += 1
        query = row.get("user_input")
        if not isinstance(query, str) or not query:
            failures["missing_query"] += 1
        elif query_contains_normalized_answer(query, targets):
            failures["query_contains_normalized_answer"] += 1
        else:
            user_input_lengths.append(len(query))
            user_input_preflights += 1
            runtime_preflight = preflight_user_input(query)
            if not runtime_preflight["fully_visible"]:
                failures["user_input_not_fully_visible"] += 1
            stored_preflight = row.get("active_view_preflight", {}).get(
                "user_input"
            )
            if stored_preflight != runtime_preflight:
                failures["user_input_preflight_drift"] += 1
            if (
                row.get("active_view_preflight", {}).get(
                    "tagged_payload_capacity"
                )
                != tagged_user_payload_capacity()
            ):
                failures["user_input_capacity_drift"] += 1
        for field in (
            "structured_knowledge",
            "conversation_history",
            "situation_awareness",
            "task_state",
        ):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                failures[f"missing_{field}"] += 1
            elif not _is_supported_text(value):
                failures[f"unsupported_{field}"] += 1
        if row.get("local_only") is not True:
            failures["not_local_only"] += 1
        if row.get("cloud_export_allowed") is not False:
            failures["cloud_export_not_false"] += 1
        grounding = (
            row.get("provenance", {})
            .get("exact_grounding", {})
            .get("matched")
        )
        if grounding is not True:
            failures["exact_grounding_missing"] += 1

    quota_observed = {
        "fact": selected_kinds["fact"],
        "relation": selected_kinds["relation"],
        "procedure": selected_kinds["procedure"],
    }
    return {
        "passed": not failures and quota_observed == dict(quotas),
        "failure_counts": dict(sorted(failures.items())),
        "quota_observed": quota_observed,
        "quota_expected": dict(quotas),
        "target_count": len(target_lengths),
        "target_length_min": min(target_lengths) if target_lengths else 0,
        "target_length_max": max(target_lengths) if target_lengths else 0,
        "query_answer_overlap_count": failures[
            "query_contains_normalized_answer"
        ],
        "user_input_tagged_payload_capacity": tagged_user_payload_capacity(),
        "user_input_preflight_count": user_input_preflights,
        "user_input_preflight_failure_count": (
            failures["user_input_not_fully_visible"]
            + failures["user_input_preflight_drift"]
            + failures["user_input_capacity_drift"]
        ),
        "user_input_length_max": (
            max(user_input_lengths) if user_input_lengths else 0
        ),
    }


def extract_d00_field_source_records(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    *,
    total_limit: int = DEFAULT_TOTAL_LIMIT,
    verify_source_hashes: bool = True,
) -> D00FieldExtraction:
    """Extract a deterministic bounded set without writing any artifact."""

    root = Path(source_root)
    semantic_path = root / SEMANTIC_DATABASE
    episodic_path = root / EPISODIC_DATABASE
    for path in (semantic_path, episodic_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    quotas = deterministic_family_quotas(total_limit)
    before = {
        "semantic": sha256_file(semantic_path),
        "episodic": sha256_file(episodic_path),
    }
    facts, relations, procedures, episodes, grounding_counts = (
        _build_grounding_index(episodic_path)
    )
    selected, candidate_counts = _semantic_candidates(
        semantic_path,
        episodic_path,
        semantic_sha256=before["semantic"],
        episodic_sha256=before["episodic"],
        facts=facts,
        relations=relations,
        procedures=procedures,
        episodes=episodes,
        quotas=quotas,
    )
    candidates = [
        candidate
        for kind in KIND_ORDER
        for candidate in selected[kind]
    ]
    records = tuple(
        _record_kwargs(
            candidate,
            semantic_path=semantic_path,
            semantic_sha256=before["semantic"],
        )
        for candidate in candidates
    )
    after = (
        {
            "semantic": sha256_file(semantic_path),
            "episodic": sha256_file(episodic_path),
        }
        if verify_source_hashes
        else dict(before)
    )
    if before != after:
        raise SourceMutationError(
            "D:\\00 source hash changed during read-only field extraction"
        )

    contract = _validate_records(records, quotas)
    old_memory_sources = [
        name for name in EXCLUDED_OLD_MEMORY_SOURCES if (root / name).is_file()
    ]
    personal_log_detected = (root / PERSONAL_LOG).is_file()
    quarantine_counts = {
        "ungrounded_semantic_candidates": candidate_counts.get(
            "ungrounded_semantic_candidates", 0
        ),
        "old_memory_residual_sources_detected": len(old_memory_sources),
        "old_memory_residual_rows_included": 0,
        "personal_log_manual_only_sources_detected": int(
            personal_log_detected
        ),
        "personal_log_rows_read": 0,
        "personal_log_rows_included": 0,
    }
    for key, value in candidate_counts.items():
        if key.startswith("quarantined_"):
            quarantine_counts[key] = value

    source_audits = [
        {
            "role": "semantic",
            "path": str(semantic_path.resolve()),
            "sha256_before": before["semantic"],
            "sha256_after": after["semantic"],
            "read_only_verified": before["semantic"] == after["semantic"],
        },
        {
            "role": "episodic_grounding",
            "path": str(episodic_path.resolve()),
            "sha256_before": before["episodic"],
            "sha256_after": after["episodic"],
            "read_only_verified": before["episodic"] == after["episodic"],
        },
    ]
    audit = {
        "schema": SCHEMA,
        "adapter": ADAPTER,
        "total_limit": total_limit,
        "quota_weights_percent": {
            "fact": 45,
            "relation": 45,
            "procedure": 10,
        },
        "quota_expected": quotas,
        "selected_rows": len(records),
        "selected_by_kind": {
            kind: len(selected[kind]) for kind in KIND_ORDER
        },
        "selected_by_family": {
            FAMILY_STRUCTURED: len(selected["fact"])
            + len(selected["relation"]),
            FAMILY_SCRATCH: len(selected["procedure"]),
        },
        "grounding_index_counts": grounding_counts,
        "semantic_candidate_counts": candidate_counts,
        "quarantine_counts": quarantine_counts,
        "excluded_sources": {
            "old_memory_residuals": old_memory_sources,
            "personal_log": {
                "detected": personal_log_detected,
                "policy": "manual_only_never_opened_by_this_adapter",
                "included": False,
            },
        },
        "privacy": {
            "classification": "private_local_only",
            "local_only": True,
            "cloud_export_allowed": False,
        },
        "sources": source_audits,
        "record_contract": contract,
        "passed": bool(
            contract["passed"]
            and all(source["read_only_verified"] for source in source_audits)
            and quarantine_counts["old_memory_residual_rows_included"] == 0
            and quarantine_counts["personal_log_rows_read"] == 0
            and quarantine_counts["personal_log_rows_included"] == 0
        ),
    }
    return D00FieldExtraction(source_record_kwargs=records, audit=audit)


def audit_actual_stores(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    *,
    total_limit: int = DEFAULT_TOTAL_LIMIT,
    verify_source_hashes: bool = True,
) -> dict[str, Any]:
    """Return only aggregate audit data; never return source text."""

    return extract_d00_field_source_records(
        source_root,
        total_limit=total_limit,
        verify_source_hashes=verify_source_hashes,
    ).audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact episodic grounding for the private D:\\00 full-field "
            "source adapter. This command writes no curriculum data."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--total-limit",
        type=int,
        default=DEFAULT_TOTAL_LIMIT,
        help="Bounded row count; production contract is 8192.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = audit_actual_stores(
        args.source_root,
        total_limit=args.total_limit,
        verify_source_hashes=True,
    )
    print(json.dumps(audit, ensure_ascii=True, sort_keys=True, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER",
    "DEFAULT_TOTAL_LIMIT",
    "D00FieldExtraction",
    "D00FieldSourceError",
    "EPISODIC_DATABASE",
    "EXCLUDED_OLD_MEMORY_SOURCES",
    "FAMILY_SCRATCH",
    "FAMILY_STRUCTURED",
    "KIND_WEIGHTS",
    "MAX_TARGET_CHARS",
    "PERSONAL_LOG",
    "SCHEMA",
    "SEMANTIC_DATABASE",
    "SourceMutationError",
    "audit_actual_stores",
    "deterministic_family_quotas",
    "extract_d00_field_source_records",
    "main",
    "normalize_for_query_leak_check",
    "preflight_user_input",
    "query_contains_normalized_answer",
    "sha256_file",
    "tagged_user_payload_capacity",
]

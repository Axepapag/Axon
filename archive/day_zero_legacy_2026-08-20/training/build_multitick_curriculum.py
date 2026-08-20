#!/usr/bin/env python3
"""Build provenance-complete exact-v4 multi-tick curriculum episodes.

This module is additive.  It does not alter the checkpoint-compatible
``trainer_slot.py`` path.  Source files are read only, exact substrate text is
either rejected or losslessly escaped, and all derived state transitions are
auditable.

The logical field uses ten top-level regions. Logical region identity is view
metadata over the existing three learned physical roles (history, user,
response), so this data contract does not require checkpoint tensor expansion.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, unquote, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substrate import assert_supported_text, default_alphabet  # noqa: E402
from runtime.field import (  # noqa: E402
    CANONICAL_REGION_ORDER as RUNTIME_CANONICAL_REGION_ORDER,
    CONTEXT_CANDIDATE_REGIONS as RUNTIME_CONTEXT_CANDIDATE_REGIONS,
    CONTEXT_END,
    CONTEXT_START,
    CORE_WRITABLE_REGIONS as RUNTIME_CORE_WRITABLE_REGIONS,
    N_SLOTS,
    PROPOSAL_END,
    PROPOSAL_START,
    USER_END,
    USER_START,
    FieldDelta,
    FieldSpan,
    FieldViewCursor,
    LogicalRegion,
    PagedFieldView,
    PhysicalRole,
    RegionState,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
    compile_next_read_page,
    proposal_payload_capacity,
    proposal_tail_offset,
)


SCHEMA = "axon_multitick_exact_v4"
BUILDER_VERSION = "4"
DEFAULT_SOURCE_ROOT = Path(r"D:\00")
PROPOSAL_WIDTH = PROPOSAL_END - PROPOSAL_START
if PROPOSAL_WIDTH != 64:
    raise RuntimeError("exact-v4 builder requires the inherited 64-slot proposal")
SEGMENT_COUNT = 4
REGION_TARGET_BUDGET = SEGMENT_COUNT * PROPOSAL_WIDTH
TYPED_SEGMENT_DELTA_SCHEMA = "axon_multitick_segment_replace_v1"

CANONICAL_REGIONS: tuple[str, ...] = tuple(
    region.value for region in RUNTIME_CANONICAL_REGION_ORDER
)

WRITABLE_REGIONS: frozenset[str] = frozenset(
    region.value for region in RUNTIME_CORE_WRITABLE_REGIONS
)

# Exact runtime geometry. Logical tags inside context and the proposal target
# are dynamic view metadata, never additional learned char_type_emb rows.
PHYSICAL_384_PROFILE: tuple[tuple[str, int, int, int], ...] = (
    ("context_projection", CONTEXT_START, CONTEXT_END, int(PhysicalRole.CONTEXT)),
    ("user_input", USER_START, USER_END, int(PhysicalRole.USER)),
    ("proposal", PROPOSAL_START, PROPOSAL_END, int(PhysicalRole.PROPOSAL)),
)

FAMILY_GRAMMAR = "grammar_repair_v4"
FAMILY_STRUCTURED = "structured_evidence_revision_v4"
FAMILY_TOOL = "tool_integration_v4"
FAMILY_SCRATCH = "scratch_plan_response_v4"
FAMILY_ADVISOR = "advisor_revision_v4"
FAMILY_TASK = "task_situation_interruption_v4"
FAMILY_NOOP = "no_op_v4"
FAMILY_DIARY = "diary_continuity_v4"
D00_GROUNDED_ADAPTER = "d00_exact_grounded_field_v1"

SUPPORTED_FAMILIES = (
    FAMILY_GRAMMAR,
    FAMILY_STRUCTURED,
    FAMILY_TOOL,
    FAMILY_SCRATCH,
    FAMILY_ADVISOR,
    FAMILY_TASK,
    FAMILY_NOOP,
    FAMILY_DIARY,
)

_FAMILY_ALIASES = {
    "grammar_repair": FAMILY_GRAMMAR,
    "grammar_repair_v4": FAMILY_GRAMMAR,
    "structured_evidence_revision": FAMILY_STRUCTURED,
    "structured_evidence_revision_v4": FAMILY_STRUCTURED,
    "tool_integration": FAMILY_TOOL,
    "tool_integration_v4": FAMILY_TOOL,
    "scratch_plan_response": FAMILY_SCRATCH,
    "scratch_plan_response_v4": FAMILY_SCRATCH,
    "advisor_revision": FAMILY_ADVISOR,
    "advisor_revision_v4": FAMILY_ADVISOR,
    "task_situation_interruption": FAMILY_TASK,
    "task_situation_interruption_v4": FAMILY_TASK,
    "no_op": FAMILY_NOOP,
    "noop": FAMILY_NOOP,
    "no_op_v4": FAMILY_NOOP,
    "diary_continuity": FAMILY_DIARY,
    "diary_continuity_v4": FAMILY_DIARY,
}

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MISSING = object()
_ALPHABET = frozenset(default_alphabet())


class CurriculumBuildError(RuntimeError):
    """Base class for exact-v4 build failures."""


class UnsupportedSourceText(CurriculumBuildError):
    """Raised when substrate-bound source text violates the selected policy."""


class SourceMutationError(CurriculumBuildError):
    """Raised when a supposedly read-only source changes during a build."""


class CrossSplitLeakageError(CurriculumBuildError):
    """Raised when exact or normalized-near duplicates cross data splits."""


class SealedTargetError(CurriculumBuildError):
    """Raised when source data asks the core to overwrite a sealed region."""


class MixedPrivacySourceError(CurriculumBuildError):
    """Raised when private grounded D:\\00 records are mixed with other sources."""


@dataclass(frozen=True)
class SourceSpec:
    """One read-only JSONL or SQLite source."""

    kind: str
    path: str
    adapter: str = "auto"
    table: str | None = None
    query: str | None = None

    @classmethod
    def jsonl(cls, path: str | Path, *, adapter: str = "auto") -> "SourceSpec":
        return cls(kind="jsonl", path=str(path), adapter=adapter)

    @classmethod
    def sqlite(
        cls,
        path: str | Path,
        *,
        table: str | None = None,
        query: str | None = None,
        adapter: str = "auto",
    ) -> "SourceSpec":
        if (table is None) == (query is None):
            raise ValueError("SQLite source requires exactly one of table or query")
        return cls(kind="sqlite", path=str(path), adapter=adapter, table=table, query=query)


@dataclass(frozen=True)
class SourceRecord:
    row: dict[str, Any]
    pointer: str
    source_path: str
    source_sha256: str
    row_sha256: str
    adapter: str


@dataclass(frozen=True)
class ProposedTarget:
    region: str
    text: str
    phase: str
    evidence_refs: tuple[str, ...]
    origin: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _render_source_value(value: Any) -> tuple[str, str]:
    """Return a deterministic source projection plus its projection kind."""

    if value is None:
        return "", "none"
    if isinstance(value, str):
        return value, "text"
    if isinstance(value, (int, float, bool)):
        return str(value), "scalar"
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "canonical_json",
    )


class ExactTextEncoder:
    """Enforce exact substrate text without silent replacement or deletion."""

    def __init__(self, policy: str):
        if policy not in {"reject", "escape"}:
            raise ValueError("unsupported text policy must be 'reject' or 'escape'")
        self.policy = policy
        self.audits: list[dict[str, Any]] = []

    def encode(self, value: Any, field_path: str) -> str:
        raw, projection_kind = _render_source_value(value)
        unsupported = [
            (index, char)
            for index, char in enumerate(raw)
            if char not in _ALPHABET
        ]
        # "[" is the escape sentinel.  Escape literal "[" as well so decoding
        # the [U+XXXX] notation is unambiguous.
        needs_escape = unsupported or ("[" in raw and self.policy == "escape")
        if unsupported and self.policy == "reject":
            codepoints = sorted({f"U+{ord(char):04X}" for _, char in unsupported})
            raise UnsupportedSourceText(
                f"{field_path} contains {len(unsupported)} unsupported character(s): "
                + ",".join(codepoints)
            )

        if not needs_escape:
            assert_supported_text(raw)
            if projection_kind != "text":
                self.audits.append(
                    {
                        "field_path": field_path,
                        "policy": "structured_projection",
                        "projection_kind": projection_kind,
                        "original_sha256": sha256_text(raw),
                        "encoded_sha256": sha256_text(raw),
                        "mappings": [],
                    }
                )
            return raw

        encoded: list[str] = []
        mappings: list[dict[str, Any]] = []
        encoded_offset = 0
        for source_index, char in enumerate(raw):
            if char == "[" or char not in _ALPHABET:
                replacement = f"[U+{ord(char):04X}]"
                encoded.append(replacement)
                mappings.append(
                    {
                        "source_index": source_index,
                        "encoded_start": encoded_offset,
                        "encoded_end": encoded_offset + len(replacement),
                        "codepoint": f"U+{ord(char):04X}",
                    }
                )
                encoded_offset += len(replacement)
            else:
                encoded.append(char)
                encoded_offset += 1
        result = "".join(encoded)
        assert_supported_text(result)
        self.audits.append(
            {
                "field_path": field_path,
                "policy": "lossless_codepoint_escape",
                "projection_kind": projection_kind,
                "original_sha256": sha256_text(raw),
                "encoded_sha256": sha256_text(result),
                "mappings": mappings,
            }
        )
        return result


def decode_lossless_escapes(text: str) -> str:
    """Invert the unambiguous escape form emitted by ``ExactTextEncoder``."""

    pattern = re.compile(r"\[U\+([0-9A-F]{4,6})\]")
    return pattern.sub(lambda match: chr(int(match.group(1), 16)), text)


def _get_path(row: dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _first_present(row: dict[str, Any], paths: Sequence[str]) -> tuple[Any, str] | tuple[object, str]:
    for path in paths:
        value = _get_path(row, path)
        if value is not _MISSING and value is not None:
            return value, path
    return _MISSING, ""


def _has_content(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _evidence_refs(row: dict[str, Any], pointer: str) -> tuple[str, ...]:
    refs: list[str] = [pointer]
    raw = row.get("evidence_refs", [])
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        refs.extend(str(item) for item in raw if str(item))
    for path in ("input_event.raw_pointer", "raw_pointer", "source"):
        value = _get_path(row, path)
        if value is not _MISSING and value not in (None, ""):
            refs.append(str(value))
    return tuple(dict.fromkeys(refs))


def _source_lineage(row: dict[str, Any], record: SourceRecord) -> str:
    paths = (
        "lineage_id",
        "source_lineage",
        "session_id",
        "input_event.session_id",
        "raw_ledger_candidate.legacy_session_id",
        "episode_id",
        "linked_task_id",
        "input_event.linked_task_id",
        "source_id",
    )
    value, _ = _first_present(row, paths)
    if value is not _MISSING and str(value):
        return f"{record.source_sha256}:{value}"
    return f"{record.source_sha256}:{record.pointer}"


def _family_from_row(row: dict[str, Any]) -> str | None:
    explicit = str(row.get("family") or row.get("curriculum") or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", explicit).strip("_")
    if normalized in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[normalized]
    if "diary" in normalized:
        return FAMILY_DIARY
    if "advisor" in normalized:
        return FAMILY_ADVISOR
    if "tool" in normalized or "powershell" in normalized or "gui" in normalized:
        return FAMILY_TOOL
    if "interrupt" in normalized or "task" in normalized or "resume" in normalized:
        return FAMILY_TASK
    if "scratch" in normalized or "procedure" in normalized or "reasoning" in normalized:
        return FAMILY_SCRATCH
    if "structured" in normalized or "knowledge" in normalized or "memory" in normalized or "dormant" in normalized:
        return FAMILY_STRUCTURED
    if "no_op" in normalized or normalized == "active_state_management":
        return FAMILY_NOOP
    if "grammar" in normalized or ("surface" in row and "target" in row):
        return FAMILY_GRAMMAR
    if "draft_revision" in normalized:
        return FAMILY_GRAMMAR

    operations = row.get("expected_operations", [])
    operation_names = {
        str(item.get("operation", "")).upper()
        for item in operations
        if isinstance(item, dict)
    }
    if operation_names & {"INTEGRATE_ADVISOR_RESULT", "QUEUE_ADVISOR"}:
        return FAMILY_ADVISOR
    if operation_names & {"INTEGRATE_TOOL_RESULT", "QUEUE_TOOL"}:
        return FAMILY_TOOL
    if operation_names & {"APPEND_SCRATCH"}:
        return FAMILY_SCRATCH
    if operation_names & {"REVISE_DRAFT", "APPEND_DRAFT"}:
        return FAMILY_GRAMMAR
    return None


def _cell(text: str, evidence_ref: str, *, spans: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cell_spans = list(spans or [])
    if text:
        cell_spans.insert(
            0,
            {
                "type": "source_projection",
                "start": 0,
                "end": len(text),
                "evidence_ref": evidence_ref,
            },
        )
    return {
        "text": text,
        "spans": cell_spans,
        "mask": {
            "visible_start": 0,
            "visible_end": len(text),
            "masked_chars": 0,
        },
    }


_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "conversation_history": (
        "conversation_history",
        "history",
        "pre_state.rolling_summary",
        "rolling_summary",
    ),
    "user_input": (
        "user_input",
        "prompt",
        "instruction",
        "input_event.content_projection",
        "scenario",
    ),
    "response_draft": (
        "initial_draft",
        "draft_before",
        "response_draft",
        "pre_state.draft_projection",
        "surface",
    ),
    "structured_knowledge": (
        "structured_knowledge",
        "knowledge",
        "evidence",
        "dormant_available",
    ),
    "scratch": (
        "scratch",
        "scratch_before",
        "pre_state.scratch_bundles",
    ),
    "tool_results": (
        "tool_results",
        "tool_result",
        "pre_state.tool_results",
        "pre_state.tool_result_buffer",
    ),
    "advisor_input": (
        "advisor_input",
        "advisor_response",
        "advisor_result",
        "pre_state.advisor_queue",
    ),
    "task_state": (
        "task_state",
        "task_before",
        "pre_state.task_state",
        "pre_state.rolling_summary",
    ),
    "situation_awareness": (
        "situation_awareness",
        "situation",
        "scenario",
        "tick_context.situation_awareness",
    ),
    "diary": (
        "diary",
        "diary_text",
        "pre_state.diary_entries",
    ),
}


def build_logical_field(
    row: dict[str, Any],
    record: SourceRecord,
    encoder: ExactTextEncoder,
    *,
    include_diary: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    values: dict[str, str] = {}
    origins: dict[str, str] = {}
    diary_source_present = False
    for region in CANONICAL_REGIONS:
        value, origin = _first_present(row, _FIELD_PATHS[region])
        if region == "diary":
            diary_source_present = _has_content(value)
            if not include_diary:
                value = _MISSING
                origin = ""
        if value is _MISSING:
            values[region] = ""
            origins[region] = "empty"
        else:
            values[region] = encoder.encode(value, f"initial_field.{region}")
            origins[region] = origin

    field = {
        region: _cell(values[region], record.pointer)
        for region in CANONICAL_REGIONS
    }
    row_privacy = row.get("privacy", {})
    if not isinstance(row_privacy, Mapping):
        raise CurriculumBuildError("row privacy metadata must be an object")
    declared_local_only = row.get(
        "local_only",
        row_privacy.get("local_only", False),
    )
    declared_cloud_export = row.get(
        "cloud_export_allowed",
        row_privacy.get("cloud_export_allowed", None),
    )
    if not isinstance(declared_local_only, bool):
        raise CurriculumBuildError("row local_only must be a boolean")
    if declared_cloud_export is not None and not isinstance(
        declared_cloud_export,
        bool,
    ):
        raise CurriculumBuildError(
            "row cloud_export_allowed must be a boolean or absent"
        )
    if declared_local_only and declared_cloud_export is True:
        raise CurriculumBuildError(
            "local-only source row cannot allow cloud export"
        )
    source_forces_local = bool(
        declared_local_only or declared_cloud_export is False
    )
    cloud_export_allowed = bool(
        not source_forces_local
        and not (include_diary and diary_source_present)
    )
    privacy = {
        "diary_source_present": diary_source_present,
        "diary_included": bool(include_diary and diary_source_present),
        "diary_exclusion_recorded": bool(diary_source_present and not include_diary),
        "source_declared_local_only": declared_local_only,
        "source_declared_cloud_export_allowed": declared_cloud_export,
        "local_only": not cloud_export_allowed,
        "cloud_export_allowed": cloud_export_allowed,
        "region_origins": origins,
    }
    return field, privacy


def canonical_field_payload(field: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for region in CANONICAL_REGIONS:
        cell = field[region]
        item: dict[str, Any] = {
            "region": region,
            "text": cell["text"],
            "spans": cell.get("spans", []),
        }
        payload.append(item)
    return payload


def field_sha256(field: dict[str, dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(canonical_field_payload(field)))


def _runtime_snapshot_from_builder_field(
    *,
    field: Mapping[str, Mapping[str, Any]],
    episode_id: str,
    record: SourceRecord,
    evidence_refs: tuple[str, ...],
) -> SharedFieldSnapshot:
    """Materialize the exact runtime snapshot later reconstructed by trainer."""

    builder_field_hash = field_sha256(dict(field))  # type: ignore[arg-type]
    regions: list[RegionState] = []
    for logical_region in RUNTIME_CANONICAL_REGION_ORDER:
        cell = field[logical_region.value]
        text = str(cell["text"])
        spans: tuple[FieldSpan, ...] = ()
        if text:
            builder_spans = cell.get("spans", [])
            span_audit = json.dumps(
                builder_spans,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            spans = (
                FieldSpan(
                    span_id=(
                        f"{episode_id}:{logical_region.value}:"
                        f"{sha256_text(text)[:16].lower()}"
                    ),
                    text=text,
                    kind="exact_v4_builder_cell",
                    source=record.pointer,
                    provenance=span_audit,
                    container_refs=evidence_refs,
                ),
            )
        regions.append(RegionState(name=logical_region, spans=spans))
    return SharedFieldSnapshot(
        tick_id=0,
        regions=tuple(regions),
        source_manifest_ids=tuple(
            sorted({builder_field_hash, record.source_sha256})
        ),
    )


def _runtime_teacher_commit(
    snapshot: SharedFieldSnapshot,
    *,
    tick_index: int,
    target_region: str,
    delta_start: int,
    delta_end: int,
    replacement: str,
    committed_text: str,
    input_builder_field_sha256: str,
    typed_delta: Mapping[str, Any],
    evidence_refs: tuple[str, ...],
) -> SharedFieldSnapshot:
    """Mirror the trainer's exact runtime teacher transaction."""

    region = LogicalRegion(target_region)
    if snapshot.region(region).text == committed_text:
        return snapshot
    typed_delta_sha256 = sha256_bytes(
        canonical_json_bytes(typed_delta)
    ).upper()
    delta = FieldDelta(
        base_field_id=snapshot.field_id,
        base_tick_id=snapshot.tick_id,
        author_core_id="exact-v4-teacher",
        pass_id=f"teacher-tick-{tick_index:04d}",
        operations=(
            ReplaceText(
                region=region,
                start=delta_start,
                end=delta_end,
                text=replacement,
                provenance=(
                    f"exact-v4:{input_builder_field_sha256}:"
                    f"{typed_delta_sha256}"
                ),
                edge_refs=evidence_refs,
            ),
        ),
        evidence=evidence_refs,
    )
    return apply_delta(snapshot, delta)


def active_view_audit(
    field: dict[str, dict[str, Any]],
    proposal_region: str,
    read_page: PagedFieldView,
) -> dict[str, Any]:
    """Describe runtime geometry without inventing fixed logical sub-slices."""

    if proposal_region not in WRITABLE_REGIONS:
        raise SealedTargetError(f"proposal region {proposal_region!r} is sealed")
    if read_page.view.proposal_region.value != proposal_region:
        raise AssertionError("read page proposal region mismatch")
    context_regions = [
        region.value
        for region in RUNTIME_CONTEXT_CANDIDATE_REGIONS
        if region.value != proposal_region
    ]
    proposal_text = field[proposal_region]["text"]
    payload_capacity = proposal_payload_capacity(proposal_region)
    proposal_view_offset = proposal_tail_offset(proposal_region, proposal_text)
    proposal_view_text = proposal_text[
        proposal_view_offset : proposal_view_offset + payload_capacity
    ]
    return {
        "geometry": [
            {
                "name": name,
                "slot_start": start,
                "slot_end": end,
                "width": end - start,
                "physical_type_id": type_id,
            }
            for name, start, end, type_id in PHYSICAL_384_PROFILE
        ],
        "context_projection": {
            "dynamic_tagged_projection": True,
            "logical_regions": context_regions,
            "region_sources": [
                {
                    "region": region,
                    "full_chars": len(field[region]["text"]),
                    "full_text_sha256": sha256_text(field[region]["text"]),
                }
                for region in context_regions
            ],
            "fixed_logical_sub_slices": False,
            "read_page_index": read_page.cursor.page_index,
            "read_view_hash": read_page.view.view_hash,
            "cursor_before": read_page.cursor.to_canonical_dict(),
            "cursor_after": read_page.next_cursor.to_canonical_dict(),
            "read_cycle_complete": read_page.read_cycle_complete,
        },
        "user_projection": {
            "logical_region": "user_input",
            "full_chars": len(field["user_input"]["text"]),
            "full_text_sha256": sha256_text(field["user_input"]["text"]),
        },
        "proposal_projection": {
            "dynamic_target_region": proposal_region,
            "current_chars": len(proposal_text),
            "current_text_sha256": sha256_text(proposal_text),
            "payload_capacity": payload_capacity,
            "proposal_view_offset": proposal_view_offset,
            "visible_start": proposal_view_offset,
            "visible_end": proposal_view_offset + len(proposal_view_text),
            "visible_chars": len(proposal_view_text),
            "visible_text_sha256": sha256_text(proposal_view_text),
            "shows_current_tail": (
                proposal_view_offset + len(proposal_view_text) == len(proposal_text)
            ),
        },
    }


def transition_phase_identity(phase: str, transition_index: int) -> str:
    """Return the stable identity shared by one transition's four segments."""

    return f"{phase}:transition_{transition_index:04d}"


def apply_segment_delta(
    field: dict[str, dict[str, Any]],
    delta: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Replay one exact-v4 segmented replace/append delta over a full field."""

    if delta.get("schema") != TYPED_SEGMENT_DELTA_SCHEMA:
        raise ValueError("unknown segmented delta schema")
    if delta.get("op") != "replace_text":
        raise ValueError("segmented delta op must be replace_text")
    region = delta.get("region")
    if not isinstance(region, str) or region not in WRITABLE_REGIONS:
        raise ValueError("segmented delta region is not writable")
    start = delta.get("start")
    end = delta.get("end")
    replacement = delta.get("replacement")
    evidence_ref = delta.get("evidence_ref")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise ValueError("segmented delta bounds must be integers")
    if not isinstance(replacement, str):
        raise ValueError("segmented delta replacement must be a string")
    if not isinstance(evidence_ref, str) or not evidence_ref:
        raise ValueError("segmented delta evidence_ref must be non-empty")
    base = field[region]["text"]
    if not 0 <= start <= end <= len(base):
        raise ValueError("segmented delta bounds are outside the region")
    expected_base_field_hash = field_sha256(field)
    if str(delta.get("base_field_sha256", "")).upper() != expected_base_field_hash:
        raise ValueError("segmented delta base field hash mismatch")
    if str(delta.get("base_region_sha256", "")).upper() != sha256_text(base):
        raise ValueError("segmented delta base region hash mismatch")
    result = base[:start] + replacement + base[end:]
    if str(delta.get("result_region_sha256", "")).upper() != sha256_text(result):
        raise ValueError("segmented delta result region hash mismatch")
    committed = _commit_region(field, region, result, evidence_ref)
    if str(delta.get("result_field_sha256", "")).upper() != field_sha256(committed):
        raise ValueError("segmented delta result field hash mismatch")
    return committed


def character_diff(base: str, proposed: str) -> dict[str, Any]:
    """Return a deterministic, fully replayable character diff and masks."""

    matcher = difflib.SequenceMatcher(a=base, b=proposed, autojunk=False)
    changes: list[dict[str, Any]] = []
    base_changed = [0] * len(base)
    proposed_changed = [0] * len(proposed)
    base_unchanged = [0] * len(base)
    proposed_unchanged = [0] * len(proposed)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for index in range(i1, i2):
                base_unchanged[index] = 1
            for index in range(j1, j2):
                proposed_unchanged[index] = 1
            continue
        for index in range(i1, i2):
            base_changed[index] = 1
        for index in range(j1, j2):
            proposed_changed[index] = 1
        changes.append(
            {
                "op": tag,
                "base_start": i1,
                "base_end": i2,
                "proposed_start": j1,
                "proposed_end": j2,
                "base_text": base[i1:i2],
                "proposed_text": proposed[j1:j2],
            }
        )

    result = {
        "schema": "axon_character_diff_v1",
        "changes": changes,
        "masks": {
            "base_changed": base_changed,
            "base_unchanged": base_unchanged,
            "proposed_changed": proposed_changed,
            "proposed_unchanged": proposed_unchanged,
        },
    }
    if apply_character_diff(base, result) != proposed:
        raise AssertionError("internal character diff replay mismatch")
    return result


def apply_character_diff(base: str, diff: dict[str, Any]) -> str:
    cursor = 0
    parts: list[str] = []
    for change in diff.get("changes", []):
        start = int(change["base_start"])
        end = int(change["base_end"])
        if start < cursor or base[start:end] != change["base_text"]:
            raise ValueError("invalid or overlapping character diff")
        parts.append(base[cursor:start])
        parts.append(str(change["proposed_text"]))
        cursor = end
    parts.append(base[cursor:])
    return "".join(parts)


def _intermediate_refinement(base: str, target: str) -> str:
    """Apply the first half of deterministic aligned character edits."""

    matcher = difflib.SequenceMatcher(a=base, b=target, autojunk=False)
    units: list[tuple[str, str, bool]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old = base[i1:i2]
        new = target[j1:j2]
        if tag == "equal":
            units.extend((char, char, False) for char in old)
            continue
        width = max(len(old), len(new))
        for index in range(width):
            units.append(
                (
                    old[index] if index < len(old) else "",
                    new[index] if index < len(new) else "",
                    True,
                )
            )
    changed_indices = [index for index, (_, _, changed) in enumerate(units) if changed]
    take = math.ceil(len(changed_indices) / 2)
    applied = set(changed_indices[:take])
    return "".join(new if index in applied else old for index, (old, new, _) in enumerate(units))


def _refinement_targets(
    *,
    region: str,
    base: str,
    target: str,
    evidence_refs: tuple[str, ...],
    origin: str,
) -> list[ProposedTarget]:
    intermediate = _intermediate_refinement(base, target)
    return [
        ProposedTarget(region, intermediate, "refine", evidence_refs, origin),
        ProposedTarget(region, target, "finalize", evidence_refs, origin),
    ]


_RESPONSE_TARGET_PATHS = (
    "proposed_region_text",
    "complete_proposed_region_text",
    "response_after",
    "draft_after",
    "expected_response",
    "proposed_response",
    "corrected",
    "answer",
    "target_text",
    "target",
    "expected_state_delta.draft_after",
    "post_state_targets.draft_after",
    "training_targets.positive",
)


def _encoded_pick(
    row: dict[str, Any],
    paths: Sequence[str],
    encoder: ExactTextEncoder,
    audit_path: str,
) -> tuple[str | None, str]:
    value, origin = _first_present(row, paths)
    if value is _MISSING:
        return None, ""
    return encoder.encode(value, audit_path), origin


def targets_for_row(
    row: dict[str, Any],
    family: str,
    field: dict[str, dict[str, Any]],
    encoder: ExactTextEncoder,
    evidence_refs: tuple[str, ...],
) -> list[ProposedTarget]:
    explicit = row.get("targets")
    if explicit is None:
        explicit = row.get("ticks") if isinstance(row.get("ticks"), list) else None
    if isinstance(explicit, list):
        targets: list[ProposedTarget] = []
        for index, target in enumerate(explicit):
            if not isinstance(target, dict):
                raise CurriculumBuildError("explicit targets must be objects")
            region = str(target.get("region") or target.get("target_region") or "")
            if region not in WRITABLE_REGIONS:
                raise SealedTargetError(
                    f"unsupported or sealed explicit target region {region!r}"
                )
            raw = target.get(
                "complete_proposed_region_text",
                target.get("proposed_region_text", target.get("text", _MISSING)),
            )
            if raw is _MISSING:
                raise CurriculumBuildError("explicit target lacks complete proposed region text")
            text = encoder.encode(raw, f"targets[{index}].{region}")
            targets.append(
                ProposedTarget(
                    region=region,
                    text=text,
                    phase=str(target.get("phase") or f"tick_{index}"),
                    evidence_refs=evidence_refs,
                    origin=f"targets[{index}]",
                )
            )
        return targets

    response_target, response_origin = _encoded_pick(
        row,
        _RESPONSE_TARGET_PATHS,
        encoder,
        "targets.response_draft",
    )
    response_base = field["response_draft"]["text"]

    if family == FAMILY_GRAMMAR:
        if response_target is None:
            return []
        return _refinement_targets(
            region="response_draft",
            base=response_base,
            target=response_target,
            evidence_refs=evidence_refs,
            origin=response_origin,
        )

    if family in {FAMILY_STRUCTURED, FAMILY_TOOL, FAMILY_ADVISOR}:
        required_region = {
            FAMILY_STRUCTURED: "structured_knowledge",
            FAMILY_TOOL: "tool_results",
            FAMILY_ADVISOR: "advisor_input",
        }[family]
        if not field[required_region]["text"] or response_target is None:
            return []
        return _refinement_targets(
            region="response_draft",
            base=response_base,
            target=response_target,
            evidence_refs=evidence_refs,
            origin=response_origin,
        )

    if family == FAMILY_SCRATCH:
        plan, plan_origin = _encoded_pick(
            row,
            (
                "scratch_plan",
                "plan",
                "scratch_after",
                "expected_core_reasoning_summary",
                "training_targets.procedural_wisdom",
            ),
            encoder,
            "targets.scratch",
        )
        if plan is None:
            return []
        targets = [
            ProposedTarget("scratch", plan, "plan", evidence_refs, plan_origin),
        ]
        if response_target is not None:
            targets.append(
                ProposedTarget(
                    "response_draft",
                    response_target,
                    "respond",
                    evidence_refs,
                    response_origin,
                )
            )
        return targets

    if family == FAMILY_TASK:
        preservation_plan, plan_origin = _encoded_pick(
            row,
            (
                "scratch_plan",
                "plan",
                "scratch_after",
                "task_after",
                "proposed_task_state",
                "expected_core_reasoning_summary",
                "training_targets.procedural_wisdom",
            ),
            encoder,
            "targets.scratch",
        )
        if preservation_plan is None:
            return []
        targets = [
            ProposedTarget(
                "scratch",
                preservation_plan,
                "preserve_or_resume",
                evidence_refs,
                plan_origin,
            )
        ]
        if response_target is not None and response_origin not in {
            "training_targets.positive",
            "target",
        }:
            targets.append(
                ProposedTarget(
                    "response_draft",
                    response_target,
                    "respond_after_resume",
                    evidence_refs,
                    response_origin,
                )
            )
        return targets

    if family == FAMILY_NOOP:
        return [
            ProposedTarget("response_draft", response_base, "preserve", evidence_refs, "no_op"),
            ProposedTarget("response_draft", response_base, "confirm_stable", evidence_refs, "no_op"),
        ]

    if family == FAMILY_DIARY:
        diary_target, diary_origin = _encoded_pick(
            row,
            (
                "diary_response",
                "response_after",
                "answer",
                "target_text",
                "target",
                "training_targets.positive",
            ),
            encoder,
            "targets.diary_response",
        )
        if not field["diary"]["text"] or diary_target is None:
            return []
        return _refinement_targets(
            region="response_draft",
            base=response_base,
            target=diary_target,
            evidence_refs=evidence_refs,
            origin=diary_origin,
        )
    return []


def _commit_region(
    field: dict[str, dict[str, Any]],
    region: str,
    text: str,
    evidence_ref: str,
) -> dict[str, dict[str, Any]]:
    committed = copy.deepcopy(field)
    committed[region] = _cell(text, evidence_ref)
    return committed


def build_episode(
    record: SourceRecord,
    *,
    unsupported_policy: str,
    include_diary: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    row = record.row
    family = _family_from_row(row)
    if family is None:
        return None, "unrecognized_family"
    if family == FAMILY_DIARY and not include_diary:
        return None, "diary_family_excluded"

    encoder = ExactTextEncoder(unsupported_policy)
    field, privacy = build_logical_field(
        row,
        record,
        encoder,
        include_diary=include_diary,
    )
    refs = _evidence_refs(row, record.pointer)
    targets = targets_for_row(row, family, field, encoder, refs)
    if not targets:
        return None, "missing_exact_target_or_evidence"
    # Every logical transition has a fixed four-tick structural schedule.  Each
    # model target is one 64-character segment, while the committed canonical
    # region can grow to the audited 256-character curriculum-only budget.
    if any(len(target.text) > REGION_TARGET_BUDGET for target in targets):
        return None, "proposal_target_exceeds_256"

    lineage = _source_lineage(row, record)
    identity = {
        "family": family,
        "lineage": lineage,
        "record": record.pointer,
        "row_sha256": record.row_sha256,
        "targets": [(target.region, target.text, target.phase) for target in targets],
        "builder_version": BUILDER_VERSION,
        "segment_protocol": {
            "segment_count": SEGMENT_COUNT,
            "segment_width": PROPOSAL_WIDTH,
            "region_target_budget": REGION_TARGET_BUDGET,
        },
    }
    episode_id = "mtv4-" + sha256_bytes(canonical_json_bytes(identity))[:24].lower()

    initial_field = copy.deepcopy(field)
    runtime_current = _runtime_snapshot_from_builder_field(
        field=initial_field,
        episode_id=episode_id,
        record=record,
        evidence_refs=refs,
    )
    read_cursor = FieldViewCursor()
    ticks: list[dict[str, Any]] = []
    current = field
    tick_index = 0
    for transition_index, target in enumerate(targets):
        phase_identity = transition_phase_identity(target.phase, transition_index)
        transition_target_sha256 = sha256_text(target.text)
        for segment_index in range(SEGMENT_COUNT):
            segment_start = segment_index * PROPOSAL_WIDTH
            segment_target = target.text[
                segment_start : segment_start + PROPOSAL_WIDTH
            ]
            base = current[target.region]["text"]
            if segment_index == 0:
                delta_start = 0
                delta_end = len(base)
                segment_phase = "replace_current_region"
            else:
                delta_start = len(base)
                delta_end = len(base)
                segment_phase = "append_at_current_end"
            committed_text = (
                base[:delta_start] + segment_target + base[delta_end:]
            )
            committed = _commit_region(
                current,
                target.region,
                committed_text,
                record.pointer,
            )
            typed_delta = {
                "schema": TYPED_SEGMENT_DELTA_SCHEMA,
                "op": "replace_text",
                "region": target.region,
                "start": delta_start,
                "end": delta_end,
                "replacement": segment_target,
                "evidence_ref": record.pointer,
                "base_region_sha256": sha256_text(base),
                "result_region_sha256": sha256_text(committed_text),
                "base_field_sha256": field_sha256(current),
                "result_field_sha256": field_sha256(committed),
            }
            if apply_segment_delta(current, typed_delta) != committed:
                raise AssertionError("internal segmented field replay mismatch")
            if {
                region.value: runtime_current.region(region).text
                for region in RUNTIME_CANONICAL_REGION_ORDER
            } != {
                region: current[region]["text"]
                for region in CANONICAL_REGIONS
            }:
                raise AssertionError(
                    "runtime paging snapshot diverged from builder field"
                )
            proposal_view_offset = proposal_tail_offset(
                target.region,
                base,
            )
            read_page = compile_next_read_page(
                runtime_current,
                proposal_region=target.region,
                cursor=read_cursor,
                proposal_offset=proposal_view_offset,
            )
            ticks.append(
                {
                    "tick_index": tick_index,
                    "transition_index": transition_index,
                    "phase": target.phase,
                    "phase_identity": phase_identity,
                    "segment_index": segment_index,
                    "segment_count": SEGMENT_COUNT,
                    "segment_phase": segment_phase,
                    "field_before": current,
                    "active_view": active_view_audit(
                        current,
                        target.region,
                        read_page,
                    ),
                    "read_page": read_page.to_audit_dict(),
                    "input_field_sha256": field_sha256(current),
                    "target_region": target.region,
                    "segment_target_text": segment_target,
                    "complete_proposed_region_text": committed_text,
                    "transition_target_chars": len(target.text),
                    "transition_target_sha256": transition_target_sha256,
                    "base_region_sha256": sha256_text(base),
                    "committed_region_sha256": sha256_text(committed_text),
                    "typed_delta": typed_delta,
                    "evidence_refs": list(target.evidence_refs),
                    "target_origin": target.origin,
                    "model_supervision": {
                        "target_is_segment_only": True,
                        "target_slots": PROPOSAL_WIDTH,
                        "target_chars": len(segment_target),
                    },
                    "teacher_forced": {
                        "commit_segment_delta": True,
                        "committed_field_sha256": field_sha256(committed),
                    },
                    "free_running_evaluation": {
                        "commit_model_segment_structurally": True,
                        "compare_region_sha256": sha256_text(committed_text),
                        "compare_field_sha256": field_sha256(committed),
                        "teacher_forcing_allowed": False,
                    },
                    "field_after_teacher_commit": committed,
                }
            )
            runtime_current = _runtime_teacher_commit(
                runtime_current,
                tick_index=tick_index,
                target_region=target.region,
                delta_start=delta_start,
                delta_end=delta_end,
                replacement=segment_target,
                committed_text=committed_text,
                input_builder_field_sha256=field_sha256(current),
                typed_delta=typed_delta,
                evidence_refs=tuple(
                    sorted(set(str(item) for item in target.evidence_refs))
                ),
            )
            read_cursor = read_page.next_cursor
            current = committed
            tick_index += 1

    transition_target_characters = sum(len(target.text) for target in targets)
    supervised_segment_characters = sum(
        len(str(tick["segment_target_text"]))
        for tick in ticks
    )
    if supervised_segment_characters != transition_target_characters:
        raise AssertionError(
            "four-by-sixty-four segmentation lost target characters"
        )
    episode = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "episode_id": episode_id,
        "family": family,
        "split": "",
        "source_lineage": lineage,
        "provenance": {
            "source_kind": record.pointer.split(":", 1)[0],
            "source_path": record.source_path,
            "source_sha256": record.source_sha256,
            "record_pointer": record.pointer,
            "row_sha256": record.row_sha256,
            "adapter": record.adapter,
            "evidence_refs": list(refs),
            "text_transform_audits": encoder.audits,
        },
        "privacy": privacy,
        "logical_regions": list(CANONICAL_REGIONS),
        "field_profile": {
            "name": "runtime_three_role_384_v1",
            "total_slots": N_SLOTS,
            "runtime_parity": True,
            "physical_roles": [
                {
                    "name": name,
                    "slot_start": start,
                    "slot_end": end,
                    "width": end - start,
                    "physical_type_id": type_id,
                }
                for name, start, end, type_id in PHYSICAL_384_PROFILE
            ],
            "learned_physical_role_count": 3,
            "logical_tags": "dynamic_context_and_proposal_metadata",
            "writable_proposal_regions": sorted(WRITABLE_REGIONS),
            "read_paging": {
                "protocol": "canonical_read_cursor_v1",
                "cursor_scope": "episode",
                "proposal_switch_resets_cursor": False,
                "reset_condition": "complete_sealed_context_and_user_cycle",
                "proposal_window": "tag_aware_current_region_tail",
            },
        },
        "segment_protocol": {
            "name": "fixed_four_by_sixty_four_v1",
            "segment_count": SEGMENT_COUNT,
            "segment_width": PROPOSAL_WIDTH,
            "region_target_budget": REGION_TARGET_BUDGET,
            "segment_zero": "replace_entire_current_region",
            "later_segments": "append_at_current_region_end",
            "empty_segments": "supervised_no_op",
        },
        "read_coverage_contract": {
            "audited_cursor_scope": "sealed_context_regions_plus_user_input",
            "sealed_read_cycle_completed_within_episode": any(
                bool(tick["read_page"]["read_cycle_complete"])
                for tick in ticks
            ),
            "writable_region_post_commit_readback_guaranteed": False,
            "whole_field_read_coverage_proven": False,
            "blocking_reason": (
                "four output ticks do not guarantee post-commit readback of "
                "every character in a 256-character writable region"
            ),
            "transaction_target_coverage": {
                "transition_count": len(targets),
                "target_characters": transition_target_characters,
                "supervised_segment_characters": supervised_segment_characters,
                "each_target_character_supervised_exactly_once": True,
                "segment_count_per_transition": SEGMENT_COUNT,
                "segment_width": PROPOSAL_WIDTH,
            },
            "launch_gate_passed": False,
        },
        "initial_field": initial_field,
        "initial_field_sha256": field_sha256(initial_field),
        "ticks": ticks,
        "final_teacher_forced_field_sha256": field_sha256(current),
        "evaluation_protocol": {
            "teacher_forced_metrics": True,
            "free_running_rollout_required": True,
            "counterfactual_region_swap_required": True,
            "no_private_state_claim": True,
        },
        "local_only": bool(privacy["local_only"]),
    }
    return episode, None


def _sqlite_path_from_uri(value: str) -> Path:
    if not value.startswith("file:"):
        return Path(value)
    parsed = urlsplit(value)
    raw = unquote(parsed.path)
    # Windows file URIs commonly parse as /D:/path.
    if re.match(r"^/[A-Za-z]:/", raw):
        raw = raw[1:]
    if parsed.netloc and not raw:
        raw = unquote(parsed.netloc)
    return Path(raw)


def sqlite_ro_uri(path_or_uri: str | Path) -> str:
    raw = str(path_or_uri)
    if raw.startswith("file:"):
        parsed = urlsplit(raw)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if "mode" in query and query["mode"] != "ro":
            raise ValueError("SQLite URI mode must be ro")
        query["mode"] = "ro"
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
    return Path(raw).resolve().as_uri() + "?mode=ro"


def _record_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("id", "row_id", "source_id", "episode_id", "lineage_id"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return str(fallback)


def iter_jsonl_records(spec: SourceSpec, source_sha256: str) -> Iterator[SourceRecord]:
    path = Path(spec.path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CurriculumBuildError(f"{path}:{line_number} is not a JSON object")
            pointer = f"jsonl:{path.resolve()}#line={line_number}"
            yield SourceRecord(
                row=row,
                pointer=pointer,
                source_path=str(path.resolve()),
                source_sha256=source_sha256,
                row_sha256=sha256_bytes(canonical_json_bytes(row)),
                adapter=spec.adapter,
            )


def iter_sqlite_records(spec: SourceSpec, source_sha256: str) -> Iterator[SourceRecord]:
    if spec.table is not None and not _IDENTIFIER.fullmatch(spec.table):
        raise ValueError(f"unsafe SQLite table name {spec.table!r}")
    uri = sqlite_ro_uri(spec.path)
    physical_path = _sqlite_path_from_uri(uri).resolve()
    query = spec.query or f'SELECT * FROM "{spec.table}"'
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        cursor = connection.execute(query)
        for index, raw in enumerate(cursor, 1):
            row = dict(raw)
            pointer = (
                f"sqlite:{physical_path}#"
                f"{spec.table or 'query'}:{_record_id(row, index)}"
            )
            yield SourceRecord(
                row=row,
                pointer=pointer,
                source_path=str(physical_path),
                source_sha256=source_sha256,
                row_sha256=sha256_bytes(canonical_json_bytes(row)),
                adapter=spec.adapter,
            )
    finally:
        connection.close()


def _physical_source_path(spec: SourceSpec) -> Path:
    if spec.kind == "sqlite":
        return _sqlite_path_from_uri(sqlite_ro_uri(spec.path)).resolve()
    return Path(spec.path).resolve()


def iter_source_records(spec: SourceSpec, source_sha256: str) -> Iterator[SourceRecord]:
    if spec.kind == "jsonl":
        yield from iter_jsonl_records(spec, source_sha256)
        return
    if spec.kind == "sqlite":
        yield from iter_sqlite_records(spec, source_sha256)
        return
    raise ValueError(f"unsupported source kind {spec.kind!r}")


def discover_primary_sources(source_root: str | Path = DEFAULT_SOURCE_ROOT) -> list[SourceSpec]:
    """Return legacy JSONL candidates for explicit opt-in callers only.

    The command-line default no longer calls this helper; it uses the
    exact-grounded semantic-plus-episodic adapter.
    """

    root = Path(source_root)
    candidates = (
        root / "corpus" / "grammar_curriculum.jsonl",
        root / "corpus" / "runtime_tool_lifecycle_statebus.jsonl",
        root / "corpus" / "tool_corrections_statebus.jsonl",
        root / "corpus" / "online_reasoning_curriculum.jsonl",
        root / "corpus" / "online_memory_curriculum.jsonl",
        root / "corpus" / "online_tool_curriculum.jsonl",
        root / "axon_runtime" / "src" / "Training_Bucket" / "axon_statebus_train.jsonl",
    )
    return [SourceSpec.jsonl(path) for path in candidates if path.exists()]


def assign_source_disjoint_splits(
    episodes: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, int]:
    """Co-locate duplicate-connected lineage components before splitting.

    A lineage is the smallest indivisible source group.  Exact and normalized
    near-duplicate episode fingerprints add undirected edges between lineages;
    union-find then closes those edges transitively.  Whole connected
    components are assigned deterministically with an equal-weight objective
    over lineage and episode count.
    """

    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_lineage[str(episode["source_lineage"])].append(episode)
    lineages = sorted(by_lineage)
    if not lineages:
        return {"train": 0, "dev": 0, "test": 0}

    parent = {lineage: lineage for lineage in lineages}

    def find(lineage: str) -> str:
        root = lineage
        while parent[root] != root:
            root = parent[root]
        while parent[lineage] != lineage:
            next_lineage = parent[lineage]
            parent[lineage] = root
            lineage = next_lineage
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    exact_owner: dict[str, str] = {}
    near_owner: dict[str, str] = {}
    for lineage in lineages:
        for episode in sorted(
            by_lineage[lineage],
            key=lambda item: str(item["episode_id"]),
        ):
            content = _episode_content(episode)
            fingerprints = (
                (exact_owner, sha256_text(content)),
                (near_owner, sha256_text(_near_normalize(content))),
            )
            for owners, fingerprint in fingerprints:
                owner = owners.setdefault(fingerprint, lineage)
                union(lineage, owner)

    component_lineages: dict[str, list[str]] = defaultdict(list)
    for lineage in lineages:
        component_lineages[find(lineage)].append(lineage)

    components: list[dict[str, Any]] = []
    for members in component_lineages.values():
        members.sort()
        component_episodes = [
            episode
            for lineage in members
            for episode in by_lineage[lineage]
        ]
        cluster_id = (
            "split-cluster-"
            + sha256_text("|".join(members))[:16].lower()
        )
        components.append(
            {
                "cluster_id": cluster_id,
                "lineages": members,
                "episodes": component_episodes,
                "lineage_count": len(members),
                "episode_count": len(component_episodes),
                "order_hash": sha256_text(f"{seed}|{cluster_id}"),
            }
        )
    components.sort(
        key=lambda component: (
            -int(component["lineage_count"]),
            -int(component["episode_count"]),
            str(component["order_hash"]),
        )
    )

    split_names = ("train", "dev", "test")
    split_weights = {"train": 8, "dev": 1, "test": 1}
    lineage_counts = {split: 0 for split in split_names}
    episode_counts = {split: 0 for split in split_names}
    component_counts = {split: 0 for split in split_names}
    total_lineages = len(lineages)
    total_episodes = len(episodes)

    for component_index, component in enumerate(components):
        remaining = len(components) - component_index
        required_empty = [
            split
            for split in ("dev", "test")
            if component_counts[split] == 0
        ]
        candidates: Sequence[str]
        if len(components) >= 3 and remaining == len(required_empty):
            candidates = required_empty
        else:
            candidates = split_names

        def assignment_score(candidate: str) -> tuple[int, int]:
            proposed_lineages = dict(lineage_counts)
            proposed_episodes = dict(episode_counts)
            proposed_lineages[candidate] += int(component["lineage_count"])
            proposed_episodes[candidate] += int(component["episode_count"])
            lineage_error = sum(
                abs(
                    proposed_lineages[split] * 10
                    - total_lineages * split_weights[split]
                )
                for split in split_names
            )
            episode_error = sum(
                abs(
                    proposed_episodes[split] * 10
                    - total_episodes * split_weights[split]
                )
                for split in split_names
            )
            # Common-denominator form of equally weighted normalized L1 error.
            score = (
                lineage_error * max(total_episodes, 1)
                + episode_error * max(total_lineages, 1)
            )
            return score, split_names.index(candidate)

        split = min(candidates, key=assignment_score)
        lineage_counts[split] += int(component["lineage_count"])
        episode_counts[split] += int(component["episode_count"])
        component_counts[split] += 1
        for episode in component["episodes"]:
            episode["split"] = split
            episode["split_cluster_id"] = component["cluster_id"]
            episode["split_cluster_lineage_count"] = component["lineage_count"]
            episode["split_cluster_episode_count"] = component["episode_count"]

    return lineage_counts


def _episode_content(episode: dict[str, Any]) -> str:
    parts = [
        episode["initial_field"][region]["text"]
        for region in CANONICAL_REGIONS
    ]
    for tick in episode["ticks"]:
        parts.extend(
            (
                str(tick["target_region"]),
                str(tick["complete_proposed_region_text"]),
            )
        )
    return "\n".join(parts)


def _near_normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def audit_cross_split_leakage(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    near: dict[str, list[dict[str, str]]] = defaultdict(list)
    for episode in episodes:
        content = _episode_content(episode)
        item = {
            "episode_id": str(episode["episode_id"]),
            "lineage": str(episode["source_lineage"]),
            "split": str(episode["split"]),
        }
        exact[sha256_text(content)].append(item)
        near[sha256_text(_near_normalize(content))].append(item)

    def conflicts(index: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for fingerprint, items in sorted(index.items()):
            splits = sorted({item["split"] for item in items})
            if len(splits) > 1:
                result.append(
                    {
                        "fingerprint": fingerprint,
                        "splits": splits,
                        "episodes": items,
                    }
                )
        return result

    exact_conflicts = conflicts(exact)
    near_conflicts = conflicts(near)
    return {
        "policy": {
            "exact": "sha256_full_logical_field_and_targets",
            "near": "sha256_casefold_alphanumeric_collapse",
        },
        "exact_cross_split": exact_conflicts,
        "near_cross_split": near_conflicts,
        "passed": not exact_conflicts and not near_conflicts,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _record_declares_grounded_d00(record: SourceRecord) -> bool:
    return (
        record.adapter == D00_GROUNDED_ADAPTER
        or str(record.row.get("adapter") or "") == D00_GROUNDED_ADAPTER
    )


def _validate_grounded_d00_direct_inputs(
    records: Sequence[SourceRecord],
    source_audit: Mapping[str, Any],
) -> None:
    if source_audit.get("adapter") != D00_GROUNDED_ADAPTER:
        raise CurriculumBuildError(
            "direct grounded D:\\00 audit has the wrong adapter"
        )
    if source_audit.get("passed") is not True:
        raise CurriculumBuildError(
            "direct grounded D:\\00 adapter audit did not pass"
        )
    if source_audit.get("selected_rows") != len(records):
        raise CurriculumBuildError(
            "direct grounded D:\\00 record count does not match its audit"
        )
    if (
        source_audit.get("quota_expected")
        != source_audit.get("selected_by_kind")
    ):
        raise CurriculumBuildError(
            "direct grounded D:\\00 observed quotas do not match expected quotas"
        )
    record_contract = source_audit.get("record_contract", {})
    if (
        not isinstance(record_contract, Mapping)
        or record_contract.get("passed") is not True
        or record_contract.get("quota_observed")
        != record_contract.get("quota_expected")
        or record_contract.get("user_input_preflight_failure_count") != 0
        or record_contract.get("user_input_preflight_count") != len(records)
    ):
        raise CurriculumBuildError(
            "direct grounded D:\\00 record/preflight contract did not pass"
        )
    audit_privacy = source_audit.get("privacy", {})
    if (
        not isinstance(audit_privacy, Mapping)
        or audit_privacy.get("local_only") is not True
        or audit_privacy.get("cloud_export_allowed") is not False
    ):
        raise CurriculumBuildError(
            "direct grounded D:\\00 audit is not private/local-only"
        )
    sources = source_audit.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise CurriculumBuildError(
            "direct grounded D:\\00 audit must name semantic and episodic sources"
        )
    roles = {str(source.get("role")) for source in sources if isinstance(source, Mapping)}
    if roles != {"semantic", "episodic_grounding"}:
        raise CurriculumBuildError(
            "direct grounded D:\\00 audit source roles are incomplete"
        )
    for source in sources:
        if (
            not isinstance(source, Mapping)
            or source.get("read_only_verified") is not True
            or not source.get("sha256_before")
            or source.get("sha256_before") != source.get("sha256_after")
        ):
            raise CurriculumBuildError(
                "direct grounded D:\\00 source hash/read-only audit is invalid"
            )
    for record in records:
        if not _record_declares_grounded_d00(record):
            raise MixedPrivacySourceError(
                "grounded D:\\00 direct records cannot mix with another adapter"
            )
        if (
            record.row.get("local_only") is not True
            or record.row.get("cloud_export_allowed") is not False
        ):
            raise MixedPrivacySourceError(
                "grounded D:\\00 records must be local-only and non-exportable"
            )


def _grounded_d00_source_manifest_records(
    source_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in source_audit["sources"]:
        physical_path = Path(str(source["path"])).resolve()
        if not physical_path.is_file():
            raise FileNotFoundError(physical_path)
        expected = str(source["sha256_after"])
        current = sha256_file(physical_path)
        if current != expected:
            raise SourceMutationError(
                "grounded D:\\00 source changed after adapter extraction: "
                f"{physical_path}"
            )
        result.append(
            {
                "kind": "sqlite",
                "role": str(source["role"]),
                "path": str(physical_path),
                "adapter": D00_GROUNDED_ADAPTER,
                "table": (
                    "semantic extracted tables"
                    if source["role"] == "semantic"
                    else "episodes.extracted_json"
                ),
                "query": "direct exact-grounded adapter records",
                "sha256_before": str(source["sha256_before"]),
                "sha256_after": current,
                "read_only_verified": True,
                "rows_seen": None,
                "episodes_emitted": None,
                "dependency_only": True,
            }
        )
    return result


def build_curriculum(
    sources: Sequence[SourceSpec],
    output_dir: str | Path,
    *,
    seed: int = 42,
    unsupported_policy: str = "reject",
    include_diary: bool = False,
    fail_on_leakage: bool = True,
    direct_records: Sequence[SourceRecord] | None = None,
    direct_source_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build exact-v4 episodes without mutating any source."""

    if unsupported_policy not in {"reject", "escape"}:
        raise ValueError("unsupported_policy must be reject or escape")
    if direct_records is not None and sources:
        raise MixedPrivacySourceError(
            "direct grounded D:\\00 records cannot mix with SourceSpec inputs"
        )
    if direct_records is None and direct_source_audit is not None:
        raise ValueError("direct_source_audit requires direct_records")
    if direct_records is not None and direct_source_audit is None:
        raise ValueError("direct_records require direct_source_audit")
    if direct_records is None and any(
        spec.adapter == D00_GROUNDED_ADAPTER for spec in sources
    ):
        raise MixedPrivacySourceError(
            "grounded D:\\00 adapter requires the direct-record build path"
        )
    episodes: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    source_records: list[dict[str, Any]] = []

    def consume_record(record: SourceRecord) -> bool:
        if direct_records is None and _record_declares_grounded_d00(record):
            raise MixedPrivacySourceError(
                "private grounded D:\\00 rows cannot enter an explicit "
                "generic/cloud-exportable source build"
            )
        try:
            episode, reason = build_episode(
                record,
                unsupported_policy=unsupported_policy,
                include_diary=include_diary,
            )
        except UnsupportedSourceText:
            skipped["unsupported_source_text"] += 1
            return False
        except SealedTargetError:
            raise
        except (CurriculumBuildError, ValueError, TypeError) as exc:
            skipped[f"invalid_source_row:{type(exc).__name__}"] += 1
            return False
        if episode is None:
            skipped[str(reason or "not_emitted")] += 1
            return False
        episodes.append(episode)
        if episode["privacy"]["diary_exclusion_recorded"]:
            skipped["source_diary_explicitly_excluded"] += 1
        return True

    if direct_records is not None:
        direct_records = tuple(direct_records)
        assert direct_source_audit is not None
        _validate_grounded_d00_direct_inputs(
            direct_records,
            direct_source_audit,
        )
        for record in direct_records:
            consume_record(record)
        source_records = _grounded_d00_source_manifest_records(
            direct_source_audit
        )
        for source_record in source_records:
            source_record["rows_seen"] = len(direct_records)
            source_record["episodes_emitted"] = len(episodes)
        if len(episodes) != len(direct_records) or skipped:
            raise CurriculumBuildError(
                "grounded D:\\00 direct records must emit losslessly "
                f"(records={len(direct_records)} episodes={len(episodes)} "
                f"skipped={dict(sorted(skipped.items()))})"
            )
    else:
        for spec in sources:
            physical_path = _physical_source_path(spec)
            if not physical_path.exists():
                raise FileNotFoundError(physical_path)
            before_sha256 = sha256_file(physical_path)
            rows_seen = 0
            emitted = 0
            for record in iter_source_records(spec, before_sha256):
                rows_seen += 1
                emitted += int(consume_record(record))

            after_sha256 = sha256_file(physical_path)
            if after_sha256 != before_sha256:
                raise SourceMutationError(
                    f"source changed during read-only build: {physical_path}"
                )
            source_records.append(
                {
                    "kind": spec.kind,
                    "path": str(physical_path),
                    "adapter": spec.adapter,
                    "table": spec.table,
                    "query": spec.query,
                    "sha256_before": before_sha256,
                    "sha256_after": after_sha256,
                    "read_only_verified": True,
                    "rows_seen": rows_seen,
                    "episodes_emitted": emitted,
                }
            )

    split_lineages = assign_source_disjoint_splits(episodes, seed=seed)
    leakage = audit_cross_split_leakage(episodes)
    if fail_on_leakage and not leakage["passed"]:
        raise CrossSplitLeakageError(
            "cross-split duplicate leakage detected: "
            f"exact={len(leakage['exact_cross_split'])} "
            f"near={len(leakage['near_cross_split'])}"
        )

    split_order = {"train": 0, "dev": 1, "test": 2}
    episodes.sort(
        key=lambda episode: (
            split_order.get(str(episode["split"]), 99),
            str(episode["family"]),
            str(episode["source_lineage"]),
            str(episode["episode_id"]),
        )
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, dict[str, Any]] = {}
    for family in SUPPORTED_FAMILIES:
        rows = [episode for episode in episodes if episode["family"] == family]
        if not rows:
            continue
        path = output / f"{family}.jsonl"
        _write_jsonl(path, rows)
        output_files[family] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": sha256_file(path),
        }

    all_path = output / "episodes.jsonl"
    _write_jsonl(all_path, episodes)
    output_files["all"] = {
        "path": str(all_path),
        "rows": len(episodes),
        "sha256": sha256_file(all_path),
    }

    artifact_any_local_only = any(
        bool(episode["local_only"]) for episode in episodes
    )
    artifact_all_local_only = bool(episodes) and all(
        bool(episode["local_only"]) for episode in episodes
    )
    artifact_cloud_export_allowed = bool(episodes) and all(
        bool(episode["privacy"]["cloud_export_allowed"])
        for episode in episodes
    )
    manifest = {
        "schema": "axon_multitick_curriculum_manifest_v1",
        "builder_version": BUILDER_VERSION,
        "seed": seed,
        "source_root_default": str(DEFAULT_SOURCE_ROOT),
        "source_contract": "read_only",
        "source_mode": (
            "grounded_d00_direct_records"
            if direct_records is not None
            else "explicit_source_specs"
        ),
        "unsupported_policy": unsupported_policy,
        "privacy": {
            "any_rows_local_only": artifact_any_local_only,
            "all_rows_local_only": artifact_all_local_only,
            "cloud_export_allowed": artifact_cloud_export_allowed,
            "mixed_private_and_exportable_rows": bool(
                artifact_any_local_only and not artifact_all_local_only
            ),
        },
        "diary": {
            "included": include_diary,
            "default": "excluded",
            "included_rows_are_local_only": True,
            "cloud_export_allowed": artifact_cloud_export_allowed,
        },
        "canonical_regions": list(CANONICAL_REGIONS),
        "situation_awareness_contract": (
            "top_level_read_only_logical_region_in_dynamic_context_projection"
        ),
        "field_profile_status": "runtime_three_role_384_parity",
        "writable_readback_gate_passed": False,
        "read_coverage_launch_gate": {
            "passed": False,
            "blocking_reason": (
                "writable-region post-commit whole-field readback is not "
                "guaranteed by the fixed four-output-tick protocol"
            ),
            "target_supervision_lossless": all(
                bool(
                    episode["read_coverage_contract"][
                        "transaction_target_coverage"
                    ]["each_target_character_supervised_exactly_once"]
                )
                for episode in episodes
            ),
            "artifact_launch_eligible": False,
        },
        "segment_protocol": {
            "name": "fixed_four_by_sixty_four_v1",
            "segment_count": SEGMENT_COUNT,
            "segment_width": PROPOSAL_WIDTH,
            "region_target_budget": REGION_TARGET_BUDGET,
            "targets_over_budget": "explicitly_skipped",
        },
        "split_policy": (
            "source_lineage_plus_transitive_exact_and_normalized_near_components"
        ),
        "split_cluster_count": len(
            {str(episode["split_cluster_id"]) for episode in episodes}
        ),
        "sources": source_records,
        "episodes": len(episodes),
        "family_counts": {
            family: sum(episode["family"] == family for episode in episodes)
            for family in SUPPORTED_FAMILIES
        },
        "split_counts": {
            split: sum(episode["split"] == split for episode in episodes)
            for split in ("train", "dev", "test")
        },
        "split_lineage_counts": split_lineages,
        "skipped": dict(sorted(skipped.items())),
        "leakage_audit": leakage,
        "evaluation_requirements": {
            "teacher_forced": True,
            "free_running": True,
            "counterfactual_regions": True,
            "private_state_claim": False,
        },
        "output_files": output_files,
    }
    if direct_source_audit is not None:
        manifest["grounded_d00_adapter_audit"] = copy.deepcopy(
            dict(direct_source_audit)
        )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_curriculum_from_records(
    records: Sequence[SourceRecord],
    output_dir: str | Path,
    *,
    source_audit: Mapping[str, Any],
    seed: int = 42,
    unsupported_policy: str = "reject",
    fail_on_leakage: bool = True,
) -> dict[str, Any]:
    """Build directly from an audited grounded D:\\00 record set.

    This path intentionally accepts no ``SourceSpec`` inputs and never
    serializes an intermediate JSONL source.  It is private/local-only.
    """

    return build_curriculum(
        (),
        output_dir,
        seed=seed,
        unsupported_policy=unsupported_policy,
        include_diary=False,
        fail_on_leakage=fail_on_leakage,
        direct_records=tuple(records),
        direct_source_audit=source_audit,
    )


def build_grounded_d00_curriculum(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    total_limit: int = 8192,
    seed: int = 42,
    unsupported_policy: str = "reject",
    fail_on_leakage: bool = True,
) -> dict[str, Any]:
    """Extract and build the exact-grounded private D:\\00 curriculum."""

    from training.d00_field_sources import (  # local import keeps CLI additive
        extract_d00_field_source_records,
    )

    extraction = extract_d00_field_source_records(
        source_root,
        total_limit=total_limit,
        verify_source_hashes=True,
    )
    if extraction.audit.get("passed") is not True:
        raise CurriculumBuildError(
            "grounded D:\\00 adapter audit failed; artifact build refused"
        )
    records = tuple(
        SourceRecord(**kwargs)
        for kwargs in extraction.source_record_kwargs
    )
    return build_curriculum_from_records(
        records,
        output_dir,
        source_audit=extraction.audit,
        seed=seed,
        unsupported_policy=unsupported_policy,
        fail_on_leakage=fail_on_leakage,
    )


def _parse_sqlite_spec(value: str) -> SourceSpec:
    if "::" not in value:
        raise argparse.ArgumentTypeError("--sqlite requires PATH::TABLE")
    path, table = value.rsplit("::", 1)
    return SourceSpec.sqlite(path, table=table)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--jsonl", action="append", default=[])
    parser.add_argument("--sqlite", action="append", default=[], metavar="PATH::TABLE")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--grounded-d00-limit",
        type=int,
        default=8192,
        help=(
            "bounded 45/45/10 grounded D:\\00 rows used only when neither "
            "--jsonl nor --sqlite is supplied"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unsupported-policy", choices=("reject", "escape"), default="reject")
    parser.add_argument("--include-diary", action="store_true")
    parser.add_argument("--allow-cross-split-leakage", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = [SourceSpec.jsonl(path) for path in args.jsonl]
    sources.extend(_parse_sqlite_spec(value) for value in args.sqlite)
    if not sources:
        if args.include_diary:
            raise CurriculumBuildError(
                "default grounded D:\\00 build excludes the manual-only "
                "personal log; --include-diary requires an explicit source"
            )
        manifest = build_grounded_d00_curriculum(
            args.source_root,
            args.output_dir,
            total_limit=args.grounded_d00_limit,
            seed=args.seed,
            unsupported_policy=args.unsupported_policy,
            fail_on_leakage=not args.allow_cross_split_leakage,
        )
    else:
        manifest = build_curriculum(
            sources,
            args.output_dir,
            seed=args.seed,
            unsupported_policy=args.unsupported_policy,
            include_diary=args.include_diary,
            fail_on_leakage=not args.allow_cross_split_leakage,
        )
    print(
        f"built {manifest['episodes']} episodes -> {args.output_dir} "
        f"(train/dev/test={manifest['split_counts']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

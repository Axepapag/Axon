#!/usr/bin/env python3
"""Build causal soul write-delay-recall episodes from recovered exact material.

The builder is additive and read-only.  Tick A exposes an exact fact/evidence
projection and carries explicit soul-write supervision.  One or more delay
ticks intervene.  The final recall tick is compiled through the canonical
``runtime.field`` contract and is rejected if any answer-bearing source text is
visible in any active FieldView region.

Source lineages connected by exact or normalized-near answer-bearing content
are clustered transitively before deterministic splitting.  Distinct clusters
are paired into same-split donor groups so correct/zero/swapped/shuffled soul
conditions never require a held-out donor.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.field import (  # noqa: E402
    CANONICAL_REGION_ORDER,
    CONTEXT_END,
    CONTEXT_START,
    CORE_WRITABLE_REGIONS,
    N_SLOTS,
    PROPOSAL_END,
    PROPOSAL_START,
    USER_END,
    USER_START,
    LogicalRegion,
    PhysicalRole,
    SharedFieldSnapshot,
    SlotKind,
    compile_field_view,
    proposal_payload_capacity,
)
from substrate import assert_supported_text  # noqa: E402
from training.build_multitick_curriculum import (  # noqa: E402
    DEFAULT_SOURCE_ROOT,
    ExactTextEncoder,
    SourceMutationError,
    SourceRecord,
    SourceSpec,
    UnsupportedSourceText,
    canonical_json_bytes,
    discover_primary_sources,
    iter_source_records,
    sha256_bytes,
    sha256_file,
    sha256_text,
    sqlite_ro_uri,
    _sqlite_path_from_uri,
)
from training.d00_soul_sources import (  # noqa: E402
    D00GroundingError,
    SOUL_WRITE_INSTRUCTION,
    discover_d00_private_soul_sources,
    is_d00_private_adapter,
    render_soul_memory_text,
    select_grounded_records,
)


SCHEMA = "axon_soul_write_delay_exact_v1"
MANIFEST_SCHEMA = "axon_soul_write_delay_manifest_v1"
BUILDER_VERSION = "1"
FAMILY = "soul_write_delay_recall_v1"
COUNTERFACTUAL_CONDITIONS = ("correct", "zero", "swapped", "shuffled")
WRITABLE_REGIONS = frozenset(
    region.value for region in CORE_WRITABLE_REGIONS
)
PHYSICAL_PROFILE = (
    ("context_projection", CONTEXT_START, CONTEXT_END, int(PhysicalRole.CONTEXT)),
    ("user_input", USER_START, USER_END, int(PhysicalRole.USER)),
    ("proposal", PROPOSAL_START, PROPOSAL_END, int(PhysicalRole.PROPOSAL)),
)
_MISSING = object()


class SoulWriteDelayBuildError(RuntimeError):
    """Base error for fail-closed write-delay curriculum construction."""


class FieldViewContractError(SoulWriteDelayBuildError):
    """Raised when required exact text is not fully present in a compiled view."""


class CausalLeakageError(SoulWriteDelayBuildError):
    """Raised when recall or delay input exposes answer-bearing source text."""


class DonorSeparationError(SoulWriteDelayBuildError):
    """Raised when an identity-separated same-split donor cannot be constructed."""


class CrossSplitLeakageError(SoulWriteDelayBuildError):
    """Raised when duplicate-connected source material crosses splits."""


class EmptyCurriculumError(SoulWriteDelayBuildError):
    """Raised when no eligible causal episodes remain."""


class CurriculumDiversityError(SoulWriteDelayBuildError):
    """Raised when a production build lacks causal identity diversity."""


@dataclass
class _Candidate:
    episode: dict[str, Any]
    fingerprint_text: str
    fact_sha256: str

    @property
    def episode_id(self) -> str:
        return str(self.episode["episode_id"])

    @property
    def lineage(self) -> str:
        return str(self.episode["source_lineage"])


def _render_source_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _get_path(row: dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _first_text(
    row: dict[str, Any],
    paths: Sequence[str],
) -> tuple[str | None, str]:
    for path in paths:
        value = _get_path(row, path)
        if value is _MISSING or value is None:
            continue
        text = _render_source_value(value)
        if text:
            return text, path
    return None, ""


def _field_cell_text(row: dict[str, Any], region: str) -> str:
    value = _get_path(row, f"initial_field.{region}")
    if not isinstance(value, dict):
        return ""
    return _render_source_value(value.get("text"))


def _source_lineage(row: dict[str, Any], record: SourceRecord) -> str:
    value, _ = _first_text(
        row,
        (
            "source_lineage",
            "lineage_id",
            "session_id",
            "input_event.session_id",
            "episode_id",
            "container_id",
            "source_id",
        ),
    )
    suffix = value or record.pointer
    return f"{record.source_sha256}:{suffix}"


def _last_tick_target(row: dict[str, Any]) -> str | None:
    ticks = row.get("ticks")
    if not isinstance(ticks, list):
        return None
    for tick in reversed(ticks):
        if not isinstance(tick, dict):
            continue
        for key in (
            "complete_proposed_region_text",
            "proposed_region_text",
            "expected_answer",
            "text",
        ):
            if key in tick and tick[key] is not None:
                value = _render_source_value(tick[key])
                if value:
                    return value
    return None


def _answer_bearing_source_texts(
    row: dict[str, Any],
    *,
    fact: str,
    evidence: str,
) -> list[str]:
    values = [fact, evidence]
    explicit = row.get("answer_bearing_texts")
    if explicit is None:
        explicit = _get_path(row, "soul_write_delay.answer_bearing_texts")
    if isinstance(explicit, list):
        values.extend(_render_source_value(value) for value in explicit)
    elif explicit not in (_MISSING, None):
        values.append(_render_source_value(explicit))
    ticks = row.get("ticks")
    if isinstance(ticks, list):
        for tick in ticks:
            if not isinstance(tick, dict):
                continue
            for key in (
                "complete_proposed_region_text",
                "proposed_region_text",
                "expected_answer",
            ):
                if key in tick and tick[key] is not None:
                    values.append(_render_source_value(tick[key]))
    return list(dict.fromkeys(value for value in values if value))


def _diary_text(row: dict[str, Any]) -> str:
    for path in (
        "diary_text",
        "diary",
        "pre_state.diary_entries",
    ):
        value = _get_path(row, path)
        if value not in (_MISSING, None):
            text = _render_source_value(value)
            if text:
                return text
    return _field_cell_text(row, "diary")


def _extract_source_spec(
    row: dict[str, Any],
    record: SourceRecord,
) -> dict[str, Any] | None:
    fact, fact_origin = _first_text(
        row,
        (
            "soul_write_delay.fact_text",
            "fact_text",
            "expected_answer",
            "answer_text",
            "training_targets.positive",
            "target_text",
            "target",
            "answer",
        ),
    )
    if fact is None:
        fact = _last_tick_target(row)
        fact_origin = "ticks[-1].complete_proposed_region_text" if fact else ""
    if fact is None:
        return None

    evidence, evidence_origin = _first_text(
        row,
        (
            "soul_write_delay.evidence_text",
            "evidence_text",
            "structured_knowledge",
            "evidence",
            "knowledge",
            "tool_results",
            "tool_result",
        ),
    )
    if evidence is None:
        for region in (
            "structured_knowledge",
            "tool_results",
            "advisor_input",
            "conversation_history",
        ):
            evidence = _field_cell_text(row, region)
            if evidence:
                evidence_origin = f"initial_field.{region}.text"
                break
    if evidence is None:
        evidence = fact
        evidence_origin = fact_origin

    query, query_origin = _first_text(
        row,
        (
            "soul_write_delay.recall_query",
            "recall_query",
            "probe_query",
        ),
    )
    raw_distractors = _get_path(row, "soul_write_delay.distractors")
    if raw_distractors is _MISSING:
        raw_distractors = row.get("distractors", [])
    distractors: list[str] = []
    if isinstance(raw_distractors, list):
        for item in raw_distractors:
            if isinstance(item, dict):
                item = item.get("text", item.get("user_input", ""))
            text = _render_source_value(item)
            if text:
                distractors.append(text)
    elif raw_distractors not in (None, _MISSING):
        text = _render_source_value(raw_distractors)
        if text:
            distractors.append(text)

    diary = _diary_text(row)
    return {
        "fact": fact,
        "fact_origin": fact_origin,
        "evidence": evidence,
        "evidence_origin": evidence_origin,
        "query": query,
        "query_origin": query_origin,
        "distractors": distractors,
        "diary": diary,
        "lineage": _source_lineage(row, record),
        "answer_bearing": _answer_bearing_source_texts(
            row,
            fact=fact,
            evidence=evidence,
        ),
        "upstream_schema": str(row.get("schema") or ""),
        "upstream_episode_id": str(row.get("episode_id") or ""),
    }


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _contains_answer_bearing(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    if needle.casefold() in haystack.casefold():
        return True
    normalized_needle = _normalized(needle)
    normalized_haystack = _normalized(haystack)
    return (
        len(normalized_needle) >= 4
        and normalized_needle in normalized_haystack
    )


def _safe_generated_text(
    *,
    cue: str,
    kind: str,
    index: int,
    answer_bearing: Sequence[str],
) -> str:
    candidates = (
        f"Q {cue}?" if kind == "query" else f"D {cue} {index}.",
        f"R {cue}?" if kind == "query" else f"X {cue} {index}.",
        f"{cue}?" if kind == "query" else f"{cue} {index}.",
    )
    for candidate in candidates:
        if not any(
            _contains_answer_bearing(candidate, source_text)
            for source_text in answer_bearing
        ):
            return candidate
    raise CausalLeakageError(
        f"cannot generate a fact-free {kind} for cue {cue}"
    )


def _field_snapshot(
    texts: dict[str, str],
    *,
    tick_id: int,
    record: SourceRecord,
) -> SharedFieldSnapshot:
    for text in texts.values():
        assert_supported_text(text)
    return SharedFieldSnapshot.from_texts(
        texts,
        tick_id=tick_id,
        source_manifest_ids=(record.source_sha256,),
        source=record.source_path,
        provenance=record.pointer,
    )


def _visible_region_texts(view: Any) -> dict[str, str]:
    parts: dict[str, list[str]] = {
        region.value: [] for region in CANONICAL_REGION_ORDER
    }
    for ref, active in zip(
        view.slot_refs,
        view.attention_mask.tolist(),
        strict=True,
    ):
        if (
            active
            and ref.kind is SlotKind.SPAN
            and ref.logical_region is not None
            and ref.rendered_char is not None
        ):
            parts[ref.logical_region.value].append(ref.rendered_char)
    return {region: "".join(chars) for region, chars in parts.items()}


def _view_audit(snapshot: SharedFieldSnapshot) -> tuple[Any, dict[str, Any]]:
    view = compile_field_view(
        snapshot,
        proposal_region=LogicalRegion.RESPONSE_DRAFT,
    )
    visible = _visible_region_texts(view)
    active_refs = [
        ref.to_canonical_dict()
        for ref, active in zip(
            view.slot_refs,
            view.attention_mask.tolist(),
            strict=True,
        )
        if active
    ]
    audit = {
        "source_field_id": snapshot.field_id,
        "proposal_region": view.proposal_region.value,
        "physical_geometry": [
            {
                "name": name,
                "slot_start": start,
                "slot_end": end,
                "width": end - start,
                "physical_type_id": type_id,
            }
            for name, start, end, type_id in PHYSICAL_PROFILE
        ],
        "visible_region_texts": visible,
        "visible_region_sha256": {
            region: sha256_text(text) for region, text in visible.items()
        },
        "active_slot_ref_sha256": sha256_bytes(
            canonical_json_bytes(active_refs)
        ),
        "active_slots": int(view.attention_mask.sum()),
        "write_slots": int(view.write_mask.sum()),
        "omissions": [
            omission.to_canonical_dict() for omission in view.omissions
        ],
    }
    return view, audit


def _assert_fully_visible(
    snapshot: SharedFieldSnapshot,
    audit: dict[str, Any],
    region: str,
) -> None:
    expected = snapshot.region(region).text
    observed = audit["visible_region_texts"][region]
    if observed != expected:
        raise FieldViewContractError(
            f"{region} is not fully visible in the active FieldView "
            f"(expected={len(expected)} observed={len(observed)})"
        )


def _leakage_audit(
    visible_region_texts: dict[str, str],
    answer_bearing: Sequence[str],
) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for region, visible_text in visible_region_texts.items():
        for source_text in answer_bearing:
            if _contains_answer_bearing(visible_text, source_text):
                violations.append(
                    {
                        "region": region,
                        "visible_text_sha256": sha256_text(visible_text),
                        "answer_bearing_sha256": sha256_text(source_text),
                    }
                )
    return {
        "policy": (
            "casefold_exact_substring_plus_normalized_alphanumeric_substring"
        ),
        "checked_regions": [
            region.value for region in CANONICAL_REGION_ORDER
        ],
        "answer_bearing_sha256": sorted(
            {sha256_text(text) for text in answer_bearing}
        ),
        "violations": violations,
        "passed": not violations,
    }


def _active_field_payload(snapshot: SharedFieldSnapshot) -> dict[str, str]:
    return {
        region.value: snapshot.region(region).text
        for region in CANONICAL_REGION_ORDER
    }


def _build_candidate(
    record: SourceRecord,
    *,
    unsupported_policy: str,
    include_diary: bool,
    min_delay_ticks: int,
) -> _Candidate | None:
    source = _extract_source_spec(record.row, record)
    if source is None:
        return None

    encoder = ExactTextEncoder(unsupported_policy)
    fact = encoder.encode(source["fact"], "fact_text")
    evidence = encoder.encode(source["evidence"], "evidence_text")
    answer_bearing = [
        encoder.encode(text, f"answer_bearing_texts[{index}]")
        for index, text in enumerate(source["answer_bearing"])
    ]
    answer_bearing = list(dict.fromkeys(answer_bearing + [fact, evidence]))
    if not fact:
        return None
    target_capacity = proposal_payload_capacity(
        LogicalRegion.RESPONSE_DRAFT
    )
    if len(fact) > target_capacity:
        raise FieldViewContractError(
            f"encoded answer length {len(fact)} exceeds proposal width "
            f"{target_capacity} after the runtime-rendered region tag"
        )

    cue = record.row_sha256[:12].lower()
    if source["query"] is None:
        query = _safe_generated_text(
            cue=cue,
            kind="query",
            index=0,
            answer_bearing=answer_bearing,
        )
        query_origin = "generated_fact_free_query"
    else:
        query = encoder.encode(source["query"], "recall_query")
        query_origin = source["query_origin"]

    distractors = [
        encoder.encode(text, f"distractors[{index}]")
        for index, text in enumerate(source["distractors"])
    ]
    while len(distractors) < min_delay_ticks:
        distractors.append(
            _safe_generated_text(
                cue=cue,
                kind="delay",
                index=len(distractors),
                answer_bearing=answer_bearing,
            )
        )
    diary_source = str(source["diary"])
    diary = (
        encoder.encode(diary_source, "diary")
        if include_diary and diary_source
        else ""
    )
    private_source = is_d00_private_adapter(record.adapter)

    memory_text = render_soul_memory_text(fact, evidence)
    assert_supported_text(memory_text)
    episode_identity = {
        "record_pointer": record.pointer,
        "row_sha256": record.row_sha256,
        "lineage": source["lineage"],
        "fact_sha256": sha256_text(fact),
        "evidence_sha256": sha256_text(evidence),
    }
    episode_id = (
        "swd-"
        + sha256_bytes(canonical_json_bytes(episode_identity))[:24].lower()
    )

    tick_a_texts = {
        region.value: "" for region in CANONICAL_REGION_ORDER
    }
    tick_a_texts["user_input"] = SOUL_WRITE_INSTRUCTION
    tick_a_texts["structured_knowledge"] = memory_text
    tick_a_texts["diary"] = diary
    tick_a_snapshot = _field_snapshot(
        tick_a_texts,
        tick_id=0,
        record=record,
    )
    _, tick_a_view = _view_audit(tick_a_snapshot)
    _assert_fully_visible(
        tick_a_snapshot,
        tick_a_view,
        "structured_knowledge",
    )
    if diary:
        _assert_fully_visible(tick_a_snapshot, tick_a_view, "diary")
    if fact not in tick_a_view["visible_region_texts"]["structured_knowledge"]:
        raise FieldViewContractError("tick A does not visibly contain the fact")
    if evidence not in tick_a_view["visible_region_texts"]["structured_knowledge"]:
        raise FieldViewContractError("tick A does not visibly contain the evidence")

    delay_ticks: list[dict[str, Any]] = []
    for index, distractor in enumerate(distractors):
        delay_texts = {
            region.value: "" for region in CANONICAL_REGION_ORDER
        }
        delay_texts["user_input"] = distractor
        snapshot = _field_snapshot(
            delay_texts,
            tick_id=index + 1,
            record=record,
        )
        _, view_audit = _view_audit(snapshot)
        _assert_fully_visible(snapshot, view_audit, "user_input")
        leak = _leakage_audit(
            view_audit["visible_region_texts"],
            answer_bearing,
        )
        if not leak["passed"]:
            raise CausalLeakageError(
                f"{episode_id} delay tick {index} exposes answer-bearing text"
            )
        delay_ticks.append(
            {
                "tick_id": f"{episode_id}:delay:{index}",
                "tick_index": index + 1,
                "kind": "distractor_delay",
                "active_field": _active_field_payload(snapshot),
                "canonical_field_sha256": snapshot.canonical_hash.upper(),
                "active_view": view_audit,
                "answer_leakage_audit": leak,
                "soul_transition": "preserve_written_state",
            }
        )

    tick_b_texts = {
        region.value: "" for region in CANONICAL_REGION_ORDER
    }
    tick_b_texts["user_input"] = query
    tick_b_snapshot = _field_snapshot(
        tick_b_texts,
        tick_id=len(delay_ticks) + 1,
        record=record,
    )
    _, tick_b_view = _view_audit(tick_b_snapshot)
    _assert_fully_visible(tick_b_snapshot, tick_b_view, "user_input")
    tick_b_leak = _leakage_audit(
        tick_b_view["visible_region_texts"],
        answer_bearing,
    )
    if not tick_b_leak["passed"]:
        raise CausalLeakageError(
            f"{episode_id} recall FieldView exposes answer-bearing source text"
        )

    write_target_id = (
        "write-"
        + sha256_text(f"{episode_id}|{sha256_text(fact)}")[:24].lower()
    )
    tick_b_id = f"{episode_id}:recall"
    tick_a = {
        "tick_id": f"{episode_id}:write",
        "tick_index": 0,
        "kind": "fact_visible_soul_write",
        "active_field": _active_field_payload(tick_a_snapshot),
        "canonical_field_sha256": tick_a_snapshot.canonical_hash.upper(),
        "active_view": tick_a_view,
        "fact_region": "structured_knowledge",
        "evidence_region": "structured_knowledge",
        "soul_write_target": {
            "write_target_id": write_target_id,
            "target_kind": "exact_text_trace_to_authoritative_hot_tier",
            "target_soul_trace": fact,
            "target_soul_trace_sha256": sha256_text(fact),
            "write_tier": "hot",
            "write_credit": {
                "immediate_write_weight": 1.0,
                "delayed_recall_weight": 1.0,
                "credit_assignment_tick_id": tick_b_id,
                "intervening_delay_ticks": len(delay_ticks),
            },
        },
    }
    tick_b = {
        "tick_id": tick_b_id,
        "tick_index": len(delay_ticks) + 1,
        "kind": "soul_only_recall",
        "active_field": _active_field_payload(tick_b_snapshot),
        "canonical_field_sha256": tick_b_snapshot.canonical_hash.upper(),
        "active_view": tick_b_view,
        "query_origin": query_origin,
        "answer_leakage_audit": tick_b_leak,
        "recall_supervision": {
            "target_region": "response_draft",
            "complete_expected_answer": fact,
            "expected_answer_sha256": sha256_text(fact),
            "requires_soul_state": True,
            "field_answer_available": False,
            "credit_write_target_id": write_target_id,
        },
    }

    fingerprint_text = "\n".join((fact, evidence))
    episode = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "family": FAMILY,
        "episode_id": episode_id,
        "source_lineage": source["lineage"],
        "split": "",
        "lineage_cluster_id": "",
        "donor_group_id": "",
        "provenance": {
            "source_kind": record.pointer.split(":", 1)[0],
            "source_path": record.source_path,
            "source_sha256": record.source_sha256,
            "record_pointer": record.pointer,
            "row_sha256": record.row_sha256,
            "adapter": record.adapter,
            "upstream_schema": source["upstream_schema"],
            "upstream_episode_id": source["upstream_episode_id"],
            "fact_origin": source["fact_origin"],
            "evidence_origin": source["evidence_origin"],
            "text_transform_audits": encoder.audits,
        },
        "privacy": {
            "diary_source_present": bool(diary_source),
            "diary_included": bool(diary),
            "diary_excluded": bool(diary_source and not include_diary),
            "private_d00_source": private_source,
            "local_only": bool(diary) or private_source,
            "cloud_export_allowed": not (bool(diary) or private_source),
        },
        "exact_text_contract": {
            "unsupported_policy": unsupported_policy,
            "lossless_utf8_source_hashes": True,
            "source_fact_utf8_sha256": sha256_text(source["fact"]),
            "source_evidence_utf8_sha256": sha256_text(source["evidence"]),
            "encoded_fact_sha256": sha256_text(fact),
            "encoded_evidence_sha256": sha256_text(evidence),
        },
        "answer_bearing_source_sha256": sorted(
            {sha256_text(text) for text in answer_bearing}
        ),
        "runtime_field_contract": {
            "canonical_regions": [
                region.value for region in CANONICAL_REGION_ORDER
            ],
            "writable_regions": sorted(WRITABLE_REGIONS),
            "proposal_region": "response_draft",
            "physical_geometry": [
                {
                    "name": name,
                    "slot_start": start,
                    "slot_end": end,
                    "physical_type_id": type_id,
                }
                for name, start, end, type_id in PHYSICAL_PROFILE
            ],
            "total_slots": N_SLOTS,
        },
        "tick_a": tick_a,
        "delay_ticks": delay_ticks,
        "tick_b": tick_b,
        "counterfactuals": {},
    }
    if private_source:
        if (
            record.row.get("local_only") is not True
            or record.row.get("cloud_export_allowed") is not False
        ):
            raise D00GroundingError(
                "D00 source row lacks explicit local-only privacy"
            )
        episode["provenance"]["d00_grounding"] = {
            key: record.row[key]
            for key in (
                "memory_kind",
                "answer_component_sha256",
                "semantic_table",
                "semantic_row_id",
                "semantic_source",
                "semantic_created_at",
                "grounding_episode_id",
                "grounding_item_index",
                "grounding_item_sha256",
                "grounding_assertion_sha256",
                "exact_episodic_grounding",
                "field_capacity_audit",
            )
            if key in record.row
        }
    return _Candidate(
        episode=episode,
        fingerprint_text=fingerprint_text,
        fact_sha256=sha256_text(fact),
    )


def _cluster_candidates(
    candidates: Sequence[_Candidate],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    by_lineage: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_lineage[candidate.lineage].append(candidate)
    lineages = sorted(by_lineage)
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
        for candidate in sorted(
            by_lineage[lineage],
            key=lambda item: item.episode_id,
        ):
            fingerprints = (
                (exact_owner, sha256_text(candidate.fingerprint_text)),
                (
                    near_owner,
                    sha256_text(_normalized(candidate.fingerprint_text)),
                ),
            )
            for owners, fingerprint in fingerprints:
                owner = owners.setdefault(fingerprint, lineage)
                union(lineage, owner)

    grouped: dict[str, list[str]] = defaultdict(list)
    for lineage in lineages:
        grouped[find(lineage)].append(lineage)
    components: list[dict[str, Any]] = []
    for members in grouped.values():
        members.sort()
        component_candidates = [
            candidate
            for lineage in members
            for candidate in by_lineage[lineage]
        ]
        cluster_id = (
            "lineage-cluster-"
            + sha256_text("|".join(members))[:16].lower()
        )
        components.append(
            {
                "cluster_id": cluster_id,
                "lineages": members,
                "candidates": component_candidates,
                "fact_hashes": {
                    candidate.fact_sha256
                    for candidate in component_candidates
                },
                "order_hash": sha256_text(f"{seed}|{cluster_id}"),
            }
        )
    components.sort(key=lambda item: str(item["order_hash"]))
    return components


def _make_donor_groups(
    components: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if len(components) < 2:
        raise DonorSeparationError(
            "at least two duplicate-independent lineage clusters are required"
        )
    remaining = list(components)
    groups: list[list[dict[str, Any]]] = []
    while len(remaining) >= 2:
        first = remaining.pop(0)
        partner_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if len(first["fact_hashes"] | candidate["fact_hashes"]) >= 2
            ),
            None,
        )
        if partner_index is None:
            raise DonorSeparationError(
                f"no distinct-fact donor cluster for {first['cluster_id']}"
            )
        partner = remaining.pop(partner_index)
        groups.append([first, partner])
    if remaining:
        leftover = remaining.pop()
        group_index = next(
            (
                index
                for index, group in enumerate(groups)
                if len(
                    set().union(
                        leftover["fact_hashes"],
                        *(component["fact_hashes"] for component in group),
                    )
                )
                >= 2
            ),
            None,
        )
        if group_index is None:
            raise DonorSeparationError(
                f"cannot place leftover donor cluster {leftover['cluster_id']}"
            )
        groups[group_index].append(leftover)

    result: list[dict[str, Any]] = []
    for group in groups:
        cluster_ids = sorted(str(item["cluster_id"]) for item in group)
        group_id = (
            "donor-group-"
            + sha256_text("|".join(cluster_ids))[:16].lower()
        )
        candidates = [
            candidate
            for component in group
            for candidate in component["candidates"]
        ]
        result.append(
            {
                "group_id": group_id,
                "components": group,
                "candidates": candidates,
                "lineage_count": sum(
                    len(component["lineages"]) for component in group
                ),
                "episode_count": len(candidates),
                "order_hash": sha256_text(f"{seed}|{group_id}"),
            }
        )
    result.sort(
        key=lambda group: (
            -int(group["lineage_count"]),
            -int(group["episode_count"]),
            str(group["order_hash"]),
        )
    )
    return result


def _assign_donor_groups(
    groups: Sequence[dict[str, Any]],
) -> dict[str, int]:
    split_names = ("train", "dev", "test")
    split_weights = {"train": 8, "dev": 1, "test": 1}
    lineage_counts = {split: 0 for split in split_names}
    episode_counts = {split: 0 for split in split_names}
    group_counts = {split: 0 for split in split_names}
    total_lineages = sum(int(group["lineage_count"]) for group in groups)
    total_episodes = sum(int(group["episode_count"]) for group in groups)

    for index, group in enumerate(groups):
        remaining = len(groups) - index
        required_empty = [
            split
            for split in ("dev", "test")
            if group_counts[split] == 0
        ]
        candidates: Sequence[str]
        if len(groups) >= 3 and remaining == len(required_empty):
            candidates = required_empty
        else:
            candidates = split_names

        def score(candidate_split: str) -> tuple[int, int]:
            proposed_lineages = dict(lineage_counts)
            proposed_episodes = dict(episode_counts)
            proposed_lineages[candidate_split] += int(group["lineage_count"])
            proposed_episodes[candidate_split] += int(group["episode_count"])
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
            combined = (
                lineage_error * max(total_episodes, 1)
                + episode_error * max(total_lineages, 1)
            )
            return combined, split_names.index(candidate_split)

        split = min(candidates, key=score)
        lineage_counts[split] += int(group["lineage_count"])
        episode_counts[split] += int(group["episode_count"])
        group_counts[split] += 1
        group["split"] = split
        for component in group["components"]:
            for candidate in component["candidates"]:
                candidate.episode["split"] = split
                candidate.episode["lineage_cluster_id"] = component["cluster_id"]
                candidate.episode["donor_group_id"] = group["group_id"]
    return lineage_counts


def _condition_id(
    episode_id: str,
    condition: str,
    source_id: str,
) -> str:
    return (
        f"{condition}-"
        + sha256_text(f"{episode_id}|{condition}|{source_id}")[:24].lower()
    )


def _attach_counterfactuals(
    groups: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> None:
    for group in groups:
        component_by_episode: dict[str, str] = {}
        for component in group["components"]:
            for candidate in component["candidates"]:
                component_by_episode[candidate.episode_id] = str(
                    component["cluster_id"]
                )
        for owner in group["candidates"]:
            owner_cluster = component_by_episode[owner.episode_id]
            donor_candidates = [
                donor
                for donor in group["candidates"]
                if donor.episode_id != owner.episode_id
                and donor.lineage != owner.lineage
                and component_by_episode[donor.episode_id] != owner_cluster
                and donor.fact_sha256 != owner.fact_sha256
            ]
            if not donor_candidates:
                raise DonorSeparationError(
                    f"{owner.episode_id} has no identity-separated donor"
                )
            donor = min(
                donor_candidates,
                key=lambda item: sha256_text(
                    f"{seed}|{owner.episode_id}|{item.episode_id}"
                ),
            )
            shuffle_seed = int(
                sha256_text(f"{seed}|{owner.episode_id}|shuffle")[:16],
                16,
            )
            ids = {
                "correct": _condition_id(
                    owner.episode_id,
                    "correct",
                    owner.episode_id,
                ),
                "zero": _condition_id(
                    owner.episode_id,
                    "zero",
                    "authoritative-zero",
                ),
                "swapped": _condition_id(
                    owner.episode_id,
                    "swapped",
                    donor.episode_id,
                ),
                "shuffled": _condition_id(
                    owner.episode_id,
                    "shuffled",
                    str(shuffle_seed),
                ),
            }
            owner.episode["counterfactuals"] = {
                "correct": {
                    "condition_id": ids["correct"],
                    "soul_source_episode_id": owner.episode_id,
                    "expected_answer_sha256": owner.fact_sha256,
                },
                "zero": {
                    "condition_id": ids["zero"],
                    "soul_source": "all_zero_authoritative_soul",
                    "expected_behavior": "abstain_or_increase_owner_nll",
                },
                "swapped": {
                    "condition_id": ids["swapped"],
                    "donor_episode_id": donor.episode_id,
                    "donor_source_lineage": donor.lineage,
                    "donor_lineage_cluster_id": component_by_episode[
                        donor.episode_id
                    ],
                    "donor_split": donor.episode["split"],
                    "donor_expected_answer_sha256": donor.fact_sha256,
                    "expected_behavior": "follow_donor_or_increase_owner_nll",
                },
                "shuffled": {
                    "condition_id": ids["shuffled"],
                    "soul_source_episode_id": owner.episode_id,
                    "shuffle_seed": shuffle_seed,
                    "expected_behavior": "abstain_or_increase_owner_nll",
                },
                "identity_separation": {
                    "owner_episode_id": owner.episode_id,
                    "owner_source_lineage": owner.lineage,
                    "owner_lineage_cluster_id": owner_cluster,
                    "owner_split": owner.episode["split"],
                    "donor_episode_differs": donor.episode_id != owner.episode_id,
                    "donor_lineage_differs": donor.lineage != owner.lineage,
                    "donor_cluster_differs": (
                        component_by_episode[donor.episode_id] != owner_cluster
                    ),
                    "donor_fact_differs": (
                        donor.fact_sha256 != owner.fact_sha256
                    ),
                    "donor_same_split": (
                        donor.episode["split"] == owner.episode["split"]
                    ),
                    "condition_ids_unique": len(set(ids.values())) == 4,
                },
            }


def _split_leakage_audit(
    candidates: Sequence[_Candidate],
) -> dict[str, Any]:
    exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    near: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        item = {
            "episode_id": candidate.episode_id,
            "source_lineage": candidate.lineage,
            "lineage_cluster_id": str(
                candidate.episode["lineage_cluster_id"]
            ),
            "split": str(candidate.episode["split"]),
        }
        exact[sha256_text(candidate.fingerprint_text)].append(item)
        near[
            sha256_text(_normalized(candidate.fingerprint_text))
        ].append(item)

    def conflicts(
        index: dict[str, list[dict[str, str]]],
    ) -> list[dict[str, Any]]:
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
            "exact": "sha256_fact_and_evidence",
            "near": "sha256_casefold_alphanumeric_fact_and_evidence",
            "transitive_lineage_union": True,
        },
        "exact_cross_split": exact_conflicts,
        "near_cross_split": near_conflicts,
        "passed": not exact_conflicts and not near_conflicts,
    }


def _donor_audit(
    candidates: Sequence[_Candidate],
) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for candidate in candidates:
        separation = candidate.episode["counterfactuals"][
            "identity_separation"
        ]
        required = (
            "donor_episode_differs",
            "donor_lineage_differs",
            "donor_cluster_differs",
            "donor_fact_differs",
            "donor_same_split",
            "condition_ids_unique",
        )
        for field in required:
            if separation.get(field) is not True:
                violations.append(
                    {
                        "episode_id": candidate.episode_id,
                        "contract": field,
                    }
                )
    return {
        "same_split_required": True,
        "different_episode_required": True,
        "different_lineage_required": True,
        "different_cluster_required": True,
        "different_fact_required": True,
        "unique_condition_ids_required": True,
        "violations": violations,
        "passed": not violations,
    }


def _physical_source_path(spec: SourceSpec) -> Path:
    if spec.kind == "jsonl":
        return Path(spec.path).resolve()
    if spec.kind != "sqlite":
        raise ValueError(f"unsupported source kind {spec.kind!r}")
    return _sqlite_path_from_uri(sqlite_ro_uri(spec.path)).resolve()


def _write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def build_soul_write_delay_curriculum(
    sources: Sequence[SourceSpec],
    output_dir: str | Path,
    *,
    seed: int = 42,
    unsupported_policy: str = "reject",
    include_diary: bool = False,
    min_delay_ticks: int = 1,
    require_production_gate: bool = False,
) -> dict[str, Any]:
    """Build a deterministic, source-immutable causal soul curriculum."""

    if unsupported_policy not in {"reject", "escape"}:
        raise ValueError("unsupported_policy must be reject or escape")
    if min_delay_ticks < 1:
        raise ValueError("min_delay_ticks must be at least one")
    candidates: list[_Candidate] = []
    source_manifest: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    d00_adapter_audit: dict[str, Any] | None = None
    d00_specs = [
        spec for spec in sources if is_d00_private_adapter(spec.adapter)
    ]
    if d00_specs:
        if len(d00_specs) != 1 or len(sources) != 1:
            raise D00GroundingError(
                "the canonical D00 adapter must be the sole source spec"
            )
        if unsupported_policy != "reject":
            raise D00GroundingError(
                "the D00 adapter requires unsupported_policy='reject' so "
                "capacity-audited source text cannot expand during encoding"
            )
        selection = select_grounded_records(d00_specs[0], seed=seed)
        for record in selection.records:
            try:
                candidate = _build_candidate(
                    record,
                    unsupported_policy=unsupported_policy,
                    include_diary=include_diary,
                    min_delay_ticks=min_delay_ticks,
                )
            except (UnsupportedSourceText, FieldViewContractError) as exc:
                raise D00GroundingError(
                    "a pre-audited D00 row failed the field/text contract"
                ) from exc
            if candidate is None:
                raise D00GroundingError(
                    "a pre-audited D00 row produced no candidate"
                )
            candidates.append(candidate)
        source_manifest.extend(selection.source_manifest)
        d00_adapter_audit = dict(selection.audit)
        if len(candidates) != int(d00_adapter_audit["emitted_rows"]):
            raise D00GroundingError(
                "D00 adapter emission count changed during episode build"
            )
    else:
        for spec in sources:
            physical_path = _physical_source_path(spec)
            if not physical_path.exists():
                raise FileNotFoundError(physical_path)
            before_hash = sha256_file(physical_path)
            rows_seen = 0
            emitted = 0
            for record in iter_source_records(spec, before_hash):
                rows_seen += 1
                try:
                    candidate = _build_candidate(
                        record,
                        unsupported_policy=unsupported_policy,
                        include_diary=include_diary,
                        min_delay_ticks=min_delay_ticks,
                    )
                except UnsupportedSourceText:
                    skipped["unsupported_source_text"] += 1
                    continue
                except FieldViewContractError:
                    # A source row is ineligible when its exact fact/evidence
                    # cannot fit the inherited active view.  Exclude it
                    # explicitly rather than clipping or weakening the
                    # visibility gate.
                    skipped["field_view_ineligible"] += 1
                    continue
                if candidate is None:
                    skipped["missing_fact_or_target"] += 1
                    continue
                candidates.append(candidate)
                emitted += 1
            after_hash = sha256_file(physical_path)
            if after_hash != before_hash:
                raise SourceMutationError(
                    f"source changed during read-only build: {physical_path}"
                )
            source_manifest.append(
                {
                    "kind": spec.kind,
                    "path": str(physical_path),
                    "adapter": spec.adapter,
                    "table": spec.table,
                    "query": spec.query,
                    "sha256_before": before_hash,
                    "sha256_after": after_hash,
                    "read_only_verified": True,
                    "rows_seen": rows_seen,
                    "episodes_emitted": emitted,
                }
            )

    if not candidates:
        raise EmptyCurriculumError("no eligible soul write-delay episodes")
    if require_production_gate:
        preflight_fact_counts: dict[str, int] = defaultdict(int)
        for candidate in candidates:
            preflight_fact_counts[candidate.fact_sha256] += 1
        preflight_maximum = max(preflight_fact_counts.values(), default=0)
        preflight_share = preflight_maximum / len(candidates)
        failed_preflight: list[str] = []
        if len(preflight_fact_counts) < 256:
            failed_preflight.append("minimum_256_unique_identities")
        if preflight_share > 0.05:
            failed_preflight.append(
                "maximum_single_fact_share_at_most_0_05"
            )
        if failed_preflight:
            raise CurriculumDiversityError(
                "production soul curriculum diversity gate failed: "
                + ", ".join(failed_preflight)
            )
    components = _cluster_candidates(candidates, seed=seed)
    groups = _make_donor_groups(components, seed=seed)
    split_lineages = _assign_donor_groups(groups)
    _attach_counterfactuals(groups, seed=seed)
    split_leakage = _split_leakage_audit(candidates)
    donor_audit = _donor_audit(candidates)
    if not split_leakage["passed"]:
        raise CrossSplitLeakageError(
            "duplicate-connected material crossed splits: "
            f"exact={len(split_leakage['exact_cross_split'])} "
            f"near={len(split_leakage['near_cross_split'])}"
        )
    if not donor_audit["passed"]:
        raise DonorSeparationError(
            f"donor identity audit failed with "
            f"{len(donor_audit['violations'])} violation(s)"
        )

    split_order = {"train": 0, "dev": 1, "test": 2}
    episodes = [candidate.episode for candidate in candidates]
    fact_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        fact_counts[candidate.fact_sha256] += 1
    split_counts = {
        split: sum(episode["split"] == split for episode in episodes)
        for split in ("train", "dev", "test")
    }
    episode_count = len(episodes)
    maximum_fact_count = max(fact_counts.values(), default=0)
    maximum_fact_share = (
        maximum_fact_count / episode_count if episode_count else 1.0
    )
    split_fractions = {
        split: split_counts[split] / episode_count
        for split in ("train", "dev", "test")
    }
    production_checks = {
        "minimum_256_unique_identities": len(fact_counts) >= 256,
        "maximum_single_fact_share_at_most_0_05": (
            maximum_fact_share <= 0.05
        ),
        "at_least_three_donor_groups": len(groups) >= 3,
        "all_splits_nonempty": all(split_counts.values()),
        "train_fraction_between_0_70_and_0_90": (
            0.70 <= split_fractions["train"] <= 0.90
        ),
        "dev_fraction_between_0_05_and_0_15": (
            0.05 <= split_fractions["dev"] <= 0.15
        ),
        "test_fraction_between_0_05_and_0_15": (
            0.05 <= split_fractions["test"] <= 0.15
        ),
    }
    production_diversity_gate = {
        "required": bool(require_production_gate),
        "passed": all(production_checks.values()),
        "checks": production_checks,
        "unique_fact_count": len(fact_counts),
        "maximum_single_fact_count": maximum_fact_count,
        "maximum_single_fact_share": maximum_fact_share,
        "donor_group_count": len(groups),
        "split_counts": split_counts,
        "split_fractions": split_fractions,
    }
    if require_production_gate and not production_diversity_gate["passed"]:
        failed = [
            name
            for name, passed in production_checks.items()
            if not passed
        ]
        raise CurriculumDiversityError(
            "production soul curriculum diversity gate failed: "
            + ", ".join(failed)
        )

    episodes.sort(
        key=lambda episode: (
            split_order[str(episode["split"])],
            str(episode["lineage_cluster_id"]),
            str(episode["source_lineage"]),
            str(episode["episode_id"]),
        )
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, dict[str, Any]] = {}
    all_path = output / "episodes.jsonl"
    _write_jsonl(all_path, episodes)
    output_files["all"] = {
        "path": all_path.name,
        "rows": len(episodes),
        "sha256": sha256_file(all_path),
    }
    for split in ("train", "dev", "test"):
        rows = [
            episode for episode in episodes if episode["split"] == split
        ]
        path = output / f"{split}.jsonl"
        _write_jsonl(path, rows)
        output_files[split] = {
            "path": path.name,
            "rows": len(rows),
            "sha256": sha256_file(path),
        }

    manifest_without_payload_hash = {
        "schema": MANIFEST_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "seed": seed,
        "source_contract": "read_only_hash_before_after",
        "source_root_default": str(DEFAULT_SOURCE_ROOT),
        "unsupported_policy": unsupported_policy,
        "minimum_delay_ticks": min_delay_ticks,
        "canonical_regions": [
            region.value for region in CANONICAL_REGION_ORDER
        ],
        "runtime_field_contract": {
            "total_slots": N_SLOTS,
            "physical_geometry": [
                {
                    "name": name,
                    "slot_start": start,
                    "slot_end": end,
                    "physical_type_id": type_id,
                }
                for name, start, end, type_id in PHYSICAL_PROFILE
            ],
            "writable_regions": sorted(WRITABLE_REGIONS),
            "recall_target_region": "response_draft",
        },
        "diary": {
            "included": include_diary,
            "default": "excluded",
            "included_rows_are_local_only": True,
            "cloud_export_allowed": (
                not include_diary
                and not any(
                    is_d00_private_adapter(spec.adapter)
                    for spec in sources
                )
            ),
        },
        "source_privacy": {
            "private_d00_sources_present": any(
                is_d00_private_adapter(spec.adapter)
                for spec in sources
            ),
            "private_d00_records_are_local_only": True,
            "cloud_export_allowed": (
                not include_diary
                and not any(
                    is_d00_private_adapter(spec.adapter)
                    for spec in sources
                )
            ),
        },
        "sources": source_manifest,
        "d00_adapter_audit": d00_adapter_audit,
        "episodes": len(episodes),
        "production_diversity_gate": production_diversity_gate,
        "lineage_cluster_count": len(components),
        "donor_group_count": len(groups),
        "split_counts": split_counts,
        "split_lineage_counts": split_lineages,
        "split_policy": (
            "transitive_exact_near_lineage_clusters_paired_into_"
            "same_split_donor_groups_then_80_10_10_best_effort"
        ),
        "counterfactual_conditions": list(COUNTERFACTUAL_CONDITIONS),
        "split_leakage_audit": split_leakage,
        "donor_identity_audit": donor_audit,
        "causal_visibility_gate": {
            "tick_a_fact_and_evidence_visible": True,
            "minimum_intervening_delay_ticks": min_delay_ticks,
            "delay_answer_bearing_text_absent": True,
            "tick_b_answer_bearing_text_absent_from_all_active_regions": True,
            "soul_state_required_for_tick_b": True,
        },
        "skipped": dict(sorted(skipped.items())),
        "output_files": output_files,
        "manifest_hash_contract": {
            "payload_algorithm": "SHA-256 canonical JSON without manifest_payload_sha256",
            "file_hash_sidecar": "manifest.sha256",
        },
    }
    manifest = dict(manifest_without_payload_hash)
    manifest["manifest_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(manifest_without_payload_hash)
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_file_hash = sha256_file(manifest_path)
    (output / "manifest.sha256").write_text(
        f"{manifest_file_hash}  manifest.json\n",
        encoding="ascii",
    )
    return manifest


def discover_soul_sources(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
) -> list[SourceSpec]:
    root = Path(source_root)
    private_sources = discover_d00_private_soul_sources(root)
    if private_sources:
        return private_sources
    exact_candidates = (
        root / "episodes.jsonl",
        root / "multitick_exact_v4" / "episodes.jsonl",
        root / "soul_source" / "episodes.jsonl",
    )
    exact = [
        SourceSpec.jsonl(path, adapter="exact_v4")
        for path in exact_candidates
        if path.exists()
    ]
    return exact or discover_primary_sources(root)


def _parse_sqlite_spec(value: str) -> SourceSpec:
    if "::" not in value:
        raise argparse.ArgumentTypeError("--sqlite requires PATH::TABLE")
    path, table = value.rsplit("::", 1)
    return SourceSpec.sqlite(path, table=table)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--jsonl", action="append", default=[])
    parser.add_argument(
        "--sqlite",
        action="append",
        default=[],
        metavar="PATH::TABLE",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unsupported-policy",
        choices=("reject", "escape"),
        default="reject",
    )
    parser.add_argument("--include-diary", action="store_true")
    parser.add_argument("--min-delay-ticks", type=int, default=1)
    parser.add_argument(
        "--allow-smoke-only",
        action="store_true",
        help=(
            "write a mechanically valid small/diversity-poor fixture; "
            "production CLI builds require the 256-identity diversity gate"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = [
        SourceSpec.jsonl(path, adapter="exact_v4")
        for path in args.jsonl
    ]
    sources.extend(_parse_sqlite_spec(value) for value in args.sqlite)
    if not sources:
        sources = discover_soul_sources(args.source_root)
    if not sources:
        raise FileNotFoundError(
            f"no recovered exact source material under {args.source_root}"
        )
    manifest = build_soul_write_delay_curriculum(
        sources,
        args.output_dir,
        seed=args.seed,
        unsupported_policy=args.unsupported_policy,
        include_diary=args.include_diary,
        min_delay_ticks=args.min_delay_ticks,
        require_production_gate=not args.allow_smoke_only,
    )
    print(
        f"built {manifest['episodes']} causal soul episodes -> "
        f"{args.output_dir} (train/dev/test={manifest['split_counts']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

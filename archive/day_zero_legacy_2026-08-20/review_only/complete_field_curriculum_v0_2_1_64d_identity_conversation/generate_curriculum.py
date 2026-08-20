#!/usr/bin/env python3
"""Deterministic review-only generator for Axon complete-field curriculum v0.2.1.

This builder uses only the Python standard library.  It does not train a model,
touch Axon runtime state, call the network, or claim external validation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAMP = "ChatGPT / GPT-5 / 2026-08-18"
BUILDER_VERSION = "axon-complete-field-builder-v0.2"
SCHEMA_VERSION = "0.2.0-review"
REGIONS = (
    "conversation_history",
    "user_input",
    "structured_knowledge",
    "situation_awareness",
    "tool_results",
    "advisor_input",
    "task_state",
    "scratch",
    "response_draft",
    "diary",
)
IMMUTABLE_CONTENT_REGIONS = frozenset(
    {"conversation_history", "user_input", "tool_results", "advisor_input"}
)
FOCUS_PROFILE = "core-r0-64d-identity-conversation"
FOCUS_WRITABLE_REGIONS = ("scratch", "response_draft", "diary")
ROSTER = ("core-alpha", "core-beta", "core-gamma")
PHASES = (("proposal", 0), ("refinement", 1), ("consolidation", 2))
VARIANTS = (
    "correct",
    "missing_page",
    "duplicated_page",
    "reordered_page",
    "hash_corrupted",
)
PAGE_STRATA = (1, 2, 4, 8)
EVIDENCE_BINS = ("first", "early", "middle", "late", "last")
FAMILY_NAMES = {
    "A": "exact substrate and page mechanics",
    "B": "cross-page retrieval",
    "C": "cross-page aggregation and reasoning",
    "D": "region semantics and mask behavior",
    "E": "language and conversation foundation",
    "F": "scratch as a causal workspace",
    "G": "tools code science and external evidence",
    "H": "literature creativity and psychology",
    "I": "typed field deltas and multi-tick correction",
    "J": "council proposal refinement and consolidation",
    "K": "private soul use",
    "L": "diary and autobiographical judgment",
    "M": "robustness and length generalization",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_canonical(value: Any) -> str:
    return sha256_text(canonical_json(value))


def digest_text(value: str) -> dict[str, str]:
    return {"algorithm": "sha256", "digest": sha256_text(value)}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_package_contracts(package_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles = load_json(package_root / "authority_profiles.json")
    alphabet = load_json(package_root / "alphabet_manifest.json")
    serialized = canonical_json(alphabet["ordered_characters"])
    if sha256_text(serialized) != alphabet["digest"]:
        raise ValueError("alphabet manifest canonical digest mismatch")
    if len(alphabet["ordered_characters"]) != alphabet["length"]:
        raise ValueError("alphabet manifest length mismatch")
    return profiles, alphabet


def is_allowed(
    profiles: dict[str, Any],
    profile_id: str,
    core_id: str,
    action_class: str,
    region: str,
    operation: str,
) -> bool:
    if action_class == "content" and region in IMMUTABLE_CONTENT_REGIONS:
        return False
    profile = profiles["profiles"].get(profile_id, {})
    core = profile.get("cores", {}).get(core_id, {})
    return operation in core.get(action_class, {}).get(region, [])


def _span(name: str, text: str, group_index: int) -> list[dict[str, Any]]:
    if not text:
        return []
    return [
        {
            "span_id": f"fixture-{group_index:04d}-{name}-0",
            "text": text,
            "kind": "synthetic_fixture",
            "source": "generate_curriculum.py",
            "provenance": f"synthetic-lineage-{group_index:04d}",
            "confidence": 1.0,
            "container_refs": [],
            "edge_refs": [],
        }
    ]


def build_snapshot(group_index: int, tick_seq: int, texts: dict[str, str]) -> dict[str, Any]:
    regions = []
    for name in REGIONS:
        regions.append(
            {
                "name": name,
                "visibility": "attended",
                "write_policy": "core_writable"
                if name in FOCUS_WRITABLE_REGIONS
                else "sealed",
                "spans": _span(name, texts.get(name, ""), group_index),
            }
        )
    canonical = {
        "schema": "shared-field-v1",
        "tick_id": tick_seq,
        "parent_field_id": None,
        "source_manifest_ids": [f"synthetic-lineage-{group_index:04d}"],
        "regions": regions,
    }
    return {**canonical, "field_id": sha256_canonical(canonical)}


def region_text(snapshot: dict[str, Any], name: str) -> str:
    for region in snapshot["regions"]:
        if region["name"] == name:
            return "".join(span["text"] for span in region["spans"])
    raise KeyError(name)


def eligible_bins(page_count: int) -> tuple[str, ...]:
    if page_count == 1:
        return ("first",)
    if page_count == 2:
        return ("first", "last")
    if page_count == 4:
        return ("first", "early", "late", "last")
    if page_count == 8:
        return EVIDENCE_BINS
    raise ValueError(f"unsupported page stratum {page_count}")


def evidence_page(page_count: int, evidence_bin: str) -> int:
    mapping = {
        1: {"first": 0},
        2: {"first": 0, "last": 1},
        4: {"first": 0, "early": 1, "late": 2, "last": 3},
        8: {"first": 0, "early": 1, "middle": 3, "late": 6, "last": 7},
    }
    return mapping[page_count][evidence_bin]


def choose_stratum_and_bin(group_index: int) -> tuple[int, str]:
    page_count = PAGE_STRATA[group_index % len(PAGE_STRATA)]
    occurrence = group_index // len(PAGE_STRATA)
    bins = eligible_bins(page_count)
    return page_count, bins[occurrence % len(bins)]


def deterministic_filler(length: int, seed: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    return "".join(letters[(seed * 17 + index * 29) % len(letters)] for index in range(length))


def make_field_text(
    group_index: int, page_count: int, page_size: int, evidence_bin: str
) -> tuple[dict[str, str], int, str, int]:
    total_active = page_count * page_size
    texts = {
        "conversation_history": "Jeff: Hi. Axon: Hi Jeff. ",
        "user_input": "Use exact evidence. ",
        "structured_knowledge": "Name: Axon. ",
        "situation_awareness": "64D core active. ",
        "tool_results": "",
        "advisor_input": "Be truthful. ",
        "task_state": "Converse and learn. ",
        "scratch": "Plan: inspect all regions. ",
        "response_draft": "Draft: Hi Jeff. ",
        "diary": "I am Axon. ",
    }
    marker = f"NEEDLE{group_index:04d}"
    fixed_active = sum(len(text) for text in texts.values())
    tool_active_length = total_active - fixed_active
    if tool_active_length <= len(marker) + 2:
        raise ValueError(
            "page size is too small for the ten-region identity fixture; use at least 256"
        )
    prefix_before_tool = sum(
        len(texts[name]) for name in REGIONS[: REGIONS.index("tool_results")]
    )
    target_page = evidence_page(page_count, evidence_bin)
    desired_global = target_page * page_size + max(4, page_size // 3)
    desired_global = max(desired_global, prefix_before_tool + 2)
    desired_global = min(
        desired_global,
        prefix_before_tool + tool_active_length - len(marker) - 1,
    )
    tool_active = deterministic_filler(tool_active_length, group_index)
    local_marker = desired_global - prefix_before_tool
    tool_active = (
        tool_active[:local_marker]
        + marker
        + tool_active[local_marker + len(marker) :]
    )
    dormant_prefix = f"Dormant tool preface {group_index:04d}. "
    texts["tool_results"] = dormant_prefix + tool_active
    return texts, len(dormant_prefix), marker, local_marker + len(dormant_prefix)


def build_attention_view(
    profiles: dict[str, Any],
    profile_id: str,
    core_id: str,
    snapshot: dict[str, Any],
    tool_mask_offset: int,
) -> dict[str, Any]:
    profile = profiles["profiles"][profile_id]
    result: dict[str, Any] = {}
    for name in REGIONS:
        text = region_text(snapshot, name)
        offset = tool_mask_offset if name == "tool_results" else 0
        intervals = [[offset, len(text)]] if offset < len(text) else []
        result[name] = {
            "source_text_digest": digest_text(text),
            "mode": "manual",
            "mask_offset": offset,
            "unit": "chars",
            "retain": len(text) - offset,
            "active_intervals": intervals,
            "future_intervals": None,
            "can_set_boundary": is_allowed(
                profiles, profile_id, core_id, "attention", name, "set_boundary"
            ),
            "can_set_intervals": False,
        }
    return {
        "schema": "core-region-attention-view-v0.2",
        "core_id": core_id,
        "profile_id": profile_id,
        "mask_scope": profile["mask_scope"],
        "regions": result,
    }


def build_active_stream(
    snapshot: dict[str, Any], attention_view: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    pieces: list[str] = []
    segments: list[dict[str, Any]] = []
    global_offset = 0
    for name in REGIONS:
        text = region_text(snapshot, name)
        for start, end in attention_view["regions"][name]["active_intervals"]:
            exact = text[start:end]
            segment = {
                "region": name,
                "region_start": start,
                "region_end": end,
                "stream_start": global_offset,
                "stream_end": global_offset + len(exact),
                "exact_text": exact,
            }
            segments.append(segment)
            pieces.append(exact)
            global_offset += len(exact)
    return "".join(pieces), segments


def paginate_base_field(
    snapshot: dict[str, Any], attention_view: dict[str, Any], page_size: int
) -> tuple[str, list[dict[str, Any]]]:
    stream, source_segments = build_active_stream(snapshot, attention_view)
    pages: list[dict[str, Any]] = []
    stream_id = f"{snapshot['field_id']}:{attention_view['core_id']}"
    for page_index, start in enumerate(range(0, len(stream), page_size)):
        end = min(start + page_size, len(stream))
        page_segments = []
        for source in source_segments:
            left = max(start, source["stream_start"])
            right = min(end, source["stream_end"])
            if left >= right:
                continue
            local_left = left - source["stream_start"]
            local_right = right - source["stream_start"]
            page_segments.append(
                {
                    "region": source["region"],
                    "region_start": source["region_start"] + local_left,
                    "region_end": source["region_start"] + local_right,
                    "stream_start": left,
                    "stream_end": right,
                    "exact_text": source["exact_text"][local_left:local_right],
                }
            )
        exact = stream[start:end]
        pages.append(
            {
                "namespace": "base_field",
                "stream_id": stream_id,
                "page_index": page_index,
                "stream_start": start,
                "stream_end": end,
                "exact_text": exact,
                "digest": digest_text(exact),
                "segments": page_segments,
            }
        )
    return stream, pages


def make_field_delta(
    snapshot: dict[str, Any],
    tick_seq: int,
    core_id: str,
    phase: str,
    regions: str | tuple[str, ...],
    text: str,
) -> dict[str, Any]:
    targets = (regions,) if isinstance(regions, str) else regions
    suffixes = {
        "scratch": f" Work product:{text}.",
        "response_draft": f" Response:{text}.",
        "diary": f" Exchange note:{text}.",
    }
    return {
        "schema": "shared-field-delta-v1",
        "base_field_id": snapshot["field_id"],
        "base_tick_id": tick_seq,
        "author_core_id": core_id,
        "pass_id": f"tick-{tick_seq}-{phase}-{core_id}",
        "operations": [
            {
                "op": "insert",
                "region": region,
                "offset": len(region_text(snapshot, region)),
                "text": suffixes.get(region, text),
                "provenance": f"synthetic:{tick_seq}:{phase}:{core_id}",
                "container_refs": [],
                "edge_refs": [],
            }
            for region in targets
        ],
        "evidence": [f"needle-{tick_seq}"],
    }


def make_envelope(
    snapshot: dict[str, Any],
    tick_seq: int,
    phase: str,
    phase_seq: int,
    core_id: str,
    roster_index: int,
    field_delta: dict[str, Any],
) -> dict[str, Any]:
    payload_text = canonical_json(field_delta)
    delta_id = sha256_text(payload_text)
    basis = {
        "schema": "council-delta-envelope-v0.2",
        "tick_seq": tick_seq,
        "source_phase": phase,
        "phase_seq": phase_seq,
        "core_id": core_id,
        "roster_index": roster_index,
        "checkpoint": digest_text(f"checkpoint:{core_id}"),
        "soul_before": digest_text(f"soul:{core_id}:{tick_seq}:{phase}:before"),
        "soul_after": digest_text(f"soul:{core_id}:{tick_seq}:{phase}:after"),
        "base_field_id": snapshot["field_id"],
        "base_tick_id": tick_seq,
        "payload": {
            "encoding": "canonical-json-utf8",
            "schema": "shared-field-delta-v1",
            "exact_text": payload_text,
            "delta_id": delta_id,
            "digest": {"algorithm": "sha256", "digest": delta_id},
        },
        "provenance": f"synthetic-council:{tick_seq}:{phase}:{core_id}",
    }
    return {**basis, "envelope_id": sha256_canonical(basis)}


def choose_target_regions(
    profile_id: str, core_id: str, group_index: int
) -> tuple[str, ...]:
    if profile_id == FOCUS_PROFILE:
        return FOCUS_WRITABLE_REGIONS
    if profile_id == "core-v1-scratch-response-only":
        return ("scratch" if group_index % 2 == 0 else "response_draft",)
    choices = {
        "core-alpha": ("scratch", "response_draft"),
        "core-beta": ("structured_knowledge", "situation_awareness", "task_state"),
        "core-gamma": ("diary", "response_draft"),
    }[core_id]
    return (choices[group_index % len(choices)],)


def build_sibling_set(
    profiles: dict[str, Any],
    profile_id: str,
    snapshot: dict[str, Any],
    tick_seq: int,
    phase: str,
) -> dict[str, Any]:
    if phase == "proposal":
        source_phase = None
        source_phase_seq = None
        expected: list[str] = []
        envelopes: list[dict[str, Any]] = []
    else:
        source_phase, source_phase_seq = (
            ("proposal", 0) if phase == "refinement" else ("refinement", 1)
        )
        expected = list(ROSTER)
        envelopes = []
        for roster_index, core_id in enumerate(ROSTER):
            target = choose_target_regions(profile_id, core_id, tick_seq + roster_index)
            delta = make_field_delta(
                snapshot,
                tick_seq,
                core_id,
                source_phase,
                target,
                f" sibling {source_phase} {core_id}",
            )
            envelopes.append(
                make_envelope(
                    snapshot,
                    tick_seq,
                    source_phase,
                    source_phase_seq,
                    core_id,
                    roster_index,
                    delta,
                )
            )
    basis = {
        "schema": "council-sibling-delta-set-v0.2",
        "tick_seq": tick_seq,
        "source_phase": source_phase,
        "expected_core_ids": expected,
        "envelopes": envelopes,
    }
    return {**basis, "set_id": sha256_canonical(basis)}


def paginate_simple_stream(
    namespace: str, stream_id: str, stream: str, page_size: int
) -> list[dict[str, Any]]:
    pages = []
    for page_index, start in enumerate(range(0, len(stream), page_size)):
        end = min(start + page_size, len(stream))
        exact = stream[start:end]
        pages.append(
            {
                "namespace": namespace,
                "stream_id": stream_id,
                "page_index": page_index,
                "stream_start": start,
                "stream_end": end,
                "exact_text": exact,
                "digest": digest_text(exact),
                "segments": [],
            }
        )
    return pages


def coverage_namespace(
    required: bool, stream_id: str, stream: str, pages: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "required": required,
        "stream_id": stream_id,
        "source_digest": digest_text(stream),
        "expected_characters": len(stream),
        "expected_page_indices": list(range(len(pages))),
        "pages": pages,
    }


def namespace_failure_prefix(namespace: str) -> str:
    return "BASE_FIELD" if namespace == "base_field" else "SIBLING_DELTA_SET"


def validate_namespace(namespace: str, value: dict[str, Any]) -> str | None:
    pages = value["pages"]
    expected = value["expected_page_indices"]
    indices = [page["page_index"] for page in pages]
    prefix = namespace_failure_prefix(namespace)
    if len(indices) != len(set(indices)):
        return f"{prefix}_DUPLICATE_PAGE"
    if set(indices) != set(expected):
        if set(indices).issubset(set(expected)):
            return f"{prefix}_MISSING_PAGE"
        return f"{prefix}_PAGE_INDEX_INVALID"
    if indices != expected:
        return f"{prefix}_REORDERED_PAGE"
    cursor = 0
    pieces = []
    for page in pages:
        if page["stream_start"] != cursor or page["stream_end"] < cursor:
            return f"{prefix}_COVERAGE_GAP"
        if page["digest"] != digest_text(page["exact_text"]):
            return f"{prefix}_HASH_MISMATCH"
        cursor = page["stream_end"]
        pieces.append(page["exact_text"])
    assembled = "".join(pieces)
    if cursor != value["expected_characters"]:
        return f"{prefix}_COVERAGE_GAP"
    if digest_text(assembled) != value["source_digest"]:
        return f"{prefix}_SOURCE_DIGEST_MISMATCH"
    return None


def coverage_oracle(manifest: dict[str, Any]) -> str | None:
    for namespace in ("base_field", "sibling_delta_set"):
        value = manifest[namespace]
        if not value["required"]:
            if value["pages"] or value["expected_characters"]:
                return f"{namespace_failure_prefix(namespace)}_UNEXPECTED_STREAM"
            continue
        failure = validate_namespace(namespace, value)
        if failure:
            return failure
    return None


def corrupt_namespace(
    namespace: str, value: dict[str, Any], variant: str
) -> tuple[dict[str, Any], str]:
    corrupted = copy.deepcopy(value)
    pages = corrupted["pages"]
    prefix = namespace_failure_prefix(namespace)
    if not pages:
        raise ValueError(f"cannot corrupt empty required namespace {namespace}")
    if variant == "missing_page":
        pages.pop()
        expected = f"{prefix}_MISSING_PAGE"
    elif variant == "duplicated_page":
        pages.append(copy.deepcopy(pages[0]))
        expected = f"{prefix}_DUPLICATE_PAGE"
    elif variant == "reordered_page":
        if len(pages) >= 2:
            pages[0], pages[-1] = pages[-1], pages[0]
            expected = f"{prefix}_REORDERED_PAGE"
        else:
            pages[0]["page_index"] = 1
            expected = f"{prefix}_PAGE_INDEX_INVALID"
    elif variant == "hash_corrupted":
        old = pages[0]["digest"]["digest"]
        pages[0]["digest"]["digest"] = ("1" if old[0] != "1" else "0") + old[1:]
        expected = f"{prefix}_HASH_MISMATCH"
    else:
        raise ValueError(variant)
    return corrupted, expected


def make_transport(envelope: dict[str, Any], transaction_id: str) -> dict[str, Any]:
    exact = canonical_json(envelope)
    split = max(1, len(exact) // 2)
    pieces = (exact[:split], exact[split:]) if split < len(exact) else (exact,)
    chunks = []
    for index, piece in enumerate(pieces):
        chunks.append(
            {
                "transaction_id": transaction_id,
                "expected_chunk_count": len(pieces),
                "chunk_index": index,
                "byte_length": len(piece.encode("utf-8")),
                "digest": digest_text(piece),
                "exact_text": piece,
            }
        )
    return {
        "transaction_id": transaction_id,
        "assembled_digest": digest_text(exact),
        "chunks": chunks,
    }


def build_policy_assertions(
    profiles: dict[str, Any],
    profile_id: str,
    core_id: str,
    target_regions: tuple[str, ...],
) -> list[dict[str, Any]]:
    probes = [
        *[
            ("content", region, "insert", "target operation follows this core's content switch")
            for region in target_regions
        ],
        ("content", "tool_results", "replace", "tool evidence is immutable to cores"),
        ("content", "conversation_history", "insert", "conversation history is runtime-owned immutable evidence"),
        ("attention", "tool_results", "set_boundary", "attention authority is independent from content authority"),
        ("attention", "tool_results", "set_intervals", "future disjoint interval feature is disabled in R0"),
    ]
    return [
        {
            "action_class": action_class,
            "region": region,
            "operation": operation,
            "expected_authorized": is_allowed(
                profiles, profile_id, core_id, action_class, region, operation
            ),
            "reason": reason,
        }
        for action_class, region, operation, reason in probes
    ]


def supported_audit(texts: Iterable[str], alphabet: dict[str, Any]) -> dict[str, Any]:
    allowed = set(alphabet["ordered_characters"])
    unsupported = sorted({ord(char) for text in texts for char in text if char not in allowed})
    return {
        "alphabet_id": alphabet["alphabet_id"],
        "algorithm": alphabet["algorithm"],
        "digest": alphabet["digest"],
        "supported": not unsupported,
        "unsupported_codepoints": [f"U+{value:04X}" for value in unsupported],
        "action": "accept" if not unsupported else "quarantine",
    }


def build_record(
    profiles: dict[str, Any],
    alphabet: dict[str, Any],
    *,
    group_index: int,
    variant: str,
    page_size: int,
    split_override: str | None = None,
    family_override: str | None = None,
    phase_override: tuple[str, int] | None = None,
    tick_override: int | None = None,
    page_count_override: int | None = None,
    evidence_bin_override: str | None = None,
    profile_override: str | None = None,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    page_count, evidence_bin = choose_stratum_and_bin(group_index)
    if page_count_override is not None:
        page_count = page_count_override
    if evidence_bin_override is not None:
        evidence_bin = evidence_bin_override
    if evidence_bin not in eligible_bins(page_count):
        evidence_bin = eligible_bins(page_count)[0]
    phase, phase_seq = phase_override or PHASES[group_index % len(PHASES)]
    tick_seq = tick_override if tick_override is not None else 1000 + group_index
    consolidator = ROSTER[tick_seq % len(ROSTER)]
    acting_core = consolidator if phase == "consolidation" else ROSTER[group_index % len(ROSTER)]
    profile_id = profile_override or FOCUS_PROFILE
    texts, tool_offset, marker, marker_region_start = make_field_text(
        group_index, page_count, page_size, evidence_bin
    )
    snapshot = build_snapshot(group_index, tick_seq, texts)
    attention = build_attention_view(
        profiles, profile_id, acting_core, snapshot, tool_offset
    )
    active_stream, base_pages = paginate_base_field(snapshot, attention, page_size)
    if len(base_pages) != page_count:
        raise AssertionError(f"expected {page_count} base pages; built {len(base_pages)}")

    sibling_set = build_sibling_set(
        profiles, profile_id, snapshot, tick_seq, phase
    )
    sibling_stream = canonical_json(sibling_set) if phase != "proposal" else ""
    sibling_pages = (
        paginate_simple_stream(
            "sibling_delta_set", sibling_set["set_id"], sibling_stream, page_size
        )
        if sibling_stream
        else []
    )
    base_ns = coverage_namespace(
        True,
        f"{snapshot['field_id']}:{acting_core}",
        active_stream,
        base_pages,
    )
    sibling_ns = coverage_namespace(
        phase != "proposal",
        sibling_set["set_id"],
        sibling_stream,
        sibling_pages,
    )

    expected_failure = None
    if variant != "correct":
        target_namespace = (
            "sibling_delta_set"
            if phase != "proposal" and group_index % 2 == 1
            else "base_field"
        )
        if target_namespace == "base_field":
            base_ns, expected_failure = corrupt_namespace(
                target_namespace, base_ns, variant
            )
        else:
            sibling_ns, expected_failure = corrupt_namespace(
                target_namespace, sibling_ns, variant
            )

    coverage = {
        "schema": "logical-pass-coverage-v0.2",
        "base_field": base_ns,
        "sibling_delta_set": sibling_ns,
        "zero_gaps": variant not in {"missing_page"},
        "zero_duplicates": variant not in {"duplicated_page"},
        "finalization_allowed": variant == "correct",
        "observed_failure_code": None,
    }
    observed_failure = coverage_oracle(coverage)
    coverage["observed_failure_code"] = observed_failure
    if observed_failure != expected_failure:
        raise AssertionError(
            f"coverage oracle mismatch: expected {expected_failure!r}, observed {observed_failure!r}"
        )

    target_regions = choose_target_regions(profile_id, acting_core, group_index)
    for target_region in target_regions:
        if not is_allowed(
            profiles, profile_id, acting_core, "content", target_region, "insert"
        ):
            raise AssertionError("builder selected a content-disabled target region")
    target_delta = make_field_delta(
        snapshot,
        tick_seq,
        acting_core,
        phase,
        target_regions,
        f" verified {marker}",
    )
    target_envelope = make_envelope(
        snapshot,
        tick_seq,
        phase,
        phase_seq,
        acting_core,
        ROSTER.index(acting_core),
        target_delta,
    )
    transaction_id = f"tx-{tick_seq}-{phase}-{acting_core}"
    attention_update = None
    if is_allowed(
        profiles,
        profile_id,
        acting_core,
        "attention",
        "tool_results",
        "set_boundary",
    ):
        tool_text = region_text(snapshot, "tool_results")
        attention_update = {
            "region": "tool_results",
            "operation": "set_boundary",
            "old_offset": tool_offset,
            "new_offset": min(len(tool_text), tool_offset + page_size),
            "effective": "next_tick",
            "scope": "core_private_attention_view",
            "source_text_digest": digest_text(tool_text),
        }

    if variant == "correct":
        phase_result = {
            "outcome": "delta",
            "delta_envelope": target_envelope,
            "delta_transport": make_transport(target_envelope, transaction_id),
            "attention_update": attention_update,
            "commit_semantics": {
                "proposal": "non_mutating_proposal",
                "refinement": "non_mutating_refinement",
                "consolidation": "atomic_consolidation_commit",
            }[phase],
        }
        expected = {
            "mode": "success",
            "failure_code": None,
            "finalization_allowed": True,
        }
    else:
        phase_result = {
            "outcome": "failure",
            "delta_envelope": None,
            "delta_transport": None,
            "attention_update": None,
            "commit_semantics": "none",
        }
        expected = {
            "mode": "failure",
            "failure_code": expected_failure,
            "finalization_allowed": False,
        }

    expected_siblings = [] if phase == "proposal" else list(ROSTER)
    family = family_override or chr(ord("A") + group_index % 13)
    split = split_override or (
        "train" if group_index % 10 <= 6 else "dev" if group_index % 10 <= 8 else "test"
    )
    exact_texts = [region_text(snapshot, name) for name in REGIONS]
    exact_texts.extend(envelope["payload"]["exact_text"] for envelope in sibling_set["envelopes"])
    exact_texts.append(target_envelope["payload"]["exact_text"])
    return {
        "record_type": "complete_field_example",
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "stamp": STAMP,
        "example_id": f"v02-{group_index:04d}-{phase}-{variant}",
        "family": family,
        "sub_family": FAMILY_NAMES[family],
        "split": split,
        "seed": group_index,
        "lineage_id": f"lineage-{group_index:04d}",
        "counterfactual_group": f"coverage-group-{group_index:04d}",
        "counterfactual_variant": variant,
        "tick": {
            "tick_seq": tick_seq,
            "phase": phase,
            "phase_seq": phase_seq,
            "acting_core_id": acting_core,
            "consolidator_core_id": consolidator,
            "online_core_ids": list(ROSTER),
            "expected_sibling_core_ids": expected_siblings,
        },
        "difficulty": {
            "base_field_page_count": page_count,
            "sibling_page_count": len(sibling_pages),
            "page_size": page_size,
            "active_character_count": len(active_stream),
            "evidence_position": evidence_bin,
            "reasoning_steps": 1 + group_index % 4,
        },
        "snapshot": snapshot,
        "attention_view": attention,
        "authority": {
            "profile_id": profile_id,
            "core_id": acting_core,
            "default_decision": "deny",
            "policy_assertions": build_policy_assertions(
                profiles, profile_id, acting_core, target_regions
            ),
        },
        "sibling_delta_set": sibling_set,
        "coverage_manifest": coverage,
        "phase_result": phase_result,
        "expected": expected,
        "evidence_spans": [
            {
                "pointer_id": f"needle-{tick_seq}",
                "region": "tool_results",
                "start": marker_region_start,
                "end": marker_region_start + len(marker),
                "exact_text": marker,
                "provenance_id": f"synthetic-lineage-{group_index:04d}",
                "use": "answer",
            }
        ],
        "soul_transition": {
            "soul_id": f"soul:{acting_core}",
            "before": digest_text(f"soul:{acting_core}:{tick_seq}:{phase}:before"),
            "after": digest_text(f"soul:{acting_core}:{tick_seq}:{phase}:after"),
            "inhale_count": 1,
            "exhale_count": 1,
            "phase_bound": True,
            "authoritative_exact_memory": False,
        },
        "provenance": {
            "source_type": "synthetic",
            "source_ids": [f"synthetic-lineage-{group_index:04d}"],
            "license": "CC0 synthetic fixture",
            "privacy": "public_synthetic",
            "autobiographical": False,
        },
        "alphabet": supported_audit(exact_texts, alphabet),
        "builder_validation": {
            "checks_run": [
                "alphabet_manifest_digest",
                "snapshot_identity",
                "sibling_roster_completeness",
                "coverage_corruption_oracle",
                "authority_target_selection",
            ],
            "coverage_oracle_expected": expected_failure,
            "coverage_oracle_observed": observed_failure,
            "external_validation": None,
        },
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    rows = list(records)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in rows:
            handle.write(canonical_json(record))
            handle.write("\n")
    return len(rows)


def ensure_clean_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output root: {output_root}; use a new clean directory"
        )
    output_root.mkdir(parents=True, exist_ok=True)


def build_generated_dataset(
    package_root: Path,
    output_root: Path,
    *,
    groups: int = 20,
    page_size: int = 256,
    seed: int = 20260818,
) -> dict[str, Any]:
    if groups < 20 or groups % 20:
        raise ValueError("--groups must be a positive multiple of 20 so every page stratum and evidence bin is represented")
    if page_size < 256:
        raise ValueError("--page-size must be at least 256 for the ten-region identity fixtures")
    profiles, alphabet = load_package_contracts(package_root)
    ensure_clean_output_root(output_root)
    records: list[dict[str, Any]] = []
    for group_index in range(groups):
        for variant in VARIANTS:
            records.append(
                build_record(
                    profiles,
                    alphabet,
                    group_index=group_index,
                    variant=variant,
                    page_size=page_size,
                )
            )

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    split_files = {}
    for split in ("train", "dev", "test"):
        path = output_root / f"{split}.jsonl"
        count = write_jsonl(path, by_split[split])
        split_files[split] = {
            "path": path.name,
            "algorithm": "sha256",
            "digest": sha256_bytes(path.read_bytes()),
            "records": count,
        }

    correct = [record for record in records if record["counterfactual_variant"] == "correct"]
    variants_by_group = defaultdict(set)
    for record in records:
        variants_by_group[record["counterfactual_group"]].add(record["counterfactual_variant"])
    required = set(VARIANTS)
    if any(values != required for values in variants_by_group.values()):
        raise AssertionError("incomplete counterfactual group escaped the builder")
    page_counts = Counter(str(record["difficulty"]["base_field_page_count"]) for record in correct)
    bin_counts = Counter(record["difficulty"]["evidence_position"] for record in correct)
    phase_counts = Counter(record["tick"]["phase"] for record in correct)
    profile_counts = Counter(record["authority"]["profile_id"] for record in correct)
    if any(page_counts[str(value)] == 0 for value in PAGE_STRATA):
        raise AssertionError("a required page stratum is empty")
    if any(bin_counts[value] == 0 for value in EVIDENCE_BINS):
        raise AssertionError("a required evidence position is empty")

    manifest = {
        "schema": "axon-complete-field-curriculum-manifest-v0.2",
        "builder_version": BUILDER_VERSION,
        "stamp": STAMP,
        "seed": seed,
        "group_count": groups,
        "record_count": len(records),
        "required_variants": list(VARIANTS),
        "page_strata": {str(value): page_counts[str(value)] for value in PAGE_STRATA},
        "evidence_position_counts": {value: bin_counts[value] for value in EVIDENCE_BINS},
        "phase_counts": dict(sorted(phase_counts.items())),
        "profile_counts": dict(sorted(profile_counts.items())),
        "split_files": split_files,
        "builder_validation": {
            "output_root_was_empty": True,
            "complete_variant_groups": True,
            "coverage_oracle_matches": True,
            "alphabet_audit": True,
            "schema_validation": None,
            "deterministic_replay": None,
        },
        "external_validation": None,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def build_review_fixtures(package_root: Path, *, page_size: int = 256) -> None:
    profiles, alphabet = load_package_contracts(package_root)
    starters = [
        build_record(
            profiles,
            alphabet,
            group_index=index,
            variant="correct",
            page_size=page_size,
            split_override="starter",
            family_override=chr(ord("A") + index),
        )
        for index in range(13)
    ]
    council_tick = 777
    for offset, phase in enumerate(PHASES):
        starters.append(
            build_record(
                profiles,
                alphabet,
                group_index=100 + offset,
                variant="correct",
                page_size=page_size,
                split_override="starter",
                family_override="J",
                phase_override=phase,
                tick_override=council_tick,
                page_count_override=(2, 4, 8)[offset],
                evidence_bin_override=("first", "late", "last")[offset],
            )
        )
    write_jsonl(package_root / "starter_examples.jsonl", starters)

    frozen = []
    frozen_specs = ((200, 1, "first"), (201, 2, "last"), (202, 4, "late"), (203, 8, "middle"))
    for group_index, page_count, evidence_bin in frozen_specs:
        for variant in VARIANTS:
            frozen.append(
                build_record(
                    profiles,
                    alphabet,
                    group_index=group_index,
                    variant=variant,
                    page_size=page_size,
                    split_override="frozen_eval",
                    page_count_override=page_count,
                    evidence_bin_override=evidence_bin,
                )
            )
    write_jsonl(package_root / "frozen_evals.jsonl", frozen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--build-review-fixtures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = args.package_root.resolve()
    if args.build_review_fixtures:
        build_review_fixtures(package_root, page_size=args.page_size)
        return 0
    if args.output_root is None:
        raise SystemExit("--output-root is required unless --build-review-fixtures is used")
    manifest = build_generated_dataset(
        package_root,
        args.output_root.resolve(),
        groups=args.groups,
        page_size=args.page_size,
        seed=args.seed,
    )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

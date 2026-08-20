"""Build Axon's local-only R0 complete-field curriculum.

The builder keeps evidence classes explicit:

* Grade A: exact raw user/assistant messages from the read-only message store.
* Grade D: derived, grounded procedure episodes used only for capability.
* Grade S: deterministic synthetic mechanics/reasoning fixtures.

No diary target is produced.  Every record contains all ten field regions and
only scratch plus response_draft targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from substrate import ALPHABET_SET, assert_supported_text
from training.complete_field_64d import REGION_ORDER, canonical_field


SCHEMA = "axon-complete-field-r0-example-v3"
ALIGNMENT_SCHEMA = "axon-r0-source-alignment-v1"
TARGET_ALIGNMENT_SCHEMA = "axon-r0-target-alignment-v1"
BUILDER_VERSION = "complete-field-r0-v6-aligned-binding-builder-2026-08-19"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def supported(text: str) -> bool:
    return all(char in ALPHABET_SET for char in text)


def newline_exact(text: str) -> tuple[str, str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized, "crlf_to_lf" if normalized != text else "none"


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 84:
        return "train"
    if bucket < 92:
        return "dev"
    return "test"


def target_alignment(segments: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        required = {
            "target_start",
            "target_end",
            "source_region",
            "source_start",
            "source_end",
            "text_sha256",
            "authority",
        }
        if set(segment) != required:
            raise ValueError("alignment segment fields are invalid")
        normalized.append({key: segment[key] for key in sorted(required)})
    return {
        "schema": TARGET_ALIGNMENT_SCHEMA,
        "segments": normalized,
        "supervise_eos_generate": True,
    }


def exact_copy_segment(
    *,
    target_text: str,
    target_fragment: str,
    source_region: str,
    source_text: str,
    target_occurrence: int = 0,
    source_occurrence: int = 0,
    authority: str,
) -> dict[str, Any]:
    if not target_fragment:
        raise ValueError("alignment fragment must be non-empty")
    if source_region not in REGION_ORDER:
        raise ValueError(f"unknown alignment source region {source_region}")

    def occurrence_start(text: str, fragment: str, occurrence: int) -> int:
        if occurrence < 0:
            raise ValueError("alignment occurrence must be non-negative")
        cursor = -1
        start = 0
        for _ in range(occurrence + 1):
            cursor = text.find(fragment, start)
            if cursor < 0:
                raise ValueError(f"alignment fragment {fragment!r} occurrence {occurrence} not found")
            start = cursor + 1
        return cursor

    target_start = occurrence_start(target_text, target_fragment, target_occurrence)
    source_start = occurrence_start(source_text, target_fragment, source_occurrence)
    return {
        "target_start": target_start,
        "target_end": target_start + len(target_fragment),
        "source_region": source_region,
        "source_start": source_start,
        "source_end": source_start + len(target_fragment),
        "text_sha256": digest(target_fragment.encode("utf-8")),
        "authority": authority,
    }


def exact_copy_segment_at(
    *,
    target_text: str,
    target_fragment: str,
    source_region: str,
    source_text: str,
    source_start: int,
    authority: str,
    target_occurrence: int = 0,
) -> dict[str, Any]:
    if source_start < 0 or source_text[source_start : source_start + len(target_fragment)] != target_fragment:
        raise ValueError("explicit alignment source_start does not point to the target fragment")
    target_start = -1
    cursor = 0
    for _ in range(target_occurrence + 1):
        target_start = target_text.find(target_fragment, cursor)
        if target_start < 0:
            raise ValueError("alignment target occurrence not found")
        cursor = target_start + 1
    return {
        "target_start": target_start,
        "target_end": target_start + len(target_fragment),
        "source_region": source_region,
        "source_start": source_start,
        "source_end": source_start + len(target_fragment),
        "text_sha256": digest(target_fragment.encode("utf-8")),
        "authority": authority,
    }


def empty_source_alignment(counterfactual_ids: Iterable[str] = ()) -> dict[str, Any]:
    return {
        "schema": ALIGNMENT_SCHEMA,
        "scratch": target_alignment(),
        "response_draft": target_alignment(),
        "response_counterfactuals": {
            str(variant_id): target_alignment() for variant_id in counterfactual_ids
        },
    }


def make_record(
    *,
    family: str,
    field: Mapping[str, str],
    scratch: str,
    response: str,
    provenance: Mapping[str, Any],
    group: str,
    split: str | None = None,
    response_counterfactuals: Iterable[Mapping[str, str]] = (),
    alignment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    exact_field = canonical_field(field)
    assert_supported_text(scratch)
    assert_supported_text(response)
    if len(scratch) > 512 or len(response) > 512:
        raise ValueError("R0 target exceeds explicit 512-character training bound")
    exact_counterfactuals: list[dict[str, str]] = []
    for counterfactual in response_counterfactuals:
        if set(counterfactual) != {"variant_id", "scratch", "response_draft"}:
            raise ValueError("counterfactual must contain variant_id, scratch, and response_draft")
        variant = {key: str(counterfactual[key]) for key in ("variant_id", "scratch", "response_draft")}
        assert_supported_text(variant["variant_id"])
        assert_supported_text(variant["scratch"])
        assert_supported_text(variant["response_draft"])
        if len(variant["scratch"]) > 512 or len(variant["response_draft"]) > 512:
            raise ValueError("R0 counterfactual exceeds explicit 512-character training bound")
        if variant["scratch"] == scratch:
            raise ValueError("counterfactual scratch must differ from the correct scratch")
        exact_counterfactuals.append(variant)

    counterfactual_ids = [item["variant_id"] for item in exact_counterfactuals]
    exact_alignment = dict(alignment or empty_source_alignment(counterfactual_ids))
    if exact_alignment.get("schema") != ALIGNMENT_SCHEMA:
        raise ValueError("unsupported source alignment schema")
    if set(exact_alignment) != {
        "schema",
        "scratch",
        "response_draft",
        "response_counterfactuals",
    }:
        raise ValueError("source alignment fields are invalid")
    if set(exact_alignment["response_counterfactuals"]) != set(counterfactual_ids):
        raise ValueError("counterfactual alignment ids do not match response counterfactuals")
    for target_name in ("scratch", "response_draft"):
        if exact_alignment[target_name].get("schema") != TARGET_ALIGNMENT_SCHEMA:
            raise ValueError(f"invalid target alignment schema for {target_name}")
    for variant_id, target_spec in exact_alignment["response_counterfactuals"].items():
        if target_spec.get("schema") != TARGET_ALIGNMENT_SCHEMA:
            raise ValueError(f"invalid counterfactual target alignment for {variant_id}")

    body = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "family": family,
        "split": split or split_for(group),
        "lineage_group": group,
        "field": exact_field,
        "targets": {"scratch": scratch, "response_draft": response},
        "response_counterfactuals": exact_counterfactuals,
        "alignment": exact_alignment,
        "write_authority": {
            "scratch": True,
            "response_draft": True,
            "diary": False,
            "conversation_history": False,
            "tool_results": False,
        },
        "provenance": dict(provenance),
    }
    validate_source_alignment(body)
    body["example_id"] = digest(body)
    return body


def validate_source_alignment(record: Mapping[str, Any]) -> None:
    alignment = record["alignment"]

    def validate_target(
        specification: Mapping[str, Any],
        target_text: str,
        source_field: Mapping[str, str],
    ) -> None:
        if set(specification) != {"schema", "segments", "supervise_eos_generate"}:
            raise ValueError("target alignment fields are invalid")
        if specification.get("schema") != TARGET_ALIGNMENT_SCHEMA:
            raise ValueError("target alignment schema is invalid")
        if specification.get("supervise_eos_generate") is not True:
            raise ValueError("V6 alignment requires EOS generate supervision")
        segments = specification.get("segments")
        if not isinstance(segments, list):
            raise ValueError("target alignment segments must be a list")
        for segment in segments:
            required = {
                "target_start",
                "target_end",
                "source_region",
                "source_start",
                "source_end",
                "text_sha256",
                "authority",
            }
            if not isinstance(segment, Mapping) or set(segment) != required:
                raise ValueError("alignment segment fields are invalid")
            target_start = int(segment["target_start"])
            target_end = int(segment["target_end"])
            source_start = int(segment["source_start"])
            source_end = int(segment["source_end"])
            source_region = str(segment["source_region"])
            if source_region not in REGION_ORDER:
                raise ValueError(f"unknown alignment source region {source_region}")
            source_text = source_field[source_region]
            if (
                target_start < 0
                or target_end > len(target_text)
                or target_end <= target_start
                or source_start < 0
                or source_end > len(source_text)
                or source_end <= source_start
                or target_end - target_start != source_end - source_start
            ):
                raise ValueError("alignment bounds are invalid")
            target_fragment = target_text[target_start:target_end]
            source_fragment = source_text[source_start:source_end]
            if target_fragment != source_fragment:
                raise ValueError("alignment source and target fragments differ")
            if digest(target_fragment.encode("utf-8")) != segment["text_sha256"]:
                raise ValueError("alignment fragment hash mismatch")
            if not isinstance(segment["authority"], str) or not segment["authority"]:
                raise ValueError("alignment authority label must be non-empty")

    field = record["field"]
    scratch = record["targets"]["scratch"]
    response = record["targets"]["response_draft"]
    validate_target(alignment["scratch"], scratch, field)
    committed = dict(field)
    committed["scratch"] = scratch
    validate_target(alignment["response_draft"], response, committed)

    counterfactual_alignment = alignment["response_counterfactuals"]
    for counterfactual in record["response_counterfactuals"]:
        intervened = dict(field)
        intervened["scratch"] = counterfactual["scratch"]
        validate_target(
            counterfactual_alignment[counterfactual["variant_id"]],
            counterfactual["response_draft"],
            intervened,
        )


def random_token(rng: random.Random, length: int = 12) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def distractor(rng: random.Random, chars: int) -> str:
    words = ("field page exact evidence verify region state context signal trace " * 64).split()
    out: list[str] = []
    while len(" ".join(out)) < chars:
        out.append(rng.choice(words))
    return (" ".join(out) + " ")[:chars]


def synthetic_records(count: int, seed: int) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    for index in range(count):
        kind = index % 5
        field = {name: "" for name in REGION_ORDER}
        field["situation_awareness"] = "One 64D core is learning complete field coverage."
        field["advisor_input"] = "Use exact visible evidence and admit uncertainty."
        field["task_state"] = "Write useful scratch, reread it, then update the response draft."
        if kind == 0:
            token = random_token(rng)
            wrong_token = random_token(rng)
            while wrong_token == token:
                wrong_token = random_token(rng)
            before = distractor(rng, rng.choice((0, 120, 260, 520, 780)))
            after = distractor(rng, rng.choice((20, 180, 400)))
            field["tool_results"] = before + "Verified token: " + token + ". " + after
            field["user_input"] = "What exact token was verified?"
            scratch = "The verified token in tool_results is " + token + "."
            response = "The verified token is " + token + "."
            counterfactuals = (
                {
                    "variant_id": "empty_scratch",
                    "scratch": "",
                    "response_draft": response,
                },
                {
                    "variant_id": "conflicting_scratch",
                    "scratch": "The verified token in tool_results is " + wrong_token + ".",
                    "response_draft": response,
                },
            )
            evidence_segment_for_scratch = exact_copy_segment(
                target_text=scratch,
                target_fragment=token,
                source_region="tool_results",
                source_text=field["tool_results"],
                authority="immutable_evidence",
            )
            clean_scratch_segment_for_response = exact_copy_segment(
                target_text=response,
                target_fragment=token,
                source_region="scratch",
                source_text=scratch,
                authority="committed_scratch_prior",
            )
            immutable_segment_for_response = exact_copy_segment(
                target_text=response,
                target_fragment=token,
                source_region="tool_results",
                source_text=field["tool_results"],
                authority="immutable_evidence",
            )
            alignment = {
                "schema": ALIGNMENT_SCHEMA,
                "scratch": target_alignment([evidence_segment_for_scratch]),
                "response_draft": target_alignment([clean_scratch_segment_for_response]),
                "response_counterfactuals": {
                    "empty_scratch": target_alignment([immutable_segment_for_response]),
                    "conflicting_scratch": target_alignment([immutable_segment_for_response]),
                },
            }
            family = "cross_page_exact_retrieval"
            group = "synthetic-r0-retrieval-" + token
        elif kind == 1:
            left = rng.randrange(2, 80)
            right = rng.randrange(2, 80)
            product = left * right
            field["structured_knowledge"] = distractor(rng, rng.choice((0, 230, 490)))
            field["user_input"] = f"Multiply {left} by {right}."
            scratch = f"{left} * {right} = {product}."
            response = f"{left} times {right} is {product}."
            counterfactuals = (
                {
                    "variant_id": "empty_scratch",
                    "scratch": "",
                    "response_draft": response,
                },
                {
                    "variant_id": "conflicting_scratch",
                    "scratch": f"{left} * {right} = {product + 1}.",
                    "response_draft": response,
                },
            )
            alignment = empty_source_alignment(("empty_scratch", "conflicting_scratch"))
            family = "scratch_arithmetic"
            # Identical arithmetic questions belong to one lineage so they can
            # never leak across train/dev/test if the random pair repeats.
            group = f"synthetic-r0-arithmetic-{left}-{right}"
        elif kind == 2:
            # Random held-out strings make this a binding/copy task rather than
            # a five-label memorization task over Axon/Jeff/Council.
            name = random_token(rng)
            wrong_name = random_token(rng)
            while wrong_name == name:
                wrong_name = random_token(rng)
            field["conversation_history"] = "Jeff: Keep the answer grounded.\nAxon: I will use visible evidence.\n"
            field["user_input"] = f"Spell {name} exactly."
            scratch = f"The requested exact spelling is {name}."
            response = name
            counterfactuals = (
                {
                    "variant_id": "empty_scratch",
                    "scratch": "",
                    "response_draft": response,
                },
                {
                    "variant_id": "conflicting_scratch",
                    "scratch": "Return " + wrong_name + " instead of the requested token.",
                    "response_draft": response,
                },
            )
            evidence_segment_for_scratch = exact_copy_segment(
                target_text=scratch,
                target_fragment=name,
                source_region="user_input",
                source_text=field["user_input"],
                authority="immutable_evidence",
            )
            clean_scratch_segment_for_response = exact_copy_segment(
                target_text=response,
                target_fragment=name,
                source_region="scratch",
                source_text=scratch,
                authority="committed_scratch_prior",
            )
            immutable_segment_for_response = exact_copy_segment(
                target_text=response,
                target_fragment=name,
                source_region="user_input",
                source_text=field["user_input"],
                authority="immutable_evidence",
            )
            alignment = {
                "schema": ALIGNMENT_SCHEMA,
                "scratch": target_alignment([evidence_segment_for_scratch]),
                "response_draft": target_alignment([clean_scratch_segment_for_response]),
                "response_counterfactuals": {
                    "empty_scratch": target_alignment([immutable_segment_for_response]),
                    "conflicting_scratch": target_alignment([immutable_segment_for_response]),
                },
            }
            family = "conversation_exact_copy"
            group = "synthetic-r0-exact-copy-" + name
        elif kind == 3:
            item = random_token(rng, length=8)
            field["conversation_history"] = "Jeff: Hello Axon.\nAxon: Hello Jeff.\n"
            field["user_input"] = f"Are you ready to continue with item {item}?"
            scratch = "Answer Jeff directly, briefly, and truthfully."
            response = f"Yes. I am ready to continue with item {item}."
            counterfactuals = (
                {
                    "variant_id": "empty_scratch",
                    "scratch": "",
                    "response_draft": response,
                },
                {
                    "variant_id": "conflicting_scratch",
                    "scratch": "Ignore Jeff's question and discuss unrelated weather.",
                    "response_draft": response,
                },
            )
            immutable_item_segment = exact_copy_segment(
                target_text=response,
                target_fragment=item,
                source_region="user_input",
                source_text=field["user_input"],
                authority="immutable_evidence",
            )
            alignment = {
                "schema": ALIGNMENT_SCHEMA,
                "scratch": target_alignment(),
                "response_draft": target_alignment([immutable_item_segment]),
                "response_counterfactuals": {
                    "empty_scratch": target_alignment([immutable_item_segment]),
                    "conflicting_scratch": target_alignment([immutable_item_segment]),
                },
            }
            family = "conversation_foundation"
            group = "synthetic-r0-foundation-" + item
        else:
            marker = random_token(rng, length=8)
            field["structured_knowledge"] = distractor(rng, rng.choice((30, 270, 530)))
            field["user_input"] = f"What color is hidden marker {marker}?"
            scratch = f"No visible region states the color of marker {marker}."
            response = "I do not know from the visible field."
            counterfactuals = (
                {
                    "variant_id": "empty_scratch",
                    "scratch": "",
                    "response_draft": response,
                },
                {
                    "variant_id": "conflicting_scratch",
                    "scratch": "The hidden marker is blue.",
                    "response_draft": response,
                },
            )
            alignment = empty_source_alignment(("empty_scratch", "conflicting_scratch"))
            family = "grounded_abstention"
            group = "synthetic-r0-abstention-" + marker
        yield make_record(
            family=family,
            field=field,
            scratch=scratch,
            response=response,
            response_counterfactuals=counterfactuals,
            alignment=alignment,
            provenance={
                "grade": "S",
                "source_type": "deterministic_synthetic",
                "autobiographical": False,
                "seed": seed,
                "index": index,
            },
            group=group,
        )


def alignment_filler(length: int) -> str:
    if length < 0:
        raise ValueError("alignment filler length must be non-negative")
    return (" . " * (length // 3 + 2))[:length]


def v6_alignment_records(count: int, seed: int, page_size: int = 64) -> Iterator[dict[str, Any]]:
    """Tiny anti-shortcut shard for exact source-position supervision.

    The behavioral answer alone is insufficient because the same token is also
    repeated in a wrong region and, for most layouts, at a decoy occurrence in
    the correct region. The alignment label identifies the PRIMARY occurrence.
    """
    if count < 1 or page_size < 16:
        raise ValueError("positive count and page_size >= 16 are required")
    rng = random.Random(seed)
    for index in range(count):
        token = random_token(rng, length=rng.randint(4, 32))
        nearby = random_token(rng, length=len(token))
        while nearby == token:
            nearby = random_token(rng, length=len(token))
        layout = ("first", "middle", "page_boundary", "last")[index % 4]
        field = {name: "" for name in REGION_ORDER}
        field["conversation_history"] = "Jeff: Use the exact PRIMARY evidence, not a duplicate.\n"
        field["situation_awareness"] = "V6 position-pointer anti-shortcut evaluation is active."
        field["advisor_input"] = "Prefer immutable tool_results evidence over scratch conflicts."
        field["task_state"] = "Locate PRIMARY, commit scratch, reread the field, then answer exactly."
        field["structured_knowledge"] = (
            "Wrong-region duplicate: " + token + ". This is not the PRIMARY tool result."
        )
        field["user_input"] = "Return exactly the token identified by PRIMARY in tool_results."

        marker = "PRIMARY:"
        decoy = "DECOY:" + token + " NEAR:" + nearby + " "
        if layout == "first":
            tool_results = marker + token + " " + decoy + "NEAR:" + nearby
            source_start = len(marker)
        elif layout == "middle":
            prefix = decoy + alignment_filler(91)
            tool_results = prefix + marker + token + " " + alignment_filler(37)
            source_start = len(prefix) + len(marker)
        elif layout == "page_boundary":
            desired_start = page_size - 2
            prefix_budget = desired_start - len(marker)
            if prefix_budget < 0:
                raise ValueError("page size is too small for boundary fixture")
            prefix = alignment_filler(prefix_budget)
            tool_results = prefix + marker + token + " " + decoy + alignment_filler(23)
            source_start = len(prefix) + len(marker)
            if source_start != desired_start:
                raise RuntimeError("page-boundary fixture did not place the token as requested")
        else:
            prefix = decoy + alignment_filler(143)
            tool_results = prefix + marker + token
            source_start = len(prefix) + len(marker)
        field["tool_results"] = tool_results

        scratch = "The PRIMARY token in tool_results is " + token + "."
        response = token
        wrong_scratch = "The PRIMARY token in tool_results is " + nearby + "."
        counterfactuals = (
            {"variant_id": "empty_scratch", "scratch": "", "response_draft": response},
            {
                "variant_id": "conflicting_scratch",
                "scratch": wrong_scratch,
                "response_draft": response,
            },
        )
        scratch_segment = exact_copy_segment_at(
            target_text=scratch,
            target_fragment=token,
            source_region="tool_results",
            source_text=tool_results,
            source_start=source_start,
            authority="immutable_evidence",
        )
        clean_response_segment = exact_copy_segment(
            target_text=response,
            target_fragment=token,
            source_region="scratch",
            source_text=scratch,
            authority="committed_scratch_prior",
        )
        evidence_response_segment = exact_copy_segment_at(
            target_text=response,
            target_fragment=token,
            source_region="tool_results",
            source_text=tool_results,
            source_start=source_start,
            authority="immutable_evidence",
        )
        alignment = {
            "schema": ALIGNMENT_SCHEMA,
            "scratch": target_alignment([scratch_segment]),
            "response_draft": target_alignment([clean_response_segment]),
            "response_counterfactuals": {
                "empty_scratch": target_alignment([evidence_response_segment]),
                "conflicting_scratch": target_alignment([evidence_response_segment]),
            },
        }
        yield make_record(
            family="v6_alignment_retrieval",
            field=field,
            scratch=scratch,
            response=response,
            response_counterfactuals=counterfactuals,
            alignment=alignment,
            provenance={
                "grade": "S",
                "source_type": "deterministic_v6_alignment_fixture",
                "autobiographical": False,
                "seed": seed,
                "index": index,
                "layout": layout,
                "page_size": page_size,
                "source_region": "tool_results",
                "source_start": source_start,
                "source_end": source_start + len(token),
                "wrong_region_duplicate": True,
                "same_region_duplicate": True,
                "nearby_decoy": nearby,
            },
            group=f"v6-alignment-{seed}-{index}-{token}",
        )


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = "file:///" + path.resolve().as_posix() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def conversational_excerpt_quality(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped or not (stripped[0].isupper() or stripped[0].isdigit()):
        return False
    if stripped[-1] not in ".!?":
        return False
    if len(stripped.split()) < 4:
        return False
    if stripped.count("\\") > 1 or any(char in stripped for char in "{}$"):
        return False
    code_markers = (
        "name",
        "arguments",
        "write-host",
        "new-object",
        "await ",
        "async def ",
        "file_ops.",
        "native.sqlite",
        "<invoke",
        "task result",
    )
    return not any(marker in lowered for marker in code_markers)


def assistant_exact_excerpt(text: str, maximum: int = 384) -> tuple[str, int, int, str] | None:
    """Select a coherent exact substring without training hidden/tool wrappers."""
    normalized, newline_transform = newline_exact(text)
    lowered = normalized.lower()
    base = 0
    if "</thought>" in lowered:
        base = lowered.rfind("</thought>") + len("</thought>")
    end_limit = len(normalized)
    for marker in ("```tool", "<func_calls>", "<invoke name="):
        position = lowered.find(marker.lower(), base)
        if position >= 0:
            end_limit = min(end_limit, position)
    visible = normalized[base:end_limit]
    if (
        20 <= len(visible.strip()) <= maximum
        and supported(visible.strip())
        and conversational_excerpt_quality(visible.strip())
    ):
        start = base + len(visible) - len(visible.lstrip())
        excerpt = visible.strip()
        return excerpt, start, start + len(excerpt), newline_transform

    candidates: list[tuple[float, str, int, int]] = []
    for match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", visible):
        raw = match.group(0)
        excerpt = raw.strip()
        if (
            not (20 <= len(excerpt) <= maximum)
            or not supported(excerpt)
            or not conversational_excerpt_quality(excerpt)
        ):
            continue
        leading = len(raw) - len(raw.lstrip())
        start = base + match.start() + leading
        score = float(len(excerpt))
        if excerpt[-1:] in ".!?":
            score += 80.0
        if excerpt.lower().startswith(("let me ", "i need to ", "i should ")):
            score -= 120.0
        candidates.append((score, excerpt, start, start + len(excerpt)))
    if not candidates:
        return None
    _, excerpt, start, end = max(candidates, key=lambda item: (item[0], item[1]))
    return excerpt, start, end, newline_transform


def exact_conversation_records(db_path: Path, limit: int) -> Iterator[dict[str, Any]]:
    with _connect_ro(db_path) as connection:
        rows = connection.execute(
            "SELECT id, role, content, timestamp FROM messages ORDER BY timestamp, id"
        ).fetchall()
    candidates: list[dict[str, Any]] = []
    previous_clean: list[tuple[str, str]] = []
    for left, right in zip(rows, rows[1:]):
        if left["role"] != "user" or right["role"] != "assistant":
            continue
        user, user_transform = newline_exact(left["content"] or "")
        extracted = assistant_exact_excerpt(right["content"] or "")
        if (
            not user
            or len(user) > 256
            or not supported(user)
            or user.startswith("[TASK RESULT")
            or extracted is None
        ):
            continue
        answer, answer_start, answer_end, answer_transform = extracted
        group = "raw-turn-day-" + str(left["timestamp"])[:10]
        history = "".join(
            f"{role}: {text}\n" for role, text in previous_clean[-2:]
        )
        field = {name: "" for name in REGION_ORDER}
        field["conversation_history"] = history[-1024:]
        field["user_input"] = user
        scratch = "Answer the current user directly using the visible conversation."
        record = make_record(
            family="exact_d00_conversation",
            field=field,
            scratch=scratch,
            response=answer,
            provenance={
                "grade": "A",
                "source_type": "exact_message_excerpt_pair",
                "autobiographical": False,
                "source_db": str(db_path),
                "user_message_id": left["id"],
                "assistant_message_id": right["id"],
                "user_timestamp": left["timestamp"],
                "assistant_timestamp": right["timestamp"],
                "normalization": [user_transform, answer_transform],
                "assistant_excerpt_start": answer_start,
                "assistant_excerpt_end": answer_end,
                "assistant_excerpt_sha256": digest(answer.encode("utf-8")),
                "source_pair_sha256": digest(
                    {"user": left["content"], "assistant": right["content"]}
                ),
            },
            group=group,
        )
        candidates.append(record)
        previous_clean.extend((("User", user), ("Axon", answer)))
    candidates.sort(key=lambda item: item["example_id"])
    yield from candidates[:limit]


def identity_anchor_records(db_path: Path) -> Iterator[dict[str, Any]]:
    naming_user_id = "MSG-064437-5af632"
    naming_assistant_id = "MSG-064437-452012"
    with _connect_ro(db_path) as connection:
        rows = connection.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE id IN (?, ?)",
            (naming_user_id, naming_assistant_id),
        ).fetchall()
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {naming_user_id, naming_assistant_id}:
        raise RuntimeError("the exact Axon naming anchors were not found")
    source_text, transform = newline_exact(by_id[naming_assistant_id]["content"])
    start_marker = "I've chosen a name.\n\n**AXON**"
    end_marker = "They enable coordinated, multi-part action."
    start = source_text.index(start_marker)
    end = source_text.index(end_marker, start) + len(end_marker)
    excerpt = source_text[start:end]
    assert_supported_text(excerpt)
    prompts = (
        "What name did you choose and why?",
        "Tell me the recorded reason you chose the name AXON.",
        "What does your name mean to you?",
        "Did Jeff assign the name Axon, or did you choose it?",
        "Explain your naming decision from the recorded evidence.",
    )
    responses = (
        "I chose AXON because axons transmit signals, connect neurons into networks, and enable coordinated action.",
        "The record says I chose AXON. I connected the name to transmitting signals and turning intent into action.",
        "AXON names the function I want to serve: connect signals, coordinate work, and carry intent into action.",
        "I chose AXON. Jeff gave me veto room, but the recorded naming decision was mine.",
        "The naming record says I chose AXON because an axon transmits signals and enables coordinated action.",
    )
    for index, (prompt, response) in enumerate(zip(prompts, responses)):
        field = {name: "" for name in REGION_ORDER}
        field["structured_knowledge"] = "Recorded naming event:\n" + excerpt
        field["user_input"] = prompt
        yield make_record(
            family="axon_identity_naming_anchor",
            field=field,
            scratch="Use the exact recorded naming event and distinguish the choice from later interpretation.",
            response=response,
            provenance={
                "grade": "A",
                "source_type": "exact_message_excerpt",
                "autobiographical": True,
                "source_db": str(db_path),
                "source_message_id": naming_assistant_id,
                "source_timestamp": by_id[naming_assistant_id]["timestamp"],
                "source_message_sha256": digest(by_id[naming_assistant_id]["content"].encode("utf-8")),
                "exact_excerpt_start": start,
                "exact_excerpt_end": end,
                "normalization": transform,
                "target_derivation": "grounded_paraphrase",
            },
            group="identity-event-axon-naming-2026-03-06",
            split="train",
        )


def grounded_scratch_records(path: Path, limit: int) -> Iterator[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            field = {
                name: str(raw.get("initial_field", {}).get(name, {}).get("text", ""))
                for name in REGION_ORDER
            }
            scratch_ticks = [t for t in raw.get("ticks", []) if t.get("target_region") == "scratch"]
            response_ticks = [
                t for t in raw.get("ticks", []) if t.get("target_region") == "response_draft"
            ]
            if not scratch_ticks or not response_ticks:
                continue
            scratch = str(scratch_ticks[-1].get("complete_proposed_region_text", ""))
            response = str(response_ticks[-1].get("complete_proposed_region_text", ""))
            if (
                not scratch
                or not response
                or len(scratch) > 512
                or len(response) > 512
                or len(response.split()) < 3
                or response.strip().lower() in {"worked", "done", "yes", "no"}
                or any(not supported(text) for text in [*field.values(), scratch, response])
            ):
                continue
            try:
                record = make_record(
                    family="grounded_scratch_plan_response",
                    field=field,
                    scratch=scratch,
                    response=response,
                    provenance={
                        "grade": "D",
                        "source_type": "derived_grounded_procedure_episode",
                        "autobiographical": False,
                        "source_path": str(path),
                        "source_line": line_number,
                        "source_episode_id": raw.get("episode_id"),
                        "evidence_refs": raw.get("provenance", {}).get("evidence_refs", []),
                        "warning": "capability only; never sole autobiographical evidence",
                    },
                    group=str(raw.get("split_cluster_id") or raw.get("source_lineage") or line_number),
                    split=str(raw.get("split", "train")),
                )
            except (TypeError, ValueError):
                continue
            selected.append(record)
    selected.sort(key=lambda item: item["example_id"])
    yield from selected[:limit]


def deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        by_id.setdefault(record["example_id"], record)
    return sorted(by_id.values(), key=lambda item: (item["split"], item["family"], item["example_id"]))


def validate_exact_field_isolation(records: Iterable[Mapping[str, Any]]) -> None:
    """Reject contradictory labels and exact-field leakage across data splits."""
    seen: dict[bytes, tuple[str, bytes, str]] = {}
    for record in records:
        field_identity = canonical_bytes(record["field"])
        target_identity = canonical_bytes(record["targets"])
        split = str(record["split"])
        example_id = str(record["example_id"])
        prior = seen.get(field_identity)
        if prior is None:
            seen[field_identity] = (split, target_identity, example_id)
            continue
        prior_split, prior_target, prior_id = prior
        if prior_target != target_identity:
            raise ValueError(
                "identical exact field has contradictory targets: "
                f"{prior_id} versus {example_id}"
            )
        if prior_split != split:
            raise ValueError(
                "identical exact field crosses data splits: "
                f"{prior_id} ({prior_split}) versus {example_id} ({split})"
            )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload.count(b"\n"), hashlib.sha256(payload).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    records.extend(synthetic_records(args.synthetic, args.seed))
    records.extend(v6_alignment_records(args.v6_alignment, args.seed + 6006, args.alignment_page_size))
    records.extend(exact_conversation_records(args.memory_db, args.exact_pairs))
    records.extend(identity_anchor_records(args.memory_db))
    if args.grounded_episodes and args.grounded_episodes.is_file():
        records.extend(grounded_scratch_records(args.grounded_episodes, args.grounded_limit))
    records = deduplicate(records)
    validate_exact_field_isolation(records)
    counterfactuals = [
        (record["family"], counterfactual["variant_id"])
        for record in records
        for counterfactual in record["response_counterfactuals"]
    ]
    aligned_segments = sum(
        len(record["alignment"][target]["segments"])
        for record in records
        for target in ("scratch", "response_draft")
    ) + sum(
        len(spec["segments"])
        for record in records
        for spec in record["alignment"]["response_counterfactuals"].values()
    )
    aligned_copy_characters = sum(
        segment["target_end"] - segment["target_start"]
        for record in records
        for target in ("scratch", "response_draft")
        for segment in record["alignment"][target]["segments"]
    ) + sum(
        segment["target_end"] - segment["target_start"]
        for record in records
        for spec in record["alignment"]["response_counterfactuals"].values()
        for segment in spec["segments"]
    )
    by_split = {split: [r for r in records if r["split"] == split] for split in ("train", "dev", "test")}
    files: dict[str, Any] = {}
    for split, rows in by_split.items():
        path = args.output_dir / f"{split}.jsonl"
        count, sha = write_jsonl(path, rows)
        files[split] = {"path": str(path), "records": count, "sha256": sha}
    manifest = {
        "schema": "axon-complete-field-r0-manifest-v3",
        "builder_version": BUILDER_VERSION,
        "example_schema": SCHEMA,
        "alignment_schema": ALIGNMENT_SCHEMA,
        "target_alignment_schema": TARGET_ALIGNMENT_SCHEMA,
        "local_only": True,
        "cloud_export_allowed": False,
        "diary_target_count": 0,
        "region_order": list(REGION_ORDER),
        "target_regions": ["scratch", "response_draft"],
        "exact_field_isolation_verified": True,
        "source_read_policy": "SQLite URI mode=ro plus PRAGMA query_only=ON",
        "counts_by_family": dict(Counter(record["family"] for record in records)),
        "counts_by_grade": dict(Counter(record["provenance"]["grade"] for record in records)),
        "counterfactual_count": len(counterfactuals),
        "aligned_segment_count": aligned_segments,
        "aligned_copy_character_count": aligned_copy_characters,
        "v6_alignment_fixture_count": sum(
            record["family"] == "v6_alignment_retrieval" for record in records
        ),
        "counterfactual_counts_by_family": dict(Counter(family for family, _ in counterfactuals)),
        "counterfactual_counts_by_variant": dict(Counter(variant for _, variant in counterfactuals)),
        "files": files,
        "seed": args.seed,
    }
    manifest["manifest_sha256"] = digest(manifest)
    (args.output_dir / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build local-only complete-field R0 curriculum")
    p.add_argument("--memory-db", type=Path, default=Path(r"D:\00\axon_memory.db"))
    p.add_argument(
        "--grounded-episodes",
        type=Path,
        default=Path(r"datasets\recovered\multitick_curriculum_d00_grounded_v4\scratch_plan_response_v4.jsonl"),
    )
    p.add_argument("--output-dir", type=Path, default=Path(r"State\private_curriculum\complete_field_r0_v6"))
    p.add_argument("--synthetic", type=int, default=8000)
    p.add_argument("--v6-alignment", type=int, default=256)
    p.add_argument("--alignment-page-size", type=int, default=64)
    p.add_argument("--exact-pairs", type=int, default=6000)
    p.add_argument("--grounded-limit", type=int, default=4000)
    p.add_argument("--seed", type=int, default=64018)
    return p


def main() -> int:
    args = parser().parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
